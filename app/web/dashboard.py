"""Boss dashboard: one read-only, server-rendered page behind a signed, expiring link.

Auth: the bot's /dashboard command issues `?t=<signed token>` (HMAC-SHA256, expiring). Opening
it validates the token, sets an HttpOnly SameSite=Strict cookie holding the same token and
redirects to a clean URL, so the token does not linger in the address bar or a Referer. No
passwords, no OTP. Every request re-checks that the user is still an active boss.

Read-only except one action: correcting a confirmed record, which writes a Correction row
(CSRF-protected). Everything shown from the database is untrusted text; Jinja2 autoescape is on.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, time, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from fastapi import APIRouter, Form, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse, Response
from fastapi.templating import Jinja2Templates

from app.config import Settings
from app.db import TenantRepository
from app.models import (
    ActivityLog,
    ApprovalRequest,
    Extraction,
    Message,
    SyncJob,
    Tenant,
    User,
)
from app.models.enums import ActivityStatus, ApprovalStatus, Role, SyncStatus
from app.pack import Pack, load_pack
from app.pipeline.confirm import TenantSettings
from app.pipeline.media import MediaStore
from app.security import csrf_token, verify_token
from app.services import corrections
from app.services.approvals import visit_of
from app.units import format_inr, format_weight_mg

router = APIRouter(prefix="/dashboard")
TEMPLATES = Jinja2Templates(
    directory=str(Path(__file__).resolve().parent / "templates" / "dashboard")
)
COOKIE = "bd_session"
FEED_LIMIT = 50
_AUDIO = {"ogg": "audio/ogg", "mp3": "audio/mpeg", "m4a": "audio/mp4", "webm": "audio/webm",
          "wav": "audio/wav", "flac": "audio/flac"}  # fmt: skip
_IMAGE = {"jpg": "image/jpeg", "png": "image/png", "webp": "image/webp"}
_STATUS_CLASS = {
    "confirmed": "bg-emerald-100 text-emerald-800",
    "synced": "bg-emerald-100 text-emerald-800",
    "pending_approval": "bg-amber-100 text-amber-800",
    "pending_confirmation": "bg-slate-100 text-slate-700",
    "rejected": "bg-rose-100 text-rose-800",
}


@dataclass
class DashSession:
    tenant_id: uuid.UUID
    user_id: uuid.UUID
    token: str


def _settings(request: Request) -> Settings:
    return request.app.state.settings  # type: ignore[no-any-return]


def _secret(request: Request) -> str:
    s = _settings(request).dashboard_secret
    if s is None:
        raise HTTPException(status_code=404)  # the dashboard does not exist without a secret
    return s.get_secret_value()


def _now(request: Request) -> datetime:
    clock = getattr(request.app.state, "clock", None)
    return clock() if clock else datetime.now(ZoneInfo("UTC"))


def _harden(resp: Response) -> Response:
    resp.headers["Referrer-Policy"] = "no-referrer"
    resp.headers["X-Frame-Options"] = "DENY"
    resp.headers["X-Content-Type-Options"] = "nosniff"
    resp.headers["Cache-Control"] = "no-store"
    return resp


def _expired_page(request: Request) -> Response:
    return _harden(TEMPLATES.TemplateResponse(request, "expired.html", {}, status_code=401))


async def _session(request: Request) -> DashSession:
    token = request.cookies.get(COOKIE)
    payload = verify_token(_secret(request), token or "", now=int(_now(request).timestamp()))
    if payload is None or payload.get("k") != "dash" or not token:
        raise HTTPException(status_code=401)
    try:
        sess = DashSession(uuid.UUID(hex=payload["t"]), uuid.UUID(hex=payload["u"]), token)
    except (KeyError, ValueError) as exc:
        raise HTTPException(status_code=401) from exc
    async with request.app.state.session_factory() as db:
        user = await db.get(User, sess.user_id)
    if (
        user is None
        or user.tenant_id != sess.tenant_id
        or user.role is not Role.BOSS
        or not user.is_active
    ):
        raise HTTPException(status_code=401)  # a removed/demoted boss loses access at once
    return sess


async def _auth(request: Request) -> DashSession | Response:
    try:
        return await _session(request)
    except HTTPException as exc:
        if exc.status_code == 401:
            return _expired_page(request)
        raise


# --------------------------------------------------------------------------- data


def _local_window(tenant: Tenant, now: datetime) -> tuple[datetime, datetime]:
    tz = ZoneInfo(tenant.timezone)
    start = datetime.combine(now.astimezone(tz).date(), time.min, tzinfo=tz)
    return start, start + timedelta(days=1)


async def _load(request: Request, sess: DashSession) -> dict[str, Any]:
    now = _now(request)
    async with request.app.state.session_factory() as db:
        repo = TenantRepository(db, sess.tenant_id)
        tenant = await db.get(Tenant, sess.tenant_id)
        assert tenant is not None
        tz = ZoneInfo(tenant.timezone)
        start, end = _local_window(tenant, now)
        users = {u.id: u for u in await repo.list(User)}
        acts = await repo.list(
            ActivityLog, ActivityLog.intent == "LOG_VISIT", order_by=[ActivityLog.created_at.desc()]
        )
        today = [
            a
            for a in acts
            if start <= a.created_at < end and a.status is not ActivityStatus.REJECTED
        ]
        counted = [a for a in today if a.status is not ActivityStatus.PENDING_CONFIRMATION]
        # Order value counts only confirmed/synced orders; one awaiting approval is not yet an order.
        value_today = sum(_value(a) for a in counted if a.status in corrections.CORRECTABLE)
        value_waiting = sum(
            _value(a) for a in counted if a.status is ActivityStatus.PENDING_APPROVAL
        )
        per_rep: dict[str, int] = {}
        for a in counted:
            name = users[a.user_id].name if a.user_id in users else "-"
            per_rep[name] = per_rep.get(name, 0) + 1

        feed_acts = [a for a in acts][:FEED_LIMIT]
        message_ids = [a.message_id for a in feed_acts if a.message_id]
        messages = {m.id: m for m in await repo.list(Message) if m.id in message_ids}
        transcripts = {
            e.message_id: e.transcript
            for e in await repo.list(Extraction)
            if e.message_id in message_ids
        }
        feed = []
        for a in feed_acts:
            v = visit_of(a)
            msg = messages.get(a.message_id) if a.message_id else None
            media = None
            purged = msg is not None and msg.media_purged_at is not None
            if msg is not None and msg.sha256 and not purged and msg.kind in ("voice", "photo"):
                media = "audio" if msg.kind == "voice" else "image"
            feed.append(
                {
                    "id": a.id.hex,
                    "time": a.created_at.astimezone(tz).strftime("%d %b %H:%M"),
                    "rep": users[a.user_id].name if a.user_id in users else "-",
                    "client": v.client_name or "-",
                    "amount": format_inr(v.order.value_paise)
                    if v.order and v.order.value_paise is not None
                    else "-",
                    "weight": format_weight_mg(v.order.weight_mg)
                    if v.order and v.order.weight_mg is not None
                    else "-",
                    "status": a.status.value,
                    "status_class": _STATUS_CLASS.get(
                        a.status.value, "bg-slate-100 text-slate-700"
                    ),
                    "transcript": transcripts.get(a.message_id) if a.message_id else None,
                    "media": media,
                    "media_purged": purged,
                    "corrected": bool(a.payload.get("meta", {}).get("corrected_by_boss")),
                    "can_correct": a.status in corrections.CORRECTABLE,
                }
            )
        approvals = []
        for ap in await repo.list(
            ApprovalRequest, ApprovalRequest.status == ApprovalStatus.PENDING,
            order_by=[ApprovalRequest.created_at],
        ):  # fmt: skip
            act = next((a for a in acts if a.id == ap.activity_log_id), None)
            av = visit_of(act) if act else None
            approvals.append(
                {
                    "rep": users[ap.requested_by].name if ap.requested_by in users else "-",
                    "client": (av.client_name if av else None) or "-",
                    "amount": format_inr(v.order.value_paise)
                    if v and v.order and v.order.value_paise is not None
                    else "-",
                    "reason": ap.reason_en,
                    "expires": ap.expires_at.astimezone(tz).strftime("%d %b %H:%M"),
                }
            )
        jobs = await repo.list(SyncJob)
        ts = TenantSettings.from_json(tenant.settings)
        sync = {
            "configured": bool(ts.sheet_spreadsheet_id),
            "url": f"https://docs.google.com/spreadsheets/d/{ts.sheet_spreadsheet_id}"
            if ts.sheet_spreadsheet_id
            else None,
            "pending": sum(1 for j in jobs if j.status is SyncStatus.PENDING),
            "done": sum(1 for j in jobs if j.status is SyncStatus.DONE),
            "failed": sum(1 for j in jobs if j.status is SyncStatus.FAILED),
            "last_error": next(
                (
                    j.last_error
                    for j in sorted(jobs, key=lambda j: j.updated_at, reverse=True)
                    if j.last_error
                ),
                None,
            ),
        }
    return {
        "tenant": tenant.name,
        "visits_today": len(counted),
        "value_today": format_inr(value_today),
        "value_waiting": format_inr(value_waiting) if value_waiting else None,
        "per_rep": sorted(per_rep.items(), key=lambda kv: (-kv[1], kv[0])),
        "approvals": approvals,
        "feed": feed,
        "sync": sync,
    }


def _value(a: ActivityLog) -> int:
    v = visit_of(a)
    return v.order.value_paise or 0 if v.order else 0


# --------------------------------------------------------------------------- routes


@router.get("", response_class=HTMLResponse)
async def index(request: Request, t: str | None = None) -> Response:
    secret = _secret(request)
    if t is not None:  # first visit from the bot link: swap the token for a cookie
        payload = verify_token(secret, t, now=int(_now(request).timestamp()))
        if payload is None or payload.get("k") != "dash":
            return _expired_page(request)
        resp = RedirectResponse("/dashboard", status_code=303)
        ttl = max(int(payload["exp"]) - int(_now(request).timestamp()), 1)
        https = (_settings(request).public_base_url or "").startswith("https")
        resp.set_cookie(
            COOKIE,
            t,
            max_age=ttl,
            httponly=True,
            samesite="strict",
            secure=https,
            path="/dashboard",
        )
        return _harden(resp)
    sess = await _auth(request)
    if isinstance(sess, Response):
        return sess
    data = await _load(request, sess)
    return _harden(TEMPLATES.TemplateResponse(request, "index.html", data))


@router.get("/partials/{name}", response_class=HTMLResponse)
async def partial(request: Request, name: str) -> Response:
    if name not in ("summary", "approvals", "feed", "sync"):
        raise HTTPException(status_code=404)
    sess = await _auth(request)
    if isinstance(sess, Response):
        return sess
    return _harden(TEMPLATES.TemplateResponse(request, f"_{name}.html", await _load(request, sess)))


@router.get("/media/{activity_id}")
async def media(request: Request, activity_id: str) -> Response:
    sess = await _auth(request)
    if isinstance(sess, Response):
        return sess
    try:
        aid = uuid.UUID(hex=activity_id)
    except ValueError as exc:
        raise HTTPException(status_code=404) from exc
    async with request.app.state.session_factory() as db:
        repo = TenantRepository(db, sess.tenant_id)  # tenant-scoped: no cross-tenant media
        act = await repo.get(ActivityLog, aid)
        msg = await repo.get(Message, act.message_id) if act and act.message_id else None
    if msg is None or not msg.sha256:
        raise HTTPException(status_code=404)
    store = MediaStore(_settings(request).media_dir)
    for ext, mime in {**_AUDIO, **_IMAGE}.items():
        path = store.path_for(sess.tenant_id, msg.sha256, ext)
        if path.is_file():
            return _harden(FileResponse(path, media_type=mime))
    raise HTTPException(status_code=404)


async def _load_correctable(
    request: Request, sess: DashSession, activity_id: str
) -> tuple[ActivityLog, Any]:
    try:
        aid = uuid.UUID(hex=activity_id)
    except ValueError as exc:
        raise HTTPException(status_code=404) from exc
    async with request.app.state.session_factory() as db:
        act = await TenantRepository(db, sess.tenant_id).get(ActivityLog, aid)
    if act is None:
        raise HTTPException(status_code=404)
    return act, visit_of(act)


def _form_context(
    request: Request,
    sess: DashSession,
    act: ActivityLog,
    values: dict[str, str],
    errors: dict[str, str],
) -> dict[str, Any]:
    return {
        "activity_id": act.id.hex,
        "fields": corrections.FIELDS,
        "values": values,
        "errors": errors,
        "csrf": csrf_token(_secret(request), sess.token, act.id.hex),
        "editable": act.status in corrections.CORRECTABLE,
        "status": act.status.value,
    }


@router.get("/correct/{activity_id}", response_class=HTMLResponse)
async def correct_form(request: Request, activity_id: str) -> Response:
    sess = await _auth(request)
    if isinstance(sess, Response):
        return sess
    act, visit = await _load_correctable(request, sess, activity_id)
    ctx = _form_context(request, sess, act, corrections.current_values(visit), {})
    return _harden(TEMPLATES.TemplateResponse(request, "correct.html", ctx))


@router.post("/correct/{activity_id}")
async def correct_submit(
    request: Request,
    activity_id: str,
    csrf: str = Form(""),
    weight_g: str = Form(""),
    value_inr: str = Form(""),
    rate_inr: str = Form(""),
    advance_inr: str = Form(""),
    delivery_date: str = Form(""),
    followup_date: str = Form(""),
) -> Response:
    sess = await _auth(request)
    if isinstance(sess, Response):
        return sess
    import hmac

    if not hmac.compare_digest(csrf, csrf_token(_secret(request), sess.token, activity_id)):
        raise HTTPException(status_code=403, detail="bad CSRF token")
    form = {"weight_g": weight_g, "value_inr": value_inr, "rate_inr": rate_inr,
            "advance_inr": advance_inr, "delivery_date": delivery_date, "followup_date": followup_date}  # fmt: skip
    pack: Pack = request.app.state.pack if hasattr(request.app.state, "pack") else load_pack()
    async with request.app.state.session_factory() as db, db.begin():
        repo = TenantRepository(db, sess.tenant_id)
        act = await repo.get(ActivityLog, uuid.UUID(hex=activity_id))
        boss = await db.get(User, sess.user_id)
        if act is None or boss is None:
            raise HTTPException(status_code=404)
        if act.status not in corrections.CORRECTABLE:
            raise HTTPException(status_code=409, detail="only confirmed records can be corrected")
        parsed = corrections.parse_form(form, visit_of(act), pack)
        if parsed.errors or not parsed.changes:
            errors = parsed.errors or {"_": "Nothing was changed"}
            ctx = _form_context(
                request, sess, act, {**corrections.current_values(visit_of(act)), **form}, errors
            )
            return _harden(
                TEMPLATES.TemplateResponse(request, "correct.html", ctx, status_code=422)
            )
        await corrections.apply_correction(repo, act, boss, parsed.changes, _now(request))
    return _harden(RedirectResponse("/dashboard", status_code=303))
