"""/sheet (boss only): connect the tenant's Google Sheet. Read back before it is applied, then
verified in the background so the boss learns right away if the sheet was not shared."""

from __future__ import annotations

import re

from app.bot.context import Ctx
from app.bot.states import consume, expiry
from app.models import ConversationState, SyncJob, Tenant
from app.models.enums import Role, SyncStatus
from app.pipeline.confirm import TenantSettings
from app.schemas.events import Button, OutMessage

SHEET_CONFIRM_KIND = "sheet_confirm"
_URL_ID = re.compile(r"/d/([A-Za-z0-9_-]{20,})")
_RAW_ID = re.compile(r"[A-Za-z0-9_-]{20,}")


def parse_sheet_id(text: str) -> str | None:
    """Accepts a full Google Sheets URL or the bare spreadsheet id."""
    text = text.strip()
    m = _URL_ID.search(text)
    if m:
        return m.group(1)
    return text if _RAW_ID.fullmatch(text) else None


def sheet_url(spreadsheet_id: str) -> str:
    return f"https://docs.google.com/spreadsheets/d/{spreadsheet_id}"


async def handle_sheet_command(ctx: Ctx, arg: str) -> None:
    assert ctx.user is not None and ctx.deps is not None
    if ctx.user.role is not Role.BOSS:
        ctx.say(ctx.t("boss_only"))
        return
    sheets = ctx.deps.sheets
    if sheets is None:
        ctx.say(ctx.t("sheet_no_provider"))
        return
    tenant = await ctx.session.get(Tenant, ctx.user.tenant_id)
    assert tenant is not None
    current = TenantSettings.from_json(tenant.settings).sheet_spreadsheet_id
    if not arg:
        if current is None:
            ctx.say(ctx.t("sheet_none", email=sheets.service_account_email))
            return
        jobs = await ctx.repo.list(SyncJob)
        pending = sum(1 for j in jobs if j.status is SyncStatus.PENDING)
        failed = sum(1 for j in jobs if j.status is SyncStatus.FAILED)
        ctx.say(ctx.t("sheet_current", url=sheet_url(current), pending=pending, failed=failed))
        return
    sid = parse_sheet_id(arg)
    if sid is None:
        ctx.say(ctx.t("sheet_bad"))
        return
    state = ctx.repo.add(
        ConversationState(
            user_id=ctx.user.id, kind=SHEET_CONFIRM_KIND, data={"id": sid}, expires_at=expiry(ctx)
        )
    )
    await ctx.repo.flush()
    h = state.id.hex
    ctx.say(
        ctx.t("sheet_confirm", id=sheet_url(sid)),
        (Button(ctx.t("btn_yes"), f"sh:y:{h}"), Button(ctx.t("btn_no"), f"sh:n:{h}")),
    )


async def handle_callback(ctx: Ctx, action: str, raw_id: str) -> None:
    assert ctx.user is not None and ctx.deps is not None
    if ctx.user.role is not Role.BOSS:
        ctx.say(ctx.t("boss_only"))
        return
    state = await consume(ctx, raw_id, (SHEET_CONFIRM_KIND,))
    if state is None:
        ctx.callback_text = ctx.t("state_expired")
        return
    if action != "y":
        ctx.say(ctx.t("addworker_cancelled"))
        ctx.callback_text = ctx.t("action_toast_cancelled")
        return
    sid = str(state.data["id"])
    tenant = await ctx.session.get(Tenant, ctx.user.tenant_id)
    assert tenant is not None
    tenant.settings = {**tenant.settings, "sheet_spreadsheet_id": sid}  # reassign: JSON column
    ctx.repo.audit(
        "sheet_configured", entity="tenant", entity_id=str(tenant.id), actor_user_id=ctx.user.id,
        detail={"spreadsheet_id": sid},
    )  # fmt: skip
    ctx.say(ctx.t("sheet_set"))
    ctx.callback_text = ctx.t("action_toast_done")
    sheets, pack, chat = ctx.deps.sheets, ctx.pack, ctx.event.chat_id

    async def verify() -> list[OutMessage]:
        assert sheets is not None
        ok = await sheets.check_access(sid)
        key = "sheet_check_ok" if ok else "sheet_check_fail"
        return [OutMessage(chat, pack.t(key, email=sheets.service_account_email))]

    if sheets is not None:
        ctx.defer(verify)
