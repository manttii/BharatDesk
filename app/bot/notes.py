"""Voice/text notes end to end: intake -> transcript -> extraction -> read-back -> commit.

Flow (one rep message):
  Phase A  (inside the event transaction) validate, store a Message row, defer the job.
  Phase B  (no DB lock held)            download, speech-to-text, LLM extraction.
  Phase C  (new transaction)            persist the Extraction, resolve the client, ask ONE
                                        missing-field question or read the values back.
  Callback (Haan / Nahi / Sudharo)      commit through the policy engine, or discard, or redo.

Nothing is committed without a tap unless auto-confirm is enabled for the tenant AND every
check passes (see pipeline/confirm.py).
"""

from __future__ import annotations

import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import date
from zoneinfo import ZoneInfo

import structlog

from app.bot import approvals_ui, benchmark, shop_ui
from app.bot.context import Ctx
from app.bot.deps import Deps
from app.bot.states import consume, expiry
from app.models import (
    ActivityLog,
    Client,
    ConversationState,
    Correction,
    Extraction,
    Message,
    Tenant,
    User,
)
from app.models.enums import ActivityStatus, ProcessingState
from app.pipeline import commit
from app.pipeline.assess import (
    REQUIRED_CLIENT,
    REQUIRED_ORDER_VALUE,
    REQUIRED_RATE,
    assess_visit,
    missing_fields,
)
from app.pipeline.confirm import TenantSettings, can_auto_confirm, readback_message
from app.pipeline.extract import ExtractionRun, StructuredRun, run_extraction
from app.pipeline.media import (
    MediaError,
    sha256_hex,
    sniff_audio_extension,
    sniff_image,
    validate_photo_event,
    validate_voice_event,
)
from app.pipeline.numerals import parse_quantities, to_paise
from app.pipeline.slip import assess_slip, run_slip_extraction
from app.pipeline.transcribe import transcribe_note
from app.providers.llm import ImagePart
from app.providers.messaging import DownloadError
from app.providers.stt import STTError
from app.schemas.events import Button, IncomingEvent, OutMessage
from app.schemas.extraction import Flag, Intent, VisitPayload
from app.schemas.slip import LLMSlip
from app.services import approvals
from app.services.clients import match_client
from app.services.dossier import build_dossier, dossier_text
from app.units import ceil_div

log = structlog.get_logger()

SLOT_KIND, CORRECTION_KIND, CONFIRM_KIND, CHOICE_KIND = (
    "slot",
    "correction",
    "confirm",
    "client_choice",
)
TEXT_CONSUMING = (SLOT_KIND, CORRECTION_KIND, approvals_ui.COUNTER_KIND)
MAX_TEXT_CHARS = 2000
MAX_SLOT_ATTEMPTS = 2

_ASK_KEYS = {
    REQUIRED_CLIENT: "ask_client",
    REQUIRED_ORDER_VALUE: "ask_value",
    REQUIRED_RATE: "ask_rate",
}


@dataclass(frozen=True)
class NoteJob:
    event: IncomingEvent
    tenant_id: uuid.UUID
    user_id: uuid.UUID
    message_id: uuid.UUID
    kind: str  # "voice" | "text"
    file_id: str | None
    mime_type: str | None
    duration_s: int | None
    text: str | None
    state_id: uuid.UUID | None
    state_kind: str | None


# --------------------------------------------------------------------------- state helpers


async def active_state(ctx: Ctx) -> ConversationState | None:
    assert ctx.user is not None
    rows = await ctx.repo.list(
        ConversationState,
        ConversationState.user_id == ctx.user.id,
        ConversationState.kind.in_(TEXT_CONSUMING),
        ConversationState.expires_at > ctx.now,
        order_by=[ConversationState.created_at.desc()],
    )
    return rows[0] if rows else None


# --------------------------------------------------------------------------- Phase A


def _voice_error_text(ctx: Ctx, code: str) -> str:
    if code == "too_long":
        return ctx.t("voice_too_long", max_s=ctx.settings.max_voice_seconds)
    return ctx.t(
        {"too_short": "voice_too_short", "too_large": "media_too_large"}.get(code, "voice_bad_type")
    )


async def accept_voice(ctx: Ctx) -> None:
    assert ctx.user is not None and ctx.deps is not None
    deps = ctx.deps
    if deps.stt is None or deps.llm is None or deps.media is None:
        ctx.say(ctx.t("stt_unavailable"))
        return
    try:
        validate_voice_event(ctx.event, ctx.settings)
    except MediaError as exc:
        ctx.say(_voice_error_text(ctx, exc.code))
        return
    await _schedule(ctx, kind="voice", text=None)


async def accept_photo(ctx: Ctx) -> None:
    """A photo of a handwritten slip. It never answers an open question: it is a new note."""
    assert ctx.user is not None and ctx.deps is not None
    deps = ctx.deps
    if deps.llm is None or deps.media is None:
        ctx.say(ctx.t("not_configured"))
        return
    try:
        validate_photo_event(ctx.event, ctx.settings)
    except MediaError as exc:
        ctx.say(ctx.t(f"photo_{exc.code}"))
        return
    await _schedule(ctx, kind="photo", text=None, use_state=False)


async def accept_text(ctx: Ctx, text: str) -> None:
    assert ctx.user is not None and ctx.deps is not None
    state = await active_state(ctx)
    if state is not None and state.kind == SLOT_KIND:
        await answer_slot(ctx, state, text)  # deterministic, no network
        return
    if state is not None and state.kind == approvals_ui.COUNTER_KIND:
        await approvals_ui.answer_counter(ctx, state, text)
        return
    query = shop_ui.classify_query(text)
    if query is not None:  # a short, clear question about stock or today's sales
        await shop_ui.answer(ctx, query)
        return
    if ctx.deps.llm is None:
        ctx.say(ctx.t("not_configured"))
        return
    await _schedule(ctx, kind="text", text=text[:MAX_TEXT_CHARS], state=state)


async def _schedule(
    ctx: Ctx,
    *,
    kind: str,
    text: str | None,
    state: ConversationState | None = None,
    use_state: bool = True,
) -> None:
    assert ctx.user is not None and ctx.deps is not None
    if state is None and use_state:
        state = await active_state(ctx)
    ev = ctx.event
    msg = ctx.repo.add(
        Message(
            created_at=ctx.now,
            user_id=ctx.user.id,
            update_id=ev.update_id,
            kind=kind,
            file_id=ev.file_id,
            processing_state=ProcessingState.PROCESSING,
        )
    )
    await ctx.repo.flush()
    job = NoteJob(
        event=ev,
        tenant_id=ctx.user.tenant_id,
        user_id=ctx.user.id,
        message_id=msg.id,
        kind=kind,
        file_id=ev.file_id,
        mime_type=ev.mime_type,
        duration_s=ev.duration_s,
        text=text,
        state_id=state.id if state else None,
        state_kind=state.kind if state else None,
    )
    deps = ctx.deps
    ctx.defer(lambda: run_note_job(deps, job))


# --------------------------------------------------------------------------- Phase B / C


async def _in_txn(
    deps: Deps, job: NoteJob, fn: Callable[[Ctx], Awaitable[None]]
) -> list[OutMessage]:
    async with deps.session_factory() as session, session.begin():
        user = await session.get(User, job.user_id)
        ctx = Ctx(
            session=session,
            settings=deps.settings,
            pack=deps.pack,
            event=job.event,
            now=deps.clock(),
            bot_username=deps.messaging.username,
            deps=deps,
            user=user,
        )
        await fn(ctx)
        return ctx.replies


async def _set_message(
    deps: Deps, job: NoteJob, state: ProcessingState, sha: str | None = None
) -> None:
    async def go(ctx: Ctx) -> None:
        msg = await ctx.repo.get(Message, job.message_id)
        if msg is not None:
            msg.processing_state = state
            if sha:
                msg.sha256 = sha

    await _in_txn(deps, job, go)


async def _client_names(deps: Deps, job: NoteJob) -> list[str]:
    names: list[str] = []

    async def go(ctx: Ctx) -> None:
        names.extend(
            c.name for c in await ctx.repo.list(Client, order_by=[Client.created_at.desc()])
        )

    await _in_txn(deps, job, go)
    return names


async def run_note_job(deps: Deps, job: NoteJob) -> list[OutMessage]:
    chat = job.event.chat_id
    pack = deps.pack

    def say(key: str, **kw: object) -> list[OutMessage]:
        return [OutMessage(chat, pack.t(key, **kw))]

    try:
        await deps.messaging.send_typing(chat)
        if job.kind == "photo":
            return await _run_photo_job(deps, job)
        names = await _client_names(deps, job)
        reasons: tuple[str, ...] = ()
        stt_ms, stt_model = 0, None
        if job.kind == "voice":
            assert job.file_id and deps.stt and deps.media
            try:
                data = await deps.messaging.download(job.file_id, deps.settings.max_media_bytes)
            except DownloadError:
                await _set_message(deps, job, ProcessingState.FAILED)
                return say("media_download_failed")
            ext = sniff_audio_extension(data)
            if ext is None:
                await _set_message(deps, job, ProcessingState.REJECTED)
                return say("voice_bad_type")
            sha = sha256_hex(data)
            deps.media.save(job.tenant_id, sha, ext, data)
            try:
                tr = await transcribe_note(
                    deps.stt,
                    data,
                    filename=f"voice.{ext}",
                    mime_type=job.mime_type,
                    client_names=names,
                    vocabulary=deps.pack.vocabulary,
                    settings=deps.settings,
                )
            except STTError as exc:
                log.warning("stt_failed", status=exc.status)
                await _set_message(deps, job, ProcessingState.FAILED, sha)
                return say("stt_failed")
            text, reasons, stt_ms, stt_model = tr.text, tr.reasons, tr.latency_ms, tr.model
            deps.stage("media", sha256=sha, bytes=len(data), ext=ext)
            deps.stage(
                "transcript",
                text=text,
                reasons=list(reasons),
                attempts=tr.attempts,
                model=tr.model,
                avg_logprob=tr.avg_logprob,
                no_speech_prob=tr.no_speech_prob,
            )
            await _set_message(deps, job, ProcessingState.PROCESSING, sha)
            if not text.strip():
                await _set_message(deps, job, ProcessingState.FAILED)
                return say("empty_transcript")
        else:
            text = job.text or ""
            deps.stage("transcript", text=text, reasons=[], attempts=0, model="text-input")

        if job.state_kind in (SLOT_KIND, approvals_ui.COUNTER_KIND):
            return await _in_txn(deps, job, lambda ctx: _answer_state(ctx, job, text))

        assert deps.llm is not None
        deps.stage(
            "numerals",
            quantities=[
                {"text": q.text, "value": str(q.value), "unit": q.unit.value if q.unit else None}
                for q in parse_quantities(text)
            ],
        )
        tz, today = await _tenant_today(deps, job)
        run = await run_extraction(deps.llm, text, today=today, timezone=tz, known_clients=names)
        deps.stage(
            "llm",
            raw=run.raw_text,
            model=run.model,
            prompt_version=run.prompt_version,
            attempts=run.attempts,
            error=run.error,
            latency_ms=run.latency_ms,
        )
        return await _in_txn(
            deps,
            job,
            lambda ctx: handle_extraction(ctx, job, text, reasons, run, stt_ms, stt_model),
        )
    except Exception:
        log.exception("note_job_failed", chat_id=chat)
        return say("error_generic")


async def _run_photo_job(deps: Deps, job: NoteJob) -> list[OutMessage]:
    chat = job.event.chat_id
    pack = deps.pack
    assert job.file_id and deps.llm and deps.media
    try:
        data = await deps.messaging.download(job.file_id, deps.settings.max_photo_bytes)
    except DownloadError:
        await _set_message(deps, job, ProcessingState.FAILED)
        return [OutMessage(chat, pack.t("media_download_failed"))]
    kind = sniff_image(data)
    if kind is None:
        await _set_message(deps, job, ProcessingState.REJECTED)
        return [OutMessage(chat, pack.t("photo_bad_type"))]
    ext, mime = kind
    sha = sha256_hex(data)

    duplicate: list[bool] = []

    async def check(ctx: Ctx) -> None:
        earlier = await ctx.repo.list(
            Message,
            Message.sha256 == sha,
            Message.id != job.message_id,
            Message.processing_state == ProcessingState.DONE,
        )
        duplicate.append(bool(earlier))
        msg = await ctx.repo.get(Message, job.message_id)
        if msg is not None:
            msg.sha256 = sha
            if earlier:
                msg.processing_state = ProcessingState.REJECTED

    await _in_txn(deps, job, check)
    if duplicate[0]:  # the same slip photo sent twice must not become two orders
        return [OutMessage(chat, pack.t("photo_duplicate"))]
    deps.media.save(job.tenant_id, sha, ext, data)
    deps.stage("media", sha256=sha, bytes=len(data), ext=ext, kind="photo")

    tz, today = await _tenant_today(deps, job)
    run = await run_slip_extraction(deps.llm, ImagePart(data, mime), today=today, timezone=tz)
    deps.stage(
        "slip_llm", raw=run.raw_text, model=run.model, prompt_version=run.prompt_version,
        attempts=run.attempts, error=run.error, latency_ms=run.latency_ms,
    )  # fmt: skip
    return await _in_txn(deps, job, lambda ctx: handle_slip_extraction(ctx, job, run))


async def handle_slip_extraction(ctx: Ctx, job: NoteJob, run: StructuredRun[LLMSlip]) -> None:
    assert ctx.user is not None
    repo = ctx.repo
    msg = await repo.get(Message, job.message_id)
    extraction = repo.add(
        Extraction(
            message_id=job.message_id,
            llm_model=run.model,
            prompt_version=run.prompt_version,
            raw_output={"text": run.raw_text, "error": run.error},
            latency_ms=run.latency_ms,
        )
    )
    if run.parsed is None:
        if msg:
            msg.processing_state = ProcessingState.FAILED
        down = (run.error or "").startswith("llm_error")
        ctx.say(ctx.t("llm_unavailable" if down else "extract_failed"))
        return
    tenant = await ctx.session.get(Tenant, ctx.user.tenant_id)
    assert tenant is not None
    today = ctx.now.astimezone(ZoneInfo(tenant.timezone)).date()
    a = assess_slip(
        run.parsed,
        today=today,
        pack=ctx.pack,
        min_confidence=ctx.settings.low_confidence_flag,
        min_legibility=ctx.settings.slip_min_legibility,
    )
    if a.unreadable or a.payload is None:
        if msg:
            msg.processing_state = ProcessingState.FAILED  # re-sent clear photo is a new message
        ctx.say(ctx.t("photo_unclear"))
        if ctx.deps:
            ctx.deps.stage("slip_assessment", unreadable=True)
        return
    if msg:
        msg.processing_state = ProcessingState.DONE
    extraction.validated_payload = a.payload.model_dump(mode="json")
    extraction.field_confidences = a.confidences
    meta = {
        "flags": [f.model_dump() for f in a.flags],
        "confidences": a.confidences,
        "client_new": False,
        "client_decided": False,
        "spoken_client": a.payload.client_name,
        "slot_attempts": 0,
        "source": "slip",
        "slot_question": a.slot_question,
    }
    if ctx.deps:
        ctx.deps.stage(
            "slip_assessment", payload=a.payload.model_dump(mode="json"), flags=meta["flags"],
            written_total_paise=a.written_total_paise, computed_total_paise=a.computed_total_paise,
        )  # fmt: skip
    activity = repo.add(
        ActivityLog(
            created_at=ctx.now,
            user_id=ctx.user.id,
            message_id=job.message_id,
            intent=Intent.LOG_VISIT.value,
            status=ActivityStatus.PENDING_CONFIRMATION,
            payload={},
        )
    )
    approvals.with_visit(activity, a.payload, **meta)
    await repo.flush()
    await advance(ctx, activity)


async def _tenant_today(deps: Deps, job: NoteJob) -> tuple[str, date]:
    out: list[tuple[str, date]] = []

    async def go(ctx: Ctx) -> None:
        assert ctx.user is not None
        tenant = await ctx.session.get(Tenant, ctx.user.tenant_id)
        assert tenant is not None
        out.append((tenant.timezone, ctx.now.astimezone(ZoneInfo(tenant.timezone)).date()))

    await _in_txn(deps, job, go)
    return out[0]


async def _answer_state(ctx: Ctx, job: NoteJob, text: str) -> None:
    assert job.state_id is not None
    state = await ctx.repo.first(
        ConversationState,
        ConversationState.id == job.state_id,
        ConversationState.expires_at > ctx.now,
    )
    if state is None:
        ctx.say(ctx.t("state_expired"))
        return
    if state.kind == SLOT_KIND:
        await answer_slot(ctx, state, text)
    else:
        await approvals_ui.answer_counter(ctx, state, text)


def _estimate_cost(ctx: Ctx, duration_s: int | None, run: ExtractionRun) -> int | None:
    parts: list[int] = []
    s = ctx.settings
    if s.stt_cost_paise_per_minute is not None and duration_s is not None:
        parts.append(ceil_div(s.stt_cost_paise_per_minute * duration_s, 60))
    if s.llm_cost_paise_per_1k_tokens is not None and (run.prompt_tokens or run.output_tokens):
        parts.append(
            ceil_div(
                s.llm_cost_paise_per_1k_tokens
                * ((run.prompt_tokens or 0) + (run.output_tokens or 0)),
                1000,
            )
        )
    return sum(parts) if parts else None


async def handle_extraction(
    ctx: Ctx,
    job: NoteJob,
    transcript: str,
    reasons: tuple[str, ...],
    run: ExtractionRun,
    stt_ms: int,
    stt_model: str | None,
) -> None:
    assert ctx.user is not None
    repo = ctx.repo
    msg = await repo.get(Message, job.message_id)
    extraction = repo.add(
        Extraction(
            message_id=job.message_id,
            transcript=transcript,
            stt_model=stt_model,
            llm_model=run.model,
            prompt_version=run.prompt_version,
            raw_output={"text": run.raw_text, "error": run.error},
            latency_ms=stt_ms + run.latency_ms,
            cost_paise=_estimate_cost(ctx, job.duration_s, run),
        )
    )
    if run.parsed is None:
        if msg:
            msg.processing_state = ProcessingState.FAILED
        down = (run.error or "").startswith("llm_error")  # provider failure vs. unusable output
        ctx.say(ctx.t("llm_unavailable" if down else "extract_failed"))
        return
    if msg:
        msg.processing_state = ProcessingState.DONE
    parsed = run.parsed
    if parsed.intent is Intent.HELP:
        ctx.say(ctx.t("help_boss" if ctx.user.role.value == "boss" else "help_worker"))
        return
    if parsed.intent is Intent.UNKNOWN:
        ctx.say(ctx.t("unknown_intent"))
        return
    if parsed.intent is Intent.SET_BENCHMARK:
        await benchmark.from_extraction(ctx, parsed, transcript)
        return
    if parsed.intent is Intent.CLIENT_DOSSIER_QUERY:
        await dossier_reply(ctx, parsed.client_name)
        return
    if parsed.intent is not Intent.LOG_VISIT:
        ctx.say(ctx.t("intent_unsupported"))
        return

    tenant = await ctx.session.get(Tenant, ctx.user.tenant_id)
    assert tenant is not None
    today = ctx.now.astimezone(ZoneInfo(tenant.timezone)).date()
    required = await commit.required_by_policy(repo)
    a = assess_visit(
        parsed,
        transcript,
        today=today,
        pack=ctx.pack,
        min_confidence=ctx.settings.low_confidence_flag,
        transcript_reasons=reasons,
        rate_required=required.rate,
        value_required=required.value,
    )
    extraction.validated_payload = a.payload.model_dump(mode="json")
    extraction.field_confidences = a.confidences
    meta = {
        "flags": [f.model_dump() for f in a.flags],
        "confidences": a.confidences,
        "client_new": False,
        "client_decided": False,
        "spoken_client": a.payload.client_name,
        "slot_attempts": 0,
    }
    if ctx.deps:
        ctx.deps.stage(
            "assessment",
            payload=a.payload.model_dump(mode="json"),
            flags=meta["flags"],
            missing=a.missing,
            confidences=a.confidences,
        )

    activity = await _target_activity(ctx, job)
    if activity is not None:  # Sudharo: the corrected note replaces the pending one
        original = dict(activity.payload.get("visit", {}))
        repo.add(
            Correction(
                created_at=ctx.now,
                activity_log_id=activity.id,
                original=original,
                corrected=a.payload.model_dump(mode="json"),
                corrected_by=ctx.user.id,
            )
        )
        activity.message_id = job.message_id
        repo.audit(
            "activity_corrected",
            entity="activity_log",
            entity_id=str(activity.id),
            actor_user_id=ctx.user.id,
        )
    else:
        activity = repo.add(
            ActivityLog(
                created_at=ctx.now,
                user_id=ctx.user.id,
                message_id=job.message_id,
                intent=Intent.LOG_VISIT.value,
                status=ActivityStatus.PENDING_CONFIRMATION,
                payload={},
            )
        )
    approvals.with_visit(activity, a.payload, **meta)
    await repo.flush()
    await advance(ctx, activity)


async def dossier_reply(ctx: Ctx, name: str | None) -> None:
    """3-line briefing from stored visits; ambiguity is asked about, never guessed."""
    assert ctx.user is not None
    if not name or not name.strip():
        ctx.say(ctx.t("dossier_ask_name"))
        return
    m = await match_client(ctx.repo, name)
    if m.status == "new":
        ctx.say(ctx.t("dossier_not_found", name=name.strip()))
    elif m.status == "ambiguous":
        ctx.say(ctx.t("dossier_ambiguous", names=" / ".join(c.name for c in m.candidates)))
    else:
        assert m.client is not None
        tenant = await ctx.session.get(Tenant, ctx.user.tenant_id)
        assert tenant is not None
        today = ctx.now.astimezone(ZoneInfo(tenant.timezone)).date()
        ctx.say(dossier_text(await build_dossier(ctx.repo, m.client, today=today), ctx.pack, today))


async def _target_activity(ctx: Ctx, job: NoteJob) -> ActivityLog | None:
    """For a Sudharo (correction) note, the pending activity being replaced."""
    if job.state_kind != CORRECTION_KIND or job.state_id is None:
        return None
    state = await ctx.repo.first(
        ConversationState,
        ConversationState.id == job.state_id,
        ConversationState.expires_at > ctx.now,
    )
    if state is None:
        return None
    assert ctx.user is not None
    await ctx.repo.delete_where(ConversationState, ConversationState.id == state.id)
    activity = await ctx.repo.get(ActivityLog, uuid.UUID(hex=str(state.data["activity_id"])))
    if (
        activity is None
        or activity.status is not ActivityStatus.PENDING_CONFIRMATION
        or activity.user_id != ctx.user.id
    ):
        return None
    return activity


# --------------------------------------------------------------------------- advance


def _meta(activity: ActivityLog) -> dict[str, object]:
    return dict(activity.payload.get("meta", {}))


async def advance(ctx: Ctx, activity: ActivityLog) -> None:
    """Decide the next step for a pending visit: resolve the client, ask ONE missing field,
    then read back (or auto-confirm if every condition holds)."""
    assert ctx.user is not None
    repo = ctx.repo
    payload = approvals.visit_of(activity)
    meta = _meta(activity)

    if payload.client_name and payload.client_id is None and not meta.get("client_decided"):
        m = await match_client(repo, payload.client_name)
        if m.status in ("exact", "likely") and m.client is not None:
            payload = payload.model_copy(
                update={"client_id": str(m.client.id), "client_name": m.client.name}
            )
            meta.update(client_decided=True, client_exact=m.status == "exact")
        elif m.status == "ambiguous":
            approvals.with_visit(activity, payload, **meta)
            await _ask_client_choice(
                ctx, activity, payload.client_name or "", [(c.id.hex, c.name) for c in m.candidates]
            )
            return
        else:
            meta.update(client_decided=True, client_new=True, client_exact=False)
        approvals.with_visit(activity, payload, **meta)

    required = await commit.required_by_policy(repo)
    missing = missing_fields(
        payload,
        rate_required=required.rate,
        # a slip is about its total, so a slip without one is always asked for it
        value_required=required.value or meta.get("source") == "slip",
    )
    if missing:
        await _ask_slot(ctx, activity, missing[0], attempts=int(meta.get("slot_attempts", 0)))  # type: ignore[call-overload]
        return

    flags = [Flag.model_validate(f) for f in meta.get("flags", [])]  # type: ignore[attr-defined]
    tenant = await ctx.session.get(Tenant, ctx.user.tenant_id)
    assert tenant is not None
    tsettings = TenantSettings.from_json(tenant.settings)
    confidences: dict[str, float] = dict(meta.get("confidences", {}))  # type: ignore[call-overload]
    if can_auto_confirm(
        flags, confidences, tsettings, client_exact=bool(meta.get("client_exact")), missing=missing
    ):
        await finalize(ctx, activity)
        return

    state = repo.add(
        ConversationState(
            user_id=ctx.user.id,
            kind=CONFIRM_KIND,
            data={"activity_id": activity.id.hex},
            expires_at=expiry(ctx),
        )
    )
    await repo.flush()
    h = state.id.hex
    ctx.say(
        readback_message(
            payload, flags, ctx.pack, new_client=bool(meta.get("client_new")), today=ctx.now.date()
        ),
        (
            Button(ctx.t("btn_yes"), f"cf:y:{h}"),
            Button(ctx.t("btn_no"), f"cf:n:{h}"),
            Button(ctx.t("btn_edit"), f"cf:e:{h}"),
        ),
    )


async def _ask_client_choice(
    ctx: Ctx, activity: ActivityLog, spoken: str, candidates: list[tuple[str, str]]
) -> None:
    assert ctx.user is not None
    state = ctx.repo.add(
        ConversationState(
            user_id=ctx.user.id,
            kind=CHOICE_KIND,
            data={"activity_id": activity.id.hex, "candidates": [c[0] for c in candidates]},
            expires_at=expiry(ctx),
        )
    )
    await ctx.repo.flush()
    h = state.id.hex
    rows = tuple((Button(name, f"cl:{i}:{h}"),) for i, (_, name) in enumerate(candidates))
    ctx.say(
        ctx.t("ask_client_choice", name=spoken),
        *rows,
        (Button(ctx.t("btn_new_client"), f"cl:n:{h}"),),
    )


async def _ask_slot(
    ctx: Ctx, activity: ActivityLog, field: str, *, attempts: int, retry: bool = False
) -> None:
    assert ctx.user is not None
    ctx.repo.add(
        ConversationState(
            user_id=ctx.user.id,
            kind=SLOT_KIND,
            data={"activity_id": activity.id.hex, "field": field, "attempts": attempts},
            expires_at=expiry(ctx),
        )
    )
    question = ctx.t(_ASK_KEYS[field])
    meta = _meta(activity)
    custom = meta.get("slot_question")
    if isinstance(custom, dict) and custom.get("field") == field:
        # e.g. a slip whose written total disagrees with its lines: say exactly what disagrees
        fmt = {k: v for k, v in custom.items() if k not in ("field", "key")}
        question = ctx.t(str(custom["key"]), **fmt)
        approvals.with_visit(activity, approvals.visit_of(activity), slot_question=None)
    ctx.say(ctx.t("ask_retry", question=question) if retry else question)


# --------------------------------------------------------------------------- slot answers


def _clean_client_name(text: str) -> str | None:
    name = " ".join(text.split())
    if not (2 <= len(name) <= 120) or not name.isprintable():
        return None
    return name


def _parse_amount(ctx: Ctx, text: str, lo_key: str, hi_key: str) -> int | None:
    quantities = parse_quantities(text)
    if len(quantities) != 1:
        return None
    paise = to_paise(quantities[0], unitless_as_rupees=True)
    if paise is None:
        return None
    lo, hi = ctx.pack.bounds.get(lo_key), ctx.pack.bounds.get(hi_key)
    if (lo is not None and paise < lo) or (hi is not None and paise > hi):
        return None
    return paise


async def answer_slot(ctx: Ctx, state: ConversationState, text: str) -> None:
    """Merge the rep's answer to the one question we asked. Parsed by code (numerals,
    bounds), not by the LLM; a failed parse re-asks once, then gives up cleanly."""
    assert ctx.user is not None
    repo = ctx.repo
    field = str(state.data["field"])
    attempts = int(state.data.get("attempts", 0))
    activity = await repo.get(ActivityLog, uuid.UUID(hex=str(state.data["activity_id"])))
    if activity is None or activity.status is not ActivityStatus.PENDING_CONFIRMATION:
        await repo.delete_where(ConversationState, ConversationState.id == state.id)
        ctx.say(ctx.t("state_expired"))
        return
    payload = approvals.visit_of(activity)
    meta = _meta(activity)
    updated: VisitPayload | None = None
    if field == REQUIRED_CLIENT:
        name = _clean_client_name(text)
        if name:
            updated = payload.model_copy(update={"client_name": name, "client_id": None})
            meta.update(client_decided=False, spoken_client=name)
    elif field in (REQUIRED_ORDER_VALUE, REQUIRED_RATE) and payload.order is not None:
        lo, hi = (
            ("value_paise_min", "value_paise_max")
            if field == REQUIRED_ORDER_VALUE
            else ("rate_paise_per_10g_min", "rate_paise_per_10g_max")
        )
        paise = _parse_amount(ctx, text, lo, hi)
        if paise is not None:
            key = "value_paise" if field == REQUIRED_ORDER_VALUE else "rate_paise_per_10g"
            updated = payload.model_copy(
                update={"order": payload.order.model_copy(update={key: paise})}
            )
    await repo.delete_where(ConversationState, ConversationState.id == state.id)
    if updated is None:
        attempts += 1
        if attempts >= MAX_SLOT_ATTEMPTS:
            activity.status = ActivityStatus.REJECTED
            repo.audit(
                "activity_abandoned",
                entity="activity_log",
                entity_id=str(activity.id),
                actor_user_id=ctx.user.id,
            )
            ctx.say(ctx.t("slot_giveup"))
            return
        meta["slot_attempts"] = attempts
        approvals.with_visit(activity, payload, **meta)
        await _ask_slot(ctx, activity, field, attempts=attempts, retry=True)
        return
    meta["flags"] = [f for f in meta.get("flags", []) if f["field"] != field]  # type: ignore[attr-defined]
    meta["slot_attempts"] = 0
    approvals.with_visit(activity, updated, **meta)
    await repo.flush()
    await advance(ctx, activity)


# --------------------------------------------------------------------------- callbacks


async def _pending_activity(ctx: Ctx, state: ConversationState) -> ActivityLog | None:
    activity = await ctx.repo.get(ActivityLog, uuid.UUID(hex=str(state.data["activity_id"])))
    if activity is None or activity.status is not ActivityStatus.PENDING_CONFIRMATION:
        return None
    return activity


async def finalize(ctx: Ctx, activity: ActivityLog) -> None:
    assert ctx.user is not None
    tenant = await ctx.session.get(Tenant, ctx.user.tenant_id)
    assert tenant is not None
    payload = approvals.visit_of(activity)
    meta = _meta(activity)
    excerpt = ""
    if activity.message_id is not None:
        ex = await ctx.repo.first(Extraction, Extraction.message_id == activity.message_id)
        excerpt = (ex.transcript or "") if ex else ""
    res = await commit.commit_visit(
        ctx.repo,
        user=ctx.user,
        tenant=tenant,
        activity=activity,
        payload=payload,
        pack=ctx.pack,
        settings=ctx.settings,
        now=ctx.now,
        excerpt=excerpt,
        spoken_client=str(meta.get("spoken_client") or "") or None,
    )
    ctx.extend(res.replies)
    if ctx.deps:
        ctx.deps.stage("commit", status=res.status.value, approval_id=res.approval_id)


async def handle_confirm_callback(ctx: Ctx, action: str, raw_id: str) -> None:
    state = await consume(ctx, raw_id, (CONFIRM_KIND,))
    if state is None:
        ctx.callback_text = ctx.t("state_expired")
        return
    activity = await _pending_activity(ctx, state)
    if activity is None:
        ctx.callback_text = ctx.t("action_expired")
        return
    assert ctx.user is not None
    if action == "y":
        await finalize(ctx, activity)
        ctx.callback_text = ctx.t("action_toast_done")
    elif action == "n":
        activity.status = ActivityStatus.REJECTED
        ctx.repo.audit(
            "activity_discarded",
            entity="activity_log",
            entity_id=str(activity.id),
            actor_user_id=ctx.user.id,
        )
        ctx.say(ctx.t("discarded"))
        ctx.callback_text = ctx.t("action_toast_cancelled")
    else:  # "e": Sudharo; the next note (voice or text) replaces this one
        ctx.repo.add(
            ConversationState(
                user_id=ctx.user.id,
                kind=CORRECTION_KIND,
                data={"activity_id": activity.id.hex},
                expires_at=expiry(ctx),
            )
        )
        ctx.say(ctx.t("ask_correction"))


async def handle_client_choice_callback(ctx: Ctx, choice: str, raw_id: str) -> None:
    state = await consume(ctx, raw_id, (CHOICE_KIND,))
    if state is None:
        ctx.callback_text = ctx.t("state_expired")
        return
    activity = await _pending_activity(ctx, state)
    if activity is None:
        ctx.callback_text = ctx.t("action_expired")
        return
    payload = approvals.visit_of(activity)
    meta = _meta(activity)
    candidates: list[str] = list(state.data["candidates"])
    if choice == "n":
        meta.update(client_decided=True, client_new=True, client_exact=False)
    else:
        try:
            client = await ctx.repo.get(Client, uuid.UUID(hex=candidates[int(choice)]))
        except (ValueError, IndexError):
            client = None
        if client is None:
            ctx.callback_text = ctx.t("action_expired")
            return
        payload = payload.model_copy(
            update={"client_id": str(client.id), "client_name": client.name}
        )
        meta.update(client_decided=True, client_exact=False)
    approvals.with_visit(activity, payload, **meta)
    await ctx.repo.flush()
    ctx.callback_text = ctx.t("action_toast_done")
    await advance(ctx, activity)


async def cancel_pending(ctx: Ctx) -> None:
    """/cancel: drop every open question/read-back for this user."""
    assert ctx.user is not None
    await ctx.repo.delete_where(ConversationState, ConversationState.user_id == ctx.user.id)
    ctx.say(ctx.t("cancel_done"))
