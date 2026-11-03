"""Bills over Telegram: send a PDF (or a photo captioned "bill"), the bot reads every page, and
the owner confirms the clean ones with one tap and decides the flagged ones one by one.

Anyone on the team may upload; only the boss may confirm, accept or reject. Reading runs after
the database transaction commits (it downloads and calls a model), and the file itself is not
stored, only the extracted fields.
"""

from __future__ import annotations

import re
import uuid
from collections import Counter

from app.bot.context import Ctx
from app.bot.deps import Deps
from app.db import TenantRepository
from app.models import Bill, User
from app.models.enums import Role
from app.pack import Pack
from app.pipeline.documents import DocumentError
from app.providers.messaging import DownloadError
from app.schemas.events import Button, IncomingEvent, OutMessage
from app.services import bill_batch
from app.services.approvals import bosses_to_notify
from app.services.bill_ingest import ingest_document
from app.units import format_inr

MAX_REVIEW_ITEMS = 8  # per message burst; the rest wait for /bills
_CAPTION = re.compile(r"^\s*(bill|invoice|challan)\b", re.IGNORECASE)
_FLAGGED = ("needs_review", "unreadable", "failed")


def is_bill_caption(caption: str | None) -> bool:
    return bool(caption and _CAPTION.match(caption))


def _is_boss(ctx: Ctx) -> bool:
    return ctx.user is not None and ctx.user.role is Role.BOSS


# ---------------------------------------------------------------------- receiving a file


async def accept_document(ctx: Ctx) -> None:
    assert ctx.user is not None and ctx.deps is not None
    deps, ev = ctx.deps, ctx.event
    if deps.llm is None:
        ctx.say(ctx.t("not_configured"))
        return
    if ev.file_id is None:
        ctx.say(ctx.t("bills_pdf_only"))
        return
    if ev.file_size is not None and ev.file_size > ctx.settings.max_pdf_bytes:
        ctx.say(ctx.t("bills_too_big"))
        return
    ctx.say(ctx.t("bills_reading"))
    tenant_id, user_id, chat_id = ctx.user.tenant_id, ctx.user.id, ev.chat_id
    ctx.defer(lambda: _read(deps, ev, tenant_id, user_id, chat_id))


async def _read(
    deps: Deps, ev: IncomingEvent, tenant_id: uuid.UUID, user_id: uuid.UUID, chat_id: int
) -> list[OutMessage]:
    assert ev.file_id is not None and deps.llm is not None
    pack = deps.pack
    try:
        data = await deps.messaging.download(ev.file_id, deps.settings.max_pdf_bytes)
    except DownloadError:
        return [OutMessage(chat_id, pack.t("bills_download_failed"))]
    try:
        summary = await ingest_document(
            deps.session_factory,
            deps.llm,
            deps.settings,
            tenant_id=tenant_id,
            user_id=user_id,
            source_name=(ev.file_name or "telegram upload")[:200],
            data=data,
            now=deps.clock(),
        )
    except DocumentError as exc:
        return [OutMessage(chat_id, pack.t("bills_file_error", reason=str(exc)))]
    async with deps.session_factory() as session:
        repo = TenantRepository(session, tenant_id)
        bills = await repo.list(Bill, Bill.batch_id == summary.batch_id, order_by=[Bill.page_no])
        uploader = await repo.get(User, user_id)
        bosses = await bosses_to_notify(repo)
    counts = Counter(b.status for b in bills)
    headline = pack.t(
        "bills_summary",
        total=len(bills),
        ready=counts["ready"],
        review=counts["needs_review"],
        bad=counts["unreadable"] + counts["failed"],
    )
    if uploader is not None and uploader.role is Role.BOSS:
        return review_messages(pack, chat_id, summary.batch_id, bills, headline)
    out = [OutMessage(chat_id, f"{headline}\n{pack.t('bills_sent_to_boss')}")]
    for boss in bosses:
        if boss.telegram_chat_id is not None:
            out += review_messages(pack, boss.telegram_chat_id, summary.batch_id, bills, headline)
    return out


# ---------------------------------------------------------------------- what the boss sees


def _reasons(pack: Pack, bill: Bill) -> str:
    seen: list[str] = []
    for f in bill.flags:
        text = pack.strings.get(f"bill_flag_{f.get('reason')}") or pack.t("bill_flag_other")
        if text not in seen:
            seen.append(text)
    return ", ".join(seen)


def review_messages(
    pack: Pack, chat_id: int, batch_id: uuid.UUID, bills: list[Bill], headline: str
) -> list[OutMessage]:
    ready = [b for b in bills if b.status == "ready"]
    flagged = [b for b in bills if b.status in _FLAGGED]
    head = headline
    buttons: tuple[tuple[Button, ...], ...] | None = None
    if ready:
        head += "\n" + pack.t("bills_confirm_prompt", ready=len(ready))
        buttons = (
            (Button(pack.t("btn_bills_confirm", ready=len(ready)), f"bc:y:{batch_id.hex}"),),
        )
    out = [OutMessage(chat_id, head, buttons)]
    out += [review_item(pack, chat_id, b) for b in flagged[:MAX_REVIEW_ITEMS]]
    if len(flagged) > MAX_REVIEW_ITEMS:
        out.append(OutMessage(chat_id, pack.t("bills_more", count=len(flagged) - MAX_REVIEW_ITEMS)))
    return out


def review_item(pack: Pack, chat_id: int, b: Bill) -> OutMessage:
    reject = Button(pack.t("btn_bill_reject"), f"ba:n:{b.id.hex}")
    if b.status in ("unreadable", "failed"):
        return OutMessage(chat_id, pack.t("bills_unreadable_item", page=b.page_no), ((reject,),))
    text = pack.t(
        "bills_review_item",
        page=b.page_no,
        party=b.party_name or "-",
        bill_no=b.bill_no or "-",
        total=format_inr(b.total_paise) if b.total_paise is not None else "-",
        reasons=_reasons(pack, b),
    )
    accept = Button(pack.t("btn_bill_accept"), f"ba:y:{b.id.hex}")
    return OutMessage(chat_id, text, ((accept, reject),))


async def pending(ctx: Ctx) -> None:
    """/bills: whatever still waits for the boss, across all uploads."""
    assert ctx.user is not None
    if not _is_boss(ctx):
        ctx.say(ctx.t("boss_only"))
        return
    repo = ctx.repo
    ready_by_batch = Counter(
        b.batch_id
        for b in await repo.list(Bill, Bill.status == "ready", order_by=[Bill.created_at])
    )
    ready = list(ready_by_batch.items())[:3]
    flagged = await repo.list(
        Bill, Bill.status.in_(_FLAGGED), order_by=[Bill.created_at, Bill.page_no]
    )
    if not ready and not flagged:
        ctx.say(ctx.t("bills_none_pending"))
        return
    for batch_id, n in ready:
        ctx.say(
            ctx.t("bills_confirm_prompt", ready=n),
            (Button(ctx.t("btn_bills_confirm", ready=n), f"bc:y:{batch_id.hex}"),),
        )
    ctx.extend([review_item(ctx.pack, ctx.event.chat_id, b) for b in flagged[:MAX_REVIEW_ITEMS]])
    if len(flagged) > MAX_REVIEW_ITEMS:
        ctx.say(ctx.t("bills_more", count=len(flagged) - MAX_REVIEW_ITEMS))


# ---------------------------------------------------------------------- the boss decides


def _uuid(raw: str) -> uuid.UUID | None:
    try:
        return uuid.UUID(hex=raw)
    except ValueError:
        return None


async def handle_batch_callback(ctx: Ctx, action: str, ident: str) -> None:
    assert ctx.user is not None
    if not _is_boss(ctx):
        ctx.callback_text = ctx.t("boss_only")
        return
    batch_id = _uuid(ident)
    if batch_id is None or action != "y":
        ctx.callback_text = ctx.t("action_expired")
        return
    n = await bill_batch.confirm_ready(ctx.repo, batch_id, ctx.user.id, ctx.now)
    if n == 0:
        ctx.callback_text = ctx.t("bills_already")
        return
    ctx.say(ctx.t("bills_confirmed", count=n))


async def handle_bill_callback(ctx: Ctx, action: str, ident: str) -> None:
    assert ctx.user is not None
    if not _is_boss(ctx):
        ctx.callback_text = ctx.t("boss_only")
        return
    bill_id = _uuid(ident)
    if bill_id is None or action not in ("y", "n"):
        ctx.callback_text = ctx.t("action_expired")
        return
    accept = action == "y"
    if not await bill_batch.decide_bill(ctx.repo, bill_id, ctx.user.id, ctx.now, accept=accept):
        ctx.callback_text = ctx.t("bills_already")
        return
    ctx.say(ctx.t("bills_accepted" if accept else "bills_rejected"))


# ---------------------------------------------------------------------- reversing a bill

REVERSIBLE = ("confirmed", "synced")


def _bill_label(pack: Pack, b: Bill) -> str:
    return pack.t(
        "reverse_item",
        party=b.party_name or "-",
        bill_no=b.bill_no or "-",
        date=b.bill_date.isoformat() if b.bill_date else "-",
        total=format_inr(b.total_paise) if b.total_paise is not None else "-",
    )


async def reverse_command(ctx: Ctx, arg: str) -> None:
    """/reverse lists recent confirmed bills; /reverse <bill no> finds one directly."""
    assert ctx.user is not None
    if not _is_boss(ctx):
        ctx.say(ctx.t("boss_only"))
        return
    bills = await ctx.repo.list(
        Bill,
        Bill.status.in_(REVERSIBLE),
        order_by=[Bill.decided_at.desc(), Bill.created_at.desc()],
    )
    key = arg.strip().casefold()
    if key:
        bills = [b for b in bills if (b.bill_no or "").casefold() == key]
    if not bills:
        ctx.say(ctx.t("reverse_none"))
        return
    ctx.say(ctx.t("reverse_header"))
    for b in bills[:MAX_REVIEW_ITEMS]:
        ctx.say(
            _bill_label(ctx.pack, b),
            (Button(ctx.t("btn_reverse"), f"rv:c:{b.id.hex}"),),
        )


async def handle_reverse_callback(ctx: Ctx, action: str, ident: str) -> None:
    """Two taps on purpose: 'Ulta karo' asks, 'Haan, ulta karo' does it."""
    assert ctx.user is not None
    if not _is_boss(ctx):
        ctx.callback_text = ctx.t("boss_only")
        return
    bill_id = _uuid(ident)
    if bill_id is None or action not in ("c", "y", "n"):
        ctx.callback_text = ctx.t("action_expired")
        return
    if action == "n":
        ctx.say(ctx.t("reverse_cancelled"))
        return
    bill = await ctx.repo.get(Bill, bill_id)
    if bill is None or bill.status not in REVERSIBLE:
        ctx.callback_text = ctx.t("bills_already")
        return
    if action == "c":
        ctx.say(
            ctx.t("reverse_ask", label=_bill_label(ctx.pack, bill)),
            (
                Button(ctx.t("btn_reverse_yes"), f"rv:y:{bill_id.hex}"),
                Button(ctx.t("btn_no"), f"rv:n:{bill_id.hex}"),
            ),
        )
        return
    done = await bill_batch.reverse_bill(ctx.repo, bill_id, ctx.user.id, ctx.now)
    if done is None:
        ctx.callback_text = ctx.t("bills_already")
        return
    ctx.say(ctx.t("reverse_done", label=_bill_label(ctx.pack, done)))
