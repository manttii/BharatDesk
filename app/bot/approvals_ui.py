"""Boss-side approval interactions: Approve / Reject / Counter buttons and the counter flow.

Every action is re-authorised (the tapper must be a boss of the same tenant), idempotent
(a second tap is a no-op) and answers the callback.
"""

from __future__ import annotations

import uuid
from datetime import timedelta

from app.bot.context import Ctx
from app.models import ActivityLog, ApprovalRequest, ConversationState
from app.models.enums import ApprovalStatus, Role
from app.schemas.events import Button
from app.services import approvals
from app.services.approvals import CounterProposal, ResolveOutcome
from app.units import format_bps, format_inr

COUNTER_KIND = "counter"
COUNTER_CONFIRM_KIND = "counter_confirm"


def _parse_id(raw: str) -> uuid.UUID | None:
    try:
        return uuid.UUID(hex=raw)
    except ValueError:
        return None


def _notify(
    ctx: Ctx, out: ResolveOutcome, *, approve: bool, counter: CounterProposal | None
) -> None:
    pack = ctx.pack
    assert out.activity is not None and out.approval is not None
    summary = approvals.describe(out.activity, pack, ctx.now.date())
    status_label = pack.t("btn_approve") if approve else pack.t("btn_reject")
    ctx.say(pack.t("approval_done_boss", status=status_label))
    rep = out.rep
    if rep is None or rep.telegram_chat_id is None:
        return
    if not approve:
        ctx.say_to(rep.telegram_chat_id, pack.t("approval_rejected_rep", summary=summary))
    elif counter is not None:
        old = out.activity.payload.get("meta", {}).get("counter", {}).get("from_rate")
        change = approvals.counter_change_text(counter, old, pack)
        ctx.say_to(
            rep.telegram_chat_id, pack.t("approval_countered_rep", change=change, summary=summary)
        )
    else:
        ctx.say_to(rep.telegram_chat_id, pack.t("approval_approved_rep", summary=summary))


async def _resolve_and_notify(
    ctx: Ctx, approval_id: uuid.UUID, *, approve: bool, counter: CounterProposal | None = None
) -> None:
    assert ctx.user is not None
    out = await approvals.resolve(
        ctx.repo, approval_id, actor=ctx.user, approve=approve, now=ctx.now, counter=counter
    )
    if out.kind == "applied":
        ctx.callback_text = ctx.t("action_toast_done")
        _notify(ctx, out, approve=approve, counter=counter)
    elif out.kind == "already":
        ctx.callback_text = ctx.t("approval_already")
    elif out.kind == "expired":
        ctx.callback_text = ctx.t("state_expired")
    elif out.kind == "forbidden":
        ctx.say(ctx.t("boss_only"))
    else:
        ctx.callback_text = ctx.t("action_expired")


async def handle_approval_callback(ctx: Ctx, action: str, raw_id: str) -> None:
    assert ctx.user is not None
    if ctx.user.role is not Role.BOSS:
        ctx.say(ctx.t("boss_only"))
        return
    approval_id = _parse_id(raw_id)
    if approval_id is None:
        ctx.callback_text = ctx.t("action_expired")
        return
    if action in ("y", "n"):
        await _resolve_and_notify(ctx, approval_id, approve=action == "y")
        return
    if action == "c":
        await _start_counter(ctx, approval_id)
        return
    ctx.callback_text = ctx.t("action_expired")


async def _start_counter(ctx: Ctx, approval_id: uuid.UUID) -> None:
    assert ctx.user is not None
    approval = await ctx.repo.get(ApprovalRequest, approval_id)
    kind = approvals.counter_kind_of(approval) if approval else None
    if (
        approval is None
        or approval.status is not ApprovalStatus.PENDING
        or approval.expires_at <= ctx.now
    ):
        ctx.callback_text = ctx.t("approval_already")
        return
    if kind is None:
        ctx.callback_text = ctx.t("action_expired")
        return
    ctx.repo.add(
        ConversationState(
            user_id=ctx.user.id,
            kind=COUNTER_KIND,
            data={"approval_id": approval.id.hex, "kind": kind},
            expires_at=ctx.now + timedelta(minutes=ctx.settings.conversation_ttl_minutes),
        )
    )
    ctx.say(ctx.t("counter_prompt_rate" if kind == "rate" else "counter_prompt_discount"))


async def answer_counter(ctx: Ctx, state: ConversationState, text: str) -> None:
    """The boss's reply to 'send the new rate/discount'. Read back before applying."""
    assert ctx.user is not None
    pack = ctx.pack
    kind = str(state.data["kind"])
    approval = await ctx.repo.get(ApprovalRequest, uuid.UUID(hex=str(state.data["approval_id"])))
    prompt = pack.t("counter_prompt_rate" if kind == "rate" else "counter_prompt_discount")
    if approval is None or approval.status is not ApprovalStatus.PENDING:
        await ctx.repo.delete_where(ConversationState, ConversationState.id == state.id)
        ctx.say(pack.t("approval_already"))
        return
    list_value = next(
        (
            v["numbers"].get("list_paise")
            for v in approval.numbers["violations"]
            if v["code"] == "discount_over_cap"
        ),
        None,
    )
    proposal = approvals.parse_counter(kind, text, pack=pack, list_value=list_value)  # type: ignore[arg-type]
    if proposal is None:
        ctx.say(pack.t("counter_bad", prompt=prompt))
        return
    await ctx.repo.delete_where(ConversationState, ConversationState.id == state.id)
    confirm = ctx.repo.add(
        ConversationState(
            user_id=ctx.user.id,
            kind=COUNTER_CONFIRM_KIND,
            data={
                "approval_id": approval.id.hex,
                "kind": proposal.kind,
                "value": proposal.value,
                "new_order_value": proposal.new_order_value,
            },
            expires_at=ctx.now + timedelta(minutes=ctx.settings.conversation_ttl_minutes),
        )
    )
    await ctx.repo.flush()
    activity = await ctx.repo.get(ActivityLog, approval.activity_log_id)
    assert activity is not None
    visit = approvals.visit_of(activity)

    if proposal.kind == "rate":
        old = visit.order.rate_paise_per_10g if visit.order else None
        text_out = pack.t(
            "counter_readback_rate",
            old=format_inr(old) if old is not None else "-",
            new=format_inr(proposal.value),
        )
    else:
        assert proposal.new_order_value is not None
        text_out = pack.t(
            "counter_readback_discount",
            pct=format_bps(proposal.value),
            value=format_inr(proposal.new_order_value),
        )
    h = confirm.id.hex
    ctx.say(
        text_out,
        (Button(pack.t("btn_yes"), f"cc:y:{h}"), Button(pack.t("btn_no"), f"cc:n:{h}")),
    )


async def handle_counter_confirm(ctx: Ctx, action: str, raw_id: str) -> None:
    assert ctx.user is not None
    if ctx.user.role is not Role.BOSS:
        ctx.say(ctx.t("boss_only"))
        return
    state_id = _parse_id(raw_id)
    state = (
        await ctx.repo.first(
            ConversationState,
            ConversationState.id == state_id,
            ConversationState.user_id == ctx.user.id,
            ConversationState.kind == COUNTER_CONFIRM_KIND,
            ConversationState.expires_at > ctx.now,
        )
        if state_id
        else None
    )
    # Consume exactly once; a second tap deletes zero rows.
    if (
        state is None
        or await ctx.repo.delete_where(ConversationState, ConversationState.id == state.id) != 1
    ):
        ctx.callback_text = ctx.t("action_expired")
        return
    if action != "y":
        ctx.say(ctx.t("addworker_cancelled"))  # the approval itself stays pending
        ctx.callback_text = ctx.t("action_toast_cancelled")
        return
    d = state.data
    proposal = CounterProposal(d["kind"], int(d["value"]), d.get("new_order_value"))
    await _resolve_and_notify(
        ctx, uuid.UUID(hex=str(d["approval_id"])), approve=True, counter=proposal
    )
