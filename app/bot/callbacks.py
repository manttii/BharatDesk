"""Inline-button callbacks. Payloads are short ids (Telegram caps callback_data at 64
bytes); every action is re-authorised and consumed exactly once."""

from __future__ import annotations

import uuid

from app.bot import approvals_ui, benchmark, bills_ui, notes, sheets_ui
from app.bot.commands import ADDWORKER_KIND, create_worker_from_state
from app.bot.context import Ctx
from app.models import ConversationState
from app.models.enums import Role
from app.services import consent


async def handle_callback(ctx: Ctx) -> None:
    assert ctx.user is not None
    parts = (ctx.event.callback_data or "").split(":")
    ns = parts[0]
    if ns == "co" and len(parts) == 2 and parts[1] in ("y", "n"):
        await _consent(ctx, accept=parts[1] == "y")
        return
    # Everything below acts for a user, so it needs current consent.
    if not consent.has_consent(ctx.user, ctx.settings.consent_version):
        ctx.say(ctx.t("consent_required"))
        return
    if len(parts) == 3:
        action, ident = parts[1], parts[2]
        if ns == "aw" and action in ("y", "n"):
            await _addworker(ctx, confirm=action == "y", token=ident)
            return
        if ns == "cf" and action in ("y", "n", "e"):  # read-back: Haan / Nahi / Sudharo
            await notes.handle_confirm_callback(ctx, action, ident)
            return
        if ns == "cl":  # client choice: index or "n" (new client)
            await notes.handle_client_choice_callback(ctx, action, ident)
            return
        if ns == "ap" and action in ("y", "n", "c"):  # boss: Approve / Reject / Counter
            await approvals_ui.handle_approval_callback(ctx, action, ident)
            return
        if ns == "sh" and action in ("y", "n"):  # boss: connect the Google Sheet
            await sheets_ui.handle_callback(ctx, action, ident)
            return
        if ns == "bc" and action == "y":  # boss: confirm the clean bills of an upload
            await bills_ui.handle_batch_callback(ctx, action, ident)
            return
        if ns == "rv" and action in ("c", "y", "n"):  # boss: reverse a confirmed bill
            await bills_ui.handle_reverse_callback(ctx, action, ident)
            return
        if ns == "ba" and action in ("y", "n"):  # boss: accept / reject one flagged bill
            await bills_ui.handle_bill_callback(ctx, action, ident)
            return
        if ns == "rt" and action in ("y", "n"):  # boss: confirm today's benchmark rate
            await benchmark.handle_callback(ctx, action, ident)
            return
        if ns == "cc" and action in ("y", "n"):  # boss: confirm the counter read-back
            await approvals_ui.handle_counter_confirm(ctx, action, ident)
            return
    ctx.callback_text = ctx.t("action_expired")


async def _consent(ctx: Ctx, *, accept: bool) -> None:
    assert ctx.user is not None
    if accept:
        if not consent.has_consent(ctx.user, ctx.settings.consent_version):
            consent.accept(ctx.repo, ctx.user, version=ctx.settings.consent_version, now=ctx.now)
        ctx.say(ctx.t("consent_accepted"))
        ctx.callback_text = ctx.t("consent_toast")
    else:
        ctx.say(ctx.t("consent_declined"))


async def _addworker(ctx: Ctx, *, confirm: bool, token: str) -> None:
    assert ctx.user is not None
    if ctx.user.role is not Role.BOSS:
        ctx.say(ctx.t("boss_only"))
        return
    try:
        state_id = uuid.UUID(hex=token)
    except ValueError:
        ctx.callback_text = ctx.t("action_expired")
        return
    state = await ctx.repo.first(
        ConversationState,
        ConversationState.id == state_id,
        ConversationState.user_id == ctx.user.id,
        ConversationState.kind == ADDWORKER_KIND,
        ConversationState.expires_at > ctx.now,
    )
    # Consume exactly once: a second tap deletes zero rows and becomes a no-op.
    if (
        state is None
        or await ctx.repo.delete_where(ConversationState, ConversationState.id == state.id) != 1
    ):
        ctx.callback_text = ctx.t("action_expired")
        return
    if not confirm:
        ctx.say(ctx.t("addworker_cancelled"))
        ctx.callback_text = ctx.t("action_toast_cancelled")
        return
    name = str(state.data["name"])
    link = await create_worker_from_state(ctx, name)
    ctx.say(ctx.t("addworker_done", name=name, link=link))
    ctx.callback_text = ctx.t("action_toast_done")
