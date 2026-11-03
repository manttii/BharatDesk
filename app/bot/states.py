"""Short-lived, button-driven conversation state shared by the bot flows."""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta

from app.bot.context import Ctx
from app.models import ConversationState


def expiry(ctx: Ctx) -> datetime:
    return ctx.now + timedelta(minutes=ctx.settings.conversation_ttl_minutes)


async def consume(ctx: Ctx, raw_id: str, kinds: tuple[str, ...]) -> ConversationState | None:
    """Fetch and delete a state exactly once (a second tap gets None). Owner-checked."""
    assert ctx.user is not None
    try:
        sid = uuid.UUID(hex=raw_id)
    except ValueError:
        return None
    state = await ctx.repo.first(
        ConversationState,
        ConversationState.id == sid,
        ConversationState.user_id == ctx.user.id,
        ConversationState.kind.in_(kinds),
        ConversationState.expires_at > ctx.now,
    )
    if (
        state is None
        or await ctx.repo.delete_where(ConversationState, ConversationState.id == state.id) != 1
    ):
        return None
    return state
