"""Resolve a Telegram chat to a user. This is the ONE deliberate cross-tenant lookup:
we cannot scope by tenant until we know who is talking."""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import User


async def resolve_user(session: AsyncSession, chat_id: int) -> User | None:
    stmt = select(User).where(User.telegram_chat_id == chat_id)
    return (await session.execute(stmt)).scalar_one_or_none()
