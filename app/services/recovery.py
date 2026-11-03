"""Recover notes stranded by a crash.

A note is accepted in one transaction and processed (download, speech-to-text, LLM) after it
commits. If the process dies in between, the Message stays `processing` forever and the rep
never hears back. This sweep gives up on such notes: it marks them failed, audits it, and tells
the rep to resend. The deduplication claim was already consumed, so a resend is a new update and
is processed normally. Safe to run repeatedly and concurrently (compare-and-set).
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

from app.db import TenantRepository, list_tenant_ids
from app.models import Message, User
from app.models.enums import ProcessingState
from app.pack import Pack
from app.schemas.events import OutMessage


async def recover_stranded(
    session_factory: Any, pack: Pack, now: datetime, *, older_than: timedelta
) -> list[OutMessage]:
    out: list[OutMessage] = []
    cutoff = now - older_than
    async with session_factory() as session:
        tenant_ids = await list_tenant_ids(session)
    for tenant_id in tenant_ids:
        async with session_factory() as session, session.begin():
            repo = TenantRepository(session, tenant_id)
            stuck = await repo.list(
                Message,
                Message.processing_state == ProcessingState.PROCESSING,
                Message.created_at < cutoff,
            )
            for msg in stuck:
                changed = await repo.update_where(
                    Message,
                    {"processing_state": ProcessingState.FAILED},
                    Message.id == msg.id,
                    Message.processing_state == ProcessingState.PROCESSING,
                )
                if changed != 1:
                    continue
                repo.audit(
                    "note_stranded",
                    entity="message",
                    entity_id=str(msg.id),
                    actor_user_id=msg.user_id,
                    detail={"kind": msg.kind},
                )
                user = await repo.get(User, msg.user_id)
                if user is not None and user.telegram_chat_id is not None:
                    out.append(OutMessage(user.telegram_chat_id, pack.t("note_stranded")))
    return out
