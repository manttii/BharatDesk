"""Short-lived conversation state and the global idempotency ledger."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import JSON, ForeignKey, String
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, TenantOwned, UTCDateTime, utcnow


class ConversationState(TenantOwned, Base):
    """A pending question or a read-back awaiting a button tap. Consumed on use."""

    __tablename__ = "conversation_states"

    user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id"), index=True)
    kind: Mapped[str] = mapped_column(String(32))
    data: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    expires_at: Mapped[datetime] = mapped_column(UTCDateTime)


class ProcessedEvent(Base):
    """Global dedupe ledger. Keys: `u:<update_id>` and `cb:<callback_id>`.

    Deliberately not tenant-owned: it must work for senders we cannot resolve to a
    tenant (unknown chats), and it stores no user content.
    """

    __tablename__ = "processed_events"

    key: Mapped[str] = mapped_column(String(80), primary_key=True)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)
