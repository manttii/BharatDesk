"""Outbox for Google Sheets: one durable row per record to publish."""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import Boolean, ForeignKey, Integer, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, TenantOwned, UTCDateTime, enum_column
from app.models.enums import SyncStatus


class SyncJob(TenantOwned, Base):
    """Created in the SAME transaction that confirms a record, so a crash cannot lose it.
    The sheet row is keyed by the activity id, so a retried send updates instead of duplicating."""

    __tablename__ = "sync_jobs"

    activity_log_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("activity_logs.id"), unique=True, index=True
    )
    status: Mapped[SyncStatus] = mapped_column(enum_column(SyncStatus), default=SyncStatus.PENDING)
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    next_attempt_at: Mapped[datetime] = mapped_column(UTCDateTime)
    last_error: Mapped[str | None] = mapped_column(Text, nullable=True)
    synced_at: Mapped[datetime | None] = mapped_column(UTCDateTime, nullable=True)
    failure_notified: Mapped[bool] = mapped_column(Boolean, default=False)
