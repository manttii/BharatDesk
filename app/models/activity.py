"""Per-message traceability and the business records built from messages."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import JSON, BigInteger, ForeignKey, Integer, String, Text, Uuid
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, TenantOwned, UTCDateTime, enum_column
from app.models.enums import ActivityStatus, ProcessingState


class Message(TenantOwned, Base):
    __tablename__ = "messages"

    user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id"))
    update_id: Mapped[int] = mapped_column(BigInteger, unique=True)
    kind: Mapped[str] = mapped_column(String(16))  # text | voice | photo
    file_id: Mapped[str | None] = mapped_column(String(256), nullable=True)
    sha256: Mapped[str | None] = mapped_column(String(64), nullable=True)
    processing_state: Mapped[ProcessingState] = mapped_column(
        enum_column(ProcessingState), default=ProcessingState.RECEIVED
    )
    # Set when the stored audio/image was deleted (retention purge or worker erasure).
    media_purged_at: Mapped[datetime | None] = mapped_column(UTCDateTime, nullable=True)


class Extraction(TenantOwned, Base):
    __tablename__ = "extractions"

    message_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("messages.id"), index=True)
    transcript: Mapped[str | None] = mapped_column(Text, nullable=True)
    stt_model: Mapped[str | None] = mapped_column(String(120), nullable=True)
    llm_model: Mapped[str | None] = mapped_column(String(120), nullable=True)
    prompt_version: Mapped[str | None] = mapped_column(String(32), nullable=True)
    raw_output: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    validated_payload: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    field_confidences: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    latency_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    cost_paise: Mapped[int | None] = mapped_column(Integer, nullable=True)


class ActivityLog(TenantOwned, Base):
    __tablename__ = "activity_logs"

    user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id"))
    client_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("clients.id"), nullable=True)
    message_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("messages.id"), nullable=True)
    intent: Mapped[str] = mapped_column(String(32))
    payload: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    status: Mapped[ActivityStatus] = mapped_column(enum_column(ActivityStatus))


class Correction(TenantOwned, Base):
    """Human fixes; these become labelled examples for the eval set."""

    __tablename__ = "corrections"

    activity_log_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("activity_logs.id"), index=True)
    original: Mapped[dict[str, Any]] = mapped_column(JSON)
    corrected: Mapped[dict[str, Any]] = mapped_column(JSON)
    corrected_by: Mapped[uuid.UUID] = mapped_column(Uuid)
