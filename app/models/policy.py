"""Pricing policy, benchmark rates and approval requests."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import JSON, BigInteger, Boolean, ForeignKey, Integer, String, Text, Uuid
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, TenantOwned, UTCDateTime, enum_column
from app.models.enums import ApprovalStatus


class PolicyRule(TenantOwned, Base):
    __tablename__ = "policy_rules"

    rule_type: Mapped[str] = mapped_column(String(48))
    params: Mapped[dict[str, Any]] = mapped_column(JSON)
    version: Mapped[int] = mapped_column(Integer, default=1)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)


class BenchmarkRate(TenantOwned, Base):
    __tablename__ = "benchmark_rates"

    # Integer paise. `unit` records what the rate is quoted per (ASSUMPTION: confirm
    # with the first customer before Phase 1 sets any policy on it).
    value_paise: Mapped[int] = mapped_column(BigInteger)
    unit: Mapped[str] = mapped_column(String(16), default="10g")
    set_by: Mapped[uuid.UUID] = mapped_column(Uuid)
    set_at: Mapped[datetime] = mapped_column(UTCDateTime)


class ApprovalRequest(TenantOwned, Base):
    __tablename__ = "approval_requests"

    activity_log_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("activity_logs.id"), index=True)
    requested_by: Mapped[uuid.UUID] = mapped_column(Uuid)
    reason_hi: Mapped[str] = mapped_column(Text)
    reason_en: Mapped[str] = mapped_column(Text)
    numbers: Mapped[dict[str, Any]] = mapped_column(JSON)
    status: Mapped[ApprovalStatus] = mapped_column(
        enum_column(ApprovalStatus), default=ApprovalStatus.PENDING
    )
    resolved_by: Mapped[uuid.UUID | None] = mapped_column(Uuid, nullable=True)
    resolved_at: Mapped[datetime | None] = mapped_column(UTCDateTime, nullable=True)
    comment: Mapped[str | None] = mapped_column(Text, nullable=True)
    expires_at: Mapped[datetime] = mapped_column(UTCDateTime)
    renotified_at: Mapped[datetime | None] = mapped_column(UTCDateTime, nullable=True)
