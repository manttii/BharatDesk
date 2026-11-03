"""Tenant, user, invite and client tables."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import JSON, BigInteger, Boolean, ForeignKey, String, Uuid
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, IdTimestampMixin, TenantOwned, UTCDateTime, enum_column
from app.models.enums import Role


class Tenant(IdTimestampMixin, Base):
    __tablename__ = "tenants"

    name: Mapped[str] = mapped_column(String(120))
    timezone: Mapped[str] = mapped_column(String(64), default="Asia/Kolkata")
    # auto_confirm_enabled, retention days, etc. Typed access lives in services.
    settings: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)


class User(TenantOwned, Base):
    __tablename__ = "users"

    # Nullable until the invite link is redeemed; unique across all tenants because a
    # Telegram chat resolves to exactly one identity.
    telegram_chat_id: Mapped[int | None] = mapped_column(BigInteger, unique=True, nullable=True)
    name: Mapped[str] = mapped_column(String(120))
    role: Mapped[Role] = mapped_column(enum_column(Role))
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    consent_at: Mapped[datetime | None] = mapped_column(UTCDateTime, nullable=True)
    consent_version: Mapped[str | None] = mapped_column(String(32), nullable=True)
    # Set when the worker's personal data was erased on request (see services/worker_data.py).
    erased_at: Mapped[datetime | None] = mapped_column(UTCDateTime, nullable=True)


class InviteToken(TenantOwned, Base):
    """Single-use, expiring, stored hashed, bound to tenant + role + user record."""

    __tablename__ = "invite_tokens"

    user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id"))
    role: Mapped[Role] = mapped_column(enum_column(Role))
    token_hash: Mapped[str] = mapped_column(String(64), unique=True)
    expires_at: Mapped[datetime] = mapped_column(UTCDateTime)
    used_at: Mapped[datetime | None] = mapped_column(UTCDateTime, nullable=True)
    created_by: Mapped[uuid.UUID | None] = mapped_column(Uuid, nullable=True)


class Client(TenantOwned, Base):
    __tablename__ = "clients"

    name: Mapped[str] = mapped_column(String(160))
    normalized_name: Mapped[str] = mapped_column(String(160), index=True)
    aliases: Mapped[list[str]] = mapped_column(JSON, default=list)
