"""Consent gate. Nothing about a worker is processed until they accept the current notice."""

from __future__ import annotations

from datetime import datetime

from app.db import TenantRepository
from app.models import User


def has_consent(user: User, current_version: str) -> bool:
    return user.consent_at is not None and user.consent_version == current_version


def accept(repo: TenantRepository, user: User, *, version: str, now: datetime) -> None:
    user.consent_at = now
    user.consent_version = version
    repo.audit(
        "consent_accepted",
        entity="user",
        entity_id=str(user.id),
        actor_user_id=user.id,
        detail={"version": version},
    )


def withdraw(repo: TenantRepository, user: User) -> bool:
    """Returns True if there was consent to withdraw."""
    had = user.consent_at is not None
    user.consent_at = None
    user.consent_version = None
    repo.audit(
        "consent_withdrawn",
        entity="user",
        entity_id=str(user.id),
        actor_user_id=user.id,
        detail={"had_consent": had},
    )
    return had
