"""Enqueue a record for Google Sheets. Tiny and dependency-light so that confirming a
record (commit, approval, dashboard correction) can call it without import cycles."""

from __future__ import annotations

import uuid
from datetime import datetime

from app.db import TenantRepository
from app.models import SyncJob
from app.models.enums import SyncStatus


async def enqueue(repo: TenantRepository, activity_id: uuid.UUID, now: datetime) -> None:
    """Idempotent: an existing job is reset to pending (used after a correction)."""
    job = await repo.first(SyncJob, SyncJob.activity_log_id == activity_id)
    if job is None:
        repo.add(SyncJob(activity_log_id=activity_id, next_attempt_at=now))
        return
    await repo.update_where(
        SyncJob,
        {
            "status": SyncStatus.PENDING,
            "attempts": 0,
            "next_attempt_at": now,
            "last_error": None,
            "failure_notified": False,
        },
        SyncJob.id == job.id,
    )
