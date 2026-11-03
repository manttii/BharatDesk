"""Media retention: delete stored audio and slip photos after the tenant's retention period.

Per tenant (TenantSettings.audio_retention_days / image_retention_days, default 90, None keeps
forever). Only the media files are removed; transcripts and business records stay, and every
purge is audited. Files are content-addressed, so a file is deleted only once EVERY message that
references it is due. The job is idempotent and safe to re-run after a crash.
"""

from __future__ import annotations

import uuid
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from app.db import TenantRepository, list_tenant_ids
from app.models import Extraction, Message, Tenant
from app.pipeline.confirm import TenantSettings
from app.pipeline.media import MediaStore


@dataclass
class PurgeReport:
    tenants: int = 0
    messages_purged: int = 0
    files_deleted: int = 0


def retention_days(settings: TenantSettings, kind: str) -> int | None:
    return settings.audio_retention_days if kind == "voice" else settings.image_retention_days


async def purge_media(
    session_factory: Any,
    store: MediaStore,
    now: datetime,
    *,
    only_tenant: uuid.UUID | None = None,
) -> PurgeReport:
    report = PurgeReport()
    async with session_factory() as session:
        tenant_ids = await list_tenant_ids(session)
    for tenant_id in tenant_ids:
        if only_tenant is not None and tenant_id != only_tenant:
            continue
        async with session_factory() as session, session.begin():
            repo = TenantRepository(session, tenant_id)
            tenant = await session.get(Tenant, tenant_id)
            assert tenant is not None
            ts = TenantSettings.from_json(tenant.settings)
            report.tenants += 1
            rows = await repo.list(
                Message, Message.sha256.is_not(None), Message.kind.in_(("voice", "photo"))
            )
            due: list[Message] = []
            for m in rows:
                if m.media_purged_at is not None:
                    continue
                days = retention_days(ts, m.kind)
                if days is not None and m.created_at < now - timedelta(days=days):
                    due.append(m)
            if not due:
                continue
            by_sha: dict[str, list[Message]] = defaultdict(list)
            for m in rows:
                if m.sha256:
                    by_sha[m.sha256].append(m)
            due_ids = {m.id for m in due}
            for sha, refs in by_sha.items():
                mine = [m for m in refs if m.id in due_ids]
                if not mine:
                    continue
                still_needed = any(m.media_purged_at is None and m.id not in due_ids for m in refs)
                if not still_needed:
                    report.files_deleted += store.delete(tenant_id, sha)
                for m in mine:
                    m.media_purged_at = now
                    report.messages_purged += 1
                    repo.audit(
                        "media_purged",
                        entity="message",
                        entity_id=str(m.id),
                        detail={"kind": m.kind, "reason": "retention"},
                    )
    return report


async def purge_transcripts(
    session_factory: Any, now: datetime, *, only_tenant: uuid.UUID | None = None
) -> int:
    """Blank transcripts and raw model output past `transcript_retention_days`.

    Off unless a tenant sets it. The validated business payload, confidences and the record
    itself stay; only the free text of what was said (and the model's raw echo of it) goes.
    Returns the number of extractions scrubbed. Idempotent.
    """
    total = 0
    async with session_factory() as session:
        tenant_ids = await list_tenant_ids(session)
    for tenant_id in tenant_ids:
        if only_tenant is not None and tenant_id != only_tenant:
            continue
        async with session_factory() as session, session.begin():
            repo = TenantRepository(session, tenant_id)
            tenant = await session.get(Tenant, tenant_id)
            assert tenant is not None
            days = TenantSettings.from_json(tenant.settings).transcript_retention_days
            if days is None:
                continue
            cutoff = now - timedelta(days=days)
            sent = {m.id: m.created_at for m in await repo.list(Message)}
            scrubbed = 0
            for e in await repo.list(Extraction):
                if (e.transcript is None and e.raw_output is None) or sent.get(
                    e.message_id, now
                ) >= cutoff:
                    continue
                e.transcript, e.raw_output = None, None
                scrubbed += 1
            if scrubbed:
                repo.audit(
                    "transcripts_purged",
                    entity="tenant",
                    entity_id=str(tenant_id),
                    detail={"count": scrubbed, "reason": "retention"},
                )
            total += scrubbed
    return total
