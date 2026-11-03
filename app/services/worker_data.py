"""A worker's right to see and erase their personal data.

EXPORT returns everything we hold about one worker as plain JSON.
ERASE removes the personal data and keeps the company's business records:
  * deleted: their stored audio/photos, Telegram file ids and hashes, transcripts and raw model
    output, open conversation state, name, Telegram chat id and consent;
  * kept: confirmed/rejected business records (what was sold to whom), now attributed to
    "Deleted worker", and the append-only audit log (it holds ids, not names);
  * propagated: their Google Sheet rows are re-synced so the rep column no longer shows a name.
Erasing is idempotent. A boss (the tenant owner) cannot be erased this way.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from app.db import TenantRepository
from app.models import (
    ActivityLog,
    AuditLog,
    ConversationState,
    Correction,
    Extraction,
    Message,
    User,
)
from app.models.enums import ActivityStatus, Role
from app.pipeline.media import MediaStore
from app.services import sync_queue

ERASED_NAME = "Deleted worker"


def _iso(value: datetime | None) -> str | None:
    return value.isoformat() if value else None


async def export_worker(repo: TenantRepository, worker: User, now: datetime) -> dict[str, Any]:
    messages = await repo.list(Message, Message.user_id == worker.id, order_by=[Message.created_at])
    ids = {m.id for m in messages}
    extractions = [e for e in await repo.list(Extraction) if e.message_id in ids]
    by_message = {e.message_id: e for e in extractions}
    activities = await repo.list(ActivityLog, ActivityLog.user_id == worker.id)
    corrections = await repo.list(Correction, Correction.corrected_by == worker.id)
    audit = await repo.list(AuditLog, AuditLog.actor_user_id == worker.id)
    return {
        "exported_at": _iso(now),
        "user": {
            "id": str(worker.id),
            "name": worker.name,
            "role": worker.role.value,
            "telegram_chat_id": worker.telegram_chat_id,
            "is_active": worker.is_active,
            "created_at": _iso(worker.created_at),
            "erased_at": _iso(worker.erased_at),
        },
        "consent": {
            "accepted_at": _iso(worker.consent_at),
            "notice_version": worker.consent_version,
        },
        "messages": [
            {
                "id": str(m.id),
                "kind": m.kind,
                "received_at": _iso(m.created_at),
                "state": m.processing_state.value,
                "telegram_file_id": m.file_id,
                "media_sha256": m.sha256,
                "media_deleted_at": _iso(m.media_purged_at),
                "transcript": by_message[m.id].transcript if m.id in by_message else None,
                "stt_model": by_message[m.id].stt_model if m.id in by_message else None,
                "llm_model": by_message[m.id].llm_model if m.id in by_message else None,
            }
            for m in messages
        ],
        "records": [
            {
                "id": str(a.id),
                "created_at": _iso(a.created_at),
                "status": a.status.value,
                "payload": a.payload,
            }
            for a in activities
        ],
        "corrections": [
            {
                "id": str(c.id),
                "created_at": _iso(c.created_at),
                "original": c.original,
                "corrected": c.corrected,
            }
            for c in corrections
        ],
        "audit_events": [
            {"at": _iso(a.created_at), "action": a.action, "entity": a.entity, "detail": a.detail}
            for a in audit
        ],
    }


@dataclass
class ErasureReport:
    already_erased: bool = False
    messages_scrubbed: int = 0
    media_files_deleted: int = 0
    records_resynced: int = 0


async def erase_worker(
    repo: TenantRepository,
    store: MediaStore,
    worker: User,
    *,
    actor: User | None,
    now: datetime,
) -> ErasureReport:
    if worker.role is Role.BOSS:
        raise ValueError("the tenant owner cannot be erased with this tool")
    if worker.erased_at is not None:
        return ErasureReport(already_erased=True)
    report = ErasureReport()
    mine = await repo.list(Message, Message.user_id == worker.id)
    mine_ids = {m.id for m in mine}
    others = [
        m for m in await repo.list(Message, Message.sha256.is_not(None)) if m.id not in mine_ids
    ]
    shared = {
        m.sha256 for m in others if m.media_purged_at is None
    }  # another message still needs it
    for m in mine:
        if m.sha256 and m.sha256 not in shared:
            report.media_files_deleted += store.delete(repo.tenant_id, m.sha256)
        m.file_id, m.sha256, m.media_purged_at = None, None, now
        report.messages_scrubbed += 1
    for e in await repo.list(Extraction):
        if e.message_id in mine_ids:
            e.transcript, e.raw_output, e.validated_payload, e.field_confidences = (
                None,
                None,
                None,
                None,
            )
    await repo.delete_where(ConversationState, ConversationState.user_id == worker.id)

    worker.name, worker.telegram_chat_id, worker.is_active = ERASED_NAME, None, False
    worker.consent_at, worker.consent_version, worker.erased_at = None, None, now

    for activity in await repo.list(ActivityLog, ActivityLog.user_id == worker.id):
        if activity.status in (ActivityStatus.CONFIRMED, ActivityStatus.SYNCED):
            await sync_queue.enqueue(repo, activity.id, now)  # the sheet row drops the name
            report.records_resynced += 1
    repo.audit(
        "worker_erased",
        entity="user",
        entity_id=str(worker.id),
        actor_user_id=actor.id if actor else None,
        detail={
            "messages_scrubbed": report.messages_scrubbed,
            "media_files_deleted": report.media_files_deleted,
        },
    )
    return report
