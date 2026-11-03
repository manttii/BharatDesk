"""Google Sheets sync worker: drains the outbox, retries with backoff, never blocks a reply.

Only CONFIRMED (or already SYNCED) records are ever sent. A job whose record has since been
rejected, expired or is still awaiting approval is cancelled, not sent. The decision is made
again right before sending, so a stale job cannot leak an unapproved record.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

import structlog

from app.db import TenantRepository, list_tenant_ids
from app.models import ActivityLog, SyncJob, Tenant, User
from app.models.enums import ActivityStatus, SyncStatus
from app.pack import Pack
from app.pipeline.confirm import TenantSettings
from app.providers.sheets import SheetsError, SheetsProvider
from app.schemas.events import OutMessage
from app.services.approvals import bosses_to_notify, visit_of
from app.units import MG_PER_GRAM

log = structlog.get_logger()

BASE_BACKOFF_S = 30
MAX_BACKOFF_S = 3600
_FORMULA_STARTS = ("=", "+", "-", "@", "\t", "\r")


def backoff_seconds(attempts: int) -> int:
    """30 s, 60 s, 120 s ... capped at 1 h."""
    return min(BASE_BACKOFF_S * (1 << max(attempts - 1, 0)), MAX_BACKOFF_S)


def safe_text(value: str | None) -> str:
    """Defence in depth on top of RAW cell input: neutralise anything formula-like."""
    if not value:
        return ""
    return "'" + value if value.startswith(_FORMULA_STARTS) else value


def amount_cell(paise: int | None) -> int | str:
    """Whole rupees as a number; otherwise an exact decimal string. Integer maths only."""
    if paise is None:
        return ""
    rupees, rem = divmod(paise, 100)
    return rupees if rem == 0 else f"{rupees}.{rem:02d}"


def weight_cell(mg: int | None) -> int | str:
    if mg is None:
        return ""
    grams, rem = divmod(mg, MG_PER_GRAM)
    return grams if rem == 0 else f"{grams}.{rem:03d}".rstrip("0")


def status_label(activity: ActivityLog) -> str:
    meta: dict[str, Any] = activity.payload.get("meta", {})
    if meta.get("counter"):
        return "approved (countered)"
    return "approved" if meta.get("approved") else "confirmed"


def activity_row(activity: ActivityLog, rep_name: str, now: datetime) -> list[Any]:
    v = visit_of(activity)
    order, pay = v.order, v.payment
    meta: dict[str, Any] = activity.payload.get("meta", {})
    return [
        activity.id.hex,
        v.visit_date.isoformat(),
        safe_text(rep_name),
        safe_text(v.client_name),
        weight_cell(order.weight_mg if order else None),
        amount_cell(order.value_paise if order else None),
        amount_cell(order.rate_paise_per_10g if order else None),
        amount_cell(pay.received_paise if pay else None),
        amount_cell(pay.pending_paise if pay else None),
        v.delivery_date.isoformat() if v.delivery_date else "",
        v.next_followup_date.isoformat() if v.next_followup_date else "",
        status_label(activity),
        str(meta.get("source", "note")),
        safe_text(v.notes),
        now.strftime("%Y-%m-%d %H:%M:%S"),
    ]


@dataclass
class SyncReport:
    sent: int = 0
    failed: int = 0  # attempts that errored (will retry unless exhausted)
    exhausted: int = 0
    cancelled: int = 0
    skipped_tenants: int = 0  # no sheet configured: jobs wait, attempts are not burned
    messages: list[OutMessage] = field(default_factory=list)


@dataclass
class _Prepared:
    job_id: uuid.UUID
    activity_id: uuid.UUID
    key: str
    row: list[Any]


async def sync_once(
    session_factory: Any,
    sheets: SheetsProvider,
    pack: Pack,
    *,
    now: datetime,
    max_attempts: int,
    batch_size: int = 20,
) -> SyncReport:
    report = SyncReport()
    async with session_factory() as session:
        tenant_ids = await list_tenant_ids(session)
    for tenant_id in tenant_ids:
        await _sync_tenant(
            session_factory, sheets, pack, tenant_id, now, max_attempts, batch_size, report
        )
    return report


async def _sync_tenant(
    sf: Any,
    sheets: SheetsProvider,
    pack: Pack,
    tenant_id: uuid.UUID,
    now: datetime,
    max_attempts: int,
    batch_size: int,
    report: SyncReport,
) -> None:
    # 1) read + decide eligibility (short transaction, no network)
    async with sf() as session, session.begin():
        repo = TenantRepository(session, tenant_id)
        tenant = await session.get(Tenant, tenant_id)
        assert tenant is not None
        ts = TenantSettings.from_json(tenant.settings)
        due = await repo.list(
            SyncJob,
            SyncJob.status == SyncStatus.PENDING,
            SyncJob.next_attempt_at <= now,
            order_by=[SyncJob.next_attempt_at],
        )
        if not ts.sheet_spreadsheet_id:
            if due:
                report.skipped_tenants += 1
            return
        prepared: list[_Prepared] = []
        for job in due[:batch_size]:
            activity = await repo.get(ActivityLog, job.activity_log_id)
            eligible = activity is not None and activity.status in (
                ActivityStatus.CONFIRMED,
                ActivityStatus.SYNCED,
            )
            if not eligible or activity is None:
                job.status = SyncStatus.CANCELLED
                repo.audit(
                    "sync_cancelled", entity="activity_log", entity_id=str(job.activity_log_id)
                )
                report.cancelled += 1
                continue
            rep = await repo.get(User, activity.user_id)
            prepared.append(
                _Prepared(
                    job.id,
                    activity.id,
                    activity.id.hex,
                    activity_row(activity, rep.name if rep else "-", now),
                )
            )
        sheet_id, worksheet = ts.sheet_spreadsheet_id, ts.sheet_worksheet

    # 2) network, outside any transaction
    results: list[tuple[_Prepared, str | None]] = []
    for p in prepared:
        try:
            await sheets.upsert_row(sheet_id, worksheet, p.key, p.row)
            results.append((p, None))
        except SheetsError as exc:
            results.append((p, str(exc)))
        except Exception as exc:  # never let one bad row stop the batch
            log.exception("sheets_unexpected_error")
            results.append((p, f"unexpected: {type(exc).__name__}"))

    # 3) record outcomes (compare-and-set, so concurrent workers cannot double count)
    async with sf() as session, session.begin():
        repo = TenantRepository(session, tenant_id)
        for p, error in results:
            current = await repo.get(SyncJob, p.job_id)
            if current is None or current.status is not SyncStatus.PENDING:
                continue
            if error is None:
                await repo.update_where(
                    SyncJob,
                    {
                        "status": SyncStatus.DONE,
                        "synced_at": now,
                        "last_error": None,
                        "attempts": current.attempts + 1,
                    },
                    SyncJob.id == current.id,
                    SyncJob.status == SyncStatus.PENDING,
                )
                await repo.update_where(
                    ActivityLog,
                    {"status": ActivityStatus.SYNCED},
                    ActivityLog.id == p.activity_id,
                    ActivityLog.status == ActivityStatus.CONFIRMED,
                )
                repo.audit("activity_synced", entity="activity_log", entity_id=str(p.activity_id))
                report.sent += 1
                continue
            attempts = current.attempts + 1
            report.failed += 1
            if attempts >= max_attempts:
                await repo.update_where(
                    SyncJob,
                    {
                        "status": SyncStatus.FAILED,
                        "attempts": attempts,
                        "last_error": error[:300],
                        "failure_notified": True,
                    },
                    SyncJob.id == current.id,
                    SyncJob.status == SyncStatus.PENDING,
                )
                repo.audit(
                    "sync_failed",
                    entity="activity_log",
                    entity_id=str(p.activity_id),
                    detail={"attempts": attempts, "error": error[:200]},
                )
                report.exhausted += 1
                client = visit_of_client(await repo.get(ActivityLog, p.activity_id))
                for boss in await bosses_to_notify(repo):
                    if boss.telegram_chat_id is not None:
                        report.messages.append(
                            OutMessage(
                                boss.telegram_chat_id,
                                pack.t("sync_failed_boss", count=attempts, client=client),
                            )
                        )
            else:
                await repo.update_where(
                    SyncJob,
                    {
                        "attempts": attempts,
                        "last_error": error[:300],
                        "next_attempt_at": now + timedelta(seconds=backoff_seconds(attempts)),
                    },
                    SyncJob.id == current.id,
                    SyncJob.status == SyncStatus.PENDING,
                )


def visit_of_client(activity: ActivityLog | None) -> str:
    return (visit_of(activity).client_name or "-") if activity is not None else "-"
