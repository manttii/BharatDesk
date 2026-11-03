"""Optional daily summary for the boss.

Off unless a tenant sets `daily_report_hour` (the boss does it with `/dailyreport 20`). Once a
day, after that local hour, each consented boss gets a short Hindi summary computed from the
database: visits, confirmed order value, approvals waiting, per-rep counts, failed Sheet syncs.

Delivery is at-most-once per day: the "sent" marker is committed before the message goes out, so
a crash or a Telegram failure skips that day's report rather than ever sending two.
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta
from typing import Any
from zoneinfo import ZoneInfo

from app.db import TenantRepository, list_tenant_ids
from app.models import ActivityLog, SyncJob, Tenant, User
from app.models.enums import ActivityStatus, SyncStatus
from app.pack import Pack
from app.pipeline.confirm import TenantSettings, format_date
from app.schemas.events import OutMessage
from app.services.approvals import bosses_to_notify, visit_of
from app.units import format_inr

_SAVED = (ActivityStatus.CONFIRMED, ActivityStatus.SYNCED)


def _value(a: ActivityLog) -> int:
    order = visit_of(a).order
    return (order.value_paise or 0) if order else 0


async def build_text(repo: TenantRepository, tenant: Tenant, now: datetime, pack: Pack) -> str:
    tz = ZoneInfo(tenant.timezone)
    local_day: date = now.astimezone(tz).date()
    start = datetime.combine(local_day, time.min, tzinfo=tz)
    end = start + timedelta(days=1)
    users = {u.id: u.name for u in await repo.list(User)}
    todays = [
        a
        for a in await repo.list(ActivityLog, ActivityLog.intent == "LOG_VISIT")
        if start <= a.created_at < end
    ]
    saved = [a for a in todays if a.status in _SAVED]
    waiting = [a for a in todays if a.status is ActivityStatus.PENDING_APPROVAL]
    counted = [*saved, *waiting]  # a visit awaiting approval still happened
    per_rep: dict[str, int] = {}
    for a in counted:
        name = users.get(a.user_id, "-")
        per_rep[name] = per_rep.get(name, 0) + 1
    reps = ", ".join(f"{n} {c}" for n, c in sorted(per_rep.items(), key=lambda kv: (-kv[1], kv[0])))
    failed = sum(1 for j in await repo.list(SyncJob) if j.status is SyncStatus.FAILED)
    return pack.t(
        "daily_report",
        date=format_date(local_day, pack, local_day),
        visits=len(counted),
        value=format_inr(sum(_value(a) for a in saved)),  # only confirmed money counts
        pending=len(waiting),
        pending_value=format_inr(sum(_value(a) for a in waiting)),
        reps=reps or "-",
        failed=failed,
    )


async def send_due(session_factory: Any, pack: Pack, now: datetime) -> list[OutMessage]:
    """Reports that are due now. Marks them sent in the same transaction that builds them."""
    out: list[OutMessage] = []
    async with session_factory() as session:
        tenant_ids = await list_tenant_ids(session)
    for tenant_id in tenant_ids:
        async with session_factory() as session, session.begin():
            tenant = await session.get(Tenant, tenant_id)
            assert tenant is not None
            ts = TenantSettings.from_json(tenant.settings)
            if ts.daily_report_hour is None:
                continue
            local = now.astimezone(ZoneInfo(tenant.timezone))
            today = local.date().isoformat()
            if local.hour < ts.daily_report_hour or ts.last_daily_report == today:
                continue
            repo = TenantRepository(session, tenant_id)
            bosses = await bosses_to_notify(repo)
            tenant.settings = {**tenant.settings, "last_daily_report": today}  # reassign: JSON
            if not bosses:
                continue
            text = await build_text(repo, tenant, now, pack)
            out += [
                OutMessage(b.telegram_chat_id, text)
                for b in bosses
                if b.telegram_chat_id is not None
            ]
            repo.audit(
                "daily_report_sent",
                entity="tenant",
                entity_id=str(tenant_id),
                detail={"date": today},
            )
    return out
