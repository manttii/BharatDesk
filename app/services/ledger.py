"""Read-side aggregates over committed records.

ASSUMPTION (documented, revisit with real data): a client's outstanding balance is the sum
of the unpaid portion of each confirmed order. Payments against older balances are not
modelled yet.
"""

from __future__ import annotations

import uuid
from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo

from app.db import TenantRepository
from app.models import ActivityLog
from app.models.enums import ActivityStatus

_COUNTED = (ActivityStatus.CONFIRMED, ActivityStatus.SYNCED)


def pending_of(payload: dict[str, object]) -> int:
    """Unpaid portion recorded on one activity payload (0 if none)."""
    visit = payload.get("visit")
    if not isinstance(visit, dict):
        return 0
    pay, order = visit.get("payment"), visit.get("order")
    if isinstance(pay, dict) and isinstance(pay.get("pending_paise"), int):
        return int(pay["pending_paise"])
    value = order.get("value_paise") if isinstance(order, dict) else None
    received = pay.get("received_paise") if isinstance(pay, dict) else None
    if isinstance(value, int):
        return max(value - (received if isinstance(received, int) else 0), 0)
    return 0


async def outstanding_paise(
    repo: TenantRepository, client_id: uuid.UUID, *, exclude: uuid.UUID | None = None
) -> int:
    rows = await repo.list(
        ActivityLog,
        ActivityLog.client_id == client_id,
        ActivityLog.intent == "LOG_VISIT",
        ActivityLog.status.in_(_COUNTED),
    )
    return sum(pending_of(r.payload) for r in rows if r.id != exclude)


async def visits_today(
    repo: TenantRepository, user_id: uuid.UUID, tz_name: str, now: datetime
) -> int:
    """Visits logged by this rep on the tenant's local calendar day (confirmed or awaiting
    approval; rejected and pending-confirmation ones do not count)."""
    tz = ZoneInfo(tz_name)
    local_midnight = datetime.combine(now.astimezone(tz).date(), time.min, tzinfo=tz)
    rows = await repo.list(
        ActivityLog,
        ActivityLog.user_id == user_id,
        ActivityLog.intent == "LOG_VISIT",
        ActivityLog.status.in_([*_COUNTED, ActivityStatus.PENDING_APPROVAL]),
        ActivityLog.created_at >= local_midnight,
        ActivityLog.created_at < local_midnight + timedelta(days=1),
    )
    return len(rows)
