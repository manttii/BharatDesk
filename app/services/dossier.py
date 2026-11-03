"""Client dossier: a 3-line Hindi briefing built ONLY from saved visits.

No LLM writes this text. Every number comes from confirmed or synced records, so the
briefing cannot invent a visit, an amount or a date. Pending and rejected records are
excluded (a pending approval is only counted, never shown as fact).
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import date

from app.db import TenantRepository
from app.models import ActivityLog, Client, User
from app.models.enums import ActivityStatus
from app.pack import Pack
from app.pipeline.confirm import format_date
from app.schemas.extraction import VisitPayload
from app.services.approvals import visit_of
from app.services.ledger import pending_of
from app.units import format_inr

_SAVED = (ActivityStatus.CONFIRMED, ActivityStatus.SYNCED)


@dataclass(frozen=True)
class Dossier:
    client: str
    visits: int
    last_visit: date | None
    last_rep: str | None
    total_order_paise: int
    last_order_paise: int | None
    last_advance_paise: int | None
    outstanding_paise: int
    next_delivery: date | None
    next_followup: date | None
    pending_approvals: int


async def build_dossier(repo: TenantRepository, client: Client, *, today: date) -> Dossier:
    rows = await repo.list(
        ActivityLog,
        ActivityLog.client_id == client.id,
        ActivityLog.intent == "LOG_VISIT",
        order_by=[ActivityLog.created_at],
    )
    saved = [r for r in rows if r.status in _SAVED]
    waiting = sum(1 for r in rows if r.status is ActivityStatus.PENDING_APPROVAL)
    users: dict[uuid.UUID, str] = {u.id: u.name for u in await repo.list(User)}
    parsed: list[tuple[ActivityLog, VisitPayload]] = [(r, visit_of(r)) for r in saved]
    parsed.sort(key=lambda rp: (rp[1].visit_date, rp[0].created_at))
    last_act, last_visit = parsed[-1] if parsed else (None, None)
    with_order = [(a, v) for a, v in parsed if v.order and v.order.value_paise is not None]
    last_order = with_order[-1][1] if with_order else None
    deliveries = [
        v.delivery_date for _, v in parsed if v.delivery_date and v.delivery_date >= today
    ]
    followups = [
        v.next_followup_date
        for _, v in parsed
        if v.next_followup_date and v.next_followup_date >= today
    ]
    return Dossier(
        client=client.name,
        visits=len(parsed),
        last_visit=last_visit.visit_date if last_visit else None,
        last_rep=users.get(last_act.user_id) if last_act else None,
        total_order_paise=sum(v.order.value_paise or 0 for _, v in parsed if v.order is not None),
        last_order_paise=last_order.order.value_paise if last_order and last_order.order else None,
        last_advance_paise=(
            (last_order.payment.received_paise or 0) if last_order and last_order.payment else 0
        )
        if last_order
        else None,
        outstanding_paise=sum(pending_of(a.payload) for a, _ in parsed),
        next_delivery=min(deliveries) if deliveries else None,
        next_followup=min(followups) if followups else None,
        pending_approvals=waiting,
    )


def dossier_text(d: Dossier, pack: Pack, today: date) -> str:
    if d.visits == 0:
        return pack.t("dossier_empty", client=d.client)
    none = pack.t("dossier_none")

    def money(v: int | None) -> str:
        return format_inr(v) if v is not None else none

    def day(v: date | None) -> str:
        return format_date(v, pack, today) if v else none

    assert d.last_visit is not None
    lines = [
        pack.t(
            "dossier_line1",
            client=d.client,
            visits=d.visits,
            last=day(d.last_visit),
            rep=d.last_rep or "-",
            total=format_inr(d.total_order_paise),
        ),
        pack.t(
            "dossier_line2",
            outstanding=format_inr(d.outstanding_paise),
            last_order=money(d.last_order_paise),
            last_advance=money(d.last_advance_paise),
        ),
        pack.t(
            "dossier_line3",
            delivery=day(d.next_delivery),
            followup=day(d.next_followup),
            pending=pack.t("dossier_pending", count=d.pending_approvals)
            if d.pending_approvals
            else "",
        ),
    ]
    return "\n".join(lines)
