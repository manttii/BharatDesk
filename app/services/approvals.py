"""Approval requests: create, resolve (exactly once), counter, remind, expire.

Resolution is a compare-and-set (`UPDATE ... WHERE status = 'pending'`), so a double tap,
two bosses tapping at once, or a retried callback can only ever apply once. The approval,
the activity record and the audit log change in the same transaction.
"""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from fractions import Fraction
from typing import Any, Literal

from app.db import TenantRepository, list_tenant_ids
from app.models import ActivityLog, ApprovalRequest, Extraction, User
from app.models.enums import ActivityStatus, ApprovalStatus, Role
from app.pack import Pack
from app.pipeline.confirm import summary_text
from app.pipeline.numerals import parse_quantities, to_paise
from app.schemas.events import Button, OutMessage
from app.schemas.extraction import VisitPayload
from app.services import sync_queue
from app.units import format_bps, format_inr

COUNTER_CODES = {"rate_below_floor": "rate", "discount_over_cap": "discount"}
CounterKind = Literal["rate", "discount"]
_PCT_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(?:%|percent|pratishat|प्रतिशत)", re.IGNORECASE)


# --- reading an activity back ------------------------------------------------------------


def visit_of(activity: ActivityLog) -> VisitPayload:
    # stored as JSON (dates are ISO strings): validate in strict JSON mode, not python mode
    return VisitPayload.model_validate_json(json.dumps(activity.payload["visit"]))


def with_visit(activity: ActivityLog, payload: VisitPayload, **meta_updates: Any) -> None:
    """Write a new payload dict (JSON columns are not mutation-tracked)."""
    meta = {**activity.payload.get("meta", {}), **meta_updates}
    activity.payload = {"visit": payload.model_dump(mode="json"), "meta": meta}


def describe(activity: ActivityLog, pack: Pack, today: date) -> str:
    return summary_text(
        visit_of(activity),
        pack,
        new_client=bool(activity.payload.get("meta", {}).get("client_new")),
        today=today,
    )


def _short(text: str, n: int = 140) -> str:
    text = " ".join(text.split())
    return text if len(text) <= n else text[: n - 1] + "…"


# --- creating a request ------------------------------------------------------------------


async def bosses_to_notify(repo: TenantRepository) -> list[User]:
    return await repo.list(
        User,
        User.role == Role.BOSS,
        User.is_active.is_(True),
        User.telegram_chat_id.is_not(None),
        User.consent_at.is_not(None),
    )


def violation_numbers(violations: list[Any]) -> dict[str, Any]:
    return {
        "violations": [
            {"rule_id": v.rule_id, "code": v.code, "numbers": v.numbers} for v in violations
        ]
    }


def counter_kind_of(approval: ApprovalRequest) -> CounterKind | None:
    for v in approval.numbers.get("violations", []):
        if v["code"] in COUNTER_CODES:
            return COUNTER_CODES[v["code"]]  # type: ignore[return-value]
    return None


def approval_buttons(approval: ApprovalRequest, pack: Pack) -> tuple[tuple[Button, ...], ...]:
    h = approval.id.hex
    row = [Button(pack.t("btn_approve"), f"ap:y:{h}"), Button(pack.t("btn_reject"), f"ap:n:{h}")]
    if counter_kind_of(approval) is not None:
        row.append(Button(pack.t("btn_counter"), f"ap:c:{h}"))
    return (tuple(row),)


def boss_request_message(
    boss_chat: int,
    approval: ApprovalRequest,
    *,
    rep: str,
    client: str,
    summary: str,
    excerpt: str,
    pack: Pack,
    reminder: bool = False,
) -> OutMessage:
    reasons = "\n".join(f"- {line}" for line in approval.reason_hi.split("\n"))
    key = "approval_reminder_boss" if reminder else "approval_request_boss"
    text = pack.t(
        key, rep=rep, client=client, summary=summary, reasons=reasons, excerpt=_short(excerpt)
    )
    return OutMessage(boss_chat, text, approval_buttons(approval, pack))


# --- resolving ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CounterProposal:
    kind: CounterKind
    value: int  # rate: paise per 10 g; discount: basis points
    new_order_value: int | None = None  # discount only


def parse_counter(
    kind: CounterKind, text: str, *, pack: Pack, list_value: int | None
) -> CounterProposal | None:
    """Parse the boss's counter. Anything unclear returns None and the boss is asked again."""
    if kind == "rate":
        quantities = [q for q in parse_quantities(text)]
        paise = [to_paise(q, unitless_as_rupees=True) for q in quantities]
        if len(quantities) != 1 or paise[0] is None:
            return None
        lo, hi = (
            pack.bounds.get("rate_paise_per_10g_min"),
            pack.bounds.get("rate_paise_per_10g_max"),
        )
        if (lo is not None and paise[0] < lo) or (hi is not None and paise[0] > hi):
            return None
        return CounterProposal("rate", paise[0])
    m = _PCT_RE.search(text)
    if m is None or list_value is None:
        return None
    bps = Fraction(m.group(1)) * 100
    if bps.denominator != 1 or not 0 <= bps <= 10_000:
        return None  # e.g. 0.333%: not a whole number of basis points
    discount = (list_value * int(bps)) // 10_000  # floor: ties go to the company
    return CounterProposal("discount", int(bps), list_value - discount)


def apply_counter(payload: VisitPayload, proposal: CounterProposal) -> VisitPayload:
    if payload.order is None:
        return payload
    if proposal.kind == "rate":
        order = payload.order.model_copy(update={"rate_paise_per_10g": proposal.value})
    else:
        order = payload.order.model_copy(update={"value_paise": proposal.new_order_value})
    return payload.model_copy(update={"order": order})


def counter_change_text(proposal: CounterProposal, old_rate: int | None, pack: Pack) -> str:
    if proposal.kind == "rate":
        return pack.t(
            "counter_change_rate",
            old=format_inr(old_rate) if old_rate is not None else "-",
            new=format_inr(proposal.value),
        )
    assert proposal.new_order_value is not None
    return pack.t(
        "counter_change_discount",
        pct=format_bps(proposal.value),
        value=format_inr(proposal.new_order_value),
    )


@dataclass
class ResolveOutcome:
    kind: Literal["applied", "already", "expired", "notfound", "forbidden"]
    approval: ApprovalRequest | None = None
    activity: ActivityLog | None = None
    rep: User | None = None


async def resolve(
    repo: TenantRepository,
    approval_id: uuid.UUID,
    *,
    actor: User,
    approve: bool,
    now: datetime,
    counter: CounterProposal | None = None,
) -> ResolveOutcome:
    if actor.role is not Role.BOSS:
        return ResolveOutcome("forbidden")
    approval = await repo.get(ApprovalRequest, approval_id)
    if approval is None:
        return ResolveOutcome("notfound")
    if approval.status is not ApprovalStatus.PENDING:
        return ResolveOutcome("already", approval)
    if approval.expires_at <= now:
        return ResolveOutcome("expired", approval)
    new_status = ApprovalStatus.APPROVED if approve else ApprovalStatus.REJECTED
    changed = await repo.update_where(
        ApprovalRequest,
        {
            "status": new_status,
            "resolved_by": actor.id,
            "resolved_at": now,
            "comment": f"counter:{counter.kind}={counter.value}" if counter else None,
        },
        ApprovalRequest.id == approval_id,
        ApprovalRequest.status == ApprovalStatus.PENDING,
    )
    if changed != 1:  # someone else got there first
        return ResolveOutcome("already", approval)
    activity = await repo.get(ActivityLog, approval.activity_log_id)
    assert activity is not None
    if approve:
        activity.status = ActivityStatus.CONFIRMED
        with_visit(activity, visit_of(activity), approved=True)
        await sync_queue.enqueue(repo, activity.id, now)
        if counter is not None:
            before = visit_of(activity)
            with_visit(
                activity,
                apply_counter(before, counter),
                counter={"kind": counter.kind, "value": counter.value,
                         "from_rate": before.order.rate_paise_per_10g if before.order else None},
            )  # fmt: skip
    else:
        activity.status = ActivityStatus.REJECTED
    action = (
        "approval_countered"
        if counter
        else ("approval_approved" if approve else "approval_rejected")
    )
    repo.audit(
        action,
        entity="approval_request",
        entity_id=str(approval.id),
        actor_user_id=actor.id,
        detail={"activity_id": str(activity.id), "counter": bool(counter)},
    )
    rep = await repo.get(User, activity.user_id)
    return ResolveOutcome("applied", approval, activity, rep)


# --- reminders and expiry ----------------------------------------------------------------


async def _context_for(
    repo: TenantRepository, approval: ApprovalRequest, pack: Pack, today: date
) -> tuple[ActivityLog, User | None, str, str, str]:
    activity = await repo.get(ActivityLog, approval.activity_log_id)
    assert activity is not None
    rep = await repo.get(User, activity.user_id)
    visit = visit_of(activity)
    excerpt = ""
    if activity.message_id is not None:
        ex = await repo.first(Extraction, Extraction.message_id == activity.message_id)
        excerpt = ex.transcript or "" if ex else ""
    return activity, rep, visit.client_name or "-", describe(activity, pack, today), excerpt


async def sweep(
    session_factory: Any, pack: Pack, ttl_hours: int, now: datetime
) -> list[OutMessage]:
    """Expire overdue approvals and remind the boss once, halfway to expiry. Returns the
    messages to send; call from a background loop. Safe to run repeatedly and concurrently."""
    out: list[OutMessage] = []
    async with session_factory() as session:
        tenant_ids = await list_tenant_ids(session)
    for tenant_id in tenant_ids:
        async with session_factory() as session, session.begin():
            repo = TenantRepository(session, tenant_id)
            pending = await repo.list(
                ApprovalRequest, ApprovalRequest.status == ApprovalStatus.PENDING
            )
            for approval in pending:
                activity, rep, client, summary, excerpt = await _context_for(
                    repo, approval, pack, now.date()
                )
                bosses = await bosses_to_notify(repo)
                if approval.expires_at <= now:
                    done = await repo.update_where(
                        ApprovalRequest,
                        {"status": ApprovalStatus.EXPIRED, "resolved_at": now},
                        ApprovalRequest.id == approval.id,
                        ApprovalRequest.status == ApprovalStatus.PENDING,
                    )
                    if done != 1:
                        continue
                    activity.status = ActivityStatus.REJECTED  # never syncs
                    repo.audit(
                        "approval_expired", entity="approval_request", entity_id=str(approval.id)
                    )
                    rep_name = rep.name if rep else "-"
                    if rep is not None and rep.telegram_chat_id is not None:
                        out.append(OutMessage(rep.telegram_chat_id, pack.t(
                            "approval_expired_rep", hours=ttl_hours, summary=summary)))  # fmt: skip
                    out += [OutMessage(b.telegram_chat_id, pack.t(
                        "approval_expired_boss", rep=rep_name, client=client, summary=summary))
                        for b in bosses if b.telegram_chat_id is not None]  # fmt: skip
                    continue
                half = approval.created_at + (approval.expires_at - approval.created_at) // 2
                if approval.renotified_at is None and now >= half:
                    done = await repo.update_where(
                        ApprovalRequest,
                        {"renotified_at": now},
                        ApprovalRequest.id == approval.id,
                        ApprovalRequest.renotified_at.is_(None),
                    )
                    if done == 1:
                        repo.audit(
                            "approval_reminded",
                            entity="approval_request",
                            entity_id=str(approval.id),
                        )
                        out += [
                            boss_request_message(
                                b.telegram_chat_id,
                                approval,
                                rep=rep.name if rep else "-",
                                client=client,
                                summary=summary,
                                excerpt=excerpt,
                                pack=pack,
                                reminder=True,
                            )
                            for b in bosses
                            if b.telegram_chat_id is not None
                        ]
    return out


def default_expiry(now: datetime, ttl_hours: int) -> datetime:
    return now + timedelta(hours=ttl_hours)
