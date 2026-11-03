"""Boss corrections from the dashboard.

A correction is authoritative (the boss is fixing a confirmed record), so the policy engine
is not re-run, but everything is traceable: a Correction row (original -> corrected), an audit
entry, and a re-queued Sheets sync. Inputs are human units (rupees, grams, ISO dates) parsed
with exact decimals, never floats.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any

from app.db import TenantRepository
from app.models import ActivityLog, Correction, User
from app.models.enums import ActivityStatus
from app.pack import Pack
from app.pipeline.amounts import within_bounds
from app.schemas.extraction import OrderPayload, PaymentPayload, VisitPayload
from app.services import sync_queue
from app.services.approvals import visit_of, with_visit
from app.services.sheets_sync import amount_cell, weight_cell
from app.units import grams_to_mg, parse_decimal, rupees_to_paise

CORRECTABLE = (ActivityStatus.CONFIRMED, ActivityStatus.SYNCED)

# form field -> label (shown in the page); all optional, blank = unchanged
FIELDS: dict[str, str] = {
    "weight_g": "Weight (g)",
    "value_inr": "Order value (INR)",
    "rate_inr": "Rate per 10 g (INR)",
    "advance_inr": "Advance received (INR)",
    "delivery_date": "Delivery date (YYYY-MM-DD)",
    "followup_date": "Follow-up date (YYYY-MM-DD)",
}


@dataclass
class CorrectionInput:
    changes: dict[str, Any]
    errors: dict[str, str]


def current_values(payload: VisitPayload) -> dict[str, str]:
    """The record in the form's human units (exact, for display)."""
    o, p = payload.order, payload.payment
    return {
        "weight_g": str(weight_cell(o.weight_mg if o else None)),
        "value_inr": str(amount_cell(o.value_paise if o else None)),
        "rate_inr": str(amount_cell(o.rate_paise_per_10g if o else None)),
        "advance_inr": str(amount_cell(p.received_paise if p else None)),
        "delivery_date": payload.delivery_date.isoformat() if payload.delivery_date else "",
        "followup_date": payload.next_followup_date.isoformat()
        if payload.next_followup_date
        else "",
    }


def parse_form(form: dict[str, str], current: VisitPayload, pack: Pack) -> CorrectionInput:
    changes: dict[str, Any] = {}
    errors: dict[str, str] = {}
    shown = current_values(current)
    for name in FIELDS:
        raw = (form.get(name) or "").strip()
        if raw == shown[name]:
            continue  # unchanged
        if name in ("delivery_date", "followup_date"):
            if raw == "":
                changes[name] = None
                continue
            try:
                changes[name] = date.fromisoformat(raw)
            except ValueError:
                errors[name] = "Use the format YYYY-MM-DD"
            continue
        if raw == "":
            errors[name] = "Cannot be blank; leave the existing value instead"
            continue
        value = parse_decimal(raw)
        try:
            if value is None:
                raise ValueError
            changes[name] = grams_to_mg(value) if name == "weight_g" else rupees_to_paise(value)
        except (ValueError, TypeError):
            errors[name] = "Enter a plain number with at most whole paise / milligrams"
    bounds = {
        "weight_g": ("weight_mg_min", "weight_mg_max"),
        "value_inr": ("value_paise_min", "value_paise_max"),
        "rate_inr": ("rate_paise_per_10g_min", "rate_paise_per_10g_max"),
    }
    for name, (lo, hi) in bounds.items():
        if name in changes and not within_bounds(pack, changes[name], lo, hi):
            errors[name] = "Outside the allowed range for this business"
    return CorrectionInput(changes, errors)


def apply_changes(old: VisitPayload, changes: dict[str, Any]) -> VisitPayload:
    order = (old.order or OrderPayload()).model_copy()
    payment = (old.payment or PaymentPayload()).model_copy()
    if "weight_g" in changes:
        order = order.model_copy(update={"weight_mg": changes["weight_g"]})
    if "value_inr" in changes:
        order = order.model_copy(update={"value_paise": changes["value_inr"]})
    if "rate_inr" in changes:
        order = order.model_copy(update={"rate_paise_per_10g": changes["rate_inr"]})
    if "advance_inr" in changes:
        payment = payment.model_copy(update={"received_paise": changes["advance_inr"]})
    if {"value_inr", "advance_inr"} & changes.keys() and order.value_paise is not None:
        received = payment.received_paise or 0
        payment = payment.model_copy(update={"pending_paise": max(order.value_paise - received, 0)})
    update: dict[str, Any] = {"order": order, "payment": payment}
    if "delivery_date" in changes:
        update["delivery_date"] = changes["delivery_date"]
    if "followup_date" in changes:
        update["next_followup_date"] = changes["followup_date"]
    return old.model_copy(update=update)


async def apply_correction(
    repo: TenantRepository,
    activity: ActivityLog,
    actor: User,
    changes: dict[str, Any],
    now: Any,
) -> Correction:
    if activity.status not in CORRECTABLE:
        raise ValueError("only confirmed records can be corrected")
    if not changes:
        raise ValueError("nothing to change")
    old = visit_of(activity)
    new = apply_changes(old, changes)
    correction = repo.add(
        Correction(
            created_at=now,
            activity_log_id=activity.id,
            original=old.model_dump(mode="json"),
            corrected=new.model_dump(mode="json"),
            corrected_by=actor.id,
        )
    )
    with_visit(activity, new, corrected_by_boss=True)
    repo.audit(
        "activity_corrected_by_boss",
        entity="activity_log",
        entity_id=str(activity.id),
        actor_user_id=actor.id,
        detail={"fields": sorted(changes)},
    )
    await sync_queue.enqueue(repo, activity.id, now)  # the sheet row is updated, not duplicated
    return correction
