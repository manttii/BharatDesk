"""Turn a validated LLMExtraction into a VisitPayload plus the reasons to distrust it.

Code decides here: dates are resolved, every number is cross-checked against the
deterministic numeral parser, sanity bounds from the pack are applied, and low-confidence
fields are flagged. Any flag forces a read-back; nothing here auto-confirms by itself.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta

from app.pack import Pack
from app.pipeline.dates import resolve_date_phrase
from app.pipeline.numerals import (
    Quantity,
    cross_check_money,
    cross_check_weight,
    parse_quantities,
)
from app.schemas.extraction import (
    Flag,
    ItemPayload,
    LLMExtraction,
    OrderPayload,
    PaymentPayload,
    VisitPayload,
)

REQUIRED_ORDER_VALUE = "order.value_paise"
REQUIRED_RATE = "order.rate_paise_per_10g"
REQUIRED_CLIENT = "client_name"


@dataclass
class Assessment:
    payload: VisitPayload
    flags: list[Flag] = field(default_factory=list)
    confidences: dict[str, float] = field(default_factory=dict)
    missing: list[str] = field(default_factory=list)  # required fields, in the order to ask

    @property
    def flagged_fields(self) -> list[str]:
        return list(dict.fromkeys(f.field for f in self.flags))


def _check_money(
    field_name: str, value: int, quantities: list[Quantity], flags: list[Flag]
) -> None:
    check = cross_check_money(value, quantities)
    if not check.agrees:
        flags.append(
            Flag(
                field=field_name,
                reason="numerals_disagree",
                detail={"candidates": list(check.candidates)},
            )
        )


def _bounded(flags: list[Flag], name: str, value: int, lo: int | None, hi: int | None) -> None:
    if (lo is not None and value < lo) or (hi is not None and value > hi):
        flags.append(
            Flag(field=name, reason="out_of_bounds", detail={"value": value, "min": lo, "max": hi})
        )


def assess_visit(
    ext: LLMExtraction,
    transcript: str,
    *,
    today: date,
    pack: Pack,
    min_confidence: float,
    transcript_reasons: tuple[str, ...] = (),
    rate_required: bool = False,
    value_required: bool = False,
) -> Assessment:
    flags: list[Flag] = []
    b = pack.bounds
    conf = ext.confidence

    # --- dates (resolved in code, never by the LLM) ---
    def resolve(text: str | None, name: str, direction: str) -> date | None:
        if not text or not text.strip():
            return None
        d = resolve_date_phrase(text, today, direction)  # type: ignore[arg-type]
        if d is None:
            flags.append(Flag(field=name, reason="date_unresolved", detail={"said": text}))
        return d

    visit_date = resolve(ext.visit_date_text, "visit_date", "past") or today
    delivery = resolve(ext.delivery_date_text, "delivery_date", "future")
    followup = resolve(ext.next_followup_date_text, "next_followup_date", "future")
    for name, d in (("delivery_date", delivery), ("next_followup_date", followup)):
        if d is not None and (
            d < visit_date or d > today + timedelta(days=b.get("max_future_days", 365))
        ):
            flags.append(Flag(field=name, reason="out_of_bounds", detail={"date": d.isoformat()}))
    if visit_date > today or visit_date < today - timedelta(days=b.get("max_past_days", 400)):
        flags.append(
            Flag(
                field="visit_date", reason="out_of_bounds", detail={"date": visit_date.isoformat()}
            )
        )

    # --- order / payment ---
    quantities = parse_quantities(transcript)
    order: OrderPayload | None = None
    if ext.order is not None:
        o = ext.order
        order = OrderPayload(
            items=[ItemPayload(name=i.name, quantity=i.quantity) for i in o.items],
            weight_mg=o.weight_mg,
            value_paise=o.value_paise,
            rate_paise_per_10g=o.rate_paise_per_10g,
        )
        if o.weight_mg is not None:
            if not cross_check_weight(o.weight_mg, quantities).agrees:
                flags.append(Flag(field="order.weight_mg", reason="numerals_disagree",
                                  detail={"candidates": list(cross_check_weight(o.weight_mg, quantities).candidates)}))  # fmt: skip
            _bounded(
                flags,
                "order.weight_mg",
                o.weight_mg,
                b.get("weight_mg_min"),
                b.get("weight_mg_max"),
            )
        if o.value_paise is not None:
            _check_money("order.value_paise", o.value_paise, quantities, flags)
            _bounded(
                flags,
                "order.value_paise",
                o.value_paise,
                b.get("value_paise_min"),
                b.get("value_paise_max"),
            )
        if o.rate_paise_per_10g is not None:
            _check_money(REQUIRED_RATE, o.rate_paise_per_10g, quantities, flags)
            _bounded(flags, REQUIRED_RATE, o.rate_paise_per_10g,
                     b.get("rate_paise_per_10g_min"), b.get("rate_paise_per_10g_max"))  # fmt: skip

    payment: PaymentPayload | None = None
    if ext.payment is not None:
        p = ext.payment
        received, pending = p.received_paise, p.pending_paise
        value = order.value_paise if order else None
        derived = max(value - received, 0) if value is not None and received is not None else None
        if received is not None:
            _check_money("payment.received_paise", received, quantities, flags)
            if value is not None and received > value:
                flags.append(Flag(field="payment.received_paise", reason="out_of_bounds",
                                  detail={"received": received, "order_value": value}))  # fmt: skip
        if pending is None:
            pending = derived  # derived by code, so there is nothing for the LLM to get wrong
        elif pending != derived:
            _check_money("payment.pending_paise", pending, quantities, flags)
        payment = PaymentPayload(received_paise=received, pending_paise=pending)

    # --- confidences and low-confidence flags ---
    confidences: dict[str, float] = {}

    def track(name: str, present: bool, c: float) -> None:
        if not present:
            return
        confidences[name] = c
        if c < min_confidence:
            flags.append(Flag(field=name, reason="low_confidence", detail={"confidence": c}))

    track("client_name", ext.client_name is not None, conf.client_name)
    if order:
        track("order.weight_mg", order.weight_mg is not None, conf.weight_mg)
        track("order.value_paise", order.value_paise is not None, conf.value_paise)
        track(REQUIRED_RATE, order.rate_paise_per_10g is not None, conf.rate_paise_per_10g)
    if ext.payment:
        track("payment.received_paise", ext.payment.received_paise is not None, conf.received_paise)
        track("payment.pending_paise", ext.payment.pending_paise is not None, conf.pending_paise)
    track("delivery_date", delivery is not None, conf.delivery_date)
    track("next_followup_date", followup is not None, conf.next_followup_date)

    for reason in transcript_reasons:  # the audio itself was doubtful
        flags.append(Flag(field="transcript", reason=reason))

    payload = VisitPayload(
        client_name=ext.client_name.strip()
        if ext.client_name and ext.client_name.strip()
        else None,
        location=ext.location,
        person_met=ext.person_met,
        samples_shown=list(ext.samples_shown),
        order=order,
        payment=payment,
        visit_date=visit_date,
        delivery_date=delivery,
        next_followup_date=followup,
        notes=ext.notes,
    )

    return Assessment(
        payload=payload,
        flags=flags,
        confidences=confidences,
        missing=missing_fields(payload, rate_required=rate_required, value_required=value_required),
    )


def missing_fields(
    payload: VisitPayload, *, rate_required: bool, value_required: bool = False
) -> list[str]:
    """Required fields still absent, in the order to ask.

    The client is always required. The order value and the quoted rate are required only
    when a policy rule needs them (the engine cannot check what was never stated, so the rep
    is asked rather than the order being sent to approval blindly); a visit with no rules to
    satisfy is read back as spoken."""
    missing: list[str] = []
    if payload.client_name is None:
        missing.append(REQUIRED_CLIENT)
    if payload.order is not None:
        if value_required and payload.order.value_paise is None:
            missing.append(REQUIRED_ORDER_VALUE)
        if rate_required and payload.order.rate_paise_per_10g is None:
            missing.append(REQUIRED_RATE)
    return missing
