"""Read-back text and the auto-confirm decision.

Principle: every extracted amount, weight, rate and date is read back and committed only
after the rep taps Haan, UNLESS auto-confirm is enabled for the tenant AND confidence is at
or above the threshold AND all values pass sanity bounds AND the numerals cross-check
agreed AND the client matched exactly. Any flag, or any doubt, means ask.
"""

from __future__ import annotations

from datetime import date

from pydantic import BaseModel, ConfigDict, Field

from app.pack import Pack
from app.schemas.extraction import Flag, VisitPayload
from app.units import format_inr, format_weight_mg

_FIELD_LABELS = {
    "client_name": "label_client",
    "order.weight_mg": "label_weight",
    "order.value_paise": "label_value",
    "order.rate_paise_per_10g": "label_rate",
    "payment.received_paise": "label_advance",
    "payment.pending_paise": "label_pending",
    "delivery_date": "label_delivery",
    "next_followup_date": "label_followup",
}


class TenantSettings(BaseModel):
    """Typed view of Tenant.settings. Auto-confirm is OFF until a tenant opts in: with no
    eval data there is no evidence yet that skipping the read-back is safe."""

    model_config = ConfigDict(extra="ignore")

    auto_confirm_enabled: bool = False
    auto_confirm_min_confidence: float = Field(default=0.98, ge=0.0, le=1.0)
    sheet_spreadsheet_id: str | None = None
    sheet_worksheet: str = "Visits"
    # Days to keep stored audio / slip photos; None = keep forever. Default 90.
    audio_retention_days: int | None = Field(default=90, ge=1)
    image_retention_days: int | None = Field(default=90, ge=1)
    # Days to keep transcripts and raw model output; None (default) keeps them. Business
    # records are never purged by this.
    transcript_retention_days: int | None = Field(default=None, ge=1)
    # Local hour (0-23) at which the boss gets a daily summary; None (default) = off.
    daily_report_hour: int | None = Field(default=None, ge=0, le=23)
    last_daily_report: str | None = None  # ISO date of the last report sent (at most once a day)
    demo: bool = False  # seeded demo tenants are excluded from pilot reports

    @classmethod
    def from_json(cls, raw: dict[str, object] | None) -> TenantSettings:
        return cls.model_validate(raw or {})


def format_date(d: date, pack: Pack, today: date) -> str:
    month = pack.months[d.month - 1] if pack.months else str(d.month)
    return f"{d.day} {month}" + (f" {d.year}" if d.year != today.year else "")


def summary_text(payload: VisitPayload, pack: Pack, *, new_client: bool, today: date) -> str:
    parts: list[str] = []
    if payload.client_name:
        parts.append(
            pack.t("readback_new_client", name=payload.client_name)
            if new_client
            else payload.client_name
        )
    if payload.order:
        o = payload.order
        if o.weight_mg is not None:
            parts.append(format_weight_mg(o.weight_mg))
        if o.value_paise is not None:
            parts.append(format_inr(o.value_paise))
        if o.rate_paise_per_10g is not None:
            parts.append(pack.t("readback_rate", amount=format_inr(o.rate_paise_per_10g)))
    if payload.payment:
        if payload.payment.received_paise is not None:
            parts.append(
                pack.t("readback_advance", amount=format_inr(payload.payment.received_paise))
            )
        if payload.payment.pending_paise is not None:
            parts.append(
                pack.t("readback_pending", amount=format_inr(payload.payment.pending_paise))
            )
    if payload.delivery_date:
        parts.append(
            pack.t("readback_delivery", date=format_date(payload.delivery_date, pack, today))
        )
    if payload.next_followup_date:
        parts.append(
            pack.t("readback_followup", date=format_date(payload.next_followup_date, pack, today))
        )
    return ", ".join(parts)


def warning_text(flags: list[Flag], pack: Pack) -> str | None:
    labels: list[str] = []
    for f in flags:
        key = _FIELD_LABELS.get(f.field, "label_other")
        label = pack.t(key)
        if label not in labels:
            labels.append(label)
    return pack.t("readback_warn", fields=", ".join(labels)) if labels else None


def readback_message(
    payload: VisitPayload, flags: list[Flag], pack: Pack, *, new_client: bool, today: date
) -> str:
    lines = [
        summary_text(payload, pack, new_client=new_client, today=today)
        + ". "
        + pack.t("readback_question")
    ]
    warn = warning_text(flags, pack)
    if warn:
        lines.append(warn)
    return "\n".join(lines)


def can_auto_confirm(
    flags: list[Flag],
    confidences: dict[str, float],
    settings: TenantSettings,
    *,
    client_exact: bool,
    missing: list[str],
) -> bool:
    if not settings.auto_confirm_enabled or flags or missing or not client_exact or not confidences:
        return False
    return all(c >= settings.auto_confirm_min_confidence for c in confidences.values())
