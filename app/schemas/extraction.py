"""Extraction schemas.

`LLMExtraction` is what the model may return (validated strictly: money/weight must be
real integers, so a float or a numeric string is rejected). `VisitPayload` is what the
system stores after code has resolved dates, cross-checked numbers and applied bounds.
The LLM never sees or produces a resolved date, a total, or a policy decision.
"""

from __future__ import annotations

import enum
from datetime import date
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field

from app.units import MoneyPaise, WeightMg

Confidence = Annotated[float, Field(ge=0.0, le=1.0)]


class Intent(enum.StrEnum):
    LOG_VISIT = "LOG_VISIT"
    CLIENT_DOSSIER_QUERY = "CLIENT_DOSSIER_QUERY"
    SET_BENCHMARK = "SET_BENCHMARK"
    HELP = "HELP"
    UNKNOWN = "UNKNOWN"


# --- what the LLM returns ----------------------------------------------------------------


class _LLMModel(BaseModel):
    model_config = ConfigDict(strict=True, extra="ignore")


class LLMItem(_LLMModel):
    name: str
    quantity: Annotated[int, Field(strict=True, ge=0)] | None = None


class LLMOrder(_LLMModel):
    items: list[LLMItem] = Field(default_factory=list)
    weight_mg: WeightMg | None = None
    value_paise: MoneyPaise | None = None
    rate_paise_per_10g: MoneyPaise | None = None


class LLMPayment(_LLMModel):
    received_paise: MoneyPaise | None = None
    pending_paise: MoneyPaise | None = None


class LLMConfidence(_LLMModel):
    intent: Confidence = 0.0
    client_name: Confidence = 0.0
    weight_mg: Confidence = 0.0
    value_paise: Confidence = 0.0
    rate_paise_per_10g: Confidence = 0.0
    received_paise: Confidence = 0.0
    pending_paise: Confidence = 0.0
    delivery_date: Confidence = 0.0
    next_followup_date: Confidence = 0.0
    visit_date: Confidence = 0.0
    benchmark_rate_paise: Confidence = 0.0


class LLMExtraction(_LLMModel):
    intent: Intent
    client_name: str | None = None
    location: str | None = None
    person_met: str | None = None
    samples_shown: list[str] = Field(default_factory=list)
    order: LLMOrder | None = None
    payment: LLMPayment | None = None
    # Dates are returned AS SPOKEN ("kal", "agle somvar", "8 November"); code resolves them.
    delivery_date_text: str | None = None
    next_followup_date_text: str | None = None
    visit_date_text: str | None = None
    notes: str | None = None
    benchmark_rate_paise: MoneyPaise | None = None
    confidence: LLMConfidence = Field(default_factory=LLMConfidence)


def _nullable(json_type: str) -> dict[str, Any]:
    return {"type": [json_type, "null"]}


_CONF_KEYS = list(LLMConfidence.model_fields)

EXTRACTION_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "intent": {"type": "string", "enum": [i.value for i in Intent]},
        "client_name": _nullable("string"),
        "location": _nullable("string"),
        "person_met": _nullable("string"),
        "samples_shown": {"type": "array", "items": {"type": "string"}},
        "order": {
            "type": ["object", "null"],
            "properties": {
                "items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {"name": {"type": "string"}, "quantity": _nullable("integer")},
                        "required": ["name", "quantity"],
                    },
                },
                "weight_mg": _nullable("integer"),
                "value_paise": _nullable("integer"),
                "rate_paise_per_10g": _nullable("integer"),
            },
            "required": ["items", "weight_mg", "value_paise", "rate_paise_per_10g"],
        },
        "payment": {
            "type": ["object", "null"],
            "properties": {
                "received_paise": _nullable("integer"),
                "pending_paise": _nullable("integer"),
            },
            "required": ["received_paise", "pending_paise"],
        },
        "delivery_date_text": _nullable("string"),
        "next_followup_date_text": _nullable("string"),
        "visit_date_text": _nullable("string"),
        "notes": _nullable("string"),
        "benchmark_rate_paise": _nullable("integer"),
        "confidence": {
            "type": "object",
            "properties": {k: {"type": "number"} for k in _CONF_KEYS},
            "required": _CONF_KEYS,
        },
    },
    "required": [
        "intent", "client_name", "location", "person_met", "samples_shown", "order", "payment",
        "delivery_date_text", "next_followup_date_text", "visit_date_text", "notes",
        "benchmark_rate_paise", "confidence",
    ],
}  # fmt: skip


# --- what the system stores --------------------------------------------------------------


class _Payload(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")


class ItemPayload(_Payload):
    name: str
    quantity: Annotated[int, Field(strict=True, ge=0)] | None = None


class OrderPayload(_Payload):
    items: list[ItemPayload] = Field(default_factory=list)
    weight_mg: WeightMg | None = None
    value_paise: MoneyPaise | None = None
    rate_paise_per_10g: MoneyPaise | None = None


class PaymentPayload(_Payload):
    received_paise: MoneyPaise | None = None
    pending_paise: MoneyPaise | None = None


class VisitPayload(_Payload):
    client_name: str | None = None
    client_id: str | None = None  # set once resolved to an existing Client
    location: str | None = None
    person_met: str | None = None
    samples_shown: list[str] = Field(default_factory=list)
    order: OrderPayload | None = None
    payment: PaymentPayload | None = None
    visit_date: date
    delivery_date: date | None = None
    next_followup_date: date | None = None
    notes: str | None = None


class Flag(BaseModel):
    """Why a value must be read back instead of trusted."""

    model_config = ConfigDict(extra="forbid")

    field: str
    reason: str  # low_confidence | numerals_disagree | out_of_bounds | date_unresolved | ...
    detail: dict[str, Any] = Field(default_factory=dict)
