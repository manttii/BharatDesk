"""Handwritten-slip ("kacha parcha") extraction schema, validated as strictly as notes."""

from __future__ import annotations

from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.extraction import Confidence
from app.units import MoneyPaise


class _SlipModel(BaseModel):
    model_config = ConfigDict(strict=True, extra="ignore")


class SlipItem(_SlipModel):
    name: str
    quantity: Annotated[int, Field(strict=True, ge=0)] | None = None
    rate_paise: MoneyPaise | None = None
    amount_paise: MoneyPaise | None = None
    confidence: Confidence = 0.0


class SlipConfidence(_SlipModel):
    party_name: Confidence = 0.0
    date: Confidence = 0.0
    total: Confidence = 0.0
    overall: Confidence = 0.0  # how legible the whole photo is


class LLMSlip(_SlipModel):
    legible: bool  # false for blur, darkness, glare, cropped or rotated beyond reading
    party_name: str | None = None
    date_text: str | None = None  # as written; resolved in code
    items: list[SlipItem] = Field(default_factory=list)
    total_paise: MoneyPaise | None = None  # the total AS WRITTEN on the slip
    confidence: SlipConfidence = Field(default_factory=SlipConfidence)


def _nullable(t: str) -> dict[str, Any]:
    return {"type": [t, "null"]}


SLIP_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "legible": {"type": "boolean"},
        "party_name": _nullable("string"),
        "date_text": _nullable("string"),
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "quantity": _nullable("integer"),
                    "rate_paise": _nullable("integer"),
                    "amount_paise": _nullable("integer"),
                    "confidence": {"type": "number"},
                },
                "required": ["name", "quantity", "rate_paise", "amount_paise", "confidence"],
            },
        },
        "total_paise": _nullable("integer"),
        "confidence": {
            "type": "object",
            "properties": {
                k: {"type": "number"} for k in ("party_name", "date", "total", "overall")
            },
            "required": ["party_name", "date", "total", "overall"],
        },
    },
    "required": ["legible", "party_name", "date_text", "items", "total_paise", "confidence"],
}
