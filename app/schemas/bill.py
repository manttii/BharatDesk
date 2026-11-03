"""What a vision/text model may return for ONE bill page, validated as strictly as notes.

Quantities are integer thousandths of a unit and money is integer paise: the model never
returns a decimal. Totals are exactly as WRITTEN on the page; the model does no arithmetic.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.extraction import Confidence
from app.units import MoneyPaise


class _BillModel(BaseModel):
    model_config = ConfigDict(strict=True, extra="ignore")


class BillLineIn(_BillModel):
    name: str
    unit: str | None = None  # as written: pcs, kg, m, box ...
    quantity_milli: Annotated[int, Field(strict=True, ge=0)] | None = None
    rate_paise: MoneyPaise | None = None
    amount_paise: MoneyPaise | None = None
    confidence: Confidence = 0.0


class BillConfidence(_BillModel):
    party_name: Confidence = 0.0
    bill_no: Confidence = 0.0
    date: Confidence = 0.0
    total: Confidence = 0.0
    overall: Confidence = 0.0  # how legible the whole page is


class LLMBill(_BillModel):
    legible: bool
    doc_type: Literal["purchase", "sales", "other"] = "other"
    party_name: str | None = None  # the OTHER party: supplier on a purchase, buyer on a sale
    bill_no: str | None = None
    date_text: str | None = None  # as written; resolved in code
    lines: list[BillLineIn] = Field(default_factory=list)
    subtotal_paise: MoneyPaise | None = None
    tax_paise: MoneyPaise | None = None
    total_paise: MoneyPaise | None = None
    confidence: BillConfidence = Field(default_factory=BillConfidence)


def _nullable(t: str) -> dict[str, Any]:
    return {"type": [t, "null"]}


BILL_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "legible": {"type": "boolean"},
        "doc_type": {"type": "string", "enum": ["purchase", "sales", "other"]},
        "party_name": _nullable("string"),
        "bill_no": _nullable("string"),
        "date_text": _nullable("string"),
        "lines": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "unit": _nullable("string"),
                    "quantity_milli": _nullable("integer"),
                    "rate_paise": _nullable("integer"),
                    "amount_paise": _nullable("integer"),
                    "confidence": {"type": "number"},
                },
                "required": [
                    "name",
                    "unit",
                    "quantity_milli",
                    "rate_paise",
                    "amount_paise",
                    "confidence",
                ],
            },
        },
        "subtotal_paise": _nullable("integer"),
        "tax_paise": _nullable("integer"),
        "total_paise": _nullable("integer"),
        "confidence": {
            "type": "object",
            "properties": {
                k: {"type": "number"} for k in ("party_name", "bill_no", "date", "total", "overall")
            },
            "required": ["party_name", "bill_no", "date", "total", "overall"],
        },
    },
    "required": [
        "legible",
        "doc_type",
        "party_name",
        "bill_no",
        "date_text",
        "lines",
        "subtotal_paise",
        "tax_paise",
        "total_paise",
        "confidence",
    ],
}
