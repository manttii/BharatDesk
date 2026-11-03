"""Typed inputs and outputs of the policy engine. Everything numeric is a strict int:
pydantic refuses floats, numeric strings and bools."""

from __future__ import annotations

import enum
from datetime import datetime
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from app.pipeline.expr import parse_expression
from app.units import MoneyPaise, WeightMg

Bps = Annotated[int, Field(strict=True, ge=0, le=10_000)]  # 1 bps = 0.01%


class _Strict(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")


class RuleType(enum.StrEnum):
    DISCOUNT_CAP_PCT = "discount_cap_pct"
    RATE_FLOOR_VS_BENCHMARK = "rate_floor_vs_benchmark"
    CREDIT_LIMIT_PER_CLIENT = "credit_limit_per_client"
    MAX_ORDER_VALUE_WITHOUT_APPROVAL = "max_order_value_without_approval"
    CUSTOM_EXPRESSION = "custom_expression"


# --- rule parameters (what the owner configures) ---------------------------------------


class DiscountCapParams(_Strict):
    max_discount_bps: Bps  # 30 = 0.3%


class RateFloorParams(_Strict):
    max_below_bps: Bps  # how far below benchmark a quote may go; 0 = never below
    max_benchmark_age_seconds: Annotated[int, Field(strict=True, gt=0)]


class CreditLimitParams(_Strict):
    default_limit_paise: MoneyPaise
    per_client_limit_paise: dict[str, MoneyPaise] = Field(default_factory=dict)


class MaxOrderParams(_Strict):
    max_value_paise: MoneyPaise


class CustomExpressionParams(_Strict):
    """An owner-defined rule. `expression` describes the VIOLATION (see pipeline/expr.py);
    the reasons are shown verbatim to the boss and rep, so write them for people."""

    expression: Annotated[str, Field(min_length=1, max_length=300)]
    reason_hi: Annotated[str, Field(min_length=1, max_length=300)]
    reason_en: Annotated[str, Field(min_length=1, max_length=300)]

    @field_validator("expression")
    @classmethod
    def _safe_expression(cls, v: str) -> str:
        parse_expression(v)  # raises ExpressionError (a ValueError) for anything not whitelisted
        return v


_PARAMS: dict[RuleType, type[_Strict]] = {
    RuleType.DISCOUNT_CAP_PCT: DiscountCapParams,
    RuleType.RATE_FLOOR_VS_BENCHMARK: RateFloorParams,
    RuleType.CREDIT_LIMIT_PER_CLIENT: CreditLimitParams,
    RuleType.MAX_ORDER_VALUE_WITHOUT_APPROVAL: MaxOrderParams,
    RuleType.CUSTOM_EXPRESSION: CustomExpressionParams,
}


class RuleSpec(_Strict):
    rule_id: str
    rule_type: RuleType
    version: Annotated[int, Field(strict=True, ge=1)] = 1
    params: _Strict


def parse_rule(rule_id: str, rule_type: str, params: dict[str, Any], version: int = 1) -> RuleSpec:
    """Validate a stored PolicyRule row into a typed rule. Raises on unknown type or
    malformed params, so a bad rule fails at load time, never silently at evaluation."""
    try:
        rtype = RuleType(rule_type)
    except ValueError as exc:
        raise ValueError(f"unknown policy rule type {rule_type!r}") from exc
    try:
        typed = _PARAMS[rtype].model_validate(params)
    except ValidationError as exc:
        raise ValueError(f"invalid params for rule {rule_id!r} ({rule_type}): {exc}") from exc
    return RuleSpec(rule_id=rule_id, rule_type=rtype, version=version, params=typed)


# --- facts (what we know about the order being checked) --------------------------------


class BenchmarkFacts(_Strict):
    value_paise: Annotated[int, Field(strict=True, gt=0)]
    unit: str  # what the rate is quoted per, e.g. "10g"
    set_at: datetime

    @field_validator("set_at")
    @classmethod
    def _aware(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            raise ValueError("set_at must be timezone-aware")
        return v


class PolicyFacts(_Strict):
    client_id: str
    order_value_paise: MoneyPaise
    payment_received_paise: MoneyPaise = 0
    list_value_paise: MoneyPaise | None = None  # value at list price, for discount checks
    quoted_rate_paise: MoneyPaise | None = None
    rate_unit: str | None = None
    benchmark: BenchmarkFacts | None = None
    outstanding_paise: MoneyPaise | None = None  # client's existing unpaid balance
    weight_mg: WeightMg | None = None


# --- result ----------------------------------------------------------------------------


class Violation(_Strict):
    rule_id: str
    rule_type: RuleType
    code: str
    reason_hi: str
    reason_en: str
    numbers: dict[str, int]


class PolicyResult(_Strict):
    violations: tuple[Violation, ...] = ()

    @property
    def passed(self) -> bool:
        return not self.violations
