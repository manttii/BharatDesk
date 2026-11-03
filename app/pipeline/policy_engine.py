"""Deterministic policy engine. Integer arithmetic only; no LLM, no float, no eval.

Principle: the engine FAILS CLOSED. If a fact a rule needs is missing, or the benchmark is
absent, stale or in a different unit, the result is a violation (-> boss approval), never
a pass. Percentages are integer basis points and are compared by cross-multiplication,
so there is no rounding at the boundary: exactly at the cap passes, one paisa over fails.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timedelta

from app.pack import Pack
from app.pipeline.expr import ExpressionError, expression_names
from app.pipeline.expr import evaluate as evaluate_expression
from app.schemas.policy import (
    BenchmarkFacts,
    CreditLimitParams,
    CustomExpressionParams,
    DiscountCapParams,
    MaxOrderParams,
    PolicyFacts,
    PolicyResult,
    RateFloorParams,
    RuleSpec,
    RuleType,
    Violation,
)
from app.units import ceil_div, format_bps, format_inr

BPS = 10_000


def _violation(
    rule: RuleSpec, code: str, key: str, numbers: dict[str, int], pack: Pack, **text: str
) -> Violation:
    return Violation(
        rule_id=rule.rule_id,
        rule_type=rule.rule_type,
        code=code,
        reason_hi=pack.t(key, **text),
        reason_en=pack.te(key, **text),
        numbers=numbers,
    )


def _missing(rule: RuleSpec, fact: str, pack: Pack) -> Violation:
    return _violation(rule, "fact_missing", "policy_fact_missing", {}, pack, fact=fact)


def _rate_floor(rule: RuleSpec, f: PolicyFacts, now: datetime, pack: Pack) -> list[Violation]:
    p = rule.params
    assert isinstance(p, RateFloorParams)
    if f.quoted_rate_paise is None:
        return [_missing(rule, "quoted_rate", pack)]
    if f.rate_unit is None:
        return [_missing(rule, "rate_unit", pack)]
    b: BenchmarkFacts | None = f.benchmark
    if b is None:
        return [_violation(rule, "benchmark_missing", "policy_benchmark_missing", {}, pack)]
    if b.set_at > now:
        return [_violation(rule, "benchmark_future", "policy_benchmark_future", {}, pack)]
    age = now - b.set_at
    if age > timedelta(seconds=p.max_benchmark_age_seconds):
        age_h = age // timedelta(hours=1)
        max_h = ceil_div(p.max_benchmark_age_seconds, 3600)
        return [
            _violation(
                rule,
                "benchmark_stale",
                "policy_benchmark_stale",
                {
                    "age_seconds": age // timedelta(seconds=1),
                    "max_age_seconds": p.max_benchmark_age_seconds,
                },
                pack,
                age_h=str(age_h),
                max_h=str(max_h),
            )
        ]
    if f.rate_unit != b.unit:
        return [
            _violation(
                rule, "unit_mismatch", "policy_unit_mismatch", {}, pack,
                quoted_unit=f.rate_unit, benchmark_unit=b.unit,
            )
        ]  # fmt: skip
    # quoted >= benchmark * (1 - bps/10000)   <=>   quoted*10000 >= benchmark*(10000-bps)
    if f.quoted_rate_paise * BPS >= b.value_paise * (BPS - p.max_below_bps):
        return []
    floor = ceil_div(b.value_paise * (BPS - p.max_below_bps), BPS)  # smallest whole-paisa pass
    return [
        _violation(
            rule,
            "rate_below_floor",
            "policy_rate_below_floor",
            {
                "quoted_paise": f.quoted_rate_paise,
                "benchmark_paise": b.value_paise,
                "floor_paise": floor,
                "max_below_bps": p.max_below_bps,
            },
            pack,
            quoted=format_inr(f.quoted_rate_paise),
            benchmark=format_inr(b.value_paise),
            pct=format_bps(p.max_below_bps),
            floor=format_inr(floor),
        )
    ]


def _discount_cap(rule: RuleSpec, f: PolicyFacts, now: datetime, pack: Pack) -> list[Violation]:
    p = rule.params
    assert isinstance(p, DiscountCapParams)
    if f.list_value_paise is None or f.list_value_paise == 0:
        return [_missing(rule, "list_value", pack)]
    discount = max(f.list_value_paise - f.order_value_paise, 0)
    # discount/list <= cap/10000   <=>   discount*10000 <= cap*list
    if discount * BPS <= p.max_discount_bps * f.list_value_paise:
        return []
    return [
        _violation(
            rule,
            "discount_over_cap",
            "policy_discount_over_cap",
            {
                "list_paise": f.list_value_paise,
                "order_paise": f.order_value_paise,
                "discount_paise": discount,
                "discount_bps_ceil": ceil_div(discount * BPS, f.list_value_paise),
                "max_discount_bps": p.max_discount_bps,
            },
            pack,
            discount=format_inr(discount),
            pct=format_bps(ceil_div(discount * BPS, f.list_value_paise)),
            pct_cap=format_bps(p.max_discount_bps),
        )
    ]


def _credit_limit(rule: RuleSpec, f: PolicyFacts, now: datetime, pack: Pack) -> list[Violation]:
    p = rule.params
    assert isinstance(p, CreditLimitParams)
    if f.outstanding_paise is None:
        return [_missing(rule, "outstanding", pack)]
    limit = p.per_client_limit_paise.get(f.client_id, p.default_limit_paise)
    new_pending = max(f.order_value_paise - f.payment_received_paise, 0)
    total = f.outstanding_paise + new_pending
    if total <= limit:
        return []
    return [
        _violation(
            rule,
            "credit_over_limit",
            "policy_credit_over_limit",
            {
                "outstanding_paise": f.outstanding_paise,
                "new_pending_paise": new_pending,
                "total_paise": total,
                "limit_paise": limit,
            },
            pack,
            total=format_inr(total),
            limit=format_inr(limit),
        )
    ]


def _max_order(rule: RuleSpec, f: PolicyFacts, now: datetime, pack: Pack) -> list[Violation]:
    p = rule.params
    assert isinstance(p, MaxOrderParams)
    if f.order_value_paise <= p.max_value_paise:
        return []
    return [
        _violation(
            rule,
            "order_over_max",
            "policy_order_over_max",
            {"order_paise": f.order_value_paise, "max_paise": p.max_value_paise},
            pack,
            value=format_inr(f.order_value_paise),
            limit=format_inr(p.max_value_paise),
        )
    ]


def _custom_values(f: PolicyFacts, now: datetime) -> dict[str, int | None]:
    b = f.benchmark
    return {
        "order_value_paise": f.order_value_paise,
        "payment_received_paise": f.payment_received_paise,
        "new_pending_paise": max(f.order_value_paise - f.payment_received_paise, 0),
        "list_value_paise": f.list_value_paise,
        "quoted_rate_paise": f.quoted_rate_paise,
        "benchmark_paise": b.value_paise if b else None,
        "benchmark_age_seconds": (now - b.set_at) // timedelta(seconds=1) if b else None,
        "outstanding_paise": f.outstanding_paise,
        "weight_mg": f.weight_mg,
    }


def _custom(rule: RuleSpec, f: PolicyFacts, now: datetime, pack: Pack) -> list[Violation]:
    p = rule.params
    assert isinstance(p, CustomExpressionParams)
    names = sorted(expression_names(p.expression))
    values = _custom_values(f, now)
    for name in names:  # a fact the rule needs but we do not have: fail closed
        if values[name] is None:
            return [_missing(rule, name, pack)]
    used = {n: int(values[n]) for n in names}  # type: ignore[arg-type]
    try:
        violated = evaluate_expression(p.expression, used)
    except ExpressionError:
        return [_violation(rule, "config_invalid", "policy_config_invalid", {}, pack)]
    if not violated:
        return []
    return [
        Violation(
            rule_id=rule.rule_id,
            rule_type=rule.rule_type,
            code="custom_rule",
            reason_hi=p.reason_hi,
            reason_en=p.reason_en,
            numbers=used,
        )
    ]


_CHECKS: dict[RuleType, Callable[[RuleSpec, PolicyFacts, datetime, Pack], list[Violation]]] = {
    RuleType.RATE_FLOOR_VS_BENCHMARK: _rate_floor,
    RuleType.DISCOUNT_CAP_PCT: _discount_cap,
    RuleType.CREDIT_LIMIT_PER_CLIENT: _credit_limit,
    RuleType.MAX_ORDER_VALUE_WITHOUT_APPROVAL: _max_order,
    RuleType.CUSTOM_EXPRESSION: _custom,
}


def evaluate(
    facts: PolicyFacts, rules: list[RuleSpec], *, pack: Pack, now: datetime
) -> PolicyResult:
    """Run every rule; return ALL violations (deterministic order: by rule_id)."""
    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    violations: list[Violation] = []
    for rule in sorted(rules, key=lambda r: (r.rule_id, r.version)):
        violations.extend(_CHECKS[rule.rule_type](rule, facts, now, pack))
    return PolicyResult(violations=tuple(violations))
