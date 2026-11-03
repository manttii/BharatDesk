"""Policy engine: boundary behaviour, fail-closed cases, strict typing, exact-arithmetic oracle."""

from __future__ import annotations

import random
from datetime import UTC, datetime, timedelta
from fractions import Fraction
from typing import Any

import pytest
from pydantic import ValidationError

from app.pack import load_pack
from app.pipeline.policy_engine import evaluate
from app.schemas.policy import BenchmarkFacts, PolicyFacts, PolicyResult, RuleSpec, parse_rule

NOW = datetime(2026, 11, 2, 9, 0, tzinfo=UTC)
PACK = load_pack()
BENCH = 7_520_000  # ₹75,200 per 10g, in paise
DAY = 86_400


def rate_rule(bps: int = 100, age: int = DAY) -> RuleSpec:
    return parse_rule("floor", "rate_floor_vs_benchmark",
                      {"max_below_bps": bps, "max_benchmark_age_seconds": age})  # fmt: skip


def bench(value: int = BENCH, unit: str = "10g", age_s: int = 3600) -> BenchmarkFacts:
    return BenchmarkFacts(value_paise=value, unit=unit, set_at=NOW - timedelta(seconds=age_s))


def facts(**kw: Any) -> PolicyFacts:
    base: dict[str, Any] = {"client_id": "c1", "order_value_paise": 1_000_000}
    return PolicyFacts(**{**base, **kw})


def run(f: PolicyFacts, *rules: RuleSpec, now: datetime = NOW) -> PolicyResult:
    return evaluate(f, list(rules), pack=PACK, now=now)


def codes(res: PolicyResult) -> list[str]:
    return [v.code for v in res.violations]


# --- rate floor vs benchmark -------------------------------------------------------------


def test_quote_exactly_at_floor_passes_and_one_paisa_below_fails() -> None:
    floor = BENCH * 9900 // 10_000  # 7,444,800 exactly
    at = run(facts(quoted_rate_paise=floor, rate_unit="10g", benchmark=bench()), rate_rule())
    below = run(facts(quoted_rate_paise=floor - 1, rate_unit="10g", benchmark=bench()), rate_rule())
    assert at.passed and codes(below) == ["rate_below_floor"]


def test_floor_with_fractional_paisa_rounds_in_the_strict_direction() -> None:
    b = bench(value=7_520_001)  # exact floor 7,444,800.99 -> smallest passing quote 7,444,801
    low = run(facts(quoted_rate_paise=7_444_800, rate_unit="10g", benchmark=b), rate_rule())
    ok = run(facts(quoted_rate_paise=7_444_801, rate_unit="10g", benchmark=b), rate_rule())
    assert codes(low) == ["rate_below_floor"] and ok.passed
    assert low.violations[0].numbers["floor_paise"] == 7_444_801


def test_zero_tolerance_means_never_below_benchmark() -> None:
    rule = rate_rule(bps=0)
    assert run(facts(quoted_rate_paise=BENCH, rate_unit="10g", benchmark=bench()), rule).passed
    below = run(facts(quoted_rate_paise=BENCH - 1, rate_unit="10g", benchmark=bench()), rule)
    assert codes(below) == ["rate_below_floor"]


def test_quote_above_benchmark_passes() -> None:
    f = facts(quoted_rate_paise=BENCH + 50_000, rate_unit="10g", benchmark=bench())
    assert run(f, rate_rule()).passed


def test_missing_benchmark_fails_closed() -> None:
    res = run(facts(quoted_rate_paise=BENCH, rate_unit="10g"), rate_rule())
    assert codes(res) == ["benchmark_missing"]
    assert "benchmark rate set nahi" in res.violations[0].reason_hi
    assert "No benchmark rate" in res.violations[0].reason_en


def test_benchmark_exactly_at_max_age_is_fresh_one_second_over_is_stale() -> None:
    q = {"quoted_rate_paise": BENCH, "rate_unit": "10g"}
    fresh = run(facts(**q, benchmark=bench(age_s=DAY)), rate_rule())
    stale = run(facts(**q, benchmark=bench(age_s=DAY + 1)), rate_rule())
    assert fresh.passed and codes(stale) == ["benchmark_stale"]
    assert stale.violations[0].numbers == {"age_seconds": DAY + 1, "max_age_seconds": DAY}


def test_stale_benchmark_blocks_even_a_great_quote() -> None:
    f = facts(quoted_rate_paise=BENCH * 2, rate_unit="10g", benchmark=bench(age_s=DAY * 3))
    assert codes(run(f, rate_rule())) == ["benchmark_stale"]
    assert "72 ghante" in run(f, rate_rule()).violations[0].reason_hi


def test_benchmark_from_the_future_fails_closed() -> None:
    b = BenchmarkFacts(value_paise=BENCH, unit="10g", set_at=NOW + timedelta(seconds=1))
    f = facts(quoted_rate_paise=BENCH, rate_unit="10g", benchmark=b)
    assert codes(run(f, rate_rule())) == ["benchmark_future"]


def test_unit_mismatch_fails_closed() -> None:
    f = facts(quoted_rate_paise=BENCH, rate_unit="gram", benchmark=bench(unit="10g"))
    assert codes(run(f, rate_rule())) == ["unit_mismatch"]


@pytest.mark.parametrize("missing", ["quoted_rate_paise", "rate_unit"])
def test_missing_rate_facts_fail_closed(missing: str) -> None:
    kw: dict[str, Any] = {"quoted_rate_paise": BENCH, "rate_unit": "10g", "benchmark": bench()}
    kw[missing] = None
    assert codes(run(facts(**kw), rate_rule())) == ["fact_missing"]


def test_reason_text_carries_indian_formatted_numbers_in_both_languages() -> None:
    f = facts(quoted_rate_paise=7_000_000, rate_unit="10g", benchmark=bench())
    (v,) = run(f, rate_rule()).violations
    for text in (v.reason_hi, v.reason_en):
        assert "₹70,000" in text and "₹75,200" in text and "₹74,448" in text and "1%" in text
    assert v.numbers == {"quoted_paise": 7_000_000, "benchmark_paise": BENCH,
                         "floor_paise": 7_444_800, "max_below_bps": 100}  # fmt: skip


# --- discount cap ------------------------------------------------------------------------


def disc_rule(bps: int = 30) -> RuleSpec:
    return parse_rule("disc", "discount_cap_pct", {"max_discount_bps": bps})


def test_discount_exactly_at_cap_passes_one_paisa_over_fails() -> None:
    lst = 10_000_000  # cap 0.3% = 30,000 paise
    at = run(facts(list_value_paise=lst, order_value_paise=lst - 30_000), disc_rule())
    over = run(facts(list_value_paise=lst, order_value_paise=lst - 30_001), disc_rule())
    assert at.passed and codes(over) == ["discount_over_cap"]
    assert over.violations[0].numbers["discount_paise"] == 30_001
    assert "0.3%" in over.violations[0].reason_hi


def test_discount_cap_with_non_divisible_list_has_no_rounding_slack() -> None:
    lst = 10_000_001  # 0.3% = 30,000.003 -> 30,000 passes, 30,001 fails
    assert run(facts(list_value_paise=lst, order_value_paise=lst - 30_000), disc_rule()).passed
    over = run(facts(list_value_paise=lst, order_value_paise=lst - 30_001), disc_rule())
    assert codes(over) == ["discount_over_cap"]


def test_order_above_list_is_not_a_discount() -> None:
    assert run(facts(list_value_paise=1_000_000, order_value_paise=1_200_000), disc_rule()).passed


@pytest.mark.parametrize("lst", [None, 0])
def test_discount_without_a_list_price_fails_closed(lst: int | None) -> None:
    assert codes(run(facts(list_value_paise=lst), disc_rule())) == ["fact_missing"]


# --- credit limit ------------------------------------------------------------------------


def credit_rule(default: int = 5_000_000, per: dict[str, int] | None = None) -> RuleSpec:
    return parse_rule("credit", "credit_limit_per_client",
                      {"default_limit_paise": default, "per_client_limit_paise": per or {}})  # fmt: skip


def test_credit_exactly_at_limit_passes_one_paisa_over_fails() -> None:
    at = run(facts(outstanding_paise=3_000_000, order_value_paise=2_000_000), credit_rule())
    over = run(facts(outstanding_paise=3_000_001, order_value_paise=2_000_000), credit_rule())
    assert at.passed and codes(over) == ["credit_over_limit"]
    assert over.violations[0].numbers["total_paise"] == 5_000_001


def test_payment_received_reduces_new_credit_and_overpayment_never_goes_negative() -> None:
    f = facts(
        outstanding_paise=4_000_000, order_value_paise=2_000_000, payment_received_paise=1_000_000
    )
    assert run(f, credit_rule()).passed  # 4.0M + (2.0M - 1.0M) = 5.0M
    f2 = facts(
        outstanding_paise=5_000_000, order_value_paise=2_000_000, payment_received_paise=9_000_000
    )
    assert run(f2, credit_rule()).passed  # new pending is floored at 0


def test_per_client_limit_overrides_the_default() -> None:
    rule = credit_rule(default=5_000_000, per={"vip": 9_000_000})
    f = {"outstanding_paise": 6_000_000, "order_value_paise": 1_000_000}
    assert codes(run(facts(client_id="c1", **f), rule)) == ["credit_over_limit"]
    assert run(facts(client_id="vip", **f), rule).passed


def test_unknown_outstanding_balance_fails_closed() -> None:
    assert codes(run(facts(outstanding_paise=None), credit_rule())) == ["fact_missing"]


# --- max order value ---------------------------------------------------------------------


def test_max_order_boundary() -> None:
    rule = parse_rule("max", "max_order_value_without_approval", {"max_value_paise": 10_000_000})
    assert run(facts(order_value_paise=10_000_000), rule).passed
    over = run(facts(order_value_paise=10_000_001), rule)
    assert codes(over) == ["order_over_max"] and "₹1,00,000" in over.violations[0].reason_hi


# --- combining rules ---------------------------------------------------------------------


def test_all_violations_are_reported_in_stable_rule_id_order() -> None:
    rules = [
        parse_rule("z_max", "max_order_value_without_approval", {"max_value_paise": 1}),
        rate_rule(),
        credit_rule(),
    ]
    f = facts(order_value_paise=900_000, outstanding_paise=9_000_000,
              quoted_rate_paise=1, rate_unit="10g", benchmark=bench())  # fmt: skip
    res = run(f, *rules)
    assert [v.rule_id for v in res.violations] == ["credit", "floor", "z_max"]
    assert run(f, *reversed(rules)) == res  # input order does not matter


def test_no_rules_means_no_violations() -> None:
    assert run(facts()).passed


def test_evaluation_is_repeatable() -> None:
    f = facts(quoted_rate_paise=1, rate_unit="10g", benchmark=bench())
    assert run(f, rate_rule()) == run(f, rate_rule())


# --- exact-arithmetic oracle -------------------------------------------------------------


def test_floor_check_matches_exact_fraction_arithmetic() -> None:
    rng = random.Random(20260211)
    for _ in range(3000):
        benchmark = rng.randint(1, 50_000_000)
        bps = rng.randint(0, 10_000)
        quoted = rng.randint(0, 60_000_000)
        # Make a good share of cases land within a paisa of the boundary.
        if rng.random() < 0.5:
            quoted = max(0, int(Fraction(benchmark * (10_000 - bps), 10_000)) + rng.randint(-2, 2))
        expected_pass = Fraction(quoted) >= Fraction(benchmark) * (1 - Fraction(bps, 10_000))
        f = facts(quoted_rate_paise=quoted, rate_unit="10g", benchmark=bench(value=benchmark))
        assert run(f, rate_rule(bps=bps)).passed is expected_pass, (benchmark, bps, quoted)


def test_discount_check_matches_exact_fraction_arithmetic() -> None:
    rng = random.Random(7)
    for _ in range(3000):
        lst = rng.randint(1, 100_000_000)
        bps = rng.randint(0, 10_000)
        order = rng.randint(0, lst)
        if rng.random() < 0.5:
            order = max(0, lst - int(Fraction(lst * bps, 10_000)) + rng.randint(-2, 2))
        discount = max(lst - order, 0)
        expected_pass = Fraction(discount, lst) <= Fraction(bps, 10_000)
        res = run(facts(list_value_paise=lst, order_value_paise=order), disc_rule(bps))
        assert res.passed is expected_pass, (lst, bps, order)


# --- strict typing -----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("rule_type", "params"),
    [
        ("discount_cap_pct", {"max_discount_bps": 0.3}),  # float
        ("discount_cap_pct", {"max_discount_bps": True}),  # bool
        ("discount_cap_pct", {"max_discount_bps": "30"}),  # numeric string
        ("discount_cap_pct", {"max_discount_bps": -1}),
        ("discount_cap_pct", {"max_discount_bps": 10_001}),
        ("discount_cap_pct", {}),  # missing
        ("discount_cap_pct", {"max_discount_bps": 30, "extra": 1}),  # unknown key
        ("rate_floor_vs_benchmark", {"max_below_bps": 100, "max_benchmark_age_seconds": 0}),
        ("credit_limit_per_client", {"default_limit_paise": 1.5}),
        (
            "credit_limit_per_client",
            {"default_limit_paise": 1, "per_client_limit_paise": {"a": 2.5}},
        ),
        ("max_order_value_without_approval", {"max_value_paise": -5}),
        ("totally_unknown_rule", {}),
    ],
)
def test_malformed_rules_are_rejected_at_load_time(rule_type: str, params: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match=r"rule|policy"):
        parse_rule("r", rule_type, params)


@pytest.mark.parametrize(
    "bad",
    [
        {"order_value_paise": 1_000_000.0},
        {"order_value_paise": "1000000"},
        {"order_value_paise": True},
        {"order_value_paise": -1},
        {"quoted_rate_paise": 7.5},
        {"outstanding_paise": 1e6},
        {"payment_received_paise": 0.5},
    ],
)
def test_facts_reject_floats_strings_bools_and_negatives(bad: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        facts(**bad)


def test_benchmark_facts_need_positive_int_and_aware_time() -> None:
    with pytest.raises(ValidationError):
        BenchmarkFacts(value_paise=0, unit="10g", set_at=NOW)
    with pytest.raises(ValidationError):
        BenchmarkFacts(value_paise=75_200.5, unit="10g", set_at=NOW)  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        BenchmarkFacts(value_paise=1, unit="10g", set_at=datetime(2026, 1, 1))  # naive


def test_evaluate_refuses_a_naive_clock() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        evaluate(facts(), [], pack=PACK, now=datetime(2026, 1, 1))
