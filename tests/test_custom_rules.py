"""Owner-defined policy rules: a whitelisted expression evaluator that fails closed."""

from __future__ import annotations

import random
import re
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from app.db import TenantRepository
from app.models import PolicyRule
from app.pack import load_pack
from app.pipeline.commit import load_rules
from app.pipeline.expr import (
    MAX_CONSTANT,
    NAMES,
    ExpressionError,
    evaluate,
    expression_names,
    parse_expression,
)
from app.pipeline.policy_engine import evaluate as evaluate_policy
from app.schemas.policy import BenchmarkFacts, PolicyFacts, parse_rule
from tests.conftest import (
    BOSS_CHAT,
    REP_CHAT,
    TRANSCRIPT,
    Env,
    button_data,
    extraction_json,
    to,
)

NOW = datetime(2026, 11, 2, 9, 0, tzinfo=UTC)
PACK = load_pack()


# --- what the language accepts and refuses ---------------------------------------------------

ACCEPTED = [
    "order_value_paise > 5000000",
    "order_value_paise >= 5000000 and payment_received_paise * 10 < order_value_paise",
    "not (order_value_paise <= 100)",
    "quoted_rate_paise < benchmark_paise - 50000",
    "(order_value_paise + outstanding_paise) // 2 > 100 or weight_mg == 0",
    "order_value_paise % 1000 != 0",
    "-order_value_paise < -5",
    "0 < weight_mg < 1000000",
    "benchmark_age_seconds > 3600 and quoted_rate_paise > 0",
    "new_pending_paise * 100 > order_value_paise * 50",
    "list_value_paise - order_value_paise > 0",
]


@pytest.mark.parametrize("text", ACCEPTED)
def test_valid_expressions_parse(text: str) -> None:
    parse_expression(text)
    assert expression_names(text) <= NAMES


REJECTED = [
    ("abs(order_value_paise) > 1", "Call"),
    ("order_value_paise.real > 1", "Attribute"),
    ("order_value_paise[0] > 1", "Subscript"),
    ("(lambda: 1)() > 0", "Lambda"),
    ("[x for x in (1, 2)] == 1", "ListComp"),
    ("'a' == 'a'", "whole-number"),
    ("order_value_paise > 1.5", "whole-number"),
    ("order_value_paise / 2 > 1", "no '/'"),
    ("order_value_paise ** 2 > 1", "'**'"),
    ("order_value_paise & 1 == 1", "bit operations"),
    ("order_value_paise | 1 == 1", "bit operations"),
    ("~order_value_paise < 0", "unary"),
    ("+order_value_paise > 0", "unary"),
    ("order_value_paise << 2 > 1", "bit operations"),
    ("order_value_paise in (1, 2)", "comparisons"),
    ("order_value_paise is 1", "comparisons"),
    ("order_value_paise > 1 if True else 0", "IfExp"),
    ("True", "whole-number"),
    ("None == None", "whole-number"),
    ("secret_key > 1", "unknown name"),
    ("__import__('os').system('x') > 0", "Call"),
    ("(x := 5) > 1", "unknown name"),
    ("f'{order_value_paise}' == '1'", "JoinedStr"),
    ("1; 2", "not a valid expression"),
    ("order_value_paise >", "not a valid expression"),
    ("", "empty"),
    ("   ", "empty"),
    ("a = 1", "not a valid expression"),
    (f"order_value_paise > {MAX_CONSTANT + 1}", "too large"),
    (" and ".join(["order_value_paise > 1"] * 12), "too complex"),
    ("x" * 301, "longer than"),
    ("{} == {}", "Dict"),
    ("await order_value_paise", "not a valid expression"),
]


@pytest.mark.parametrize(("text", "why"), REJECTED)
def test_everything_else_is_refused(text: str, why: str) -> None:
    with pytest.raises(ExpressionError):
        parse_expression(text)  # the exact wording depends on which offending node is met first


STABLE_MESSAGES = {
    "order_value_paise > 1.5": "whole-number",
    "order_value_paise / 2 > 1": "no '/'",
    "order_value_paise ** 2 > 1": "'**'",
    "secret_key > 1": "unknown name",
    "": "empty",
    "x" * 301: "longer than",
    f"order_value_paise > {MAX_CONSTANT + 1}": "too large",
}


@pytest.mark.parametrize(("text", "why"), list(STABLE_MESSAGES.items()))
def test_the_reason_for_a_refusal_is_explained(text: str, why: str) -> None:
    with pytest.raises(ExpressionError, match=re.escape(why)):
        parse_expression(text)


def test_deeply_nested_parentheses_do_not_crash_the_parser() -> None:
    nest = "(" * 150 + "order_value_paise > 1" + ")" * 150
    with pytest.raises(ExpressionError):
        parse_expression(nest)


# --- evaluation ------------------------------------------------------------------------------

V = {n: 0 for n in NAMES} | {"order_value_paise": 1_000_000, "payment_received_paise": 100_000}


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("order_value_paise > 999999", True),
        ("order_value_paise > 1000000", False),  # strict
        ("order_value_paise >= 1000000", True),  # inclusive: exactly at the boundary
        ("payment_received_paise * 10 < order_value_paise", False),  # 1,000,000 < 1,000,000
        ("payment_received_paise * 10 <= order_value_paise", True),
        ("1 + 2 * 3 == 7", True),  # precedence
        ("(1 + 2) * 3 == 9", True),
        ("-7 // 2 == -4", True),  # floor division, like Python
        ("7 % 3 == 1", True),
        ("0 < 5 < 10", True),
        ("0 < 15 < 10", False),
        ("1 < 2 and 3 < 2", False),
        ("1 < 2 or 3 < 2", True),
        ("not (1 < 2)", False),
        ("not 1 < 2 or 2 < 3", True),
    ],
)
def test_evaluation_semantics(text: str, expected: bool) -> None:
    assert evaluate(text, V) is expected


@pytest.mark.parametrize(
    "text",
    [
        "order_value_paise",  # an int is not a condition
        "order_value_paise and 1 > 0",  # 'and' needs conditions
        "not order_value_paise",
        "1 + (2 > 1) == 2",  # no true/false in arithmetic
        "-(1 < 2) < 0",
        "(1 < 2) < 3",
        "order_value_paise // 0 > 1",
        "order_value_paise % 0 > 1",
    ],
)
def test_type_errors_and_division_by_zero_are_errors(text: str) -> None:
    with pytest.raises(ExpressionError):
        evaluate(text, V)


def test_overflow_is_an_error_not_a_slow_multiplication() -> None:
    big = {**V, "order_value_paise": 10**12}
    with pytest.raises(ExpressionError, match="too large"):
        evaluate("order_value_paise * order_value_paise * order_value_paise > 1", big)


def test_a_missing_value_is_an_error() -> None:
    with pytest.raises(ExpressionError, match="no value"):
        evaluate("weight_mg > 1", {"order_value_paise": 1})


def test_all_parts_are_evaluated_so_an_error_is_never_masked_by_short_circuiting() -> None:
    with pytest.raises(ExpressionError, match="division by zero"):
        evaluate("weight_mg > 0 and 10 // weight_mg > 1", {**V, "weight_mg": 0})


def test_random_expressions_match_python_as_an_oracle() -> None:
    """Differential test: our tree walk must agree with Python's own arithmetic on random programs."""
    rng = random.Random(20261102)
    names = ["order_value_paise", "payment_received_paise", "weight_mg", "outstanding_paise"]
    ops = ["+", "-", "*"]
    cmps = ["<", "<=", ">", ">=", "==", "!="]

    def arith(depth: int) -> str:
        if depth == 0 or rng.random() < 0.3:
            return rng.choice(names) if rng.random() < 0.6 else str(rng.randint(0, 50))
        return f"({arith(depth - 1)} {rng.choice(ops)} {arith(depth - 1)})"

    def cond(depth: int) -> str:
        if depth == 0 or rng.random() < 0.4:
            return f"{arith(2)} {rng.choice(cmps)} {arith(2)}"
        joiner = rng.choice(["and", "or"])
        inner = f"({cond(depth - 1)} {joiner} {cond(depth - 1)})"
        return f"not {inner}" if rng.random() < 0.2 else inner

    checked = 0
    for _ in range(2000):
        values = {n: rng.randint(0, 40) for n in names}
        text = cond(2)
        try:
            parse_expression(text)
        except ExpressionError:
            continue  # a program over the size limits is refused, which is its own tested behaviour
        expected = eval(text, {"__builtins__": {}}, dict(values))  # noqa: S307  (test oracle only)
        assert evaluate(text, values) is bool(expected), (text, values)
        checked += 1
    assert checked >= 500, "the generator produced too few in-limit programs to be meaningful"


# --- the policy engine -----------------------------------------------------------------------


def _rule(expression: str, **kw: Any):  # type: ignore[no-untyped-def]
    params = {
        "expression": expression,
        "reason_hi": "Advance bahut kam hai.",
        "reason_en": "Advance is too low.",
    }
    return parse_rule("c1", "custom_expression", {**params, **kw})


def _facts(**kw: Any) -> PolicyFacts:
    base: dict[str, Any] = {
        "client_id": "c",
        "order_value_paise": 6_000_000,
        "payment_received_paise": 100_000,
    }
    return PolicyFacts(**{**base, **kw})


def _run(facts: PolicyFacts, *rules: Any) -> Any:
    return evaluate_policy(facts, list(rules), pack=PACK, now=NOW)


LOW_ADVANCE = "order_value_paise > 5000000 and payment_received_paise * 10 < order_value_paise"


def test_the_expression_describes_the_violation() -> None:
    res = _run(_facts(), _rule(LOW_ADVANCE))
    (v,) = res.violations
    assert (v.code, v.rule_type.value) == ("custom_rule", "custom_expression")
    assert (v.reason_hi, v.reason_en) == ("Advance bahut kam hai.", "Advance is too low.")
    assert v.numbers == {"order_value_paise": 6_000_000, "payment_received_paise": 100_000}


@pytest.mark.parametrize(
    ("order", "received", "violates"),
    [
        (5_000_000, 0, False),
        (5_000_001, 500_000, True),
        (5_000_001, 500_001, False),
        (6_000_000, 600_000, False),
    ],
)
def test_boundaries_of_a_custom_rule(order: int, received: int, violates: bool) -> None:
    res = _run(_facts(order_value_paise=order, payment_received_paise=received), _rule(LOW_ADVANCE))
    assert (not res.passed) is violates


def test_a_fact_the_rule_needs_but_is_missing_fails_closed() -> None:
    res = _run(_facts(), _rule("outstanding_paise > 1000000"))
    assert [v.code for v in res.violations] == ["fact_missing"]
    assert "outstanding_paise" in res.violations[0].reason_en


def test_the_benchmark_derived_facts_are_available() -> None:
    bench = BenchmarkFacts(value_paise=7_520_000, unit="10g", set_at=NOW - timedelta(hours=2))
    f = _facts(quoted_rate_paise=7_000_000, benchmark=bench)
    assert _run(f, _rule("benchmark_paise - quoted_rate_paise > 500000")).violations
    assert _run(
        f, _rule("benchmark_age_seconds == 7200 and quoted_rate_paise < benchmark_paise")
    ).violations
    assert _run(f, _rule("benchmark_age_seconds > 7200")).passed
    assert _run(_facts(), _rule("benchmark_paise > 0")).violations[0].code == "fact_missing"


def test_pending_and_weight_facts() -> None:
    f = _facts(order_value_paise=1_000_000, payment_received_paise=300_000, weight_mg=250_000)
    assert _run(f, _rule("new_pending_paise == 700000 and weight_mg == 250000")).violations
    assert _run(
        _facts(order_value_paise=1_000, payment_received_paise=9_000),
        _rule("new_pending_paise > 0"),
    ).passed


def test_custom_rules_combine_with_builtin_rules_in_stable_order() -> None:
    rules = [
        _rule("order_value_paise > 1"),
        parse_rule("a_max", "max_order_value_without_approval", {"max_value_paise": 1}),
    ]
    res = _run(_facts(), *rules)
    assert [v.rule_id for v in res.violations] == ["a_max", "c1"]
    assert _run(_facts(), *reversed(rules)) == res


@pytest.mark.parametrize(
    "bad",
    [
        {"expression": "abs(1) > 0"},
        {"expression": "unknown_name > 1"},
        {"expression": "order_value_paise > 1.5"},
        {"expression": ""},
        {"reason_hi": ""},
        {"reason_en": "x" * 301},
        {"extra": "field"},
    ],
)
def test_a_bad_custom_rule_is_rejected_when_it_is_loaded(bad: dict[str, str]) -> None:
    good = {"expression": "order_value_paise > 1", "reason_hi": "h", "reason_en": "e"}
    with pytest.raises(ValueError, match=r"rule|policy"):
        parse_rule("c", "custom_expression", {**good, **bad})


def test_reasons_are_plain_text_not_format_strings() -> None:
    odd = _rule("order_value_paise > 1", reason_hi="{0} {name} %s", reason_en="{__class__}")
    (v,) = _run(_facts(), odd).violations
    assert v.reason_hi == "{0} {name} %s" and v.reason_en == "{__class__}"


# --- end to end ------------------------------------------------------------------------------

RULE = ("custom_expression", {"expression": LOW_ADVANCE, "reason_hi": "Advance order ka 10% se kam hai.",
                             "reason_en": "Advance is under 10% of a large order."})  # fmt: skip


async def _org(p: Env):  # type: ignore[no-untyped-def]
    org = await p.org()
    await p.add_client(org, "Rajendra Jewellers")
    return org


async def test_a_custom_rule_routes_a_visit_to_the_boss_with_the_owners_own_words(
    pipe: Env,
) -> None:
    org = await _org(pipe)
    await pipe.add_rules(org, RULE)
    out = await pipe.voice(
        REP_CHAT, TRANSCRIPT, extraction_json()
    )  # Rs 4,12,000 order, Rs 45,000 advance: fine
    out = await pipe.tap(REP_CHAT, button_data(to(REP_CHAT, out)[0], "Haan"))
    assert "Save ho gaya" in to(REP_CHAT, out)[0].text
    low = TRANSCRIPT.replace("paintalis hazaar", "paanch hazaar")  # advance Rs 5,000
    out = await pipe.voice(REP_CHAT, low, extraction_json(received=500_000, value_paise=41_200_000))
    out = await pipe.tap(REP_CHAT, button_data(to(REP_CHAT, out)[0], "Haan"))
    assert "Advance order ka 10% se kam hai." in to(REP_CHAT, out)[0].text
    boss = to(BOSS_CHAT, out)[0]
    assert "Advance order ka 10% se kam hai." in boss.text and "Approve" in [
        b.text for b in boss.buttons[0]
    ]  # type: ignore[index]


async def test_a_custom_rule_that_reads_the_value_makes_it_a_required_question(pipe: Env) -> None:
    org = await _org(pipe)
    await pipe.add_rules(org, RULE)
    out = await pipe.voice(REP_CHAT, TRANSCRIPT, extraction_json(value_paise=None))
    assert to(REP_CHAT, out)[0].text == "Order ki total keemat kitni thi?"


async def test_a_custom_rule_that_reads_the_rate_makes_the_rate_a_required_question(
    pipe: Env,
) -> None:
    org = await _org(pipe)
    await pipe.add_rules(org, ("custom_expression", {"expression": "quoted_rate_paise < 7000000",
                              "reason_hi": "Rate kam hai.", "reason_en": "Rate too low."}))  # fmt: skip
    out = await pipe.voice(REP_CHAT, TRANSCRIPT, extraction_json())
    assert to(REP_CHAT, out)[0].text == "Rate kya tha (10 gram ka)?"


async def test_a_custom_rule_that_needs_neither_asks_nothing_extra(pipe: Env) -> None:
    org = await _org(pipe)
    await pipe.add_rules(org, ("custom_expression", {"expression": "weight_mg > 1000000",
                              "reason_hi": "Wazan zyada.", "reason_en": "Heavy."}))  # fmt: skip
    out = await pipe.voice(REP_CHAT, TRANSCRIPT, extraction_json(value_paise=None))
    assert to(REP_CHAT, out)[0].text.endswith("Sahi hai?")


async def test_a_malformed_custom_rule_in_the_database_fails_closed(pipe: Env) -> None:
    org = await _org(pipe)
    async with pipe.sf() as s, s.begin():
        TenantRepository(s, org.tenant_id).add(
            PolicyRule(
                rule_type="custom_expression",
                params={
                    "expression": "__import__('os').getcwd() == 1",
                    "reason_hi": "x",
                    "reason_en": "y",
                },
            )
        )
        rules, errors = await load_rules(TenantRepository(s, org.tenant_id))
    assert rules == [] and len(errors) == 1  # reported, not silently skipped
    out = await pipe.voice(REP_CHAT, TRANSCRIPT, extraction_json())
    out = await pipe.tap(REP_CHAT, button_data(to(REP_CHAT, out)[0], "Haan"))
    assert to(BOSS_CHAT, out), (
        "a broken rule must send the order to the boss, never wave it through"
    )
