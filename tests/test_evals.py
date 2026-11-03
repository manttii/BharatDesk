"""The eval harness maths, tested on tiny synthetic data (never presented as results)."""

from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path

import pytest

from evals.metrics import (
    Gold,
    Pred,
    levenshtein,
    load_gold,
    percentile,
    render_markdown,
    score,
    score_numerals,
    word_errors,
)
from evals.run_evals import main as run_evals_main
from scripts.label_clip import append_entry, make_entry, parse_field, readback


def test_levenshtein_known_values() -> None:
    assert levenshtein(list("kitten"), list("sitting")) == 3
    assert levenshtein([], list("abc")) == 3
    assert levenshtein(list("abc"), list("abc")) == 0


def test_word_error_counts_ignore_case_and_punctuation() -> None:
    assert word_errors("Dedh sau gram.", "dedh sau gram") == (0, 3)
    assert word_errors("dedh sau gram", "dedh so gram") == (1, 3)
    assert word_errors("paintalis hazaar", "chalis hazaar") == (1, 2)


def test_percentile_nearest_rank() -> None:
    vals = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    assert (percentile(vals, 50), percentile(vals, 95), percentile(vals, 100)) == (50, 100, 100)
    assert percentile([7], 95) == 7 and percentile([], 50) is None


GOLD = [
    Gold("a", "a.ogg", "dedh sau gram paintalis hazaar", "LOG_VISIT",
         {"client": "Rajendra", "order.weight_mg": 150_000, "payment.received_paise": 4_500_000}),
    Gold("b", "b.ogg", "do sau gram", "LOG_VISIT", {"order.weight_mg": 200_000}),
    Gold("c", "c.ogg", "aaj ka rate", "SET_BENCHMARK", {"benchmark_paise": 7_520_000}),
]  # fmt: skip


def _pred(id_: str, fields: dict[str, object], **kw: object) -> Pred:
    return Pred(id=id_, transcript="x", intent="LOG_VISIT", fields=fields, **kw)  # type: ignore[arg-type]


def test_numeric_exact_match_and_false_auto_confirm() -> None:
    preds = [
        # a: right numbers, auto-confirmed (fine)
        _pred("a", {"client": "Rajendra", "order.weight_mg": 150_000,
                    "payment.received_paise": 4_500_000}, auto_confirmed=True),
        # b: WRONG weight but auto-confirmed -> a false auto-confirm
        _pred("b", {"order.weight_mg": 210_000}, auto_confirmed=True),
        # c: wrong, but it asked the rep -> NOT a false auto-confirm
        Pred("c", "x", "SET_BENCHMARK", {"benchmark_paise": 7_250_000}, asked_confirmation=True),
    ]  # fmt: skip
    rep = score(GOLD, preds)
    assert rep.numeric_total == 4 and rep.numeric_wrong == 2
    assert rep.numeric_exact == Fraction(1, 2)
    assert (rep.auto_confirmed, rep.false_auto_confirms) == (2, 1)
    assert rep.false_auto_confirm_rate == Fraction(1, 2)
    assert rep.confirmation_rate == Fraction(1, 3)
    assert rep.intent_accuracy == Fraction(3, 3)
    assert rep.field_exact["order.weight_mg"] == Fraction(1, 2)
    assert any(f.startswith("b:") for f in rep.failures)


def test_float_is_not_equal_to_the_right_integer() -> None:
    rep = score(GOLD[1:2], [_pred("b", {"order.weight_mg": 200_000.0})])
    assert rep.numeric_wrong == 1  # 200000.0 is a float: counted wrong, never silently equal


def test_missing_prediction_counts_every_numeric_field_wrong() -> None:
    rep = score(GOLD, [])
    assert rep.n_predicted == 0 and rep.numeric_wrong == rep.numeric_total == 4
    assert rep.false_auto_confirm_rate is None  # nothing was auto-confirmed
    assert all("no prediction" in f or "gold=" in f for f in rep.failures)


def test_perfect_run_scores_one_and_zero_false_confirms() -> None:
    preds = [_pred(g.id, dict(g.fields), auto_confirmed=True) for g in GOLD]
    preds[2] = Pred("c", "x", "SET_BENCHMARK", dict(GOLD[2].fields), auto_confirmed=True)
    rep = score(GOLD, preds)
    assert rep.numeric_exact == 1 and rep.false_auto_confirms == 0
    assert rep.false_auto_confirm_rate == 0


def test_latency_and_cost_aggregation() -> None:
    preds = [
        _pred("a", {}, latency_ms=1000, cost_paise=10),
        _pred("b", {}, latency_ms=2000, cost_paise=20),
        _pred("c", {}, latency_ms=9000, cost_paise=30),
    ]
    rep = score(GOLD, preds)
    assert (rep.latency_p50_ms, rep.latency_p95_ms) == (2000, 9000)
    assert rep.mean_cost_paise == Fraction(20)


def test_numerals_support_against_gold_transcripts() -> None:
    rep = score_numerals(GOLD)
    # a: weight + advance found; b: weight found; c: benchmark never spoken -> unsupported
    assert (rep.checked, rep.supported) == (4, 3)
    assert len(rep.unsupported) == 1 and rep.unsupported[0].startswith("c.benchmark_paise")


def test_markdown_report_labels_headline_metrics() -> None:
    md = render_markdown(score(GOLD, []), score_numerals(GOLD), "synthetic")
    assert "numeric exact-match (headline)" in md and "false auto-confirm rate (target 0)" in md


def test_load_gold_rejects_float_numeric_labels(tmp_path: Path) -> None:
    p = tmp_path / "manifest.jsonl"
    p.write_text(json.dumps({"id": "x", "fields": {"order.value_paise": 4500000.0}}) + "\n")
    with pytest.raises(ValueError, match="must be an integer"):
        load_gold(p)


def test_load_gold_rejects_duplicate_ids(tmp_path: Path) -> None:
    p = tmp_path / "manifest.jsonl"
    p.write_text('{"id": "x"}\n{"id": "x"}\n')
    with pytest.raises(ValueError, match="duplicate"):
        load_gold(p)


# --- CLI: never fabricates -----------------------------------------------------------------


def test_run_evals_with_no_data_says_so_and_invents_nothing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert run_evals_main(["--dataset", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert out.startswith("no data") and "%" not in out and "headline" not in out


def test_run_evals_numerals_only_on_a_real_manifest(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    entry = make_entry("c1", "c1.ogg", "0" * 64, "dedh sau gram", "LOG_VISIT",
                       {"order.weight_mg": 150_000})  # fmt: skip
    append_entry(tmp_path / "manifest.jsonl", entry)
    assert run_evals_main(["--dataset", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "found in transcript by parser: 1 (100.0%)" in out
    assert "No predictions file" in out  # STT/LLM metrics are NOT made up


# --- labelling helper ----------------------------------------------------------------------


def test_label_fields_enforce_integers_for_money_and_weight() -> None:
    assert parse_field("order.weight_mg=150000") == ("order.weight_mg", 150_000)
    assert parse_field("client=Rajendra Jewellers") == ("client", "Rajendra Jewellers")
    for bad in ("order.weight_mg=150.5", "x_paise=4,500,000", "x_paise=abc", "novalue", "=3"):
        with pytest.raises(ValueError, match=r"whole number|name=value"):
            parse_field(bad)


def test_label_readback_shows_rupees_and_grams() -> None:
    assert "₹45,000" in readback("payment.received_paise", 4_500_000)
    assert "150 gram" in readback("order.weight_mg", 150_000)


def test_label_manifest_refuses_duplicate_ids(tmp_path: Path) -> None:
    m = tmp_path / "manifest.jsonl"
    entry = make_entry("c1", "c1.ogg", "0" * 64, "x", "HELP", {})
    append_entry(m, entry)
    with pytest.raises(ValueError, match="already in"):
        append_entry(m, entry)
    with pytest.raises(ValueError, match="intent must be"):
        make_entry("c2", "c2.ogg", "0" * 64, "x", "NOT_AN_INTENT", {})
