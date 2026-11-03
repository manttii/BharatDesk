"""Pure scoring functions for the eval harness. No network, no fabricated data: every
number comes from a gold manifest and a predictions file you provide.

Gold manifest line (evals/datasets/<name>/manifest.jsonl):
  {"id": "c001", "audio": "c001.ogg", "transcript": "...", "intent": "LOG_VISIT",
   "fields": {"client": "Rajendra Jewellers", "order.weight_mg": 150000,
              "payment.received_paise": 4500000}}
Prediction line (written by the pipeline runner, Phase 2):
  {"id": "c001", "transcript": "...", "intent": "LOG_VISIT", "fields": {...},
   "auto_confirmed": false, "asked_confirmation": true, "latency_ms": 2400,
   "cost_paise": 12}
Fields whose name ends in _paise or _mg are NUMERIC (integers) and drive the headline
metrics. Rates here are Fractions of counts, so nothing is rounded until display.
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any

from app.pipeline.numerals import (
    cross_check_money,
    cross_check_weight,
    parse_quantities,
)

NUMERIC_SUFFIXES = ("_paise", "_mg")
_PUNCT = re.compile(r"[\s,.;:!?।\"'()\[\]]+")


def is_numeric_field(name: str) -> bool:
    return name.endswith(NUMERIC_SUFFIXES)


def normalise_text(text: str) -> str:
    return _PUNCT.sub(" ", unicodedata.normalize("NFC", text).casefold()).strip()


def levenshtein(a: Sequence[str], b: Sequence[str]) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, start=1):
        cur = [i]
        for j, y in enumerate(b, start=1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def word_errors(ref: str, hyp: str) -> tuple[int, int]:
    r, h = normalise_text(ref).split(), normalise_text(hyp).split()
    return levenshtein(r, h), len(r)


def char_errors(ref: str, hyp: str) -> tuple[int, int]:
    r, h = normalise_text(ref), normalise_text(hyp)
    return levenshtein(list(r), list(h)), len(r)


def percentile(values: Sequence[int], pct: int) -> int | None:
    """Nearest-rank percentile (integer in, integer out)."""
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, -(-pct * len(ordered) // 100))
    return ordered[rank - 1]


def _rate(num: int, den: int) -> Fraction | None:
    return Fraction(num, den) if den else None


# --------------------------------------------------------------------------- data


@dataclass(frozen=True)
class Gold:
    id: str
    audio: str
    transcript: str
    intent: str
    fields: dict[str, Any]


@dataclass(frozen=True)
class Pred:
    id: str
    transcript: str = ""
    intent: str = ""
    fields: dict[str, Any] = field(default_factory=dict)
    auto_confirmed: bool = False
    asked_confirmation: bool = False
    latency_ms: int | None = None
    cost_paise: int | None = None


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{n}: invalid JSON ({exc.msg})") from exc
    return rows


def load_gold(path: Path) -> list[Gold]:
    out: list[Gold] = []
    for r in read_jsonl(path):
        for k, v in r.get("fields", {}).items():
            if is_numeric_field(k) and (isinstance(v, bool) or not isinstance(v, int)):
                raise ValueError(f"{path}: gold {r['id']}.{k} must be an integer (got {v!r})")
        out.append(Gold(r["id"], r.get("audio", ""), r.get("transcript", ""),
                        r.get("intent", ""), r.get("fields", {})))  # fmt: skip
    ids = [g.id for g in out]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{path}: duplicate ids in manifest")
    return out


def load_preds(path: Path) -> list[Pred]:
    return [
        Pred(
            id=r["id"],
            transcript=r.get("transcript", ""),
            intent=r.get("intent", ""),
            fields=r.get("fields", {}),
            auto_confirmed=bool(r.get("auto_confirmed", False)),
            asked_confirmation=bool(r.get("asked_confirmation", False)),
            latency_ms=r.get("latency_ms"),
            cost_paise=r.get("cost_paise"),
        )
        for r in read_jsonl(path)
    ]


# --------------------------------------------------------------------------- scoring


@dataclass
class Report:
    n_gold: int = 0
    n_predicted: int = 0
    wer: Fraction | None = None
    cer: Fraction | None = None
    intent_accuracy: Fraction | None = None
    field_exact: dict[str, Fraction] = field(default_factory=dict)
    numeric_exact: Fraction | None = None  # HEADLINE
    numeric_total: int = 0
    numeric_wrong: int = 0
    confirmation_rate: Fraction | None = None
    auto_confirmed: int = 0
    false_auto_confirms: int = 0
    false_auto_confirm_rate: Fraction | None = None  # target: 0
    latency_p50_ms: int | None = None
    latency_p95_ms: int | None = None
    mean_cost_paise: Fraction | None = None
    failures: list[str] = field(default_factory=list)


def score(gold: list[Gold], preds: list[Pred]) -> Report:
    by_id = {p.id: p for p in preds}
    rep = Report(n_gold=len(gold))
    w_err = w_len = c_err = c_len = intent_ok = intent_n = 0
    per_field_ok: dict[str, int] = {}
    per_field_n: dict[str, int] = {}
    asked = 0
    latencies: list[int] = []
    costs: list[int] = []

    for g in gold:
        p = by_id.get(g.id)
        if p is None:
            rep.failures.append(f"{g.id}: no prediction")
        else:
            rep.n_predicted += 1
        pred_fields = p.fields if p else {}

        if p is not None and g.transcript:
            e, n = word_errors(g.transcript, p.transcript)
            w_err, w_len = w_err + e, w_len + n
            e, n = char_errors(g.transcript, p.transcript)
            c_err, c_len = c_err + e, c_len + n
        if g.intent:
            intent_n += 1
            intent_ok += p is not None and p.intent == g.intent

        wrong_numeric: list[str] = []
        for name, want in g.fields.items():
            got = pred_fields.get(name)
            ok = got == want and type(got) is type(want)  # 150000 != 150000.0
            per_field_n[name] = per_field_n.get(name, 0) + 1
            per_field_ok[name] = per_field_ok.get(name, 0) + ok
            if is_numeric_field(name):
                rep.numeric_total += 1
                if not ok:
                    rep.numeric_wrong += 1
                    wrong_numeric.append(f"{name}: gold={want!r} pred={got!r}")
        if wrong_numeric:
            rep.failures.append(f"{g.id}: " + "; ".join(wrong_numeric))

        if p is not None:
            asked += p.asked_confirmation
            if p.auto_confirmed:
                rep.auto_confirmed += 1
                if wrong_numeric:
                    rep.false_auto_confirms += 1
            if p.latency_ms is not None:
                latencies.append(p.latency_ms)
            if p.cost_paise is not None:
                costs.append(p.cost_paise)

    rep.wer, rep.cer = _rate(w_err, w_len), _rate(c_err, c_len)
    rep.intent_accuracy = _rate(intent_ok, intent_n)
    rep.field_exact = {k: Fraction(per_field_ok[k], n) for k, n in sorted(per_field_n.items())}
    rep.numeric_exact = _rate(rep.numeric_total - rep.numeric_wrong, rep.numeric_total)
    rep.confirmation_rate = _rate(asked, rep.n_predicted)
    rep.false_auto_confirm_rate = _rate(rep.false_auto_confirms, rep.auto_confirmed)
    rep.latency_p50_ms, rep.latency_p95_ms = percentile(latencies, 50), percentile(latencies, 95)
    rep.mean_cost_paise = _rate(sum(costs), len(costs))
    return rep


@dataclass
class NumeralsReport:
    checked: int = 0
    supported: int = 0
    unsupported: list[str] = field(default_factory=list)

    @property
    def rate(self) -> Fraction | None:
        return _rate(self.supported, self.checked)


def score_numerals(gold: list[Gold]) -> NumeralsReport:
    """How often does the deterministic parser find each gold amount/weight in the GOLD
    transcript? Needs no STT or LLM. A miss means a parser gap OR a number the rep never
    said aloud (e.g. a derived total), so read the list; it is not an accuracy figure."""
    rep = NumeralsReport()
    for g in gold:
        if not g.transcript:
            continue
        quantities = parse_quantities(g.transcript)
        for name, want in g.fields.items():
            if not is_numeric_field(name):
                continue
            rep.checked += 1
            check = (
                cross_check_money(want, quantities)
                if name.endswith("_paise")
                else cross_check_weight(want, quantities)
            )
            if check.agrees:
                rep.supported += 1
            else:
                rep.unsupported.append(
                    f"{g.id}.{name}: gold={want} parser={list(check.candidates)}"
                )
    return rep


# --------------------------------------------------------------------------- report


def _pct(x: Fraction | None) -> str:
    return "n/a" if x is None else f"{float(x):.1%}"


def render_markdown(
    rep: Report | None, numerals: NumeralsReport | None, dataset: str, max_failures: int = 25
) -> str:
    lines = [f"# BharatDesk eval report: {dataset}", ""]
    if numerals is not None:
        lines += [
            "## Numerals parser vs gold transcripts (no STT/LLM involved)",
            f"- numeric gold fields checked: {numerals.checked}",
            f"- found in transcript by parser: {numerals.supported} ({_pct(numerals.rate)})",
            "",
        ]
        if numerals.unsupported:
            lines += ["Not found (parser gap, or never spoken aloud):", ""]
            lines += [f"- {u}" for u in numerals.unsupported[:max_failures]] + [""]
    if rep is None:
        lines += ["## Pipeline metrics", "No predictions file: STT/LLM metrics not computed.", ""]
        return "\n".join(lines)
    lines += [
        "## Pipeline metrics",
        f"- clips in gold set: {rep.n_gold}; with predictions: {rep.n_predicted}",
        f"- **numeric exact-match (headline): {_pct(rep.numeric_exact)}** "
        f"({rep.numeric_total - rep.numeric_wrong}/{rep.numeric_total})",
        f"- **false auto-confirm rate (target 0): {_pct(rep.false_auto_confirm_rate)}** "
        f"({rep.false_auto_confirms}/{rep.auto_confirmed} auto-confirmed)",
        f"- confirmation-trigger rate: {_pct(rep.confirmation_rate)}",
        f"- intent accuracy: {_pct(rep.intent_accuracy)}",
        f"- transcript WER: {_pct(rep.wer)}; CER: {_pct(rep.cer)}",
        f"- latency p50/p95: {rep.latency_p50_ms} / {rep.latency_p95_ms} ms",
        f"- mean cost per message: "
        f"{'n/a' if rep.mean_cost_paise is None else f'{float(rep.mean_cost_paise):.2f} paise'}",
        "",
        "### Per-field exact match",
        *[f"- {k}: {_pct(v)}" for k, v in rep.field_exact.items()],
        "",
        "### Failure cases",
        *([f"- {f}" for f in rep.failures[:max_failures]] or ["- none"]),
        "",
    ]
    return "\n".join(lines)
