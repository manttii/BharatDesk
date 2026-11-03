"""Run the eval harness.

    python -m evals.run_evals                      # scan evals/datasets/*/manifest.jsonl
    python -m evals.run_evals --dataset evals/datasets/pilot1 --predictions preds.jsonl
    python -m evals.run_evals --out report.md

With no dataset it prints "no data" and how to collect one. It never invents results.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from evals.metrics import (
    NumeralsReport,
    Report,
    load_gold,
    load_preds,
    render_markdown,
    score,
    score_numerals,
)

DATASETS_DIR = Path(__file__).resolve().parent / "datasets"

NO_DATA = """no data

There is no labelled dataset to evaluate, so no metrics were computed (none are invented).
To collect one:
  1. Record 50-100 real voice notes from the customer's reps: different people, noisy
     shops, fast speech, Hinglish. See evals/README.md for the checklist.
  2. Label each clip:
       python -m scripts.label_clip --dataset evals/datasets/pilot1 --audio clip.ogg
  3. Re-run:
       python -m evals.run_evals --dataset evals/datasets/pilot1
"""


def find_manifests(dataset: Path | None) -> list[Path]:
    if dataset is not None:
        manifest = dataset / "manifest.jsonl"
        return [manifest] if manifest.is_file() else []
    return sorted(DATASETS_DIR.glob("*/manifest.jsonl"))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", type=Path, help="dataset directory containing manifest.jsonl")
    ap.add_argument("--predictions", type=Path, help="predictions.jsonl from the pipeline runner")
    ap.add_argument("--out", type=Path, help="write the Markdown report here")
    args = ap.parse_args(argv)

    manifests = find_manifests(args.dataset)
    gold_total = 0
    sections: list[str] = []
    for manifest in manifests:
        gold = load_gold(manifest)
        gold_total += len(gold)
        if not gold:
            continue
        numerals: NumeralsReport = score_numerals(gold)
        report: Report | None = None
        if args.predictions is not None:
            report = score(gold, load_preds(args.predictions))
        sections.append(render_markdown(report, numerals, manifest.parent.name))
    if gold_total == 0:
        print(NO_DATA)
        return 0
    text = "\n".join(sections)
    if args.out:
        args.out.write_text(text, encoding="utf-8")
        print(f"report written to {args.out}")
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
