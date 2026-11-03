"""Add one labelled clip to a dataset manifest.

    python -m scripts.label_clip --dataset evals/datasets/pilot1 --audio path/to/clip.ogg

You are asked for what was actually SAID (gold transcript), the intent, and the fields.
Numeric fields (names ending _paise or _mg) must be whole integers; they are echoed back
in rupees / grams so a slipped zero is obvious before it becomes a gold label.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

from app.units import format_inr, format_weight_mg
from evals.metrics import is_numeric_field, read_jsonl

INTENTS = ("LOG_VISIT", "CLIENT_DOSSIER_QUERY", "SET_BENCHMARK", "HELP", "UNKNOWN")


def parse_field(raw: str) -> tuple[str, Any]:
    """'order.weight_mg=150000' -> ('order.weight_mg', 150000); text fields stay strings."""
    name, sep, value = raw.partition("=")
    name, value = name.strip(), value.strip()
    if not sep or not name or not value:
        raise ValueError("use name=value")
    if is_numeric_field(name):
        if not value.lstrip("-").isdigit():
            raise ValueError(f"{name} must be a whole number (integer paise / milligrams)")
        return name, int(value)
    return name, value


def readback(name: str, value: Any) -> str:
    if name.endswith("_paise"):
        return f"{name} = {value}  ({format_inr(value)})"
    if name.endswith("_mg"):
        return f"{name} = {value}  ({format_weight_mg(value)})"
    return f"{name} = {value}"


def make_entry(
    clip_id: str,
    audio_name: str,
    audio_sha256: str,
    transcript: str,
    intent: str,
    fields: dict[str, Any],
    notes: str = "",
) -> dict[str, Any]:
    if intent not in INTENTS:
        raise ValueError(f"intent must be one of {INTENTS}")
    return {
        "id": clip_id,
        "audio": audio_name,
        "audio_sha256": audio_sha256,
        "transcript": transcript,
        "intent": intent,
        "fields": fields,
        "notes": notes,
    }


def append_entry(manifest: Path, entry: dict[str, Any]) -> None:
    if manifest.exists() and any(r["id"] == entry["id"] for r in read_jsonl(manifest)):
        raise ValueError(f"id {entry['id']!r} already in {manifest}")
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--audio", type=Path, required=True)
    ap.add_argument("--id", help="clip id (default: audio file stem)")
    args = ap.parse_args()

    clip_id = args.id or args.audio.stem
    data = args.audio.read_bytes()
    transcript = input("Gold transcript (exactly what was said): ").strip()
    intent = input(f"Intent {INTENTS}: ").strip().upper()
    fields: dict[str, Any] = {}
    print("Fields as name=value, blank line to finish (e.g. order.weight_mg=150000):")
    while raw := input("> ").strip():
        try:
            name, value = parse_field(raw)
        except ValueError as exc:
            print(f"  skipped: {exc}")
            continue
        fields[name] = value
        print("  " + readback(name, value))
    notes = input("Notes (noise, speaker, accent; optional): ").strip()
    entry = make_entry(
        clip_id,
        args.audio.name,
        hashlib.sha256(data).hexdigest(),
        transcript,
        intent,
        fields,
        notes,
    )
    args.dataset.mkdir(parents=True, exist_ok=True)
    dest = args.dataset / args.audio.name
    if not dest.exists():
        shutil.copy2(args.audio, dest)
    append_entry(args.dataset / "manifest.jsonl", entry)
    print(f"added {clip_id} to {args.dataset / 'manifest.jsonl'}")


if __name__ == "__main__":
    main()
