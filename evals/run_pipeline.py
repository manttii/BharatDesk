"""Run the REAL speech + extraction pipeline over a labelled dataset and write
predictions.jsonl for `python -m evals.run_evals --predictions`.

    python -m evals.run_pipeline --dataset evals/datasets/pilot1 --out preds.jsonl [--today 2026-11-02]

Needs GROQ_API_KEY/WHISPER_MODEL and GEMINI_API_KEY/GEMINI_MODEL; without them it stops
with a message and writes nothing (no fake predictions). It uses the same code as the bot:
transcribe_note -> run_extraction -> assess_visit, then asks the same question the bot asks
("would this auto-confirm?") under a strict hypothetical tenant setting, so the false
auto-confirm rate measures the real gate. Client matching needs a database, so the client is
treated as an exact match here; that is the optimistic case for auto-confirm.

Field names written (use them in gold labels): intent, client, order.weight_mg,
order.value_paise, order.rate_paise_per_10g, payment.received_paise, payment.pending_paise,
delivery_date, next_followup_date, benchmark_paise.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from datetime import date
from pathlib import Path
from typing import Any

from app.config import Settings, get_settings
from app.pack import Pack, load_pack
from app.pipeline.assess import assess_visit
from app.pipeline.confirm import TenantSettings, can_auto_confirm
from app.pipeline.extract import run_extraction
from app.pipeline.transcribe import transcribe_note
from app.providers.llm import LLMProvider
from app.providers.stt import STTProvider
from evals.metrics import Gold, load_gold

STRICT = TenantSettings(auto_confirm_enabled=True, auto_confirm_min_confidence=0.98)


async def predict_clip(
    gold: Gold,
    audio: bytes,
    *,
    stt: STTProvider,
    llm: LLMProvider,
    pack: Pack,
    settings: Settings,
    today: date,
    timezone: str = "Asia/Kolkata",
) -> dict[str, Any]:
    tr = await transcribe_note(
        stt,
        audio,
        filename=gold.audio or "clip.ogg",
        mime_type=None,
        client_names=[],
        vocabulary=pack.vocabulary,
        settings=settings,
    )
    run = await run_extraction(llm, tr.text, today=today, timezone=timezone, known_clients=[])
    out: dict[str, Any] = {
        "id": gold.id,
        "transcript": tr.text,
        "intent": run.parsed.intent.value if run.parsed else "",
        "fields": {},
        "auto_confirmed": False,
        "asked_confirmation": True,
        "latency_ms": tr.latency_ms + run.latency_ms,
        "cost_paise": None,
    }
    if run.parsed is None:
        return out
    ext = run.parsed
    fields: dict[str, Any] = {}
    if ext.benchmark_rate_paise is not None:
        fields["benchmark_paise"] = ext.benchmark_rate_paise
    if ext.intent.value == "LOG_VISIT":
        a = assess_visit(
            ext, tr.text, today=today, pack=pack,
            min_confidence=settings.low_confidence_flag, transcript_reasons=tr.reasons,
        )  # fmt: skip
        p = a.payload
        fields["client"] = p.client_name
        if p.order:
            fields["order.weight_mg"] = p.order.weight_mg
            fields["order.value_paise"] = p.order.value_paise
            fields["order.rate_paise_per_10g"] = p.order.rate_paise_per_10g
        if p.payment:
            fields["payment.received_paise"] = p.payment.received_paise
            fields["payment.pending_paise"] = p.payment.pending_paise
        fields["delivery_date"] = p.delivery_date.isoformat() if p.delivery_date else None
        fields["next_followup_date"] = (
            p.next_followup_date.isoformat() if p.next_followup_date else None
        )
        auto = can_auto_confirm(
            a.flags, a.confidences, STRICT, client_exact=True, missing=a.missing
        )
        out["auto_confirmed"], out["asked_confirmation"] = auto, not auto
    out["fields"] = {k: v for k, v in fields.items() if v is not None}
    return out


async def run_dataset(
    gold: list[Gold], dataset_dir: Path, *, stt: STTProvider, llm: LLMProvider,
    pack: Pack, settings: Settings, today: date,
) -> list[dict[str, Any]]:  # fmt: skip
    preds = []
    for g in gold:
        audio_path = dataset_dir / g.audio
        if not audio_path.is_file():
            raise FileNotFoundError(f"{g.id}: audio file {audio_path} not found")
        preds.append(
            await predict_clip(
                g,
                audio_path.read_bytes(),
                stt=stt,
                llm=llm,
                pack=pack,
                settings=settings,
                today=today,
            )
        )
    return preds


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--today", type=date.fromisoformat, default=date.today())
    args = ap.parse_args(argv)

    cfg = get_settings()
    from app.providers.llm import build_llm

    llm = build_llm(cfg)
    if not (cfg.groq_api_key and cfg.whisper_model and llm):
        print("Cannot run: set GROQ_API_KEY, WHISPER_MODEL and an LLM (GEMINI_* or NVIDIA_*).")
        print("Nothing was written; no predictions are ever invented.")
        return 2
    from app.providers.stt import GroqSTT

    manifest = args.dataset / "manifest.jsonl"
    if not manifest.is_file():
        print(f"no data: {manifest} does not exist (see evals/README.md)")
        return 0
    gold = load_gold(manifest)
    stt = GroqSTT(
        cfg.groq_api_key.get_secret_value(), cfg.whisper_model, force_convert=cfg.stt_force_convert
    )

    async def go() -> list[dict[str, Any]]:
        await llm.ensure_available()
        return await run_dataset(gold, args.dataset, stt=stt, llm=llm, pack=load_pack(),
                                 settings=cfg, today=args.today)  # fmt: skip

    preds = asyncio.run(go())
    args.out.write_text(
        "\n".join(json.dumps(p, ensure_ascii=False) for p in preds) + "\n", encoding="utf-8"
    )
    print(f"wrote {len(preds)} predictions to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
