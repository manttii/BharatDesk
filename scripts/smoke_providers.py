"""One live call to each configured provider, to prove your keys and model names work.

    python -m scripts.smoke_providers

Checks (and prints OK/FAIL, never a key):
  * Groq:   sends 1 second of silence as OGG-less WAV and expects an HTTP 200.
  * Gemini: GEMINI_MODEL is available, and a tiny structured-JSON call round-trips.
These are the two integrations the automated tests can only mock, so run this once after
filling in .env. It costs a fraction of a paisa.
"""

from __future__ import annotations

import asyncio
import io
import json
import os
import wave
from pathlib import Path

from app.config import get_settings
from app.providers.llm import LLMError, LLMUnavailableError, build_llm
from app.providers.sheets import GSpreadSheets
from app.providers.stt import GroqSTT, STTError


def silence_wav(seconds: int = 1, rate: int = 16_000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"\x00\x00" * rate * seconds)
    return buf.getvalue()


async def main() -> int:
    cfg = get_settings()
    failures = 0
    if cfg.groq_api_key and cfg.whisper_model:
        stt = GroqSTT(cfg.groq_api_key.get_secret_value(), cfg.whisper_model)
        try:
            res = await stt.transcribe(
                silence_wav(),
                filename="silence.wav",
                mime_type="audio/wav",
                language="hi",
                prompt=None,
            )
            print(
                f"Groq   OK   model={res.model} latency={res.latency_ms}ms "
                f"no_speech_prob={res.no_speech_prob} text={res.text!a}"
            )
        except STTError as exc:
            failures += 1
            print(f"Groq   FAIL {exc}")
    else:
        print("Groq   skipped (set GROQ_API_KEY and WHISPER_MODEL)")
    llm = build_llm(cfg)
    if llm is not None:
        try:
            await llm.ensure_available()
            schema = {
                "type": "object",
                "properties": {"ok": {"type": "boolean"}},
                "required": ["ok"],
            }
            res2 = await llm.generate_json(
                system="Reply with JSON.", user='Return {"ok": true}', schema=schema
            )
            print(
                f"LLM    OK   model={res2.model} latency={res2.latency_ms}ms reply={res2.text.strip()!a}"
            )
        except (LLMUnavailableError, LLMError) as exc:
            failures += 1
            print(f"LLM    FAIL {exc}")
    else:
        print("LLM    skipped (set GEMINI_API_KEY+GEMINI_MODEL or NVIDIA_API_KEY+NVIDIA_MODEL)")
    sheet_id = os.environ.get("SMOKE_SHEET_ID")
    if cfg.google_service_account_json or cfg.google_service_account_file:
        raw = (
            cfg.google_service_account_json.get_secret_value()
            if cfg.google_service_account_json
            else await asyncio.to_thread(
                Path(str(cfg.google_service_account_file)).read_text, encoding="utf-8"
            )
        )
        sheets = GSpreadSheets(json.loads(raw))
        print(
            f"Sheets service account: {sheets.service_account_email} (share your sheet with this email)"
        )
        if sheet_id:
            ok = await sheets.check_access(sheet_id)
            failures += 0 if ok else 1
            print(f"Sheets {'OK  ' if ok else 'FAIL'} access to the spreadsheet in SMOKE_SHEET_ID")
        else:
            print("Sheets access check skipped (set SMOKE_SHEET_ID to a sheet id to test)")
    else:
        print("Sheets skipped (set GOOGLE_SERVICE_ACCOUNT_FILE or GOOGLE_SERVICE_ACCOUNT_JSON)")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
