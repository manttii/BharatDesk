"""The eval pipeline runner (with fakes), and a guard that every pack string used in code exists."""

from __future__ import annotations

import io
import json
import re
import wave
from datetime import date
from pathlib import Path

import pytest

from app.bot import notes
from app.config import Settings
from app.pack import load_pack
from app.providers.llm import FakeLLM
from app.providers.stt import FakeSTT
from evals.metrics import load_gold, score
from evals.run_pipeline import main as run_pipeline_main
from evals.run_pipeline import run_dataset
from scripts.label_clip import append_entry, make_entry
from scripts.smoke_providers import silence_wav
from tests.conftest import OGG, TRANSCRIPT, extraction_json

APP = Path(__file__).resolve().parent.parent / "app"


# --- eval pipeline runner ------------------------------------------------------------------


def _dataset(tmp_path: Path) -> Path:
    (tmp_path / "c1.ogg").write_bytes(OGG)
    append_entry(
        tmp_path / "manifest.jsonl",
        make_entry(
            "c1", "c1.ogg", "0" * 64, TRANSCRIPT, "LOG_VISIT",
            {"client": "Rajendra Jewellers", "order.weight_mg": 150_000, "order.value_paise": 41_200_000,
             "payment.received_paise": 4_500_000, "delivery_date": "2026-11-08"},
        ),
    )  # fmt: skip
    return tmp_path


async def test_pipeline_runner_produces_predictions_the_scorer_can_use(tmp_path: Path) -> None:
    ds = _dataset(tmp_path)
    stt, llm = FakeSTT(), FakeLLM()
    stt.say(TRANSCRIPT)
    llm.replies.append(extraction_json())
    gold = load_gold(ds / "manifest.jsonl")
    (pred,) = await run_dataset(
        gold, ds, stt=stt, llm=llm, pack=load_pack(),
        settings=Settings(_env_file=None), today=date(2026, 11, 2),  # type: ignore[call-arg]
    )  # fmt: skip
    assert pred["fields"]["order.value_paise"] == 41_200_000 and isinstance(
        pred["fields"]["order.weight_mg"], int
    )
    assert (
        pred["auto_confirmed"] is True and pred["asked_confirmation"] is False
    )  # clean + confident
    from evals.metrics import Pred

    rep = score(gold, [Pred(id=pred["id"], transcript=pred["transcript"], intent=pred["intent"],
                            fields=pred["fields"], auto_confirmed=pred["auto_confirmed"],
                            asked_confirmation=pred["asked_confirmation"], latency_ms=pred["latency_ms"])])  # fmt: skip
    assert rep.numeric_exact == 1 and rep.false_auto_confirms == 0


async def test_runner_does_not_auto_confirm_a_wrong_number(tmp_path: Path) -> None:
    ds = _dataset(tmp_path)
    stt, llm = FakeSTT(), FakeLLM()
    stt.say(TRANSCRIPT)
    llm.replies.append(extraction_json(received=4_000_000))  # transcript says paintalis (45,000)
    gold = load_gold(ds / "manifest.jsonl")
    (pred,) = await run_dataset(
        gold, ds, stt=stt, llm=llm, pack=load_pack(),
        settings=Settings(_env_file=None), today=date(2026, 11, 2),  # type: ignore[call-arg]
    )  # fmt: skip
    assert pred["auto_confirmed"] is False and pred["asked_confirmation"] is True
    assert pred["fields"]["payment.received_paise"] == 4_000_000  # wrong, but it would have ASKED


async def test_runner_reports_a_missing_audio_file(tmp_path: Path) -> None:
    ds = _dataset(tmp_path)
    (ds / "c1.ogg").unlink()
    with pytest.raises(FileNotFoundError, match="c1"):
        await run_dataset(load_gold(ds / "manifest.jsonl"), ds, stt=FakeSTT(), llm=FakeLLM(),
                          pack=load_pack(), settings=Settings(_env_file=None),  # type: ignore[call-arg]
                          today=date(2026, 11, 2))  # fmt: skip


def test_runner_refuses_without_keys_and_writes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    for var in ("GROQ_API_KEY", "WHISPER_MODEL", "GEMINI_API_KEY", "GEMINI_MODEL"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.chdir(tmp_path)  # no .env here
    from app.config import get_settings

    get_settings.cache_clear()
    out = tmp_path / "preds.jsonl"
    assert run_pipeline_main(["--dataset", str(_dataset(tmp_path)), "--out", str(out)]) == 2
    assert not out.exists() and "ever invented" in capsys.readouterr().out
    get_settings.cache_clear()


def test_smoke_script_silence_is_a_valid_wav() -> None:
    with wave.open(io.BytesIO(silence_wav(2)), "rb") as w:
        assert (w.getnchannels(), w.getframerate(), w.getnframes()) == (1, 16_000, 32_000)


# --- pack strings used in code must exist ---------------------------------------------------


def _used_keys() -> set[str]:
    keys: set[str] = set()
    pattern = re.compile(r"""\.(?:t|te)\(\s*\n?\s*["']([a-z0-9_]+)["']""")
    for path in APP.rglob("*.py"):
        keys |= set(pattern.findall(path.read_text(encoding="utf-8")))
    keys |= set(notes._ASK_KEYS.values())
    from app.pipeline.confirm import _FIELD_LABELS

    keys |= set(_FIELD_LABELS.values())
    # keys chosen at runtime from small literal maps / conditionals:
    keys |= {"voice_too_short", "voice_too_long", "voice_bad_type", "media_too_large",
             "llm_unavailable", "extract_failed", "help_boss", "help_worker",
             "counter_prompt_rate", "counter_prompt_discount", "approval_reminder_boss",
             "approval_request_boss"}  # fmt: skip
    return keys


def test_every_pack_string_used_in_code_exists_in_the_yaml() -> None:
    pack = load_pack()
    missing = sorted(k for k in _used_keys() if k not in pack.strings)
    assert missing == []


def test_every_policy_violation_code_has_hindi_and_english_text() -> None:
    pack = load_pack()
    for key in pack.strings:
        if key.startswith("policy_"):
            assert key in pack.strings_en, key
    assert json.dumps(sorted(pack.bounds)) and pack.bounds["weight_mg_min"] > 0
