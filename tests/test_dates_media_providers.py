"""Date resolver, media gates, and the Groq / Gemini provider adapters."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx
from google.genai import errors as genai_errors
from pydantic import ValidationError

from app.config import Settings
from app.pipeline.dates import resolve_date_phrase
from app.pipeline.media import (
    MediaError,
    MediaStore,
    sha256_hex,
    sniff_audio_extension,
    validate_voice_event,
)
from app.pipeline.transcribe import build_stt_prompt, has_foreign_script
from app.providers.llm import GeminiLLM, LLMError, LLMUnavailableError
from app.providers.stt import GROQ_TRANSCRIPTIONS_URL, GroqSTT, STTError
from app.schemas.events import IncomingEvent

MON = date(2026, 11, 2)  # a Monday


@pytest.mark.parametrize(
    ("text", "direction", "expected"),
    [
        ("aaj", "future", date(2026, 11, 2)),
        ("kal", "future", date(2026, 11, 3)),
        ("kal", "past", date(2026, 11, 1)),
        ("parso", "future", date(2026, 11, 4)),
        ("parso", "past", date(2026, 10, 31)),
        ("tomorrow", "future", date(2026, 11, 3)),
        ("day after tomorrow", "future", date(2026, 11, 4)),
        ("yesterday", "past", date(2026, 11, 1)),
        ("agle somvar", "future", date(2026, 11, 9)),  # today is Monday: NEXT Monday
        ("somvar", "future", date(2026, 11, 9)),
        ("mangalvar", "future", date(2026, 11, 3)),
        ("agle shukravar", "future", date(2026, 11, 6)),
        ("pichle shukravar", "past", date(2026, 10, 30)),
        ("next friday", "future", date(2026, 11, 6)),
        ("teen din baad", "future", date(2026, 11, 5)),
        ("3 din baad", "future", date(2026, 11, 5)),
        ("2 din pehle", "past", date(2026, 10, 31)),
        ("2 hafte baad", "future", date(2026, 11, 16)),
        ("agle hafte", "future", date(2026, 11, 9)),
        ("8 November", "future", date(2026, 11, 8)),
        ("8th nov", "future", date(2026, 11, 8)),
        ("november 8", "future", date(2026, 11, 8)),
        ("aath november", "future", date(2026, 11, 8)),
        ("8/11", "future", date(2026, 11, 8)),
        ("08-11-2026", "future", date(2026, 11, 8)),
        ("8.11.26", "future", date(2026, 11, 8)),
        ("1 november", "future", date(2027, 11, 1)),  # already past this year -> next year
        ("1 november", "past", date(2026, 11, 1)),
        ("15 december", "past", date(2025, 12, 15)),
        ("2 November 2027", "future", date(2027, 11, 2)),
        ("कल", "future", date(2026, 11, 3)),
        ("परसों", "future", date(2026, 11, 4)),
        ("अगले मंगलवार", "future", date(2026, 11, 3)),
        ("15 दिसंबर", "future", date(2026, 12, 15)),
    ],
)
def test_resolve_date_phrase(text: str, direction: str, expected: date) -> None:
    assert resolve_date_phrase(text, MON, direction) == expected  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "text", ["", "   ", None, "jaldi hi", "blah", "31 february", "32 november", "13/13", "november"]
)
def test_unresolvable_phrases_return_none_not_a_guess(text: str | None) -> None:
    assert resolve_date_phrase(text, MON, "future") is None


def test_dates_across_a_year_end_and_leap_day() -> None:
    assert resolve_date_phrase("kal", date(2026, 12, 31), "future") == date(2027, 1, 1)
    assert resolve_date_phrase("29 february", date(2027, 6, 1), "future") == date(2028, 2, 29)
    # only this year and next are considered, so a leap day two years out is asked, not guessed
    assert resolve_date_phrase("29 february", date(2026, 6, 1), "future") is None


# --- media -----------------------------------------------------------------------------------


def _s() -> Settings:
    return Settings(_env_file=None)  # type: ignore[call-arg]


def _ev(**kw: Any) -> IncomingEvent:
    base: dict[str, Any] = {"update_id": 1, "chat_id": 1, "kind": "voice", "file_id": "f",
                            "duration_s": 5, "mime_type": "audio/ogg", "file_size": 1000}  # fmt: skip
    return IncomingEvent(**{**base, **kw})


@pytest.mark.parametrize(
    ("kw", "code"),
    [
        ({"duration_s": 1}, "too_short"),
        ({"duration_s": 0}, "too_short"),
        ({"duration_s": 121}, "too_long"),
        ({"mime_type": "text/html"}, "bad_type"),
        ({"file_size": 20 * 1024 * 1024 + 1}, "too_large"),
        ({"file_id": None}, "bad_type"),
    ],
)
def test_validate_voice_event_rejects(kw: dict[str, Any], code: str) -> None:
    with pytest.raises(MediaError) as exc:
        validate_voice_event(_ev(**kw), _s())
    assert exc.value.code == code


@pytest.mark.parametrize("kw", [{"duration_s": 2}, {"duration_s": 120}, {"mime_type": None},
                                {"file_size": 20 * 1024 * 1024}, {"duration_s": None}])  # fmt: skip
def test_validate_voice_event_accepts_boundaries(kw: dict[str, Any]) -> None:
    validate_voice_event(_ev(**kw), _s())


@pytest.mark.parametrize(
    ("data", "ext"),
    [
        (b"OggS\x00\x02", "ogg"),
        (b"RIFF\x00\x00\x00\x00WAVEfmt ", "wav"),
        (b"\x1a\x45\xdf\xa3\x01", "webm"),
        (b"\x00\x00\x00\x20ftypM4A ", "m4a"),
        (b"ID3\x04\x00", "mp3"),
        (b"\xff\xfb\x90\x00", "mp3"),
        (b"fLaC\x00", "flac"),
        (b"%PDF-1.7", None),
        (b"", None),
        (b"MZ\x90\x00", None),
    ],
)
def test_sniff_audio_extension(data: bytes, ext: str | None) -> None:
    assert sniff_audio_extension(data) == ext


def test_media_store_is_content_addressed_atomic_and_traversal_safe(tmp_path: Path) -> None:
    import uuid

    store, tenant = MediaStore(tmp_path), uuid.uuid4()
    sha = sha256_hex(b"abc")
    path = store.save(tenant, sha, "ogg", b"abc")
    assert path == tmp_path / str(tenant) / f"{sha}.ogg" and path.read_bytes() == b"abc"
    assert store.save(tenant, sha, "ogg", b"abc") == path  # idempotent
    assert not list(tmp_path.rglob("*.tmp"))
    for bad in ("../../etc/passwd", "a" * 63, "g" * 64, ""):
        with pytest.raises(ValueError, match="bad sha256"):
            store.path_for(tenant, bad, "ogg")
    assert store.path_for(tenant, sha, "../x").name == f"{sha}.x"  # extension sanitised


# --- transcript helpers --------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "foreign"),
    [("राजेंद्र ज्वैलर्स dedh sau", False), ("rajendra", False), ("راجندر", True),
     ("mix रा and ج", True), ("₹45,000", False), ("", False)],
)  # fmt: skip
def test_has_foreign_script(text: str, foreign: bool) -> None:
    assert has_foreign_script(text) is foreign


def test_stt_prompt_lists_names_and_vocabulary_and_caps_names() -> None:
    names = [f"Client {i}" for i in range(100)]
    prompt = build_stt_prompt(names, ["kundan"], strong=False)
    assert "Client 0" in prompt and "Client 39" in prompt and "Client 40" not in prompt
    assert prompt.endswith("kundan") and "देवनागरी" in build_stt_prompt([], [], strong=True)


# --- Groq adapter (HTTP mocked) -------------------------------------------------------------

GROQ_OK = {
    "text": " dedh sau gram ",
    "segments": [
        {"avg_logprob": -0.2, "no_speech_prob": 0.01},
        {"avg_logprob": -0.4, "no_speech_prob": 0.05},
    ],
}


def _groq(**kw: Any) -> GroqSTT:
    return GroqSTT("gsk-test-key-123", "whisper-test-model", retry_delay_s=0, **kw)


@respx.mock
async def test_groq_request_shape_and_response_parsing() -> None:
    route = respx.post(GROQ_TRANSCRIPTIONS_URL).mock(return_value=httpx.Response(200, json=GROQ_OK))
    res = await _groq().transcribe(b"OggS-bytes", filename="voice.ogg", mime_type="audio/ogg",
                                   language="hi", prompt="x" * 2000)  # fmt: skip
    req = route.calls[0].request
    body = req.content
    assert req.headers["authorization"] == "Bearer gsk-test-key-123"
    for needle in (b'name="model"', b"whisper-test-model", b'name="language"', b"\r\nhi\r\n",
                   b'name="response_format"', b"verbose_json", b'name="temperature"', b"OggS-bytes"):  # fmt: skip
        assert needle in body, needle
    prompt_part = body.split(b'name="prompt"')[1].split(b"\r\n--")[0].strip()
    assert len(prompt_part) <= 450  # Groq caps the prompt (224 tokens)
    assert res.text == "dedh sau gram" and res.model == "whisper-test-model"
    assert res.avg_logprob == pytest.approx(-0.3) and res.no_speech_prob == pytest.approx(0.05)


@respx.mock
async def test_groq_retries_once_on_429_then_succeeds() -> None:
    route = respx.post(GROQ_TRANSCRIPTIONS_URL).mock(
        side_effect=[httpx.Response(429), httpx.Response(200, json=GROQ_OK)]
    )
    res = await _groq().transcribe(
        b"x", filename="a.ogg", mime_type=None, language="hi", prompt=None
    )
    assert route.call_count == 2 and res.text == "dedh sau gram"


@respx.mock
async def test_groq_gives_up_after_one_retry_and_never_leaks_the_key() -> None:
    respx.post(GROQ_TRANSCRIPTIONS_URL).mock(
        return_value=httpx.Response(503, text="gsk-test-key-123")
    )
    with pytest.raises(STTError) as exc:
        await _groq().transcribe(b"x", filename="a.ogg", mime_type=None, language="hi", prompt=None)
    assert exc.value.status == 503 and "gsk-test-key-123" not in str(exc.value)


@respx.mock
async def test_groq_does_not_retry_auth_errors() -> None:
    route = respx.post(GROQ_TRANSCRIPTIONS_URL).mock(return_value=httpx.Response(401))
    with pytest.raises(STTError) as exc:
        await _groq().transcribe(b"x", filename="a.ogg", mime_type=None, language="hi", prompt=None)
    assert route.call_count == 1 and exc.value.status == 401


@respx.mock
async def test_groq_converts_with_ffmpeg_only_when_the_file_is_rejected() -> None:
    converted: list[bytes] = []

    async def fake_convert(data: bytes) -> bytes:
        converted.append(data)
        return b"RIFFwav"

    route = respx.post(GROQ_TRANSCRIPTIONS_URL).mock(
        side_effect=[
            httpx.Response(400, json={"error": "bad file"}),
            httpx.Response(200, json=GROQ_OK),
        ]
    )
    res = await _groq(convert=fake_convert).transcribe(
        b"OggS-opus", filename="voice.ogg", mime_type="audio/ogg", language="hi", prompt=None
    )
    assert converted == [b"OggS-opus"] and route.call_count == 2
    assert (
        b"RIFFwav" in route.calls[1].request.content
        and b"audio.wav" in route.calls[1].request.content
    )
    assert res.text == "dedh sau gram"


@respx.mock
async def test_groq_force_convert_sends_wav_first_and_a_400_is_final() -> None:
    async def fake_convert(data: bytes) -> bytes:
        return b"RIFFwav"

    route = respx.post(GROQ_TRANSCRIPTIONS_URL).mock(return_value=httpx.Response(400))
    stt = _groq(force_convert=True, convert=fake_convert)
    with pytest.raises(STTError):
        await stt.transcribe(
            b"OggS", filename="v.ogg", mime_type="audio/ogg", language="hi", prompt=None
        )
    assert route.call_count == 1 and b"RIFFwav" in route.calls[0].request.content


@respx.mock
async def test_groq_network_error_is_an_stt_error() -> None:
    respx.post(GROQ_TRANSCRIPTIONS_URL).mock(side_effect=httpx.ConnectError("no route"))
    with pytest.raises(STTError, match="ConnectError"):
        await _groq().transcribe(b"x", filename="a.ogg", mime_type=None, language="hi", prompt=None)


# --- Gemini adapter (SDK client stubbed) ---------------------------------------------------


class _Usage:
    prompt_token_count = 120
    candidates_token_count = 45


class _Resp:
    text = '{"intent": "HELP"}'
    model_version = "gemini-test-001"
    usage_metadata = _Usage()


class _Models:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.fail: Exception | None = None

    async def generate_content(self, **kw: Any) -> _Resp:
        self.calls.append(kw)
        if self.fail:
            raise self.fail
        return _Resp()

    async def get(self, **kw: Any) -> None:
        if self.fail:
            raise self.fail


class _Aio:
    def __init__(self) -> None:
        self.models = _Models()


class _Client:
    def __init__(self) -> None:
        self.aio = _Aio()


async def test_gemini_generate_json_uses_temperature_zero_and_a_json_schema() -> None:
    client = _Client()
    llm = GeminiLLM("k", "gemini-test", client=client)
    schema = {"type": "object"}
    res = await llm.generate_json(system="SYS", user="USER", schema=schema)
    call = client.aio.models.calls[0]
    cfg = call["config"]
    assert call["model"] == "gemini-test" and call["contents"] == "USER"
    assert cfg.temperature == 0 and cfg.response_mime_type == "application/json"
    assert cfg.response_json_schema == schema and cfg.system_instruction == "SYS"
    assert (res.text, res.model, res.prompt_tokens, res.output_tokens) == (
        '{"intent": "HELP"}', "gemini-test-001", 120, 45)  # fmt: skip


async def test_gemini_api_errors_become_llm_errors() -> None:
    client = _Client()
    client.aio.models.fail = genai_errors.ClientError(429, {"error": {"message": "quota"}})
    with pytest.raises(LLMError, match="429"):
        await GeminiLLM("k", "m", client=client).generate_json(system="s", user="u", schema={})


async def test_unavailable_model_fails_startup_with_a_clear_error() -> None:
    client = _Client()
    client.aio.models.fail = genai_errors.ClientError(404, {"error": {"message": "not found"}})
    with pytest.raises(LLMUnavailableError, match=r"GEMINI_MODEL 'no-such-model'.*404"):
        await GeminiLLM("k", "no-such-model", client=client).ensure_available()
    await GeminiLLM("k", "ok-model", client=_Client()).ensure_available()  # available: no error


# --- configuration -------------------------------------------------------------------------


def test_provider_keys_require_explicit_model_names() -> None:
    with pytest.raises(ValidationError, match="WHISPER_MODEL"):
        Settings(_env_file=None, groq_api_key="k")  # type: ignore[call-arg,arg-type]
    with pytest.raises(ValidationError, match="GEMINI_MODEL"):
        Settings(_env_file=None, gemini_api_key="k")  # type: ignore[call-arg,arg-type]
    ok = Settings(
        _env_file=None, groq_api_key="k", whisper_model="w", gemini_api_key="k", gemini_model="g"
    )  # type: ignore[call-arg,arg-type]
    assert (ok.whisper_model, ok.gemini_model) == ("w", "g")


def test_sandbox_is_forced_off_in_production() -> None:
    assert Settings(_env_file=None, env="prod", sandbox_enabled=True).sandbox_enabled is False  # type: ignore[call-arg]
    assert Settings(_env_file=None, env="dev").sandbox_enabled is True  # type: ignore[call-arg]
