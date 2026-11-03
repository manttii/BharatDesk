"""Speech-to-text boundary: an interface, Groq Whisper over its REST API, and a Fake.

Verified against Groq's docs (console.groq.com/docs/speech-to-text): POST
/openai/v1/audio/transcriptions, multipart `file`, `model`, `language` (ISO-639-1),
`prompt` (<= 224 tokens; spellings/context, not instructions), `response_format`
(`verbose_json` adds per-segment avg_logprob / no_speech_prob), `temperature`.
ogg is a listed format but the docs say to test OPUS-encoded files, so a rejected file is
retried once after an ffmpeg conversion to 16 kHz mono WAV (or always, if configured).
"""

from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

import httpx
import structlog

log = structlog.get_logger()

GROQ_TRANSCRIPTIONS_URL = "https://api.groq.com/openai/v1/audio/transcriptions"
MAX_PROMPT_CHARS = 450  # Groq caps the prompt at 224 tokens; Devanagari is token-heavy


class STTError(Exception):
    def __init__(self, message: str, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


class _FormatRejected(STTError):
    pass


@dataclass(frozen=True)
class STTResult:
    text: str
    model: str
    latency_ms: int
    avg_logprob: float | None = None  # mean over segments; closer to 0 = more confident
    no_speech_prob: float | None = None  # max over segments


class STTProvider(ABC):
    model: str

    @abstractmethod
    async def transcribe(
        self,
        audio: bytes,
        *,
        filename: str,
        mime_type: str | None,
        language: str,
        prompt: str | None,
    ) -> STTResult: ...


async def ffmpeg_to_wav(data: bytes) -> bytes:
    """16 kHz mono WAV via ffmpeg (argument list, no shell). Raises STTError if unavailable."""
    try:
        proc = await asyncio.create_subprocess_exec(
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-i", "pipe:0",
            "-ar", "16000", "-ac", "1", "-f", "wav", "pipe:1",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )  # fmt: skip
    except FileNotFoundError as exc:
        raise STTError("ffmpeg is not installed; cannot convert audio") from exc
    out, _ = await proc.communicate(data)
    if proc.returncode != 0 or not out:
        raise STTError("ffmpeg could not convert the audio")
    return out


class GroqSTT(STTProvider):
    def __init__(
        self,
        api_key: str,
        model: str,
        *,
        client: httpx.AsyncClient | None = None,
        force_convert: bool = False,
        convert: Callable[[bytes], Awaitable[bytes]] = ffmpeg_to_wav,
        retry_delay_s: float = 1.0,
    ) -> None:
        self.model = model
        self._key = api_key
        self._client = client or httpx.AsyncClient(timeout=httpx.Timeout(60.0))
        self._force_convert = force_convert
        self._convert = convert
        self._retry_delay = retry_delay_s

    async def transcribe(
        self,
        audio: bytes,
        *,
        filename: str,
        mime_type: str | None,
        language: str,
        prompt: str | None,
    ) -> STTResult:
        started = time.monotonic()
        payload, name, mime = audio, filename, mime_type or "application/octet-stream"
        converted = False
        if self._force_convert:
            payload, name, mime, converted = (
                await self._convert(audio),
                "audio.wav",
                "audio/wav",
                True,
            )
        try:
            body = await self._post(payload, name, mime, language, prompt)
        except _FormatRejected:
            if converted:
                raise
            log.info("stt_format_rejected_retrying_with_wav")
            payload = await self._convert(audio)
            body = await self._post(payload, "audio.wav", "audio/wav", language, prompt)
        return self._parse(body, started)

    async def _post(
        self, payload: bytes, name: str, mime: str, language: str, prompt: str | None
    ) -> dict[str, Any]:
        data = {
            "model": self.model,
            "language": language,
            "response_format": "verbose_json",
            "temperature": "0",
        }
        if prompt:
            data["prompt"] = prompt[:MAX_PROMPT_CHARS]
        last: STTError | None = None
        for attempt in range(2):
            try:
                resp = await self._client.post(
                    GROQ_TRANSCRIPTIONS_URL,
                    headers={"Authorization": f"Bearer {self._key}"},
                    data=data,
                    files={"file": (name, payload, mime)},
                )
            except httpx.HTTPError as exc:
                last = STTError(f"groq request failed: {type(exc).__name__}")
            else:
                if resp.status_code == 200:
                    result: dict[str, Any] = resp.json()
                    return result
                if resp.status_code == 400:
                    raise _FormatRejected("groq rejected the audio", status=400)
                last = STTError(f"groq returned HTTP {resp.status_code}", status=resp.status_code)
                if resp.status_code not in (429, 500, 502, 503, 504):
                    raise last
            if attempt == 0:
                await asyncio.sleep(self._retry_delay)
        assert last is not None
        raise last

    def _parse(self, body: dict[str, Any], started: float) -> STTResult:
        segments = body.get("segments") or []
        logprobs = [s["avg_logprob"] for s in segments if s.get("avg_logprob") is not None]
        nospeech = [s["no_speech_prob"] for s in segments if s.get("no_speech_prob") is not None]
        return STTResult(
            text=str(body.get("text", "")).strip(),
            model=self.model,
            latency_ms=int((time.monotonic() - started) * 1000),
            avg_logprob=sum(logprobs) / len(logprobs) if logprobs else None,
            no_speech_prob=max(nospeech) if nospeech else None,
        )


class FakeSTT(STTProvider):
    """Scripted results for tests; records every call (including the prompt it was given)."""

    def __init__(self, results: list[STTResult | Exception] | None = None) -> None:
        self.model = "fake-whisper"
        self.results: list[STTResult | Exception] = list(results or [])
        self.calls: list[dict[str, Any]] = []

    def say(self, text: str, **kw: Any) -> None:
        self.results.append(STTResult(text=text, model=self.model, latency_ms=5, **kw))

    async def transcribe(
        self,
        audio: bytes,
        *,
        filename: str,
        mime_type: str | None,
        language: str,
        prompt: str | None,
    ) -> STTResult:
        self.calls.append({"audio": audio, "language": language, "prompt": prompt})
        if not self.results:
            raise STTError("FakeSTT has no scripted result")
        nxt = self.results.pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        return nxt
