"""Transcription with a vocabulary hint and a script check.

Known failure mode: Whisper sometimes renders Hindi audio in Urdu (Arabic) script. We retry
once with a stronger Devanagari hint; if the script is still wrong the transcript is kept
but flagged, which forces a read-back instead of being trusted.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from app.config import Settings
from app.providers.stt import STTProvider, STTResult

MAX_NAMES_IN_PROMPT = 40

_HINT_DEVANAGARI = (
    "यह एक सेल्स विज़िट का हिंदी वॉइस नोट है। हिंदी देवनागरी और अंग्रेज़ी शब्दों में लिखें, अंक और रकम सही लिखें। "
)
_HINT_PLAIN = "Hindi/Hinglish sales visit note. Names and terms: "


def has_foreign_script(text: str) -> bool:
    """True if the text contains Arabic/Urdu-range characters (never expected here)."""
    return any(
        0x0600 <= ord(c) <= 0x06FF
        or 0x0750 <= ord(c) <= 0x077F
        or 0xFB50 <= ord(c) <= 0xFDFF
        or 0xFE70 <= ord(c) <= 0xFEFF
        for c in text
    )


def build_stt_prompt(
    client_names: Sequence[str], vocabulary: Sequence[str], *, strong: bool
) -> str:
    terms = ", ".join([*list(client_names)[:MAX_NAMES_IN_PROMPT], *vocabulary])
    return (_HINT_DEVANAGARI if strong else _HINT_PLAIN) + terms


@dataclass(frozen=True)
class Transcript:
    text: str
    model: str
    latency_ms: int
    attempts: int
    reasons: tuple[
        str, ...
    ]  # why it must be read back: foreign_script | low_logprob | no_speech | empty
    avg_logprob: float | None = None
    no_speech_prob: float | None = None

    @property
    def low_confidence(self) -> bool:
        return bool(self.reasons)


def _reasons(r: STTResult, settings: Settings) -> list[str]:
    out: list[str] = []
    if not r.text.strip():
        out.append("empty")
    if has_foreign_script(r.text):
        out.append("foreign_script")
    if r.avg_logprob is not None and r.avg_logprob < settings.stt_min_avg_logprob:
        out.append("low_logprob")
    if r.no_speech_prob is not None and r.no_speech_prob > settings.stt_max_no_speech_prob:
        out.append("no_speech")
    return out


async def transcribe_note(
    stt: STTProvider,
    audio: bytes,
    *,
    filename: str,
    mime_type: str | None,
    client_names: Sequence[str],
    vocabulary: Sequence[str],
    settings: Settings,
) -> Transcript:
    """STT errors propagate (the caller tells the rep); script problems become reasons."""
    first = await stt.transcribe(
        audio,
        filename=filename,
        mime_type=mime_type,
        language="hi",
        prompt=build_stt_prompt(client_names, vocabulary, strong=False),
    )
    result, attempts, latency = first, 1, first.latency_ms
    if has_foreign_script(first.text):
        second = await stt.transcribe(
            audio,
            filename=filename,
            mime_type=mime_type,
            language="hi",
            prompt=build_stt_prompt(client_names, vocabulary, strong=True),
        )
        attempts, latency = 2, latency + second.latency_ms
        result = second  # the retry is what we keep; its reasons decide whether to trust it
    return Transcript(
        text=result.text.strip(),
        model=result.model,
        latency_ms=latency,
        attempts=attempts,
        reasons=tuple(_reasons(result, settings)),
        avg_logprob=result.avg_logprob,
        no_speech_prob=result.no_speech_prob,
    )
