"""Transport-neutral shapes that cross the messaging boundary."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

MAX_CALLBACK_DATA_BYTES = 64  # Telegram's hard limit for inline-button callback_data


@dataclass(frozen=True)
class Button:
    text: str
    data: str

    def __post_init__(self) -> None:
        if len(self.data.encode("utf-8")) > MAX_CALLBACK_DATA_BYTES:
            raise ValueError("callback data exceeds 64 bytes; send a short id, not a payload")


@dataclass(frozen=True)
class OutMessage:
    chat_id: int
    text: str
    buttons: tuple[tuple[Button, ...], ...] | None = None


@dataclass(frozen=True)
class IncomingEvent:
    update_id: int
    chat_id: int
    kind: Literal["text", "callback", "voice", "photo", "document", "other"]
    text: str | None = None
    callback_id: str | None = None
    callback_data: str | None = None
    # voice notes (all as reported by Telegram, i.e. sender-supplied: validate, don't trust)
    file_id: str | None = None
    duration_s: int | None = None
    mime_type: str | None = None
    file_size: int | None = None
    width: int | None = None  # photos
    height: int | None = None
    file_name: str | None = None  # documents, sender-supplied
    caption: str | None = None  # photo/document caption; NEVER parsed as a command
