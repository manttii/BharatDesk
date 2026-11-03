"""Messaging boundary: one interface, a Telegram adapter, and a Fake for tests/sandbox."""

from __future__ import annotations

from abc import ABC, abstractmethod

import structlog
from aiogram import Bot
from aiogram.exceptions import TelegramAPIError
from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup, Update

from app.schemas.events import IncomingEvent, OutMessage

log = structlog.get_logger()

TELEGRAM_MAX_TEXT = 4096


class DownloadError(Exception):
    """A media file could not be fetched (missing, too large, or the transport failed)."""


class MessagingProvider(ABC):
    username: str

    @abstractmethod
    async def send(self, msg: OutMessage) -> None: ...

    @abstractmethod
    async def download(self, file_id: str, max_bytes: int) -> bytes:
        """Fetch a file by id. Raises DownloadError; never returns more than max_bytes."""

    @abstractmethod
    async def answer_callback(self, callback_id: str, text: str | None = None) -> None: ...

    @abstractmethod
    async def send_typing(self, chat_id: int) -> None: ...


class FakeProvider(MessagingProvider):
    """Records everything; used by tests and (later) the sandbox. No network."""

    def __init__(self, username: str = "FakeBharatDeskBot") -> None:
        self.username = username
        self.sent: list[OutMessage] = []
        self.callback_answers: list[tuple[str, str | None]] = []
        self.typing: list[int] = []
        self.files: dict[str, bytes] = {}  # file_id -> content, for tests and the sandbox

    async def send(self, msg: OutMessage) -> None:
        self.sent.append(msg)

    async def download(self, file_id: str, max_bytes: int) -> bytes:
        data = self.files.get(file_id)
        if data is None:
            raise DownloadError(f"unknown file_id {file_id!r}")
        if len(data) > max_bytes:
            raise DownloadError("file too large")
        return data

    async def answer_callback(self, callback_id: str, text: str | None = None) -> None:
        self.callback_answers.append((callback_id, text))

    async def send_typing(self, chat_id: int) -> None:
        self.typing.append(chat_id)


class TelegramProvider(MessagingProvider):
    """Adapter over aiogram's Bot. Plain text only: no parse_mode, so user-supplied
    names can never inject markup."""

    def __init__(self, bot: Bot, username: str) -> None:
        self._bot = bot
        self.username = username

    async def send(self, msg: OutMessage) -> None:
        markup: InlineKeyboardMarkup | None = None
        if msg.buttons:
            markup = InlineKeyboardMarkup(
                inline_keyboard=[
                    [InlineKeyboardButton(text=b.text, callback_data=b.data) for b in row]
                    for row in msg.buttons
                ]
            )
        await self._bot.send_message(
            chat_id=msg.chat_id, text=msg.text[:TELEGRAM_MAX_TEXT], reply_markup=markup
        )

    async def download(self, file_id: str, max_bytes: int) -> bytes:
        try:
            info = await self._bot.get_file(file_id)
            if info.file_size is not None and info.file_size > max_bytes:
                raise DownloadError("file too large")
            if not info.file_path:
                raise DownloadError("telegram returned no file_path")
            buf = await self._bot.download_file(info.file_path)
        except TelegramAPIError as exc:
            raise DownloadError(type(exc).__name__) from exc
        if buf is None:
            raise DownloadError("empty download")
        data = buf.read()
        if len(data) > max_bytes:
            raise DownloadError("file too large")
        return data

    async def answer_callback(self, callback_id: str, text: str | None = None) -> None:
        # Stale callback ids (>~15 min old, or already answered) are harmless: log, move on.
        try:
            await self._bot.answer_callback_query(callback_query_id=callback_id, text=text)
        except TelegramAPIError as exc:
            log.warning("answer_callback_failed", error=type(exc).__name__)

    async def send_typing(self, chat_id: int) -> None:
        try:
            await self._bot.send_chat_action(chat_id=chat_id, action="typing")
        except TelegramAPIError as exc:
            log.warning("send_typing_failed", error=type(exc).__name__)


def parse_update(update: Update) -> IncomingEvent | None:
    """Normalise a Telegram Update. Returns None for anything we deliberately ignore
    (group chats, edited messages, channel posts, ...)."""
    if update.callback_query is not None:
        cq = update.callback_query
        chat = cq.message.chat if cq.message is not None else None
        # Private chats only; the tapper's id equals the chat id there.
        if chat is not None and chat.type != "private":
            return None
        return IncomingEvent(
            update_id=update.update_id,
            chat_id=chat.id if chat is not None else cq.from_user.id,
            kind="callback",
            callback_id=cq.id,
            callback_data=cq.data,
        )
    msg = update.message
    if msg is None or msg.chat.type != "private":
        return None
    if msg.voice is not None:
        v = msg.voice
        return IncomingEvent(
            update_id=update.update_id,
            chat_id=msg.chat.id,
            kind="voice",
            file_id=v.file_id,
            duration_s=v.duration,
            mime_type=v.mime_type,
            file_size=v.file_size,
        )
    if msg.photo:
        # Telegram sends several sizes of the same photo; take the largest.
        best = max(msg.photo, key=lambda p: (p.width or 0) * (p.height or 0))
        return IncomingEvent(
            update_id=update.update_id,
            chat_id=msg.chat.id,
            kind="photo",
            file_id=best.file_id,
            file_size=best.file_size,
            width=best.width,
            height=best.height,
            caption=msg.caption,
        )
    doc = msg.document
    if doc is not None and (doc.mime_type or "").startswith("image/"):
        # A slip sent "as a file" (uncompressed). Handled like a photo; dimensions are unknown
        # here, so the legibility gate is left to the vision model.
        return IncomingEvent(
            update_id=update.update_id,
            chat_id=msg.chat.id,
            kind="photo",
            file_id=doc.file_id,
            file_size=doc.file_size,
            mime_type=doc.mime_type,
            caption=msg.caption,
        )
    if doc is not None and (
        doc.mime_type == "application/pdf" or (doc.file_name or "").lower().endswith(".pdf")
    ):
        return IncomingEvent(
            update_id=update.update_id,
            chat_id=msg.chat.id,
            kind="document",
            file_id=doc.file_id,
            file_size=doc.file_size,
            mime_type=doc.mime_type,
            file_name=doc.file_name,
            caption=msg.caption,
        )
    return IncomingEvent(
        update_id=update.update_id,
        chat_id=msg.chat.id,
        kind="text" if msg.text is not None else "other",
        text=msg.text,
    )
