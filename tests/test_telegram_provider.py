"""Telegram adapter: update normalisation, keyboard building, 64-byte limit."""

from __future__ import annotations

from typing import Any, cast

import pytest
from aiogram import Bot
from aiogram.exceptions import TelegramBadRequest
from aiogram.methods import AnswerCallbackQuery
from aiogram.types import Update

from app.providers.messaging import TelegramProvider, parse_update
from app.schemas.events import Button, OutMessage


def _msg_update(**message: Any) -> Update:
    base = {
        "message_id": 1,
        "date": 1_700_000_000,
        "chat": {"id": 7, "type": "private"},
        "from": {"id": 7, "is_bot": False, "first_name": "T"},
    }
    return Update.model_validate({"update_id": 10, "message": {**base, **message}})


def test_text_message_is_normalised() -> None:
    ev = parse_update(_msg_update(text="namaste"))
    assert ev is not None and (ev.kind, ev.text, ev.chat_id, ev.update_id) == (
        "text",
        "namaste",
        7,
        10,
    )


def test_voice_message_is_normalised_with_its_metadata() -> None:
    ev = parse_update(
        _msg_update(
            voice={
                "file_id": "f",
                "file_unique_id": "u",
                "duration": 3,
                "mime_type": "audio/ogg",
                "file_size": 4096,
            }
        )
    )
    assert ev is not None and ev.kind == "voice" and ev.text is None
    assert (ev.file_id, ev.duration_s, ev.mime_type, ev.file_size) == ("f", 3, "audio/ogg", 4096)


def test_photo_message_picks_the_largest_size() -> None:
    photo = [
        {"file_id": "small", "file_unique_id": "a", "width": 90, "height": 90, "file_size": 1000},
        {"file_id": "big", "file_unique_id": "b", "width": 1280, "height": 960, "file_size": 90000},
        {"file_id": "mid", "file_unique_id": "c", "width": 320, "height": 240, "file_size": 9000},
    ]
    ev = parse_update(_msg_update(photo=photo))
    assert ev is not None and ev.kind == "photo"
    assert (ev.file_id, ev.width, ev.height, ev.file_size) == ("big", 1280, 960, 90000)


def test_other_attachments_are_kind_other() -> None:
    doc = {"file_id": "d", "file_unique_id": "du", "mime_type": "application/zip"}
    ev = parse_update(_msg_update(document=doc))
    assert ev is not None and ev.kind == "other"
    ev = parse_update(_msg_update(document={"file_id": "d", "file_unique_id": "du"}))
    assert ev is not None and ev.kind == "other"  # no mime type: not assumed to be anything


def test_a_pdf_is_a_document_by_mime_type_or_by_name() -> None:
    doc = {"file_id": "d", "file_unique_id": "du", "mime_type": "application/pdf", "file_size": 9}
    ev = parse_update(_msg_update(document={**doc, "file_name": "bills.pdf"}, caption="march"))
    assert ev is not None and ev.kind == "document"
    assert (ev.file_id, ev.file_name, ev.file_size, ev.caption) == ("d", "bills.pdf", 9, "march")
    assert ev.text is None  # a caption is never command text
    named = {"file_id": "d", "file_unique_id": "du", "file_name": "SCAN.PDF"}
    ev = parse_update(_msg_update(document=named))
    assert ev is not None and ev.kind == "document"


def test_a_photo_caption_is_carried_but_never_becomes_text() -> None:
    photo = [{"file_id": "p", "file_unique_id": "a", "width": 800, "height": 600}]
    ev = parse_update(_msg_update(photo=photo, caption="/sheet https://evil"))
    assert ev is not None and ev.kind == "photo"
    assert ev.caption == "/sheet https://evil" and ev.text is None


def test_an_image_sent_as_a_file_is_a_photo() -> None:
    doc = {"file_id": "img", "file_unique_id": "iu", "mime_type": "image/png", "file_size": 123_456}
    ev = parse_update(_msg_update(document=doc))
    assert ev is not None and ev.kind == "photo"
    assert (ev.file_id, ev.file_size, ev.mime_type, ev.width) == ("img", 123_456, "image/png", None)


def test_group_messages_are_ignored() -> None:
    upd = _msg_update(text="hi", chat={"id": -5, "type": "supergroup", "title": "g"})
    assert parse_update(upd) is None


def test_callback_query_is_normalised() -> None:
    upd = Update.model_validate(
        {
            "update_id": 11,
            "callback_query": {
                "id": "cbq-1",
                "chat_instance": "ci",
                "from": {"id": 7, "is_bot": False, "first_name": "T"},
                "data": "co:y",
                "message": {
                    "message_id": 2,
                    "date": 1_700_000_000,
                    "chat": {"id": 7, "type": "private"},
                },
            },
        }
    )
    ev = parse_update(upd)
    assert ev is not None
    assert (ev.kind, ev.callback_id, ev.callback_data, ev.chat_id) == (
        "callback",
        "cbq-1",
        "co:y",
        7,
    )


def test_edited_message_is_ignored() -> None:
    upd = Update.model_validate(
        {
            "update_id": 12,
            "edited_message": {
                "message_id": 1,
                "date": 1_700_000_000,
                "edit_date": 1_700_000_001,
                "chat": {"id": 7, "type": "private"},
                "text": "x",
            },
        }
    )
    assert parse_update(upd) is None


def test_button_rejects_oversize_callback_data() -> None:
    Button("ok", "x" * 64)
    with pytest.raises(ValueError, match="64 bytes"):
        Button("bad", "x" * 65)
    with pytest.raises(ValueError, match="64 bytes"):
        Button("bad", "ह" * 22)  # 22 Devanagari chars = 66 bytes: the limit is bytes, not chars


class _StubBot:
    def __init__(self) -> None:
        self.sent: list[dict[str, Any]] = []
        self.fail_callback = False

    async def send_message(self, **kw: Any) -> None:
        self.sent.append(kw)

    async def answer_callback_query(self, **kw: Any) -> None:
        if self.fail_callback:
            raise TelegramBadRequest(
                method=AnswerCallbackQuery(callback_query_id="x"), message="query is too old"
            )

    async def send_chat_action(self, **kw: Any) -> None:
        self.sent.append({"action": kw["action"]})


async def test_send_builds_inline_keyboard_without_parse_mode() -> None:
    bot = _StubBot()
    provider = TelegramProvider(cast(Bot, bot), "TestBot")
    await provider.send(
        OutMessage(7, "Sahi hai?", ((Button("Haan", "y:1"), Button("Nahi", "n:1")),))
    )
    kw = bot.sent[0]
    assert kw["chat_id"] == 7 and "parse_mode" not in kw
    row = kw["reply_markup"].inline_keyboard[0]
    assert [(b.text, b.callback_data) for b in row] == [("Haan", "y:1"), ("Nahi", "n:1")]


async def test_send_truncates_to_telegram_limit() -> None:
    bot = _StubBot()
    await TelegramProvider(cast(Bot, bot), "TestBot").send(OutMessage(7, "a" * 5000))
    assert len(bot.sent[0]["text"]) == 4096


async def test_stale_callback_answer_does_not_raise() -> None:
    bot = _StubBot()
    bot.fail_callback = True
    await TelegramProvider(cast(Bot, bot), "TestBot").answer_callback("cbq-old")


async def test_typing_action() -> None:
    bot = _StubBot()
    await TelegramProvider(cast(Bot, bot), "TestBot").send_typing(7)
    assert bot.sent == [{"action": "typing"}]
