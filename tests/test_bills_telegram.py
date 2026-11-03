"""Bills over Telegram: upload, one-tap confirmation, per-bill decisions, boss-only authority."""

from __future__ import annotations

from typing import Any

from sqlalchemy import select

from app.models import Bill
from app.schemas.events import IncomingEvent, OutMessage
from tests.conftest import BOSS_CHAT, JPEG, REP_CHAT, Env, button_data
from tests.test_bills import bill_json, text_pdf


async def send_file(
    pipe: Env,
    chat: int,
    replies: list[str],
    *,
    data: bytes | None = None,
    pages: int = 2,
    size: int | None = None,
    kind: str = "document",
    caption: str | None = None,
    name: str = "bills.pdf",
) -> list[OutMessage]:
    assert pipe.llm is not None
    blob = data if data is not None else text_pdf(pages)
    uid = pipe._update_ids()
    file_id = f"doc-{uid}"
    pipe.fake.files[file_id] = blob
    pipe.llm.replies.extend(replies)
    ev = IncomingEvent(
        uid,
        chat,
        kind,  # type: ignore[arg-type]
        file_id=file_id,
        file_size=size if size is not None else len(blob),
        mime_type="application/pdf" if kind == "document" else None,
        file_name=name,
        caption=caption,
    )
    return await pipe.send_event(ev)


async def org(pipe: Env) -> None:
    await pipe.org()
    await pipe.connect_sheet()


async def bills(pipe: Env) -> dict[str, str]:
    async with pipe.sf() as s:
        return {
            b.bill_no or f"p{b.page_no}": b.status
            for b in (await s.execute(select(Bill))).scalars()
        }


def texts(msgs: list[OutMessage]) -> str:
    return "\n".join(m.text for m in msgs)


FLAGGED = bill_json(bill_no="F1", total_paise=7)


async def test_the_boss_uploads_a_pdf_sees_a_summary_and_confirms_with_one_tap(pipe: Env) -> None:
    await org(pipe)
    out = await send_file(pipe, BOSS_CHAT, [bill_json(bill_no="A1"), FLAGGED])
    assert "Padh raha hoon" in out[0].text  # immediate acknowledgement
    summary = next(m for m in out if "page padhe" in m.text)
    assert "2 page padhe: 1 sahi, 1 dekhne wale, 0 padh nahi paaye" in summary.text
    confirm = button_data(summary, "Haan, 1 confirm karo")
    assert len(confirm.encode()) <= 64
    flagged = next(m for m in out if "bill F1" in m.text)
    assert "total nahi mil raha" in flagged.text and "₹0.07" in flagged.text
    assert await bills(pipe) == {"A1": "ready", "F1": "needs_review"}

    done = await pipe.tap(BOSS_CHAT, confirm)
    assert "1 bills save ho gaye" in texts(done)
    assert await bills(pipe) == {"A1": "confirmed", "F1": "needs_review"}  # flagged untouched

    again = await pipe.tap(BOSS_CHAT, confirm)  # a second tap changes nothing
    assert "save ho gaye" not in texts(again)
    assert pipe.fake.callback_answers[-1][1] == "Ye pehle hi ho chuka hai."


async def test_flagged_bills_are_accepted_or_rejected_individually(pipe: Env) -> None:
    await org(pipe)
    out = await send_file(
        pipe,
        BOSS_CHAT,
        [bill_json(bill_no="X1", total_paise=1), bill_json(bill_no="X2", total_paise=2)],
    )
    items = [m for m in out if "Dhyan:" in m.text]
    assert len(items) == 2
    await pipe.tap(BOSS_CHAT, button_data(items[0], "Aise hi rakho"))
    await pipe.tap(BOSS_CHAT, button_data(items[1], "Hatao"))
    assert await bills(pipe) == {"X1": "confirmed", "X2": "rejected"}


async def test_a_rep_uploads_but_only_the_boss_gets_the_buttons_and_the_authority(
    pipe: Env,
) -> None:
    await org(pipe)
    out = await send_file(
        pipe, REP_CHAT, [bill_json(bill_no="R1"), bill_json(bill_no="R2", party_name="Y")]
    )
    rep_msgs = [m for m in out if m.chat_id == REP_CHAT]
    boss_msgs = [m for m in out if m.chat_id == BOSS_CHAT]
    assert any("Boss ko bhej diya" in m.text for m in rep_msgs)
    assert all(m.buttons is None for m in rep_msgs)
    confirm = button_data(boss_msgs[0], "Haan, 2 confirm karo")

    denied = await pipe.tap(REP_CHAT, confirm)  # the rep cannot confirm even with the button data
    assert "save ho gaye" not in texts(denied)
    assert await bills(pipe) == {"R1": "ready", "R2": "ready"}
    assert "sirf boss" in (pipe.fake.callback_answers[-1][1] or "")
    await pipe.tap(BOSS_CHAT, confirm)
    assert set((await bills(pipe)).values()) == {"confirmed"}


async def test_an_unreadable_page_is_reported_and_can_be_dismissed(pipe: Env) -> None:
    await org(pipe)
    out = await send_file(pipe, BOSS_CHAT, [bill_json(legible=False)], pages=1)
    assert "0 sahi, 0 dekhne wale, 1 padh nahi paaye" in texts(out)
    item = next(m for m in out if "padh nahi paya" in m.text)
    await pipe.tap(BOSS_CHAT, button_data(item, "Hatao"))
    assert list((await bills(pipe)).values()) == ["rejected"]


async def test_bad_oversize_and_unconfigured_uploads_are_refused_politely(pipe: Env) -> None:
    await org(pipe)
    junk = await send_file(pipe, BOSS_CHAT, [], data=b"MZ not a document")
    assert "Ye file padh nahi sakta" in texts(junk) and "only PDF" in texts(junk)
    big = await send_file(pipe, BOSS_CHAT, [], size=pipe.settings.max_pdf_bytes + 1)
    assert "bahut badi" in texts(big)
    assert await bills(pipe) == {}


async def test_without_a_model_the_bot_says_so(env: Env) -> None:
    await env.org()
    ev = IncomingEvent(env._update_ids(), BOSS_CHAT, "document", file_id="x", file_size=10)
    out = await env.send_event(ev)
    assert "chalu nahi hai" in texts(out)


async def test_a_photo_captioned_bill_is_read_as_a_bill_not_a_visit_slip(pipe: Env) -> None:
    await org(pipe)
    assert pipe.llm is not None
    out = await send_file(
        pipe, BOSS_CHAT, [bill_json(bill_no="PH1")], data=JPEG, kind="photo", caption="Bill 12 Nov"
    )
    assert "1 page padhe" in texts(out)
    assert "business document" in pipe.llm.calls[0]["system"]
    assert pipe.llm.calls[0]["image"] is not None and await bills(pipe) == {"PH1": "ready"}


async def test_a_plain_photo_is_still_a_visit_slip(pipe: Env) -> None:
    await org(pipe)
    assert pipe.llm is not None
    await send_file(pipe, BOSS_CHAT, [], data=JPEG, kind="photo", caption="/sheet https://x")
    assert await bills(pipe) == {}  # no bill was created: the slip route handled it


async def test_bills_command_lists_what_is_pending_for_the_boss_only(pipe: Env) -> None:
    await org(pipe)
    await send_file(pipe, BOSS_CHAT, [bill_json(bill_no="L1"), FLAGGED])
    out = await pipe.say(BOSS_CHAT, "/bills")
    assert any("1 sahi bills confirm" in m.text for m in out)
    assert any("bill F1" in m.text for m in out)
    deny = await pipe.say(REP_CHAT, "/bills")
    assert "sirf boss" in texts(deny)
    for m in out:
        for row in m.buttons or ():
            for b in row:
                assert len(b.data.encode()) <= 64


async def test_nothing_pending_is_said_plainly(pipe: Env) -> None:
    await org(pipe)
    assert "Koi bill pending nahi" in texts(await pipe.say(BOSS_CHAT, "/bills"))


async def test_confirmed_bills_from_telegram_reach_the_sheet(pipe: Env) -> None:
    from datetime import UTC, datetime

    from app.services import bill_sync

    assert pipe.sheets is not None
    await org(pipe)
    out = await send_file(
        pipe, BOSS_CHAT, [bill_json(bill_no="S1"), bill_json(bill_no="S2", party_name="Z")]
    )
    summary: Any = next(m for m in out if "page padhe" in m.text)
    await pipe.tap(BOSS_CHAT, button_data(summary, "Haan, 2 confirm karo"))
    report = await bill_sync.sync_bills(pipe.sf, pipe.sheets, now=datetime.now(UTC))
    assert report.bills_sent == 2
