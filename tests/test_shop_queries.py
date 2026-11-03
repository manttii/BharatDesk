"""/stock, /today and plain-language shop questions: confirmed bills only, exact numbers."""

from __future__ import annotations

import json
from datetime import date
from zoneinfo import ZoneInfo

import pytest

from app.bot.shop_ui import ShopQuery, classify_query
from app.db import TenantRepository
from app.services import bill_batch
from tests.conftest import BOSS_CHAT, REP_CHAT, Env
from tests.test_bills import NOW, bill_json, save


def texts(msgs: list) -> str:  # type: ignore[type-arg]
    return "\n".join(m.text for m in msgs)


# --- what counts as a question ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("stock kitna hai", ShopQuery("stock")),
        ("Stock kitna hai?", ShopQuery("stock")),
        ("basmati rice ka stock kitna hai?", ShopQuery("stock", "basmati rice")),
        ("sugar kitna bacha hai", ShopQuery("stock", "sugar")),
        ("stock dikhao", ShopQuery("stock")),
        ("aaj kya bika?", ShopQuery("today")),
        ("aaj ki sale kitni hui", ShopQuery("today")),
        ("today sales?", ShopQuery("today")),
        ("आज कितना बिका?", None),  # Devanagari is not guessed at
    ],
)
def test_clear_short_questions_are_recognised(text: str, expected: ShopQuery | None) -> None:
    assert classify_query(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "Rajendra ko stock dikhaya",  # a visit note: no question cue
        "stock",  # no cue: use /stock
        "aaj delivery hai",  # no sale word
        "Rajendra Jewellers gaya tha stock kitna hai",  # too long to be a bare question
        "stock kitna hai 50 gram",  # digits mean a note
        "",
    ],
)
def test_visit_notes_and_odd_text_are_not_mistaken_for_questions(text: str) -> None:
    assert classify_query(text) is None


# --- answers ------------------------------------------------------------------------------------


def today_text(pipe: Env) -> str:
    d = pipe.clock().astimezone(ZoneInfo("Asia/Kolkata")).date()
    return f"{d.day} {d.strftime('%b')} {d.year}"


def sale(
    day: str, qty: int = 8000, rate: int = 9000, bill_no: str = "S1", name: str = "Basmati rice"
) -> str:
    amount = qty * rate // 1000
    return bill_json(
        doc_type="sales",
        bill_no=bill_no,
        party_name="Anil Kirana",
        date_text=day,
        lines=[{"name": name, "unit": "kg", "quantity_milli": qty, "rate_paise": rate,
                "amount_paise": amount, "confidence": 0.97}],
        subtotal_paise=amount,
        tax_paise=0,
        total_paise=amount,
    )  # fmt: skip


async def shop(pipe: Env, bills: list[str]) -> None:
    org = await pipe.org()
    s = await save(pipe, org.tenant_id, org.boss_id, bills)
    async with pipe.sf() as session, session.begin():
        await bill_batch.confirm_ready(
            TenantRepository(session, org.tenant_id), s.batch_id, org.boss_id, NOW
        )


async def test_stock_lists_bought_minus_sold_for_the_whole_team(pipe: Env) -> None:
    day = today_text(pipe)
    await shop(pipe, [bill_json(date_text=day), sale(day)])
    for chat in (BOSS_CHAT, REP_CHAT):
        out = texts(await pipe.say(chat, "/stock"))
        assert "Basmati rice: 17 kg" in out and "Sugar: 10.5 kg" in out
        assert "confirmed bills" in out


async def test_stock_can_be_narrowed_to_an_item_and_reports_an_unknown_one(pipe: Env) -> None:
    day = today_text(pipe)
    await shop(pipe, [bill_json(date_text=day)])
    only = texts(await pipe.say(BOSS_CHAT, "/stock rice"))
    assert "Basmati rice: 25 kg" in only and "Sugar" not in only
    assert "koi item nahi mila" in texts(await pipe.say(BOSS_CHAT, "/stock laptop"))


async def test_overselling_shows_minus_with_a_hint(pipe: Env) -> None:
    day = today_text(pipe)
    await shop(pipe, [bill_json(date_text=day), sale(day, qty=30000)])
    out = texts(await pipe.say(BOSS_CHAT, "/stock rice"))
    assert "Basmati rice: -5 kg" in out and "minus" in out


async def test_unconfirmed_bills_never_count_and_are_mentioned(pipe: Env) -> None:
    org = await pipe.org()
    await save(pipe, org.tenant_id, org.boss_id, [bill_json(bill_no="W1")])  # ready, not confirmed
    out = texts(await pipe.say(BOSS_CHAT, "/stock"))
    assert "nahi bata sakta" in out  # nothing confirmed yet: no stock claimed
    day = today_text(pipe)
    await save(pipe, org.tenant_id, org.boss_id, [bill_json(bill_no="W2", date_text=day)])
    assert "nahi bata sakta" in texts(await pipe.say(BOSS_CHAT, "/stock"))


async def test_a_flagged_bill_is_noted_under_confirmed_stock(pipe: Env) -> None:
    day = today_text(pipe)
    org = await pipe.org()
    s = await save(pipe, org.tenant_id, org.boss_id,
                   [bill_json(bill_no="OK", date_text=day), bill_json(bill_no="BAD", date_text=day, total_paise=5)])  # fmt: skip
    async with pipe.sf() as session, session.begin():
        await bill_batch.confirm_ready(
            TenantRepository(session, org.tenant_id), s.batch_id, org.boss_id, NOW
        )
    out = texts(await pipe.say(BOSS_CHAT, "/stock"))
    assert "Basmati rice: 25 kg" in out and "1 bill abhi review mein" in out


async def test_today_summarises_todays_bills_for_the_boss_only(pipe: Env) -> None:
    day = today_text(pipe)
    await shop(pipe, [bill_json(date_text=day), sale(day), sale("1 Jan 2020", bill_no="OLD")])
    out = texts(await pipe.say(BOSS_CHAT, "/today"))
    assert "Bikri: 1 bills, ₹720" in out and "Kharid: 1 bills, ₹2,462" in out
    assert "Sabse zyada bika: Basmati rice 8 kg" in out
    assert "boss ke liye" in texts(await pipe.say(REP_CHAT, "/today"))  # money stays with the boss


async def test_today_with_nothing_says_so(pipe: Env) -> None:
    await pipe.org()
    assert "koi confirmed bill nahi" in texts(await pipe.say(BOSS_CHAT, "/today"))


async def test_plain_questions_get_the_same_answers_as_the_commands(pipe: Env) -> None:
    day = today_text(pipe)
    await shop(pipe, [bill_json(date_text=day), sale(day)])
    assert "Bikri: 1 bills" in texts(await pipe.say(BOSS_CHAT, "aaj kya bika?"))
    assert "Basmati rice: 17 kg" in texts(
        await pipe.say(REP_CHAT, "basmati rice ka stock kitna hai?")
    )
    assert "boss ke liye" in texts(await pipe.say(REP_CHAT, "aaj kya bika?"))
    assert pipe.llm is not None and pipe.llm.calls == []  # answered by code, no model call


async def test_a_visit_note_mentioning_stock_still_goes_to_the_note_pipeline(pipe: Env) -> None:
    await pipe.org()
    assert pipe.llm is not None
    await pipe.say(REP_CHAT, "Rajendra ko stock dikhaya")
    assert len(pipe.llm.calls) >= 1  # it reached the model (which retries once)


def test_json_helper_sanity() -> None:
    assert json.loads(sale("1 Nov 2026"))["doc_type"] == "sales" and isinstance(date.today(), date)
