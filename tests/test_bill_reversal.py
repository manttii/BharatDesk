"""Reversing a confirmed bill: stock and reports drop it, the sheet shows it as reversed."""

from __future__ import annotations

from typing import Any

from sqlalchemy import select

from app.db import TenantRepository
from app.models import AuditLog, Bill
from app.providers.sheets import GSpreadSheets, SheetsError
from app.services import bill_batch, bill_sync
from app.services.shop_reports import stock_report
from tests.conftest import BOSS_CHAT, REP_CHAT, SHEET_ID, Env, button_data
from tests.test_bills import NOW, bill_json, save
from tests.test_shop_queries import sale, today_text


async def confirmed(
    pipe: Env, replies: list[str], *, sheet: bool = True
) -> tuple[Any, Any, list[Any]]:
    org = await pipe.org()
    if sheet:
        await pipe.connect_sheet()
    s = await save(pipe, org.tenant_id, org.boss_id, replies)
    async with pipe.sf() as session, session.begin():
        await bill_batch.confirm_ready(
            TenantRepository(session, org.tenant_id), s.batch_id, org.boss_id, NOW
        )
    return org.tenant_id, org.boss_id, s.bill_ids


async def reverse(pipe: Env, tenant_id: Any, boss_id: Any, bill_id: Any) -> Any:
    async with pipe.sf() as session, session.begin():
        return await bill_batch.reverse_bill(
            TenantRepository(session, tenant_id), bill_id, boss_id, NOW, reason="entered twice"
        )


async def status_of(pipe: Env, bill_id: Any) -> str:
    async with pipe.sf() as s:
        return (await s.execute(select(Bill.status).where(Bill.id == bill_id))).scalar_one()


# --- the service ---------------------------------------------------------------------------------


async def test_a_confirmed_bill_is_reversed_audited_and_leaves_stock(pipe: Env) -> None:
    tenant_id, boss_id, (a, b) = await confirmed(
        pipe, [bill_json(bill_no="A"), bill_json(bill_no="B", party_name="Other")], sheet=False
    )
    assert await reverse(pipe, tenant_id, boss_id, a) is not None
    assert await status_of(pipe, a) == "reversed" and await status_of(pipe, b) == "confirmed"
    async with pipe.sf() as s:
        stock = await stock_report(TenantRepository(s, tenant_id))
        audit = (
            await s.execute(select(AuditLog.detail).where(AuditLog.action == "bill_reversed"))
        ).scalar_one()
    assert {x.name: x.on_hand for x in stock.lines}["Basmati rice"] == 25000  # only bill B left
    assert audit["bill_no"] == "A" and audit["reason"] == "entered twice" and audit["prior_status"]


async def test_only_confirmed_bills_can_be_reversed_and_only_once(pipe: Env) -> None:
    org = await pipe.org()
    s = await save(pipe, org.tenant_id, org.boss_id,
                   [bill_json(bill_no="R"), bill_json(bill_no="F", total_paise=1)])  # fmt: skip
    ready, flagged = s.bill_ids
    assert await reverse(pipe, org.tenant_id, org.boss_id, ready) is None  # not confirmed yet
    assert await reverse(pipe, org.tenant_id, org.boss_id, flagged) is None
    async with pipe.sf() as session, session.begin():
        await bill_batch.confirm_ready(
            TenantRepository(session, org.tenant_id), s.batch_id, org.boss_id, NOW
        )
    assert await reverse(pipe, org.tenant_id, org.boss_id, ready) is not None
    assert await reverse(pipe, org.tenant_id, org.boss_id, ready) is None  # already reversed


async def test_another_business_cannot_reverse_my_bill(pipe: Env) -> None:
    _, _, (a,) = await confirmed(pipe, [bill_json()], sheet=False)
    other, other_boss, _ = await pipe.new_tenant("Other Shop", "Owner")
    assert await reverse(pipe, other, other_boss, a) is None
    assert await status_of(pipe, a) == "confirmed"


async def test_a_reversed_bill_can_be_uploaded_again_without_a_duplicate_flag(pipe: Env) -> None:
    tenant_id, boss_id, (a,) = await confirmed(pipe, [bill_json(bill_no="D")], sheet=False)
    await reverse(pipe, tenant_id, boss_id, a)
    again = await save(pipe, tenant_id, boss_id, [bill_json(bill_no="D")], name="again.pdf")
    assert dict(again.counts) == {"ready": 1}


# --- the sheet -----------------------------------------------------------------------------------


async def test_a_synced_bill_shows_as_reversed_and_stock_is_recomputed(pipe: Env) -> None:
    assert pipe.sheets is not None
    tenant_id, boss_id, (a, _) = await confirmed(
        pipe, [bill_json(bill_no="A"), bill_json(bill_no="B", party_name="Other")]
    )
    await bill_sync.sync_bills(pipe.sf, pipe.sheets, now=NOW)
    assert {r[9] for r in pipe.sheets.rows(SHEET_ID, "Bills")} == {"confirmed"}
    await reverse(pipe, tenant_id, boss_id, a)
    report = await bill_sync.sync_bills(pipe.sf, pipe.sheets, now=NOW)
    assert report.bills_sent == 1
    status = {r[4]: r[9] for r in pipe.sheets.rows(SHEET_ID, "Bills")}
    assert status == {"A": "reversed", "B": "confirmed"}  # the row is kept, not deleted
    items = pipe.sheets.rows(SHEET_ID, "Bill items")
    assert sorted({r[10] for r in items}) == ["confirmed", "reversed"]
    stock = {r[0]: r[4] for r in pipe.sheets.rows(SHEET_ID, "Stock")}
    assert stock["Basmati rice"] == 25 and stock["Sugar"] == "10.5"  # one bill, not two
    again = await bill_sync.sync_bills(pipe.sf, pipe.sheets, now=NOW)
    assert again.bills_sent == 0  # the reversal is delivered once


async def test_a_bill_reversed_before_it_reached_the_sheet_never_appears(pipe: Env) -> None:
    assert pipe.sheets is not None
    tenant_id, boss_id, (a,) = await confirmed(pipe, [bill_json(bill_no="E")])
    await reverse(pipe, tenant_id, boss_id, a)
    report = await bill_sync.sync_bills(pipe.sf, pipe.sheets, now=NOW)
    assert report.bills_sent == 0 and pipe.sheets.rows(SHEET_ID, "Bills") == []


async def test_a_sheet_failure_keeps_the_reversal_pending_for_the_next_run(pipe: Env) -> None:
    assert pipe.sheets is not None
    tenant_id, boss_id, (a,) = await confirmed(pipe, [bill_json(bill_no="G")])
    await bill_sync.sync_bills(pipe.sf, pipe.sheets, now=NOW)
    await reverse(pipe, tenant_id, boss_id, a)
    pipe.sheets.fail.append(SheetsError("sheets API error HTTP 503"))
    assert (await bill_sync.sync_bills(pipe.sf, pipe.sheets, now=NOW)).tenants_failed == 1
    assert pipe.sheets.rows(SHEET_ID, "Bills")[0][9] == "confirmed"  # not yet updated
    assert (await bill_sync.sync_bills(pipe.sf, pipe.sheets, now=NOW)).bills_sent == 1
    assert pipe.sheets.rows(SHEET_ID, "Bills")[0][9] == "reversed"


# --- the real adapter, with a recording stand-in for gspread -------------------------------------


class _WS:
    def __init__(self, rows: list[list[str]]) -> None:
        self.rows = rows
        self.batches: list[Any] = []
        self.appended: list[Any] = []

    def col_values(self, n: int) -> list[str]:
        return [r[0] for r in self.rows]

    def row_values(self, n: int) -> list[str]:
        return self.rows[0] if self.rows else []

    def batch_update(self, data: Any) -> None:
        self.batches.append(data)

    def append_rows(self, rows: Any) -> None:
        self.appended.append(rows)

    def append_row(self, row: Any) -> None:
        self.appended.append([row])


class _Book:
    def __init__(self, ws: _WS) -> None:
        self.ws = ws

    def worksheet(self, title: str) -> _WS:
        return self.ws


async def test_the_gspread_adapter_updates_in_bulk_and_repairs_a_changed_header() -> None:
    ws = _WS([["Bill ID", "Old col"], ["k1", "x"]])
    adapter = GSpreadSheets({"client_email": "bot@example.iam"})
    adapter._client = type("C", (), {"open_by_key": lambda self, _id: _Book(ws)})()  # type: ignore[assignment]
    await adapter.upsert_rows(
        "sheet", "Bills", ["Bill ID", "New col"], {"k1": ["k1", "y"], "k2": ["k2", "z"]}
    )
    (updates,) = ws.batches
    assert {"range": "A1", "values": [["Bill ID", "New col"]]} in updates  # header repaired
    assert {"range": "A2", "values": [["k1", "y"]]} in updates  # existing key updated in place
    assert ws.appended == [[["k2", "z"]]]  # new key appended, in one call


# --- in the chat ----------------------------------------------------------------------------------


def texts(msgs: list[Any]) -> str:
    return "\n".join(m.text for m in msgs)


async def test_the_boss_reverses_a_bill_with_two_taps_and_stock_follows(pipe: Env) -> None:
    day = today_text(pipe)
    await confirmed(
        pipe, [bill_json(bill_no="P1", date_text=day), sale(day, bill_no="S1")], sheet=False
    )
    listing = await pipe.say(BOSS_CHAT, "/reverse")
    assert "ulta karna hai" in listing[0].text and len(listing) == 3
    sale_msg = next(m for m in listing if "bill S1" in m.text)
    ask = await pipe.tap(BOSS_CHAT, button_data(sale_msg, "Ulta karo"))
    assert "Pakka?" in texts(ask)
    done = await pipe.tap(BOSS_CHAT, button_data(ask[-1], "Haan, ulta karo"))
    assert "Bill ulta kar diya" in texts(done)
    stock = texts(await pipe.say(BOSS_CHAT, "/stock rice"))
    assert "Basmati rice: 25 kg" in stock  # the 8 kg sale no longer counts
    assert "Bikri: 0 bills" in texts(await pipe.say(BOSS_CHAT, "/today"))
    again = await pipe.tap(BOSS_CHAT, button_data(ask[-1], "Haan, ulta karo"))
    assert "ulta kar diya" not in texts(again)  # a second tap changes nothing


async def test_the_no_button_leaves_the_bill_alone_and_a_bill_number_finds_it(pipe: Env) -> None:
    await confirmed(pipe, [bill_json(bill_no="ST-77")], sheet=False)
    found = await pipe.say(BOSS_CHAT, "/reverse st-77")
    ask = await pipe.tap(BOSS_CHAT, button_data(found[-1], "Ulta karo"))
    cancelled = await pipe.tap(BOSS_CHAT, button_data(ask[-1], "Nahi"))
    assert "waisa hi hai" in texts(cancelled)
    assert "Basmati rice: 25 kg" in texts(await pipe.say(BOSS_CHAT, "/stock rice"))
    assert "nahi mila" in texts(await pipe.say(BOSS_CHAT, "/reverse zzz"))


async def test_a_rep_cannot_reverse_even_with_the_button_data(pipe: Env) -> None:
    await confirmed(pipe, [bill_json(bill_no="X")], sheet=False)
    out = await pipe.say(BOSS_CHAT, "/reverse")
    data = button_data(out[-1], "Ulta karo")
    assert "boss ke liye" in texts(await pipe.say(REP_CHAT, "/reverse"))
    await pipe.tap(REP_CHAT, data)
    assert "boss" in (pipe.fake.callback_answers[-1][1] or "").lower()
    ask = await pipe.tap(BOSS_CHAT, data)
    await pipe.tap(REP_CHAT, button_data(ask[-1], "Haan, ulta karo"))
    assert "Basmati rice: 25 kg" in texts(await pipe.say(BOSS_CHAT, "/stock rice"))
