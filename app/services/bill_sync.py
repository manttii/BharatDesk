"""Confirmed bills -> the owner's sheet, many rows per call, plus a live stock tab.

Three tabs: "Bills" (one row per bill), "Bill items" (one row per line) and "Stock"
(bought minus sold per item, rebuilt from every confirmed bill). Only CONFIRMED or SYNCED
bills are ever sent or counted. Rows are keyed by id, so a retry updates instead of duplicating.
Stock groups by exact normalised item name and unit; it never merges names fuzzily.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import structlog

from app.db import TenantRepository, list_tenant_ids
from app.models import Bill, BillLine, Tenant
from app.models.bills import COUNTED_STATUSES
from app.pipeline.confirm import TenantSettings
from app.providers.sheets import SheetsError, SheetsProvider
from app.services.sheets_sync import amount_cell, safe_text

log = structlog.get_logger()

BILLS_TAB, ITEMS_TAB, STOCK_TAB = "Bills", "Bill items", "Stock"
BILL_HEADER = [
    "Bill ID",
    "Date",
    "Type",
    "Party",
    "Bill no",
    "Items",
    "Subtotal (INR)",
    "Tax (INR)",
    "Total (INR)",
    "Status",
    "Source file",
    "Page",
    "Flags",
    "Updated (UTC)",
]
ITEM_HEADER = [
    "Line ID",
    "Bill ID",
    "Date",
    "Type",
    "Party",
    "Item",
    "Unit",
    "Quantity",
    "Rate (INR)",
    "Amount (INR)",
    "Status",
]
STOCK_HEADER = ["Item", "Unit", "Bought", "Sold", "In stock", "Updated (UTC)"]
_CHUNK = 400


def milli_cell(milli: int | None) -> int | str:
    """Quantity in thousandths -> a number: whole units as an int, else an exact decimal string."""
    if milli is None:
        return ""
    sign = "-" if milli < 0 else ""
    whole, rem = divmod(abs(milli), 1000)
    if rem == 0:
        return -whole if sign else whole
    return f"{sign}{whole}.{rem:03d}".rstrip("0")


def status_cell(bill: Bill) -> str:
    """What the sheet shows: confirmed (including synced), or reversed."""
    return "confirmed" if bill.status in ("confirmed", "synced") else bill.status


def bill_row(bill: Bill, item_count: int, now: datetime) -> list[Any]:
    reasons = ", ".join(sorted({str(f.get("reason", "")) for f in bill.flags if f.get("reason")}))
    return [
        bill.id.hex,
        bill.bill_date.isoformat() if bill.bill_date else "",
        bill.doc_type,
        safe_text(bill.party_name),
        safe_text(bill.bill_no),
        item_count,
        amount_cell(bill.subtotal_paise),
        amount_cell(bill.tax_paise),
        amount_cell(bill.total_paise),
        status_cell(bill),
        safe_text(bill.source_name),
        bill.page_no,
        safe_text(reasons),  # flags the owner accepted, kept visible
        now.strftime("%Y-%m-%d %H:%M:%S"),
    ]


def item_row(line: BillLine, bill: Bill) -> list[Any]:
    return [
        line.id.hex,
        bill.id.hex,
        bill.bill_date.isoformat() if bill.bill_date else "",
        bill.doc_type,
        safe_text(bill.party_name),
        safe_text(line.name),
        safe_text(line.unit),
        milli_cell(line.quantity_milli),
        amount_cell(line.rate_paise),
        amount_cell(line.amount_paise),
        status_cell(bill),
    ]


@dataclass
class StockLine:
    name: str
    unit: str
    bought: int = 0
    sold: int = 0

    @property
    def on_hand(self) -> int:
        return self.bought - self.sold


def build_stock(pairs: list[tuple[Bill, list[BillLine]]]) -> list[StockLine]:
    """Purchases add, sales subtract, 'other' documents are ignored; lines with no quantity
    are skipped (they cannot be counted honestly)."""
    table: dict[tuple[str, str], StockLine] = {}
    for bill, lines in pairs:
        if bill.doc_type not in ("purchase", "sales"):
            continue
        for ln in lines:
            if ln.quantity_milli is None or not ln.name_key:
                continue
            unit = (ln.unit or "").casefold()
            row = table.setdefault((ln.name_key, unit), StockLine(ln.name, unit))
            if bill.doc_type == "purchase":
                row.bought += ln.quantity_milli
            else:
                row.sold += ln.quantity_milli
    return sorted(table.values(), key=lambda r: (r.name.casefold(), r.unit))


def stock_rows(stock: list[StockLine], now: datetime) -> list[list[Any]]:
    stamp = now.strftime("%Y-%m-%d %H:%M:%S")
    return [
        [
            safe_text(s.name),
            safe_text(s.unit),
            milli_cell(s.bought),
            milli_cell(s.sold),
            milli_cell(s.on_hand),
            stamp,
        ]
        for s in stock
    ]


async def lines_by_bill(
    repo: TenantRepository, bill_ids: list[uuid.UUID]
) -> dict[uuid.UUID, list[BillLine]]:
    out: dict[uuid.UUID, list[BillLine]] = {b: [] for b in bill_ids}
    for i in range(0, len(bill_ids), _CHUNK):
        chunk = bill_ids[i : i + _CHUNK]
        for ln in await repo.list(
            BillLine, BillLine.bill_id.in_(chunk), order_by=[BillLine.bill_id, BillLine.position]
        ):
            out[ln.bill_id].append(ln)
    return out


@dataclass
class BillSyncReport:
    bills_sent: int = 0
    tenants_failed: int = 0


async def sync_bills(
    session_factory: Any, sheets: SheetsProvider, *, now: datetime
) -> BillSyncReport:
    report = BillSyncReport()
    async with session_factory() as session:
        tenant_ids = await list_tenant_ids(session)
    for tenant_id in tenant_ids:
        try:
            report.bills_sent += await _sync_tenant(session_factory, sheets, tenant_id, now)
        except SheetsError as exc:
            report.tenants_failed += 1
            log.warning("bill_sync_failed", tenant_id=str(tenant_id), error=str(exc)[:120])
        except Exception:
            report.tenants_failed += 1
            log.exception("bill_sync_unexpected", tenant_id=str(tenant_id))
    return report


async def _sync_tenant(sf: Any, sheets: SheetsProvider, tenant_id: uuid.UUID, now: datetime) -> int:
    async with sf() as session, session.begin():
        repo = TenantRepository(session, tenant_id)
        tenant = await session.get(Tenant, tenant_id)
        assert tenant is not None
        sheet_id = TenantSettings.from_json(tenant.settings).sheet_spreadsheet_id
        pending = await repo.list(Bill, Bill.status == "confirmed", order_by=[Bill.created_at])
        # reversed bills that are on the sheet as "confirmed" and must now show as "reversed"
        undone = await repo.list(
            Bill,
            Bill.status == "reversed",
            Bill.synced_at.is_not(None),
            Bill.reversal_synced_at.is_(None),
            order_by=[Bill.reversed_at],
        )
        if not sheet_id or not (pending or undone):
            return 0
        to_write = [*pending, *undone]
        written_lines = await lines_by_bill(repo, [b.id for b in to_write])
        counted = await repo.list(Bill, Bill.status.in_(COUNTED_STATUSES))
        counted_lines = await lines_by_bill(repo, [b.id for b in counted])
        bill_rows = {b.id.hex: bill_row(b, len(written_lines[b.id]), now) for b in to_write}
        line_rows = {ln.id.hex: item_row(ln, b) for b in to_write for ln in written_lines[b.id]}
        stock = stock_rows(build_stock([(b, counted_lines[b.id]) for b in counted]), now)
        ids = [b.id for b in pending]
        undone_ids = [b.id for b in undone]

    # network, outside any transaction: three tabs, a handful of API calls in total
    await sheets.upsert_rows(sheet_id, BILLS_TAB, BILL_HEADER, bill_rows)
    if line_rows:
        await sheets.upsert_rows(sheet_id, ITEMS_TAB, ITEM_HEADER, line_rows)
    await sheets.replace_table(sheet_id, STOCK_TAB, STOCK_HEADER, stock)

    async with sf() as session, session.begin():
        repo = TenantRepository(session, tenant_id)
        n = await repo.update_where(
            Bill,
            {"status": "synced", "synced_at": now},
            Bill.id.in_(ids),
            Bill.status == "confirmed",  # compare-and-set: a concurrent worker cannot double count
        )
        if undone_ids:
            await repo.update_where(
                Bill,
                {"reversal_synced_at": now},
                Bill.id.in_(undone_ids),
                Bill.status == "reversed",
            )
        repo.audit("bills_synced", entity="bill", detail={"count": n, "reversals": len(undone_ids)})
    return n + len(undone_ids)
