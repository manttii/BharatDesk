"""Answers about the shop, computed from CONFIRMED bills only (the same rule the sheet uses).

Stock is bought minus sold per exact item name and unit, via the same function that builds the
sheet's Stock tab, so the bot and the sheet can never disagree. Money is integer paise.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import date

from app.db import TenantRepository
from app.models import Bill
from app.models.bills import COUNTED_STATUSES
from app.pipeline.bills import name_key
from app.services.bill_sync import StockLine, build_stock, lines_by_bill

REVIEW_STATUSES = ("ready", "needs_review")


@dataclass
class StockReport:
    lines: list[StockLine]
    pending_bills: int  # waiting for the boss: NOT counted above


async def stock_report(repo: TenantRepository, item: str | None = None) -> StockReport:
    counted = await repo.list(Bill, Bill.status.in_(COUNTED_STATUSES))
    by_bill = await lines_by_bill(repo, [b.id for b in counted])
    lines = build_stock([(b, by_bill[b.id]) for b in counted])
    key = name_key(item) if item else ""
    if key:
        lines = [s for s in lines if key in name_key(s.name)]
    pending = len(await repo.list(Bill, Bill.status.in_(REVIEW_STATUSES)))
    return StockReport(lines, pending)


@dataclass
class DaySummary:
    day: date
    sales_count: int = 0
    sales_paise: int = 0
    purchase_count: int = 0
    purchase_paise: int = 0
    unknown_totals: int = 0  # confirmed bills with no usable total: not in the sums
    top_sold: list[tuple[str, str, int]] = field(default_factory=list)  # name, unit, qty_milli
    pending_bills: int = 0


async def day_summary(repo: TenantRepository, day: date, *, top: int = 3) -> DaySummary:
    out = DaySummary(day)
    bills = await repo.list(Bill, Bill.status.in_(COUNTED_STATUSES), Bill.bill_date == day)
    by_bill = await lines_by_bill(repo, [b.id for b in bills])
    sold: Counter[tuple[str, str]] = Counter()
    names: dict[tuple[str, str], str] = {}
    for b in bills:
        if b.doc_type not in ("sales", "purchase"):
            continue
        if b.total_paise is None:
            out.unknown_totals += 1
        elif b.doc_type == "sales":
            out.sales_paise += b.total_paise
        else:
            out.purchase_paise += b.total_paise
        if b.doc_type == "sales":
            out.sales_count += 1
            for ln in by_bill[b.id]:
                if ln.quantity_milli is not None and ln.name_key:
                    k = (ln.name_key, (ln.unit or "").casefold())
                    sold[k] += ln.quantity_milli
                    names.setdefault(k, ln.name)
        else:
            out.purchase_count += 1
    out.top_sold = [(names[k], k[1], q) for k, q in sold.most_common(top)]
    out.pending_bills = len(await repo.list(Bill, Bill.status.in_(REVIEW_STATUSES)))
    return out
