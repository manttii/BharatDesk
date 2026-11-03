"""Bills read from uploaded PDFs/photos: one row per bill, one per line item.

Money is integer paise and quantity is integer thousandths of a unit (2.5 kg = 2500), so no
float ever touches stock or ledger arithmetic. `name_key` is a normalised item name used to
group stock; matching is exact on that key, never fuzzy, because a wrong merge silently
corrupts stock.
"""

from __future__ import annotations

import uuid
from datetime import date, datetime
from typing import Any

from sqlalchemy import JSON, BigInteger, Date, ForeignKey, Integer, String, Uuid
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, TenantOwned, UTCDateTime

# ready: clean, waiting for the owner's one-tap confirmation   needs_review: flagged by code
# unreadable / failed: nothing usable (blurry page / provider error)   confirmed: counted in stock
# synced: written to the sheet   rejected: dismissed by the owner
# reversed: a confirmed bill the owner took back (kept for the record; not counted anywhere)
BILL_STATUSES = (
    "ready", "needs_review", "unreadable", "failed", "confirmed", "synced", "rejected", "reversed"
)  # fmt: skip
COUNTED_STATUSES = ("confirmed", "synced")


class Bill(TenantOwned, Base):
    __tablename__ = "bills"

    uploaded_by: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id"))
    batch_id: Mapped[uuid.UUID] = mapped_column(index=True)
    source_name: Mapped[str] = mapped_column(String(200))
    source_sha256: Mapped[str] = mapped_column(String(64))
    page_no: Mapped[int] = mapped_column(Integer)
    route: Mapped[str] = mapped_column(String(8))  # text | image
    doc_type: Mapped[str] = mapped_column(String(16), default="other")  # purchase|sales|other
    party_name: Mapped[str | None] = mapped_column(String(200), nullable=True)
    party_key: Mapped[str | None] = mapped_column(String(200), nullable=True, index=True)
    bill_no: Mapped[str | None] = mapped_column(String(64), nullable=True)
    bill_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    subtotal_paise: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    tax_paise: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    total_paise: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    status: Mapped[str] = mapped_column(String(16), index=True)
    flags: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)
    llm_model: Mapped[str | None] = mapped_column(String(120), nullable=True)
    decided_by: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("users.id"), nullable=True)
    decided_at: Mapped[datetime | None] = mapped_column(UTCDateTime, nullable=True)
    synced_at: Mapped[datetime | None] = mapped_column(UTCDateTime, nullable=True)
    reversed_by: Mapped[uuid.UUID | None] = mapped_column(Uuid, nullable=True)
    reversed_at: Mapped[datetime | None] = mapped_column(UTCDateTime, nullable=True)
    reversal_reason: Mapped[str | None] = mapped_column(String(200), nullable=True)
    # set once the sheet shows the bill as reversed (or at once if it never reached the sheet)
    reversal_synced_at: Mapped[datetime | None] = mapped_column(UTCDateTime, nullable=True)


class BillLine(TenantOwned, Base):
    __tablename__ = "bill_lines"

    bill_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("bills.id"), index=True)
    position: Mapped[int] = mapped_column(Integer)
    name: Mapped[str] = mapped_column(String(200))
    name_key: Mapped[str] = mapped_column(String(200), index=True)
    unit: Mapped[str | None] = mapped_column(String(16), nullable=True)
    quantity_milli: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    rate_paise: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    amount_paise: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
