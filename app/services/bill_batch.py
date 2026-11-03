"""A whole uploaded document: extract every page, save the results, and let the owner decide.

Pages are read with bounded concurrency (provider rate limits are real). Nothing is confirmed
here: clean pages become `ready` and flagged pages `needs_review`; a person confirms the ready
ones with one action and accepts or rejects each flagged one. Duplicates (same party, bill
number and date) are flagged against both earlier uploads and the same batch.
"""

from __future__ import annotations

import asyncio
import uuid
from collections import Counter
from dataclasses import dataclass, field
from datetime import date, datetime

from app.db import TenantRepository
from app.models import Bill, BillLine
from app.pipeline.bills import BillAssessment, assess_bill, name_key, run_bill_extraction
from app.pipeline.documents import PageInput
from app.providers.llm import LLMProvider
from app.schemas.extraction import Flag

DUPLICATE_STATUSES = ("ready", "needs_review", "confirmed", "synced")


@dataclass
class PageOutcome:
    page_no: int
    route: str
    status: str  # ready | needs_review | unreadable | failed
    assessment: BillAssessment | None = None
    model: str | None = None
    error: str | None = None


async def extract_pages(
    llm: LLMProvider,
    pages: list[PageInput],
    *,
    today: date,
    timezone: str,
    min_confidence: float,
    min_legibility: float,
    concurrency: int = 3,
    retry_delay_s: float = 3.0,
    business: str = "",
) -> list[PageOutcome]:
    gate = asyncio.Semaphore(max(1, concurrency))

    async def one(page: PageInput) -> PageOutcome:
        async with gate:
            try:
                run = await run_bill_extraction(
                    llm, page, today=today, timezone=timezone, business=business
                )
            except Exception as exc:  # one bad page never sinks the batch
                return PageOutcome(page.page_no, page.route, "failed", error=type(exc).__name__)
        if run.parsed is None:
            return PageOutcome(page.page_no, page.route, "failed", model=run.model, error=run.error)
        a = assess_bill(
            run.parsed, today=today, min_confidence=min_confidence, min_legibility=min_legibility
        )
        return PageOutcome(page.page_no, page.route, a.status, a, run.model)

    return list(await asyncio.gather(*(one(p) for p in pages)))


@dataclass
class BatchSummary:
    batch_id: uuid.UUID
    bill_ids: list[uuid.UUID] = field(default_factory=list)
    counts: Counter[str] = field(default_factory=Counter)


async def save_batch(
    repo: TenantRepository,
    *,
    uploaded_by: uuid.UUID,
    source_name: str,
    source_sha256: str,
    outcomes: list[PageOutcome],
) -> BatchSummary:
    summary = BatchSummary(batch_id=uuid.uuid4())
    seen: dict[tuple[str, str, date | None, str], int] = {}
    for o in sorted(outcomes, key=lambda x: x.page_no):
        a = o.assessment
        flags: list[Flag] = list(a.flags) if a else []
        status = o.status
        party_key = name_key(a.party_name) if a and a.party_name else None
        if a and status in ("ready", "needs_review") and party_key and a.bill_no:
            key = (party_key, a.bill_no.casefold(), a.bill_date, a.doc_type)
            earlier = await repo.first(
                Bill,
                Bill.party_key == party_key,
                Bill.bill_no == a.bill_no,
                Bill.bill_date == a.bill_date,
                Bill.doc_type == a.doc_type,
                Bill.status.in_(DUPLICATE_STATUSES),
            )
            if earlier is not None:
                flags.append(
                    Flag(
                        field="bill_no",
                        reason="duplicate",
                        detail={"of_bill": earlier.id.hex, "earlier_upload": True},
                    )
                )
                status = "needs_review"
            elif key in seen:
                flags.append(
                    Flag(
                        field="bill_no",
                        reason="duplicate",
                        detail={"of_page": seen[key], "earlier_upload": False},
                    )
                )
                status = "needs_review"
            else:
                seen[key] = o.page_no
        bill = Bill(
            uploaded_by=uploaded_by,
            batch_id=summary.batch_id,
            source_name=source_name[:200],
            source_sha256=source_sha256,
            page_no=o.page_no,
            route=o.route,
            doc_type=a.doc_type if a else "other",
            party_name=a.party_name if a else None,
            party_key=party_key,
            bill_no=a.bill_no if a else None,
            bill_date=a.bill_date if a else None,
            subtotal_paise=a.subtotal_paise if a else None,
            tax_paise=a.tax_paise if a else None,
            total_paise=a.total_paise if a else None,
            status=status,
            flags=[f.model_dump() for f in flags]
            + (
                [{"field": "page", "reason": o.status, "detail": {"error": (o.error or "")[:200]}}]
                if not a
                else []
            ),
            llm_model=o.model,
        )
        repo.add(bill)
        await repo.flush()
        if a:
            for i, ln in enumerate(a.lines, start=1):
                repo.add(
                    BillLine(
                        bill_id=bill.id,
                        position=i,
                        name=ln.name,
                        name_key=ln.name_key,
                        unit=ln.unit,
                        quantity_milli=ln.quantity_milli,
                        rate_paise=ln.rate_paise,
                        amount_paise=ln.amount_paise,
                    )
                )
        summary.bill_ids.append(bill.id)
        summary.counts[status] += 1
    repo.audit(
        "bill_batch_saved",
        entity="bill_batch",
        entity_id=str(summary.batch_id),
        actor_user_id=uploaded_by,
        detail={"source": source_name[:80], "pages": len(outcomes), **summary.counts},
    )
    return summary


async def confirm_ready(
    repo: TenantRepository, batch_id: uuid.UUID, actor_id: uuid.UUID, now: datetime
) -> int:
    """One action confirms every CLEAN bill of the batch. Flagged bills are never included."""
    n = await repo.update_where(
        Bill,
        {"status": "confirmed", "decided_by": actor_id, "decided_at": now},
        Bill.batch_id == batch_id,
        Bill.status == "ready",
    )
    repo.audit(
        "bill_batch_confirmed",
        entity="bill_batch",
        entity_id=str(batch_id),
        actor_user_id=actor_id,
        detail={"confirmed": n},
    )
    return n


async def decide_bill(
    repo: TenantRepository, bill_id: uuid.UUID, actor_id: uuid.UUID, now: datetime, *, accept: bool
) -> bool:
    """A person accepts a flagged bill as it stands, or rejects any not-yet-confirmed bill.

    The flags the owner saw are recorded in the audit log with the decision."""
    bill = await repo.get(Bill, bill_id)
    if bill is None:
        return False
    allowed = ("needs_review",) if accept else ("ready", "needs_review", "unreadable", "failed")
    n = await repo.update_where(
        Bill,
        {
            "status": "confirmed" if accept else "rejected",
            "decided_by": actor_id,
            "decided_at": now,
        },
        Bill.id == bill_id,
        Bill.status.in_(allowed),
    )
    if n:
        repo.audit(
            "bill_accepted" if accept else "bill_rejected",
            entity="bill",
            entity_id=bill_id.hex,
            actor_user_id=actor_id,
            detail={"flags": [f.get("reason") for f in bill.flags]},
        )
    return bool(n)


async def reverse_bill(
    repo: TenantRepository,
    bill_id: uuid.UUID,
    actor_id: uuid.UUID,
    now: datetime,
    *,
    reason: str | None = None,
) -> Bill | None:
    """Take back a CONFIRMED bill. It leaves stock and every report at once, and the sheet row is
    marked "reversed" on the next sync (the row is kept as the record, never deleted).

    Returns the bill, or None if it was not in a reversible state (compare-and-set, so a double
    tap or a concurrent reversal changes nothing). A bill that never reached the sheet has
    nothing to update there, so its reversal counts as already synced."""
    bill = await repo.get(Bill, bill_id)
    if bill is None or bill.status not in ("confirmed", "synced"):
        return None
    prior = bill.status
    never_sent = bill.synced_at is None
    n = await repo.update_where(
        Bill,
        {
            "status": "reversed",
            "reversed_by": actor_id,
            "reversed_at": now,
            "reversal_reason": reason[:200] if reason else None,
            "reversal_synced_at": now if never_sent else None,
        },
        Bill.id == bill_id,
        Bill.status.in_(("confirmed", "synced")),
    )
    if not n:
        return None
    repo.audit(
        "bill_reversed",
        entity="bill",
        entity_id=bill_id.hex,
        actor_user_id=actor_id,
        detail={
            "prior_status": prior,
            "party": bill.party_name,
            "bill_no": bill.bill_no,
            "total_paise": bill.total_paise,
            "reason": reason[:200] if reason else None,
        },
    )
    return bill
