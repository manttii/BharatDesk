"""Pilot evidence, computed ONLY from the live database.

Nothing here is estimated or filled in: every figure is a count or a duration read from real
rows. Seeded demo tenants are excluded (their data is not evidence). When there is nothing to
count, the report says "no data" instead of printing zeros that look like results.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from fractions import Fraction
from typing import Any

from app.db import TenantRepository, list_tenant_ids
from app.models import ActivityLog, ApprovalRequest, Correction, Message, Tenant, User
from app.models.enums import ActivityStatus, ApprovalStatus, ProcessingState
from app.pipeline.confirm import TenantSettings

NOTE_KINDS = ("voice", "text", "photo")
_SAVED = (ActivityStatus.CONFIRMED, ActivityStatus.SYNCED)


@dataclass
class RepUsage:
    name: str
    notes: int = 0
    saved_visits: int = 0
    corrections: int = 0
    approvals_requested: int = 0


@dataclass
class PilotReport:
    tenants: int = 0
    notes_by_kind: dict[str, int] = field(default_factory=dict)  # processed (state DONE)
    notes_not_processed: int = 0  # failed or rejected
    corrections: int = 0
    approvals_by_status: dict[str, int] = field(default_factory=dict)
    approval_seconds: list[int] = field(default_factory=list)  # resolved approvals only
    reps: list[RepUsage] = field(default_factory=list)

    @property
    def notes_processed(self) -> int:
        return sum(self.notes_by_kind.values())

    @property
    def voice_notes(self) -> int:
        return self.notes_by_kind.get("voice", 0)

    @property
    def corrections_per_100_notes(self) -> Fraction | None:
        n = self.notes_processed
        return Fraction(100 * self.corrections, n) if n else None

    @property
    def approvals_total(self) -> int:
        return sum(self.approvals_by_status.values())

    @property
    def median_seconds_to_approve(self) -> Fraction | None:
        xs = sorted(self.approval_seconds)
        if not xs:
            return None
        mid = len(xs) // 2
        return Fraction(xs[mid]) if len(xs) % 2 else Fraction(xs[mid - 1] + xs[mid], 2)


def _in_window(t: datetime, since: datetime | None, until: datetime | None) -> bool:
    return (since is None or t >= since) and (until is None or t < until)


async def build_report(
    session_factory: Any,
    *,
    since: datetime | None = None,
    until: datetime | None = None,
    include_demo: bool = False,
    only_tenant: Any = None,
) -> PilotReport:
    rep = PilotReport()
    usage: dict[Any, RepUsage] = {}
    async with session_factory() as session:
        tenant_ids = await list_tenant_ids(session)
    for tenant_id in tenant_ids:
        if only_tenant is not None and tenant_id != only_tenant:
            continue
        async with session_factory() as session:
            tenant = await session.get(Tenant, tenant_id)
            assert tenant is not None
            if TenantSettings.from_json(tenant.settings).demo and not include_demo:
                continue
            repo = TenantRepository(session, tenant_id)
            rep.tenants += 1
            users = {u.id: u for u in await repo.list(User)}

            def row(user_id: Any, users: dict[Any, User] = users) -> RepUsage:
                if user_id not in usage:
                    u = users.get(user_id)
                    usage[user_id] = RepUsage(name=u.name if u else "-")
                return usage[user_id]

            for m in await repo.list(Message):
                if m.kind not in NOTE_KINDS or not _in_window(m.created_at, since, until):
                    continue
                if m.processing_state is ProcessingState.DONE:
                    rep.notes_by_kind[m.kind] = rep.notes_by_kind.get(m.kind, 0) + 1
                    row(m.user_id).notes += 1
                elif m.processing_state in (ProcessingState.FAILED, ProcessingState.REJECTED):
                    rep.notes_not_processed += 1
            for a in await repo.list(ActivityLog):
                if a.status in _SAVED and _in_window(a.created_at, since, until):
                    row(a.user_id).saved_visits += 1
            for c in await repo.list(Correction):
                if _in_window(c.created_at, since, until):
                    rep.corrections += 1
                    row(c.corrected_by).corrections += 1
            for ap in await repo.list(ApprovalRequest):
                if not _in_window(ap.created_at, since, until):
                    continue
                rep.approvals_by_status[ap.status.value] = (
                    rep.approvals_by_status.get(ap.status.value, 0) + 1
                )
                row(ap.requested_by).approvals_requested += 1
                if (
                    ap.status in (ApprovalStatus.APPROVED, ApprovalStatus.REJECTED)
                    and ap.resolved_at
                ):
                    rep.approval_seconds.append(
                        int((ap.resolved_at - ap.created_at).total_seconds())
                    )
    rep.reps = sorted(usage.values(), key=lambda r: (-r.notes, r.name))
    return rep


def _one_decimal(x: Fraction) -> str:
    tenths = (x * 10 + Fraction(1, 2)) // 1  # round half up, exactly
    return f"{tenths // 10}.{tenths % 10}"


def _duration(seconds: Fraction) -> str:
    total = int(seconds)
    h, rest = divmod(total, 3600)
    m, s = divmod(rest, 60)
    return f"{h}h {m}m {s}s" if h else (f"{m}m {s}s" if m else f"{s}s")


def render_markdown(rep: PilotReport, *, title: str = "BharatDesk pilot report") -> str:
    lines = [f"# {title}", ""]
    if rep.tenants == 0 or (
        rep.notes_processed == 0 and rep.approvals_total == 0 and rep.corrections == 0
    ):
        lines += [
            "**no data**",
            "",
            "No processed notes, approvals or corrections were found for this period (demo tenants",
            "are excluded). Figures appear here only once real reps have used the bot.",
            "",
        ]
        return "\n".join(lines)
    per100 = rep.corrections_per_100_notes
    median = rep.median_seconds_to_approve
    kinds = ", ".join(f"{k}: {v}" for k, v in sorted(rep.notes_by_kind.items())) or "none"
    lines += [
        f"- tenants included: {rep.tenants}",
        f"- **voice notes processed: {rep.voice_notes}** (all notes processed: {rep.notes_processed}; {kinds})",
        f"- notes that failed or were rejected: {rep.notes_not_processed}",
        f"- **corrections per 100 notes: {_one_decimal(per100) if per100 is not None else 'n/a'}** "
        f"({rep.corrections} corrections over {rep.notes_processed} notes)",
        f"- **approval requests: {rep.approvals_total}** "
        f"({', '.join(f'{k}: {v}' for k, v in sorted(rep.approvals_by_status.items())) or 'none'})",
        f"- **median time to approve: {_duration(median) if median is not None else 'n/a (none resolved yet)'}** "
        f"(over {len(rep.approval_seconds)} resolved)",
        "",
        "## Per-rep usage",
        "",
        "| Rep | Notes | Saved visits | Corrections | Approvals requested |",
        "|---|---:|---:|---:|---:|",
        *[
            f"| {r.name} | {r.notes} | {r.saved_visits} | {r.corrections} | {r.approvals_requested} |"
            for r in rep.reps
        ],
        "",
        "_A correction is a rep's Sudharo or a boss's dashboard fix. Counts come straight from the database._",
        "",
    ]
    return "\n".join(lines)
