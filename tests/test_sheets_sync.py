"""Google Sheets sync: outbox, retry/backoff, never-in-the-reply-path, never-unconfirmed."""

from __future__ import annotations

from datetime import timedelta
from typing import Any

import pytest
from gspread.exceptions import APIError, SpreadsheetNotFound, WorksheetNotFound
from sqlalchemy import select

from app.models import ActivityLog, ApprovalRequest, AuditLog, SyncJob, Tenant
from app.models.enums import ActivityStatus, SyncStatus
from app.pack import load_pack
from app.providers.sheets import HEADER, FakeSheets, GSpreadSheets, SheetsError
from app.services import approvals
from app.services.sheets_sync import (
    amount_cell,
    backoff_seconds,
    safe_text,
    status_label,
    sync_once,
    weight_cell,
)
from tests.conftest import (
    BOSS_CHAT,
    REP_CHAT,
    SHEET_ID,
    SHEET_URL,
    TRANSCRIPT,
    Env,
    button_data,
    extraction_json,
    to,
)

RATE_TRANSCRIPT = TRANSCRIPT + ", sattar hazaar rate"
RATE_RULE = ("rate_floor_vs_benchmark", {"max_below_bps": 100, "max_benchmark_age_seconds": 86_400})


async def _sync(
    p: Env, *, max_attempts: int = 8, batch: int = 20, sheets: FakeSheets | None = None
):  # type: ignore[no-untyped-def]
    assert p.sheets is not None
    return await sync_once(
        p.sf,
        sheets or p.sheets,
        load_pack(),
        now=p.clock(),
        max_attempts=max_attempts,
        batch_size=batch,
    )


async def _confirmed_visit(p: Env, transcript: str = TRANSCRIPT, **kw: Any) -> None:
    out = await p.voice(REP_CHAT, transcript, extraction_json(**kw))
    await p.tap(REP_CHAT, button_data(to(REP_CHAT, out)[0], "Haan"))


async def _org(p: Env, *, sheet: bool = True):  # type: ignore[no-untyped-def]
    org = await p.org()
    await p.add_client(org, "Rajendra Jewellers")
    if sheet:
        await p.connect_sheet()
    return org


async def _jobs(p: Env) -> list[SyncJob]:
    return await p.all(SyncJob)  # type: ignore[no-any-return]


# --- /sheet --------------------------------------------------------------------------------


async def test_boss_connects_a_sheet_with_a_readback_and_a_background_check(pipe: Env) -> None:
    org = await pipe.org()
    out = await pipe.say(BOSS_CHAT, "/sheet")
    assert "bharatdesk@example.iam.gserviceaccount.com" in out[-1].text  # who to share it with
    out = await pipe.say(BOSS_CHAT, f"/sheet {SHEET_URL}")
    assert SHEET_ID in out[-1].text
    async with pipe.sf() as s:
        assert "sheet_spreadsheet_id" not in (await s.get(Tenant, org.tenant_id)).settings  # type: ignore[union-attr]
    out = await pipe.tap(BOSS_CHAT, button_data(out[-1], "Haan"))
    assert out[0].text == "Sheet set ho gayi." and "Sheet connect ho gayi" in out[-1].text
    async with pipe.sf() as s:
        assert (await s.get(Tenant, org.tenant_id)).settings["sheet_spreadsheet_id"] == SHEET_ID  # type: ignore[union-attr]
    assert any(a.action == "sheet_configured" for a in await pipe.all(AuditLog))
    assert (
        "docs.google.com/spreadsheets/d/" + SHEET_ID
        in (await pipe.say(BOSS_CHAT, "/sheet"))[-1].text
    )


async def test_a_sheet_that_was_not_shared_is_reported_with_the_email_to_share_with(
    pipe: Env,
) -> None:
    await pipe.org()
    pipe.sheets.accessible = False  # type: ignore[union-attr]
    out = await pipe.connect_sheet()
    assert "access nahi mila" in out[-1].text and "bharatdesk@example.iam" in out[-1].text


async def test_bare_ids_are_accepted_and_junk_is_not(pipe: Env) -> None:
    await pipe.org()
    assert SHEET_ID in (await pipe.say(BOSS_CHAT, f"/sheet {SHEET_ID}"))[-1].text
    for junk in ("hello", "https://example.com/not-a-sheet", "short", "../../etc/passwd"):
        assert "samajh nahi aaya" in (await pipe.say(BOSS_CHAT, f"/sheet {junk}"))[-1].text


async def test_only_the_boss_can_connect_a_sheet(pipe: Env) -> None:
    await pipe.org()
    assert "sirf boss" in (await pipe.say(REP_CHAT, f"/sheet {SHEET_URL}"))[-1].text
    out = await pipe.say(BOSS_CHAT, f"/sheet {SHEET_URL}")
    await pipe.tap(REP_CHAT, button_data(out[-1], "Haan"))  # a rep pressing the boss's button
    assert "sheet_spreadsheet_id" not in (await pipe.all(Tenant))[0].settings


async def test_declining_and_double_confirming_the_sheet(pipe: Env) -> None:
    org = await pipe.org()
    out = await pipe.say(BOSS_CHAT, f"/sheet {SHEET_URL}")
    await pipe.tap(BOSS_CHAT, button_data(out[-1], "Nahi"))
    out = await pipe.say(BOSS_CHAT, f"/sheet {SHEET_URL}")
    yes = button_data(out[-1], "Haan")
    await pipe.tap(BOSS_CHAT, yes)
    assert await pipe.tap(BOSS_CHAT, yes) == []
    assert len([a for a in await pipe.all(AuditLog) if a.action == "sheet_configured"]) == 1
    assert org.tenant_id


async def test_sheet_command_without_a_provider_says_so(env: Env) -> None:
    await env.onboard_boss(BOSS_CHAT)
    assert "service account set nahi" in (await env.say(BOSS_CHAT, "/sheet"))[-1].text


# --- the outbox ----------------------------------------------------------------------------


async def test_confirming_creates_an_outbox_job_and_never_calls_sheets_in_the_reply_path(
    pipe: Env,
) -> None:
    await _org(pipe)
    await _confirmed_visit(pipe)
    assert pipe.sheets.calls == 0  # type: ignore[union-attr]  # nothing sent while replying
    (job,) = await _jobs(pipe)
    assert (job.status, job.attempts) == (SyncStatus.PENDING, 0)
    (act,) = await pipe.all(ActivityLog)
    assert act.status is ActivityStatus.CONFIRMED  # not SYNCED until the worker succeeds


async def test_the_reply_is_unaffected_when_sheets_is_down(pipe: Env) -> None:
    await _org(pipe)
    pipe.sheets.fail[:] = [SheetsError("down")] * 5  # type: ignore[union-attr]
    out = await pipe.voice(REP_CHAT, TRANSCRIPT, extraction_json())
    out = await pipe.tap(REP_CHAT, button_data(to(REP_CHAT, out)[0], "Haan"))
    assert "Save ho gaya" in to(REP_CHAT, out)[0].text and pipe.sheets.calls == 0  # type: ignore[union-attr]


async def test_the_sync_writes_the_row_and_marks_the_record_synced(pipe: Env) -> None:
    await _org(pipe)
    await _confirmed_visit(pipe)
    report = await _sync(pipe)
    assert report.sent == 1 and report.failed == 0
    (row,) = pipe.sheets.rows(SHEET_ID)  # type: ignore[union-attr]
    (act,) = await pipe.all(ActivityLog)
    assert row[:14] == [
        act.id.hex, "2026-11-02", "Ramesh", "Rajendra Jewellers", 150, 412_000, "", 45_000, 367_000,
        "2026-11-08", "", "confirmed", "note", "",
    ]  # fmt: skip
    assert all(isinstance(row[i], int) for i in (4, 5, 7, 8)) and not any(
        isinstance(c, float) for c in row
    )
    assert act.status is ActivityStatus.SYNCED
    (job,) = await _jobs(pipe)
    assert (job.status, job.attempts, job.synced_at) == (SyncStatus.DONE, 1, pipe.clock())
    assert any(a.action == "activity_synced" for a in await pipe.all(AuditLog))


async def test_jobs_wait_without_burning_attempts_until_a_sheet_is_connected(pipe: Env) -> None:
    await _org(pipe, sheet=False)
    await _confirmed_visit(pipe)
    report = await _sync(pipe)
    assert report.sent == 0 and report.skipped_tenants == 1 and pipe.sheets.calls == 0  # type: ignore[union-attr]
    assert (await _jobs(pipe))[0].attempts == 0
    await pipe.connect_sheet()
    assert (await _sync(pipe)).sent == 1  # the waiting record goes out once the sheet exists


async def test_running_the_sync_twice_never_duplicates_a_row(pipe: Env) -> None:
    await _org(pipe)
    await _confirmed_visit(pipe)
    await _sync(pipe)
    calls = pipe.sheets.calls  # type: ignore[union-attr]
    assert (await _sync(pipe)).sent == 0 and pipe.sheets.calls == calls  # type: ignore[union-attr]
    # a worker that lost its DB update would re-send the same key: still one row
    async with pipe.sf() as s, s.begin():
        job = (await s.execute(select(SyncJob))).scalar_one()
        job.status, job.attempts = SyncStatus.PENDING, 0
    await _sync(pipe)
    assert len(pipe.sheets.rows(SHEET_ID)) == 1  # type: ignore[union-attr]


async def test_the_header_and_columns_are_stable() -> None:
    assert HEADER[0] == "Record ID" and len(HEADER) == 15


# --- only confirmed records are ever sent --------------------------------------------------


async def test_a_record_awaiting_approval_is_not_queued_or_sent(pipe: Env) -> None:
    org = await _org(pipe)
    await pipe.add_rules(org, RATE_RULE)
    await pipe.set_benchmark(org)
    await _confirmed_visit(pipe, RATE_TRANSCRIPT, rate=7_000_000)  # below the floor
    (act,) = await pipe.all(ActivityLog)
    assert act.status is ActivityStatus.PENDING_APPROVAL
    assert await _jobs(pipe) == []
    assert (await _sync(pipe)).sent == 0 and pipe.sheets.rows(SHEET_ID) == []  # type: ignore[union-attr]


async def test_approval_queues_the_record_and_the_row_says_approved(pipe: Env) -> None:
    org = await _org(pipe)
    await pipe.add_rules(org, RATE_RULE)
    await pipe.set_benchmark(org)
    out = await pipe.voice(REP_CHAT, RATE_TRANSCRIPT, extraction_json(rate=7_000_000))
    out = await pipe.tap(REP_CHAT, button_data(to(REP_CHAT, out)[0], "Haan"))
    await pipe.tap(BOSS_CHAT, button_data(to(BOSS_CHAT, out)[0], "Approve"))
    assert len(await _jobs(pipe)) == 1
    await _sync(pipe)
    (row,) = pipe.sheets.rows(SHEET_ID)  # type: ignore[union-attr]
    assert row[11] == "approved" and row[6] == 70_000


async def test_a_countered_approval_syncs_the_countered_rate(pipe: Env) -> None:
    org = await _org(pipe)
    await pipe.add_rules(org, RATE_RULE)
    await pipe.set_benchmark(org)
    out = await pipe.voice(REP_CHAT, RATE_TRANSCRIPT, extraction_json(rate=7_000_000))
    out = await pipe.tap(REP_CHAT, button_data(to(REP_CHAT, out)[0], "Haan"))
    await pipe.tap(BOSS_CHAT, button_data(to(BOSS_CHAT, out)[0], "Counter"))
    out = await pipe.say(BOSS_CHAT, "74000")
    await pipe.tap(BOSS_CHAT, button_data(out[-1], "Haan"))
    await _sync(pipe)
    (row,) = pipe.sheets.rows(SHEET_ID)  # type: ignore[union-attr]
    assert row[11] == "approved (countered)" and row[6] == 74_000


async def test_rejected_and_expired_approvals_never_sync(pipe: Env) -> None:
    org = await _org(pipe)
    await pipe.add_rules(org, RATE_RULE)
    await pipe.set_benchmark(org)
    out = await pipe.voice(REP_CHAT, RATE_TRANSCRIPT, extraction_json(rate=7_000_000))
    out = await pipe.tap(REP_CHAT, button_data(to(REP_CHAT, out)[0], "Haan"))
    await pipe.tap(BOSS_CHAT, button_data(to(BOSS_CHAT, out)[0], "Reject"))
    out = await pipe.voice(
        REP_CHAT,
        RATE_TRANSCRIPT,
        extraction_json(rate=7_000_000),
    )
    await pipe.tap(REP_CHAT, button_data(to(REP_CHAT, out)[0], "Haan"))
    pipe.clock.advance(hours=24)
    await approvals.sweep(pipe.sf, load_pack(), 24, pipe.clock())
    assert await _jobs(pipe) == [] and (await _sync(pipe)).sent == 0
    assert pipe.sheets.rows(SHEET_ID) == []  # type: ignore[union-attr]
    statuses = sorted(a.status.value for a in await pipe.all(ApprovalRequest))
    assert statuses == ["expired", "rejected"]


async def test_discarded_visits_never_sync(pipe: Env) -> None:
    await _org(pipe)
    out = await pipe.voice(REP_CHAT, TRANSCRIPT, extraction_json())
    await pipe.tap(REP_CHAT, button_data(to(REP_CHAT, out)[0], "Nahi"))
    assert await _jobs(pipe) == [] and (await _sync(pipe)).sent == 0


async def test_a_stale_job_for_a_record_that_is_no_longer_confirmed_is_cancelled_not_sent(
    pipe: Env,
) -> None:
    await _org(pipe)
    await _confirmed_visit(pipe)
    async with pipe.sf() as s, s.begin():
        act = (await s.execute(select(ActivityLog))).scalar_one()
        act.status = ActivityStatus.REJECTED  # e.g. reversed after the job was queued
    report = await _sync(pipe)
    assert report.sent == 0 and report.cancelled == 1 and pipe.sheets.calls == 0  # type: ignore[union-attr]
    assert (await _jobs(pipe))[0].status is SyncStatus.CANCELLED
    assert any(a.action == "sync_cancelled" for a in await pipe.all(AuditLog))


# --- retry, backoff, give-up ---------------------------------------------------------------


async def test_failures_retry_with_exponential_backoff_then_succeed(pipe: Env) -> None:
    await _org(pipe)
    await _confirmed_visit(pipe)
    pipe.sheets.fail[:] = [SheetsError("quota"), SheetsError("quota")]  # type: ignore[union-attr]
    t0 = pipe.clock()
    r = await _sync(pipe)
    assert (r.sent, r.failed) == (0, 1)
    (job,) = await _jobs(pipe)
    assert (job.attempts, job.next_attempt_at, job.last_error) == (
        1,
        t0 + timedelta(seconds=30),
        "quota",
    )
    assert (await _sync(pipe)).failed == 0 and pipe.sheets.calls == 1  # type: ignore[union-attr]  # not due yet
    pipe.clock.advance(seconds=30)
    assert (await _sync(pipe)).failed == 1  # second failure
    assert (await _jobs(pipe))[0].next_attempt_at == pipe.clock() + timedelta(seconds=60)
    pipe.clock.advance(seconds=60)
    assert (await _sync(pipe)).sent == 1  # third attempt works
    (job,) = await _jobs(pipe)
    assert (job.status, job.attempts, job.last_error) == (SyncStatus.DONE, 3, None)
    assert len(pipe.sheets.rows(SHEET_ID)) == 1  # type: ignore[union-attr]


async def test_giving_up_marks_it_failed_and_tells_the_boss_exactly_once(pipe: Env) -> None:
    await _org(pipe)
    await _confirmed_visit(pipe)
    pipe.sheets.fail[:] = [SheetsError("spreadsheet not found or not shared", transient=False)] * 10  # type: ignore[union-attr]
    msgs = []
    for _ in range(3):
        report = await _sync(pipe, max_attempts=3)
        msgs += report.messages
        pipe.clock.advance(hours=2)
    (job,) = await _jobs(pipe)
    assert (
        job.status is SyncStatus.FAILED
        and job.attempts == 3
        and "not shared" in (job.last_error or "")
    )
    assert [m.chat_id for m in msgs] == [BOSS_CHAT] and "Sheet sync fail ho gaya" in msgs[0].text
    assert "Rajendra Jewellers" in msgs[0].text
    calls = pipe.sheets.calls  # type: ignore[union-attr]
    assert (await _sync(pipe, max_attempts=3)).messages == [] and pipe.sheets.calls == calls  # type: ignore[union-attr]
    (act,) = await pipe.all(ActivityLog)
    assert act.status is ActivityStatus.CONFIRMED  # still safely stored, simply not in the sheet
    assert any(a.action == "sync_failed" for a in await pipe.all(AuditLog))


async def test_an_unexpected_exception_is_a_failed_attempt_not_a_crash(pipe: Env) -> None:
    await _org(pipe)
    await _confirmed_visit(pipe)
    pipe.sheets.fail[:] = [RuntimeError("secret-token-should-not-leak")]  # type: ignore[union-attr]
    report = await _sync(pipe)
    assert report.failed == 1
    assert "secret-token-should-not-leak" not in ((await _jobs(pipe))[0].last_error or "")


async def test_one_bad_row_does_not_block_the_rest_of_the_batch(pipe: Env) -> None:
    await _org(pipe)
    await _confirmed_visit(pipe)
    await _confirmed_visit(pipe, TRANSCRIPT + " ")
    pipe.sheets.fail[:] = [SheetsError("boom")]  # type: ignore[union-attr]
    report = await _sync(pipe)
    assert (report.sent, report.failed) == (1, 1)


async def test_the_batch_size_is_respected(pipe: Env) -> None:
    await _org(pipe)
    for i in range(3):
        await _confirmed_visit(pipe, TRANSCRIPT + " " * i)
    assert (await _sync(pipe, batch=2)).sent == 2
    assert (await _sync(pipe, batch=2)).sent == 1


async def test_tenants_use_their_own_sheets(pipe: Env) -> None:
    await _org(pipe)
    await _confirmed_visit(pipe)
    _, _, raw = await pipe.new_tenant("Other Co", "Mallory")
    await pipe.say(3003, f"/start {raw}")
    await pipe.tap(3003, "co:y")
    await pipe.connect_sheet(
        3003, "https://docs.google.com/spreadsheets/d/OTHERSHEETIDxxxxxxxxxxxxxxxxxxxxxx/edit"
    )
    await _sync(pipe)
    assert len(pipe.sheets.rows(SHEET_ID)) == 1  # type: ignore[union-attr]
    assert pipe.sheets.rows("OTHERSHEETIDxxxxxxxxxxxxxxxxxxxxxx") == []  # type: ignore[union-attr]


# --- cell formatting and safety ------------------------------------------------------------


@pytest.mark.parametrize(
    ("paise", "cell"),
    [
        (None, ""),
        (0, 0),
        (100, 1),
        (4_500_000, 45_000),
        (41_200_000, 412_000),
        (150, "1.50"),
        (5, "0.05"),
        (12_345, "123.45"),
    ],
)
def test_amount_cells_are_exact_and_never_floats(paise: int | None, cell: Any) -> None:
    got = amount_cell(paise)
    assert got == cell and not isinstance(got, float)


@pytest.mark.parametrize(
    ("mg", "cell"),
    [(None, ""), (0, 0), (150_000, 150), (150_500, "150.5"), (250, "0.25"), (1, "0.001")],
)
def test_weight_cells(mg: int | None, cell: Any) -> None:
    got = weight_cell(mg)
    assert got == cell and not isinstance(got, float)


@pytest.mark.parametrize(
    ("raw", "safe"),
    [
        ('=HYPERLINK("http://evil")', '\'=HYPERLINK("http://evil")'),
        ("+1+1", "'+1+1"),
        ("-2", "'-2"),
        ("@SUM(A1)", "'@SUM(A1)"),
        ("\tx", "'\tx"),
        ("Rajendra", "Rajendra"),
        ("", ""),
        (None, ""),
    ],
)
def test_safe_text_neutralises_formulas(raw: str | None, safe: str) -> None:
    assert safe_text(raw) == safe


async def test_a_formula_like_client_name_is_stored_as_text(pipe: Env) -> None:
    await _org(pipe)
    await _confirmed_visit(pipe, client='=IMPORTXML("http://evil")')
    await _sync(pipe)
    row = pipe.sheets.rows(SHEET_ID)[0]  # type: ignore[union-attr]
    assert row[3].startswith("'=")


def test_backoff_doubles_and_caps() -> None:
    assert [backoff_seconds(n) for n in (1, 2, 3, 4, 8, 12, 50)] == [
        30,
        60,
        120,
        240,
        3600,
        3600,
        3600,
    ]


def test_status_labels() -> None:
    act = ActivityLog(payload={"visit": {}, "meta": {}})
    assert status_label(act) == "confirmed"
    act.payload = {"meta": {"approved": True}}
    assert status_label(act) == "approved"
    act.payload = {"meta": {"approved": True, "counter": {"kind": "rate"}}}
    assert status_label(act) == "approved (countered)"


# --- the gspread adapter (client stubbed) --------------------------------------------------


class _Resp:
    def __init__(self, code: int) -> None:
        self.status_code = code
        self.text = "x"

    def json(self) -> dict[str, Any]:
        return {"error": {"code": self.status_code, "message": "m", "status": "s"}}


class _WS:
    def __init__(self, ids: list[str] | None = None) -> None:
        self.ids = ids if ids is not None else []
        self.appended: list[list[Any]] = []
        self.updated: list[tuple[Any, str]] = []

    def col_values(self, col: int) -> list[str]:
        return list(self.ids)

    def append_row(self, row: list[Any]) -> None:
        self.appended.append(row)

    def update(self, values: Any, range_name: str) -> None:
        self.updated.append((values, range_name))


class _Sheet:
    def __init__(self, ws: _WS | None) -> None:
        self.ws = ws
        self.added: list[str] = []

    def worksheet(self, title: str) -> _WS:
        if self.ws is None:
            raise WorksheetNotFound(title)
        return self.ws

    def add_worksheet(self, title: str, rows: int, cols: int) -> _WS:
        self.added.append(title)
        self.ws = _WS()
        return self.ws


class _GC:
    def __init__(self, sheet: _Sheet | Exception) -> None:
        self.sheet = sheet

    def open_by_key(self, key: str) -> _Sheet:
        if isinstance(self.sheet, Exception):
            raise self.sheet
        return self.sheet


def _gs(sheet: _Sheet | Exception) -> GSpreadSheets:
    g = GSpreadSheets({"client_email": "svc@x.iam.gserviceaccount.com"})
    g._client = _GC(sheet)  # type: ignore[assignment]
    return g


async def test_gspread_appends_a_new_key_and_writes_the_header_to_an_empty_sheet() -> None:
    ws = _WS([])
    await _gs(_Sheet(ws)).upsert_row("sid", "Visits", "K1", ["K1", "x"])
    assert ws.appended == [HEADER, ["K1", "x"]] and ws.updated == []


async def test_gspread_updates_the_existing_row_in_place() -> None:
    ws = _WS(["Record ID", "K0", "K1"])
    await _gs(_Sheet(ws)).upsert_row("sid", "Visits", "K1", ["K1", "new"])
    assert ws.appended == [] and ws.updated == [([["K1", "new"]], "A3")]


async def test_gspread_appends_when_the_key_is_new() -> None:
    ws = _WS(["Record ID", "K0"])
    await _gs(_Sheet(ws)).upsert_row("sid", "Visits", "K9", ["K9"])
    assert ws.appended == [["K9"]] and ws.updated == []


async def test_gspread_creates_a_missing_worksheet() -> None:
    sheet = _Sheet(None)
    await _gs(sheet).upsert_row("sid", "Visits", "K1", ["K1"])
    assert sheet.added == ["Visits"] and sheet.ws is not None and sheet.ws.appended[-1] == ["K1"]


@pytest.mark.parametrize(
    ("code", "transient"), [(429, True), (500, True), (503, True), (403, False), (404, False)]
)
async def test_gspread_api_errors_are_classified(code: int, transient: bool) -> None:
    with pytest.raises(SheetsError) as exc:
        await _gs(APIError(_Resp(code))).upsert_row("sid", "Visits", "K", ["K"])  # type: ignore[arg-type]
    assert exc.value.transient is transient and str(code) in str(exc.value)


async def test_gspread_missing_or_unshared_spreadsheet_is_not_transient() -> None:
    with pytest.raises(SheetsError) as exc:
        await _gs(SpreadsheetNotFound()).upsert_row("sid", "Visits", "K", ["K"])
    assert exc.value.transient is False and "not shared" in str(exc.value)
    assert await _gs(SpreadsheetNotFound()).check_access("sid") is False
    assert await _gs(_Sheet(_WS())).check_access("sid") is True


def test_the_service_account_email_comes_from_the_credentials() -> None:
    assert _gs(_Sheet(_WS())).service_account_email == "svc@x.iam.gserviceaccount.com"
