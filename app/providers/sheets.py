"""Google Sheets boundary: an interface, a gspread adapter (service account), and a Fake.

Verified against the installed gspread 6.2.1: `service_account_from_dict`, `open_by_key`,
`worksheet`/`add_worksheet`, `col_values`, `update(values, range_name)`, and `append_row`,
whose default value_input_option is RAW (cells are stored literally, so a rep's text can
never become a formula). gspread is synchronous, so every call runs in a worker thread.

Rows are keyed by the record id in column A: a retried send UPDATES the existing row
instead of appending a duplicate.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from typing import Any

import gspread
import structlog
from gspread.exceptions import APIError, GSpreadException, SpreadsheetNotFound, WorksheetNotFound

log = structlog.get_logger()

HEADER: list[str] = [
    "Record ID", "Visit date", "Rep", "Client", "Weight (g)", "Order value (INR)",
    "Rate per 10 g (INR)", "Advance (INR)", "Pending (INR)", "Delivery date",
    "Follow-up date", "Status", "Source", "Notes", "Updated (UTC)",
]  # fmt: skip
SHEET_COLS = len(HEADER)


class SheetsError(Exception):
    """`transient` errors (quota, 5xx, network) are retried; others are retried more slowly
    and finally reported, because they usually mean the sheet was not shared."""

    def __init__(self, message: str, *, transient: bool = True) -> None:
        super().__init__(message)
        self.transient = transient


class SheetsProvider(ABC):
    service_account_email: str

    @abstractmethod
    async def upsert_row(
        self, spreadsheet_id: str, worksheet: str, key: str, row: list[Any]
    ) -> None: ...

    @abstractmethod
    async def upsert_rows(
        self, spreadsheet_id: str, worksheet: str, header: list[str], rows: dict[str, list[Any]]
    ) -> None:
        """Many keyed rows in a few API calls: existing keys are updated, new ones appended."""

    @abstractmethod
    async def replace_table(
        self, spreadsheet_id: str, worksheet: str, header: list[str], rows: list[list[Any]]
    ) -> None:
        """Rewrite a whole derived tab (e.g. stock) with the header and these rows."""

    @abstractmethod
    async def check_access(self, spreadsheet_id: str) -> bool: ...


class GSpreadSheets(SheetsProvider):
    def __init__(self, service_account_info: dict[str, Any]) -> None:
        self._info = service_account_info
        self.service_account_email = str(service_account_info.get("client_email", ""))
        self._client: gspread.Client | None = None

    def _gc(self) -> gspread.Client:
        if self._client is None:
            self._client = gspread.service_account_from_dict(self._info)
        return self._client

    async def check_access(self, spreadsheet_id: str) -> bool:
        def run() -> bool:
            try:
                self._gc().open_by_key(spreadsheet_id)
            except (SpreadsheetNotFound, APIError, GSpreadException):
                return False
            return True

        return await asyncio.to_thread(run)

    async def upsert_row(
        self, spreadsheet_id: str, worksheet: str, key: str, row: list[Any]
    ) -> None:
        try:
            await asyncio.to_thread(self._upsert, spreadsheet_id, worksheet, key, row)
        except APIError as exc:
            status = getattr(getattr(exc, "response", None), "status_code", 0)
            raise SheetsError(
                f"sheets API error HTTP {status}", transient=status in (429, 500, 502, 503, 504)
            ) from exc
        except SpreadsheetNotFound as exc:
            raise SheetsError("spreadsheet not found or not shared", transient=False) from exc
        except GSpreadException as exc:
            raise SheetsError(type(exc).__name__, transient=False) from exc
        except OSError as exc:
            raise SheetsError(f"network error: {type(exc).__name__}") from exc

    async def upsert_rows(
        self, spreadsheet_id: str, worksheet: str, header: list[str], rows: dict[str, list[Any]]
    ) -> None:
        await self._guarded(self._upsert_rows, spreadsheet_id, worksheet, header, rows)

    async def replace_table(
        self, spreadsheet_id: str, worksheet: str, header: list[str], rows: list[list[Any]]
    ) -> None:
        await self._guarded(self._replace_table, spreadsheet_id, worksheet, header, rows)

    async def _guarded(self, fn: Any, *args: Any) -> None:
        try:
            await asyncio.to_thread(fn, *args)
        except APIError as exc:
            status = getattr(getattr(exc, "response", None), "status_code", 0)
            raise SheetsError(
                f"sheets API error HTTP {status}", transient=status in (429, 500, 502, 503, 504)
            ) from exc
        except SpreadsheetNotFound as exc:
            raise SheetsError("spreadsheet not found or not shared", transient=False) from exc
        except GSpreadException as exc:
            raise SheetsError(type(exc).__name__, transient=False) from exc
        except OSError as exc:
            raise SheetsError(f"network error: {type(exc).__name__}") from exc

    def _worksheet(self, spreadsheet_id: str, title: str, cols: int) -> gspread.Worksheet:
        sheet = self._gc().open_by_key(spreadsheet_id)
        try:
            return sheet.worksheet(title)
        except WorksheetNotFound:
            return sheet.add_worksheet(title=title, rows=1000, cols=max(cols, 1))

    def _upsert_rows(
        self, spreadsheet_id: str, title: str, header: list[str], rows: dict[str, list[Any]]
    ) -> None:
        ws = self._worksheet(spreadsheet_id, title, len(header))
        ids = ws.col_values(1)  # ONE read for the whole batch
        updates: list[dict[str, Any]] = []
        if not ids:
            ws.append_row(header)
            ids = [header[0]]
        elif ws.row_values(1) != header:
            updates.append({"range": "A1", "values": [header]})  # e.g. a column was added
        new_rows: list[list[Any]] = []
        for key, row in rows.items():
            if key in ids:
                updates.append({"range": f"A{ids.index(key) + 1}", "values": [row]})
            else:
                new_rows.append(row)
        if updates:
            ws.batch_update(updates)  # RAW: cells stay literal, never formulas
        if new_rows:
            ws.append_rows(new_rows)  # RAW by default

    def _replace_table(
        self, spreadsheet_id: str, title: str, header: list[str], rows: list[list[Any]]
    ) -> None:
        ws = self._worksheet(spreadsheet_id, title, len(header))
        ws.clear()
        ws.update([header, *rows], "A1", raw=True)

    def _upsert(self, spreadsheet_id: str, title: str, key: str, row: list[Any]) -> None:
        sheet = self._gc().open_by_key(spreadsheet_id)
        try:
            ws = sheet.worksheet(title)
        except WorksheetNotFound:
            ws = sheet.add_worksheet(title=title, rows=1000, cols=SHEET_COLS)
        ids = ws.col_values(1)
        if not ids:
            ws.append_row(HEADER)
            ids = [HEADER[0]]
        if key in ids:
            ws.update([row], f"A{ids.index(key) + 1}")  # same record: update in place
        else:
            ws.append_row(row)


class FakeSheets(SheetsProvider):
    """In-memory sheet for tests. `fail` is a queue of exceptions raised by upsert_row."""

    def __init__(self, email: str = "bharatdesk@example.iam.gserviceaccount.com") -> None:
        self.service_account_email = email
        self.sheets: dict[tuple[str, str], dict[str, list[Any]]] = {}
        self.fail: list[Exception] = []
        self.calls = 0
        self.accessible = True
        self.headers: dict[tuple[str, str], list[str]] = {}

    async def upsert_row(
        self, spreadsheet_id: str, worksheet: str, key: str, row: list[Any]
    ) -> None:
        self.calls += 1
        if self.fail:
            raise self.fail.pop(0)
        self.sheets.setdefault((spreadsheet_id, worksheet), {})[key] = list(row)

    async def upsert_rows(
        self, spreadsheet_id: str, worksheet: str, header: list[str], rows: dict[str, list[Any]]
    ) -> None:
        self.calls += 1
        if self.fail:
            raise self.fail.pop(0)
        self.headers[(spreadsheet_id, worksheet)] = list(header)
        self.sheets.setdefault((spreadsheet_id, worksheet), {}).update(
            {k: list(v) for k, v in rows.items()}
        )

    async def replace_table(
        self, spreadsheet_id: str, worksheet: str, header: list[str], rows: list[list[Any]]
    ) -> None:
        self.calls += 1
        if self.fail:
            raise self.fail.pop(0)
        self.headers[(spreadsheet_id, worksheet)] = list(header)
        self.sheets[(spreadsheet_id, worksheet)] = {str(i): list(r) for i, r in enumerate(rows)}

    async def check_access(self, spreadsheet_id: str) -> bool:
        return self.accessible

    def rows(self, spreadsheet_id: str, worksheet: str = "Visits") -> list[list[Any]]:
        return list(self.sheets.get((spreadsheet_id, worksheet), {}).values())
