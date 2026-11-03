"""Background loops (sweep, sheet sync, retention) can be switched off, e.g. for fake-clock tests.

Why it exists: the loops read the REAL clock. A test whose data is dated by a fake clock in the
past would have its approvals expired by a loop tick, so the dashboard test failed now and then
(and reliably once real time passed the fake approval's expiry)."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from app.providers.messaging import FakeProvider


def _app(tmp_path: Path, **kw: object) -> TestClient:
    db = tmp_path / "bg.db"
    settings = Settings(  # type: ignore[call-arg]
        _env_file=None, database_url=f"sqlite+aiosqlite:///{db.as_posix()}", sandbox_enabled=False,
        media_dir=str(tmp_path / "media"), **kw,
    )  # fmt: skip
    return TestClient(create_app(settings, messaging=FakeProvider()))


def test_loops_run_by_default(tmp_path: Path) -> None:
    with _app(tmp_path) as client:
        names = {t.get_name() for t in client.app.state.background_tasks}  # type: ignore[attr-defined]
        assert {"approval-sweep", "media-purge"} <= names


def test_loops_can_be_switched_off(tmp_path: Path) -> None:
    with _app(tmp_path, background_tasks=False) as client:
        assert client.app.state.background_tasks == []  # type: ignore[attr-defined]
        assert client.get("/health").status_code == 200  # the app itself still works
