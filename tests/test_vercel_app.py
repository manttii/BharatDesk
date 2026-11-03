"""The dashboard-only app for Vercel, Postgres URL handling, and the deployment files."""

from __future__ import annotations

import importlib
import json
import re
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr
from sqlalchemy.pool import NullPool

from app.config import Settings, get_settings
from app.db import make_engine, normalize_database_url
from app.web.vercel_app import ConfigError, create_dashboard_app
from tests.conftest import Env
from tests.test_dashboard import BASE, SECRET, _boss_session, _org, _visit

ROOT = Path(__file__).resolve().parent.parent


# --- database URLs --------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("given", "expected"),
    [
        (
            "postgres://u:p@ep-x.neon.tech/db?sslmode=require&channel_binding=require",
            "postgresql+asyncpg://u:p@ep-x.neon.tech/db?ssl=require",
        ),
        ("postgresql://u:p@h:5432/db", "postgresql+asyncpg://u:p@h:5432/db"),
        ("postgresql+asyncpg://u:p@h/db?ssl=require", "postgresql+asyncpg://u:p@h/db?ssl=require"),
        (
            "postgres://u:p%40x@h/db?sslmode=verify-full",
            "postgresql+asyncpg://u:p%40x@h/db?ssl=verify-full",
        ),
        ("sqlite+aiosqlite:///./x.db", "sqlite+aiosqlite:///./x.db"),
    ],
)
def test_provider_urls_are_made_to_work_with_asyncpg(given: str, expected: str) -> None:
    assert normalize_database_url(given) == expected


def test_serverless_engines_have_no_pool_and_no_statement_cache() -> None:
    e = make_engine("postgres://u:p@h/db?sslmode=require", serverless=True)
    assert e.dialect.name == "postgresql" and isinstance(e.pool, NullPool)
    pooled = make_engine("postgresql://u:p@h/db")
    assert not isinstance(pooled.pool, NullPool)
    assert make_engine("sqlite+aiosqlite:///:memory:", serverless=True).dialect.name == "sqlite"


# --- the app --------------------------------------------------------------------------------------


def _settings(p: Env, **kw: Any) -> Settings:
    return Settings(  # type: ignore[call-arg]
        _env_file=None, database_url=p.settings.database_url, auto_migrate=False,
        dashboard_secret=SECRET, public_base_url=BASE, **kw,
    )  # fmt: skip


@pytest.fixture
def web(pipe: Env) -> Iterator[tuple[Env, TestClient]]:
    pipe.settings.dashboard_secret = SecretStr(SECRET)
    pipe.settings.public_base_url = BASE
    app = create_dashboard_app(_settings(pipe))
    app.state.clock = pipe.clock
    with TestClient(app, follow_redirects=False) as client:
        yield pipe, client


async def test_the_dashboard_app_serves_the_real_dashboard_without_any_background_work(
    web: tuple[Env, TestClient],
) -> None:
    p, client = web
    assert client.get("/").text.startswith("BharatDesk dashboard")
    assert client.get("/health").json()["status"] == "ok"
    assert client.get("/docs").status_code == 404  # no API docs page exposed
    assert not hasattr(client.app.state, "background_tasks")  # nothing runs in the background
    await _org(p)
    await _visit(p)
    await _boss_session(p, client)  # the bot's signed link works against this app
    page = client.get("/dashboard")
    assert page.status_code == 200 and "Ramesh" in client.get("/dashboard/partials/summary").text


async def test_the_dashboard_needs_the_signed_link_and_nothing_else_leaks(
    web: tuple[Env, TestClient],
) -> None:
    _, client = web
    assert client.get("/dashboard").status_code in (303, 401, 403)
    assert client.get("/dashboard/partials/summary").status_code in (303, 401, 403)
    assert client.get("/dashboard?t=not-a-real-token").status_code in (303, 401, 403)
    assert "sqlite" not in client.get("/health").text.lower()  # no connection details


async def test_an_https_deployment_marks_the_session_cookie_secure(pipe: Env) -> None:
    pipe.settings.dashboard_secret = SecretStr(SECRET)
    pipe.settings.public_base_url = "https://desk.example"
    await pipe.org()
    out = await pipe.say(1001, "/dashboard")
    link = re.search(r"https://desk\.example(/dashboard\?t=\S+)", out[0].text)
    assert link
    app = create_dashboard_app(
        _settings(pipe).model_copy(update={"public_base_url": "https://desk.example"})
    )
    app.state.clock = pipe.clock
    with TestClient(app, follow_redirects=False) as client:
        r = client.get(link.group(1))
    cookie = r.headers["set-cookie"].lower()
    assert "httponly" in cookie and "secure" in cookie and "samesite=strict" in cookie


def test_vercel_hostname_becomes_the_https_base_url(
    pipe: Env, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("VERCEL_PROJECT_PRODUCTION_URL", "bharatdesk.vercel.app")
    app = create_dashboard_app(_settings(pipe).model_copy(update={"public_base_url": None}))
    assert app.state.settings.public_base_url == "https://bharatdesk.vercel.app"


def test_a_bad_deployment_fails_loudly(pipe: Env, monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(ConfigError, match="DASHBOARD_SECRET"):
        create_dashboard_app(_settings(pipe).model_copy(update={"dashboard_secret": None}))
    monkeypatch.setenv("VERCEL", "1")
    with pytest.raises(ConfigError, match="hosted Postgres"):
        create_dashboard_app(_settings(pipe))  # still pointing at SQLite
    ok = _settings(pipe).model_copy(update={"database_url": "postgres://u:p@h/db?sslmode=require"})
    assert create_dashboard_app(ok).state.engine.dialect.name == "postgresql"


def test_the_vercel_entry_point_exposes_an_asgi_app(
    pipe: Env, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DATABASE_URL", pipe.settings.database_url)
    monkeypatch.setenv("DASHBOARD_SECRET", SECRET)
    get_settings.cache_clear()
    sys.modules.pop("index", None)
    try:
        module = importlib.import_module("index")
        assert module.app.title == "BharatDesk dashboard"
    finally:
        get_settings.cache_clear()
        sys.modules.pop("index", None)


# --- deployment files -----------------------------------------------------------------------------


def test_the_dashboard_is_the_entry_point_vercel_finds_first() -> None:
    cfg = json.loads((ROOT / "vercel.json").read_text(encoding="utf-8"))
    assert cfg["framework"] == "fastapi"
    entry = (ROOT / "index.py").read_text(encoding="utf-8")
    assert "create_dashboard_app" in entry
    # Vercel's detection must not stumble onto the bot's app (app/main.py) or a stale function.
    assert not (ROOT / "api").exists()
    assert (ROOT / "app" / "main.py").is_file()  # the bot lives there, so index.py must win


def test_secrets_and_local_data_can_never_be_uploaded_to_vercel() -> None:
    ignored = (ROOT / ".vercelignore").read_text(encoding="utf-8").splitlines()
    for must in (".env", ".env.*", "*.db", "media/", ".venv/", "*service-account*.json"):
        assert must in ignored, must
    assert "!.env.example" in ignored


def test_the_postgres_driver_is_pinned_for_the_deployment() -> None:
    reqs = (ROOT / "requirements.txt").read_text(encoding="utf-8")
    assert re.search(r"^asyncpg==\d", reqs, re.M)
