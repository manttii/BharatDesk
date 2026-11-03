"""The boss dashboard on its own, for serverless hosting (Vercel).

This is only the web dashboard. The Telegram bot, the model calls and the background loops
(approval sweep, Sheets sync, retention purge) keep running on a normal server, because they
need a long-lived process. Both sides talk to the SAME hosted Postgres database, and the
dashboard authenticates with the same signed links, so it needs the same DASHBOARD_SECRET.

Serverless rules this respects: no lifespan hooks or startup work, no background tasks, no
local files (so no SQLite, no migrations, no stored audio/photos), and no connection pool.
"""

from __future__ import annotations

import os

from fastapi import FastAPI
from fastapi.responses import PlainTextResponse

from app.config import Settings, get_settings
from app.db import make_engine, make_session_factory, ping
from app.pack import load_pack
from app.web import dashboard

VERSION = "0.4.0"


class ConfigError(RuntimeError):
    """Raised at import time so a bad deployment fails loudly instead of serving errors."""


def _check(cfg: Settings) -> None:
    on_vercel = bool(os.environ.get("VERCEL"))
    if cfg.dashboard_secret is None:
        raise ConfigError("DASHBOARD_SECRET is not set (use the same value as the bot)")
    if on_vercel and cfg.database_url.startswith("sqlite"):
        raise ConfigError(
            "DATABASE_URL is not set to a hosted Postgres database. Vercel has no persistent "
            "disk, so SQLite cannot be used there."
        )


def create_dashboard_app(settings: Settings | None = None) -> FastAPI:
    cfg = settings or get_settings()
    if cfg.public_base_url is None and os.environ.get("VERCEL_PROJECT_PRODUCTION_URL"):
        # Vercel provides the production hostname; cookies are marked Secure when this is https.
        cfg = cfg.model_copy(
            update={"public_base_url": f"https://{os.environ['VERCEL_PROJECT_PRODUCTION_URL']}"}
        )
    _check(cfg)

    app = FastAPI(
        title="BharatDesk dashboard",
        version=VERSION,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,  # no public listing of routes
    )
    engine = make_engine(cfg.database_url, serverless=True)
    app.state.settings = cfg
    app.state.engine = engine
    app.state.session_factory = make_session_factory(engine)
    app.state.pack = load_pack()
    app.state.clock = None  # real time; tests may inject a callable
    app.include_router(dashboard.router)

    @app.get("/", response_class=PlainTextResponse)
    async def root() -> str:
        return "BharatDesk dashboard. Ask your bot for /dashboard to get your sign-in link."

    @app.get("/health")
    async def health() -> dict[str, object]:
        return {"status": "ok" if await ping(engine) else "degraded", "version": VERSION}

    return app
