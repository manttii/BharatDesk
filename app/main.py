"""FastAPI app factory. `uvicorn app.main:app` boots with or without Telegram, Groq or Gemini."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import timedelta
from pathlib import Path

import structlog
from aiogram import Bot
from fastapi import FastAPI
from fastapi.responses import JSONResponse

from app.bot import webhook
from app.bot.polling import ALLOWED_UPDATES, run_polling
from app.bot.processor import EventProcessor
from app.config import Settings, get_settings
from app.db import make_engine, make_session_factory, migrate_to_head, ping
from app.logging_config import configure_logging
from app.models.base import utcnow
from app.pack import load_pack
from app.pipeline.media import MediaStore
from app.providers.llm import LLMProvider, build_llm
from app.providers.messaging import FakeProvider, MessagingProvider, TelegramProvider
from app.providers.sheets import GSpreadSheets, SheetsProvider
from app.providers.stt import GroqSTT, STTProvider
from app.queue import InProcessQueue
from app.schemas.events import IncomingEvent
from app.security import RateLimiter
from app.services import approvals, bill_sync, daily_report, recovery, retention, sheets_sync
from app.web import dashboard, sandbox

log = structlog.get_logger()
VERSION = "0.4.0"


def create_app(
    settings: Settings | None = None,
    messaging: MessagingProvider | None = None,
    stt: STTProvider | None = None,
    llm: LLMProvider | None = None,
    sheets: SheetsProvider | None = None,
) -> FastAPI:
    cfg = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        configure_logging()
        if cfg.auto_migrate:
            await migrate_to_head(cfg.database_url)
        engine = make_engine(cfg.database_url)
        sf = make_session_factory(engine)
        pack = load_pack()

        speech = stt
        if speech is None and cfg.groq_api_key and cfg.whisper_model:
            speech = GroqSTT(
                cfg.groq_api_key.get_secret_value(),
                cfg.whisper_model,
                force_convert=cfg.stt_force_convert,
            )
        model = llm
        if model is None:
            model = build_llm(cfg)
        grid = sheets
        if grid is None and (cfg.google_service_account_json or cfg.google_service_account_file):
            raw = (
                cfg.google_service_account_json.get_secret_value()
                if cfg.google_service_account_json
                else await asyncio.to_thread(
                    Path(str(cfg.google_service_account_file)).read_text, encoding="utf-8"
                )
            )
            grid = GSpreadSheets(json.loads(raw))
        if model is not None:
            await model.ensure_available()  # a wrong/unavailable model fails HERE, loudly
        log.info("providers", stt=bool(speech), llm=bool(model))

        bot: Bot | None = None
        provider = messaging
        telegram_state = "disabled"
        if provider is None and cfg.telegram_bot_token is not None:
            bot = Bot(token=cfg.telegram_bot_token.get_secret_value())
            me = await bot.get_me()
            provider = TelegramProvider(bot, me.username or cfg.telegram_bot_username or "")
            telegram_state = cfg.telegram_mode
        if provider is None:
            provider = FakeProvider()
            log.warning("telegram_disabled", reason="no TELEGRAM_BOT_TOKEN; using FakeProvider")

        processor = EventProcessor(
            sf, provider, cfg, pack, RateLimiter(cfg.rate_limit_per_minute),
            stt=speech, llm=model, media=MediaStore(cfg.media_dir), sheets=grid,
        )  # fmt: skip
        queue: InProcessQueue[IncomingEvent] = InProcessQueue(processor.process)
        queue.start()

        async def sweep_once() -> int:
            now = utcnow()
            msgs = await approvals.sweep(sf, pack, cfg.approval_ttl_hours, now)
            msgs += await daily_report.send_due(sf, pack, now)
            msgs += await recovery.recover_stranded(
                sf, pack, now, older_than=timedelta(minutes=cfg.stranded_after_minutes)
            )
            for m in msgs:
                try:
                    await provider.send(m)
                except Exception:
                    log.exception("sweep_send_failed", chat_id=m.chat_id)
            return len(msgs)

        async def sweep_loop() -> None:
            while True:
                try:
                    await sweep_once()
                except Exception:
                    log.exception("sweep_failed")
                await asyncio.sleep(cfg.expiry_sweep_seconds)

        sweep_task = (
            asyncio.create_task(sweep_loop(), name="approval-sweep")
            if cfg.background_tasks
            else None
        )

        async def sync_once() -> int:
            """Drain the Sheets outbox. Runs in the background, never in a reply path."""
            if grid is None:
                return 0
            report = await sheets_sync.sync_once(
                sf, grid, pack, now=utcnow(), max_attempts=cfg.sync_max_attempts,
                batch_size=cfg.sync_batch_size,
            )  # fmt: skip
            for m in report.messages:
                try:
                    await provider.send(m)
                except Exception:
                    log.exception("sync_notice_failed", chat_id=m.chat_id)
            bills = await bill_sync.sync_bills(sf, grid, now=utcnow())
            return report.sent + bills.bills_sent

        async def sync_loop() -> None:
            while True:
                try:
                    await sync_once()
                except Exception:
                    log.exception("sync_failed")
                await asyncio.sleep(cfg.sync_interval_seconds)

        sync_task = (
            asyncio.create_task(sync_loop(), name="sheets-sync")
            if grid and cfg.background_tasks
            else None
        )

        async def purge_once() -> int:
            """Delete stored audio/photos (and, if a tenant opted in, transcripts) past retention."""
            now = utcnow()
            report = await retention.purge_media(sf, MediaStore(cfg.media_dir), now)
            scrubbed = await retention.purge_transcripts(sf, now)
            if report.messages_purged or scrubbed:
                log.info(
                    "retention_purge",
                    media=report.messages_purged,
                    files=report.files_deleted,
                    transcripts=scrubbed,
                )
            return report.messages_purged + scrubbed

        async def purge_loop() -> None:
            while True:
                try:
                    await purge_once()
                except Exception:
                    log.exception("purge_failed")
                await asyncio.sleep(cfg.retention_sweep_hours * 3600)

        purge_task = (
            asyncio.create_task(purge_loop(), name="media-purge") if cfg.background_tasks else None
        )

        polling_task: asyncio.Task[None] | None = None
        if bot is not None:
            if cfg.telegram_mode == "webhook":
                assert cfg.telegram_webhook_secret is not None and cfg.telegram_webhook_url
                await bot.set_webhook(
                    url=cfg.telegram_webhook_url,
                    secret_token=cfg.telegram_webhook_secret.get_secret_value(),
                    allowed_updates=ALLOWED_UPDATES,
                )
            else:
                await bot.delete_webhook()  # polling and a webhook are mutually exclusive
                polling_task = asyncio.create_task(run_polling(bot, queue), name="tg-polling")

        app.state.settings = cfg
        app.state.engine = engine
        app.state.session_factory = sf
        app.state.queue = queue
        app.state.telegram = telegram_state
        app.state.sweep_once = sweep_once
        app.state.sync_once = sync_once
        app.state.purge_once = purge_once
        app.state.background_tasks = [t for t in (sweep_task, sync_task, purge_task) if t]
        app.state.pack = pack
        app.state.clock = None  # tests may set a callable returning the current UTC time
        app.state.providers = {"stt": bool(speech), "llm": bool(model), "sheets": bool(grid)}
        app.state.sandbox = (
            sandbox.SandboxRunner(sf, cfg, pack, speech, model) if cfg.sandbox_enabled else None
        )
        try:
            yield
        finally:
            for task in (polling_task, sweep_task, sync_task, purge_task):
                if task is not None:
                    task.cancel()
                    await asyncio.gather(task, return_exceptions=True)
            await queue.stop()
            if bot is not None:
                await bot.session.close()
            await engine.dispose()

    app = FastAPI(title="BharatDesk", version=VERSION, lifespan=lifespan)
    app.include_router(webhook.router)
    if cfg.sandbox_enabled:
        app.include_router(sandbox.router)
    if cfg.dashboard_secret is not None:
        app.include_router(dashboard.router)

    @app.get("/health")
    async def health() -> JSONResponse:
        db_ok = await ping(app.state.engine)
        body = {
            "status": "ok" if db_ok else "degraded",
            "db": "ok" if db_ok else "unreachable",
            "telegram": app.state.telegram,
            "providers": app.state.providers,
            "version": VERSION,
        }
        return JSONResponse(body, status_code=200 if db_ok else 503)

    return app


app = create_app()
