"""Long polling for local development (webhook mode is for deployment)."""

from __future__ import annotations

import asyncio

import structlog
from aiogram import Bot
from aiogram.exceptions import TelegramAPIError

from app.providers.messaging import parse_update
from app.queue import Queue
from app.schemas.events import IncomingEvent

log = structlog.get_logger()

ALLOWED_UPDATES = ["message", "callback_query"]
POLL_TIMEOUT_S = 30


async def run_polling(bot: Bot, queue: Queue[IncomingEvent]) -> None:
    offset: int | None = None
    backoff = 1.0
    while True:
        try:
            updates = await bot.get_updates(
                offset=offset,
                timeout=POLL_TIMEOUT_S,
                allowed_updates=ALLOWED_UPDATES,
                request_timeout=POLL_TIMEOUT_S + 15,
            )
            backoff = 1.0
        except asyncio.CancelledError:
            raise
        except (TelegramAPIError, OSError, TimeoutError) as exc:
            log.warning("polling_error", error=type(exc).__name__, retry_in=backoff)
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 30.0)
            continue
        for update in updates:
            offset = update.update_id + 1
            event = parse_update(update)
            if event is not None:
                await queue.put(event)
