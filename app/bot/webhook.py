"""Telegram webhook endpoint. Authenticated by the secret-token header Telegram echoes
back on every call (set via setWebhook secret_token); compared in constant time."""

from __future__ import annotations

import asyncio
from typing import Annotated

from aiogram.types import Update
from fastapi import APIRouter, Header, HTTPException, Request
from pydantic import ValidationError

from app.providers.messaging import parse_update
from app.security import secrets_match

router = APIRouter()


@router.post("/telegram/webhook")
async def telegram_webhook(
    request: Request,
    x_telegram_bot_api_secret_token: Annotated[str | None, Header()] = None,
) -> dict[str, bool]:
    state = request.app.state
    secret = state.settings.telegram_webhook_secret
    if secret is None:  # webhook mode not configured: endpoint does not exist
        raise HTTPException(status_code=404)
    if not secrets_match(x_telegram_bot_api_secret_token, secret.get_secret_value()):
        raise HTTPException(status_code=403)
    try:
        update = Update.model_validate(await request.json())
    except (ValueError, ValidationError) as exc:
        raise HTTPException(status_code=400, detail="invalid update") from exc
    event = parse_update(update)
    if event is not None:
        try:
            await state.queue.put(event)
        except asyncio.QueueFull as exc:
            raise HTTPException(status_code=503) from exc  # Telegram will retry
    return {"ok": True}
