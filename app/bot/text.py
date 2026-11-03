"""Messages that are neither commands nor notes (photos and other media, for now)."""

from __future__ import annotations

from app.bot.context import Ctx


async def handle_other(ctx: Ctx) -> None:
    ctx.say(ctx.t("unsupported"))
