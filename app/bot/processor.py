"""Event pipeline: dedupe -> rate-limit -> identity -> consent gate -> handler.

One transaction per event. Replies are sent only after commit. A failure rolls the
transaction back (including the dedupe claim), so a retried delivery is processed
again rather than silently lost.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from typing import Any

import structlog
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.bot import bills_ui, callbacks, commands, notes, text
from app.bot.context import Ctx, Deferred
from app.bot.deps import Deps
from app.config import Settings
from app.db import claim_event
from app.models.base import utcnow
from app.pack import Pack
from app.pipeline.media import MediaStore
from app.providers.llm import LLMProvider
from app.providers.messaging import MessagingProvider
from app.providers.sheets import SheetsProvider
from app.providers.stt import STTProvider
from app.schemas.events import Button, IncomingEvent, OutMessage
from app.security import RateLimiter
from app.services import consent
from app.services.identity import resolve_user
from app.services.onboarding import RedeemStatus, redeem_invite

log = structlog.get_logger()


class EventProcessor:
    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        messaging: MessagingProvider,
        settings: Settings,
        pack: Pack,
        limiter: RateLimiter,
        clock: Callable[[], datetime] = utcnow,
        *,
        stt: STTProvider | None = None,
        llm: LLMProvider | None = None,
        media: MediaStore | None = None,
        sheets: SheetsProvider | None = None,
        trace: list[dict[str, Any]] | None = None,
    ) -> None:
        self._sf = session_factory
        self._messaging = messaging
        self._settings = settings
        self._pack = pack
        self._limiter = limiter
        self._clock = clock
        self.deps = Deps(
            session_factory=session_factory,
            messaging=messaging,
            settings=settings,
            pack=pack,
            stt=stt,
            llm=llm,
            media=media,
            sheets=sheets,
            clock=clock,
            trace=trace,
        )

    async def process(self, event: IncomingEvent) -> None:
        replies: list[OutMessage] = []
        callback_text: str | None = None
        deferred: list[Deferred] = []
        try:
            async with self._sf() as session, session.begin():
                ctx = Ctx(
                    session=session,
                    settings=self._settings,
                    pack=self._pack,
                    event=event,
                    now=self._clock(),
                    bot_username=self._messaging.username,
                    deps=self.deps,
                )
                await self._route(ctx)
                replies, callback_text, deferred = ctx.replies, ctx.callback_text, ctx.deferred
        except Exception:
            log.exception("event_failed", update_id=event.update_id, chat_id=event.chat_id)
            replies = [OutMessage(event.chat_id, self._pack.t("error_generic"))]
            callback_text, deferred = None, []
        await self._deliver(event, replies, callback_text)
        # Slow work (download, STT, LLM) runs only after the transaction above is committed
        # and released, so no database lock is held across a network call.
        for job in deferred:
            try:
                more = await job()
            except Exception:
                log.exception("deferred_job_failed", update_id=event.update_id)
                more = [OutMessage(event.chat_id, self._pack.t("error_generic"))]
            await self._deliver(event, more, None, answer_callback=False)

    async def _deliver(
        self,
        event: IncomingEvent,
        replies: list[OutMessage],
        callback_text: str | None,
        *,
        answer_callback: bool = True,
    ) -> None:
        # Always answer a callback so the button's spinner stops, even for duplicates.
        if answer_callback and event.callback_id is not None:
            await self._messaging.answer_callback(event.callback_id, callback_text)
        for msg in replies:
            try:
                await self._messaging.send(msg)
            except Exception:
                log.exception("send_failed", chat_id=msg.chat_id)

    async def _route(self, ctx: Ctx) -> None:
        ev = ctx.event
        if not await claim_event(ctx.session, f"u:{ev.update_id}"):
            log.info("duplicate_update", update_id=ev.update_id)
            return
        if ev.callback_id is not None and not await claim_event(
            ctx.session, f"cb:{ev.callback_id}"
        ):
            log.info("duplicate_callback", chat_id=ev.chat_id)
            return
        if not self._limiter.allow(ev.chat_id):
            log.warning("rate_limited", chat_id=ev.chat_id)
            ctx.say(ctx.t("rate_limited"))
            return

        cmd = commands.parse_command(ev.text)

        # Deep-link onboarding is the only path open to unregistered chats.
        if cmd is not None and cmd.name == "start" and cmd.arg:
            await self._start_with_token(ctx, cmd.arg)
            return

        ctx.user = await resolve_user(ctx.session, ev.chat_id)
        user = ctx.user
        if user is None:
            ctx.say(ctx.t("not_registered"))
            return
        if not user.is_active:
            ctx.say(ctx.t("account_inactive"))
            return

        if ev.kind == "callback":
            await callbacks.handle_callback(ctx)
            return

        if (cmd is not None and cmd.name == "stop") or commands.is_stop_phrase(ev.text):
            await commands.withdraw_consent(ctx)
            return

        if not consent.has_consent(user, self._settings.consent_version):
            self._ask_consent(ctx)  # nothing else is processed or stored before consent
            return

        if cmd is not None:
            await commands.handle_command(ctx, cmd)
        elif ev.kind == "voice":
            await notes.accept_voice(ctx)
        elif ev.kind == "document" or (ev.kind == "photo" and bills_ui.is_bill_caption(ev.caption)):
            await bills_ui.accept_document(ctx)
        elif ev.kind == "photo":
            await notes.accept_photo(ctx)
        elif ev.kind == "text" and ev.text is not None:
            await notes.accept_text(ctx, ev.text)
        else:
            await text.handle_other(ctx)

    async def _start_with_token(self, ctx: Ctx, raw_token: str) -> None:
        result = await redeem_invite(
            ctx.session, raw_token=raw_token, chat_id=ctx.event.chat_id, now=ctx.now
        )
        if result.status is RedeemStatus.CHAT_ALREADY_BOUND:
            existing = await resolve_user(ctx.session, ctx.event.chat_id)
            ctx.say(ctx.t("welcome_back", name=existing.name if existing else ""))
            return
        if result.status is not RedeemStatus.OK or result.user is None or result.tenant is None:
            log.warning("invite_rejected", chat_id=ctx.event.chat_id)
            ctx.say(ctx.t("invite_invalid"))
            return
        ctx.user = result.user
        log.info("invite_redeemed", chat_id=ctx.event.chat_id)
        ctx.say(ctx.t("welcome_bound", name=result.user.name, company=result.tenant.name))
        self._ask_consent(ctx)

    def _ask_consent(self, ctx: Ctx) -> None:
        ctx.say(
            ctx.t("consent_notice"),
            (
                Button(ctx.t("btn_accept"), "co:y"),
                Button(ctx.t("btn_decline"), "co:n"),
            ),
        )
