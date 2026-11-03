"""Per-event context. Handlers queue replies here; the processor sends them only after
the database transaction commits, so a rolled-back event never tells the user it worked.

Slow work (downloads, speech-to-text, LLM calls) is registered with `defer()` and runs
AFTER the first transaction commits, so no database lock is held across a network call.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.bot.deps import Deps
from app.config import Settings
from app.db import TenantRepository
from app.models import User
from app.pack import Pack
from app.schemas.events import Button, IncomingEvent, OutMessage

Deferred = Callable[[], Awaitable[list[OutMessage]]]


@dataclass
class Ctx:
    session: AsyncSession
    settings: Settings
    pack: Pack
    event: IncomingEvent
    now: datetime
    bot_username: str
    deps: Deps | None = None
    user: User | None = None
    replies: list[OutMessage] = field(default_factory=list)
    callback_text: str | None = None
    deferred: list[Deferred] = field(default_factory=list)

    @property
    def repo(self) -> TenantRepository:
        if self.user is None:
            raise RuntimeError("no resolved user; tenant-scoped access is not available")
        return TenantRepository(self.session, self.user.tenant_id)

    def t(self, key: str, **values: Any) -> str:
        return self.pack.t(key, **values)

    def say(self, text: str, *button_rows: tuple[Button, ...]) -> None:
        self.say_to(self.event.chat_id, text, *button_rows)

    def say_to(self, chat_id: int, text: str, *button_rows: tuple[Button, ...]) -> None:
        self.replies.append(
            OutMessage(chat_id=chat_id, text=text, buttons=tuple(button_rows) or None)
        )

    def extend(self, messages: list[OutMessage]) -> None:
        self.replies.extend(messages)

    def defer(self, job: Deferred) -> None:
        self.deferred.append(job)
