"""Everything a note needs beyond the database: providers, limits, clock, optional trace."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.config import Settings
from app.models.base import utcnow
from app.pack import Pack
from app.pipeline.media import MediaStore
from app.providers.llm import LLMProvider
from app.providers.messaging import MessagingProvider
from app.providers.sheets import SheetsProvider
from app.providers.stt import STTProvider


@dataclass
class Deps:
    session_factory: async_sessionmaker[AsyncSession]
    messaging: MessagingProvider
    settings: Settings
    pack: Pack
    stt: STTProvider | None = None
    llm: LLMProvider | None = None
    media: MediaStore | None = None
    sheets: SheetsProvider | None = None
    clock: Callable[[], datetime] = utcnow
    # The sandbox sets this to a list to collect one record per pipeline stage.
    trace: list[dict[str, Any]] | None = field(default=None)

    def stage(self, name: str, **data: Any) -> None:
        if self.trace is not None:
            self.trace.append({"stage": name, **data})
