"""Shared fixtures. Each test gets its own SQLite file cloned from a database that was
built by the real Alembic migration, so migrations are exercised by the whole suite.

`env` is the bare bot (no speech/LLM providers). `pipe` is the same bot wired to FakeSTT and
FakeLLM, with helpers to run a whole voice note through it."""

from __future__ import annotations

import asyncio
import itertools
import json
import re
import shutil
import uuid
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from app.bot.processor import EventProcessor
from app.config import Settings
from app.db import TenantRepository, make_engine, make_session_factory, migrate_to_head
from app.models import BenchmarkRate, ConversationState, PolicyRule, Tenant, User
from app.pack import load_pack
from app.pipeline.media import MediaStore
from app.providers.llm import FakeLLM
from app.providers.messaging import FakeProvider
from app.providers.sheets import FakeSheets
from app.providers.stt import FakeSTT
from app.schemas.events import IncomingEvent, OutMessage
from app.security import RateLimiter
from app.services.clients import create_client
from app.services.onboarding import bootstrap_tenant

T0 = datetime(2026, 11, 2, 9, 0, tzinfo=UTC)  # Monday 2 Nov 2026, 14:30 IST
BOSS_CHAT, REP_CHAT = 1001, 2002
OGG = b"OggS" + b"\x00" * 60
JPEG = b"\xff\xd8\xff\xe0" + b"\x00" * 60
SHEET_ID = "1AbCdEfGhIjKlMnOpQrStUvWxYz0123456789"
SHEET_URL = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/edit#gid=0"

TRANSCRIPT = (
    "Rajendra Jewellers gaya tha, dedh sau gram necklace, chaar lakh baarah hazaar ka order, "
    "paintalis hazaar advance, 8 November ko delivery"
)


class Clock:
    def __init__(self) -> None:
        self.now = T0

    def __call__(self) -> datetime:
        return self.now

    def advance(self, **kwargs: float) -> None:
        self.now += timedelta(**kwargs)


def extraction_json(
    *,
    client: str | None = "Rajendra Jewellers",
    intent: str = "LOG_VISIT",
    weight_mg: int | None = 150_000,
    value_paise: int | None = 41_200_000,
    rate: int | None = None,
    received: int | None = 4_500_000,
    delivery: str | None = "8 November",
    followup: str | None = None,
    conf: float = 0.99,
    order: bool = True,
    **override: Any,
) -> str:
    """A valid LLM reply for the TRANSCRIPT above (override anything to break it)."""
    data: dict[str, Any] = {
        "intent": intent,
        "client_name": client,
        "location": None,
        "person_met": None,
        "samples_shown": [],
        "order": (
            {
                "items": [{"name": "necklace", "quantity": 1}],
                "weight_mg": weight_mg,
                "value_paise": value_paise,
                "rate_paise_per_10g": rate,
            }
            if order
            else None
        ),
        "payment": {"received_paise": received, "pending_paise": None}
        if received is not None
        else None,
        "delivery_date_text": delivery,
        "next_followup_date_text": followup,
        "visit_date_text": None,
        "notes": None,
        "benchmark_rate_paise": None,
        "confidence": {
            k: conf
            for k in (
                "intent client_name weight_mg value_paise rate_paise_per_10g received_paise "
                "pending_paise delivery_date next_followup_date visit_date benchmark_rate_paise"
            ).split()
        },
    }
    data.update(override)
    return json.dumps(data)


def slip_json(
    *,
    party: str | None = "Rajendra Jewellers",
    date_text: str | None = "1/11",
    items: list[dict[str, Any]] | None = None,
    total: int | None = 41_200_000,
    legible: bool = True,
    conf: float = 0.97,
    overall: float | None = None,
    **override: Any,
) -> str:
    """A valid vision-model reply for a slip: two lines that add up to Rs 4,12,000."""
    if items is None:
        items = [
            {"name": "necklace", "quantity": 1, "rate_paise": None,
             "amount_paise": 30_000_000, "confidence": conf},
            {"name": "chain", "quantity": 2, "rate_paise": 5_600_000,
             "amount_paise": 11_200_000, "confidence": conf},
        ]  # fmt: skip
    data: dict[str, Any] = {
        "legible": legible,
        "party_name": party,
        "date_text": date_text,
        "items": items,
        "total_paise": total,
        "confidence": {"party_name": conf, "date": conf, "total": conf,
                       "overall": conf if overall is None else overall},
    }  # fmt: skip
    data.update(override)
    return json.dumps(data)


@dataclass
class Org:
    tenant_id: uuid.UUID
    boss_id: uuid.UUID
    rep_id: uuid.UUID


@dataclass
class Env:
    engine: AsyncEngine
    sf: async_sessionmaker[AsyncSession]
    settings: Settings
    fake: FakeProvider
    clock: Clock
    processor: EventProcessor
    stt: FakeSTT | None = None
    llm: FakeLLM | None = None
    media_dir: Path | None = None
    sheets: FakeSheets | None = None
    _update_ids: Callable[[], int] = field(default_factory=lambda: itertools.count(1).__next__)

    # ---- sending events ----

    async def say(self, chat_id: int, text: str) -> list[OutMessage]:
        return await self._send(IncomingEvent(self._update_ids(), chat_id, "text", text=text))

    async def tap(
        self, chat_id: int, data: str, *, callback_id: str | None = None
    ) -> list[OutMessage]:
        cid = callback_id or uuid.uuid4().hex
        return await self._send(
            IncomingEvent(
                self._update_ids(), chat_id, "callback", callback_id=cid, callback_data=data
            )
        )

    async def send_event(self, event: IncomingEvent) -> list[OutMessage]:
        return await self._send(event)

    async def _send(self, event: IncomingEvent) -> list[OutMessage]:
        before = len(self.fake.sent)
        await self.processor.process(event)
        return self.fake.sent[before:]

    # ---- voice / note helpers (need `pipe`) ----

    async def voice(
        self,
        chat_id: int,
        transcript: str | None = TRANSCRIPT,
        reply: str | None = None,
        *,
        duration: int | None = 5,
        mime: str | None = "audio/ogg",
        size: int | None = None,
        data: bytes = OGG,
        stt: dict[str, Any] | None = None,
        update_id: int | None = None,
    ) -> list[OutMessage]:
        assert self.stt is not None and self.llm is not None
        uid = update_id or self._update_ids()
        file_id = f"file-{uid}"
        self.fake.files[file_id] = data
        if transcript is not None:
            self.stt.say(transcript, **(stt or {}))
        if reply is not None:
            self.llm.replies.append(reply)
        ev = IncomingEvent(
            uid,
            chat_id,
            "voice",
            file_id=file_id,
            duration_s=duration,
            mime_type=mime,
            file_size=size,
        )
        return await self._send(ev)

    async def photo(
        self,
        chat_id: int,
        reply: str | None = None,
        *,
        data: bytes = JPEG,
        width: int | None = 1280,
        height: int | None = 960,
        size: int | None = None,
        update_id: int | None = None,
    ) -> list[OutMessage]:
        """A photo of a slip; `reply` is the (scripted) vision-model JSON."""
        assert self.llm is not None
        uid = update_id or self._update_ids()
        file_id = f"photo-{uid}"
        self.fake.files[file_id] = data
        if reply is not None:
            self.llm.replies.append(reply)
        ev = IncomingEvent(
            uid, chat_id, "photo", file_id=file_id, file_size=size, width=width, height=height
        )
        return await self._send(ev)

    async def connect_sheet(
        self, boss_chat: int = BOSS_CHAT, sheet_url: str = SHEET_URL
    ) -> list[OutMessage]:
        """Boss runs /sheet <url> and taps Haan; returns the messages from the tap."""
        out = await self.say(boss_chat, f"/sheet {sheet_url}")
        return await self.tap(boss_chat, button_data(out[-1], "Haan"))

    async def note(self, chat_id: int, text: str, reply: str | None) -> list[OutMessage]:
        assert self.llm is not None
        if reply is not None:
            self.llm.replies.append(reply)
        return await self.say(chat_id, text)

    # ---- organisation / data setup ----

    async def new_tenant(
        self, name: str = "Sharma Jewellers", boss: str = "Rakesh"
    ) -> tuple[uuid.UUID, uuid.UUID, str]:
        async with self.sf() as session, session.begin():
            tenant, user, raw = await bootstrap_tenant(
                session,
                tenant_name=name,
                boss_name=boss,
                now=self.clock(),
                ttl_hours=self.settings.invite_ttl_hours,
            )
            return tenant.id, user.id, raw

    async def onboard_boss(self, chat_id: int = BOSS_CHAT) -> tuple[uuid.UUID, uuid.UUID]:
        """Tenant + boss bound to `chat_id` and consented. Returns (tenant_id, boss_id)."""
        tenant_id, boss_id, raw = await self.new_tenant()
        await self.say(chat_id, f"/start {raw}")
        await self.tap(chat_id, "co:y")
        return tenant_id, boss_id

    async def add_worker(self, boss_chat: int, name: str) -> str:
        """Boss runs /addworker and confirms; returns the raw invite token."""
        out = await self.say(boss_chat, f"/addworker {name}")
        assert out[-1].buttons is not None
        yes = out[-1].buttons[0][0].data
        out = await self.tap(boss_chat, yes)
        return extract_token(out[-1].text)

    async def org(self, *, with_rep: bool = True) -> Org:
        """Tenant with a consented boss (BOSS_CHAT) and a consented rep (REP_CHAT)."""
        tenant_id, boss_id = await self.onboard_boss(BOSS_CHAT)
        rep_id = uuid.UUID(int=0)
        if with_rep:
            token = await self.add_worker(BOSS_CHAT, "Ramesh")
            await self.say(REP_CHAT, f"/start {token}")
            await self.tap(REP_CHAT, "co:y")
            async with self.sf() as s:
                rep_id = (
                    await s.execute(select(User.id).where(User.telegram_chat_id == REP_CHAT))
                ).scalar_one()
        self.fake.sent.clear()
        return Org(tenant_id, boss_id, rep_id)

    async def add_rules(self, org: Org, *rules: tuple[str, dict[str, Any]]) -> None:
        async with self.sf() as s, s.begin():
            repo = TenantRepository(s, org.tenant_id)
            for rule_type, params in rules:
                repo.add(PolicyRule(rule_type=rule_type, params=params))

    async def set_benchmark(
        self, org: Org, value_paise: int = 7_520_000, *, age_s: int = 3600, unit: str = "10g"
    ) -> None:
        async with self.sf() as s, s.begin():
            TenantRepository(s, org.tenant_id).add(
                BenchmarkRate(
                    value_paise=value_paise,
                    unit=unit,
                    set_by=org.boss_id,
                    set_at=self.clock() - timedelta(seconds=age_s),
                )
            )

    async def add_client(self, org: Org, name: str) -> uuid.UUID:
        async with self.sf() as s, s.begin():
            repo = TenantRepository(s, org.tenant_id)
            client = create_client(repo, name)
            await repo.flush()
            return client.id

    async def tenant_settings(self, org: Org, **values: Any) -> None:
        async with self.sf() as s, s.begin():
            tenant = await s.get(Tenant, org.tenant_id)
            assert tenant is not None
            tenant.settings = {**tenant.settings, **values}

    async def all(self, model: type) -> list[Any]:
        async with self.sf() as s:
            return list((await s.execute(select(model))).scalars().all())

    async def states(self) -> list[ConversationState]:
        return await self.all(ConversationState)  # type: ignore[no-any-return]


def to(chat_id: int, messages: list[OutMessage]) -> list[OutMessage]:
    return [m for m in messages if m.chat_id == chat_id]


def button_data(msg: OutMessage, label: str) -> str:
    for row in msg.buttons or ():
        for b in row:
            if b.text == label:
                return b.data
    raise AssertionError(f"no {label!r} button in {msg.buttons!r}")


def extract_token(text: str) -> str:
    m = re.search(r"start=([A-Za-z0-9_-]+)", text)
    assert m, f"no deep link in: {text!r}"
    return m.group(1)


@pytest.fixture(scope="session")
def template_db(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("tpl") / "template.db"
    asyncio.run(migrate_to_head(f"sqlite+aiosqlite:///{path.as_posix()}"))
    return path


async def _build(template_db: Path, tmp_path: Path, *, providers: bool) -> Env:
    db = tmp_path / "test.db"
    shutil.copy(template_db, db)
    url = f"sqlite+aiosqlite:///{db.as_posix()}"
    settings = Settings(_env_file=None, database_url=url, auto_migrate=False)  # type: ignore[call-arg]
    engine = make_engine(url)
    sf = make_session_factory(engine)
    fake = FakeProvider()
    clock = Clock()
    stt = FakeSTT() if providers else None
    llm = FakeLLM() if providers else None
    sheets = FakeSheets() if providers else None
    media_dir = tmp_path / "media"
    processor = EventProcessor(
        sf, fake, settings, load_pack(), RateLimiter(10_000), clock,
        stt=stt, llm=llm, media=MediaStore(media_dir) if providers else None, sheets=sheets,
    )  # fmt: skip
    return Env(engine, sf, settings, fake, clock, processor, stt, llm, media_dir, sheets)


@pytest.fixture
async def env(template_db: Path, tmp_path: Path) -> AsyncIterator[Env]:
    e = await _build(template_db, tmp_path, providers=False)
    yield e
    await e.engine.dispose()


@pytest.fixture
async def pipe(template_db: Path, tmp_path: Path) -> AsyncIterator[Env]:
    e = await _build(template_db, tmp_path, providers=True)
    yield e
    await e.engine.dispose()
