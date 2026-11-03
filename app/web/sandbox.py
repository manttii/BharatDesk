"""Sandbox: run the REAL pipeline from a browser, with no Telegram involved.

A dev tool. It records a voice note (or takes text or an uploaded file), drives the same
EventProcessor the bot uses (with a FakeProvider for messaging), and shows every stage:
transcript, numerals, raw LLM JSON, validated payload, policy/commit result, final replies.
Speech and extraction use the real Groq/Gemini providers when their keys are configured;
without keys it says so rather than inventing output. It runs against demo data in its own
"Sandbox" tenant, is loopback-only, and is disabled when ENV=prod.
"""

from __future__ import annotations

import asyncio
import itertools
import time
import uuid
from datetime import timedelta
from pathlib import Path
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.bot.processor import EventProcessor
from app.config import Settings
from app.db import TenantRepository
from app.models import BenchmarkRate, Bill, PolicyRule, Tenant, User
from app.models.base import utcnow
from app.models.enums import Role
from app.pack import Pack
from app.pipeline.documents import DocumentError
from app.pipeline.media import MediaStore, sniff_image
from app.providers.llm import LLMProvider
from app.providers.messaging import FakeProvider
from app.providers.stt import STTProvider
from app.schemas.events import IncomingEvent
from app.security import RateLimiter
from app.services import bill_batch
from app.services.bill_ingest import ingest_document
from app.services.clients import create_client
from app.services.identity import resolve_user

REP_CHAT = 9_000_000_001
BOSS_CHAT = 9_000_000_002
ROLE_BY_CHAT = {REP_CHAT: "rep", BOSS_CHAT: "boss"}
DEMO_BENCHMARK_PAISE = 7_520_000  # Rs 75,200 per 10 g: DEMO DATA, not a market rate
LOOPBACK = {"127.0.0.1", "::1", "localhost", "testclient"}
PAGE = Path(__file__).resolve().parent / "templates" / "sandbox.html"
BILLS_PAGE = Path(__file__).resolve().parent / "templates" / "sandbox_bills.html"


class SandboxRunner:
    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        settings: Settings,
        pack: Pack,
        stt: STTProvider | None,
        llm: Any,
    ) -> None:
        self._sf = session_factory
        self.stt, self.llm = stt, llm
        self.fake = FakeProvider("SandboxBot")
        self.trace: list[dict[str, Any]] = []
        self.processor = EventProcessor(
            session_factory,
            self.fake,
            settings,
            pack,
            RateLimiter(10_000),
            stt=stt,
            llm=llm if isinstance(llm, LLMProvider) else None,
            media=MediaStore(Path(settings.media_dir) / "sandbox"),
            trace=self.trace,
        )
        self._lock = asyncio.Lock()
        self._ids = itertools.count(time.time_ns() // 1000)

    async def ensure_seeded(self) -> None:
        """Create the demo tenant once; keep the demo benchmark fresh (policy fails closed on
        a stale one, which is correct but unhelpful for a demo)."""
        async with self._sf() as session, session.begin():
            now = utcnow()
            rep = await resolve_user(session, REP_CHAT)
            if rep is None:
                tenant = Tenant(name="Sandbox (demo data)", settings={"sandbox": True})
                session.add(tenant)
                await session.flush()
                repo = TenantRepository(session, tenant.id)
                for chat, name, role in (
                    (REP_CHAT, "Demo Rep", Role.WORKER),
                    (BOSS_CHAT, "Demo Boss", Role.BOSS),
                ):
                    repo.add(
                        User(
                            name=name,
                            role=role,
                            telegram_chat_id=chat,
                            consent_at=now,
                            consent_version="v1",
                        )
                    )
                create_client(repo, "Rajendra Jewellers")
                repo.add(
                    PolicyRule(
                        rule_type="rate_floor_vs_benchmark",
                        params={"max_below_bps": 100, "max_benchmark_age_seconds": 86_400},
                    )
                )
                repo.add(
                    PolicyRule(
                        rule_type="max_order_value_without_approval",
                        params={"max_value_paise": 100_000_000},
                    )
                )
                await session.flush()
                rep = await resolve_user(session, REP_CHAT)
            assert rep is not None
            repo = TenantRepository(session, rep.tenant_id)
            boss = await resolve_user(session, BOSS_CHAT)
            rows = await repo.list(
                BenchmarkRate,
                order_by=[BenchmarkRate.set_at.desc(), BenchmarkRate.created_at.desc()],
            )
            if boss and (not rows or now - rows[0].set_at > timedelta(hours=12)):
                repo.add(
                    BenchmarkRate(
                        value_paise=DEMO_BENCHMARK_PAISE, unit="10g", set_by=boss.id, set_at=now
                    )
                )
            # keep the demo consent current if the notice version changed
            for u in await repo.list(User):
                if u.consent_version != "v1":
                    u.consent_version, u.consent_at = "v1", now

    async def run(
        self,
        *,
        as_user: str,
        text: str | None = None,
        audio: bytes | None = None,
        callback: str | None = None,
    ) -> dict[str, Any]:
        chat = BOSS_CHAT if as_user == "boss" else REP_CHAT
        async with self._lock:
            await self.ensure_seeded()
            self.trace.clear()
            self.fake.sent.clear()
            self.fake.callback_answers.clear()
            update_id = next(self._ids)
            if callback is not None:
                event = IncomingEvent(
                    update_id,
                    chat,
                    "callback",
                    callback_id=uuid.uuid4().hex,
                    callback_data=callback,
                )
            elif audio is not None:
                file_id = f"sandbox-{update_id}"
                self.fake.files[file_id] = audio
                kind: Literal["photo", "voice"] = (
                    "photo" if sniff_image(audio) else "voice"
                )  # image = slip
                event = IncomingEvent(update_id, chat, kind, file_id=file_id, mime_type=None)
            else:
                event = IncomingEvent(update_id, chat, "text", text=text or "")
            await self.processor.process(event)
            return {
                "stages": list(self.trace),
                "replies": [
                    {
                        "to": ROLE_BY_CHAT.get(m.chat_id, str(m.chat_id)),
                        "text": m.text,
                        "buttons": [
                            [{"text": b.text, "data": b.data} for b in row]
                            for row in (m.buttons or ())
                        ],
                    }
                    for m in self.fake.sent
                ],
                "toasts": [t for _, t in self.fake.callback_answers if t],
            }


router = APIRouter(prefix="/sandbox")


def _guard(request: Request) -> SandboxRunner:
    host = request.client.host if request.client else ""
    if host not in LOOPBACK:
        raise HTTPException(status_code=403, detail="sandbox is loopback-only")
    runner: SandboxRunner | None = request.app.state.sandbox
    if runner is None:
        raise HTTPException(status_code=404)
    return runner


@router.get("", response_class=HTMLResponse)
async def page(request: Request) -> HTMLResponse:
    _guard(request)
    return HTMLResponse(PAGE.read_text(encoding="utf-8"))


@router.get("/status")
async def status(request: Request) -> dict[str, bool]:
    runner = _guard(request)
    return {"stt": runner.stt is not None, "llm": runner.llm is not None}


@router.post("/run")
async def run(
    request: Request,
    as_user: Annotated[str, Form()] = "rep",
    text: Annotated[str | None, Form()] = None,
    audio: UploadFile | None = None,
) -> JSONResponse:
    runner = _guard(request)
    limit = request.app.state.settings.max_media_bytes
    data = await audio.read(limit + 1) if audio is not None else None
    if data is not None and len(data) > limit:
        raise HTTPException(status_code=413, detail="file too large")
    result = await runner.run(as_user=as_user, text=text, audio=data or None)
    return JSONResponse(result)


class TapBody(BaseModel):
    as_user: str = "rep"
    data: str


@router.post("/tap")
async def tap(request: Request, body: TapBody) -> JSONResponse:
    runner = _guard(request)
    return JSONResponse(await runner.run(as_user=body.as_user, callback=body.data))


# --- documents: upload a PDF/photo of bills, review, confirm ------------------------------------


def _bill_json(b: Bill) -> dict[str, Any]:
    return {
        "id": b.id.hex,
        "page": b.page_no,
        "route": b.route,
        "status": b.status,
        "type": b.doc_type,
        "party": b.party_name,
        "bill_no": b.bill_no,
        "date": b.bill_date.isoformat() if b.bill_date else None,
        "total_paise": b.total_paise,
        "flags": [
            {"field": f.get("field"), "reason": f.get("reason"), "detail": f.get("detail")}
            for f in b.flags
        ],
    }


@router.get("/bills", response_class=HTMLResponse)
async def bills_page(request: Request) -> HTMLResponse:
    _guard(request)
    return HTMLResponse(BILLS_PAGE.read_text(encoding="utf-8"))


@router.post("/bills/upload")
async def bills_upload(request: Request, file: UploadFile) -> JSONResponse:
    runner = _guard(request)
    cfg = request.app.state.settings
    if runner.llm is None:
        raise HTTPException(status_code=503, detail="no LLM configured")
    data = await file.read(cfg.max_pdf_bytes + 1)
    await runner.ensure_seeded()
    async with runner._sf() as session:
        boss = await resolve_user(session, BOSS_CHAT)
    assert boss is not None
    try:
        summary = await ingest_document(
            runner._sf,
            runner.llm,
            cfg,
            tenant_id=boss.tenant_id,
            user_id=boss.id,
            source_name=(file.filename or "upload")[:200],
            data=data,
            now=utcnow(),
        )
    except DocumentError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    async with runner._sf() as session:
        repo = TenantRepository(session, boss.tenant_id)
        bills = await repo.list(Bill, Bill.batch_id == summary.batch_id, order_by=[Bill.page_no])
        return JSONResponse(
            {
                "batch_id": summary.batch_id.hex,
                "counts": dict(summary.counts),
                "bills": [_bill_json(b) for b in bills],
            }
        )


class BatchBody(BaseModel):
    batch_id: str


class DecideBody(BaseModel):
    bill_id: str
    accept: bool


@router.post("/bills/confirm")
async def bills_confirm(request: Request, body: BatchBody) -> JSONResponse:
    runner = _guard(request)
    async with runner._sf() as session, session.begin():
        boss = await resolve_user(session, BOSS_CHAT)
        assert boss is not None
        n = await bill_batch.confirm_ready(
            TenantRepository(session, boss.tenant_id), uuid.UUID(body.batch_id), boss.id, utcnow()
        )
    return JSONResponse({"confirmed": n})


@router.post("/bills/decide")
async def bills_decide(request: Request, body: DecideBody) -> JSONResponse:
    runner = _guard(request)
    async with runner._sf() as session, session.begin():
        boss = await resolve_user(session, BOSS_CHAT)
        assert boss is not None
        ok = await bill_batch.decide_bill(
            TenantRepository(session, boss.tenant_id),
            uuid.UUID(body.bill_id),
            boss.id,
            utcnow(),
            accept=body.accept,
        )
    return JSONResponse({"ok": ok})
