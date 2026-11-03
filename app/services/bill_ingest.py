"""Upload -> pages -> extraction -> saved bills. The one entry point the web page, the CLI and
(later) the chat bot all call, so every route behaves identically."""

from __future__ import annotations

import asyncio
import hashlib
import uuid
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

from app.config import Settings
from app.db import TenantRepository
from app.models import Tenant
from app.pipeline.documents import read_document
from app.providers.llm import LLMProvider
from app.services.bill_batch import BatchSummary, extract_pages, save_batch


async def ingest_document(
    session_factory: Any,
    llm: LLMProvider,
    cfg: Settings,
    *,
    tenant_id: uuid.UUID,
    user_id: uuid.UUID,
    source_name: str,
    data: bytes,
    now: datetime,
    retry_delay_s: float = 3.0,
) -> BatchSummary:
    """Raises DocumentError (safe to show) for an unusable file. Provider failures on single
    pages are recorded on those pages instead of failing the whole upload."""
    pages = await asyncio.to_thread(
        read_document, data, max_pages=cfg.max_pdf_pages, max_bytes=cfg.max_pdf_bytes
    )
    async with session_factory() as session:
        tenant = await session.get(Tenant, tenant_id)
        assert tenant is not None
        tz, business = tenant.timezone, tenant.name
    today = now.astimezone(ZoneInfo(tz)).date()
    outcomes = await extract_pages(
        llm,
        pages,
        today=today,
        timezone=tz,
        min_confidence=cfg.low_confidence_flag,
        min_legibility=cfg.slip_min_legibility,
        concurrency=cfg.bill_concurrency,
        retry_delay_s=retry_delay_s,
        business=business,
    )
    async with session_factory() as session, session.begin():
        return await save_batch(
            TenantRepository(session, tenant_id),
            uploaded_by=user_id,
            source_name=source_name,
            source_sha256=hashlib.sha256(data).hexdigest(),
            outcomes=outcomes,
        )
