"""The audit log is append-only at the ORM and (SQLite) database level."""

from __future__ import annotations

import pytest
from sqlalchemy import delete, select, text, update
from sqlalchemy.exc import DatabaseError

from app.db import TenantRepository
from app.models import AuditLog
from app.models.audit import AuditLogImmutableError
from tests.conftest import Env


async def _seed(env: Env) -> None:
    tenant_id, _, _ = await env.new_tenant()
    async with env.sf() as s, s.begin():
        TenantRepository(s, tenant_id).audit("test_event", entity="thing", detail={"k": 1})


async def test_orm_update_is_refused(env: Env) -> None:
    await _seed(env)
    async with env.sf() as s:
        row = (
            await s.execute(select(AuditLog).where(AuditLog.action == "test_event"))
        ).scalar_one()
        row.action = "tampered"
        with pytest.raises(AuditLogImmutableError):
            await s.flush()


async def test_orm_delete_is_refused(env: Env) -> None:
    await _seed(env)
    async with env.sf() as s:
        row = (
            await s.execute(select(AuditLog).where(AuditLog.action == "test_event"))
        ).scalar_one()
        await s.delete(row)
        with pytest.raises(AuditLogImmutableError):
            await s.flush()


async def test_database_triggers_block_raw_update_and_delete(env: Env) -> None:
    await _seed(env)
    for stmt in (update(AuditLog).values(action="x"), delete(AuditLog)):
        async with env.sf() as s:
            with pytest.raises(DatabaseError, match="append-only"):
                await s.execute(stmt)
    async with env.sf() as s:
        with pytest.raises(DatabaseError, match="append-only"):
            await s.execute(text("UPDATE audit_log SET action = 'x'"))
        assert (await s.execute(select(AuditLog))).scalars().all()  # rows survived
