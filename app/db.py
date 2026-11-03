"""Engine/session wiring, the idempotency gate, and the ONE tenant-scoped repository.

Every read or write of tenant-owned data goes through `TenantRepository`. The only
code allowed to query without a tenant is the small allowlist in
tests/test_tenant_isolation.py (identity lookup, invite redemption, bootstrap).
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import Sequence
from pathlib import Path
from typing import Any, TypeVar, cast
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from alembic import command
from alembic.config import Config
from sqlalchemy import ColumnElement, CursorResult, Select, delete, event, select, text, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.pool import NullPool

from app.models import AuditLog, ProcessedEvent, Tenant
from app.models.base import TenantOwned

T = TypeVar("T", bound=TenantOwned)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


_PG_SCHEMES = ("postgres://", "postgresql://", "postgresql+asyncpg://")
_DROPPED_PG_PARAMS = {"channel_binding"}  # libpq-only options that asyncpg rejects


def normalize_database_url(url: str) -> str:
    """Accept the URL a hosted Postgres provider shows you and make it work with asyncpg.

    Providers hand out `postgres://...?sslmode=require&channel_binding=require`. SQLAlchemy's
    asyncpg driver needs `postgresql+asyncpg://` and understands `ssl=` rather than `sslmode=`,
    and rejects libpq-only options. SQLite and already-correct URLs pass through unchanged."""
    if not url.startswith(_PG_SCHEMES):
        return url
    parts = urlsplit(url)
    query = []
    for key, value in parse_qsl(parts.query, keep_blank_values=True):
        if key in _DROPPED_PG_PARAMS:
            continue
        query.append(("ssl" if key == "sslmode" else key, value))
    return urlunsplit(parts._replace(scheme="postgresql+asyncpg", query=urlencode(query)))


def make_engine(url: str, *, serverless: bool = False) -> AsyncEngine:
    """`serverless=True` is for short-lived functions (Vercel): no connection pool, because a
    pooled connection would be frozen between invocations and found dead on the next one, and no
    prepared-statement cache, which a transaction-mode pooler (pgbouncer, Neon's pooler) breaks."""
    url = normalize_database_url(url)
    if serverless and not url.startswith("sqlite"):
        engine = create_async_engine(
            url, poolclass=NullPool, connect_args={"statement_cache_size": 0}
        )
    else:
        engine = create_async_engine(url)
    if url.startswith("sqlite"):
        # The pysqlite driver's implicit transaction handling does not honour SAVEPOINT:
        # RELEASE of the outermost savepoint would COMMIT, breaking atomicity (the dedupe
        # claim would survive a later failure in the same event). Standard SQLAlchemy
        # remedy: disable the driver's autobegin and emit BEGIN ourselves. IMMEDIATE takes
        # the write lock up front, so concurrent writers queue (up to the busy timeout)
        # instead of deadlocking on a read->write lock upgrade.
        @event.listens_for(engine.sync_engine, "connect")
        def _sqlite_connect(dbapi_conn: Any, _: Any) -> None:
            dbapi_conn.isolation_level = None
            cur = dbapi_conn.cursor()
            cur.execute("PRAGMA foreign_keys=ON")
            cur.close()

        @event.listens_for(engine.sync_engine, "begin")
        def _sqlite_begin(conn: Any) -> None:
            conn.exec_driver_sql("BEGIN IMMEDIATE")

    return engine


def make_session_factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(engine, expire_on_commit=False)


async def ping(engine: AsyncEngine) -> bool:
    try:
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
    except Exception:
        return False
    return True


def _alembic_config(url: str) -> Config:
    cfg = Config(str(PROJECT_ROOT / "alembic.ini"))
    cfg.set_main_option("script_location", str(PROJECT_ROOT / "migrations"))
    cfg.set_main_option("sqlalchemy.url", normalize_database_url(url).replace("%", "%%"))
    cfg.attributes["configure_logger"] = False
    return cfg


async def migrate_to_head(url: str) -> None:
    """Run Alembic in a worker thread (its async env.py owns its own event loop)."""
    await asyncio.to_thread(command.upgrade, _alembic_config(url), "head")


async def list_tenant_ids(session: AsyncSession) -> list[uuid.UUID]:
    """Tenants themselves are not tenant-owned; background sweeps iterate them, then work
    through a TenantRepository per tenant."""
    return list((await session.execute(select(Tenant.id))).scalars().all())


async def claim_event(session: AsyncSession, key: str) -> bool:
    """Idempotency gate. True = first time we see `key`; False = duplicate delivery."""
    try:
        async with session.begin_nested():
            session.add(ProcessedEvent(key=key))
            await session.flush()
    except IntegrityError:
        return False
    return True


class TenantRepository:
    """All tenant-owned access goes through here; tenant_id is injected, never passed in."""

    def __init__(self, session: AsyncSession, tenant_id: uuid.UUID) -> None:
        self._session = session
        self.tenant_id = tenant_id

    @staticmethod
    def _require_owned(model: type[Any]) -> None:
        if not (isinstance(model, type) and issubclass(model, TenantOwned)):
            raise TypeError(f"{model!r} is not tenant-owned; TenantRepository refuses it")

    def select(self, model: type[T]) -> Select[T]:
        self._require_owned(model)
        return select(model).where(model.tenant_id == self.tenant_id)

    async def get(self, model: type[T], id_: uuid.UUID) -> T | None:
        stmt = self.select(model).where(model.id == id_)
        return (await self._session.execute(stmt)).scalar_one_or_none()

    async def first(self, model: type[T], *criteria: ColumnElement[bool]) -> T | None:
        stmt = self.select(model).where(*criteria).limit(1)
        return (await self._session.execute(stmt)).scalars().first()

    async def list(
        self,
        model: type[T],
        *criteria: ColumnElement[bool],
        order_by: Sequence[Any] = (),
    ) -> list[T]:
        stmt = self.select(model).where(*criteria).order_by(*order_by)
        return list((await self._session.execute(stmt)).scalars().all())

    def add(self, obj: T) -> T:
        self._require_owned(type(obj))
        current = getattr(obj, "tenant_id", None)
        if current is not None and current != self.tenant_id:
            raise ValueError("object belongs to a different tenant")
        obj.tenant_id = self.tenant_id
        self._session.add(obj)
        return obj

    async def flush(self) -> None:
        await self._session.flush()

    async def update_where(
        self, model: type[T], values: dict[str, Any], *criteria: ColumnElement[bool]
    ) -> int:
        """Scoped UPDATE; returns rows changed (used for atomic compare-and-set)."""
        self._require_owned(model)
        stmt = update(model).where(model.tenant_id == self.tenant_id, *criteria).values(**values)
        result = cast("CursorResult[Any]", await self._session.execute(stmt))
        return result.rowcount

    async def delete_where(self, model: type[T], *criteria: ColumnElement[bool]) -> int:
        """Scoped DELETE; returns rows removed (used for atomic 'consume once')."""
        self._require_owned(model)
        stmt = delete(model).where(model.tenant_id == self.tenant_id, *criteria)
        result = cast("CursorResult[Any]", await self._session.execute(stmt))
        return result.rowcount

    def audit(
        self,
        action: str,
        *,
        entity: str,
        entity_id: str | None = None,
        actor_user_id: uuid.UUID | None = None,
        detail: dict[str, Any] | None = None,
    ) -> AuditLog:
        """Append an audit row (the only write path; AuditLog cannot be updated)."""
        return self.add(
            AuditLog(
                action=action,
                entity=entity,
                entity_id=entity_id,
                actor_user_id=actor_user_id,
                detail=detail or {},
            )
        )
