"""Append-only audit log. Enforced in the ORM and, on SQLite, by triggers."""

from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy import DDL, JSON, String, Uuid, event
from sqlalchemy.engine import Connection
from sqlalchemy.orm import Mapped, Mapper, mapped_column

from app.models.base import Base, TenantOwned

SQLITE_TRIGGERS: tuple[str, ...] = (
    "CREATE TRIGGER IF NOT EXISTS audit_log_no_update BEFORE UPDATE ON audit_log "
    "BEGIN SELECT RAISE(ABORT, 'audit_log is append-only'); END",
    "CREATE TRIGGER IF NOT EXISTS audit_log_no_delete BEFORE DELETE ON audit_log "
    "BEGIN SELECT RAISE(ABORT, 'audit_log is append-only'); END",
)


class AuditLog(TenantOwned, Base):
    __tablename__ = "audit_log"

    actor_user_id: Mapped[uuid.UUID | None] = mapped_column(Uuid, nullable=True)
    action: Mapped[str] = mapped_column(String(64), index=True)
    entity: Mapped[str] = mapped_column(String(48))
    entity_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    detail: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)


class AuditLogImmutableError(RuntimeError):
    pass


def _refuse(mapper: Mapper[Any], connection: Connection, target: AuditLog) -> None:
    raise AuditLogImmutableError("audit_log is append-only")


event.listen(AuditLog, "before_update", _refuse)
event.listen(AuditLog, "before_delete", _refuse)

for _stmt in SQLITE_TRIGGERS:
    event.listen(
        AuditLog.__table__,
        "after_create",
        DDL(_stmt).execute_if(dialect="sqlite"),  # type: ignore[no-untyped-call]
    )
# On PostgreSQL, enforce the same with `REVOKE UPDATE, DELETE ON audit_log FROM <app_role>`
# (see SECURITY.md); the ORM guard above applies on every backend.
