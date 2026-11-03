"""Invite tokens: create (boss-initiated) and redeem (Start deep link).

Redemption is a single conditional UPDATE, so a token is consumed at most once even
under concurrent taps. Raw tokens exist only in the link we hand to the boss; the
database stores a SHA-256.
"""

from __future__ import annotations

import enum
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import TenantRepository
from app.models import InviteToken, Tenant, User
from app.models.enums import Role
from app.security import hash_token, new_token


class RedeemStatus(enum.StrEnum):
    OK = "ok"
    INVALID = "invalid"  # unknown, expired or already used: deliberately indistinguishable
    CHAT_ALREADY_BOUND = "chat_already_bound"


@dataclass(frozen=True)
class RedeemResult:
    status: RedeemStatus
    user: User | None = None
    tenant: Tenant | None = None


def issue_invite(
    repo: TenantRepository,
    *,
    user: User,
    created_by: uuid.UUID | None,
    now: datetime,
    ttl_hours: int,
) -> str:
    """Create an InviteToken for `user`; returns the raw token (shown once)."""
    raw = new_token()
    repo.add(
        InviteToken(
            user_id=user.id,
            role=user.role,
            token_hash=hash_token(raw),
            expires_at=now + timedelta(hours=ttl_hours),
            created_by=created_by,
        )
    )
    return raw


async def create_worker(
    repo: TenantRepository,
    *,
    boss: User,
    name: str,
    now: datetime,
    ttl_hours: int,
) -> tuple[User, str]:
    worker = repo.add(User(name=name, role=Role.WORKER, is_active=True))
    await repo.flush()  # assigns worker.id (UUID default fires on flush)
    raw = issue_invite(repo, user=worker, created_by=boss.id, now=now, ttl_hours=ttl_hours)
    repo.audit(
        "worker_created",
        entity="user",
        entity_id=str(worker.id),
        actor_user_id=boss.id,
        detail={"role": worker.role.value},
    )
    return worker, raw


async def bootstrap_tenant(
    session: AsyncSession,
    *,
    tenant_name: str,
    boss_name: str,
    now: datetime,
    ttl_hours: int,
) -> tuple[Tenant, User, str]:
    """Operator-run (scripts/bootstrap.py): create a tenant and its boss invite."""
    tenant = Tenant(name=tenant_name)
    session.add(tenant)
    await session.flush()
    repo = TenantRepository(session, tenant.id)
    boss = repo.add(User(name=boss_name, role=Role.BOSS, is_active=True))
    await repo.flush()
    raw = issue_invite(repo, user=boss, created_by=None, now=now, ttl_hours=ttl_hours)
    repo.audit("tenant_bootstrapped", entity="tenant", entity_id=str(tenant.id))
    return tenant, boss, raw


async def redeem_invite(
    session: AsyncSession, *, raw_token: str, chat_id: int, now: datetime
) -> RedeemResult:
    """Bind `chat_id` to the user the token was issued for. Cross-tenant by nature:
    the token itself identifies the tenant."""
    token_hash = hash_token(raw_token)

    # Refuse before consuming, so a chat that is already someone else keeps the token.
    taken = await session.execute(select(User.id).where(User.telegram_chat_id == chat_id))
    if taken.first() is not None:
        return RedeemResult(RedeemStatus.CHAT_ALREADY_BOUND)

    consume = (
        update(InviteToken)
        .where(
            InviteToken.token_hash == token_hash,
            InviteToken.used_at.is_(None),
            InviteToken.expires_at > now,
        )
        .values(used_at=now)
        .returning(InviteToken.user_id, InviteToken.tenant_id)
    )
    row = (await session.execute(consume)).first()
    if row is None:
        return RedeemResult(RedeemStatus.INVALID)
    user_id, tenant_id = row

    repo = TenantRepository(session, tenant_id)
    user = await repo.get(User, user_id)
    tenant = await session.get(Tenant, tenant_id)
    if user is None or tenant is None:  # FK makes this unreachable; fail closed anyway
        return RedeemResult(RedeemStatus.INVALID)
    user.telegram_chat_id = chat_id
    repo.audit("invite_redeemed", entity="user", entity_id=str(user.id), actor_user_id=user.id)
    return RedeemResult(RedeemStatus.OK, user=user, tenant=tenant)
