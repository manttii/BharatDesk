"""Tenant A must never read or write tenant B's data, and the code must keep it that way."""

from __future__ import annotations

import re
import uuid
from pathlib import Path

import pytest
from sqlalchemy import select

from app.db import TenantRepository
from app.models import Client, ConversationState, ProcessedEvent, Tenant, User
from app.models.enums import Role
from tests.conftest import Env

APP_DIR = Path(__file__).resolve().parent.parent / "app"
# The only modules allowed to query without a tenant scope (see their docstrings).
UNSCOPED_ALLOWLIST = {"db.py", "identity.py", "onboarding.py"}


async def _two_tenants(env: Env) -> tuple[uuid.UUID, uuid.UUID, uuid.UUID]:
    a, _, _ = await env.new_tenant("Tenant A", "Boss A")
    b, _, _ = await env.new_tenant("Tenant B", "Boss B")
    async with env.sf() as s, s.begin():
        b_client = TenantRepository(s, b).add(
            Client(name="Secret Client", normalized_name="secret")
        )
        TenantRepository(s, a).add(Client(name="A Client", normalized_name="a client"))
        await s.flush()
        return a, b, b_client.id


async def test_tenant_a_cannot_read_tenant_b_rows(env: Env) -> None:
    a, b, b_client_id = await _two_tenants(env)
    async with env.sf() as s:
        repo_a, repo_b = TenantRepository(s, a), TenantRepository(s, b)
        assert await repo_a.get(Client, b_client_id) is None
        assert await repo_b.get(Client, b_client_id) is not None
        assert [c.name for c in await repo_a.list(Client)] == ["A Client"]
        assert [u.name for u in await repo_a.list(User)] == ["Boss A"]
        assert await repo_a.first(Client, Client.name == "Secret Client") is None


async def test_tenant_a_cannot_delete_tenant_b_rows(env: Env) -> None:
    a, b, _ = await _two_tenants(env)
    async with env.sf() as s, s.begin():
        removed = await TenantRepository(s, a).delete_where(Client)
        assert removed == 1
    async with env.sf() as s:
        assert [c.name for c in await TenantRepository(s, b).list(Client)] == ["Secret Client"]


async def test_add_refuses_object_owned_by_another_tenant(env: Env) -> None:
    a, b, _ = await _two_tenants(env)
    async with env.sf() as s:
        foreign = Client(name="x", normalized_name="x", tenant_id=b)
        with pytest.raises(ValueError, match="different tenant"):
            TenantRepository(s, a).add(foreign)


async def test_repository_refuses_non_tenant_models(env: Env) -> None:
    a, _, _ = await _two_tenants(env)
    async with env.sf() as s:
        repo = TenantRepository(s, a)
        for model in (Tenant, ProcessedEvent):
            with pytest.raises(TypeError):
                repo.select(model)  # type: ignore[type-var]


async def test_bot_flow_stays_inside_the_callers_tenant(env: Env) -> None:
    """End to end: tenant A's boss lists the team and sees only tenant A."""
    _, _, raw_a = await env.new_tenant("Tenant A", "Boss A")
    _, _, raw_b = await env.new_tenant("Tenant B", "Boss B")
    await env.say(1, f"/start {raw_a}")
    await env.tap(1, "co:y")
    await env.say(2, f"/start {raw_b}")
    await env.tap(2, "co:y")
    await env.add_worker(2, "B Worker")
    out = await env.say(1, "/team")
    assert "Boss A" in out[-1].text
    assert "Boss B" not in out[-1].text and "B Worker" not in out[-1].text


async def test_pending_state_of_one_tenant_is_unusable_by_another(env: Env) -> None:
    await env.onboard_boss(1)
    out = await env.say(1, "/addworker Ramesh")
    assert out[0].buttons is not None
    yes = out[0].buttons[0][0].data
    _, _, raw_b = await env.new_tenant("Tenant B", "Boss B")
    await env.say(2, f"/start {raw_b}")
    await env.tap(2, "co:y")
    out = await env.tap(2, yes)  # Boss B taps Boss A's button
    assert out == []
    assert "expire" in (env.fake.callback_answers[-1][1] or "")
    async with env.sf() as s:
        assert (await s.execute(select(User).where(User.role == Role.WORKER))).first() is None
        assert (await s.execute(select(ConversationState))).scalars().all()  # A's is untouched


def test_no_unscoped_queries_outside_allowlist() -> None:
    """Static guard: handlers must use TenantRepository, not raw select/update/delete."""
    pattern = re.compile(
        r"from sqlalchemy import[^\n]*\b(select|update|delete)\b|\bsession\.execute\(|\.scalars\("
    )
    offenders = [
        str(p.relative_to(APP_DIR))
        for p in APP_DIR.rglob("*.py")
        if p.name not in UNSCOPED_ALLOWLIST and pattern.search(p.read_text(encoding="utf-8"))
    ]
    assert offenders == [], f"unscoped DB access outside the allowlist: {offenders}"
