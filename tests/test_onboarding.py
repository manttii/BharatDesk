"""Invite-link onboarding: single use, expiry, hashing, boss-only /addworker."""

from __future__ import annotations

import asyncio

from sqlalchemy import func, select

from app.models import InviteToken, ProcessedEvent, User
from app.models.enums import Role
from app.security import hash_token
from app.services.onboarding import RedeemStatus, redeem_invite
from tests.conftest import Env, extract_token

BOSS, WORKER, STRANGER = 1001, 2002, 9999


async def _count(env: Env, model: type) -> int:
    async with env.sf() as s:
        return (await s.execute(select(func.count()).select_from(model))).scalar_one()


async def test_valid_token_binds_chat_and_asks_consent(env: Env) -> None:
    _, boss_id, raw = await env.new_tenant()
    out = await env.say(BOSS, f"/start {raw}")
    assert "Sharma Jewellers" in out[0].text
    assert out[1].buttons is not None and out[1].buttons[0][0].data == "co:y"
    async with env.sf() as s:
        user = await s.get(User, boss_id)
        assert user is not None
        assert user.telegram_chat_id == BOSS
        assert user.consent_at is None  # binding is not consent


async def test_token_is_single_use(env: Env) -> None:
    _, _, raw = await env.new_tenant()
    await env.say(BOSS, f"/start {raw}")
    out = await env.say(STRANGER, f"/start {raw}")
    assert "kaam nahi kar raha" in out[0].text
    async with env.sf() as s:
        owner = (await s.execute(select(User).where(User.telegram_chat_id == BOSS))).scalar_one()
        assert owner.role is Role.BOSS
        hijack = await s.execute(select(User).where(User.telegram_chat_id == STRANGER))
        assert hijack.first() is None


async def test_concurrent_redemption_binds_exactly_once(env: Env) -> None:
    _, _, raw = await env.new_tenant()

    async def attempt(chat_id: int) -> RedeemStatus:
        async with env.sf() as s, s.begin():
            return (await redeem_invite(s, raw_token=raw, chat_id=chat_id, now=env.clock())).status

    results = await asyncio.gather(attempt(11), attempt(22))
    assert sorted(results) == [RedeemStatus.INVALID, RedeemStatus.OK]


async def test_expired_token_rejected(env: Env) -> None:
    _, _, raw = await env.new_tenant()
    env.clock.advance(hours=env.settings.invite_ttl_hours, seconds=1)
    out = await env.say(BOSS, f"/start {raw}")
    assert "kaam nahi kar raha" in out[0].text


async def test_token_valid_just_before_expiry(env: Env) -> None:
    _, _, raw = await env.new_tenant()
    env.clock.advance(hours=env.settings.invite_ttl_hours, seconds=-1)
    out = await env.say(BOSS, f"/start {raw}")
    assert "jud gaye" in out[0].text


async def test_token_stored_hashed_not_raw(env: Env) -> None:
    _, _, raw = await env.new_tenant()
    async with env.sf() as s:
        stored = (await s.execute(select(InviteToken.token_hash))).scalar_one()
    assert stored == hash_token(raw)
    assert raw not in stored


async def test_bogus_token_rejected(env: Env) -> None:
    out = await env.say(STRANGER, "/start not-a-real-token")
    assert "kaam nahi kar raha" in out[0].text
    assert await _count(env, User) == 0


async def test_unknown_chat_gets_one_polite_reply_and_nothing_stored(env: Env) -> None:
    out = await env.say(STRANGER, "hello")
    assert len(out) == 1 and "registered nahi" in out[0].text
    assert await _count(env, User) == 0
    # Only the dedupe ledger (no content) is written for strangers.
    assert await _count(env, ProcessedEvent) == 1


async def test_boss_adds_worker_via_readback_and_worker_joins(env: Env) -> None:
    await env.onboard_boss(BOSS)
    out = await env.say(BOSS, "/addworker Ramesh Kumar")
    assert "Ramesh Kumar" in out[0].text and out[0].buttons is not None  # read-back first
    assert await _count(env, User) == 1  # nothing created before the tap

    token = await env.add_worker(BOSS, "Ramesh Kumar")
    out = await env.say(WORKER, f"/start {token}")
    assert "Sharma Jewellers" in out[0].text
    async with env.sf() as s:
        w = (await s.execute(select(User).where(User.telegram_chat_id == WORKER))).scalar_one()
        assert (w.name, w.role) == ("Ramesh Kumar", Role.WORKER)


async def test_double_tap_on_addworker_creates_one_worker(env: Env) -> None:
    await env.onboard_boss(BOSS)
    out = await env.say(BOSS, "/addworker Ramesh")
    assert out[0].buttons is not None
    yes = out[0].buttons[0][0].data
    first = await env.tap(BOSS, yes)
    second = await env.tap(BOSS, yes)
    assert "start=" in first[-1].text
    assert second == []  # no second link
    assert "expire" in (env.fake.callback_answers[-1][1] or "")  # toast on the stale button
    assert await _count(env, User) == 2  # boss + exactly one worker
    assert await _count(env, InviteToken) == 2


async def test_addworker_no_tap_means_no_worker(env: Env) -> None:
    await env.onboard_boss(BOSS)
    out = await env.say(BOSS, "/addworker Ramesh")
    assert out[0].buttons is not None
    no = out[0].buttons[0][1].data
    out = await env.tap(BOSS, no)
    assert "cancel" in out[-1].text.lower()
    assert await _count(env, User) == 1


async def test_pending_addworker_expires(env: Env) -> None:
    await env.onboard_boss(BOSS)
    out = await env.say(BOSS, "/addworker Ramesh")
    assert out[0].buttons is not None
    yes = out[0].buttons[0][0].data
    env.clock.advance(minutes=env.settings.pending_action_ttl_minutes, seconds=1)
    out = await env.tap(BOSS, yes)
    assert out == []
    assert "expire" in (env.fake.callback_answers[-1][1] or "")
    assert await _count(env, User) == 1


async def test_worker_cannot_use_boss_commands(env: Env) -> None:
    await env.onboard_boss(BOSS)
    token = await env.add_worker(BOSS, "Ramesh")
    await env.say(WORKER, f"/start {token}")
    await env.tap(WORKER, "co:y")
    for cmd in ("/addworker Mallory", "/team"):
        out = await env.say(WORKER, cmd)
        assert "sirf boss" in out[-1].text
    assert await _count(env, User) == 2


async def test_worker_cannot_replay_boss_confirm_button(env: Env) -> None:
    """A worker who obtains the callback id of a boss's pending read-back gets nothing."""
    await env.onboard_boss(BOSS)
    token = await env.add_worker(BOSS, "Ramesh")
    await env.say(WORKER, f"/start {token}")
    await env.tap(WORKER, "co:y")
    out = await env.say(BOSS, "/addworker Mallory")
    assert out[0].buttons is not None
    yes = out[0].buttons[0][0].data
    out = await env.tap(WORKER, yes)
    assert "sirf boss" in out[-1].text
    assert await _count(env, User) == 2


async def test_team_lists_members_with_status(env: Env) -> None:
    await env.onboard_boss(BOSS)
    await env.add_worker(BOSS, "Ramesh")
    out = await env.say(BOSS, "/team")
    assert "Rakesh (boss) — active" in out[-1].text
    assert "Ramesh (worker) — link bheja" in out[-1].text


async def test_invite_link_token_fits_telegram_start_payload(env: Env) -> None:
    await env.onboard_boss(BOSS)
    token = await env.add_worker(BOSS, "Ramesh")
    assert len(token) <= 64 and extract_token(f"start={token}") == token
