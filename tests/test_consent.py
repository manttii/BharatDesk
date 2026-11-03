"""No processing before consent; withdrawal halts processing."""

from __future__ import annotations

from sqlalchemy import select

from app.models import AuditLog, User
from tests.conftest import Env

BOSS = 1001


async def _bound_but_unconsented(env: Env) -> None:
    _, _, raw = await env.new_tenant()
    await env.say(BOSS, f"/start {raw}")


async def test_text_before_consent_is_not_processed(env: Env) -> None:
    await _bound_but_unconsented(env)
    out = await env.say(BOSS, "aaj ka rate 75200")
    assert len(out) == 1 and out[0].buttons is not None  # consent prompt, no echo
    assert "Aapne likha" not in out[0].text


async def test_commands_before_consent_are_gated(env: Env) -> None:
    await _bound_but_unconsented(env)
    out = await env.say(BOSS, "/addworker Ramesh")
    assert out[0].buttons is not None and out[0].buttons[0][0].data == "co:y"


async def test_buttons_other_than_consent_are_gated(env: Env) -> None:
    await _bound_but_unconsented(env)
    out = await env.tap(BOSS, "aw:y:" + "0" * 32)
    assert "sahmati" in out[-1].text.lower()


async def test_accept_unlocks_processing_and_records_version_and_time(env: Env) -> None:
    await _bound_but_unconsented(env)
    await env.tap(BOSS, "co:y")
    out = await env.say(BOSS, "namaste")
    assert out[0].buttons is None and "processing chalu nahi" in out[0].text  # reached the handler
    async with env.sf() as s:
        u = (await s.execute(select(User))).scalar_one()
        assert u.consent_at == env.clock() and u.consent_version == "v1"


async def test_decline_keeps_gate_closed(env: Env) -> None:
    await _bound_but_unconsented(env)
    await env.tap(BOSS, "co:n")
    out = await env.say(BOSS, "namaste")
    assert out[0].buttons is not None


async def test_stop_withdraws_consent_and_halts_processing(env: Env) -> None:
    await env.onboard_boss(BOSS)
    out = await env.say(BOSS, "/stop")
    assert "wapas le li" in out[0].text
    out = await env.say(BOSS, "namaste")
    assert out[0].buttons is not None
    async with env.sf() as s:
        assert (await s.execute(select(User))).scalar_one().consent_at is None
        actions = (await s.execute(select(AuditLog.action))).scalars().all()
        assert "consent_withdrawn" in actions


async def test_band_karo_phrase_withdraws_consent(env: Env) -> None:
    await env.onboard_boss(BOSS)
    out = await env.say(BOSS, "  Band   KARO ")
    assert "wapas le li" in out[0].text


async def test_consent_can_be_given_again_after_withdrawal(env: Env) -> None:
    await env.onboard_boss(BOSS)
    await env.say(BOSS, "/stop")
    await env.say(BOSS, "/start")
    await env.tap(BOSS, "co:y")
    out = await env.say(BOSS, "wapas aa gaya")
    assert out[0].buttons is None and "processing chalu nahi" in out[0].text


async def test_new_notice_version_requires_fresh_consent(env: Env) -> None:
    await env.onboard_boss(BOSS)
    env.processor._settings = env.settings.model_copy(update={"consent_version": "v2"})
    out = await env.say(BOSS, "namaste")
    assert out[0].buttons is not None


async def test_inactive_user_gets_no_service(env: Env) -> None:
    await env.onboard_boss(BOSS)
    async with env.sf() as s, s.begin():
        (await s.execute(select(User))).scalar_one().is_active = False
    out = await env.say(BOSS, "namaste")
    assert "active nahi" in out[0].text
