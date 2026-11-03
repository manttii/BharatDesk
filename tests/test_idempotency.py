"""Telegram retries and double taps must be safe."""

from __future__ import annotations

from sqlalchemy import func, select

from app.models import ProcessedEvent, User
from app.schemas.events import IncomingEvent
from tests.conftest import Env

BOSS = 1001


async def test_same_update_id_is_processed_once(env: Env) -> None:
    await env.onboard_boss(BOSS)
    ev = IncomingEvent(update_id=777, chat_id=BOSS, kind="text", text="namaste")
    first = await env.send_event(ev)
    second = await env.send_event(ev)
    assert len(first) == 1 and second == []


async def test_same_callback_id_is_processed_once_but_always_answered(env: Env) -> None:
    await env.onboard_boss(BOSS)
    out = await env.say(BOSS, "/addworker Ramesh")
    assert out[0].buttons is not None
    yes = out[0].buttons[0][0].data
    first = await env.tap(BOSS, yes, callback_id="cb-1")
    # Same callback id arriving under a different update_id (e.g. a re-sent update).
    second = await env.tap(BOSS, yes, callback_id="cb-1")
    assert "start=" in first[-1].text and second == []
    answered = [cid for cid, _ in env.fake.callback_answers if cid == "cb-1"]
    assert len(answered) == 2  # the spinner is stopped both times
    async with env.sf() as s:
        assert (await s.execute(select(func.count()).select_from(User))).scalar_one() == 2


async def test_duplicate_onboarding_update_binds_once(env: Env) -> None:
    _, _, raw = await env.new_tenant()
    ev = IncomingEvent(update_id=5, chat_id=BOSS, kind="text", text=f"/start {raw}")
    await env.send_event(ev)
    again = await env.send_event(ev)
    assert again == []
    async with env.sf() as s:
        keys = (await s.execute(select(ProcessedEvent.key))).scalars().all()
        assert keys == ["u:5"]


async def test_failed_event_rolls_back_so_retry_is_processed(env: Env) -> None:
    await env.onboard_boss(BOSS)
    ev = IncomingEvent(update_id=900, chat_id=BOSS, kind="text", text="/help")

    original = env.processor._route
    calls = {"n": 0}

    async def flaky(ctx):  # type: ignore[no-untyped-def]
        await original(ctx)
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("boom after dedupe claim")

    env.processor._route = flaky  # type: ignore[method-assign]
    first = await env.send_event(ev)
    assert "gadbad" in first[0].text
    second = await env.send_event(ev)  # same update_id: claim was rolled back
    assert "Aap ye kar sakte hain" in second[0].text
