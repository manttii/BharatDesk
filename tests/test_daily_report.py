"""The optional daily boss report: opt-in, once a day, computed only from saved data."""

from __future__ import annotations

from datetime import timedelta

import pytest
from pydantic import ValidationError

from app.models import AuditLog, Tenant, User
from app.pack import load_pack
from app.pipeline.confirm import TenantSettings
from app.services.daily_report import send_due
from tests.conftest import (
    BOSS_CHAT,
    REP_CHAT,
    TRANSCRIPT,
    Env,
    button_data,
    extraction_json,
    to,
)

# T0 is 09:00 UTC = 14:30 IST on Mon 2 Nov 2026. 20:00 IST is 14:30 UTC (5 h 30 m later).
TO_8PM = timedelta(hours=5, minutes=30)


async def _org(p: Env):  # type: ignore[no-untyped-def]
    org = await p.org()
    await p.add_client(org, "Rajendra Jewellers")
    return org


async def _due(p: Env) -> list:  # type: ignore[type-arg]
    return await send_due(p.sf, load_pack(), p.clock())


async def _visit(p: Env, **kw: object) -> None:
    out = await p.voice(REP_CHAT, TRANSCRIPT, extraction_json(**kw))
    await p.tap(REP_CHAT, button_data(to(REP_CHAT, out)[0], "Haan"))


# --- the command -----------------------------------------------------------------------------


async def test_the_boss_turns_the_report_on_and_off(pipe: Env) -> None:
    org = await _org(pipe)
    assert "band" in (await pipe.say(BOSS_CHAT, "/dailyreport"))[-1].text  # off by default
    assert "roz 20:00 baje" in (await pipe.say(BOSS_CHAT, "/dailyreport 20"))[-1].text
    assert "roz 20:00 baje" in (await pipe.say(BOSS_CHAT, "/dailyreport"))[-1].text
    async with pipe.sf() as s:
        assert (
            TenantSettings.from_json(
                (await s.get(Tenant, org.tenant_id)).settings
            ).daily_report_hour
            == 20
        )  # type: ignore[union-attr]
    assert "band kar diya" in (await pipe.say(BOSS_CHAT, "/dailyreport off"))[-1].text
    assert [a.detail for a in await pipe.all(AuditLog) if a.action == "daily_report_configured"] == [
        {"hour": 20}, {"hour": None}]  # fmt: skip


@pytest.mark.parametrize("bad", ["24", "-1", "25", "evening", "20.5", "8pm"])
async def test_bad_hours_are_refused(pipe: Env, bad: str) -> None:
    await _org(pipe)
    assert "Jaise: /dailyreport 20" in (await pipe.say(BOSS_CHAT, f"/dailyreport {bad}"))[-1].text
    assert "band" in (await pipe.say(BOSS_CHAT, "/dailyreport"))[-1].text


async def test_hour_zero_is_valid_midnight(pipe: Env) -> None:
    await _org(pipe)
    assert "roz 0:00 baje" in (await pipe.say(BOSS_CHAT, "/dailyreport 0"))[-1].text


async def test_only_the_boss_can_configure_it(pipe: Env) -> None:
    await _org(pipe)
    assert "sirf boss" in (await pipe.say(REP_CHAT, "/dailyreport 20"))[-1].text
    assert "sirf boss" in (await pipe.say(REP_CHAT, "/dailyreport"))[-1].text


def test_the_setting_is_validated() -> None:
    assert TenantSettings().daily_report_hour is None
    assert TenantSettings.from_json({"daily_report_hour": 0}).daily_report_hour == 0
    for bad in (-1, 24):
        with pytest.raises(ValidationError):
            TenantSettings.from_json({"daily_report_hour": bad})


# --- sending ---------------------------------------------------------------------------------


async def test_nothing_is_sent_unless_the_tenant_opted_in(pipe: Env) -> None:
    await _org(pipe)
    await _visit(pipe)
    pipe.clock.advance(hours=12)
    assert await _due(pipe) == []


async def test_it_waits_for_the_local_hour_then_sends_exactly_once_a_day(pipe: Env) -> None:
    org = await _org(pipe)
    await pipe.tenant_settings(org, daily_report_hour=20)
    await _visit(pipe)
    assert await _due(pipe) == []  # 14:30 IST: too early
    pipe.clock.advance(hours=5, minutes=29)  # 19:59 IST
    assert await _due(pipe) == []
    pipe.clock.advance(minutes=1)  # 20:00 IST: exactly the hour
    (msg,) = await _due(pipe)
    assert msg.chat_id == BOSS_CHAT
    assert await _due(pipe) == []  # already sent today
    pipe.clock.advance(hours=2)
    assert await _due(pipe) == []  # still today
    pipe.clock.advance(days=1)
    assert len(await _due(pipe)) == 1  # tomorrow's report
    assert [a.detail["date"] for a in await pipe.all(AuditLog) if a.action == "daily_report_sent"] == [
        "2026-11-02", "2026-11-03"]  # fmt: skip


async def test_the_report_contains_only_saved_numbers(pipe: Env) -> None:
    org = await _org(pipe)
    await pipe.add_rules(org, ("max_order_value_without_approval", {"max_value_paise": 30_000_000}))
    await pipe.tenant_settings(org, daily_report_hour=20)
    await _visit(pipe, value_paise=20_000_000)  # saved: Rs 2,00,000
    await _visit(pipe, value_paise=41_200_000)  # over the limit: waiting for approval
    out = await pipe.voice(REP_CHAT, TRANSCRIPT, extraction_json())
    await pipe.tap(REP_CHAT, button_data(to(REP_CHAT, out)[0], "Nahi"))  # rejected: never counted
    await pipe.voice(REP_CHAT, TRANSCRIPT, extraction_json(client=None))  # an unanswered question
    pipe.clock.advance(hours=5, minutes=30)
    (msg,) = await _due(pipe)
    lines = msg.text.split("\n")
    assert lines == [
        "Aaj ka report (2 Nov)",
        "Visits: 2",  # one saved + one awaiting approval
        "Confirm order ki keemat: ₹2,00,000",  # the unapproved Rs 4,12,000 is NOT in the total
        "Approval pending: 1 (₹4,12,000)",
        "Rep-wise: Ramesh 2",
        "Sheet sync fail: 0",
    ]


async def test_earlier_days_are_not_counted(pipe: Env) -> None:
    org = await _org(pipe)
    await pipe.tenant_settings(org, daily_report_hour=20)
    await _visit(pipe)
    pipe.clock.advance(days=1, hours=5, minutes=30)  # tomorrow evening: no new visits
    (msg,) = await _due(pipe)
    assert "Visits: 0" in msg.text and "₹0" in msg.text and "Rep-wise: -" in msg.text


async def test_a_boss_without_telegram_or_consent_gets_nothing_but_the_day_is_marked(
    pipe: Env,
) -> None:
    org = await _org(pipe)
    await pipe.tenant_settings(org, daily_report_hour=0)
    async with pipe.sf() as s, s.begin():
        (await s.get(User, org.boss_id)).consent_at = None  # type: ignore[union-attr]
    assert await _due(pipe) == []
    async with pipe.sf() as s:
        t = await s.get(Tenant, org.tenant_id)
        assert TenantSettings.from_json(t.settings).last_daily_report == "2026-11-02"  # type: ignore[union-attr]


async def test_each_tenant_has_its_own_schedule_and_audience(pipe: Env) -> None:
    org = await _org(pipe)
    await pipe.tenant_settings(org, daily_report_hour=20)
    _, _, raw = await pipe.new_tenant("Other Co", "Mallory")
    await pipe.say(3003, f"/start {raw}")
    await pipe.tap(3003, "co:y")  # Other Co never opted in
    pipe.clock.advance(hours=6)
    msgs = await _due(pipe)
    assert [m.chat_id for m in msgs] == [BOSS_CHAT]


async def test_it_follows_the_tenants_timezone(pipe: Env) -> None:
    org = await _org(pipe)
    await pipe.tenant_settings(org, daily_report_hour=9)
    async with pipe.sf() as s, s.begin():
        (await s.get(Tenant, org.tenant_id)).timezone = "Asia/Tokyo"  # type: ignore[union-attr]
    # 09:00 UTC is already 18:00 in Tokyo, well past 09:00 local
    assert len(await _due(pipe)) == 1


async def test_two_runs_at_the_same_moment_send_one_report(pipe: Env) -> None:
    org = await _org(pipe)
    await pipe.tenant_settings(org, daily_report_hour=0)
    first, second = await _due(pipe), await _due(pipe)
    assert len(first) == 1 and second == []
