"""The pilot report (real data only, hand-checked numbers) and the demo seed."""

from __future__ import annotations

from datetime import timedelta
from fractions import Fraction

import pytest
from sqlalchemy import select

from app.models import ActivityLog, BenchmarkRate, Client, PolicyRule, Tenant, User
from app.models.enums import ActivityStatus
from app.providers.stt import STTError
from app.services.pilot_report import (
    PilotReport,
    _duration,
    _one_decimal,
    build_report,
    render_markdown,
)
from scripts import pilot_report as pilot_cli
from scripts.seed import DEMO_CLIENTS, SeedError, seed
from tests.conftest import (
    BOSS_CHAT,
    REP_CHAT,
    T0,
    TRANSCRIPT,
    Env,
    button_data,
    extract_token,
    extraction_json,
    to,
)

SMALL = (
    "Rajendra Jewellers, dedh sau gram, pachas hazaar ka order, paintalis hazaar advance, "
    "8 November delivery"
)


async def _scenario(p: Env):  # type: ignore[no-untyped-def]
    """4 voice notes processed (A, B, C, C'), 1 failed, 1 correction, 2 approvals (120 s, 600 s)."""
    org = await p.org()
    await p.add_client(org, "Rajendra Jewellers")
    await p.add_rules(org, ("max_order_value_without_approval", {"max_value_paise": 10_000_000}))
    # A: over the limit; the boss approves after 120 s
    out = await p.voice(REP_CHAT, TRANSCRIPT, extraction_json())
    out = await p.tap(REP_CHAT, button_data(to(REP_CHAT, out)[0], "Haan"))
    p.clock.advance(seconds=120)
    await p.tap(BOSS_CHAT, button_data(to(BOSS_CHAT, out)[0], "Approve"))
    # B: over the limit; the boss rejects after 600 s
    out = await p.voice(REP_CHAT, TRANSCRIPT, extraction_json())
    out = await p.tap(REP_CHAT, button_data(to(REP_CHAT, out)[0], "Haan"))
    p.clock.advance(seconds=600)
    await p.tap(BOSS_CHAT, button_data(to(BOSS_CHAT, out)[0], "Reject"))
    # C: under the limit; the rep corrects it (Sudharo) and sends it again
    small = extraction_json(value_paise=5_000_000)
    out = await p.voice(REP_CHAT, SMALL, small)
    out = await p.tap(REP_CHAT, button_data(to(REP_CHAT, out)[0], "Sudharo"))
    out = await p.voice(REP_CHAT, SMALL, small)
    await p.tap(REP_CHAT, button_data(to(REP_CHAT, out)[0], "Haan"))
    # one voice note that could not be processed
    p.stt.results.append(STTError("boom"))  # type: ignore[union-attr]
    await p.voice(REP_CHAT, None, None)
    return org


# --- the report ----------------------------------------------------------------------------


async def test_an_empty_database_says_no_data_and_prints_no_zero_figures(pipe: Env) -> None:
    await pipe.org()
    text = render_markdown(await build_report(pipe.sf))
    assert "**no data**" in text and "voice notes processed" not in text and "0.0" not in text


async def test_every_figure_is_computed_from_real_rows(pipe: Env) -> None:
    await _scenario(pipe)
    rep = await build_report(pipe.sf)
    assert rep.tenants == 1
    assert rep.notes_by_kind == {"voice": 4} and rep.voice_notes == 4 and rep.notes_processed == 4
    assert rep.notes_not_processed == 1  # the failed STT
    assert rep.corrections == 1 and rep.corrections_per_100_notes == Fraction(25)
    assert rep.approvals_by_status == {"approved": 1, "rejected": 1} and rep.approvals_total == 2
    assert sorted(rep.approval_seconds) == [120, 600] and rep.median_seconds_to_approve == 360
    (ramesh,) = [r for r in rep.reps if r.name == "Ramesh"]
    assert (ramesh.notes, ramesh.saved_visits, ramesh.corrections, ramesh.approvals_requested) == (
        4,
        2,
        1,
        2,
    )


async def test_the_markdown_has_the_headline_numbers(pipe: Env) -> None:
    await _scenario(pipe)
    text = render_markdown(await build_report(pipe.sf))
    for needle in (
        "**voice notes processed: 4**",
        "**corrections per 100 notes: 25.0** (1 corrections over 4 notes)",
        "**approval requests: 2** (approved: 1, rejected: 1)",
        "**median time to approve: 6m 0s** (over 2 resolved)",
        "notes that failed or were rejected: 1",
        "| Ramesh | 4 | 2 | 1 | 2 |",
    ):
        assert needle in text, needle
    assert "no data" not in text


async def test_demo_tenants_are_excluded_unless_asked(pipe: Env) -> None:
    org = await _scenario(pipe)
    await pipe.tenant_settings(org, demo=True)
    rep = await build_report(pipe.sf)
    assert rep.tenants == 0 and "**no data**" in render_markdown(rep)
    assert (await build_report(pipe.sf, include_demo=True)).voice_notes == 4


async def test_the_window_is_inclusive_of_since_and_exclusive_of_until(pipe: Env) -> None:
    await _scenario(pipe)
    end = pipe.clock()  # everything above happened at or before this instant
    assert (await build_report(pipe.sf, since=T0)).voice_notes == 4  # since is inclusive
    assert (await build_report(pipe.sf, since=end + timedelta(seconds=1))).voice_notes == 0
    assert (await build_report(pipe.sf, until=T0)).voice_notes == 0  # until is exclusive
    assert (await build_report(pipe.sf, until=end + timedelta(days=1))).voice_notes == 4


async def test_tenants_are_aggregated_separately_but_reported_together(pipe: Env) -> None:
    await _scenario(pipe)
    _, _, raw = await pipe.new_tenant("Other Co", "Mallory")
    await pipe.say(3003, f"/start {raw}")
    await pipe.tap(3003, "co:y")
    await pipe.voice(3003, TRANSCRIPT, extraction_json(client="Zed"))
    rep = await build_report(pipe.sf)
    assert rep.tenants == 2 and rep.voice_notes == 5
    only = await build_report(pipe.sf, only_tenant=(await pipe.all(Tenant))[1].id)
    assert only.tenants == 1 and only.voice_notes in (1, 4)


def test_median_and_rounding_are_exact() -> None:
    r = PilotReport(approval_seconds=[1, 2])
    assert r.median_seconds_to_approve == Fraction(3, 2)
    assert PilotReport(approval_seconds=[5, 1, 9]).median_seconds_to_approve == 5
    assert PilotReport().median_seconds_to_approve is None
    assert [_one_decimal(Fraction(n, d)) for n, d in ((1, 3), (2, 3), (1, 20), (25, 1), (0, 1), (1, 2))] == [
        "0.3", "0.7", "0.1", "25.0", "0.0", "0.5"]  # fmt: skip
    assert [_duration(Fraction(s)) for s in (0, 59, 60, 360, 3600, 3725)] == [
        "0s",
        "59s",
        "1m 0s",
        "6m 0s",
        "1h 0m 0s",
        "1h 2m 5s",
    ]


def test_no_notes_means_no_corrections_rate() -> None:
    assert PilotReport(corrections=3).corrections_per_100_notes is None


async def test_the_report_cli_prints_no_data_for_an_empty_database(
    pipe: Env, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from app.config import get_settings

    monkeypatch.setenv("DATABASE_URL", pipe.settings.database_url)
    get_settings.cache_clear()
    assert await pilot_cli.main([]) == 0
    assert "**no data**" in capsys.readouterr().out
    get_settings.cache_clear()


# --- the demo seed -------------------------------------------------------------------------


async def _seed(p: Env, **kw):  # type: ignore[no-untyped-def]
    p.settings.telegram_bot_username = "DemoBot"
    return await seed(p.sf, p.settings, tenant_name="Demo Jewellers", boss_name="Asha",
                      workers=["Ramesh", "Suresh"], history=kw.get("history", True), now=p.clock())  # fmt: skip


async def test_seed_creates_a_working_demo_tenant(pipe: Env) -> None:
    res = await _seed(pipe)
    assert res.boss_link.startswith("https://t.me/DemoBot?start=") and set(res.worker_links) == {
        "Ramesh",
        "Suresh",
    }
    rules = await pipe.all(PolicyRule)
    assert sorted(r.rule_type for r in rules) == ["credit_limit_per_client", "max_order_value_without_approval",
                                                  "rate_floor_vs_benchmark"]  # fmt: skip
    (bench,) = await pipe.all(BenchmarkRate)
    assert (bench.value_paise, bench.unit) == (7_520_000, "10g")
    assert sorted(c.name for c in await pipe.all(Client)) == sorted(DEMO_CLIENTS)
    assert len(await pipe.all(ActivityLog)) == 6
    (tenant,) = await pipe.all(Tenant)
    assert tenant.settings == {"demo": True}
    users = await pipe.all(User)
    assert sorted(u.name for u in users) == ["Asha", "Ramesh", "Suresh"] and all(
        u.telegram_chat_id is None for u in users
    )


async def test_seed_without_history_adds_no_visits(pipe: Env) -> None:
    await _seed(pipe, history=False)
    assert await pipe.all(ActivityLog) == [] and len(await pipe.all(Client)) == 3


async def test_seed_refuses_to_run_twice_for_the_same_name(pipe: Env) -> None:
    await _seed(pipe)
    with pytest.raises(SeedError, match="already exists"):
        await _seed(pipe)
    assert len(await pipe.all(Tenant)) == 1


async def test_seeded_demo_data_is_not_pilot_evidence(pipe: Env) -> None:
    await _seed(pipe)
    assert "**no data**" in render_markdown(await build_report(pipe.sf))


async def test_the_seeded_links_work_and_the_seeded_history_feeds_the_dossier(pipe: Env) -> None:
    res = await _seed(pipe)
    await pipe.say(BOSS_CHAT, f"/start {extract_token(res.boss_link)}")
    await pipe.tap(BOSS_CHAT, "co:y")
    ask = extraction_json(intent="CLIENT_DOSSIER_QUERY", client="Rajendra Jewellers", order=False,
                          received=None, delivery=None)  # fmt: skip
    out = await pipe.note(BOSS_CHAT, "status", ask)
    lines = to(BOSS_CHAT, out)[0].text.split("\n")
    assert lines[0].startswith("Rajendra Jewellers: 2 visit,") and lines[0].endswith(
        "Kul order ₹6,12,000."
    )
    assert lines[1].startswith("Baaki (udhaar): ₹5,17,000.")  # 3,67,000 + 1,50,000 unpaid


async def test_the_seeded_policy_routes_a_low_quote_to_the_boss(pipe: Env) -> None:
    res = await _seed(pipe)
    await pipe.say(BOSS_CHAT, f"/start {extract_token(res.boss_link)}")
    await pipe.tap(BOSS_CHAT, "co:y")
    await pipe.say(REP_CHAT, f"/start {extract_token(res.worker_links['Ramesh'])}")
    await pipe.tap(REP_CHAT, "co:y")
    said = TRANSCRIPT + ", sattar hazaar rate"  # Rs 70,000 vs a Rs 75,200 benchmark
    out = await pipe.voice(REP_CHAT, said, extraction_json(rate=7_000_000))
    out = await pipe.tap(REP_CHAT, button_data(to(REP_CHAT, out)[0], "Haan"))
    assert to(BOSS_CHAT, out) and "benchmark" in to(BOSS_CHAT, out)[0].text
    acts = [a for a in await pipe.all(ActivityLog) if a.status is ActivityStatus.PENDING_APPROVAL]
    assert len(acts) == 1
    async with pipe.sf() as s:
        assert (await s.execute(select(PolicyRule))).scalars().all()
