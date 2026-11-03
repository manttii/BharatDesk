"""SET_BENCHMARK (boss only, read back) and the client dossier (built only from saved visits)."""

from __future__ import annotations

from sqlalchemy import select

from app.models import ActivityLog, AuditLog, BenchmarkRate, ConversationState
from app.models.enums import ActivityStatus
from tests.conftest import (
    BOSS_CHAT,
    REP_CHAT,
    TRANSCRIPT,
    Env,
    button_data,
    extraction_json,
    to,
)

RATE_TRANSCRIPT = TRANSCRIPT + ", sattar hazaar rate"


async def _org(p: Env):  # type: ignore[no-untyped-def]
    org = await p.org()
    await p.add_client(org, "Rajendra Jewellers")
    return org


# --- benchmark: /rate ----------------------------------------------------------------------


async def test_boss_sets_the_rate_with_a_readback(pipe: Env) -> None:
    org = await _org(pipe)
    out = await pipe.say(BOSS_CHAT, "/rate 75200")
    msg = out[-1]
    assert msg.text == "Aaj ka rate ₹75,200 (10 gram) set karna hai?"
    assert await pipe.all(BenchmarkRate) == []  # nothing applied before the tap
    out = await pipe.tap(BOSS_CHAT, button_data(msg, "Haan"))
    assert "Rate set ho gaya: ₹75,200" in out[-1].text
    (row,) = await pipe.all(BenchmarkRate)
    assert (row.value_paise, row.unit, row.set_by, row.set_at) == (
        7_520_000,
        "10g",
        org.boss_id,
        pipe.clock(),
    )
    assert any(a.action == "benchmark_set" for a in await pipe.all(AuditLog))


async def test_rate_command_without_an_argument_shows_the_current_rate(pipe: Env) -> None:
    org = await _org(pipe)
    assert "koi rate set nahi" in (await pipe.say(BOSS_CHAT, "/rate"))[-1].text
    await pipe.set_benchmark(org, 7_520_000, age_s=3 * 3600)
    assert "₹75,200" in (await pipe.say(BOSS_CHAT, "/rate"))[-1].text
    assert "3 ghante" in (await pipe.say(BOSS_CHAT, "/rate"))[-1].text


async def test_spoken_amounts_work_in_the_command(pipe: Env) -> None:
    await _org(pipe)
    out = await pipe.say(BOSS_CHAT, "/rate pachattar hazaar do sau")
    assert "₹75,200" in out[-1].text


async def test_only_the_boss_can_set_the_rate(pipe: Env) -> None:
    await _org(pipe)
    out = await pipe.say(REP_CHAT, "/rate 75200")
    assert "sirf boss" in out[-1].text and await pipe.states() == []
    assert await pipe.all(BenchmarkRate) == []


async def test_out_of_range_or_unclear_rates_are_refused_not_applied(pipe: Env) -> None:
    await _org(pipe)
    assert "range se bahar" in (await pipe.say(BOSS_CHAT, "/rate 500"))[-1].text  # Rs 500 / 10 g
    assert "range se bahar" in (await pipe.say(BOSS_CHAT, "/rate 9999999"))[-1].text
    assert "samajh nahi aaya" in (await pipe.say(BOSS_CHAT, "/rate kuch bhi"))[-1].text
    assert (
        "samajh nahi aaya" in (await pipe.say(BOSS_CHAT, "/rate 100 200"))[-1].text
    )  # two numbers
    assert await pipe.all(BenchmarkRate) == [] and await pipe.states() == []


async def test_declining_or_double_confirming_the_rate(pipe: Env) -> None:
    await _org(pipe)
    out = await pipe.say(BOSS_CHAT, "/rate 75200")
    await pipe.tap(BOSS_CHAT, button_data(out[-1], "Nahi"))
    assert await pipe.all(BenchmarkRate) == []
    out = await pipe.say(BOSS_CHAT, "/rate 75200")
    yes = button_data(out[-1], "Haan")
    await pipe.tap(BOSS_CHAT, yes)
    again = await pipe.tap(BOSS_CHAT, yes)
    assert again == [] and len(await pipe.all(BenchmarkRate)) == 1


async def test_a_rep_cannot_press_the_bosss_rate_button(pipe: Env) -> None:
    await _org(pipe)
    out = await pipe.say(BOSS_CHAT, "/rate 75200")
    await pipe.tap(REP_CHAT, button_data(out[-1], "Haan"))
    assert await pipe.all(BenchmarkRate) == []


async def test_the_rate_confirmation_expires(pipe: Env) -> None:
    await _org(pipe)
    out = await pipe.say(BOSS_CHAT, "/rate 75200")
    pipe.clock.advance(minutes=pipe.settings.conversation_ttl_minutes, seconds=1)
    await pipe.tap(BOSS_CHAT, button_data(out[-1], "Haan"))
    assert await pipe.all(BenchmarkRate) == []


async def test_a_new_rate_replaces_the_old_one_and_drives_the_policy(pipe: Env) -> None:
    org = await _org(pipe)
    await pipe.add_rules(org, ("rate_floor_vs_benchmark",
                               {"max_below_bps": 100, "max_benchmark_age_seconds": 86_400}))  # fmt: skip
    out = await pipe.say(BOSS_CHAT, "/rate 75200")
    await pipe.tap(BOSS_CHAT, button_data(out[-1], "Haan"))
    # Rs 70,000 is far below the floor of the Rs 75,200 benchmark -> approval
    out = await pipe.voice(REP_CHAT, RATE_TRANSCRIPT, extraction_json(rate=7_000_000))
    out = await pipe.tap(REP_CHAT, button_data(to(REP_CHAT, out)[0], "Haan"))
    assert to(BOSS_CHAT, out), "below-floor quote must reach the boss"
    # the boss lowers today's rate to Rs 70,000: the same quote is now fine
    out = await pipe.say(BOSS_CHAT, "/rate 70000")
    await pipe.tap(BOSS_CHAT, button_data(out[-1], "Haan"))
    assert len(await pipe.all(BenchmarkRate)) == 2
    out = await pipe.voice(REP_CHAT, RATE_TRANSCRIPT, extraction_json(rate=7_000_000))
    out = await pipe.tap(REP_CHAT, button_data(to(REP_CHAT, out)[0], "Haan"))
    assert "Save ho gaya" in to(REP_CHAT, out)[0].text and not to(BOSS_CHAT, out)


# --- benchmark: by voice -------------------------------------------------------------------


def _bench_json(value: int | None, **kw: object) -> str:
    return extraction_json(intent="SET_BENCHMARK", client=None, order=False, received=None,
                           delivery=None, benchmark_rate_paise=value, **kw)  # fmt: skip


async def test_boss_voice_sets_the_rate_through_the_same_readback(pipe: Env) -> None:
    await _org(pipe)
    out = await pipe.voice(BOSS_CHAT, "aaj ka rate pachattar hazaar do sau", _bench_json(7_520_000))
    msg = to(BOSS_CHAT, out)[0]
    assert msg.text == "Aaj ka rate ₹75,200 (10 gram) set karna hai?"  # agrees with the transcript
    await pipe.tap(BOSS_CHAT, button_data(msg, "Haan"))
    (row,) = await pipe.all(BenchmarkRate)
    assert row.value_paise == 7_520_000


async def test_a_rate_that_disagrees_with_the_audio_is_flagged(pipe: Env) -> None:
    await _org(pipe)
    out = await pipe.voice(BOSS_CHAT, "aaj ka rate pachattar hazaar do sau", _bench_json(7_250_000))
    text = to(BOSS_CHAT, out)[0].text
    assert "₹72,500" in text and "Dhyan" in text and "rate" in text.split("Dhyan")[1]


async def test_a_reps_voice_cannot_set_the_rate(pipe: Env) -> None:
    await _org(pipe)
    out = await pipe.voice(REP_CHAT, "aaj ka rate pachattar hazaar do sau", _bench_json(7_520_000))
    assert "sirf boss" in to(REP_CHAT, out)[0].text
    assert await pipe.all(BenchmarkRate) == []
    assert [s for s in await pipe.all(ConversationState) if s.kind == "rate_confirm"] == []


async def test_a_voice_rate_without_a_number_asks_for_it(pipe: Env) -> None:
    await _org(pipe)
    out = await pipe.voice(BOSS_CHAT, "aaj ka rate set karo", _bench_json(None))
    assert "Rate kitna hai" in to(BOSS_CHAT, out)[0].text


# --- dossier -------------------------------------------------------------------------------


async def _log(p: Env, transcript: str, **kw: object) -> list:  # type: ignore[type-arg]
    out = await p.voice(REP_CHAT, transcript, extraction_json(**kw))
    return await p.tap(REP_CHAT, button_data(to(REP_CHAT, out)[0], "Haan"))


def _ask(client: str | None) -> str:
    return extraction_json(intent="CLIENT_DOSSIER_QUERY", client=client, order=False,
                           received=None, delivery=None)  # fmt: skip


SECOND = (
    "Rajendra Jewellers, paanch sau gram, do lakh ka order, pachas hazaar advance, "
    "20 November ko delivery, 25 November ko follow-up"
)


async def _two_visits(p: Env) -> None:
    await _log(p, TRANSCRIPT)
    await _log(p, SECOND, weight_mg=500_000, value_paise=20_000_000, received=5_000_000,
               delivery="20 November", followup="25 November")  # fmt: skip


async def test_dossier_is_three_lines_from_saved_visits(pipe: Env) -> None:
    await _org(pipe)
    await _two_visits(pipe)
    out = await pipe.note(
        BOSS_CHAT, "Rajendra Jewellers ka status batao", _ask("Rajendra Jewellers")
    )
    lines = to(BOSS_CHAT, out)[0].text.split("\n")
    assert lines == [
        "Rajendra Jewellers: 2 visit, aakhri visit 2 Nov (Ramesh). Kul order ₹6,12,000.",
        "Baaki (udhaar): ₹5,17,000. Aakhri order ₹2,00,000, advance ₹50,000.",
        "Agli delivery: 8 Nov. Follow-up: 25 Nov.",
    ]


async def test_dossier_works_for_a_rep_and_by_voice(pipe: Env) -> None:
    await _org(pipe)
    await _log(pipe, TRANSCRIPT)
    out = await pipe.voice(
        REP_CHAT, "Rajendra Jewellers ka status batao", _ask("Rajendra Jwellers")
    )
    assert to(REP_CHAT, out)[0].text.startswith("Rajendra Jewellers: 1 visit")


async def test_dossier_counts_only_saved_visits_and_mentions_pending_approvals(pipe: Env) -> None:
    org = await _org(pipe)
    await _log(pipe, TRANSCRIPT)
    await pipe.add_rules(org, ("max_order_value_without_approval", {"max_value_paise": 10_000_000}))
    await _log(pipe, SECOND, weight_mg=500_000, value_paise=20_000_000, received=5_000_000,
               delivery="20 November", followup="25 November")  # fmt: skip
    # second visit is above the limit -> awaiting approval: NOT counted as a visit or as credit
    out = await pipe.note(BOSS_CHAT, "status", _ask("Rajendra Jewellers"))
    text = to(BOSS_CHAT, out)[0].text
    assert "1 visit" in text and "₹4,12,000" in text and "₹3,67,000" in text
    assert "Approval pending: 1." in text
    assert "20 Nov" not in text and "25 Nov" not in text  # unconfirmed dates are never shown


async def test_rejected_visits_never_appear(pipe: Env) -> None:
    await _org(pipe)
    out = await pipe.voice(REP_CHAT, TRANSCRIPT, extraction_json())
    await pipe.tap(REP_CHAT, button_data(to(REP_CHAT, out)[0], "Nahi"))
    out = await pipe.note(BOSS_CHAT, "status", _ask("Rajendra Jewellers"))
    assert "koi save kiya hua record nahi" in to(BOSS_CHAT, out)[0].text


async def test_dossier_for_unknown_ambiguous_and_missing_names(pipe: Env) -> None:
    org = await pipe.org()
    await pipe.add_client(org, "Rajendra Jewellers")
    await pipe.add_client(org, "Rajendra Gems")
    out = await pipe.note(BOSS_CHAT, "x", _ask("Totally Different Traders"))
    assert '"Totally Different Traders" naam ka koi client nahi' in to(BOSS_CHAT, out)[0].text
    out = await pipe.note(BOSS_CHAT, "x", _ask("Rajendra"))
    assert (
        "Kaun sa client?" in to(BOSS_CHAT, out)[0].text
        and "Rajendra Gems" in to(BOSS_CHAT, out)[0].text
    )
    out = await pipe.note(BOSS_CHAT, "x", _ask(None))
    assert "Kis client ka status" in to(BOSS_CHAT, out)[0].text


async def test_dossier_cannot_invent_anything_the_llm_says(pipe: Env) -> None:
    """The LLM only supplies the NAME; every figure comes from the database."""
    await _org(pipe)
    reply = extraction_json(intent="CLIENT_DOSSIER_QUERY", client="Rajendra Jewellers", order=False,
                            received=None, delivery=None, notes="owes Rs 99,99,999 and ordered 5 kg")  # fmt: skip
    out = await pipe.note(BOSS_CHAT, "status", reply)
    text = to(BOSS_CHAT, out)[0].text
    assert (
        "99,99,999" not in text and "5 kg" not in text and "koi save kiya hua record nahi" in text
    )


async def test_dossier_does_not_cross_tenants(pipe: Env) -> None:
    await _org(pipe)
    await _log(pipe, TRANSCRIPT)
    _, _, raw = await pipe.new_tenant("Other Co", "Mallory")
    await pipe.say(3003, f"/start {raw}")
    await pipe.tap(3003, "co:y")
    out = await pipe.note(3003, "status", _ask("Rajendra Jewellers"))
    assert "naam ka koi client nahi" in to(3003, out)[0].text


async def test_dossier_query_creates_no_activity(pipe: Env) -> None:
    await _org(pipe)
    await pipe.note(BOSS_CHAT, "status", _ask("Rajendra Jewellers"))
    acts = await pipe.all(ActivityLog)
    assert all(a.status is not ActivityStatus.PENDING_CONFIRMATION for a in acts)
    async with pipe.sf() as s:
        assert (await s.execute(select(ActivityLog))).scalars().all() == []
