"""Policy engine -> approval request -> boss buttons -> resolution, expiry and reminders."""

from __future__ import annotations

from app.models import ActivityLog, ApprovalRequest, AuditLog, PolicyRule
from app.models.enums import ActivityStatus, ApprovalStatus
from app.pack import load_pack
from app.services import approvals
from app.units import format_inr
from tests.conftest import (
    BOSS_CHAT,
    REP_CHAT,
    TRANSCRIPT,
    Env,
    Org,
    button_data,
    extraction_json,
    to,
)

RATE_TRANSCRIPT = TRANSCRIPT + ", sattar hazaar rate"  # Rs 70,000 per 10 g = 7,000,000 paise
FLOOR = 7_444_800  # benchmark Rs 75,200 less 1%
RATE_RULE = ("rate_floor_vs_benchmark", {"max_below_bps": 100, "max_benchmark_age_seconds": 86_400})


async def _policy_org(p: Env, *rules: tuple[str, dict]) -> Org:  # type: ignore[type-arg]
    org = await p.org()
    await p.add_client(org, "Rajendra Jewellers")
    await p.add_rules(org, *(rules or (RATE_RULE,)))
    await p.set_benchmark(org)
    return org


async def _submit(p: Env, *, rate: int | None, transcript: str = RATE_TRANSCRIPT, **kw) -> list:  # type: ignore[no-untyped-def,type-arg]
    """Rep sends the note and taps Haan; returns everything sent after the tap."""
    out = await p.voice(REP_CHAT, transcript, extraction_json(rate=rate, **kw))
    readback = to(REP_CHAT, out)[0]
    return await p.tap(REP_CHAT, button_data(readback, "Haan"))


async def _only_approval(p: Env) -> ApprovalRequest:
    (a,) = await p.all(ApprovalRequest)
    return a  # type: ignore[no-any-return]


# --- a violation always creates an approval and never commits ------------------------------


async def test_quote_below_floor_creates_an_approval_and_notifies_the_boss(pipe: Env) -> None:
    await _policy_org(pipe)
    out = await _submit(pipe, rate=7_000_000)
    (act,) = await pipe.all(ActivityLog)
    assert act.status is ActivityStatus.PENDING_APPROVAL  # NOT confirmed, therefore never synced
    approval = await _only_approval(pipe)
    assert approval.status is ApprovalStatus.PENDING
    assert approval.numbers["violations"][0]["numbers"] == {
        "quoted_paise": 7_000_000, "benchmark_paise": 7_520_000,
        "floor_paise": FLOOR, "max_below_bps": 100,
    }  # fmt: skip

    (rep_msg,) = to(REP_CHAT, out)
    assert "Boss ki approval" in rep_msg.text and "benchmark" in rep_msg.text
    (boss_msg,) = to(BOSS_CHAT, out)
    for needle in ("Ramesh", "Rajendra Jewellers", "₹70,000", "₹75,200", "₹74,448", "chaar lakh"):
        assert needle in boss_msg.text, needle  # rep, client, numbers, reason, transcript excerpt
    assert [b.text for b in boss_msg.buttons[0]] == ["Approve", "Reject", "Counter"]  # type: ignore[index]


async def test_quote_exactly_at_the_floor_needs_no_approval(pipe: Env) -> None:
    await _policy_org(pipe)
    transcript = TRANSCRIPT + ", chauhattar hazaar chaar sau adtalis rate"  # Rs 74,448
    out = await _submit(pipe, rate=FLOOR, transcript=transcript)
    assert "Save ho gaya" in to(REP_CHAT, out)[0].text and to(BOSS_CHAT, out) == []
    assert await pipe.all(ApprovalRequest) == []


async def test_one_paisa_below_the_floor_needs_approval(pipe: Env) -> None:
    await _policy_org(pipe)
    transcript = TRANSCRIPT + ", chauhattar hazaar chaar sau adtalis rate"
    await _submit(pipe, rate=FLOOR - 1, transcript=transcript)
    assert (await _only_approval(pipe)).status is ApprovalStatus.PENDING


async def test_missing_benchmark_fails_closed_to_approval(pipe: Env) -> None:
    org = await pipe.org()
    await pipe.add_client(org, "Rajendra Jewellers")
    await pipe.add_rules(org, RATE_RULE)  # no benchmark set
    out = await _submit(pipe, rate=7_600_000, transcript=TRANSCRIPT + ", chhiyattar hazaar rate")
    approval = await _only_approval(pipe)
    assert approval.numbers["violations"][0]["code"] == "benchmark_missing"
    assert "benchmark rate set nahi" in to(BOSS_CHAT, out)[0].text


async def test_stale_benchmark_fails_closed_to_approval(pipe: Env) -> None:
    org = await pipe.org()
    await pipe.add_client(org, "Rajendra Jewellers")
    await pipe.add_rules(org, RATE_RULE)
    await pipe.set_benchmark(org, age_s=86_400 + 1)
    await _submit(pipe, rate=7_600_000, transcript=TRANSCRIPT + ", chhiyattar hazaar rate")
    assert (await _only_approval(pipe)).numbers["violations"][0]["code"] == "benchmark_stale"


async def test_a_visit_without_an_order_has_no_policy_to_check(pipe: Env) -> None:
    await _policy_org(pipe)
    out = await pipe.voice(REP_CHAT, "Rajendra Jewellers gaya tha",
                           extraction_json(order=False, received=None, delivery=None))  # fmt: skip
    out = await pipe.tap(REP_CHAT, button_data(to(REP_CHAT, out)[0], "Haan"))
    assert "Save ho gaya" in to(REP_CHAT, out)[0].text


async def test_malformed_rule_in_the_database_fails_closed(pipe: Env) -> None:
    org = await _policy_org(pipe)
    async with pipe.sf() as s, s.begin():
        from app.db import TenantRepository

        TenantRepository(s, org.tenant_id).add(
            PolicyRule(rule_type="discount_cap_pct", params={"max_discount_bps": 0.3})
        )
    out = await _submit(pipe, rate=7_600_000, transcript=TRANSCRIPT + ", chhiyattar hazaar rate")
    codes = [v["code"] for v in (await _only_approval(pipe)).numbers["violations"]]
    assert "config_invalid" in codes and to(BOSS_CHAT, out)


async def test_boss_not_started_holds_the_request_with_a_clear_message(pipe: Env) -> None:
    org = await pipe.org(with_rep=False)
    await pipe.add_client(org, "Rajendra Jewellers")
    await pipe.add_rules(org, RATE_RULE)
    await pipe.set_benchmark(org)
    # make the boss "not started": drop their chat binding
    from app.models import User

    async with pipe.sf() as s, s.begin():
        for u in await pipe.all(User):
            (await s.get(User, u.id)).telegram_chat_id = None  # type: ignore[union-attr]
    # a rep joins via a fresh invite from the (now unreachable) boss: simulate with a direct row
    async with pipe.sf() as s, s.begin():
        from app.db import TenantRepository
        from app.models.enums import Role

        TenantRepository(s, org.tenant_id).add(
            User(
                name="Ramesh",
                role=Role.WORKER,
                telegram_chat_id=REP_CHAT,
                consent_at=pipe.clock(),
                consent_version="v1",
            )
        )
    out = await _submit(pipe, rate=7_000_000)
    assert "hold par hai" in to(REP_CHAT, out)[0].text and to(BOSS_CHAT, out) == []
    assert (await _only_approval(pipe)).status is ApprovalStatus.PENDING


# --- resolution ----------------------------------------------------------------------------


async def test_approve_confirms_the_record_and_tells_the_rep(pipe: Env) -> None:
    await _policy_org(pipe)
    out = await _submit(pipe, rate=7_000_000)
    approve = button_data(to(BOSS_CHAT, out)[0], "Approve")
    out = await pipe.tap(BOSS_CHAT, approve)
    assert "Boss ne approve kar diya" in to(REP_CHAT, out)[0].text
    (act,) = await pipe.all(ActivityLog)
    assert act.status is ActivityStatus.CONFIRMED
    approval = await _only_approval(pipe)
    assert approval.status is ApprovalStatus.APPROVED and approval.resolved_at == pipe.clock()
    actions = [a.action for a in await pipe.all(AuditLog)]
    assert "approval_created" in actions and "approval_approved" in actions


async def test_double_approve_is_a_noop(pipe: Env) -> None:
    await _policy_org(pipe)
    out = await _submit(pipe, rate=7_000_000)
    approve = button_data(to(BOSS_CHAT, out)[0], "Approve")
    first = await pipe.tap(BOSS_CHAT, approve)
    second = await pipe.tap(BOSS_CHAT, approve)  # a different callback id: a real second tap
    assert to(REP_CHAT, first) and second == []
    assert "pehle hi nipat" in (pipe.fake.callback_answers[-1][1] or "")
    approvals_applied = [a for a in await pipe.all(AuditLog) if a.action == "approval_approved"]
    assert len(approvals_applied) == 1


async def test_approve_then_reject_keeps_the_first_decision(pipe: Env) -> None:
    await _policy_org(pipe)
    out = await _submit(pipe, rate=7_000_000)
    boss = to(BOSS_CHAT, out)[0]
    await pipe.tap(BOSS_CHAT, button_data(boss, "Approve"))
    await pipe.tap(BOSS_CHAT, button_data(boss, "Reject"))
    assert (await _only_approval(pipe)).status is ApprovalStatus.APPROVED
    (act,) = await pipe.all(ActivityLog)
    assert act.status is ActivityStatus.CONFIRMED


async def test_reject_leaves_the_record_unconfirmed_and_notifies_the_rep(pipe: Env) -> None:
    await _policy_org(pipe)
    out = await _submit(pipe, rate=7_000_000)
    out = await pipe.tap(BOSS_CHAT, button_data(to(BOSS_CHAT, out)[0], "Reject"))
    assert "Boss ne reject kar diya" in to(REP_CHAT, out)[0].text
    (act,) = await pipe.all(ActivityLog)
    assert act.status is ActivityStatus.REJECTED
    assert (await _only_approval(pipe)).status is ApprovalStatus.REJECTED


async def test_a_rep_cannot_approve_their_own_request(pipe: Env) -> None:
    await _policy_org(pipe)
    out = await _submit(pipe, rate=7_000_000)
    approve = button_data(to(BOSS_CHAT, out)[0], "Approve")
    out = await pipe.tap(REP_CHAT, approve)
    assert "sirf boss" in to(REP_CHAT, out)[0].text
    assert (await _only_approval(pipe)).status is ApprovalStatus.PENDING


async def test_another_tenants_boss_cannot_resolve_it(pipe: Env) -> None:
    await _policy_org(pipe)
    out = await _submit(pipe, rate=7_000_000)
    approve = button_data(to(BOSS_CHAT, out)[0], "Approve")
    _, _, raw = await pipe.new_tenant("Other Co", "Mallory")
    await pipe.say(3003, f"/start {raw}")
    await pipe.tap(3003, "co:y")
    await pipe.tap(3003, approve)
    assert (await _only_approval(pipe)).status is ApprovalStatus.PENDING


async def test_approving_after_expiry_is_refused(pipe: Env) -> None:
    await _policy_org(pipe)
    out = await _submit(pipe, rate=7_000_000)
    approve = button_data(to(BOSS_CHAT, out)[0], "Approve")
    pipe.clock.advance(hours=24, seconds=1)
    await pipe.tap(BOSS_CHAT, approve)
    assert (await _only_approval(pipe)).status is ApprovalStatus.PENDING  # sweeper will expire it
    (act,) = await pipe.all(ActivityLog)
    assert act.status is ActivityStatus.PENDING_APPROVAL


# --- counter -------------------------------------------------------------------------------


async def test_boss_counters_the_rate_with_a_readback(pipe: Env) -> None:
    await _policy_org(pipe)
    out = await _submit(pipe, rate=7_000_000)
    out = await pipe.tap(BOSS_CHAT, button_data(to(BOSS_CHAT, out)[0], "Counter"))
    assert "Naya rate bhejein" in to(BOSS_CHAT, out)[0].text
    out = await pipe.say(BOSS_CHAT, "74000")
    readback = to(BOSS_CHAT, out)[0]
    assert "₹70,000" in readback.text and "₹74,000" in readback.text
    assert (await _only_approval(pipe)).status is ApprovalStatus.PENDING  # not applied yet
    out = await pipe.tap(BOSS_CHAT, button_data(readback, "Haan"))
    assert "naye terms par approve" in to(REP_CHAT, out)[0].text
    (act,) = await pipe.all(ActivityLog)
    assert act.status is ActivityStatus.CONFIRMED
    assert act.payload["visit"]["order"]["rate_paise_per_10g"] == 7_400_000
    assert act.payload["meta"]["counter"] == {
        "kind": "rate",
        "value": 7_400_000,
        "from_rate": 7_000_000,
    }


async def test_counter_confirm_twice_applies_once(pipe: Env) -> None:
    await _policy_org(pipe)
    out = await _submit(pipe, rate=7_000_000)
    await pipe.tap(BOSS_CHAT, button_data(to(BOSS_CHAT, out)[0], "Counter"))
    out = await pipe.say(BOSS_CHAT, "74000")
    yes = button_data(to(BOSS_CHAT, out)[0], "Haan")
    await pipe.tap(BOSS_CHAT, yes)
    again = await pipe.tap(BOSS_CHAT, yes)
    assert (
        again == []
        and len([a for a in await pipe.all(AuditLog) if a.action == "approval_countered"]) == 1
    )


async def test_declining_the_counter_readback_leaves_the_request_pending(pipe: Env) -> None:
    await _policy_org(pipe)
    out = await _submit(pipe, rate=7_000_000)
    await pipe.tap(BOSS_CHAT, button_data(to(BOSS_CHAT, out)[0], "Counter"))
    out = await pipe.say(BOSS_CHAT, "74000")
    await pipe.tap(BOSS_CHAT, button_data(to(BOSS_CHAT, out)[0], "Nahi"))
    assert (await _only_approval(pipe)).status is ApprovalStatus.PENDING


async def test_unparseable_or_out_of_range_counter_is_asked_again(pipe: Env) -> None:
    await _policy_org(pipe)
    out = await _submit(pipe, rate=7_000_000)
    await pipe.tap(BOSS_CHAT, button_data(to(BOSS_CHAT, out)[0], "Counter"))
    for bad in ("kuch bhi", "5", "do hazaar teen hazaar"):
        out = await pipe.say(BOSS_CHAT, bad)
        assert "Samajh nahi aaya" in to(BOSS_CHAT, out)[0].text, bad
    out = await pipe.say(BOSS_CHAT, "74000")  # the state survived the bad tries
    assert "Lagu karein" in to(BOSS_CHAT, out)[0].text


async def test_boss_counters_a_discount_in_percent(pipe: Env) -> None:
    await _policy_org(pipe, ("discount_cap_pct", {"max_discount_bps": 30}))
    # weight 150 g at the Rs 75,200/10g benchmark lists at Rs 11,28,000; this order is Rs 10,00,000
    out = await _submit(pipe, rate=None, transcript=TRANSCRIPT.replace("chaar lakh baarah hazaar", "das lakh"),
                        value_paise=100_000_000)  # fmt: skip
    boss_msg = to(BOSS_CHAT, out)[0]
    assert "limit 0.3% se zyada" in boss_msg.text
    out = await pipe.tap(BOSS_CHAT, button_data(boss_msg, "Counter"))
    assert "discount %" in to(BOSS_CHAT, out)[0].text
    out = await pipe.say(BOSS_CHAT, "0.3%")
    readback = to(BOSS_CHAT, out)[0]
    new_value = (
        112_800_000 - (112_800_000 * 30) // 10_000
    )  # floor(discount): ties go to the company
    assert "0.3%" in readback.text and format_inr(new_value) in readback.text
    await pipe.tap(BOSS_CHAT, button_data(readback, "Haan"))
    (act,) = await pipe.all(ActivityLog)
    assert (
        act.status is ActivityStatus.CONFIRMED
        and act.payload["visit"]["order"]["value_paise"] == new_value
    )


async def test_a_333_basis_point_counter_that_is_not_whole_is_refused(pipe: Env) -> None:
    await _policy_org(pipe, ("discount_cap_pct", {"max_discount_bps": 30}))
    out = await _submit(pipe, rate=None, transcript=TRANSCRIPT.replace("chaar lakh baarah hazaar", "das lakh"),
                        value_paise=100_000_000)  # fmt: skip
    await pipe.tap(BOSS_CHAT, button_data(to(BOSS_CHAT, out)[0], "Counter"))
    out = await pipe.say(BOSS_CHAT, "0.333%")
    assert "Samajh nahi aaya" in to(BOSS_CHAT, out)[0].text


# --- credit --------------------------------------------------------------------------------


async def test_credit_limit_counts_earlier_confirmed_orders(pipe: Env) -> None:
    await _policy_org(pipe, ("credit_limit_per_client", {"default_limit_paise": 50_000_000}))
    # first visit: unpaid Rs 3,67,000 (36,700,000 paise) is under the Rs 5,00,000 limit
    out = await _submit(pipe, rate=None)
    assert "Save ho gaya" in to(REP_CHAT, out)[0].text
    # second visit adds another Rs 3,67,000 -> Rs 7,34,000 outstanding > limit
    out = await _submit(pipe, rate=None)
    assert to(BOSS_CHAT, out), "second order must go to the boss"
    approval = (await pipe.all(ApprovalRequest))[0]
    v = approval.numbers["violations"][0]
    assert v["code"] == "credit_over_limit" and v["numbers"]["outstanding_paise"] == 36_700_000


# --- expiry and reminders ------------------------------------------------------------------


async def _sweep(p: Env) -> list:  # type: ignore[type-arg]
    msgs = await approvals.sweep(p.sf, load_pack(), p.settings.approval_ttl_hours, p.clock())
    for m in msgs:
        await p.fake.send(m)
    return msgs


async def test_expiry_marks_it_tells_the_rep_and_the_boss_and_never_confirms(pipe: Env) -> None:
    await _policy_org(pipe)
    await _submit(pipe, rate=7_000_000)
    pipe.clock.advance(hours=24)  # exactly at expiry
    msgs = await _sweep(pipe)
    assert {m.chat_id for m in msgs} >= {REP_CHAT, BOSS_CHAT}
    assert "expire ho gayi" in to(REP_CHAT, msgs)[0].text
    assert (await _only_approval(pipe)).status is ApprovalStatus.EXPIRED
    (act,) = await pipe.all(ActivityLog)
    assert act.status is ActivityStatus.REJECTED  # never confirmed, so never synced
    assert await _sweep(pipe) == []  # idempotent: no duplicate notices


async def test_boss_is_reminded_once_halfway_to_expiry(pipe: Env) -> None:
    await _policy_org(pipe)
    await _submit(pipe, rate=7_000_000)
    pipe.clock.advance(hours=11, minutes=59)
    assert await _sweep(pipe) == []
    pipe.clock.advance(minutes=1)  # 12 h = half of 24 h
    msgs = await _sweep(pipe)
    (reminder,) = to(BOSS_CHAT, msgs)
    assert "Yaad dilana" in reminder.text and [b.text for b in reminder.buttons[0]][:2] == [
        "Approve",
        "Reject",
    ]  # type: ignore[index]
    pipe.clock.advance(hours=1)
    assert await _sweep(pipe) == []  # once only
    assert (await _only_approval(pipe)).status is ApprovalStatus.PENDING


async def test_a_resolved_approval_is_not_swept(pipe: Env) -> None:
    await _policy_org(pipe)
    out = await _submit(pipe, rate=7_000_000)
    await pipe.tap(BOSS_CHAT, button_data(to(BOSS_CHAT, out)[0], "Approve"))
    pipe.clock.advance(hours=48)
    assert await _sweep(pipe) == []
    assert (await _only_approval(pipe)).status is ApprovalStatus.APPROVED
