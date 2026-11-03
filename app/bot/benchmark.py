"""Today's benchmark rate (boss only), set by `/rate 75200` or by voice ("aaj ka rate 75200").

The rate drives every rate-floor check, so it is never applied silently: it is read back and
applied only after the boss taps Haan. Out-of-range values are refused outright.
"""

from __future__ import annotations

from datetime import timedelta

from app.bot.context import Ctx
from app.bot.states import consume, expiry
from app.models import BenchmarkRate, ConversationState
from app.models.enums import Role
from app.pipeline.amounts import parse_amount_paise, within_bounds
from app.pipeline.numerals import cross_check_money, parse_quantities
from app.schemas.events import Button
from app.schemas.extraction import LLMExtraction
from app.units import format_inr

RATE_CONFIRM_KIND = "rate_confirm"
RATE_UNIT = "10g"


def _is_boss(ctx: Ctx) -> bool:
    assert ctx.user is not None
    if ctx.user.role is Role.BOSS:
        return True
    ctx.say(ctx.t("boss_only"))
    return False


async def _propose(ctx: Ctx, value_paise: int, *, doubtful: bool) -> None:
    assert ctx.user is not None
    if not within_bounds(ctx.pack, value_paise, "rate_paise_per_10g_min", "rate_paise_per_10g_max"):
        ctx.say(ctx.t("benchmark_out_of_range", rate=format_inr(value_paise)))
        return
    state = ctx.repo.add(
        ConversationState(
            user_id=ctx.user.id,
            kind=RATE_CONFIRM_KIND,
            data={"value_paise": value_paise},
            expires_at=expiry(ctx),
        )
    )
    await ctx.repo.flush()
    h = state.id.hex
    text = ctx.t("benchmark_confirm", rate=format_inr(value_paise))
    if doubtful:
        text += "\n" + ctx.t("readback_warn", fields=ctx.t("label_rate"))
    ctx.say(
        text,
        (Button(ctx.t("btn_yes"), f"rt:y:{h}"), Button(ctx.t("btn_no"), f"rt:n:{h}")),
    )


async def show_current(ctx: Ctx) -> None:
    rows = await ctx.repo.list(
        BenchmarkRate, order_by=[BenchmarkRate.set_at.desc(), BenchmarkRate.created_at.desc()]
    )
    if not rows:
        ctx.say(ctx.t("benchmark_none"))
        return
    age_h = (ctx.now - rows[0].set_at) // timedelta(hours=1)
    ctx.say(ctx.t("benchmark_current", rate=format_inr(rows[0].value_paise), age_h=age_h))


async def handle_rate_command(ctx: Ctx, arg: str) -> None:
    """`/rate 75200` (boss only). With no argument, shows the current rate."""
    if not _is_boss(ctx):
        return
    if not arg:
        await show_current(ctx)
        return
    value = parse_amount_paise(arg)
    if value is None:
        ctx.say(ctx.t("benchmark_bad"))
        return
    await _propose(ctx, value, doubtful=False)


async def from_extraction(ctx: Ctx, parsed: LLMExtraction, transcript: str) -> None:
    """SET_BENCHMARK heard in a voice/text note."""
    if not _is_boss(ctx):
        return
    value = parsed.benchmark_rate_paise
    if value is None:
        ctx.say(ctx.t("benchmark_ask"))
        return
    agrees = cross_check_money(value, parse_quantities(transcript)).agrees
    doubtful = (
        not agrees
    ) or parsed.confidence.benchmark_rate_paise < ctx.settings.low_confidence_flag
    await _propose(ctx, value, doubtful=doubtful)


async def handle_callback(ctx: Ctx, action: str, raw_id: str) -> None:
    assert ctx.user is not None
    if ctx.user.role is not Role.BOSS:
        ctx.say(ctx.t("boss_only"))
        return
    state = await consume(ctx, raw_id, (RATE_CONFIRM_KIND,))
    if state is None:
        ctx.callback_text = ctx.t("state_expired")
        return
    if action != "y":
        ctx.say(ctx.t("addworker_cancelled"))
        ctx.callback_text = ctx.t("action_toast_cancelled")
        return
    value = int(state.data["value_paise"])
    ctx.repo.add(
        BenchmarkRate(value_paise=value, unit=RATE_UNIT, set_by=ctx.user.id, set_at=ctx.now)
    )
    ctx.repo.audit(
        "benchmark_set",
        entity="benchmark_rate",
        actor_user_id=ctx.user.id,
        detail={"value_paise": value, "unit": RATE_UNIT},
    )
    ctx.say(ctx.t("benchmark_set", rate=format_inr(value)))
    ctx.callback_text = ctx.t("action_toast_done")
