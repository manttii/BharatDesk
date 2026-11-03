"""Slash commands. Boss-only commands are gated on the verified chat_id -> role, and
state-changing ones are read back with buttons before anything is applied."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta

from app.bot import benchmark, bills_ui, notes, sheets_ui, shop_ui
from app.bot.context import Ctx
from app.models import ConversationState, InviteToken, Tenant, User
from app.models.enums import Role
from app.pipeline.confirm import TenantSettings
from app.schemas.events import Button
from app.security import sign_token
from app.services import consent, onboarding

ADDWORKER_KIND = "addworker"
MIN_NAME, MAX_NAME = 2, 60


@dataclass(frozen=True)
class Command:
    name: str
    arg: str


def parse_command(text: str | None) -> Command | None:
    if not text or not text.startswith("/"):
        return None
    head, _, rest = text.partition(" ")
    name = head[1:].split("@", 1)[0].lower()  # "/help@MyBot" -> "help"
    return Command(name=name, arg=rest.strip())


def is_stop_phrase(text: str | None) -> bool:
    return text is not None and " ".join(text.lower().split()) == "band karo"


def clean_name(raw: str) -> str | None:
    name = " ".join(raw.split())
    if not (MIN_NAME <= len(name) <= MAX_NAME) or not name.isprintable():
        return None
    if any(ch in name for ch in "<>{}[]\\"):
        return None
    return name


def _require_boss(ctx: Ctx) -> bool:
    assert ctx.user is not None
    if ctx.user.role is Role.BOSS:
        return True
    ctx.say(ctx.t("boss_only"))
    return False


async def handle_command(ctx: Ctx, cmd: Command) -> None:
    assert ctx.user is not None
    if cmd.name in ("start", "help"):
        key = "help_boss" if ctx.user.role is Role.BOSS else "help_worker"
        ctx.say(ctx.t(key))
    elif cmd.name == "cancel":
        await notes.cancel_pending(ctx)
    elif cmd.name == "sheet":
        await sheets_ui.handle_sheet_command(ctx, cmd.arg)
    elif cmd.name == "bills":
        await bills_ui.pending(ctx)
    elif cmd.name == "reverse":
        await bills_ui.reverse_command(ctx, cmd.arg)
    elif cmd.name == "stock":
        await shop_ui.stock(ctx, cmd.arg.strip() or None)
    elif cmd.name == "today":
        await shop_ui.today(ctx)
    elif cmd.name == "dashboard":
        await _dashboard(ctx)
    elif cmd.name == "dailyreport":
        await _dailyreport(ctx, cmd.arg)
    elif cmd.name == "rate":
        await benchmark.handle_rate_command(ctx, cmd.arg)
    elif cmd.name == "addworker":
        await _addworker(ctx, cmd.arg)
    elif cmd.name == "team":
        await _team(ctx)
    else:
        ctx.say(ctx.t("unknown_command"))


async def withdraw_consent(ctx: Ctx) -> None:
    assert ctx.user is not None
    consent.withdraw(ctx.repo, ctx.user)
    ctx.say(ctx.t("consent_withdrawn"))


async def _addworker(ctx: Ctx, arg: str) -> None:
    assert ctx.user is not None
    if not _require_boss(ctx):
        return
    if not arg:
        ctx.say(ctx.t("addworker_usage"))
        return
    name = clean_name(arg)
    if name is None:
        ctx.say(ctx.t("addworker_bad_name"))
        return
    state = ctx.repo.add(
        ConversationState(
            user_id=ctx.user.id,
            kind=ADDWORKER_KIND,
            data={"name": name},
            expires_at=ctx.now + timedelta(minutes=ctx.settings.pending_action_ttl_minutes),
        )
    )
    await ctx.repo.flush()
    token = state.id.hex  # 32 chars -> "aw:y:<32>" is 37 bytes, under Telegram's 64
    ctx.say(
        ctx.t("addworker_confirm", name=name),
        (
            Button(ctx.t("btn_yes"), f"aw:y:{token}"),
            Button(ctx.t("btn_no"), f"aw:n:{token}"),
        ),
    )


async def _dailyreport(ctx: Ctx, arg: str) -> None:
    """Boss-only: `/dailyreport 20` (local hour), `/dailyreport off`, or no argument to see it."""
    assert ctx.user is not None
    if not _require_boss(ctx):
        return
    tenant = await ctx.session.get(Tenant, ctx.user.tenant_id)
    assert tenant is not None
    current = TenantSettings.from_json(tenant.settings).daily_report_hour
    arg = arg.strip().lower()
    if not arg:
        state = (
            ctx.t("dailyreport_off") if current is None else ctx.t("dailyreport_on", hour=current)
        )
        ctx.say(ctx.t("dailyreport_current", state=state))
        return
    if arg == "off":
        new: int | None = None
    elif arg.isdigit() and 0 <= int(arg) <= 23:
        new = int(arg)
    else:
        ctx.say(ctx.t("dailyreport_usage"))
        return
    tenant.settings = {**tenant.settings, "daily_report_hour": new}  # reassign: JSON column
    ctx.repo.audit(
        "daily_report_configured", entity="tenant", entity_id=str(tenant.id),
        actor_user_id=ctx.user.id, detail={"hour": new},
    )  # fmt: skip
    ctx.say(ctx.t("dailyreport_disabled") if new is None else ctx.t("dailyreport_set", hour=new))


async def _dashboard(ctx: Ctx) -> None:
    """Boss-only: a signed, expiring link. No password, no OTP."""
    assert ctx.user is not None
    if not _require_boss(ctx):
        return
    secret, base = ctx.settings.dashboard_secret, ctx.settings.public_base_url
    if secret is None or not base:
        ctx.say(ctx.t("dashboard_not_configured"))
        return
    ttl = ctx.settings.dashboard_link_ttl_minutes
    exp = int(ctx.now.timestamp()) + ttl * 60
    token = sign_token(
        secret.get_secret_value(),
        {"k": "dash", "t": ctx.user.tenant_id.hex, "u": ctx.user.id.hex, "exp": exp},
    )
    ctx.repo.audit("dashboard_link_issued", entity="user", entity_id=str(ctx.user.id),
                   actor_user_id=ctx.user.id)  # fmt: skip
    ctx.say(ctx.t("dashboard_link", minutes=ttl, link=f"{base.rstrip('/')}/dashboard?t={token}"))


async def _team(ctx: Ctx) -> None:
    if not _require_boss(ctx):
        return
    users = await ctx.repo.list(User, order_by=[User.created_at])
    if not users:
        ctx.say(ctx.t("team_empty"))
        return
    pending_invites = {
        t.user_id for t in await ctx.repo.list(InviteToken, InviteToken.used_at.is_(None))
    }
    lines = [ctx.t("team_header")]
    for u in users:
        if u.telegram_chat_id is None:
            status = ctx.t("team_status_invited") if u.id in pending_invites else "-"
        elif u.consent_at is None:
            status = ctx.t("team_status_pending_consent")
        else:
            status = ctx.t("team_status_active")
        lines.append(ctx.t("team_line", name=u.name, role=u.role.value, status=status))
    ctx.say("\n".join(lines))


async def create_worker_from_state(ctx: Ctx, name: str) -> str:
    """Create the worker and return the deep link to hand to the boss."""
    assert ctx.user is not None
    _, raw = await onboarding.create_worker(
        ctx.repo,
        boss=ctx.user,
        name=name,
        now=ctx.now,
        ttl_hours=ctx.settings.invite_ttl_hours,
    )
    return f"https://t.me/{ctx.bot_username}?start={raw}"
