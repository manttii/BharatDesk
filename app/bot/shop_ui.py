"""Shop questions in chat: /stock, /today, and the same asked in plain Hindi/Hinglish.

Plain-language matching is deliberately narrow and deterministic (no model, no guessing): a short
message with no digits AND a question cue, so a visit note such as "Rajendra ko stock dikhaya"
is never mistaken for a question. Anything else goes on to the normal note pipeline.
/stock is open to the whole team; /today shows sales money, so it is boss-only.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal
from zoneinfo import ZoneInfo

from app.bot.context import Ctx
from app.models import Tenant
from app.models.enums import Role
from app.services.bill_sync import milli_cell
from app.services.shop_reports import day_summary, stock_report
from app.units import format_inr

MAX_STOCK_LINES = 25
MAX_WORDS = 7
MAX_ITEM_WORDS = 3
_WORD = re.compile(r"[a-zऀ-ॿ]+")
_CUES = {"kitna", "kitni", "kitne", "kya", "kaun", "batao", "bata", "dikhao", "dikha", "show",
         "list", "hai", "hain", "left", "much", "status"}  # fmt: skip
_STOCK_WORDS = {"stock", "bacha", "bachha", "bachi", "bache", "inventory", "maal"}
_SALE_WORDS = {"sale", "sales", "bika", "bike", "bikri", "bechi", "becha", "bik"}
_TODAY_WORDS = {"aaj", "today"}
_FILLER = _CUES | _STOCK_WORDS | {
    "ka", "ki", "ke", "mein", "me", "abhi", "aaj", "the", "of", "in", "is", "how", "what",
    "hua", "hui", "hai", "ye", "yeh", "mera", "meri", "apna", "kuch", "sab", "total",
}  # fmt: skip


@dataclass(frozen=True)
class ShopQuery:
    kind: Literal["stock", "today"]
    item: str | None = None


def classify_query(text: str) -> ShopQuery | None:
    low = text.casefold()
    if any(ch.isdigit() for ch in low):
        return None
    words = _WORD.findall(low)
    if not words or len(words) > MAX_WORDS:
        return None
    cue = bool(set(words) & _CUES) or "?" in low
    if not cue:
        return None
    if set(words) & _TODAY_WORDS and set(words) & _SALE_WORDS:
        return ShopQuery("today")
    if set(words) & _STOCK_WORDS:
        item_words = [w for w in words if w not in _FILLER]
        if len(item_words) > MAX_ITEM_WORDS:  # that is a sentence about a visit, not an item name
            return None
        return ShopQuery("stock", " ".join(item_words) or None)
    return None


def _qty(milli: int, unit: str) -> str:
    return f"{milli_cell(milli)} {unit}".strip()


async def answer(ctx: Ctx, query: ShopQuery) -> None:
    if query.kind == "stock":
        await stock(ctx, query.item)
    else:
        await today(ctx)


async def stock(ctx: Ctx, item: str | None) -> None:
    rep = await stock_report(ctx.repo, item)
    if not rep.lines:
        if item:
            ctx.say(ctx.t("stock_none_for", name=item[:60]))
        else:
            ctx.say(ctx.t("stock_empty"))
        return
    rows = [
        ctx.t("stock_line", name=s.name, qty=_qty(s.on_hand, s.unit))
        + (ctx.t("stock_minus_tag") if s.on_hand < 0 else "")
        for s in rep.lines[:MAX_STOCK_LINES]
    ]
    parts = [ctx.t("stock_header"), *rows]
    if len(rep.lines) > MAX_STOCK_LINES:
        parts.append(ctx.t("stock_more", count=len(rep.lines) - MAX_STOCK_LINES))
    if rep.pending_bills:
        parts.append(ctx.t("stock_footnote", pending=rep.pending_bills))
    ctx.say("\n".join(parts))


async def today(ctx: Ctx) -> None:
    assert ctx.user is not None
    if ctx.user.role is not Role.BOSS:
        ctx.say(ctx.t("boss_only"))
        return
    tenant = await ctx.session.get(Tenant, ctx.user.tenant_id)
    tz = tenant.timezone if tenant else "Asia/Kolkata"
    day = ctx.now.astimezone(ZoneInfo(tz)).date()
    s = await day_summary(ctx.repo, day)
    if not (s.sales_count or s.purchase_count):
        parts = [ctx.t("today_none", date=day.isoformat())]
    else:
        parts = [
            ctx.t("today_header", date=day.isoformat()),
            ctx.t("today_sales", count=s.sales_count, total=format_inr(s.sales_paise)),
            ctx.t("today_purchases", count=s.purchase_count, total=format_inr(s.purchase_paise)),
        ]
        if s.top_sold:
            names = ", ".join(f"{n} {_qty(q, u)}" for n, u, q in s.top_sold)
            parts.append(ctx.t("today_top", names=names))
        if s.unknown_totals:
            parts.append(ctx.t("today_unknown_total", count=s.unknown_totals))
    if s.pending_bills:
        parts.append(ctx.t("today_pending", pending=s.pending_bills))
    ctx.say("\n".join(parts))
