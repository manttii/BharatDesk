"""Commit a confirmed visit: run the deterministic policy engine, then either record it or
route it to the boss. Nothing here involves an LLM.

A visit with no order has no policy to check. An order is checked against every active
PolicyRule; any violation (including a missing/stale benchmark: the engine fails closed)
creates an ApprovalRequest and the record stays out of reach (never synced) until the
boss taps Approve.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime

import structlog

from app.config import Settings
from app.db import TenantRepository
from app.models import (
    ActivityLog,
    ApprovalRequest,
    BenchmarkRate,
    Client,
    PolicyRule,
    Tenant,
    User,
)
from app.models.enums import ActivityStatus
from app.pack import Pack
from app.pipeline.confirm import summary_text
from app.pipeline.expr import expression_names
from app.pipeline.policy_engine import evaluate
from app.schemas.events import OutMessage
from app.schemas.extraction import VisitPayload
from app.schemas.policy import (
    BenchmarkFacts,
    CustomExpressionParams,
    PolicyFacts,
    RuleSpec,
    RuleType,
    Violation,
    parse_rule,
)
from app.services import approvals, ledger, sync_queue
from app.services.clients import create_client, remember_alias
from app.units import ceil_div

log = structlog.get_logger()

MG_PER_10G = 10_000
RATE_UNIT = "10g"


@dataclass
class CommitResult:
    status: ActivityStatus
    replies: list[OutMessage]
    approval_id: str | None = None


async def load_rules(repo: TenantRepository) -> tuple[list[RuleSpec], list[str]]:
    """Active rules parsed into typed specs. A malformed rule is reported, not skipped."""
    rules: list[RuleSpec] = []
    errors: list[str] = []
    for row in await repo.list(
        PolicyRule, PolicyRule.is_active.is_(True), order_by=[PolicyRule.created_at]
    ):
        try:
            rules.append(
                parse_rule(
                    f"{row.rule_type}:{row.id.hex[:8]}", row.rule_type, row.params, row.version
                )
            )
        except ValueError as exc:
            errors.append(f"{row.id.hex[:8]}: {exc}")
    return rules, errors


@dataclass(frozen=True)
class RequiredByPolicy:
    value: bool  # a rule compares the order value (cap, discount, credit)
    rate: bool  # a rule compares the quoted rate to the benchmark


_VALUE_RULES = {
    RuleType.MAX_ORDER_VALUE_WITHOUT_APPROVAL,
    RuleType.DISCOUNT_CAP_PCT,
    RuleType.CREDIT_LIMIT_PER_CLIENT,
}


async def required_by_policy(repo: TenantRepository) -> RequiredByPolicy:
    """Which order fields the tenant's active rules need the rep to state."""
    rules, _ = await load_rules(repo)
    value = rate = False
    for r in rules:
        if r.rule_type in _VALUE_RULES:
            value = True
        elif r.rule_type is RuleType.RATE_FLOOR_VS_BENCHMARK:
            rate = True
        elif isinstance(r.params, CustomExpressionParams):
            names = expression_names(r.params.expression)
            value = value or bool(names & {"order_value_paise", "new_pending_paise"})
            rate = rate or "quoted_rate_paise" in names
    return RequiredByPolicy(value=value, rate=rate)


async def latest_benchmark(repo: TenantRepository) -> BenchmarkFacts | None:
    rows = await repo.list(
        BenchmarkRate, order_by=[BenchmarkRate.set_at.desc(), BenchmarkRate.created_at.desc()]
    )
    if not rows:
        return None
    return BenchmarkFacts(value_paise=rows[0].value_paise, unit=rows[0].unit, set_at=rows[0].set_at)


async def build_facts(
    repo: TenantRepository, payload: VisitPayload, client: Client | None, activity_id: object
) -> PolicyFacts:
    order = payload.order
    assert order is not None
    benchmark = await latest_benchmark(repo)
    list_value = None
    if benchmark is not None and benchmark.unit == RATE_UNIT and order.weight_mg is not None:
        # Value at the benchmark rate; ceil so any rounding makes a discount look LARGER
        # (fails closed). Making charges are not modelled.
        list_value = ceil_div(benchmark.value_paise * order.weight_mg, MG_PER_10G)
    outstanding = (
        await ledger.outstanding_paise(repo, client.id, exclude=activity_id)  # type: ignore[arg-type]
        if client is not None
        else 0
    )
    return PolicyFacts(
        client_id=str(client.id) if client else "new",
        order_value_paise=order.value_paise or 0,
        payment_received_paise=(payload.payment.received_paise or 0) if payload.payment else 0,
        list_value_paise=list_value,
        quoted_rate_paise=order.rate_paise_per_10g,
        rate_unit=RATE_UNIT if order.rate_paise_per_10g is not None else None,
        benchmark=benchmark,
        outstanding_paise=outstanding,
        weight_mg=order.weight_mg,
    )


async def commit_visit(
    repo: TenantRepository,
    *,
    user: User,
    tenant: Tenant,
    activity: ActivityLog,
    payload: VisitPayload,
    pack: Pack,
    settings: Settings,
    now: datetime,
    excerpt: str,
    spoken_client: str | None = None,
) -> CommitResult:
    client: Client | None = None
    new_client = False
    if payload.client_id is not None:
        client = await repo.get(Client, uuid.UUID(payload.client_id))
        if client is not None and spoken_client:
            remember_alias(client, spoken_client)
    elif payload.client_name:
        client = create_client(repo, payload.client_name)
        await repo.flush()
        new_client = True
        payload = payload.model_copy(update={"client_id": str(client.id)})
    activity.client_id = client.id if client else None

    violations: list[Violation] = []
    if payload.order is not None:
        rules, errors = await load_rules(repo)
        facts = await build_facts(repo, payload, client, activity.id)
        violations = list(evaluate(facts, rules, pack=pack, now=now).violations)
        for err in errors:  # a broken rule must not silently disable a control
            log.error("policy_rule_invalid", detail=err)
            violations.append(
                Violation(
                    rule_id="config",
                    rule_type=RuleType.MAX_ORDER_VALUE_WITHOUT_APPROVAL,
                    code="config_invalid",
                    reason_hi=pack.t("policy_config_invalid"),
                    reason_en=pack.te("policy_config_invalid"),
                    numbers={},
                )
            )

    meta = {**activity.payload.get("meta", {}), "client_new": new_client or bool(
        activity.payload.get("meta", {}).get("client_new"))}  # fmt: skip
    approvals.with_visit(activity, payload, **{k: v for k, v in meta.items() if k != "visit"})
    summary = summary_text(payload, pack, new_client=bool(meta["client_new"]), today=now.date())
    await repo.flush()

    if not violations:
        activity.status = ActivityStatus.CONFIRMED
        repo.audit("activity_confirmed", entity="activity_log", entity_id=str(activity.id),
                   actor_user_id=user.id)  # fmt: skip
        await repo.flush()
        await sync_queue.enqueue(repo, activity.id, now)  # outbox row, same transaction
        count = await ledger.visits_today(repo, user.id, tenant.timezone, now)
        return CommitResult(
            ActivityStatus.CONFIRMED,
            [
                OutMessage(
                    user.telegram_chat_id or 0, pack.t("committed", summary=summary, count=count)
                )
            ],
        )

    activity.status = ActivityStatus.PENDING_APPROVAL
    approval = repo.add(_approval_row(user, activity, violations, now, settings.approval_ttl_hours))
    await repo.flush()
    repo.audit("approval_created", entity="approval_request", entity_id=str(approval.id),
               actor_user_id=user.id, detail={"codes": [v.code for v in violations]})  # fmt: skip
    replies: list[OutMessage] = []
    bosses = await approvals.bosses_to_notify(repo)
    rep_chat = user.telegram_chat_id or 0
    if bosses:
        replies.append(
            OutMessage(
                rep_chat,
                pack.t("awaiting_approval", reason="; ".join(v.reason_hi for v in violations)),
            )
        )
        for boss in bosses:
            assert boss.telegram_chat_id is not None
            replies.append(
                approvals.boss_request_message(
                    boss.telegram_chat_id, approval, rep=user.name,
                    client=payload.client_name or "-", summary=summary, excerpt=excerpt, pack=pack,
                )
            )  # fmt: skip
    else:
        replies.append(OutMessage(rep_chat, pack.t("approval_held")))
    return CommitResult(ActivityStatus.PENDING_APPROVAL, replies, approval.id.hex)


def _approval_row(
    user: User, activity: ActivityLog, violations: list[Violation], now: datetime, ttl_hours: int
) -> ApprovalRequest:
    return ApprovalRequest(
        created_at=now,  # the injected clock, so expiry/reminder maths is consistent
        activity_log_id=activity.id,
        requested_by=user.id,
        reason_hi="\n".join(v.reason_hi for v in violations),
        reason_en="\n".join(v.reason_en for v in violations),
        numbers=approvals.violation_numbers(violations),
        expires_at=approvals.default_expiry(now, ttl_hours),
    )
