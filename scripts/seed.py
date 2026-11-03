"""Create a DEMO tenant with sensible policy, a benchmark, clients and optional history.

    python -m scripts.seed --tenant "Demo Jewellers" --boss "Asha" --workers "Ramesh,Suresh" --history

Prints Telegram invite links (single use, 24 h). The tenant is marked `demo`, so its data is
excluded from `scripts/pilot_report.py`: demo history is not pilot evidence. For a REAL customer
use `scripts/bootstrap.py`, which creates only the tenant and its boss.

Demo policy: a rate may be at most 1% below the benchmark (benchmark must be under 24 h old),
no order above Rs 10,00,000 without approval, and a Rs 10,00,000 credit limit per client.
The demo benchmark (Rs 75,200 per 10 g) is sample data, not a market rate.
"""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.config import Settings, get_settings
from app.db import TenantRepository, make_engine, make_session_factory, migrate_to_head
from app.models import ActivityLog, BenchmarkRate, PolicyRule, Tenant
from app.models.base import utcnow
from app.models.enums import ActivityStatus
from app.schemas.extraction import ItemPayload, OrderPayload, PaymentPayload, VisitPayload
from app.services.clients import create_client
from app.services.onboarding import bootstrap_tenant, create_worker

DEMO_CLIENTS = ("Rajendra Jewellers", "Surendra Gems", "Mehta Jewellers")
DEMO_BENCHMARK_PAISE = 7_520_000
# (days ago, client index, weight_mg, order value paise, advance paise)
DEMO_HISTORY = [
    (3, 0, 150_000, 41_200_000, 4_500_000),
    (3, 1, 80_000, 22_500_000, 22_500_000),
    (2, 0, 500_000, 20_000_000, 5_000_000),
    (2, 2, 120_000, 33_000_000, 10_000_000),
    (1, 1, 60_000, 16_800_000, 0),
    (1, 2, 200_000, 55_000_000, 20_000_000),
]


class SeedError(Exception):
    pass


@dataclass
class SeedResult:
    tenant_id: str
    boss_link: str
    worker_links: dict[str, str] = field(default_factory=dict)


def _link(settings: Settings, raw: str) -> str:
    return f"https://t.me/{settings.telegram_bot_username or '<YOUR_BOT_USERNAME>'}?start={raw}"


async def seed(
    session_factory: async_sessionmaker[AsyncSession],
    settings: Settings,
    *,
    tenant_name: str,
    boss_name: str,
    workers: list[str],
    history: bool,
    now: datetime | None = None,
) -> SeedResult:
    now = now or utcnow()
    async with session_factory() as session, session.begin():
        if (await session.execute(select(Tenant.id).where(Tenant.name == tenant_name))).first():
            raise SeedError(f"a tenant named {tenant_name!r} already exists; pick another name")
        tenant, boss, raw = await bootstrap_tenant(
            session,
            tenant_name=tenant_name,
            boss_name=boss_name,
            now=now,
            ttl_hours=settings.invite_ttl_hours,
        )
        tenant.settings = {"demo": True}
        repo = TenantRepository(session, tenant.id)
        result = SeedResult(str(tenant.id), _link(settings, raw))
        created = []
        for name in workers:
            user, token = await create_worker(
                repo, boss=boss, name=name, now=now, ttl_hours=settings.invite_ttl_hours
            )
            created.append(user)
            result.worker_links[name] = _link(settings, token)
        repo.add(PolicyRule(rule_type="rate_floor_vs_benchmark",
                            params={"max_below_bps": 100, "max_benchmark_age_seconds": 86_400}))  # fmt: skip
        repo.add(
            PolicyRule(
                rule_type="max_order_value_without_approval",
                params={"max_value_paise": 100_000_000},
            )
        )
        repo.add(
            PolicyRule(
                rule_type="credit_limit_per_client", params={"default_limit_paise": 100_000_000}
            )
        )
        repo.add(
            BenchmarkRate(value_paise=DEMO_BENCHMARK_PAISE, unit="10g", set_by=boss.id, set_at=now)
        )
        clients = [create_client(repo, n) for n in DEMO_CLIENTS]
        await repo.flush()
        if history and created:
            for i, (days_ago, ci, weight, value, advance) in enumerate(DEMO_HISTORY):
                when = now - timedelta(days=days_ago)
                visit = VisitPayload(
                    client_name=clients[ci].name,
                    client_id=str(clients[ci].id),
                    visit_date=when.date(),
                    order=OrderPayload(items=[ItemPayload(name="jewellery", quantity=1)],
                                       weight_mg=weight, value_paise=value),
                    payment=PaymentPayload(received_paise=advance, pending_paise=value - advance),
                )  # fmt: skip
                repo.add(
                    ActivityLog(
                        created_at=when,
                        user_id=created[i % len(created)].id,
                        client_id=clients[ci].id,
                        intent="LOG_VISIT",
                        status=ActivityStatus.CONFIRMED,
                        payload={
                            "visit": visit.model_dump(mode="json"),
                            "meta": {"source": "seed"},
                        },
                    )
                )
        repo.audit(
            "tenant_seeded", entity="tenant", entity_id=str(tenant.id), detail={"demo": True}
        )
    return result


async def main(args: argparse.Namespace) -> int:
    cfg = get_settings()
    if cfg.auto_migrate:
        await migrate_to_head(cfg.database_url)
    engine = make_engine(cfg.database_url)
    try:
        res = await seed(
            make_session_factory(engine), cfg, tenant_name=args.tenant, boss_name=args.boss,
            workers=[w.strip() for w in args.workers.split(",") if w.strip()], history=args.history,
        )  # fmt: skip
    except SeedError as exc:
        print(f"error: {exc}")
        return 1
    finally:
        await engine.dispose()
    print(f"demo tenant {res.tenant_id} created (excluded from pilot reports)")
    print(f"boss:   {res.boss_link}")
    for name, link in res.worker_links.items():
        print(f"worker {name}: {link}")
    print("each link is single-use and valid for 24 hours; open it in Telegram and tap Start")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tenant", default="Demo Jewellers")
    ap.add_argument("--boss", default="Demo Boss")
    ap.add_argument("--workers", default="Ramesh,Suresh", help="comma-separated names")
    ap.add_argument("--history", action="store_true", help="add six sample confirmed visits")
    raise SystemExit(asyncio.run(main(ap.parse_args())))
