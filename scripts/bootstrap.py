"""Create a tenant and its boss invite link.

    python -m scripts.bootstrap --tenant "Sharma Jewellers" --boss "Rakesh Sharma"

The boss opens the printed link in Telegram, taps Start, and accepts the consent notice.
The raw token is shown once and never stored (only its hash is).
"""

from __future__ import annotations

import argparse
import asyncio

from app.config import get_settings
from app.db import make_engine, make_session_factory, migrate_to_head
from app.models.base import utcnow
from app.services.onboarding import bootstrap_tenant


async def main(tenant_name: str, boss_name: str) -> None:
    cfg = get_settings()
    if cfg.auto_migrate:
        await migrate_to_head(cfg.database_url)
    engine = make_engine(cfg.database_url)
    try:
        async with make_session_factory(engine)() as session, session.begin():
            tenant, boss, raw = await bootstrap_tenant(
                session,
                tenant_name=tenant_name,
                boss_name=boss_name,
                now=utcnow(),
                ttl_hours=cfg.invite_ttl_hours,
            )
            tenant_id, boss_id = tenant.id, boss.id
    finally:
        await engine.dispose()
    username = cfg.telegram_bot_username or "<YOUR_BOT_USERNAME>"
    print(f"tenant {tenant_id}  boss {boss_id}")
    print(f"Boss invite link (valid {cfg.invite_ttl_hours}h, single use):")
    print(f"https://t.me/{username}?start={raw}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tenant", required=True)
    ap.add_argument("--boss", required=True)
    args = ap.parse_args()
    asyncio.run(main(args.tenant, args.boss))
