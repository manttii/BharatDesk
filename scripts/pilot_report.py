"""Pilot evidence from the live database. Real data only: demo tenants are excluded and an empty
period prints "no data". Nothing is estimated.

    python -m scripts.pilot_report [--since 2026-11-01] [--until 2026-12-01] [--out report.md]
    python -m scripts.pilot_report --include-demo      # only to eyeball the layout on seeded data
"""

from __future__ import annotations

import argparse
import asyncio
from datetime import UTC, date, datetime, time
from pathlib import Path

from app.config import get_settings
from app.db import make_engine, make_session_factory
from app.services.pilot_report import build_report, render_markdown


def _day(text: str) -> datetime:
    return datetime.combine(date.fromisoformat(text), time.min, tzinfo=UTC)


async def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--since", type=_day, help="UTC date, inclusive")
    ap.add_argument("--until", type=_day, help="UTC date, exclusive")
    ap.add_argument("--include-demo", action="store_true")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args(argv)
    engine = make_engine(get_settings().database_url)
    try:
        report = await build_report(
            make_session_factory(engine),
            since=args.since,
            until=args.until,
            include_demo=args.include_demo,
        )
    finally:
        await engine.dispose()
    text = render_markdown(report)
    if args.out:
        args.out.write_text(text, encoding="utf-8")
        print(f"wrote {args.out}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
