"""Operator commands that are deliberately NOT exposed in the bot.

    python -m scripts.admin export-worker --tenant "Sharma Jewellers" --worker "Ramesh" [--out ramesh.json]
    python -m scripts.admin erase-worker  --tenant "Sharma Jewellers" --worker "Ramesh" --yes
    python -m scripts.admin retention     --tenant "Sharma Jewellers" --audio-days 30 --image-days 30
    python -m scripts.admin purge         [--tenant "Sharma Jewellers"]

`--audio-days none` keeps audio forever. `purge` is also run automatically every
RETENTION_SWEEP_HOURS while the app is up. Run these on the host that holds the database.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import uuid
from pathlib import Path
from typing import Any

from sqlalchemy import select

from app.config import get_settings
from app.db import TenantRepository, make_engine, make_session_factory
from app.models import Tenant, User
from app.models.base import utcnow
from app.pipeline.confirm import TenantSettings
from app.pipeline.media import MediaStore
from app.services import retention, worker_data


class AdminError(Exception):
    pass


async def _tenant_id(session: Any, name: str) -> uuid.UUID:
    ids = list(
        (await session.execute(select(Tenant.id).where(Tenant.name == name))).scalars().all()
    )
    if not ids:
        raise AdminError(f"no tenant named {name!r}")
    if len(ids) > 1:
        raise AdminError(f"{len(ids)} tenants are named {name!r}; rename one first")
    return uuid.UUID(str(ids[0]))


async def _worker(repo: TenantRepository, name: str) -> User:
    found = [u for u in await repo.list(User) if u.name == name]
    if not found:
        raise AdminError(f"no user named {name!r} in that tenant")
    if len(found) > 1:
        raise AdminError(f"{len(found)} users are named {name!r}; use unique names")
    return found[0]


async def export_worker(sf: Any, tenant: str, worker: str) -> dict[str, Any]:
    async with sf() as session:
        repo = TenantRepository(session, await _tenant_id(session, tenant))
        return await worker_data.export_worker(repo, await _worker(repo, worker), utcnow())


async def erase_worker(
    sf: Any, store: MediaStore, tenant: str, worker: str
) -> worker_data.ErasureReport:
    async with sf() as session, session.begin():
        repo = TenantRepository(session, await _tenant_id(session, tenant))
        user = await _worker(repo, worker)
        try:
            return await worker_data.erase_worker(repo, store, user, actor=None, now=utcnow())
        except ValueError as exc:
            raise AdminError(str(exc)) from exc


async def set_retention(
    sf: Any,
    tenant: str,
    audio: int | None,
    image: int | None,
    *,
    set_audio: bool,
    set_image: bool,
    transcript: int | None = None,
    set_transcript: bool = False,
) -> TenantSettings:
    async with sf() as session, session.begin():
        t = await session.get(Tenant, await _tenant_id(session, tenant))
        assert t is not None
        values = dict(t.settings)
        if set_audio:
            values["audio_retention_days"] = audio
        if set_image:
            values["image_retention_days"] = image
        if set_transcript:
            values["transcript_retention_days"] = transcript
        ts = TenantSettings.from_json(values)  # validates (days >= 1 or None)
        t.settings = values
        TenantRepository(session, t.id).audit(
            "retention_changed",
            entity="tenant",
            entity_id=str(t.id),
            detail={
                "audio": ts.audio_retention_days,
                "image": ts.image_retention_days,
                "transcript": ts.transcript_retention_days,
            },
        )
        return ts


def _days(text: str) -> int | None:
    return None if text.lower() in ("none", "forever") else int(text)


async def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("export-worker", "erase-worker"):
        p = sub.add_parser(name)
        p.add_argument("--tenant", required=True)
        p.add_argument("--worker", required=True)
        if name == "export-worker":
            p.add_argument("--out", type=Path)
        else:
            p.add_argument("--yes", action="store_true", help="confirm the erasure")
    r = sub.add_parser("retention")
    r.add_argument("--tenant", required=True)
    r.add_argument("--audio-days")
    r.add_argument("--image-days")
    r.add_argument(
        "--transcript-days", help="days to keep transcripts; 'none' (default) keeps them"
    )
    pu = sub.add_parser("purge")
    pu.add_argument("--tenant")
    args = ap.parse_args(argv)

    cfg = get_settings()
    engine = make_engine(cfg.database_url)
    sf = make_session_factory(engine)
    store = MediaStore(cfg.media_dir)
    try:
        if args.cmd == "export-worker":
            data = json.dumps(
                await export_worker(sf, args.tenant, args.worker), indent=2, ensure_ascii=False
            )
            if args.out:
                args.out.write_text(data, encoding="utf-8")
                print(f"wrote {args.out}")
            else:
                print(data)
        elif args.cmd == "erase-worker":
            if not args.yes:
                print(
                    "This permanently erases the worker's personal data. Re-run with --yes to confirm."
                )
                return 2
            rep = await erase_worker(sf, store, args.tenant, args.worker)
            print(
                "already erased"
                if rep.already_erased
                else f"erased: {rep.messages_scrubbed} messages scrubbed, {rep.media_files_deleted} media files deleted, "
                f"{rep.records_resynced} sheet rows queued to drop the name"
            )
        elif args.cmd == "retention":
            ts = await set_retention(
                sf,
                args.tenant,
                _days(args.audio_days or "90"),
                _days(args.image_days or "90"),
                set_audio=args.audio_days is not None,
                set_image=args.image_days is not None,
                transcript=_days(args.transcript_days or "none"),
                set_transcript=args.transcript_days is not None,
            )
            print(
                f"retention: audio {ts.audio_retention_days} days, images {ts.image_retention_days} days, "
                f"transcripts {ts.transcript_retention_days} days (None = forever)"
            )
        else:
            tid = None
            if args.tenant:
                async with sf() as session:
                    tid = await _tenant_id(session, args.tenant)
            now = utcnow()
            rep2 = await retention.purge_media(sf, store, now, only_tenant=tid)
            scrubbed = await retention.purge_transcripts(sf, now, only_tenant=tid)
            print(
                f"purged {rep2.messages_purged} messages' media ({rep2.files_deleted} files) and "
                f"{scrubbed} transcripts across {rep2.tenants} tenant(s)"
            )
    except AdminError as exc:
        print(f"error: {exc}")
        return 1
    finally:
        await engine.dispose()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
