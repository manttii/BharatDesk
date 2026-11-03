"""Things that must hold off the developer's machine: PostgreSQL DDL, the ffmpeg fallback."""

from __future__ import annotations

import asyncio
import os
import shutil
import subprocess
import sys
import wave
from pathlib import Path

import pytest
from sqlalchemy.dialects import postgresql
from sqlalchemy.schema import CreateIndex, CreateTable

from app.models import Base
from app.providers import stt
from app.providers.stt import STTError, ffmpeg_to_wav
from scripts.smoke_providers import silence_wav

ROOT = Path(__file__).resolve().parent.parent
needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is not installed")


# --- PostgreSQL (no server needed: DDL is compiled for the dialect) ---------------------------


def test_every_table_compiles_for_postgresql() -> None:
    dialect = postgresql.dialect()
    ddl = {}
    for table in Base.metadata.sorted_tables:
        ddl[table.name] = str(CreateTable(table).compile(dialect=dialect))
        for index in table.indexes:
            str(CreateIndex(index).compile(dialect=dialect))
    assert len(ddl) == len(Base.metadata.tables) >= 15
    users = ddl["users"]
    assert "telegram_chat_id BIGINT" in users and "id UUID NOT NULL" in users
    assert "TIMESTAMP WITH TIME ZONE" in users
    assert "JSON" in ddl["tenants"]


def test_the_migration_renders_as_postgresql_sql_without_sqlite_only_features() -> None:
    env = {**os.environ, "DATABASE_URL": "postgresql+asyncpg://u:p@localhost/db"}
    run = subprocess.run(
        [sys.executable, "-m", "alembic", "upgrade", "head", "--sql"],
        cwd=ROOT, env=env, capture_output=True, text=True, timeout=120, check=False,
    )  # fmt: skip
    assert run.returncode == 0, run.stderr
    sql = run.stdout
    for table in (
        "tenants",
        "users",
        "audit_log",
        "approval_requests",
        "sync_jobs",
        "processed_events",
    ):
        assert f"CREATE TABLE {table}" in sql, table
    assert "CREATE TRIGGER" not in sql and "sqlite" not in sql.lower()  # triggers are SQLite-only
    assert "media_purged_at" in sql and "erased_at" in sql  # the latest revision is included


# --- ffmpeg fallback -------------------------------------------------------------------------


@needs_ffmpeg
async def test_ffmpeg_converts_wav_to_16k_mono_wav(tmp_path: Path) -> None:
    out = await ffmpeg_to_wav(silence_wav(2, rate=44_100))
    path = tmp_path / "o.wav"
    path.write_bytes(out)
    with wave.open(str(path), "rb") as w:
        assert (w.getnchannels(), w.getframerate()) == (1, 16_000) and w.getnframes() > 0


@needs_ffmpeg
async def test_ffmpeg_converts_ogg_opus_like_telegram_voice_notes(tmp_path: Path) -> None:
    src = tmp_path / "in.ogg"
    ffmpeg = shutil.which("ffmpeg") or "ffmpeg"
    cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i",
           "sine=frequency=440:duration=1", "-c:a", "libopus", str(src)]  # fmt: skip
    made = await asyncio.to_thread(subprocess.run, cmd, capture_output=True, check=False)
    if made.returncode != 0:
        pytest.skip("this ffmpeg build cannot encode Opus")
    assert src.read_bytes()[:4] == b"OggS"
    out = await ffmpeg_to_wav(src.read_bytes())
    assert out[:4] == b"RIFF" and out[8:12] == b"WAVE"


@needs_ffmpeg
async def test_ffmpeg_rejects_bytes_that_are_not_audio() -> None:
    with pytest.raises(STTError, match="could not convert"):
        await ffmpeg_to_wav(b"this is definitely not audio")


async def test_a_missing_ffmpeg_is_a_clear_error_not_a_crash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def boom(*args: object, **kwargs: object) -> None:
        raise FileNotFoundError("ffmpeg")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", boom)
    with pytest.raises(STTError, match="not installed"):
        await ffmpeg_to_wav(b"x")
    assert stt.ffmpeg_to_wav is ffmpeg_to_wav
