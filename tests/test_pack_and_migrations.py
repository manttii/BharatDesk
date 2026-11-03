from __future__ import annotations

import os
import string
import subprocess
import sys
from pathlib import Path

import pytest

from app.pack import REQUIRED_STRINGS, load_pack

ROOT = Path(__file__).resolve().parent.parent
KNOWN_PLACEHOLDERS = {
    "ready", "review", "bad", "qty", "label", "party", "page", "bill_no",
    "name", "company", "text", "link", "role", "status",
    "quoted", "benchmark", "pct", "floor", "discount", "pct_cap", "total", "limit", "value",
    "age_h", "max_h", "quoted_unit", "benchmark_unit", "fact",
    "max_s", "date", "count", "summary", "reason", "rep", "client", "reasons", "excerpt",
    "hours", "change", "old", "new", "amount", "question", "fields", "prompt",
    "rate", "url", "email", "id", "written", "computed", "names", "minutes", "pending", "visits",
    "last", "outstanding", "last_order", "last_advance", "delivery", "followup", "failed", "state", "hour", "pending_value", "reps",
}  # fmt: skip


def test_pack_loads_and_has_every_required_string() -> None:
    pack = load_pack()
    assert pack.name == "field_sales_jewellery"
    assert REQUIRED_STRINGS <= pack.strings.keys()


def test_pack_placeholders_are_known() -> None:
    for key, value in load_pack().strings.items():
        fields = {f for _, f, _, _ in string.Formatter().parse(value) if f}
        assert fields <= KNOWN_PLACEHOLDERS, f"{key}: {fields - KNOWN_PLACEHOLDERS}"


def test_english_policy_strings_use_the_same_placeholders_as_hindi() -> None:
    pack = load_pack()
    for key, en in pack.strings_en.items():
        hi_fields = {f for _, f, _, _ in string.Formatter().parse(pack.strings[key]) if f}
        en_fields = {f for _, f, _, _ in string.Formatter().parse(en) if f}
        assert hi_fields == en_fields, key


def test_user_supplied_text_is_substituted_as_data() -> None:
    nasty = "{name} {0} %s {__class__}"
    assert load_pack().t("echo", text=nasty) == f"Aapne likha: {nasty}"


def test_missing_string_fails_fast(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import app.pack as pack_mod

    (tmp_path / "broken.yaml").write_text("name: broken\nversion: 1\nstrings:\n  btn_yes: x\n")
    monkeypatch.setattr(pack_mod, "PACK_DIR", tmp_path)
    with pytest.raises(ValueError, match="missing strings"):
        pack_mod.load_pack("broken")


def _alembic(db: Path, *args: str) -> subprocess.CompletedProcess[str]:
    env = {**os.environ, "DATABASE_URL": f"sqlite+aiosqlite:///{db.as_posix()}"}
    return subprocess.run(  # noqa: S603
        [sys.executable, "-m", "alembic", *args],
        cwd=ROOT, env=env, capture_output=True, text=True, timeout=120, check=False,
    )  # fmt: skip


def test_migration_matches_models_and_is_reversible(tmp_path: Path) -> None:
    db = tmp_path / "m.db"
    up = _alembic(db, "upgrade", "head")
    assert up.returncode == 0, up.stderr
    check = _alembic(db, "check")  # fails if models drifted from the migration
    assert check.returncode == 0, check.stdout + check.stderr
    down = _alembic(db, "downgrade", "base")
    assert down.returncode == 0, down.stderr
