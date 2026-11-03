"""The pre-push secrets check finds leaks without ever printing the secret."""

from __future__ import annotations

from pathlib import Path

from scripts.check_secrets import scan

FAKE = "zz-not-a-real-secret-0123456789"


def _project(tmp_path: Path, gitignore: str = ".env\n") -> Path:
    (tmp_path / ".gitignore").write_text(gitignore)
    (tmp_path / ".env").write_text(f"GEMINI_API_KEY={FAKE}\nENV=dev\n")
    return tmp_path


def test_clean_project_passes(tmp_path: Path) -> None:
    root = _project(tmp_path)
    (root / "README.md").write_text("no secrets here")
    assert scan(root) == []


def test_a_leaked_env_value_is_reported_without_the_value(tmp_path: Path) -> None:
    root = _project(tmp_path)
    (root / "notes.md").write_text(f"key: {FAKE}\n")
    problems = scan(root)
    assert problems and "notes.md:1" in problems[0] and "GEMINI_API_KEY" in problems[0]
    assert all(FAKE not in p for p in problems)


def test_provider_shaped_keys_are_caught(tmp_path: Path) -> None:
    root = _project(tmp_path)
    (root / "a.py").write_text("K = 'gsk_" + "a" * 30 + "'\n")
    assert any("Groq" in p for p in scan(root))


def test_missing_gitignore_entry_is_reported(tmp_path: Path) -> None:
    root = _project(tmp_path, gitignore="*.pyc\n")
    assert any(".gitignore" in p for p in scan(root))


def test_a_database_url_password_is_a_secret_even_though_the_name_is_not(tmp_path: Path) -> None:
    (tmp_path / ".gitignore").write_text(".env\n")
    url = "postgresql://owner:" + "s3cretPassw0rd" + "@db.example.com/app?sslmode=require"
    (tmp_path / ".env").write_text(f"DATABASE_URL={url}\n")
    (tmp_path / "notes.md").write_text(f"connect with {url}\n")
    problems = scan(tmp_path)
    assert any("notes.md:1" in p and "DATABASE_URL" in p for p in problems)
    assert all("s3cretPassw0rd" not in p for p in problems)


def test_a_url_with_a_real_looking_password_is_caught_by_shape_but_placeholders_pass(
    tmp_path: Path,
) -> None:
    (tmp_path / ".gitignore").write_text(".env\n")
    (tmp_path / "x.md").write_text("postgres://owner:" + "longpasswordhere" + "@host/db\n")
    assert any("database URL" in p for p in scan(tmp_path))
    (tmp_path / "x.md").write_text("postgres://user:pass@host/db\n")  # a documentation placeholder
    assert scan(tmp_path) == []
