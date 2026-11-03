"""Make sure no secret can leave this machine by accident. Run it before every commit/push.

    python -m scripts.check_secrets

It fails (exit 1) if
  * `.env` is not git-ignored, or is tracked by git;
  * any file other than `.env` contains the ACTUAL value of a secret from `.env`
    (API keys, tokens, secrets, service-account JSON); or
  * any file contains something shaped like a provider key or private key.
It never prints a secret: only the file, line number and which setting matched.
Install it as a hook with `python -m scripts.check_secrets --install-hook` (needs a git repo).
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SENSITIVE_NAME = re.compile(r"KEY|TOKEN|SECRET|PASSWORD|JSON|CREDENTIAL")
CREDENTIAL_URL = re.compile(r"://[^/\s:@]+:[^@\s/]+@")
MIN_SECRET_LEN = 12  # shorter values (e.g. "true", "60") are settings, not secrets
SKIP_DIRS = {".git", ".venv", "venv", "__pycache__", ".mypy_cache", ".pytest_cache", ".ruff_cache",
             "media", "node_modules"}  # fmt: skip
SKIP_SUFFIXES = {".db", ".sqlite", ".pyc", ".ogg", ".jpg", ".jpeg", ".png", ".webp", ".wav", ".mp3"}
SHAPES = {
    "Google/Gemini API key": re.compile(r"AIza[0-9A-Za-z_\-]{20,}"),
    "Gemini key (AQ. format)": re.compile(r"\bAQ\.[A-Za-z0-9_\-]{30,}"),
    "Groq API key": re.compile(r"gsk_[A-Za-z0-9]{20,}"),
    "Telegram bot token": re.compile(r"\b\d{8,10}:[A-Za-z0-9_\-]{30,}"),
    "private key block": re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    "OpenAI-style key": re.compile(r"\bsk-[A-Za-z0-9]{20,}"),
    "database URL with a password": re.compile(r"://[^/\s:@]+:[^@\s/]{8,}@"),
}


def env_secrets(env_path: Path) -> dict[str, str]:
    """name -> value for every sensitive, long-enough setting in .env."""
    found: dict[str, str] = {}
    if not env_path.is_file():
        return found
    for line in env_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.strip().startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        name, value = name.strip(), value.strip().strip("'\"")
        if len(value) >= MIN_SECRET_LEN and (
            SENSITIVE_NAME.search(name) or CREDENTIAL_URL.search(value)
        ):
            found[name] = value  # a URL with user:password@ is a secret whatever it is called
    return found


def candidate_files(root: Path) -> list[Path]:
    out: list[Path] = []
    for p in root.rglob("*"):
        rel = p.relative_to(root)
        if not p.is_file() or set(rel.parts) & SKIP_DIRS or p.suffix.lower() in SKIP_SUFFIXES:
            continue
        if rel.as_posix() == ".env":
            continue  # the one place a secret belongs
        out.append(p)
    return sorted(out)


def scan(root: Path = ROOT) -> list[str]:
    """Human-readable problems; empty means clean. Never contains a secret value."""
    problems: list[str] = []
    gitignore = root / ".gitignore"
    ignored = gitignore.is_file() and re.search(
        r"^\.env\s*$", gitignore.read_text(encoding="utf-8"), re.M
    )
    if not ignored:
        problems.append(".gitignore does not ignore .env")
    if (root / ".git").exists():
        tracked = subprocess.run(  # noqa: S603
            ["git", "-C", str(root), "ls-files", "--error-unmatch", ".env"],  # noqa: S607
            capture_output=True,
            check=False,
        )
        if tracked.returncode == 0:
            problems.append(".env is TRACKED by git: run `git rm --cached .env` before pushing")
    secrets = env_secrets(root / ".env")
    for path in candidate_files(root):
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        rel = path.relative_to(root).as_posix()
        for lineno, line in enumerate(text.splitlines(), start=1):
            for name, value in secrets.items():
                if value in line:
                    problems.append(f"{rel}:{lineno} contains the value of {name} from .env")
            for label, pattern in SHAPES.items():
                if pattern.search(line):
                    problems.append(f"{rel}:{lineno} looks like a {label}")
    return problems


HOOK = """#!/bin/sh
# installed by scripts/check_secrets.py: block commits/pushes that would leak a secret
python -m scripts.check_secrets || exit 1
"""


def install_hook(root: Path = ROOT) -> str:
    hooks = root / ".git" / "hooks"
    if not hooks.is_dir():
        return "not a git repository yet: run `git init` first, then re-run with --install-hook"
    for name in ("pre-commit", "pre-push"):
        (hooks / name).write_text(HOOK, encoding="utf-8", newline="\n")
        (hooks / name).chmod(0o755)
    return "installed pre-commit and pre-push hooks"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--install-hook", action="store_true")
    args = ap.parse_args(argv)
    if args.install_hook:
        print(install_hook())
        return 0
    problems = scan()
    if problems:
        print("SECRETS CHECK FAILED. Do not commit or push until this is fixed:")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("secrets check passed: .env is ignored and no secret appears in any other file")
    return 0


if __name__ == "__main__":
    sys.exit(main())
