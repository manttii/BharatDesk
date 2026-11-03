"""Repository hygiene and privacy: the "no eval, no hardcoded models/keys/rates, no stubs, no
transcripts in logs" promises, enforced rather than asserted in prose."""

from __future__ import annotations

import ast
import re
import uuid
from pathlib import Path

import pytest
import structlog

from app.logging_config import configure_logging
from app.providers.stt import STTError
from tests.conftest import (
    BOSS_CHAT,
    REP_CHAT,
    TRANSCRIPT,
    Env,
    Org,
    button_data,
    extraction_json,
    to,
)

ROOT = Path(__file__).resolve().parent.parent
CODE_DIRS = ("app", "scripts", "evals")


def _py_files() -> list[Path]:
    return sorted(p for d in CODE_DIRS for p in (ROOT / d).rglob("*.py"))


def _text(p: Path) -> str:
    return p.read_text(encoding="utf-8")


def test_there_are_python_files_to_scan() -> None:
    assert len(_py_files()) > 60  # guards against the scans below passing on an empty set


def test_no_eval_exec_or_compile_anywhere() -> None:
    offenders = []
    for p in _py_files():
        for node in ast.walk(ast.parse(_text(p))):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id in ("eval", "exec", "compile", "__import__"):
                    offenders.append(f"{p.relative_to(ROOT)}:{node.lineno} {node.func.id}()")
    assert offenders == []


MODEL_NAMES = re.compile(
    r"gemini-[0-9a-z]|whisper-large|whisper-v|gpt-[0-9]|llama-?[0-9]|claude-[a-z0-9]"
)


def test_no_hardcoded_model_names() -> None:
    offenders = [
        f"{p.relative_to(ROOT)}: {m.group(0)}"
        for p in _py_files()
        if (m := MODEL_NAMES.search(_text(p)))
    ]
    assert offenders == []  # model names come only from GEMINI_MODEL / WHISPER_MODEL


SECRET_SHAPES = [
    r"AIza[0-9A-Za-z_\-]{20,}",  # Google API key
    r"gsk_[A-Za-z0-9]{20,}",  # Groq key
    r"\b\d{8,10}:[A-Za-z0-9_\-]{30,}",  # Telegram bot token
    r"-----BEGIN [A-Z ]*PRIVATE KEY-----",
    r"sk-[A-Za-z0-9]{20,}",
]


def test_no_secret_shaped_literals_in_the_repo() -> None:
    files = [*_py_files(), ROOT / ".env.example", ROOT / "README.md", ROOT / "SECURITY.md"]
    offenders = [
        f"{p.name}: {pat}"
        for p in files
        if p.exists()
        for pat in SECRET_SHAPES
        if re.search(pat, _text(p))
    ]
    assert offenders == []


def test_env_example_ships_no_secret_values() -> None:
    sensitive = re.compile(r"KEY|TOKEN|SECRET|JSON|PASSWORD")
    offenders = []
    for line in _text(ROOT / ".env.example").splitlines():
        if line.strip().startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        if sensitive.search(name) and value.strip():
            offenders.append(name)
    assert offenders == []


def test_no_todo_stubs_or_unimplemented_bodies() -> None:
    marker = re.compile(r"\b(TODO|FIXME|XXX|HACK)\b|NotImplementedError")
    offenders = [f"{p.relative_to(ROOT)}: marker" for p in _py_files() if marker.search(_text(p))]
    for p in _py_files():
        for node in ast.walk(ast.parse(_text(p))):
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                if len(node.body) == 1 and isinstance(node.body[0], ast.Pass):
                    offenders.append(
                        f"{p.relative_to(ROOT)}:{node.lineno} {node.name} is only `pass`"
                    )
    assert offenders == []


def test_the_demo_benchmark_is_the_only_hardcoded_rate_in_the_app() -> None:
    """No rate literal (as a number) outside the sandbox's labelled demo data. Examples inside
    docstrings or help strings are fine: this inspects numeric constants only."""
    allowed = {"web/sandbox.py"}
    offenders = []
    for p in (ROOT / "app").rglob("*.py"):
        rel = p.relative_to(ROOT / "app").as_posix()
        for node in ast.walk(ast.parse(_text(p))):
            is_rate = isinstance(node, ast.Constant) and node.value in (7_520_000, 75_200)
            if is_rate and rel not in allowed:
                offenders.append(f"{rel}:{node.lineno}")
    assert offenders == []


def test_every_python_file_parses_and_none_is_truncated() -> None:
    for p in _py_files():
        src = _text(p)
        ast.parse(src)  # raises on a half-written file
        assert src == "" or src.endswith("\n"), p
        if p.name != "__init__.py":
            assert src.strip(), f"{p} is empty"


def test_dependencies_are_pinned() -> None:
    for name in ("requirements.txt", "requirements-dev.txt"):
        for line in _text(ROOT / name).splitlines():
            line = line.strip()
            if line and not line.startswith(("#", "-r")):
                assert "==" in line or line.startswith("types-"), f"{name}: {line} is not pinned"


def test_the_app_object_uvicorn_loads_exposes_the_expected_routes() -> None:
    from app.config import Settings
    from app.main import app as module_level_app  # what `uvicorn app.main:app` loads
    from app.main import create_app

    assert module_level_app is not None
    # Built from defaults, not from whatever .env happens to sit in the working directory.
    app = create_app(Settings(_env_file=None))  # type: ignore[call-arg]
    paths = set(app.openapi()["paths"])
    assert {"/health", "/telegram/webhook", "/sandbox"} <= paths
    assert not any(p.startswith("/dashboard") for p in paths)  # absent without DASHBOARD_SECRET


def test_docs_exist_and_cover_the_required_sections() -> None:
    readme, arch, sec = (_text(ROOT / n) for n in ("README.md", "ARCHITECTURE.md", "SECURITY.md"))
    assert "5-minute demo" in readme and "Known limitations" in readme
    assert "sequenceDiagram" in arch and "LLMs EXTRACT; CODE DECIDES" in arch
    for heading in ("Threat model", "Security checklist", "Retention", "Erasure"):
        assert heading in sec, heading


# --- privacy in logs -----------------------------------------------------------------------


async def test_logs_never_contain_transcripts_names_tokens_or_raw_chat_ids(
    pipe: Env, capsys: pytest.CaptureFixture[str]
) -> None:
    configure_logging()
    try:
        tenant_id, boss_id, raw_token = await pipe.new_tenant()
        await pipe.say(BOSS_CHAT, f"/start {raw_token}")
        await pipe.tap(BOSS_CHAT, "co:y")
        worker_token = await pipe.add_worker(BOSS_CHAT, "Ramesh")
        await pipe.say(REP_CHAT, f"/start {worker_token}")
        await pipe.tap(REP_CHAT, "co:y")
        await pipe.add_client(Org(tenant_id, boss_id, uuid.uuid4()), "Rajendra Jewellers")
        out = await pipe.voice(REP_CHAT, TRANSCRIPT, extraction_json())
        await pipe.tap(REP_CHAT, button_data(to(REP_CHAT, out)[0], "Haan"))
        await pipe.say(7777, "hello")  # an unknown chat
        pipe.stt.results.append(STTError("x"))  # type: ignore[union-attr]
        await pipe.voice(REP_CHAT, None, None)  # an error path
    finally:
        structlog.reset_defaults()
    logged = capsys.readouterr().out
    assert "***" in logged, "no masked chat id was logged: this test would be vacuous"
    for secret in (
        "dedh sau gram",
        "Rajendra",
        "4,12,000",
        "paintalis",
        "gaya tha",
        raw_token,
        worker_token,
        "Ramesh",
    ):
        assert secret not in logged, secret
    for raw_id in ("1001", "2002", "7777"):
        assert not re.search(rf"\b{raw_id}\b", logged), f"raw chat id {raw_id} leaked"


def test_the_ci_workflow_is_valid_and_runs_the_full_check_on_both_pythons() -> None:
    import yaml

    wf = yaml.safe_load(_text(ROOT / ".github" / "workflows" / "ci.yml"))
    job = wf["jobs"]["test"]
    assert job["strategy"]["matrix"]["python-version"] == ["3.11", "3.12"]
    commands = " ".join(str(step.get("run", "")) for step in job["steps"])
    for needle in ("requirements-dev.txt", "ruff check", "mypy", "pytest", "ffmpeg"):
        assert needle in commands, needle


def test_every_env_var_the_settings_read_is_documented_in_env_example() -> None:
    from app.config import Settings

    documented = {
        line.split("=", 1)[0].strip().lower()
        for line in _text(ROOT / ".env.example").splitlines()
        if "=" in line and not line.strip().startswith("#")
    }
    # tuning knobs that are intentionally left to defaults
    optional = {
        "allowed_voice_mimes", "stt_min_avg_logprob", "stt_max_no_speech_prob", "low_confidence_flag",
        "max_media_bytes", "expiry_sweep_seconds", "pending_action_ttl_minutes", "sync_batch_size",
    }  # fmt: skip
    missing = sorted(set(Settings.model_fields) - documented - optional)
    assert missing == [], f"undocumented settings: {missing}"


def test_env_example_loads_as_shipped_and_means_everything_is_unset(tmp_path: Path) -> None:
    """A fresh clone copies .env.example to .env; that must not crash and must not enable anything."""
    from app.config import Settings

    shipped = tmp_path / ".env"
    shipped.write_text(_text(ROOT / ".env.example"), encoding="utf-8")
    cfg = Settings(_env_file=shipped)  # type: ignore[call-arg]
    assert (
        cfg.groq_api_key is None and cfg.gemini_api_key is None and cfg.telegram_bot_token is None
    )
    assert cfg.whisper_model is None and cfg.gemini_model is None and cfg.dashboard_secret is None
    assert cfg.stt_cost_paise_per_minute is None and cfg.llm_cost_paise_per_1k_tokens is None
    assert cfg.google_service_account_file is None and cfg.telegram_webhook_secret is None
