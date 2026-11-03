"""Provider failures: what is retried, what is not, and what the model is told."""

from __future__ import annotations

import asyncio
from datetime import date
from typing import Any

import httpx
import pytest
from google.genai import errors as genai_errors

from app.pipeline.extract import run_extraction
from app.providers.llm import FakeLLM, GeminiLLM, LLMError, LLMUnavailableError
from tests.conftest import extraction_json

TODAY = date(2026, 11, 2)


async def _run(llm: FakeLLM, **kw: Any):  # type: ignore[no-untyped-def]
    return await run_extraction(
        llm, "x", today=TODAY, timezone="Asia/Kolkata", known_clients=[], **kw
    )


@pytest.fixture
def sleeps(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    recorded: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        recorded.append(seconds)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    return recorded


# --- retry policy ----------------------------------------------------------------------------


async def test_a_transient_failure_waits_then_retries_the_same_prompt(sleeps: list[float]) -> None:
    llm = FakeLLM([LLMError("busy", transient=True), extraction_json()])
    run = await _run(llm)
    assert run.parsed is not None and run.attempts == 2
    assert sleeps == [3.0]  # a pause, not an instant retry that burns quota
    assert llm.calls[0]["user"] == llm.calls[1]["user"]  # nothing was wrong with the prompt
    assert "previous_output_was_invalid" not in llm.calls[1]["user"]


async def test_a_permanent_failure_is_retried_without_waiting(sleeps: list[float]) -> None:
    llm = FakeLLM([LLMError("bad request"), extraction_json()])
    run = await _run(llm)
    assert run.parsed is not None and sleeps == []


async def test_two_provider_failures_give_up_with_the_error_recorded(sleeps: list[float]) -> None:
    llm = FakeLLM([LLMError("quota", transient=True), LLMError("quota", transient=True)])
    run = await _run(llm)
    assert (
        run.parsed is None and run.error == "llm_error: quota" and sleeps == [3.0]
    )  # one wait only


async def test_an_invalid_reply_is_retried_with_the_error_and_no_pause(sleeps: list[float]) -> None:
    llm = FakeLLM(["not json", extraction_json()])
    run = await _run(llm)
    assert run.parsed is not None and sleeps == []
    assert "previous_output_was_invalid" in llm.calls[1]["user"]


async def test_the_retry_delay_is_configurable() -> None:
    llm = FakeLLM([LLMError("busy", transient=True), extraction_json()])
    started = asyncio.get_running_loop().time()
    await _run(llm, retry_delay_s=0)
    assert asyncio.get_running_loop().time() - started < 1


# --- the Gemini adapter ----------------------------------------------------------------------


class _Models:
    def __init__(
        self, error: Exception | None = None, generate_error: Exception | None = None
    ) -> None:
        self.error, self.generate_error = error, generate_error
        self.generated = 0

    async def get(self, **kw: Any) -> None:
        if self.error:
            raise self.error

    async def generate_content(self, **kw: Any) -> Any:
        self.generated += 1
        if self.generate_error:
            raise self.generate_error

        class R:
            text = "OK"
            model_version = "m"
            usage_metadata = None

        return R()


class _Aio:
    def __init__(self, models: _Models) -> None:
        self.models = models


class _Client:
    def __init__(self, models: _Models) -> None:
        self.aio = _Aio(models)


def _api_error(code: int, message: str = "boom") -> genai_errors.APIError:
    cls = genai_errors.ClientError if code < 500 else genai_errors.ServerError
    return cls(code, {"error": {"code": code, "message": message}})


@pytest.mark.parametrize(
    ("code", "transient"),
    [
        (429, True),
        (500, True),
        (503, True),
        (504, True),
        (400, False),
        (401, False),
        (403, False),
        (404, False),
    ],
)
async def test_gemini_errors_are_classified_as_transient_or_not(code: int, transient: bool) -> None:
    llm = GeminiLLM("k", "m", client=_Client(_Models(generate_error=_api_error(code, "why"))))
    with pytest.raises(LLMError) as exc:
        await llm.generate_json(system="s", user="u", schema={})
    assert (
        exc.value.transient is transient
        and f"HTTP {code}" in str(exc.value)
        and "why" in str(exc.value)
    )


@pytest.mark.parametrize(
    "error",
    [TimeoutError(), httpx.ReadTimeout("slow"), httpx.ConnectError("down"), ConnectionResetError()],
)
async def test_timeouts_and_network_errors_are_transient_llm_errors(error: Exception) -> None:
    llm = GeminiLLM("k", "m", client=_Client(_Models(generate_error=error)))
    with pytest.raises(LLMError) as exc:
        await llm.generate_json(system="s", user="u", schema={})
    assert exc.value.transient is True


async def test_startup_check_really_generates_not_just_lists_the_model() -> None:
    """Google still lists models that 404 on generate; metadata alone is not proof."""
    models = _Models(generate_error=_api_error(404, "no longer available to new users"))
    with pytest.raises(
        LLMUnavailableError, match=r"404.*no longer available to new users.*Pick another"
    ):
        await GeminiLLM("k", "old-model", client=_Client(models)).ensure_available()
    assert models.generated == 1


async def test_startup_check_passes_when_the_model_generates() -> None:
    models = _Models()
    await GeminiLLM("k", "good", client=_Client(models)).ensure_available()
    assert models.generated == 1


@pytest.mark.parametrize("code", [429, 500, 503, 504])
async def test_a_busy_model_does_not_block_startup(code: int) -> None:
    await GeminiLLM(
        "k", "m", client=_Client(_Models(generate_error=_api_error(code)))
    ).ensure_available()


@pytest.mark.parametrize("code", [400, 401, 403, 404])
async def test_a_misconfigured_key_or_model_does_block_startup(code: int) -> None:
    with pytest.raises(LLMUnavailableError):
        await GeminiLLM(
            "k", "m", client=_Client(_Models(generate_error=_api_error(code)))
        ).ensure_available()


async def test_a_model_the_key_cannot_see_blocks_startup() -> None:
    with pytest.raises(LLMUnavailableError, match="404"):
        await GeminiLLM(
            "k", "nope", client=_Client(_Models(error=_api_error(404, "not found")))
        ).ensure_available()


async def test_the_error_text_never_contains_the_api_key() -> None:
    secret = "AQ.super-secret-key"
    err = _api_error(403, "permission denied")
    llm = GeminiLLM(secret, "m", client=_Client(_Models(generate_error=err)))
    with pytest.raises(LLMError) as exc:
        await llm.generate_json(system="s", user="u", schema={})
    assert secret not in str(exc.value)
