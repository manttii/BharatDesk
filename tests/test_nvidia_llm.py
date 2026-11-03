"""The OpenAI-compatible (NVIDIA) provider: request shape, reply cleanup, errors, selection."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from app.config import Settings
from app.providers.llm import (
    GeminiLLM,
    ImagePart,
    LLMError,
    LLMUnavailableError,
    OpenAICompatLLM,
    build_llm,
)

SECRET = "nvapi-test-secret-value"
SCHEMA = {"type": "object", "properties": {"ok": {"type": "boolean"}}}


def _llm(handler: Any) -> tuple[OpenAICompatLLM, list[httpx.Request]]:
    seen: list[httpx.Request] = []

    def wrapped(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return handler(request)

    client = httpx.AsyncClient(transport=httpx.MockTransport(wrapped))
    return OpenAICompatLLM(SECRET, "m", base_url="https://x.test/v1/", client=client), seen


def _ok(content: str) -> Any:
    return lambda r: httpx.Response(
        200,
        json={
            "model": "served-m",
            "choices": [{"message": {"content": content}}],
            "usage": {"prompt_tokens": 11, "completion_tokens": 3},
        },
    )


async def test_request_shape_and_result() -> None:
    llm, seen = _llm(_ok('{"ok": true}'))
    res = await llm.generate_json(system="SYS", user="USER", schema=SCHEMA)
    req = seen[0]
    body = json.loads(req.content)
    assert str(req.url) == "https://x.test/v1/chat/completions"
    assert req.headers["authorization"] == f"Bearer {SECRET}"
    assert body["model"] == "m" and body["temperature"] == 0
    assert body["response_format"] == {"type": "json_object"}
    assert (
        body["messages"][0]["content"].startswith("SYS")
        and '"ok"' in body["messages"][0]["content"]
    )
    assert body["messages"][1] == {"role": "user", "content": "USER"}
    assert (res.text, res.model, res.prompt_tokens, res.output_tokens) == (
        '{"ok": true}',
        "served-m",
        11,
        3,
    )


@pytest.mark.parametrize(
    "raw",
    [
        '<think>hmm {"ok": false}</think>\n{"ok": true}',
        '```json\n{"ok": true}\n```',
        '{"ok": true}',
    ],
)
async def test_reasoning_blocks_and_code_fences_are_stripped(raw: str) -> None:
    llm, _ = _llm(_ok(raw))
    assert (await llm.generate_json(system="s", user="u", schema=SCHEMA)).text == '{"ok": true}'


async def test_an_image_is_sent_as_a_data_url() -> None:
    llm, seen = _llm(_ok("{}"))
    await llm.generate_json(
        system="s", user="u", schema=SCHEMA, image=ImagePart(b"\x89PNG", "image/png")
    )
    parts = json.loads(seen[0].content)["messages"][1]["content"]
    assert parts[0]["image_url"]["url"].startswith("data:image/png;base64,")
    assert parts[1] == {"type": "text", "text": "u"}


async def test_a_model_without_json_mode_is_retried_without_it() -> None:
    calls: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        calls.append(body)
        if "response_format" in body:
            return httpx.Response(400, json={"error": {"message": "response_format unsupported"}})
        return _ok('{"ok": true}')(request)

    llm, _ = _llm(handler)
    res = await llm.generate_json(system="s", user="u", schema=SCHEMA)
    assert res.text == '{"ok": true}' and len(calls) == 2 and "response_format" not in calls[1]


@pytest.mark.parametrize(
    ("code", "transient"), [(429, True), (500, True), (503, True), (400, False), (401, False)]
)
async def test_http_errors_are_classified(code: int, transient: bool) -> None:
    llm, _ = _llm(lambda r: httpx.Response(code, json={"detail": "why"}))
    with pytest.raises(LLMError) as exc:
        await llm.generate_json(system="s", user="u", schema=SCHEMA)
    assert exc.value.transient is transient and f"HTTP {code}" in str(exc.value)
    assert "why" in str(exc.value) and SECRET not in str(exc.value)


async def test_network_failures_are_transient() -> None:
    def boom(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("down")

    llm, _ = _llm(boom)
    with pytest.raises(LLMError) as exc:
        await llm.generate_json(system="s", user="u", schema=SCHEMA)
    assert exc.value.transient is True and SECRET not in str(exc.value)


async def test_an_unreadable_reply_is_an_error_not_a_crash() -> None:
    llm, _ = _llm(lambda r: httpx.Response(200, json={"choices": []}))
    with pytest.raises(LLMError, match="unreadable"):
        await llm.generate_json(system="s", user="u", schema=SCHEMA)


async def test_startup_check_really_generates() -> None:
    llm, seen = _llm(_ok("OK"))
    await llm.ensure_available()
    assert len(seen) == 1


async def test_startup_blocks_on_a_bad_model_but_not_on_a_busy_one() -> None:
    bad, _ = _llm(lambda r: httpx.Response(404, json={"detail": "Function not found"}))
    with pytest.raises(LLMUnavailableError, match=r"404.*Function not found.*Pick another"):
        await bad.ensure_available()
    busy, _ = _llm(lambda r: httpx.Response(429, json={"detail": "slow down"}))
    await busy.ensure_available()


# --- selection ---------------------------------------------------------------------------------


def _cfg(**kw: Any) -> Settings:
    return Settings(_env_file=None, **kw)  # type: ignore[call-arg]


def test_nvidia_is_chosen_when_only_it_is_configured() -> None:
    llm = build_llm(_cfg(nvidia_api_key="k", nvidia_model="some/model"))
    assert isinstance(llm, OpenAICompatLLM) and llm.model == "some/model"


def test_gemini_is_chosen_when_only_it_is_configured() -> None:
    assert isinstance(build_llm(_cfg(gemini_api_key="k", gemini_model="m")), GeminiLLM)


def test_nothing_configured_means_no_llm() -> None:
    assert build_llm(_cfg()) is None


def test_two_providers_require_an_explicit_choice() -> None:
    both = {"gemini_api_key": "a", "gemini_model": "m", "nvidia_api_key": "b", "nvidia_model": "n"}
    with pytest.raises(ValueError, match="LLM_PROVIDER"):
        _cfg(**both)
    assert isinstance(build_llm(_cfg(**both, llm_provider="nvidia")), OpenAICompatLLM)
    assert isinstance(build_llm(_cfg(**both, llm_provider="gemini")), GeminiLLM)


def test_a_key_without_a_model_refuses_to_start() -> None:
    with pytest.raises(ValueError, match="NVIDIA_MODEL"):
        _cfg(nvidia_api_key="k")


def test_a_chosen_provider_needs_its_key() -> None:
    with pytest.raises(ValueError, match="NVIDIA_API_KEY"):
        _cfg(llm_provider="nvidia")
