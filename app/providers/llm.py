"""LLM boundary: an interface, Gemini via google-genai, and a Fake.

google-genai 2.x (read from the installed package): `Client(api_key=...)`,
`client.aio.models.generate_content(model=, contents=, config=GenerateContentConfig(
system_instruction, temperature, response_mime_type, response_json_schema))`,
`client.aio.models.get(model=)` for availability, `response.text`,
`response.usage_metadata.{prompt,candidates}_token_count`.
The model name always comes from GEMINI_MODEL; startup fails clearly if it is unavailable.
"""

from __future__ import annotations

import base64
import json
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import httpx
import structlog
from google import genai
from google.genai import errors as genai_errors
from google.genai import types as genai_types

if TYPE_CHECKING:
    from app.config import Settings

log = structlog.get_logger()


class LLMError(Exception):
    """A provider failure. `transient` means retrying shortly may work (rate limit, busy,
    timeout); anything else (bad request, auth) will not improve by waiting."""

    def __init__(self, message: str, *, transient: bool = False) -> None:
        super().__init__(message)
        self.transient = transient


class LLMUnavailableError(LLMError):
    """The configured model cannot be used; raised at startup so it is never a surprise."""


@dataclass(frozen=True)
class ImagePart:
    """An image sent to a vision-capable model. Its contents are untrusted data."""

    data: bytes
    mime_type: str


@dataclass(frozen=True)
class LLMResult:
    text: str
    model: str
    latency_ms: int
    prompt_tokens: int | None = None
    output_tokens: int | None = None


class LLMProvider(ABC):
    model: str

    @abstractmethod
    async def ensure_available(self) -> None: ...

    @abstractmethod
    async def generate_json(
        self,
        *,
        system: str,
        user: str,
        schema: dict[str, Any],
        image: ImagePart | None = None,
    ) -> LLMResult:
        """Structured JSON output at temperature 0. `user` and `image` are untrusted data."""


def _api_message(exc: genai_errors.APIError) -> str:
    """Google's own error text (it never contains the key), trimmed to one short line."""
    text = str(getattr(exc, "message", None) or exc)
    return " ".join(text.split())[:200]


class GeminiLLM(LLMProvider):
    def __init__(
        self,
        api_key: str,
        model: str,
        *,
        client: Any | None = None,
        timeout_s: int = 60,
    ) -> None:
        self.model = model
        # An explicit timeout matters: one queue worker serves every rep, so a Gemini call
        # that hangs would freeze all notes. The SDK's default is not a safe bound.
        self._client = client or genai.Client(
            api_key=api_key,
            http_options=genai_types.HttpOptions(timeout=timeout_s * 1000),
        )

    async def ensure_available(self) -> None:
        """Fail at startup if the model cannot actually GENERATE.

        `models.get` is not enough: Google still lists models that return 404 ("no longer
        available to new users") when you generate. So after the metadata check we make one tiny
        real generation. Busy/rate-limited responses (429, 5xx) are not a configuration error
        and do not block startup.
        """
        try:
            await self._client.aio.models.get(model=self.model)
            await self._client.aio.models.generate_content(
                model=self.model,
                contents="Reply with the single word OK.",
                config=genai_types.GenerateContentConfig(temperature=0, max_output_tokens=16),
            )
        except genai_errors.APIError as exc:
            code = getattr(exc, "code", None)
            if isinstance(code, int) and (code == 429 or code >= 500):
                log.warning("gemini_busy_at_startup", code=code)
                return
            raise LLMUnavailableError(
                f"GEMINI_MODEL {self.model!r} cannot be used with this API key "
                f"(HTTP {code}: {_api_message(exc)}). Pick another model."
            ) from exc

    async def generate_json(
        self,
        *,
        system: str,
        user: str,
        schema: dict[str, Any],
        image: ImagePart | None = None,
    ) -> LLMResult:
        started = time.monotonic()
        contents: Any = user
        if image is not None:
            # Part.from_bytes(data=, mime_type=) in google-genai 2.x; the image comes first.
            contents = [
                genai_types.Part.from_bytes(data=image.data, mime_type=image.mime_type),
                user,
            ]
        config = genai_types.GenerateContentConfig(
            system_instruction=system,
            temperature=0,
            response_mime_type="application/json",
            response_json_schema=schema,
        )
        try:
            resp = await self._client.aio.models.generate_content(
                model=self.model, contents=contents, config=config
            )
        except genai_errors.APIError as exc:
            code = getattr(exc, "code", None)
            raise LLMError(
                f"gemini request failed (HTTP {code}: {_api_message(exc)})",
                transient=isinstance(code, int) and (code == 429 or code >= 500),
            ) from exc
        except (TimeoutError, httpx.HTTPError, OSError) as exc:
            raise LLMError(f"gemini request failed ({type(exc).__name__})", transient=True) from exc
        usage = getattr(resp, "usage_metadata", None)
        return LLMResult(
            text=resp.text or "",
            model=getattr(resp, "model_version", None) or self.model,
            latency_ms=int((time.monotonic() - started) * 1000),
            prompt_tokens=getattr(usage, "prompt_token_count", None),
            output_tokens=getattr(usage, "candidates_token_count", None),
        )


_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_FENCE_RE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL)


def _clean_reply(text: str) -> str:
    """Reasoning models prepend <think> blocks and chatty ones wrap JSON in a code fence."""
    text = _THINK_RE.sub("", text).strip()
    fenced = _FENCE_RE.match(text)
    return fenced.group(1) if fenced else text


class OpenAICompatLLM(LLMProvider):
    """Any OpenAI-style `/chat/completions` endpoint; used for NVIDIA's hosted NIM models.

    The model name always comes from settings. JSON is requested with `response_format` and the
    schema is also written into the system prompt, because not every hosted model enforces it;
    the reply is validated by the same strict pydantic step either way.
    """

    def __init__(
        self,
        api_key: str,
        model: str,
        *,
        base_url: str,
        provider_name: str = "nvidia",
        client: httpx.AsyncClient | None = None,
        timeout_s: int = 60,
    ) -> None:
        self.model = model
        self._url = base_url.rstrip("/") + "/chat/completions"
        self._name = provider_name
        self._headers = {"Authorization": f"Bearer {api_key}"}
        # An explicit timeout: one queue worker serves every rep, so a hang would freeze all notes.
        self._client = client or httpx.AsyncClient(timeout=timeout_s)

    @staticmethod
    def _message(resp: httpx.Response) -> str:
        """The provider's own error text (it never contains the key), one short line."""
        try:
            body = resp.json()
            err = body.get("error") if isinstance(body, dict) else None
            text = (
                (err.get("message") if isinstance(err, dict) else err)
                or (body.get("detail") if isinstance(body, dict) else None)
                or resp.text
            )
        except ValueError:
            text = resp.text
        return " ".join(str(text).split())[:200]

    async def _post(self, payload: dict[str, Any]) -> httpx.Response:
        try:
            return await self._client.post(self._url, json=payload, headers=self._headers)
        except (TimeoutError, httpx.HTTPError, OSError) as exc:
            raise LLMError(
                f"{self._name} request failed ({type(exc).__name__})", transient=True
            ) from exc

    @staticmethod
    def _transient(code: int) -> bool:
        return code == 429 or code >= 500

    async def ensure_available(self) -> None:
        """One tiny real generation: listing a model is not proof it answers for this key."""
        resp = await self._post(
            {
                "model": self.model,
                "messages": [{"role": "user", "content": "Reply with the single word OK."}],
                "temperature": 0,
                "max_tokens": 16,
            }
        )
        if resp.status_code < 400:
            return
        if self._transient(resp.status_code):
            log.warning("llm_busy_at_startup", provider=self._name, code=resp.status_code)
            return
        raise LLMUnavailableError(
            f"{self._name.upper()}_MODEL {self.model!r} cannot be used with this API key "
            f"(HTTP {resp.status_code}: {self._message(resp)}). Pick another model."
        )

    async def generate_json(
        self,
        *,
        system: str,
        user: str,
        schema: dict[str, Any],
        image: ImagePart | None = None,
    ) -> LLMResult:
        started = time.monotonic()
        content: Any = user
        if image is not None:
            data = base64.b64encode(image.data).decode("ascii")
            content = [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{image.mime_type};base64,{data}"},
                },
                {"type": "text", "text": user},
            ]
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": f"{system}\n\nReply with ONLY one JSON object that conforms to this "
                    f"JSON Schema, no prose and no code fence:\n{json.dumps(schema)}",
                },
                {"role": "user", "content": content},
            ],
            "temperature": 0,
            "response_format": {"type": "json_object"},
        }
        resp = await self._post(payload)
        if resp.status_code == 400 and "response_format" in resp.text:
            payload.pop("response_format")  # this model has no JSON mode; the prompt carries it
            resp = await self._post(payload)
        if resp.status_code >= 400:
            raise LLMError(
                f"{self._name} request failed (HTTP {resp.status_code}: {self._message(resp)})",
                transient=self._transient(resp.status_code),
            )
        try:
            body = resp.json()
            text = body["choices"][0]["message"]["content"] or ""
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"{self._name} returned an unreadable response") from exc
        usage = body.get("usage") or {}
        return LLMResult(
            text=_clean_reply(text),
            model=body.get("model") or self.model,
            latency_ms=int((time.monotonic() - started) * 1000),
            prompt_tokens=usage.get("prompt_tokens"),
            output_tokens=usage.get("completion_tokens"),
        )


def build_llm(cfg: Settings) -> LLMProvider | None:
    """The configured provider, or None. Settings guarantee at most one is chosen."""
    if cfg.llm_provider == "nvidia" or (cfg.llm_provider is None and cfg.nvidia_api_key):
        if cfg.nvidia_api_key and cfg.nvidia_model:
            return OpenAICompatLLM(
                cfg.nvidia_api_key.get_secret_value(),
                cfg.nvidia_model,
                base_url=cfg.nvidia_base_url,
                timeout_s=cfg.llm_timeout_seconds,
            )
        return None
    if cfg.gemini_api_key and cfg.gemini_model:
        return GeminiLLM(
            cfg.gemini_api_key.get_secret_value(),
            cfg.gemini_model,
            timeout_s=cfg.llm_timeout_seconds,
        )
    return None


class FakeLLM(LLMProvider):
    """Scripted JSON replies for tests; records the prompts it was given."""

    def __init__(self, replies: list[str | Exception] | None = None) -> None:
        self.model = "fake-llm"
        self.replies: list[str | Exception] = list(replies or [])
        self.calls: list[dict[str, Any]] = []
        self.available = True

    async def ensure_available(self) -> None:
        if not self.available:
            raise LLMUnavailableError("fake model unavailable")

    async def generate_json(
        self,
        *,
        system: str,
        user: str,
        schema: dict[str, Any],
        image: ImagePart | None = None,
    ) -> LLMResult:
        self.calls.append({"system": system, "user": user, "schema": schema, "image": image})
        if not self.replies:
            raise LLMError("FakeLLM has no scripted reply")
        nxt = self.replies.pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        return LLMResult(
            text=nxt, model=self.model, latency_ms=7, prompt_tokens=100, output_tokens=50
        )
