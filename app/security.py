"""Token hashing, constant-time secret checks, PII masking and rate limiting."""

from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import json
import secrets
import time
from collections import defaultdict, deque
from collections.abc import Callable
from typing import Any

from structlog.typing import EventDict


def new_token(nbytes: int = 24) -> str:
    """URL-safe token. 24 bytes -> 32 chars, within Telegram's 64-char start payload."""
    return secrets.token_urlsafe(nbytes)


def hash_token(raw: str) -> str:
    """Tokens are high-entropy random values, so an unsalted SHA-256 is sufficient."""
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def secrets_match(supplied: str | None, expected: str) -> bool:
    if supplied is None:
        return False
    return hmac.compare_digest(supplied.encode("utf-8"), expected.encode("utf-8"))


def mask_chat_id(chat_id: int | str) -> str:
    s = str(chat_id)
    return "***" + s[-3:] if len(s) > 3 else "***"


def mask_chat_id_processor(_: Any, __: str, event_dict: EventDict) -> EventDict:
    """structlog processor: a raw `chat_id` must never reach the log sink."""
    if "chat_id" in event_dict:
        event_dict["chat_id"] = mask_chat_id(event_dict["chat_id"])
    return event_dict


class RateLimiter:
    """Sliding-window limiter, per key (chat_id). In-process; fine for a single worker."""

    def __init__(
        self,
        limit: int,
        window_seconds: float = 60.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._limit = limit
        self._window = window_seconds
        self._clock = clock
        self._hits: dict[int, deque[float]] = defaultdict(deque)

    def allow(self, key: int) -> bool:
        now = self._clock()
        hits = self._hits[key]
        while hits and now - hits[0] >= self._window:
            hits.popleft()
        if len(hits) >= self._limit:
            return False
        hits.append(now)
        return True


# --- signed, expiring tokens (boss dashboard links) -------------------------------------


def _b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("ascii").rstrip("=")


def _unb64(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


def sign_token(secret: str, payload: dict[str, Any]) -> str:
    """`payload` must carry an integer `exp` (epoch seconds). HMAC-SHA256, no passwords/OTP."""
    body = _b64(json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8"))
    sig = _b64(hmac.new(secret.encode("utf-8"), body.encode("ascii"), hashlib.sha256).digest())
    return f"{body}.{sig}"


def verify_token(secret: str, token: str, *, now: int) -> dict[str, Any] | None:
    """The payload if the signature is valid and `exp` is in the future, else None."""
    try:
        body, sig = token.split(".", 1)
        expected = _b64(
            hmac.new(secret.encode("utf-8"), body.encode("ascii"), hashlib.sha256).digest()
        )
        if not hmac.compare_digest(sig.encode("ascii"), expected.encode("ascii")):
            return None
        payload = json.loads(_unb64(body))
    except (ValueError, UnicodeError, binascii.Error):
        return None
    exp = payload.get("exp") if isinstance(payload, dict) else None
    if isinstance(exp, bool) or not isinstance(exp, int) or exp <= now:
        return None
    return payload  # type: ignore[no-any-return]


def csrf_token(secret: str, session_token: str, scope: str) -> str:
    msg = f"csrf:{scope}:{session_token}".encode()
    return hmac.new(secret.encode("utf-8"), msg, hashlib.sha256).hexdigest()
