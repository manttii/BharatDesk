from __future__ import annotations

import structlog

from app.security import (
    RateLimiter,
    hash_token,
    mask_chat_id,
    mask_chat_id_processor,
    new_token,
    secrets_match,
)


def test_new_tokens_are_unique_and_fit_telegram_start_payload() -> None:
    tokens = {new_token() for _ in range(200)}
    assert len(tokens) == 200
    assert all(len(t) <= 64 for t in tokens)


def test_hash_is_deterministic_and_not_the_token() -> None:
    assert hash_token("abc") == hash_token("abc") != "abc"
    assert len(hash_token("abc")) == 64


def test_secrets_match() -> None:
    assert secrets_match("abc", "abc")
    assert not secrets_match("abd", "abc")
    assert not secrets_match(None, "abc")
    assert not secrets_match("", "abc")


def test_mask_chat_id_hides_all_but_last_three() -> None:
    assert mask_chat_id(123456789) == "***789"
    assert mask_chat_id(12) == "***"


def test_log_processor_masks_chat_id() -> None:
    out = mask_chat_id_processor(None, "info", {"event": "x", "chat_id": 987654321})
    assert out["chat_id"] == "***321" and "987654321" not in str(out)


def test_logging_pipeline_never_emits_raw_chat_id(capsys: object) -> None:
    structlog.configure(
        processors=[mask_chat_id_processor, structlog.processors.JSONRenderer()],
        logger_factory=structlog.ReturnLoggerFactory(),
    )
    rendered = structlog.get_logger().info("evt", chat_id=555123456)
    assert "555123456" not in rendered


def test_rate_limiter_sliding_window() -> None:
    now = [0.0]
    rl = RateLimiter(limit=3, window_seconds=60, clock=lambda: now[0])
    assert [rl.allow(1) for _ in range(4)] == [True, True, True, False]
    assert rl.allow(2)  # per-chat, not global
    now[0] = 59.9
    assert not rl.allow(1)
    now[0] = 60.0
    assert rl.allow(1)
