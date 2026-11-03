"""Webhook secret-token check and the health endpoint, through the real FastAPI app."""

from __future__ import annotations

import time
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.config import Settings
from app.main import create_app
from app.providers.messaging import FakeProvider

SECRET = "s3cret_token-OK"
HEADER = "X-Telegram-Bot-Api-Secret-Token"


def _update(update_id: int, text: str, chat_id: int = 42) -> dict[str, object]:
    return {
        "update_id": update_id,
        "message": {
            "message_id": 1,
            "date": 1_700_000_000,
            "chat": {"id": chat_id, "type": "private"},
            "from": {"id": chat_id, "is_bot": False, "first_name": "T"},
            "text": text,
        },
    }


@pytest.fixture
def webhook_client(tmp_path: Path) -> Iterator[tuple[TestClient, FakeProvider]]:
    fake = FakeProvider()
    settings = Settings(  # type: ignore[call-arg]
        _env_file=None,
        database_url=f"sqlite+aiosqlite:///{(tmp_path / 'wh.db').as_posix()}",
        telegram_bot_token="123:dummy",  # type: ignore[arg-type]
        telegram_mode="webhook",
        telegram_webhook_url="https://example.test/telegram/webhook",
        telegram_webhook_secret=SECRET,  # type: ignore[arg-type]
    )
    with TestClient(create_app(settings, messaging=fake)) as client:
        yield client, fake


def _wait_for(fake: FakeProvider, count: int, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while len(fake.sent) < count and time.monotonic() < deadline:
        time.sleep(0.02)


def test_missing_secret_header_is_forbidden(
    webhook_client: tuple[TestClient, FakeProvider],
) -> None:
    client, fake = webhook_client
    assert client.post("/telegram/webhook", json=_update(1, "hi")).status_code == 403
    time.sleep(0.1)
    assert fake.sent == []


def test_wrong_secret_header_is_forbidden(webhook_client: tuple[TestClient, FakeProvider]) -> None:
    client, fake = webhook_client
    r = client.post("/telegram/webhook", json=_update(2, "hi"), headers={HEADER: SECRET + "x"})
    assert r.status_code == 403
    time.sleep(0.1)
    assert fake.sent == []


def test_correct_secret_is_accepted_and_processed(
    webhook_client: tuple[TestClient, FakeProvider],
) -> None:
    client, fake = webhook_client
    r = client.post("/telegram/webhook", json=_update(3, "hi"), headers={HEADER: SECRET})
    assert r.status_code == 200 and r.json() == {"ok": True}
    _wait_for(fake, 1)
    assert len(fake.sent) == 1 and "registered nahi" in fake.sent[0].text  # unknown chat


def test_redelivered_update_is_not_processed_twice(
    webhook_client: tuple[TestClient, FakeProvider],
) -> None:
    client, fake = webhook_client
    for _ in range(2):
        r = client.post("/telegram/webhook", json=_update(4, "hi"), headers={HEADER: SECRET})
        assert r.status_code == 200
    _wait_for(fake, 1)
    time.sleep(0.3)
    assert len(fake.sent) == 1


def test_malformed_body_with_valid_secret_is_400(
    webhook_client: tuple[TestClient, FakeProvider],
) -> None:
    client, _ = webhook_client
    r = client.post("/telegram/webhook", json={"not": "an update"}, headers={HEADER: SECRET})
    assert r.status_code == 400


def test_group_chat_updates_are_ignored(webhook_client: tuple[TestClient, FakeProvider]) -> None:
    client, fake = webhook_client
    body = _update(5, "hi")
    body["message"]["chat"]["type"] = "group"  # type: ignore[index]
    assert client.post("/telegram/webhook", json=body, headers={HEADER: SECRET}).status_code == 200
    time.sleep(0.3)
    assert fake.sent == []


def test_webhook_mode_requires_a_valid_secret() -> None:
    base = {"_env_file": None, "telegram_mode": "webhook", "telegram_bot_token": "1:x",
            "telegram_webhook_url": "https://example.test/hook"}  # fmt: skip
    with pytest.raises(ValidationError):
        Settings(**base)  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        Settings(**base, telegram_webhook_secret="has spaces!")  # type: ignore[arg-type]


def test_webhook_route_absent_in_polling_mode(tmp_path: Path) -> None:
    settings = Settings(  # type: ignore[call-arg]
        _env_file=None, database_url=f"sqlite+aiosqlite:///{(tmp_path / 'p.db').as_posix()}"
    )
    with TestClient(create_app(settings, messaging=FakeProvider())) as client:
        assert client.post("/telegram/webhook", json=_update(6, "hi")).status_code == 404


def test_health_reports_db_and_telegram_state(tmp_path: Path) -> None:
    settings = Settings(  # type: ignore[call-arg]
        _env_file=None, database_url=f"sqlite+aiosqlite:///{(tmp_path / 'h.db').as_posix()}"
    )
    with TestClient(create_app(settings)) as client:  # no token, no provider: Telegram disabled
        r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {
        "status": "ok",
        "db": "ok",
        "telegram": "disabled",
        "providers": {"stt": False, "llm": False, "sheets": False},
        "version": "0.4.0",
    }
