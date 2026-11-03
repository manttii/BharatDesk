"""The sandbox page drives the real pipeline with no Telegram."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from app.providers.llm import FakeLLM
from app.providers.messaging import FakeProvider
from app.providers.stt import FakeSTT
from tests.conftest import OGG, TRANSCRIPT, extraction_json

RATE_TRANSCRIPT = TRANSCRIPT + ", sattar hazaar rate"


def _app(tmp_path: Path, *, env: str = "dev", with_providers: bool = True):  # type: ignore[no-untyped-def]
    settings = Settings(  # type: ignore[call-arg]
        _env_file=None, env=env, database_url=f"sqlite+aiosqlite:///{(tmp_path / 's.db').as_posix()}",
        media_dir=str(tmp_path / "media"),
    )  # fmt: skip
    stt, llm = (FakeSTT(), FakeLLM()) if with_providers else (None, None)
    return create_app(settings, messaging=FakeProvider(), stt=stt, llm=llm), stt, llm


def test_sandbox_runs_a_typed_note_end_to_end_and_shows_every_stage(tmp_path: Path) -> None:
    app, _, llm = _app(tmp_path)
    assert llm is not None
    llm.replies.append(extraction_json(rate=7_000_000))
    with TestClient(app) as client:
        r = client.post("/sandbox/run", data={"as_user": "rep", "text": RATE_TRANSCRIPT})
        assert r.status_code == 200
        body = r.json()
        assert [s["stage"] for s in body["stages"]] == [
            "transcript",
            "numerals",
            "llm",
            "assessment",
        ]
        (reply,) = body["replies"]
        assert reply["to"] == "rep" and "Rajendra Jewellers" in reply["text"]
        yes = next(b["data"] for row in reply["buttons"] for b in row if b["text"] == "Haan")

        r = client.post("/sandbox/tap", json={"as_user": "rep", "data": yes})
        body = r.json()
        stages = {s["stage"]: s for s in body["stages"]}
        assert stages["commit"]["status"] == "pending_approval"  # below the demo floor
        assert {m["to"] for m in body["replies"]} == {"rep", "boss"}  # boss gets Approve/Reject

        boss = next(m for m in body["replies"] if m["to"] == "boss")
        approve = next(b["data"] for row in boss["buttons"] for b in row if b["text"] == "Approve")
        r = client.post("/sandbox/tap", json={"as_user": "boss", "data": approve})
        assert any("Boss ne approve" in m["text"] for m in r.json()["replies"])


def test_sandbox_accepts_an_audio_upload(tmp_path: Path) -> None:
    app, stt, llm = _app(tmp_path)
    assert stt is not None and llm is not None
    stt.say(RATE_TRANSCRIPT)
    llm.replies.append(extraction_json(rate=7_000_000))
    with TestClient(app) as client:
        r = client.post("/sandbox/run", data={"as_user": "rep"},
                        files={"audio": ("rec.ogg", OGG, "audio/ogg")})  # fmt: skip
        stages = [s["stage"] for s in r.json()["stages"]]
        assert stages[:2] == ["media", "transcript"] and "assessment" in stages
        assert stt.calls[0]["language"] == "hi"


def test_sandbox_says_so_when_providers_are_not_configured(tmp_path: Path) -> None:
    app, _, _ = _app(tmp_path, with_providers=False)
    with TestClient(app) as client:
        assert client.get("/sandbox/status").json() == {"stt": False, "llm": False}
        body = client.post("/sandbox/run", data={"as_user": "rep", "text": "hello"}).json()
        assert (
            "chalu nahi" in body["replies"][0]["text"] and body["stages"] == []
        )  # nothing invented


def test_sandbox_page_is_served_and_never_uses_innerhtml(tmp_path: Path) -> None:
    app, _, _ = _app(tmp_path)
    with TestClient(app) as client:
        r = client.get("/sandbox")
        assert r.status_code == 200 and "BharatDesk sandbox" in r.text
        assert "innerHTML" not in r.text and "textContent" in r.text  # untrusted text stays text


def test_sandbox_is_loopback_only(tmp_path: Path) -> None:
    app, _, _ = _app(tmp_path)
    with TestClient(app, client=("203.0.113.9", 5000)) as client:
        assert client.get("/sandbox").status_code == 403
        assert client.post("/sandbox/run", data={"text": "x"}).status_code == 403


def test_sandbox_does_not_exist_in_production(tmp_path: Path) -> None:
    app, _, _ = _app(tmp_path, env="prod")
    with TestClient(app) as client:
        assert client.get("/sandbox").status_code == 404
        assert client.post("/sandbox/run", data={"text": "x"}).status_code == 404


def test_sandbox_rejects_oversized_uploads(tmp_path: Path) -> None:
    app, _, _ = _app(tmp_path)
    app_settings_limit = 20 * 1024 * 1024
    with TestClient(app) as client:
        big = b"OggS" + b"\x00" * (app_settings_limit + 1)
        r = client.post("/sandbox/run", data={"as_user": "rep"}, files={"audio": ("big.ogg", big)})
        assert r.status_code == 413


@pytest.mark.parametrize("who", ["rep", "boss"])
def test_sandbox_users_are_consented_demo_accounts(tmp_path: Path, who: str) -> None:
    app, _, llm = _app(tmp_path)
    assert llm is not None
    with TestClient(app) as client:
        body = client.post("/sandbox/run", data={"as_user": who, "text": "/help"}).json()
        assert "Aap ye kar sakte hain" in body["replies"][0]["text"]  # past the consent gate


# --- the README's 5-minute demo, replayed ---------------------------------------------------

DEMO_SENTENCE = (
    "Rajendra Jewellers gaya tha, dedh sau gram necklace, chaar lakh baarah hazaar ka order, "
    "paintalis hazaar advance, pachattar hazaar rate, kal delivery"
)


def test_the_readme_demo_steps_behave_as_documented(tmp_path: Path) -> None:
    app, _, llm = _app(tmp_path)
    assert llm is not None
    good = extraction_json(rate=7_500_000, delivery="kal")
    low = extraction_json(rate=7_000_000, delivery="kal")
    low_sentence = DEMO_SENTENCE.replace("pachattar", "sattar")
    with TestClient(app) as client:
        # step 4: a clean visit is read back and saved on Haan
        llm.replies.append(good)
        body = client.post("/sandbox/run", data={"as_user": "rep", "text": DEMO_SENTENCE}).json()
        text = body["replies"][0]["text"]
        assert text.startswith(
            "Rajendra Jewellers, 150 gram, ₹4,12,000, rate ₹75,000 (10 gram), advance ₹45,000"
        )
        assert "Dhyan" not in text
        yes = next(
            b["data"] for row in body["replies"][0]["buttons"] for b in row if b["text"] == "Haan"
        )
        saved = client.post("/sandbox/tap", json={"as_user": "rep", "data": yes}).json()
        assert "Save ho gaya" in saved["replies"][0]["text"] and {
            m["to"] for m in saved["replies"]
        } == {"rep"}
        # step 5: a quote below the benchmark floor reaches the boss, who approves
        llm.replies.append(low)
        body = client.post("/sandbox/run", data={"as_user": "rep", "text": low_sentence}).json()
        yes = next(
            b["data"] for row in body["replies"][0]["buttons"] for b in row if b["text"] == "Haan"
        )
        sent = client.post("/sandbox/tap", json={"as_user": "rep", "data": yes}).json()
        boss = next(m for m in sent["replies"] if m["to"] == "boss")
        assert [b["text"] for b in boss["buttons"][0]] == ["Approve", "Reject", "Counter"]
        approve = next(b["data"] for b in boss["buttons"][0] if b["text"] == "Approve")
        done = client.post("/sandbox/tap", json={"as_user": "boss", "data": approve}).json()
        assert any(m["to"] == "rep" and "Boss ne approve" in m["text"] for m in done["replies"])
