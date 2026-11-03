"""Boss dashboard: signed expiring link, session cookie, read-only views, corrections."""

from __future__ import annotations

import re
from collections.abc import Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr, ValidationError
from sqlalchemy import select

from app.config import Settings
from app.main import create_app
from app.models import ActivityLog, AuditLog, Correction, SyncJob, User
from app.models.enums import ActivityStatus, Role, SyncStatus
from app.pack import load_pack
from app.providers.messaging import FakeProvider
from app.schemas.extraction import VisitPayload
from app.security import csrf_token, sign_token, verify_token
from app.services.corrections import apply_changes, current_values, parse_form
from app.services.sheets_sync import sync_once
from tests.conftest import (
    BOSS_CHAT,
    OGG,
    REP_CHAT,
    SHEET_ID,
    TRANSCRIPT,
    Env,
    button_data,
    extraction_json,
    slip_json,
    to,
)

SECRET = "d" * 40
BASE = "http://desk.local"


def _link(out: list) -> str:  # type: ignore[type-arg]
    m = next(
        (
            re.search(r"http://desk\.local(/dashboard\?t=\S+)", msg.text)
            for msg in out
            if "desk.local" in msg.text
        ),
        None,
    )
    assert m, out
    return m.group(1)


@pytest.fixture
def dash(pipe: Env) -> Iterator[tuple[Env, TestClient]]:
    """The bot environment plus a dashboard app on the SAME database and clock."""
    pipe.settings.dashboard_secret = SecretStr(SECRET)
    pipe.settings.public_base_url = BASE
    settings = Settings(  # type: ignore[call-arg]
        _env_file=None, database_url=pipe.settings.database_url, auto_migrate=False,
        dashboard_secret=SECRET, public_base_url=BASE, media_dir=str(pipe.media_dir),
        sandbox_enabled=False, background_tasks=False,
    )  # fmt: skip
    with TestClient(
        create_app(settings, messaging=FakeProvider()), follow_redirects=False
    ) as client:
        client.app.state.clock = pipe.clock  # type: ignore[attr-defined]
        yield pipe, client


async def _boss_session(p: Env, client: TestClient, chat: int = BOSS_CHAT) -> None:
    out = await p.say(chat, "/dashboard")
    r = client.get(_link(out))
    assert r.status_code == 303 and r.headers["location"] == "/dashboard"


async def _visit(p: Env, transcript: str = TRANSCRIPT, **kw: Any) -> None:
    out = await p.voice(REP_CHAT, transcript, extraction_json(**kw))
    await p.tap(REP_CHAT, button_data(to(REP_CHAT, out)[0], "Haan"))


async def _org(p: Env):  # type: ignore[no-untyped-def]
    org = await p.org()
    await p.add_client(org, "Rajendra Jewellers")
    return org


# --- issuing and opening the link ----------------------------------------------------------


async def test_boss_gets_a_signed_expiring_link(dash: tuple[Env, TestClient]) -> None:
    p, _ = dash
    org = await _org(p)
    out = await p.say(BOSS_CHAT, "/dashboard")
    text = to(BOSS_CHAT, out)[0].text
    assert "60 minute mein expire" in text
    token = _link(out).split("?t=")[1]
    payload = verify_token(SECRET, token, now=int(p.clock().timestamp()))
    assert payload == {"k": "dash", "t": org.tenant_id.hex, "u": org.boss_id.hex,
                       "exp": int(p.clock().timestamp()) + 3600}  # fmt: skip
    assert any(a.action == "dashboard_link_issued" for a in await p.all(AuditLog))


async def test_only_the_boss_can_ask_for_a_link(dash: tuple[Env, TestClient]) -> None:
    p, _ = dash
    await _org(p)
    assert "sirf boss" in to(REP_CHAT, await p.say(REP_CHAT, "/dashboard"))[0].text


async def test_without_a_secret_the_bot_says_the_dashboard_is_off(pipe: Env) -> None:
    await pipe.org()
    assert (
        "Dashboard abhi chalu nahi"
        in to(BOSS_CHAT, await pipe.say(BOSS_CHAT, "/dashboard"))[0].text
    )


async def test_opening_the_link_sets_an_httponly_strict_cookie_and_hides_the_token(
    dash: tuple[Env, TestClient],
) -> None:
    p, client = dash
    await _org(p)
    r = client.get(_link(await p.say(BOSS_CHAT, "/dashboard")))
    cookie = r.headers["set-cookie"]
    assert (
        r.status_code == 303
        and r.headers["location"] == "/dashboard"
        and "?t=" not in r.headers["location"]
    )
    for attr in ("bd_session=", "HttpOnly", "SameSite=strict", "Path=/dashboard", "Max-Age=3600"):
        assert attr in cookie, attr
    page = client.get("/dashboard")
    assert page.status_code == 200 and "Sharma Jewellers" in page.text
    for header, value in (("referrer-policy", "no-referrer"), ("x-frame-options", "DENY"),
                          ("cache-control", "no-store"), ("x-content-type-options", "nosniff")):  # fmt: skip
        assert page.headers[header] == value


async def test_the_cookie_is_secure_on_https(pipe: Env) -> None:
    pipe.settings.dashboard_secret = SecretStr(SECRET)
    pipe.settings.public_base_url = "https://desk.example"
    await pipe.org()
    out = await pipe.say(BOSS_CHAT, "/dashboard")
    link = re.search(r"https://desk\.example(/dashboard\?t=\S+)", to(BOSS_CHAT, out)[0].text).group(
        1
    )  # type: ignore[union-attr]
    settings = Settings(_env_file=None, database_url=pipe.settings.database_url, auto_migrate=False,  # type: ignore[call-arg]
                        dashboard_secret=SECRET, public_base_url="https://desk.example", sandbox_enabled=False,
                        background_tasks=False)  # fmt: skip
    with TestClient(
        create_app(settings, messaging=FakeProvider()), follow_redirects=False
    ) as client:
        client.app.state.clock = pipe.clock  # type: ignore[attr-defined]
        assert "Secure" in client.get(link).headers["set-cookie"]


# --- authentication ------------------------------------------------------------------------


async def test_no_cookie_means_the_expired_page(dash: tuple[Env, TestClient]) -> None:
    _, client = dash
    r = client.get("/dashboard")
    assert r.status_code == 401 and "expired or is not valid" in r.text and "/dashboard" in r.text
    assert client.get("/dashboard/partials/feed").status_code == 401


async def test_a_tampered_token_is_refused(dash: tuple[Env, TestClient]) -> None:
    p, client = dash
    await _org(p)
    token = _link(await p.say(BOSS_CHAT, "/dashboard")).split("?t=")[1]
    body, sig = token.split(".")
    for bad in (f"{body}.{sig[:-2]}AA", f"{body[:-2]}AA.{sig}", token + "x", "garbage", "a.b", ""):
        assert client.get(f"/dashboard?t={bad}").status_code == 401, bad


async def test_an_expired_link_is_refused_and_a_session_expires_too(
    dash: tuple[Env, TestClient],
) -> None:
    p, client = dash
    await _org(p)
    link = _link(await p.say(BOSS_CHAT, "/dashboard"))
    p.clock.advance(minutes=60, seconds=1)
    assert client.get(link).status_code == 401
    p.clock.advance(minutes=-60)
    assert client.get(link).status_code == 303  # valid inside its hour
    assert client.get("/dashboard").status_code == 200
    p.clock.advance(minutes=60)
    assert client.get("/dashboard").status_code == 401  # the cookie's token has expired as well


async def test_a_token_signed_with_another_secret_or_kind_is_refused(
    dash: tuple[Env, TestClient],
) -> None:
    p, client = dash
    org = await _org(p)
    exp = int(p.clock().timestamp()) + 600
    claims = {"t": org.tenant_id.hex, "u": org.boss_id.hex, "exp": exp}
    assert (
        client.get("/dashboard?t=" + sign_token("x" * 40, {"k": "dash", **claims})).status_code
        == 401
    )
    assert (
        client.get("/dashboard?t=" + sign_token(SECRET, {"k": "other", **claims})).status_code
        == 401
    )
    assert (
        client.get("/dashboard?t=" + sign_token(SECRET, {"k": "dash", **claims})).status_code == 303
    )


async def test_a_demoted_or_deactivated_boss_loses_access_at_once(
    dash: tuple[Env, TestClient],
) -> None:
    p, client = dash
    org = await _org(p)
    await _boss_session(p, client)
    assert client.get("/dashboard").status_code == 200
    async with p.sf() as s, s.begin():
        (await s.get(User, org.boss_id)).is_active = False  # type: ignore[union-attr]
    assert client.get("/dashboard").status_code == 401
    async with p.sf() as s, s.begin():
        boss = await s.get(User, org.boss_id)
        boss.is_active, boss.role = True, Role.WORKER  # type: ignore[union-attr]
    assert client.get("/dashboard").status_code == 401


async def test_a_reps_token_is_useless(dash: tuple[Env, TestClient]) -> None:
    p, client = dash
    org = await _org(p)
    exp = int(p.clock().timestamp()) + 600
    token = sign_token(
        SECRET, {"k": "dash", "t": org.tenant_id.hex, "u": org.rep_id.hex, "exp": exp}
    )
    client.get("/dashboard?t=" + token)
    assert client.get("/dashboard").status_code == 401


# --- content -------------------------------------------------------------------------------


async def test_the_dashboard_shows_todays_numbers_the_feed_and_the_audio(
    dash: tuple[Env, TestClient],
) -> None:
    p, client = dash
    await _org(p)
    await _visit(p)
    await _boss_session(p, client)
    html = client.get("/dashboard").text
    for needle in ("Visits today", "₹4,12,000", "Ramesh", "Rajendra Jewellers", "150 gram", "confirmed",
                   "dedh sau gram necklace", "/dashboard/media/", "Correct"):  # fmt: skip
        assert needle in html, needle
    assert 'hx-trigger="every 10s"' in html and "htmx.org@2.0.4" in html
    (act,) = await p.all(ActivityLog)
    media = client.get(f"/dashboard/media/{act.id.hex}")
    assert (
        media.status_code == 200
        and media.content == OGG
        and media.headers["content-type"] == "audio/ogg"
    )


async def test_todays_counts_exclude_other_days_and_rejected_records(
    dash: tuple[Env, TestClient],
) -> None:
    p, client = dash
    await _org(p)
    await _visit(p)
    out = await p.voice(REP_CHAT, TRANSCRIPT, extraction_json())
    await p.tap(REP_CHAT, button_data(to(REP_CHAT, out)[0], "Nahi"))  # rejected: not counted
    await _boss_session(p, client)
    summary = client.get("/dashboard/partials/summary").text
    assert re.search(r">\s*1\s*<", summary) and "₹4,12,000" in summary
    p.clock.advance(days=1)
    await _boss_session(p, client)
    assert "₹0" in client.get("/dashboard/partials/summary").text  # a new day starts at zero


async def test_per_rep_counts_and_pending_approvals(dash: tuple[Env, TestClient]) -> None:
    p, client = dash
    org = await _org(p)
    await p.add_rules(org, ("max_order_value_without_approval", {"max_value_paise": 10_000_000}))
    await _visit(p)  # above the limit -> approval
    await _boss_session(p, client)
    approvals = client.get("/dashboard/partials/approvals").text
    assert (
        "Pending approvals" in approvals
        and "Ramesh" in approvals
        and "exceeds the no-approval limit" in approvals
    )
    assert "Approve or reject in Telegram" in approvals  # the dashboard itself stays read-only
    summary = client.get("/dashboard/partials/summary").text
    assert "Ramesh" in summary
    # an order awaiting approval is a visit, but its value is NOT counted as order value
    assert "₹0" in summary and "₹4,12,000 awaiting your approval (not counted)" in summary


async def test_the_sheet_link_and_sync_status_are_shown(dash: tuple[Env, TestClient]) -> None:
    p, client = dash
    await _org(p)
    await _boss_session(p, client)
    assert "No sheet connected" in client.get("/dashboard/partials/sync").text
    await p.connect_sheet()
    await _visit(p)
    html = client.get("/dashboard/partials/sync").text
    assert f"spreadsheets/d/{SHEET_ID}" in html and "0 synced" in html and "1 waiting" in html
    await sync_once(p.sf, p.sheets, load_pack(), now=p.clock(), max_attempts=8)  # type: ignore[arg-type]
    assert "1 synced" in client.get("/dashboard/partials/sync").text


async def test_slip_photos_are_viewable_from_the_feed(dash: tuple[Env, TestClient]) -> None:
    p, client = dash
    await _org(p)
    out = await p.photo(REP_CHAT, slip_json())
    await p.tap(REP_CHAT, button_data(to(REP_CHAT, out)[0], "Haan"))
    await _boss_session(p, client)
    assert "View slip photo" in client.get("/dashboard/partials/feed").text
    (act,) = await p.all(ActivityLog)
    r = client.get(f"/dashboard/media/{act.id.hex}")
    assert r.headers["content-type"] == "image/jpeg"


async def test_untrusted_text_is_escaped(dash: tuple[Env, TestClient]) -> None:
    p, client = dash
    await _org(p)
    evil_client = "<script>alert(1)</script> Co"
    evil_speech = TRANSCRIPT + " <img src=x onerror=alert(2)>"
    out = await p.voice(REP_CHAT, evil_speech, extraction_json(client=evil_client))
    await p.tap(REP_CHAT, button_data(to(REP_CHAT, out)[0], "Haan"))
    await _boss_session(p, client)
    html = client.get("/dashboard").text
    assert (
        "&lt;script&gt;alert(1)&lt;/script&gt;" in html
        and "&lt;img src=x onerror=alert(2)&gt;" in html
    )
    assert "<script>alert(1)" not in html and "<img src=x" not in html


async def test_partials_are_fragments_and_unknown_ones_404(dash: tuple[Env, TestClient]) -> None:
    p, client = dash
    await _org(p)
    await _boss_session(p, client)
    for name in ("summary", "approvals", "feed", "sync"):
        r = client.get(f"/dashboard/partials/{name}")
        assert r.status_code == 200 and "<html" not in r.text
    assert client.get("/dashboard/partials/secrets").status_code == 404
    assert client.get("/dashboard/partials/../media").status_code in (404, 422)


async def test_media_is_tenant_scoped_and_validated(dash: tuple[Env, TestClient]) -> None:
    p, client = dash
    await _org(p)
    await _visit(p)
    (act,) = await p.all(ActivityLog)
    _, _, raw = await p.new_tenant("Other Co", "Mallory")
    await p.say(3003, f"/start {raw}")
    await p.tap(3003, "co:y")
    other = TestClient(client.app, follow_redirects=False)
    await _boss_session(p, other, 3003)
    assert other.get(f"/dashboard/media/{act.id.hex}").status_code == 404  # another tenant's audio
    assert other.get(f"/dashboard/correct/{act.id.hex}").status_code == 404
    assert client.get("/dashboard/media/not-a-uuid").status_code in (401, 404)
    await _boss_session(p, client)
    assert client.get("/dashboard/media/not-a-uuid").status_code == 404
    assert client.get(f"/dashboard/media/{'0' * 32}").status_code == 404


async def test_the_dashboard_exposes_no_write_methods_except_corrections(
    dash: tuple[Env, TestClient],
) -> None:
    p, client = dash
    await _org(p)
    await _boss_session(p, client)
    for path in ("/dashboard", "/dashboard/partials/feed", "/dashboard/media/" + "0" * 32):
        assert client.post(path).status_code == 405
        assert client.delete(path).status_code == 405


async def test_the_dashboard_does_not_exist_without_a_secret(pipe: Env, tmp_path: Any) -> None:
    settings = Settings(_env_file=None, database_url=pipe.settings.database_url, auto_migrate=False,  # type: ignore[call-arg]
                        sandbox_enabled=False, background_tasks=False)  # fmt: skip
    with TestClient(create_app(settings, messaging=FakeProvider())) as client:
        assert client.get("/dashboard").status_code == 404


# --- corrections ---------------------------------------------------------------------------


async def _correction_form(p: Env, client: TestClient) -> tuple[str, str]:
    (act,) = await p.all(ActivityLog)
    r = client.get(f"/dashboard/correct/{act.id.hex}")
    assert r.status_code == 200
    return act.id.hex, re.search(r'name="csrf" value="([0-9a-f]+)"', r.text).group(1)  # type: ignore[union-attr]


async def test_the_form_shows_the_current_values_in_human_units(
    dash: tuple[Env, TestClient],
) -> None:
    p, client = dash
    await _org(p)
    await _visit(p)
    await _boss_session(p, client)
    (act,) = await p.all(ActivityLog)
    html = client.get(f"/dashboard/correct/{act.id.hex}").text
    for value in ('value="150"', 'value="412000"', 'value="45000"', 'value="2026-11-08"'):
        assert value in html, value


async def test_a_correction_changes_the_record_and_is_fully_traceable(
    dash: tuple[Env, TestClient],
) -> None:
    p, client = dash
    org = await _org(p)
    await p.connect_sheet()
    await _visit(p)
    await sync_once(p.sf, p.sheets, load_pack(), now=p.clock(), max_attempts=8)  # type: ignore[arg-type]
    await _boss_session(p, client)
    aid, csrf = await _correction_form(p, client)
    r = client.post(f"/dashboard/correct/{aid}", data={
        "csrf": csrf, "weight_g": "150", "value_inr": "4,50,000", "rate_inr": "", "advance_inr": "50000",
        "delivery_date": "2026-11-10", "followup_date": ""})  # fmt: skip
    assert r.status_code == 303 and r.headers["location"] == "/dashboard"
    (act,) = await p.all(ActivityLog)
    visit = act.payload["visit"]
    assert visit["order"]["value_paise"] == 45_000_000 and visit["payment"] == {"received_paise": 5_000_000,
                                                                                 "pending_paise": 40_000_000}  # fmt: skip
    assert visit["delivery_date"] == "2026-11-10" and visit["order"]["weight_mg"] == 150_000
    assert all(isinstance(visit["order"][k], int) for k in ("value_paise", "weight_mg"))
    assert act.payload["meta"]["corrected_by_boss"] is True
    (corr,) = await p.all(Correction)
    assert (
        corr.original["order"]["value_paise"] == 41_200_000
        and corr.corrected["order"]["value_paise"] == 45_000_000
    )
    assert corr.corrected_by == org.boss_id and corr.activity_log_id == act.id
    assert any(
        a.action == "activity_corrected_by_boss" and a.actor_user_id == org.boss_id
        for a in await p.all(AuditLog)
    )
    (job,) = await p.all(SyncJob)
    assert job.status is SyncStatus.PENDING  # re-queued so the SAME sheet row is updated
    await sync_once(p.sf, p.sheets, load_pack(), now=p.clock(), max_attempts=8)  # type: ignore[arg-type]
    (row,) = p.sheets.rows(SHEET_ID)  # type: ignore[union-attr]
    assert (row[5], row[7], row[8], row[9]) == (450_000, 50_000, 400_000, "2026-11-10")
    assert "(corrected)" in client.get("/dashboard/partials/feed").text


async def test_invalid_corrections_change_nothing(dash: tuple[Env, TestClient]) -> None:
    p, client = dash
    await _org(p)
    await _visit(p)
    await _boss_session(p, client)
    aid, csrf = await _correction_form(p, client)
    base = {"csrf": csrf, "weight_g": "150", "value_inr": "412000", "rate_inr": "", "advance_inr": "45000",
            "delivery_date": "2026-11-08", "followup_date": ""}  # fmt: skip
    cases = {
        "value_inr": [
            "abc",
            "4.5.6",
            "-5",
            "1e5",
            "0.005",
            "5",
        ],  # junk, float forms, fractional paise, below bounds
        "weight_g": ["0.0001", "999999999"],
        "delivery_date": ["8/11/2026", "2026-13-40"],
    }
    for name, bad_values in cases.items():
        for bad in bad_values:
            r = client.post(f"/dashboard/correct/{aid}", data={**base, name: bad})
            assert r.status_code == 422, (name, bad)
    (act,) = await p.all(ActivityLog)
    assert (
        act.payload["visit"]["order"]["value_paise"] == 41_200_000 and await p.all(Correction) == []
    )
    r = client.post(f"/dashboard/correct/{aid}", data=base)  # nothing changed
    assert r.status_code == 422 and "Nothing was changed" in r.text


async def test_the_form_redisplays_what_was_typed_with_errors(dash: tuple[Env, TestClient]) -> None:
    p, client = dash
    await _org(p)
    await _visit(p)
    await _boss_session(p, client)
    aid, csrf = await _correction_form(p, client)
    r = client.post(f"/dashboard/correct/{aid}", data={"csrf": csrf, "value_inr": "oops"})
    assert r.status_code == 422 and 'value="oops"' in r.text and "plain number" in r.text


async def test_csrf_is_required_and_bound_to_the_session_and_record(
    dash: tuple[Env, TestClient],
) -> None:
    p, client = dash
    await _org(p)
    await _visit(p)
    await _boss_session(p, client)
    aid, csrf = await _correction_form(p, client)
    good = {
        "value_inr": "450000",
        "weight_g": "150",
        "advance_inr": "45000",
        "delivery_date": "2026-11-08",
    }
    for token in (
        "",
        "0" * 64,
        csrf[:-1] + ("0" if csrf[-1] != "0" else "1"),
        csrf_token(SECRET, "other-session", aid),
        csrf_token(SECRET, "x", "0" * 32),
    ):
        assert (
            client.post(f"/dashboard/correct/{aid}", data={**good, "csrf": token}).status_code
            == 403
        )
    assert client.post(f"/dashboard/correct/{aid}", data=good).status_code == 403  # no token at all
    assert await p.all(Correction) == []
    assert client.post(f"/dashboard/correct/{aid}", data={**good, "csrf": csrf}).status_code == 303


async def test_a_correction_without_a_session_is_refused(dash: tuple[Env, TestClient]) -> None:
    p, client = dash
    await _org(p)
    await _visit(p)
    (act,) = await p.all(ActivityLog)
    r = client.post(f"/dashboard/correct/{act.id.hex}", data={"csrf": "x", "value_inr": "1"})
    assert r.status_code == 401 and await p.all(Correction) == []


async def test_unconfirmed_records_cannot_be_corrected(dash: tuple[Env, TestClient]) -> None:
    p, client = dash
    org = await _org(p)
    await p.add_rules(org, ("max_order_value_without_approval", {"max_value_paise": 10_000_000}))
    await _visit(p)  # awaiting approval
    await _boss_session(p, client)
    (act,) = await p.all(ActivityLog)
    aid = act.id.hex
    csrf = csrf_token(
        SECRET, client.cookies["bd_session"], aid
    )  # a valid token: still must be refused
    html = client.get(f"/dashboard/correct/{aid}").text
    assert "Only confirmed records can be corrected" in html and "<form" not in html
    r = client.post(f"/dashboard/correct/{aid}", data={"csrf": csrf, "value_inr": "450000"})
    assert r.status_code == 409 and await p.all(Correction) == []
    async with p.sf() as s:
        assert (
            await s.execute(select(ActivityLog))
        ).scalar_one().status is ActivityStatus.PENDING_APPROVAL


# --- unit level ----------------------------------------------------------------------------


def test_token_verification_matrix() -> None:
    t = sign_token(SECRET, {"exp": 2000, "k": "dash"})
    assert verify_token(SECRET, t, now=1999) == {"exp": 2000, "k": "dash"}
    assert verify_token(SECRET, t, now=2000) is None and verify_token(SECRET, t, now=2001) is None
    assert verify_token("x" * 40, t, now=1) is None
    for bad in ("", ".", "a", "a.b", t + ".x", t.replace(".", ",")):
        assert verify_token(SECRET, bad, now=1) is None
    for payload in ({"k": "dash"}, {"exp": "2000"}, {"exp": True}, {"exp": 2000.5}):
        assert (
            verify_token(SECRET, sign_token(SECRET, payload), now=1) is None
        )  # exp must be a real int


def test_csrf_tokens_are_scoped() -> None:
    a = csrf_token(SECRET, "sess", "rec1")
    assert a == csrf_token(SECRET, "sess", "rec1")
    assert a != csrf_token(SECRET, "sess", "rec2") != csrf_token(SECRET, "other", "rec2")
    assert a != csrf_token("y" * 40, "sess", "rec1")


def test_the_dashboard_secret_must_be_long_enough() -> None:
    with pytest.raises(ValidationError, match="at least 32"):
        Settings(_env_file=None, dashboard_secret="short")  # type: ignore[call-arg,arg-type]


def _payload() -> VisitPayload:
    return VisitPayload.model_validate_json(
        '{"client_name":"A","visit_date":"2026-11-02","order":{"items":[],"weight_mg":150000,'
        '"value_paise":41200000,"rate_paise_per_10g":null},"payment":{"received_paise":4500000,'
        '"pending_paise":36700000},"delivery_date":"2026-11-08","next_followup_date":null}'
    )


def test_parse_form_only_reports_real_changes_in_exact_units() -> None:
    cur = _payload()
    shown = current_values(cur)
    assert shown["value_inr"] == "412000" and shown["weight_g"] == "150"
    res = parse_form({**shown, "weight_g": "150.5", "advance_inr": "4,50,000.50"}, cur, load_pack())
    assert res.errors == {} and res.changes == {"weight_g": 150_500, "advance_inr": 45_000_050}
    assert parse_form(shown, cur, load_pack()).changes == {}


def test_apply_changes_recomputes_pending_only_when_needed() -> None:
    cur = _payload()
    new = apply_changes(cur, {"value_inr": 50_000_000})
    assert new.payment.pending_paise == 45_500_000 and cur.payment.pending_paise == 36_700_000  # type: ignore[union-attr]
    new = apply_changes(cur, {"advance_inr": 60_000_000})
    assert new.payment.pending_paise == 0  # never negative
    new = apply_changes(cur, {"delivery_date": None})
    assert new.delivery_date is None and new.payment.pending_paise == 36_700_000  # type: ignore[union-attr]
