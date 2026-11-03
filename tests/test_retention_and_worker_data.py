"""Retention purge, worker data export and erasure, and the operator CLI."""

from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path

import pytest
from pydantic import ValidationError
from sqlalchemy import select

from app.models import (
    ActivityLog,
    AuditLog,
    ConversationState,
    Correction,
    Extraction,
    Message,
    Tenant,
    User,
)
from app.models.enums import ActivityStatus, Role, SyncStatus
from app.pack import load_pack
from app.pipeline.confirm import TenantSettings
from app.pipeline.media import MediaStore, sha256_hex
from app.services import retention, worker_data
from app.services.sheets_sync import sync_once
from scripts import admin
from tests.conftest import (
    BOSS_CHAT,
    JPEG,
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

DAY = timedelta(days=1)


async def _org(p: Env):  # type: ignore[no-untyped-def]
    org = await p.org()
    await p.add_client(org, "Rajendra Jewellers")
    return org


async def _voice_visit(p: Env, data: bytes = OGG) -> None:
    out = await p.voice(REP_CHAT, TRANSCRIPT, extraction_json(), data=data)
    await p.tap(REP_CHAT, button_data(to(REP_CHAT, out)[0], "Haan"))


def _store(p: Env) -> MediaStore:
    assert p.media_dir is not None
    return MediaStore(p.media_dir)


async def _purge(p: Env) -> retention.PurgeReport:
    return await retention.purge_media(p.sf, _store(p), p.clock())


def _file(p: Env, org, data: bytes = OGG, ext: str = "ogg") -> Path:  # type: ignore[no-untyped-def]
    return _store(p).path_for(org.tenant_id, sha256_hex(data), ext)


# --- retention -----------------------------------------------------------------------------


async def test_audio_is_deleted_after_the_default_90_days_and_everything_else_stays(
    pipe: Env,
) -> None:
    org = await _org(pipe)
    await _voice_visit(pipe)
    assert _file(pipe, org).exists()
    pipe.clock.advance(days=91)
    report = await _purge(pipe)
    assert (report.messages_purged, report.files_deleted) == (1, 1) and not _file(
        pipe, org
    ).exists()
    (msg,) = await pipe.all(Message)
    assert msg.media_purged_at == pipe.clock()
    (ex,) = await pipe.all(Extraction)
    assert ex.transcript == TRANSCRIPT  # only the media goes; transcript and record stay
    assert (await pipe.all(ActivityLog))[0].status is ActivityStatus.CONFIRMED
    audit = [a for a in await pipe.all(AuditLog) if a.action == "media_purged"]
    assert len(audit) == 1 and audit[0].detail == {"kind": "voice", "reason": "retention"}


async def test_the_retention_boundary_is_exact(pipe: Env) -> None:
    org = await _org(pipe)
    await _voice_visit(pipe)
    pipe.clock.advance(days=90)
    assert (await _purge(pipe)).messages_purged == 0 and _file(
        pipe, org
    ).exists()  # exactly 90 d: kept
    pipe.clock.advance(seconds=1)
    assert (await _purge(pipe)).messages_purged == 1 and not _file(pipe, org).exists()


async def test_the_purge_is_idempotent_and_survives_a_missing_file(pipe: Env) -> None:
    org = await _org(pipe)
    await _voice_visit(pipe)
    pipe.clock.advance(days=100)
    _file(pipe, org).unlink()  # e.g. an earlier run died after deleting the file
    first = await _purge(pipe)
    assert (first.messages_purged, first.files_deleted) == (1, 0)
    again = await _purge(pipe)
    assert (again.messages_purged, again.files_deleted) == (0, 0)
    assert len([a for a in await pipe.all(AuditLog) if a.action == "media_purged"]) == 1


async def test_audio_and_images_have_independent_per_tenant_retention(pipe: Env) -> None:
    org = await _org(pipe)
    await pipe.tenant_settings(
        org, audio_retention_days=30, image_retention_days=None
    )  # photos forever
    await _voice_visit(pipe)
    out = await pipe.photo(REP_CHAT, slip_json())
    await pipe.tap(REP_CHAT, button_data(to(REP_CHAT, out)[0], "Haan"))
    pipe.clock.advance(days=400)
    report = await _purge(pipe)
    assert report.messages_purged == 1  # the voice note only
    assert not _file(pipe, org).exists() and _file(pipe, org, JPEG, "jpg").exists()
    await pipe.tenant_settings(org, image_retention_days=1)
    assert (await _purge(pipe)).messages_purged == 1 and not _file(pipe, org, JPEG, "jpg").exists()


async def test_a_shared_file_is_deleted_only_when_every_reference_is_due(pipe: Env) -> None:
    org = await _org(pipe)
    await _voice_visit(pipe)  # same bytes sent twice, 40 days apart
    pipe.clock.advance(days=40)
    await _voice_visit(pipe)
    pipe.clock.advance(days=55)  # first is 95 days old, second 55
    report = await _purge(pipe)
    assert (report.messages_purged, report.files_deleted) == (1, 0) and _file(pipe, org).exists()
    pipe.clock.advance(days=40)
    report = await _purge(pipe)
    assert (report.messages_purged, report.files_deleted) == (1, 1) and not _file(
        pipe, org
    ).exists()


async def test_one_tenants_policy_never_touches_anothers_media(pipe: Env) -> None:
    org = await _org(pipe)
    await _voice_visit(pipe)
    _, _, raw = await pipe.new_tenant("Other Co", "Mallory")
    await pipe.say(3003, f"/start {raw}")
    await pipe.tap(3003, "co:y")
    pipe.llm.replies.append(extraction_json(client="Zed"))  # type: ignore[union-attr]
    await pipe.voice(3003, TRANSCRIPT, extraction_json(client="Zed"), data=OGG + b"B")
    async with pipe.sf() as s, s.begin():
        other = next(t for t in (await s.execute(select(Tenant))).scalars() if t.name == "Other Co")
        other.settings = {"audio_retention_days": None}  # Other Co keeps audio forever
    pipe.clock.advance(days=500)
    report = await _purge(pipe)
    assert report.messages_purged == 1 and not _file(pipe, org).exists()
    other_path = _store(pipe).path_for(other.id, sha256_hex(OGG + b"B"), "ogg")
    assert other_path.exists()
    only = await retention.purge_media(pipe.sf, _store(pipe), pipe.clock(), only_tenant=other.id)
    assert only.messages_purged == 0 and other_path.exists()


async def test_text_notes_have_no_media_to_purge(pipe: Env) -> None:
    await _org(pipe)
    await pipe.note(REP_CHAT, TRANSCRIPT, extraction_json())
    pipe.clock.advance(days=1000)
    assert (await _purge(pipe)).messages_purged == 0


def test_retention_settings_are_validated() -> None:
    assert (
        TenantSettings().audio_retention_days == 90 and TenantSettings().image_retention_days == 90
    )
    assert TenantSettings.from_json({"audio_retention_days": None}).audio_retention_days is None
    for bad in (0, -5):
        with pytest.raises(ValidationError):
            TenantSettings.from_json({"audio_retention_days": bad})


def test_media_store_delete_rejects_bad_hashes_and_ignores_missing_files(tmp_path: Path) -> None:
    import uuid

    store, tenant = MediaStore(tmp_path), uuid.uuid4()
    assert store.delete(tenant, "a" * 64) == 0  # nothing there is fine
    for bad in ("../x", "a" * 63, ""):
        with pytest.raises(ValueError, match="bad sha256"):
            store.delete(tenant, bad)


# --- worker export -------------------------------------------------------------------------


async def test_export_contains_everything_held_about_the_worker_as_plain_json(pipe: Env) -> None:
    org = await _org(pipe)
    await _voice_visit(pipe)
    out = await pipe.voice(REP_CHAT, TRANSCRIPT, extraction_json())
    await pipe.tap(REP_CHAT, button_data(to(REP_CHAT, out)[0], "Sudharo"))
    await pipe.voice(REP_CHAT, TRANSCRIPT, extraction_json(delivery="kal"))
    async with pipe.sf() as s:
        from app.db import TenantRepository

        repo = TenantRepository(s, org.tenant_id)
        data = await worker_data.export_worker(
            repo, (await repo.get(User, org.rep_id)), pipe.clock()
        )  # type: ignore[arg-type]
    json.dumps(data)  # must be serialisable
    assert data["user"]["name"] == "Ramesh" and data["user"]["telegram_chat_id"] == REP_CHAT
    assert data["consent"] == {"accepted_at": pipe.clock().isoformat(), "notice_version": "v1"}
    assert [m["kind"] for m in data["messages"]] == ["voice", "voice", "voice"]
    assert all(
        m["transcript"] == TRANSCRIPT and len(m["media_sha256"]) == 64 for m in data["messages"]
    )
    assert len(data["records"]) == 2 and len(data["corrections"]) == 1
    assert {"consent_accepted", "invite_redeemed"} <= {a["action"] for a in data["audit_events"]}


async def test_export_does_not_include_other_peoples_data(pipe: Env) -> None:
    org = await _org(pipe)
    await _voice_visit(pipe)
    async with pipe.sf() as s:
        from app.db import TenantRepository

        repo = TenantRepository(s, org.tenant_id)
        boss = await worker_data.export_worker(
            repo, (await repo.get(User, org.boss_id)), pipe.clock()
        )  # type: ignore[arg-type]
    assert boss["messages"] == [] and boss["records"] == []


# --- worker erasure ------------------------------------------------------------------------


async def _erase(p: Env, org, *, actor_boss: bool = True):  # type: ignore[no-untyped-def]
    from app.db import TenantRepository

    async with p.sf() as s, s.begin():
        repo = TenantRepository(s, org.tenant_id)
        worker = await repo.get(User, org.rep_id)
        actor = await repo.get(User, org.boss_id) if actor_boss else None
        return await worker_data.erase_worker(repo, _store(p), worker, actor=actor, now=p.clock())  # type: ignore[arg-type]


async def test_erasing_a_worker_removes_personal_data_but_keeps_the_business_record(
    pipe: Env,
) -> None:
    org = await _org(pipe)
    await pipe.connect_sheet()
    await _voice_visit(pipe)
    await sync_once(pipe.sf, pipe.sheets, load_pack(), now=pipe.clock(), max_attempts=8)  # type: ignore[arg-type]
    assert pipe.sheets.rows(SHEET_ID)[0][2] == "Ramesh"  # type: ignore[union-attr]
    await pipe.voice(REP_CHAT, TRANSCRIPT, extraction_json(client=None))  # leaves an open question
    assert any(s.kind == "slot" for s in await pipe.all(ConversationState))

    report = await _erase(pipe, org)
    assert (report.messages_scrubbed, report.media_files_deleted, report.records_resynced) == (
        2,
        1,
        1,
    )
    async with pipe.sf() as s:
        rep = await s.get(User, org.rep_id)
    assert (rep.name, rep.telegram_chat_id, rep.is_active, rep.consent_at, rep.consent_version) == (  # type: ignore[union-attr]
        "Deleted worker",
        None,
        False,
        None,
        None,
    )
    assert rep.erased_at == pipe.clock()  # type: ignore[union-attr]
    assert not _file(pipe, org).exists()
    for m in await pipe.all(Message):
        assert (m.file_id, m.sha256) == (None, None) and m.media_purged_at == pipe.clock()
    for e in await pipe.all(Extraction):
        assert (e.transcript, e.raw_output, e.validated_payload) == (None, None, None)
    assert [s for s in await pipe.all(ConversationState) if s.user_id == org.rep_id] == []
    # the company's record survives, now attributed to the erased user
    saved = (ActivityStatus.CONFIRMED, ActivityStatus.SYNCED)
    (act,) = [a for a in await pipe.all(ActivityLog) if a.status in saved]
    assert act.payload["visit"]["order"]["value_paise"] == 41_200_000 and act.user_id == org.rep_id
    erased = [a for a in await pipe.all(AuditLog) if a.action == "worker_erased"]
    assert len(erased) == 1 and erased[0].actor_user_id == org.boss_id
    assert "Ramesh" not in json.dumps(
        [a.detail for a in await pipe.all(AuditLog)]
    )  # audit holds ids, not names


async def test_the_erased_workers_sheet_row_loses_their_name(pipe: Env) -> None:
    org = await _org(pipe)
    await pipe.connect_sheet()
    await _voice_visit(pipe)
    await sync_once(pipe.sf, pipe.sheets, load_pack(), now=pipe.clock(), max_attempts=8)  # type: ignore[arg-type]
    await _erase(pipe, org)
    from app.models import SyncJob

    assert (await pipe.all(SyncJob))[0].status is SyncStatus.PENDING  # re-queued
    await sync_once(pipe.sf, pipe.sheets, load_pack(), now=pipe.clock(), max_attempts=8)  # type: ignore[arg-type]
    (row,) = pipe.sheets.rows(SHEET_ID)  # type: ignore[union-attr]
    assert row[2] == "Deleted worker" and len(pipe.sheets.rows(SHEET_ID)) == 1  # updated in place


async def test_an_erased_worker_can_no_longer_use_the_bot(pipe: Env) -> None:
    org = await _org(pipe)
    await _erase(pipe, org)
    out = await pipe.say(REP_CHAT, "/help")
    assert "registered nahi" in out[0].text


async def test_erasing_twice_is_harmless_and_the_boss_cannot_be_erased(pipe: Env) -> None:
    org = await _org(pipe)
    await _erase(pipe, org)
    again = await _erase(pipe, org)
    assert (
        again.already_erased
        and len([a for a in await pipe.all(AuditLog) if a.action == "worker_erased"]) == 1
    )
    from app.db import TenantRepository

    async with pipe.sf() as s, s.begin():
        repo = TenantRepository(s, org.tenant_id)
        with pytest.raises(ValueError, match="owner"):
            await worker_data.erase_worker(
                repo, _store(pipe), await repo.get(User, org.boss_id), actor=None, now=pipe.clock()
            )  # type: ignore[arg-type]


async def test_erasure_leaves_other_workers_and_shared_media_alone(pipe: Env) -> None:
    org = await _org(pipe)
    await _voice_visit(pipe)  # Ramesh sends audio X
    token = await pipe.add_worker(BOSS_CHAT, "Suresh")
    await pipe.say(3003, f"/start {token}")
    await pipe.tap(3003, "co:y")
    out = await pipe.voice(
        3003, TRANSCRIPT, extraction_json(), data=OGG
    )  # Suresh sends identical bytes
    await pipe.tap(3003, button_data(to(3003, out)[0], "Haan"))
    report = await _erase(pipe, org)
    assert (
        report.media_files_deleted == 0 and _file(pipe, org).exists()
    )  # Suresh still needs the file
    async with pipe.sf() as s:
        suresh = (await s.execute(select(User).where(User.name == "Suresh"))).scalar_one()
        msgs = [m for m in (await s.execute(select(Message))).scalars() if m.user_id == suresh.id]
        exs = [
            e for e in (await s.execute(select(Extraction))).scalars() if e.message_id == msgs[0].id
        ]
    assert suresh.is_active and msgs[0].sha256 is not None and exs[0].transcript == TRANSCRIPT


async def test_an_open_correction_survives_erasure_of_another_worker(pipe: Env) -> None:
    org = await _org(pipe)
    out = await pipe.voice(REP_CHAT, TRANSCRIPT, extraction_json())
    await pipe.tap(REP_CHAT, button_data(to(REP_CHAT, out)[0], "Sudharo"))
    await pipe.voice(REP_CHAT, TRANSCRIPT, extraction_json(delivery="kal"))
    await _erase(pipe, org)
    assert len(await pipe.all(Correction)) == 1  # a business trace, kept


# --- operator CLI --------------------------------------------------------------------------


async def test_admin_export_erase_and_retention_commands(pipe: Env) -> None:
    await _org(pipe)
    await _voice_visit(pipe)
    data = await admin.export_worker(pipe.sf, "Sharma Jewellers", "Ramesh")
    assert data["user"]["name"] == "Ramesh" and len(data["messages"]) == 1
    ts = await admin.set_retention(
        pipe.sf, "Sharma Jewellers", 30, None, set_audio=True, set_image=True
    )
    assert (ts.audio_retention_days, ts.image_retention_days) == (30, None)
    with pytest.raises(ValidationError):
        await admin.set_retention(
            pipe.sf, "Sharma Jewellers", 0, None, set_audio=True, set_image=False
        )
    rep = await admin.erase_worker(pipe.sf, _store(pipe), "Sharma Jewellers", "Ramesh")
    assert rep.messages_scrubbed == 1 and not rep.already_erased
    assert any(a.action == "retention_changed" for a in await pipe.all(AuditLog))


async def test_admin_reports_unknown_and_ambiguous_names(pipe: Env) -> None:
    await _org(pipe)
    with pytest.raises(admin.AdminError, match="no tenant named"):
        await admin.export_worker(pipe.sf, "Nope Co", "Ramesh")
    with pytest.raises(admin.AdminError, match="no user named"):
        await admin.export_worker(pipe.sf, "Sharma Jewellers", "Nobody")
    await pipe.new_tenant("Sharma Jewellers", "Twin")
    with pytest.raises(admin.AdminError, match="2 tenants are named"):
        await admin.export_worker(pipe.sf, "Sharma Jewellers", "Ramesh")
    assert (await pipe.all(Tenant)).__len__() == 2 and Role.BOSS


async def test_the_erase_command_refuses_without_confirmation(
    pipe: Env, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from app.config import get_settings

    await _org(pipe)
    monkeypatch.setenv("DATABASE_URL", pipe.settings.database_url)
    get_settings.cache_clear()
    code = await admin.main(["erase-worker", "--tenant", "Sharma Jewellers", "--worker", "Ramesh"])
    assert code == 2 and "--yes" in capsys.readouterr().out
    async with pipe.sf() as s:
        assert (await s.execute(select(User).where(User.name == "Ramesh"))).scalar_one().is_active
    get_settings.cache_clear()


# --- transcript retention --------------------------------------------------------------------


async def test_transcripts_are_kept_unless_a_tenant_opts_in(pipe: Env) -> None:
    await _org(pipe)
    await _voice_visit(pipe)
    pipe.clock.advance(days=3000)
    assert await retention.purge_transcripts(pipe.sf, pipe.clock()) == 0
    assert (await pipe.all(Extraction))[0].transcript == TRANSCRIPT


async def test_opted_in_transcripts_and_raw_output_are_blanked_but_the_record_stays(
    pipe: Env,
) -> None:
    org = await _org(pipe)
    await pipe.tenant_settings(org, transcript_retention_days=30)
    await _voice_visit(pipe)
    pipe.clock.advance(days=30)
    assert await retention.purge_transcripts(pipe.sf, pipe.clock()) == 0  # exactly 30 d: kept
    pipe.clock.advance(seconds=1)
    assert await retention.purge_transcripts(pipe.sf, pipe.clock()) == 1
    (ex,) = await pipe.all(Extraction)
    assert (ex.transcript, ex.raw_output) == (None, None)
    assert (
        ex.validated_payload is not None and ex.field_confidences is not None
    )  # business data stays
    (act,) = await pipe.all(ActivityLog)
    assert (
        act.status is ActivityStatus.CONFIRMED
        and act.payload["visit"]["order"]["value_paise"] == 41_200_000
    )
    audit = [a for a in await pipe.all(AuditLog) if a.action == "transcripts_purged"]
    assert len(audit) == 1 and audit[0].detail == {"count": 1, "reason": "retention"}
    assert await retention.purge_transcripts(pipe.sf, pipe.clock()) == 0  # idempotent


async def test_typed_notes_are_covered_and_only_old_ones_go(pipe: Env) -> None:
    org = await _org(pipe)
    await pipe.tenant_settings(org, transcript_retention_days=10)
    await pipe.note(REP_CHAT, TRANSCRIPT, extraction_json())
    pipe.clock.advance(days=8)
    await pipe.note(REP_CHAT, TRANSCRIPT + " ", extraction_json())
    pipe.clock.advance(days=3)  # the first is 11 days old, the second 3
    assert await retention.purge_transcripts(pipe.sf, pipe.clock()) == 1
    transcripts = sorted((e.transcript is None) for e in await pipe.all(Extraction))
    assert transcripts == [False, True]


async def test_transcript_retention_is_per_tenant(pipe: Env) -> None:
    org = await _org(pipe)
    await pipe.tenant_settings(org, transcript_retention_days=1)
    await _voice_visit(pipe)
    _, _, raw = await pipe.new_tenant("Other Co", "Mallory")
    await pipe.say(3003, f"/start {raw}")
    await pipe.tap(3003, "co:y")
    await pipe.voice(3003, TRANSCRIPT, extraction_json(client="Zed"), data=OGG + b"C")
    pipe.clock.advance(days=5)
    assert await retention.purge_transcripts(pipe.sf, pipe.clock()) == 1  # only the opted-in tenant
    assert sorted(e.transcript is None for e in await pipe.all(Extraction)) == [False, True]


def test_transcript_retention_setting_is_validated() -> None:
    assert TenantSettings().transcript_retention_days is None
    assert TenantSettings.from_json({"transcript_retention_days": 7}).transcript_retention_days == 7
    with pytest.raises(ValidationError):
        TenantSettings.from_json({"transcript_retention_days": 0})


async def test_the_admin_tool_sets_and_validates_transcript_retention(pipe: Env) -> None:
    await _org(pipe)
    ts = await admin.set_retention(
        pipe.sf, "Sharma Jewellers", None, None, set_audio=False, set_image=False,
        transcript=14, set_transcript=True,
    )  # fmt: skip
    assert ts.transcript_retention_days == 14 and ts.audio_retention_days == 90  # others untouched
    with pytest.raises(ValidationError):
        await admin.set_retention(pipe.sf, "Sharma Jewellers", None, None, set_audio=False, set_image=False,
                                  transcript=0, set_transcript=True)  # fmt: skip
