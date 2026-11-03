"""Notes stranded by a crash are given up on, audited, and the rep is told to resend."""

from __future__ import annotations

import asyncio
from datetime import timedelta

import pytest

from app.models import AuditLog, Message, User
from app.models.enums import ProcessingState
from app.pack import load_pack
from app.services.recovery import recover_stranded
from tests.conftest import (
    OGG,
    REP_CHAT,
    TRANSCRIPT,
    Env,
    button_data,
    extraction_json,
    to,
)

WAIT = timedelta(minutes=10)


async def _recover(p: Env) -> list:  # type: ignore[type-arg]
    return await recover_stranded(p.sf, load_pack(), p.clock(), older_than=WAIT)


async def _crash_mid_note(p: Env) -> None:
    """Simulate the process dying after the note was accepted but before it finished."""

    async def die(*args: object, **kwargs: object) -> bytes:
        raise asyncio.CancelledError  # a BaseException: nothing in the pipeline catches it

    p.fake.download = die  # type: ignore[method-assign]
    p.fake.files["f1"] = OGG
    with pytest.raises(asyncio.CancelledError):
        await p.voice(REP_CHAT, TRANSCRIPT, extraction_json())


async def _org(p: Env):  # type: ignore[no-untyped-def]
    org = await p.org()
    await p.add_client(org, "Rajendra Jewellers")
    return org


async def test_a_crashed_note_stays_processing_until_the_sweep_recovers_it(pipe: Env) -> None:
    await _org(pipe)
    await _crash_mid_note(pipe)
    (msg,) = await pipe.all(Message)
    assert msg.processing_state is ProcessingState.PROCESSING
    assert await _recover(pipe) == []  # not old enough yet: it might still be running
    pipe.clock.advance(minutes=10)
    assert await _recover(pipe) == []  # exactly at the limit: still not recovered
    pipe.clock.advance(seconds=1)
    msgs = await _recover(pipe)
    assert [(m.chat_id, "dobara bhejein" in m.text) for m in msgs] == [(REP_CHAT, True)]
    (msg,) = await pipe.all(Message)
    assert msg.processing_state is ProcessingState.FAILED
    assert [a.action for a in await pipe.all(AuditLog) if a.action == "note_stranded"] == [
        "note_stranded"
    ]


async def test_the_sweep_is_idempotent(pipe: Env) -> None:
    await _org(pipe)
    await _crash_mid_note(pipe)
    pipe.clock.advance(minutes=11)
    assert len(await _recover(pipe)) == 1
    assert await _recover(pipe) == []  # nothing left to recover, no duplicate message


async def test_the_rep_can_simply_resend_afterwards(pipe: Env) -> None:
    await _org(pipe)
    await _crash_mid_note(pipe)
    pipe.clock.advance(minutes=11)
    await _recover(pipe)
    del pipe.fake.download  # restore the real method
    out = await pipe.voice(REP_CHAT, TRANSCRIPT, extraction_json())
    msg = to(REP_CHAT, out)[0]
    assert msg.text.endswith("Sahi hai?")  # a fresh update id, processed normally
    out = await pipe.tap(REP_CHAT, button_data(msg, "Haan"))
    assert "Save ho gaya" in to(REP_CHAT, out)[0].text


async def test_finished_and_recent_notes_are_left_alone(pipe: Env) -> None:
    await _org(pipe)
    await pipe.voice(REP_CHAT, TRANSCRIPT, extraction_json())  # completed: state done
    pipe.clock.advance(hours=5)
    assert await _recover(pipe) == []
    (msg,) = await pipe.all(Message)
    assert msg.processing_state is ProcessingState.DONE


async def test_an_erased_or_unbound_worker_is_not_messaged(pipe: Env) -> None:
    org = await _org(pipe)
    await _crash_mid_note(pipe)
    async with pipe.sf() as s, s.begin():
        (await s.get(User, org.rep_id)).telegram_chat_id = None  # type: ignore[union-attr]
    pipe.clock.advance(minutes=11)
    assert await _recover(pipe) == []
    assert (await pipe.all(Message))[
        0
    ].processing_state is ProcessingState.FAILED  # still cleaned up


async def test_each_tenants_stranded_notes_are_recovered_separately(pipe: Env) -> None:
    await _org(pipe)
    await _crash_mid_note(pipe)
    _, _, raw = await pipe.new_tenant("Other Co", "Mallory")
    await pipe.say(3003, f"/start {raw}")
    await pipe.tap(3003, "co:y")
    pipe.clock.advance(minutes=11)
    msgs = await _recover(pipe)
    assert [m.chat_id for m in msgs] == [REP_CHAT]  # Other Co has nothing stranded
