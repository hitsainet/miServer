"""Two probe records in flight at once, on PostgreSQL: each writes through ITS OWN session.

⚠ THE DEFECT (found by reading 2026-11-21, proven here the same day). `ProbeEventService` is a
process singleton (the socket throttle and the dropped-event counter live on it), and its two
repositories were REBOUND by every user: `InferenceService._probe_record` set
`service.repository`/`service.events` to its own session's before each `record()`, and so did the
`GET /api/probes/status` dependency. `record()` then read `self.events` again after every await
(`create_many` → `prune_aged` → `prune_to_cap` → the rule and verdict prunes → `_write_turn`), so a
second record (or a status poll) that rebound in between moved the FIRST record's remaining writes
onto the SECOND request's session: its retention ran and committed inside another request's
transaction, its session turn was written there, and when that collided with the other session's
own commit its failure path rolled back a session that was not its own and reported its rows
`not_persisted` although they were committed.

Production runs PostgreSQL with one connection per session, and the records overlap: every record
is a tracked task (`_spawn_probe_record`) that outlives its request's semaphore, the continuous
batching, llama.cpp and batched paths record outside any lock, and the admin UI polls status. So
the interleaving is forced here at the DRIVER — a statement held on its way to the server, which is
exactly what a slow round trip does — never by timing luck.

What is asserted is implementation-agnostic: every statement a record task issues goes through the
session THAT TASK opened, both requests' events are committed, the turn is written, and nothing is
reported `not_persisted`. PostgreSQL only (the production driver and its per-connection
transaction), so it lives under `tests/schema/` and runs in the CI `schema` job: skipped loudly
without a server, REQUIRED under `MISERVER_REQUIRE_PG=1`.
"""

from __future__ import annotations

import asyncio
import uuid
from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import Any, Callable

import pytest
import sqlalchemy as sa
from sqlalchemy import event
from sqlalchemy.dialects.postgresql import asyncpg as pg_asyncpg
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import Session

from tests.schema.test_schema_guards import PG_BASE
from tests.schema.test_weights_origin_migration import _admin, _pg_available
from tests.support.env_flags import flag

#: How long any one forced step may take before the test calls it a hang (it never should).
STEP_S = 20.0


@pytest.fixture(scope="module")
def pg_url():
    if not asyncio.run(_pg_available()):
        message = f"PostgreSQL is not reachable at {PG_BASE}"
        if flag("REQUIRE_PG") == "1":
            pytest.fail(f"{message} and MISERVER_REQUIRE_PG=1 — the record isolation test did NOT run")
        pytest.skip(f"{message} — the record isolation test needs a real server")
    name = f"pesiso_{uuid.uuid4().hex[:8]}"
    asyncio.run(_admin(f'CREATE DATABASE "{name}"'))
    try:
        yield f"{PG_BASE}/{name}"
    finally:
        asyncio.run(_admin(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)'))


class _TrackedSession(Session):
    """The production session class, so the listeners below see only this test's sessions."""


class _Ledger:
    """Which session each task OPENED, and which session each of its statements went through."""

    def __init__(self) -> None:
        self.opened: dict[Any, list[Session]] = {}
        self.used: list[tuple[Any, Session, str]] = []

    def strays(self, task: Any) -> list[str]:
        own = set(map(id, self.opened.get(task, [])))
        return [what for t, s, what in self.used if t is task and id(s) not in own]


class _Gate:
    """Hold a chosen task's chosen statement on its way to the server until released."""

    def __init__(self) -> None:
        self.rules: list[tuple[Any, Callable[[str], bool], asyncio.Event, asyncio.Event]] = []

    def hold(self, task: Any, match: Callable[[str], bool]) -> tuple[asyncio.Event, asyncio.Event]:
        reached, release = asyncio.Event(), asyncio.Event()
        self.rules.append((task, match, reached, release))
        return reached, release

    async def check(self, operation: str) -> None:
        me = asyncio.current_task()
        for i, (task, match, reached, release) in enumerate(self.rules):
            if task is me and match(operation):
                del self.rules[i]
                reached.set()
                await asyncio.wait_for(release.wait(), STEP_S)
                return


@pytest.fixture
async def world(pg_url, monkeypatch):
    """A PostgreSQL database with one armed probe, the production session factory's settings, the
    ledger and the gate wired in, and a cold `ProbeEventService` singleton."""
    import miserver.api.dependencies as deps
    import miserver.db.base as db_base
    import miserver.db.models  # noqa: F401 - every table
    import miserver.services.probe_event_service as pes
    from miserver.db.base import Base
    from miserver.db.models.probe_session import ProbeSession
    from tests.unit.services.test_probe_survives_no_restart import _probe_row

    engine = create_async_engine(pg_url, pool_size=5)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)
    # The production factory's settings (`miserver/db/base.py`), on the tracked class.
    maker = async_sessionmaker(engine, class_=AsyncSession, sync_session_class=_TrackedSession,
                               expire_on_commit=False, autoflush=False)
    async with maker() as s:
        s.add(_probe_row("pr_a", armed=True))
        # Feature 36: one conversation per request, so each record also writes a session turn.
        for sid in ("ps_a", "ps_b"):
            s.add(ProbeSession(id=sid, client_session_id=None, source="header"))
        await s.commit()

    ledger, gate = _Ledger(), _Gate()

    @asynccontextmanager
    async def factory():
        async with maker() as session:
            ledger.opened.setdefault(asyncio.current_task(), []).append(session.sync_session)
            yield session

    def on_execute(state):
        ledger.used.append((asyncio.current_task(), state.session,
                            str(state.statement).split("\n", 1)[0][:60]))

    def on_flush(session, _ctx, _instances):
        for obj in [*session.new, *session.dirty, *session.deleted]:
            ledger.used.append((asyncio.current_task(), session, f"flush {type(obj).__name__}"))

    event.listen(_TrackedSession, "do_orm_execute", on_execute)
    event.listen(_TrackedSession, "before_flush", on_flush)

    real_execute = pg_asyncpg.AsyncAdapt_asyncpg_cursor._prepare_and_execute
    real_many = pg_asyncpg.AsyncAdapt_asyncpg_cursor._executemany

    async def gated_execute(self, operation, parameters):
        await gate.check(operation)
        return await real_execute(self, operation, parameters)

    async def gated_many(self, operation, seq):
        await gate.check(operation)
        return await real_many(self, operation, seq)

    monkeypatch.setattr(pg_asyncpg.AsyncAdapt_asyncpg_cursor, "_prepare_and_execute", gated_execute)
    monkeypatch.setattr(pg_asyncpg.AsyncAdapt_asyncpg_cursor, "_executemany", gated_many)
    monkeypatch.setattr(db_base, "async_session_factory", factory)
    # A cold process: no state shared with an earlier test.
    monkeypatch.setattr(pes, "_shared_state", None)
    try:
        yield SimpleNamespace(maker=maker, ledger=ledger, gate=gate, deps=deps, pes=pes)
    finally:
        event.remove(_TrackedSession, "do_orm_execute", on_execute)
        event.remove(_TrackedSession, "before_flush", on_flush)
        await engine.dispose()


def _verdict(window: str = "all"):
    return SimpleNamespace(
        probe_id="pr_a", window=window, scored=True, not_scored_reason=None, score=1.0,
        threshold=0.5, fires=True, rung=2, top_positions=[], n_scored_tokens=3,
        provisional=False, threshold_revision=1,
    )


def _session_ref(session_id: str):
    from miserver.api.probe_request_meta import SessionRef

    # `chat=False`: a turn with no continuation hash and no transcript; the turn row is the point.
    return SessionRef(id=session_id, client_session_id=None, source="header", turn_index=1,
                      prefix_hash="p" * 64, weak_hash="w" * 64, chat=False)


def _context(request_id: str, session_id: str):
    return SimpleNamespace(request_id=request_id, overhead_ms=None, n_passes=1,
                           finish=lambda interrupted=None: [_verdict()], prompt_length=None,
                           last_user_span=None, session=_session_ref(session_id))


def _inference():
    from miserver.services.inference_service import InferenceService

    class _NoModel(InferenceService):
        """The real `_probe_record`, with no model resident (no context window to decode)."""

        _tokenizer = None

        def is_model_loaded(self) -> bool:
            return False

    return _NoModel.__new__(_NoModel)


async def _stored(maker) -> dict[str, tuple[int, int]]:
    """request id → (events committed, session turns committed)."""
    from miserver.db.models.probe import ProbeEvent
    from miserver.db.models.probe_session import ProbeSessionTurn

    async with maker() as s:
        events = dict((await s.execute(
            sa.select(ProbeEvent.request_id, sa.func.count()).group_by(ProbeEvent.request_id)
        )).all())
        turns = dict((await s.execute(
            sa.select(ProbeSessionTurn.request_id, sa.func.count())
            .group_by(ProbeSessionTurn.request_id)
        )).all())
    return {r: (events.get(r, 0), turns.get(r, 0)) for r in {*events, *turns}}


def _is(prefix: str) -> Callable[[str], bool]:
    return lambda sql: sql.lstrip().upper().startswith(prefix)


INSERT_EVENT = _is("INSERT INTO PROBE_EVENTS")


async def _record(inference, request_id: str, session_id: str) -> None:
    await inference._probe_record(_context(request_id, session_id), [_verdict()], detached=True,
                                  reply_text="ok")


def _failures(caplog) -> list[str]:
    return [r.getMessage() for r in caplog.records
            if r.getMessage() in ("probe_event_persist_failed", "probe_session_write_failed")
            or r.getMessage().startswith("probe_session_write_failed")]


async def _held_at(gate, coro, match, *, name: str):
    """Start `coro` as a task, held at its first statement matching `match`."""
    go = asyncio.Event()

    async def run():
        await go.wait()
        return await coro

    task = asyncio.create_task(run(), name=name)
    reached, release = gate.hold(task, match)
    go.set()
    await asyncio.wait_for(reached.wait(), STEP_S)
    return task, release


async def test_two_records_in_flight_each_write_through_their_own_session(world, caplog):
    """A is held while its event INSERT is on the wire; B starts and is held at its own INSERT;
    A is released and runs to the end; then B.

    Before the fix (measured 2026-11-21 on PostgreSQL 16): B's `_probe_record` rebound the
    singleton to B's session, so A's retention DELETE ran inside B's open transaction and A's
    commit collided with B's (`IllegalStateChangeError`). A's failure path rolled back B's session
    and reported A's COMMITTED event `not_persisted`; B then found its transaction closed
    (`ResourceClosedError`). Stored: A (1 event, 0 turns), B (0 events, 0 turns).
    """
    inference = _inference()
    task_a, release_a = await _held_at(world.gate, _record(inference, "req-A", "ps_a"),
                                       INSERT_EVENT, name="A")
    task_b, release_b = await _held_at(world.gate, _record(inference, "req-B", "ps_b"),
                                       INSERT_EVENT, name="B")
    release_a.set()
    await asyncio.wait_for(task_a, STEP_S)
    release_b.set()
    await asyncio.wait_for(task_b, STEP_S)

    assert world.ledger.strays(task_a) == [], (
        "request A's record issued statements through a session it did not open — the "
        "singleton was rebound to request B's session while A was in flight"
    )
    assert world.ledger.strays(task_b) == []
    assert await _stored(world.maker) == {"req-A": (1, 1), "req-B": (1, 1)}, (
        "an event or a session turn was lost: one record wrote through, committed or rolled "
        "back the other's session"
    )
    assert world.pes.shared_state().last_not_persisted == [], (
        "a committed event was reported not_persisted"
    )
    assert _failures(caplog) == []


async def test_a_status_poll_during_a_record_does_not_take_over_the_records_session(
    world, caplog
):
    """`GET /api/probes/status` resolves `ProbeEventServiceDep`. It used to REBIND the singleton to
    the poll's session on every call, so a record in flight then pruned, committed and wrote its
    turn through the poll's session. The poll is driven through the real dependency."""
    from miserver.db.repositories.probe_repository import ProbeEventRepository, ProbeRepository

    inference = _inference()
    task_a, release_a = await _held_at(world.gate, _record(inference, "req-A", "ps_a"),
                                       INSERT_EVENT, name="A")

    import miserver.db.base as db_base

    async def poll():
        async with db_base.async_session_factory() as session:
            service = await world.deps.get_probe_event_service(
                ProbeRepository(session), ProbeEventRepository(session))
            # Held at its first statement, so the poll's session is open while A resumes.
            return await service.status()

    task_p, release_p = await _held_at(world.gate, poll(), _is("SELECT"), name="poll")
    release_a.set()
    await asyncio.wait_for(task_a, STEP_S)
    release_p.set()
    status = await asyncio.wait_for(task_p, STEP_S)

    assert world.ledger.strays(task_a) == [], (
        "the record's statements went through the status poll's session"
    )
    assert world.ledger.strays(task_p) == []
    assert await _stored(world.maker) == {"req-A": (1, 1)}
    assert status["events_recorded"] == 1, "the poll ran after A committed and must count it"
    assert world.pes.shared_state().last_not_persisted == []
    assert _failures(caplog) == []


async def test_the_shared_state_still_spans_requests(world):
    """The other half: the throttle and the overhead must still be PROCESS-wide. Two records and a
    status poll, each over its own session, see one state — a fresh state per request would be the
    same as having no throttle (039 TD39-28)."""
    from miserver.db.repositories.probe_repository import ProbeEventRepository, ProbeRepository

    inference = _inference()
    first = _context("req-1", "ps_a")
    first.overhead_ms = 7.5
    await inference._probe_record(first, [_verdict()], detached=True, reply_text="ok")
    await _record(inference, "req-2", "ps_b")   # within the 0.1 s throttle of the first
    async with world.maker() as session:
        service = await world.deps.get_probe_event_service(
            ProbeRepository(session), ProbeEventRepository(session))
        status = await service.status()
    assert status["socket_events_dropped"] == {"requests": 1, "results": 1}, (
        "the second request's emit was not throttled: each record got its own throttle"
    )
    assert status["last_request_overhead_ms"] == 7.5, (
        "the status poll cannot see the overhead a record noted: the state is not shared"
    )
