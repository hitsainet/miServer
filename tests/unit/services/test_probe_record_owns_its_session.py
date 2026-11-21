"""A probe record writes ONLY through the session it opened, even with another record in flight.

The unit-shard half of `tests/schema/test_probe_record_isolation.py` (which proves the defect and
the fix on PostgreSQL, in the CI `schema` job). Here the sessions are fakes that note which task
called them, and every commit is a real yield point, so this runs in every shard with no server.

⚠ THE DEFECT: `ProbeEventService` was a process singleton whose repositories `_probe_record` (and
the status dependency) REBOUND to its own session; `record()` re-reads them after each await, so a
second record rebinding in between took over the first's remaining writes. Controls: rebind a
shared instance in `_probe_record`, or in `get_probe_event_service` → red.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest

class _FakeSession:
    """Just what `record()` calls, noting the calling task. `commit` yields to the loop at a gate,
    the way a real commit awaits its round trip."""

    def __init__(self, name: str, gates: dict) -> None:
        self.name = name
        self.gates = gates
        self.calls: list[tuple[object, str]] = []

    def _note(self, what: str) -> None:
        self.calls.append((asyncio.current_task(), what))

    def add_all(self, rows) -> None:
        self._note("add_all")

    def add(self, row) -> None:
        self._note("add")

    def __contains__(self, obj) -> bool:   # `record()` detaches its committed rows
        return False

    async def execute(self, *args, **kwargs):
        self._note("execute")
        await asyncio.sleep(0)
        return SimpleNamespace(rowcount=0, all=lambda: [], scalars=lambda: [])

    async def commit(self) -> None:
        self._note("commit")
        gate = self.gates.pop((asyncio.current_task(), "commit"), None)
        if gate is not None:
            reached, release = gate
            reached.set()
            await asyncio.wait_for(release.wait(), 10)
        await asyncio.sleep(0)

    async def rollback(self) -> None:
        self._note("rollback")


def _verdict():
    return SimpleNamespace(
        probe_id="pr_a", window="all", scored=True, not_scored_reason=None, score=1.0,
        threshold=0.5, fires=True, rung=2, top_positions=[], n_scored_tokens=3,
        provisional=False, threshold_revision=1,
    )


def _context(request_id: str):
    return SimpleNamespace(request_id=request_id, overhead_ms=None, n_passes=1,
                           finish=lambda interrupted=None: [_verdict()], prompt_length=None,
                           last_user_span=None, session=None)


def _inference():
    from miserver.services.inference_service import InferenceService

    class _NoModel(InferenceService):
        _tokenizer = None

        def is_model_loaded(self) -> bool:
            return False

    return _NoModel.__new__(_NoModel)


@pytest.fixture
def world(monkeypatch):
    import miserver.db.base as db_base
    import miserver.services.probe_event_service as pes

    gates: dict = {}
    opened: dict = {}

    @asynccontextmanager
    async def factory():
        session = _FakeSession(f"s{len(opened)}", gates)
        opened[asyncio.current_task()] = session
        yield session

    monkeypatch.setattr(db_base, "async_session_factory", factory)
    monkeypatch.setattr(pes, "_shared_state", None)
    return SimpleNamespace(gates=gates, opened=opened, pes=pes)


def _hold(world, task, what="commit"):
    reached, release = asyncio.Event(), asyncio.Event()
    world.gates[(task, what)] = (reached, release)
    return reached, release


async def test_a_record_keeps_its_own_session_while_another_record_starts(world):
    inference = _inference()
    go = asyncio.Event()

    async def run(request_id):
        await go.wait()
        await inference._probe_record(_context(request_id), [_verdict()], detached=True,
                                      reply_text="ok")

    task_a = asyncio.create_task(run("req-A"))
    a_at_commit, release_a = _hold(world, task_a)
    task_b = asyncio.create_task(run("req-B"))
    b_at_commit, release_b = _hold(world, task_b)
    go.set()
    # A and B both sit in their first commit (the event INSERT) with their own sessions open.
    await asyncio.wait_for(a_at_commit.wait(), 10)
    await asyncio.wait_for(b_at_commit.wait(), 10)
    release_a.set()
    await asyncio.wait_for(task_a, 10)
    release_b.set()
    await asyncio.wait_for(task_b, 10)

    session_a, session_b = world.opened[task_a], world.opened[task_b]
    assert session_a is not session_b
    for own, other, name in ((session_a, task_b, "B"), (session_b, task_a, "A")):
        foreign = [what for task, what in own.calls if task is other]
        assert foreign == [], (
            f"request {name}'s record called {foreign} on another request's session — the "
            "service's repositories were rebound while a record was in flight"
        )
    # The payload, not just the absence: each session carried its own whole record (the event
    # INSERT's commit, the age prune and the cap prune, each committed).
    for session, task in ((session_a, task_a), (session_b, task_b)):
        mine = [what for t, what in session.calls if t is task]
        assert mine.count("execute") == 2 and mine.count("commit") == 3, mine
        assert "rollback" not in mine, mine
    assert world.pes.shared_state().last_not_persisted == []


async def test_the_status_dependency_never_hands_out_a_shared_rebindable_instance(world):
    """Two resolutions of `ProbeEventServiceDep` are two services over two sessions, over ONE
    state. (The old dependency returned the same instance with its repositories reassigned.)"""
    import miserver.api.dependencies as deps

    one = await deps.get_probe_event_service("repo-1", "events-1")
    two = await deps.get_probe_event_service("repo-2", "events-2")
    assert (one.repository, one.events) == ("repo-1", "events-1"), (
        "the first poll's service now points at the second poll's repositories"
    )
    assert (two.repository, two.events) == ("repo-2", "events-2")
    assert one.state is two.state is world.pes.shared_state()


def _emitted(monkeypatch) -> list:
    from miserver.sockets import progress

    sent: list = []
    monkeypatch.setattr(progress.progress_emitter, "emit_probe_results", sent.append)
    return sent


def test_a_service_built_with_no_state_shares_the_process_throttle(world, monkeypatch):
    """⚠ THE SAFE CHOICE IS THE DEFAULT. A production caller that builds `ProbeEventService` and
    forgets `state=` must share the process throttle: a private one resets on every request,
    which is the same as having none. Two services built WITHOUT a state argument, as both
    production sites build them: the second request's emit, inside 0.1 s of the first, is
    dropped and counted on the one shared state. Control: default the constructor to a private
    `ProbeEventState()` → red."""
    from miserver.services.probe_event_service import ProbeEventService

    sent = _emitted(monkeypatch)
    first, second = ProbeEventService("repo-1", "events-1"), ProbeEventService("repo-2", "events-2")
    assert first.state is second.state is world.pes.shared_state()
    first._emit_request("req-1", [{"probe_id": "pr_a"}])
    second._emit_request("req-2", [{"probe_id": "pr_a"}, {"probe_id": "pr_b"}])
    assert [m["request_id"] for m in sent] == ["req-1"], (
        "the second request was not throttled: each service got its own throttle"
    )
    assert world.pes.shared_state().ws_dropped == {"requests": 1, "results": 2}


def test_isolation_is_asked_for_by_name(world, monkeypatch):
    """`isolated()` is the one way to a private state (tests), and it touches nothing shared."""
    from miserver.services.probe_event_service import ProbeEventService

    sent = _emitted(monkeypatch)
    shared = ProbeEventService("repo-1", "events-1")
    private = ProbeEventService.isolated("repo-2", "events-2")
    assert private.state is not shared.state
    shared._emit_request("req-1", [{"probe_id": "pr_a"}])
    private._emit_request("req-2", [{"probe_id": "pr_a"}])
    assert [m["request_id"] for m in sent] == ["req-1", "req-2"], (
        "the isolated service was throttled by the shared state"
    )
    assert world.pes.shared_state().ws_dropped == {"requests": 0, "results": 0}
