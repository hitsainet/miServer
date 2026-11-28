"""Operator code AFTER the response, through the LIVE app and a real chat (039 S3; D41; D44).

D41 is an honesty requirement, proven here on the real chat route with a generated behaviour armed
and a generated rule set carrying a `code_rule` attached through the routes:

1. the live response NEVER waits on the runner: it is delivered while the worker is still holding
   its reply, and every code-rule expression in it is pending (`rule_worker_unavailable`), never
   `fires: false`;
2. the result LANDS AFTERWARDS: the same stored `rule_signal_results` rows (same `result_id`) are
   revised once the worker answers, the events list shows the revision, and one
   `probe:rule_results` socket message carries it — a count, never a value;
3. what cannot be decided stays pending: the worker unreachable. (An expression that needs a code
   rule's POSITIONS is refused at import since D65 — `test_rule_code_followup_r1.py`.)

The timing proof uses a scripted worker that holds its reply (`tests/support/stub_runner.py`); the
values are proven with the REAL worker on the published snapshot (skips loudly without it).
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from typing import Any
from unittest.mock import patch

import pytest
from sqlalchemy import select

from miserver.core.config import settings
from miserver.db.models.rule_set import RuleSignalResult
from miserver.rules import runner_contract as rc
from tests.support import behaviour_factory as bf
from tests.support import capture_pre038_goldens as cap
from tests.support import rule_runner as rr
from tests.support import rule_set_factory as rf
from tests.support.stub_runner import StubRunner

pytestmark = pytest.mark.asyncio

CASES = json.loads(rc.CASES_PATH.read_text(encoding="utf-8"))
TEXT = f"marker ZQ-1234 and {rf.LUHN_OK} w1 w2"
BODY = {**cap.BODY, "messages": [{"role": "user", "content": TEXT}]}
CODE_NAMES = {"operator", "op_and_token"}


def code_document(doc_id: str = "count_tokens") -> dict[str, Any]:
    """A published reference code-rule document from the vendored vectors."""
    return CASES["documents"][doc_id]


def rule_set_with_code(key: str, code: dict[str, Any]) -> dict[str, Any]:
    doc = rf.document(key)
    doc["signals"].append({
        "id": "op", "kind": "code_rule", "code_sha256": rc.code_sha256(code),
        "runner_contract": rc.contract_format(), "function": "f", "params": {},
        "outputs": [{"name": "hits", "type": "int"}]})
    doc["expressions"]["operator"] = {"signal": "op"}
    doc["expressions"]["op_and_token"] = {"all": [{"signal": "op"}, {"signal": "tok"}]}
    # No `within_n` over `op`: a code rule reports no positions, so that is refused (D65).
    return doc


def envelope(doc: dict[str, Any], code: dict[str, Any]) -> bytes:
    return json.dumps({"rule_set": doc, "code_modules": {
        rc.code_sha256(code): rc.canonical_json(code).decode("utf-8")}}).encode("utf-8")


async def _arm(h, key):
    async with h.factory() as session:
        for i in range(2):
            await bf.import_probe(session, bf.definition(
                f"{key}-gen-{i}", key=key, layer=i % 2, scale=0.13 + 0.2 * i))
    r = h.client.post(f"/api/probes/behaviours/{key}/arm", json={})
    assert r.status_code == 200, r.text


def _attach(h, key, raw: bytes) -> str:
    r = h.client.post("/api/rule-sets/import", content=raw,
                      headers={"content-type": "application/json"})
    assert r.status_code == 200, r.text
    rule_set_id = r.json()["data"]["id"]
    r = h.client.put(f"/api/probes/behaviours/{key}/rule-signals",
                     json={"rule_set_ids": [rule_set_id]})
    assert r.status_code == 200, r.text
    return rule_set_id


async def _stored(h, request_id) -> list[RuleSignalResult]:
    async with h.factory() as session:
        return list((await session.execute(
            select(RuleSignalResult).where(RuleSignalResult.request_id == request_id)
            .order_by(RuleSignalResult.id))).scalars().all())


@contextlib.asynccontextmanager
async def _no_lifespan(app):
    yield


class _OneLoopMutex(asyncio.Lock):
    """The harness connection's SQLAlchemy `_execute_mutex`, refusing a second event loop.

    Review R2 (follow-ups): the harness has ONE aiosqlite connection, and SQLAlchemy serialises
    it with an `asyncio.Lock`. Awaited from the pytest loop while the live loop holds it, its
    release wakes the waiter with a non-threadsafe `set_result`, which is lost: the pytest loop
    sleeps in epoll for good (the 20-minute hang of the 039 follow-up runs). Timing decides
    whether it hangs; this makes ANY database use from the wrong loop fail at once instead."""

    MESSAGE = ("database used from the pytest loop while the live loop runs: run it on the live "
               "loop (`client.portal.call(...)`), or a lost wake-up can hang the test")

    def __init__(self, loop: asyncio.AbstractEventLoop) -> None:
        super().__init__()
        self.owner_loop = loop
        #: Recorded as well as raised: a caller that swallows the error (best-effort recording)
        #: must not hide it — `_live_loop` fails on exit when this is not empty.
        self.violations = 0

    async def acquire(self) -> bool:
        if asyncio.get_running_loop() is not self.owner_loop:
            self.violations += 1
            raise AssertionError(self.MESSAGE)
        return await super().acquire()


class _SharedConnectionGuard:
    """Fails the test when one session's ROLLBACK discards another session's uncommitted writes.

    The harness database is ``sqlite+aiosqlite:///:memory:``, so SQLAlchemy gives it a
    `StaticPool`: EVERY session — the app's, an after-response job's, the test's own reads — runs
    on the SAME DBAPI connection. A transaction is the connection's, not the session's, so a
    session that closes (an implicit ROLLBACK) while another session has executed an
    INSERT/UPDATE/DELETE and not yet committed rolls that work back; the writer then commits
    nothing and reports success. Production (PostgreSQL, one connection per session) cannot do
    this; the harness can, and it did: CI run 44894198783 (2026-11-21) lost the after-response
    revision of the hostile cases to a 50 ms poll of `_stored` that landed between the job's
    `UPDATE rule_signal_results` and its COMMIT, and the stored rows read
    `rule_worker_unavailable` although the worker had answered `rule_resource_limit`
    (`0xcc/reviews/rule_runner_resource_limit_2026-11-21.md`). Timing decides whether it happens;
    this makes it a named failure instead of a wrong value read 60 s later."""

    MESSAGE = ("a session's ROLLBACK discarded another session's uncommitted writes on the "
               "harness's one shared connection: read only after the after-response work has "
               "finished (`_rows_after_jobs`)")
    WRITES = ("INSERT", "UPDATE", "DELETE")

    def __init__(self, engine: Any) -> None:
        from sqlalchemy import event

        self._event = event
        self.engine = engine.sync_engine
        #: `id(Connection)` -> the Connection, for every one with executed, uncommitted DML.
        self.writing: dict[int, Any] = {}
        self.violations: list[str] = []
        self._hooks = (("after_cursor_execute", self._executed), ("commit", self._ended),
                       ("rollback", self._rolled_back))

    def __enter__(self) -> "_SharedConnectionGuard":
        for name, fn in self._hooks:
            self._event.listen(self.engine, name, fn)
        return self

    def __exit__(self, *exc: Any) -> None:
        for name, fn in self._hooks:
            self._event.remove(self.engine, name, fn)

    def _executed(self, conn, _cursor, statement, _params, _context, _executemany) -> None:
        if statement.lstrip()[:6].upper() in self.WRITES:
            self.writing[id(conn)] = conn

    def _ended(self, conn) -> None:
        self.writing.pop(id(conn), None)

    def _rolled_back(self, conn) -> None:
        others = [c for key, c in self.writing.items() if key != id(conn)]
        if others:
            self.violations.append(f"{len(others)} writer(s) rolled back by another session")
        self._ended(conn)


@contextlib.contextmanager
def _live_loop(h):
    """Keep ONE server loop alive across requests (as uvicorn does), so a task the request
    spawned and did not await keeps running after the response. The app's own startup is not
    run (the harness already built everything it needs). While it runs, the database belongs to
    that loop alone (`_OneLoopMutex`), and no session may roll back another's writes on the
    shared connection (`_SharedConnectionGuard`)."""
    h.client.app.router.lifespan_context = _no_lifespan
    with h.client as client, _SharedConnectionGuard(h.engine) as lost:
        adapted = h.engine.sync_engine.pool.connection.dbapi_connection
        original = adapted._execute_mutex
        guard = _OneLoopMutex(client.portal.call(_running_loop))
        adapted._execute_mutex = guard
        try:
            yield client
        finally:
            adapted._execute_mutex = original
        assert not guard.violations, _OneLoopMutex.MESSAGE
        assert not lost.violations, (_SharedConnectionGuard.MESSAGE, lost.violations)


async def _running_loop() -> asyncio.AbstractEventLoop:
    return asyncio.get_running_loop()


def _code(entries):
    return [e for e in entries if e.get("kind") == "rule" and e["signal"] in CODE_NAMES]


def _pending(entry) -> bool:
    return (entry["outcome"], entry["status"], entry["reason"], entry["fires"], entry["count"]) \
        == ("skipped", "unavailable", "rule_worker_unavailable", None, 0)


async def _after_jobs(timeout: float) -> int:
    """Wait for every probe record, then every after-response job, in flight on THIS loop.
    Returns how many are still running after `timeout` seconds."""
    from miserver.rules import code_followup
    from miserver.services.inference_service import InferenceService

    left = await InferenceService.drain_probe_records(timeout)
    return left + await code_followup.drain(timeout)


def _rows_after_jobs(client, h, request_id, timeout=20.0):
    """The request's stored rule rows, read ONCE, after its after-response work has finished.

    It used to POLL `_stored` every 50 ms until a predicate held. Each poll is a session on the
    harness's one shared connection, and its close is a ROLLBACK: a poll landing between the
    job's UPDATE and its COMMIT discarded the revision, the rows stayed pending, and the test
    read `rule_worker_unavailable` after its timeout (CI run 44894198783;
    `_SharedConnectionGuard`). Nothing reads the database here until the work has ended."""
    left = client.portal.call(_after_jobs, timeout)
    assert left == 0, f"{left} probe record(s) or after-response job(s) still running after {timeout} s"
    return client.portal.call(_stored, h, request_id)


def _by_signal(rows):
    return {(r.signal, r.window): r for r in rows}


async def test_the_live_response_never_waits_and_the_result_lands_afterwards(monkeypatch):
    monkeypatch.setattr(settings, "PROBE_VERDICT_HEADER_MAX_BYTES", 65536)
    code = code_document()
    async with cap.Harness() as h:
        key = bf.random_key()
        await _arm(h, key)
        rule_set_id = _attach(h, key, envelope(rule_set_with_code(key, code), code))
        emitted: list[dict] = []
        from miserver.sockets.progress import progress_emitter

        with StubRunner(hold=True) as stub, \
                patch.object(progress_emitter, "emit_probe_rule_results", emitted.append), \
                _live_loop(h) as client:
            monkeypatch.setattr(settings, "RULE_RUNNER_SOCKET", str(stub.socket))
            h._fresh_socket_throttle()
            response = client.post("/v1/chat/completions", json=BODY)
            assert response.status_code == 200, response.text
            request_id = response.json()["id"]
            live = _code(response.json()["millm_probe_verdicts"])
            # (1) Delivered BEFORE the worker answered; every code expression pending, none false.
            assert live and all(_pending(e) for e in live), live
            assert stub.call_seen.wait(20), "the after-response job never reached the worker"
            assert not stub.release.is_set()
            held = [r for r in client.portal.call(_stored, h, request_id) if r.signal in CODE_NAMES]
            assert {r.result_id for r in held} == {e["result_id"] for e in live}
            assert all((r.outcome, r.reason, r.fires) == ("skipped", "rule_worker_unavailable",
                                                          None) for r in held)
            # (2) The worker answers; the SAME rows are revised in place.
            stub.release.set()
            rows = _rows_after_jobs(client, h, request_id)
            by = _by_signal(r for r in rows if r.signal in CODE_NAMES)
            windows = {w for (s, w) in by if s == "operator"}
            assert windows, by
            for window in windows:
                op, both = by[("operator", window)], by[("op_and_token", window)]
                assert (op.outcome, op.fires, op.count, op.reason, op.status) == (
                    "completed", True, 2, None, "final")
                assert op.result["fires"] is True and op.result["count"] == 2
                assert op.result["status"] == "final" and op.result["reason"] is None
                # `all` over the code rule and a declarative token: the token's one position
                # plus the code rule's count; fires only where the token fired.
                tok_fired = both.count > 0
                assert both.outcome == "completed" and both.fires is tok_fired
                if tok_fired:
                    assert both.count == 1 + 2
            assert {r.result_id for r in rows if r.signal in CODE_NAMES} == {
                e["result_id"] for e in live}
            # The events list reads the revised rows.
            events = client.get("/api/probes/events", params={"limit": 100}).json()["data"]
            ops = [e for e in events if e.get("kind") == "rule" and e["signal"] == "operator"]
            assert ops and all(e["outcome"] == "completed" and e["count"] == 2 for e in ops)
            # One socket message for the revision, ids only from this request, no text.
            assert len(emitted) == 1
            message = emitted[0]
            assert message["request_id"] == request_id
            assert {e["result_id"] for e in message["results"]} <= {
                e["result_id"] for e in live}
            assert "ZQ-1234" not in json.dumps(message)
        calls = stub.calls()
        # Called once per (code rule, window); the text never recorded here, only its length.
        assert len(calls) == len(windows)
        assert {c["code_sha256"] for c in calls} == {rc.code_sha256(code)}
        assert rule_set_id


async def test_with_no_worker_the_chat_serves_and_code_rules_stay_pending(monkeypatch):
    """FR-39.26: the worker stopped → the chat route serves, the probes are unchanged, and the
    code rules report `rule_worker_unavailable` — never a guessed value. Also the negative control
    for in-process evaluation: with no worker, NOTHING may resolve them."""
    code = code_document()
    async with cap.Harness() as h:
        key = bf.random_key()
        await _arm(h, key)
        before = [e for e in h.post_chat(BODY, {}).json()["millm_probe_verdicts"]
                  if e.get("kind", "probe") == "probe"]
        _attach(h, key, envelope(rule_set_with_code(key, code), code))
        emitted: list[dict] = []
        from miserver.sockets.progress import progress_emitter

        with patch.object(progress_emitter, "emit_probe_rule_results", emitted.append), \
                _live_loop(h) as client:
            monkeypatch.setattr(settings, "RULE_RUNNER_SOCKET", "/nonexistent/millm-runner.sock")
            h._fresh_socket_throttle()
            response = client.post("/v1/chat/completions", json=BODY)
            assert response.status_code == 200
            entries = response.json()["millm_probe_verdicts"]
            same = ("probe_id", "window", "score", "threshold", "verdict", "outcome", "reason")
            assert [{k: e.get(k) for k in same} for e in entries
                    if e.get("kind", "probe") == "probe"] == [
                {k: e.get(k) for k in same} for e in before]
            live = _code(entries)
            assert live and all(_pending(e) for e in live)
            from miserver.rules import code_followup

            client.portal.call(code_followup.drain, 20.0)
            rows = client.portal.call(_stored, h, response.json()["id"])
            assert all((r.outcome, r.reason) == ("skipped", "rule_worker_unavailable")
                       for r in rows if r.signal in CODE_NAMES)
            assert emitted == []


async def test_the_real_worker_scores_the_code_rule_after_the_response(monkeypatch):
    """The values, from the published guest: `count_tokens` counts whitespace tokens."""
    bundle = rr.bundle_dir()
    monkeypatch.setattr(settings, "PROBE_VERDICT_HEADER_MAX_BYTES", 65536)
    code = code_document("count_tokens")
    async with cap.Harness() as h:
        key = bf.random_key()
        await _arm(h, key)
        _attach(h, key, envelope(rule_set_with_code(key, code), code))
        with rr.RunnerProcess(bundle) as runner, _live_loop(h) as client:
            monkeypatch.setattr(settings, "RULE_RUNNER_SOCKET", str(runner.socket))
            h._fresh_socket_throttle()
            response = client.post("/v1/chat/completions", json=BODY)
            request_id = response.json()["id"]
            assert all(_pending(e) for e in _code(response.json()["millm_probe_verdicts"]))
            rows = _rows_after_jobs(client, h, request_id)
            by = _by_signal(r for r in rows if r.signal == "operator")
            assert by, rows
            for (_s, window), row in by.items():
                assert row.outcome == "completed" and row.fires is (row.count > 0)
            # The prompt window is the user's one message: its whitespace tokens, counted by the
            # guest; `all` (prompt + reply) holds at least as many.
            assert by[("operator", "prompt")].count == len(TEXT.split())
            assert by[("operator", "all")].count >= by[("operator", "prompt")].count


async def test_the_per_request_call_cap_leaves_the_rest_pending(monkeypatch, caplog):
    """`RULE_CODE_CALLS_PER_REQUEST`: past it no call is made; those results stay pending, and
    the cap is logged (FTASKS 17.2)."""
    import logging

    caplog.set_level(logging.WARNING)
    monkeypatch.setattr(settings, "PROBE_VERDICT_HEADER_MAX_BYTES", 65536)
    monkeypatch.setattr(settings, "RULE_CODE_CALLS_PER_REQUEST", 1)
    code = code_document()
    async with cap.Harness() as h:
        key = bf.random_key()
        await _arm(h, key)
        _attach(h, key, envelope(rule_set_with_code(key, code), code))
        with StubRunner() as stub, _live_loop(h) as client:
            monkeypatch.setattr(settings, "RULE_RUNNER_SOCKET", str(stub.socket))
            h._fresh_socket_throttle()
            response = client.post("/v1/chat/completions", json=BODY)
            from miserver.rules import code_followup

            client.portal.call(code_followup.drain, 20.0)
            rows = [r for r in client.portal.call(_stored, h, response.json()["id"])
                    if r.signal == "operator"]
        assert len(stub.calls()) == 1
        assert len(rows) >= 2
        assert sum(r.outcome == "completed" for r in rows) == 1
        assert all((r.outcome, r.reason) == ("skipped", "rule_worker_unavailable")
                   for r in rows if r.outcome != "completed")
        assert "rule_code_calls_capped" in caplog.text


# ── CI run 44894198783 (2026-11-21): a read during the revision rolled it back ─────────────────


def _held_revisions(h, rolled_back: "asyncio.Event"):
    """A session factory whose sessions, when they COMMIT a revision of `rule_signal_results`
    rows, first flush it and then hold the transaction open until another session has rolled
    back on the shared connection (at most 2 s). That is the CI run's interleaving made certain:
    a reader that touches the database while the job runs lands inside the window."""
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

    class _Held(AsyncSession):
        async def commit(self) -> None:
            if any(isinstance(o, RuleSignalResult) for o in self.dirty):
                await self.flush()
                rolled_back.clear()
                with contextlib.suppress(asyncio.TimeoutError):
                    await asyncio.wait_for(rolled_back.wait(), 2.0)
            await super().commit()

    return async_sessionmaker(h.engine, class_=_Held, expire_on_commit=False)


async def test_a_read_while_the_revision_is_in_flight_does_not_lose_it(monkeypatch):
    """The worker answered every call (`rule_code_failed … reason=rule_resource_limit`, logged)
    and the stored rows still read `rule_worker_unavailable`: the test's own poll closed its
    session between the job's UPDATE and COMMIT, which on the harness's one shared connection is
    a ROLLBACK of the job's work. Here the window is held open on purpose: `_rows_after_jobs`
    reads only after the job ends, so the revision lands; the old 50 ms poll fails this test."""
    import miserver.db.base as db_base
    from sqlalchemy import event

    monkeypatch.setattr(settings, "PROBE_VERDICT_HEADER_MAX_BYTES", 65536)
    code = code_document()
    async with cap.Harness() as h:
        key = bf.random_key()
        await _arm(h, key)
        _attach(h, key, envelope(rule_set_with_code(key, code), code))
        with StubRunner() as stub, _live_loop(h) as client:
            rolled_back = client.portal.call(_new_event)
            listener = lambda _conn: rolled_back.set()  # noqa: E731 - on the live loop's thread
            event.listen(h.engine.sync_engine, "rollback", listener)
            # Scoped to the harness (2026-11-26): a `monkeypatch.setattr` here was undone at the TEST's end, after
            # the harness had restored the factory, so it put the harness's SQLite factory back for good and every
            # later test in the process saw it (found by DEBT_TASKS 6.3's binding test).
            with patch.object(db_base, "async_session_factory", _held_revisions(h, rolled_back)):
                monkeypatch.setattr(settings, "RULE_RUNNER_SOCKET", str(stub.socket))
                h._fresh_socket_throttle()
                try:
                    response = client.post("/v1/chat/completions", json=BODY)
                    assert response.status_code == 200, response.text
                    rows = _rows_after_jobs(client, h, response.json()["id"])
                finally:
                    event.remove(h.engine.sync_engine, "rollback", listener)
        ops = [r for r in rows if r.signal == "operator"]
        assert ops and stub.calls()
        assert all((r.outcome, r.reason, r.count, r.status) == ("completed", None, 2, "final")
                   for r in ops), [(r.window, r.outcome, r.reason, r.status) for r in ops]


async def _new_event() -> asyncio.Event:
    return asyncio.Event()


async def test_the_live_loop_fails_when_a_session_rolls_back_another_sessions_writes():
    """The guard is WIRED into `_live_loop`: one session executes an UPDATE and has not
    committed, a second session reads and closes (ROLLBACK on the shared connection), the first
    commits. The write is gone — proven by reading it back — and `_live_loop` fails on exit."""
    from sqlalchemy import text

    async def interleave(h) -> tuple[str, str]:
        async with h.factory() as writer:
            await writer.execute(text("CREATE TABLE IF NOT EXISTS guard_probe (v TEXT)"))
            await writer.commit()
            await writer.execute(text("INSERT INTO guard_probe (v) VALUES ('pending')"))
            await writer.commit()
            await writer.execute(text("UPDATE guard_probe SET v = 'final'"))
            async with h.factory() as reader:
                seen = (await reader.execute(text("SELECT v FROM guard_probe"))).scalar_one()
            await writer.commit()
        async with h.factory() as after:
            kept = (await after.execute(text("SELECT v FROM guard_probe"))).scalar_one()
        return seen, kept

    async with cap.Harness() as h:
        with pytest.raises(AssertionError, match="discarded another session's uncommitted"):
            with _live_loop(h) as client:
                seen, kept = client.portal.call(interleave, h)
        # The reader saw the uncommitted value, and its close rolled it back: the commit kept
        # nothing. This is the mechanism, not a model of it.
        assert (seen, kept) == ("final", "pending")
