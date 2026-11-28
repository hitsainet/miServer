"""A probe record's failure paths AFTER its rows are committed, on PostgreSQL.

Found in the 2026-11-21 rebind review (`0xcc/reviews/probe_event_service_rebind_2026-11-21.md`).
Both are pre-existing and both are invisible on SQLite, which is why they live here:

* **A failed retention pass relabelled committed events.** One `try` covered the event write and
  the prunes after it, so a prune that failed reported every committed result `not_persisted`.
* **A rollback expires the committed rows, and the publish then read them.** On the production
  session settings a `rollback()` expires every object in the session; `_publish` reads `id` and
  `created_at` off the persisted rows, and an expired attribute on an `AsyncSession` raises
  `MissingGreenlet`. So after a failed session-turn write (which rolls back) the record RAISED out
  of `record()`: the request's events never reached the socket or telemetry, and `_probe_record`'s
  handler skipped the after-response code-rule jobs, leaving their rule rows pending for good.

Runs in the CI `schema` job (fixtures from `test_probe_record_isolation.py`).
"""

from __future__ import annotations

import pytest

from tests.schema.test_probe_record_isolation import (  # noqa: F401 - fixtures
    _context,
    _inference,
    _stored,
    _verdict,
    pg_url,
    world,
)


@pytest.fixture
def emitted(monkeypatch):
    from miserver.services.probe_event_service import ProbeEventService

    sent: list = []
    monkeypatch.setattr(ProbeEventService, "_emit_request",
                        lambda self, request_id, payloads: sent.append((request_id, payloads)))
    return sent


@pytest.fixture
def followups(monkeypatch):
    from miserver.rules import code_followup

    calls: list = []
    monkeypatch.setattr(code_followup, "spawn",
                        lambda job, results, live: calls.append(("rules", len(list(results)))))
    monkeypatch.setattr(code_followup, "spawn_verdicts",
                        lambda job, pending, live: calls.append(("verdicts", pending)))
    return calls


def _fail_at_server(monkeypatch, prefix: str) -> None:
    """The first statement starting with `prefix` fails AT THE SERVER, inside the session's open
    transaction, which PostgreSQL then holds ABORTED until a rollback — the way a lock timeout or a
    deadlock arrives. (The statement is swapped for one the server rejects. Raising from a patched
    repository method instead fails before the session begins a transaction, and hides both the
    expiry defect and the need for the rollback.)"""
    from sqlalchemy.dialects.postgresql import asyncpg as pg_asyncpg

    gated = pg_asyncpg.AsyncAdapt_asyncpg_cursor._prepare_and_execute
    armed = [True]

    async def failing(self, operation, parameters):
        if armed[0] and operation.lstrip().upper().startswith(prefix):
            armed[0] = False
            return await gated(self, "SELECT 1 / 0", ())
        return await gated(self, operation, parameters)

    monkeypatch.setattr(pg_asyncpg.AsyncAdapt_asyncpg_cursor, "_prepare_and_execute", failing)


async def _record(request_id: str = "req-F") -> None:
    await _inference()._probe_record(_context(request_id, "ps_a"), [_verdict()], detached=True,
                                     reply_text="ok")


async def test_a_failed_retention_pass_leaves_the_committed_events_reported_and_emitted(
    world, emitted, followups, monkeypatch, caplog  # noqa: F811 - fixture
):
    # `prune_to_cap`'s DELETE (the age prune's has no `probe_id`).
    _fail_at_server(monkeypatch, "DELETE FROM PROBE_EVENTS WHERE PROBE_EVENTS.PROBE_ID")
    await _record()

    assert await _stored(world.maker) == {"req-F": (1, 1)}, "the event or its turn was lost"
    assert world.pes.shared_state().last_not_persisted == [], (
        "a committed event was reported not_persisted because a PRUNE failed"
    )
    assert [rid for rid, _p in emitted] == ["req-F"], "the committed event never reached the socket"
    assert all(p["id"] is not None for _rid, payloads in emitted for p in payloads)
    assert ("verdicts", None) in followups and ("rules", 1) in followups, (
        "the after-response code-rule jobs were never handed off"
    )
    assert "probe_event_retention_failed" in [r.getMessage() for r in caplog.records]
    assert "probe_record_failed" not in [r.getMessage() for r in caplog.records]


async def test_a_failed_turn_write_still_publishes_the_committed_events(
    world, emitted, followups, monkeypatch, caplog  # noqa: F811 - fixture
):
    from miserver.services import probe_sessions

    _fail_at_server(monkeypatch, "INSERT INTO PROBE_SESSION_TURNS")
    before = probe_sessions.counters["session_write_failures"]
    await _record()

    assert await _stored(world.maker) == {"req-F": (1, 0)}
    assert probe_sessions.counters["session_write_failures"] == before + 1
    assert world.pes.shared_state().last_not_persisted == []
    assert [rid for rid, _p in emitted] == ["req-F"], (
        "the turn write's rollback expired the committed rows and the publish raised"
    )
    assert ("verdicts", None) in followups and ("rules", 1) in followups, (
        "the record raised, so the after-response code-rule jobs were never handed off"
    )
    assert "probe_record_failed" not in [r.getMessage() for r in caplog.records]
