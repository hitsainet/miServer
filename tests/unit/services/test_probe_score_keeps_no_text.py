"""Stateless scoring keeps no input text, even with transcripts on (043 FR-043.14, task 6.1).

`test_probe_score_writes_nothing.py` holds the event count (no `probe_events` row). This adds the
text half, with COUNTS: with `PROBE_TRANSCRIPTS=on` and a session header, a scoring call over a
`messages` input and a `text` input writes no session, no turn and no transcript row.

R1 review fix (OR1-043-5), 2026-11-16: the counts cover EVERY table in `Base.metadata`, not the four
probe tables: `sensing_events`, `circuit_edge_sensing_events` and the batch tables also hold JSON or text, and a
scoring path that started writing any table would go unseen by a fixed list.

R2 review fix (OR2-043-1), 2026-11-16: the counts read the request's own session, but the writers that hold
text in production (`_probe_record`, the turn and transcript writers, the batch and legacy-name flushes) open
`miserver.db.base.async_session_factory`, the module engine on `settings.DATABASE_URL`, which the test never
counted. That factory is now bound to the test's engine, so every writer lands in the counted database.

R3 review fix (OR3-043-3), 2026-11-16 (adopted from the review draft): a writer that opens its own engine
(`create_async_engine(settings.DATABASE_URL)`) or a session on `miserver.db.base.engine` writes past both the
factory and the counts (review mutations OE1, OE2 stayed green). An engine-wide `before_cursor_execute` listener
on `sqlalchemy.engine.Engine` sees every statement on every engine during the call; no INSERT, UPDATE, DELETE,
REPLACE or MERGE may run.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select

from miserver.core.config import settings
from miserver.db.base import Base
from miserver.db.repositories.probe_repository import ProbeRepository
from miserver.services.probe_arming import armed_probe_from_row
from miserver.services.probe_runtime import ProbeRuntimeState
from miserver.services.probe_service import ProbeService
from tests.unit.f25_fixtures import clear_loaded, make_service, word_model, word_tokenizer
from tests.unit.score_fixtures import tiny_definition, tiny_identity


@pytest.fixture(autouse=True)
def clean():
    ProbeRuntimeState.reset_for_tests()
    yield
    ProbeRuntimeState.reset_for_tests()
    clear_loaded()


#: Tables the fixture itself fills before the call (the imported probe); their count must not move either.
SEEDED = {"probes"}
#: A floor on the table count, so an empty or partial metadata cannot pass.
MIN_TABLES = 20


async def _counts(session) -> dict[str, int]:
    tables = Base.metadata.sorted_tables
    assert len(tables) >= MIN_TABLES, f"only {len(tables)} tables in the metadata"
    return {t.name: int((await session.execute(select(func.count()).select_from(t))).scalar_one())
            for t in tables}


async def test_scoring_writes_no_session_turn_or_transcript(test_session, test_engine, monkeypatch):
    import miserver.db.base as db_base
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

    monkeypatch.setattr(db_base, "async_session_factory",
                        async_sessionmaker(test_engine, class_=AsyncSession, expire_on_commit=False))
    from miserver.api.dependencies import get_db, get_inference_service, get_probe_repository
    from miserver.main import create_app

    monkeypatch.setattr(settings, "PROBE_TRANSCRIPTS", "on")
    model, tokenizer = word_model(), word_tokenizer()
    inference = make_service(model, tokenizer)
    monkeypatch.setattr("miserver.services.probe_arm_bridge.loaded_identity",
                        AsyncMock(return_value=(tiny_identity(), model, tokenizer)))
    repo = ProbeRepository(test_session)
    row = await ProbeService(repo).import_definition(tiny_definition())
    ProbeRuntimeState().arm(armed_probe_from_row(row, windows=["all"]), model)

    async def db():
        yield test_session

    app = create_app()
    app.dependency_overrides[get_db] = db
    app.dependency_overrides[get_inference_service] = lambda: inference
    app.dependency_overrides[get_probe_repository] = lambda: repo
    client = TestClient(app)
    before = await _counts(test_session)
    from sqlalchemy import event
    from sqlalchemy.engine import Engine

    writes: list[tuple[str, str]] = []
    seen: list[str] = []

    def _spy(conn, cursor, statement, params, context, executemany):  # noqa: ANN001
        verb = statement.lstrip().split(None, 1)[0].upper() if statement.strip() else ""
        seen.append(verb)
        if verb in {"INSERT", "UPDATE", "DELETE", "REPLACE", "MERGE"}:
            writes.append((str(conn.engine.url), statement[:120]))

    event.listen(Engine, "before_cursor_execute", _spy)
    try:
        response = client.post("/api/probes/score", headers={"X-miLLM-Session": "held-out-1"}, json={
            "inputs": [{"messages": [{"role": "user", "content": "w1 w2 w3"}]}, {"text": "w4 w5"}],
            "windows": ["all"],
        })
    finally:
        event.remove(Engine, "before_cursor_execute", _spy)
    assert seen, "precondition: the listener saw the route's own reads"
    assert writes == [], f"scoring wrote through an engine: {writes}"
    assert response.status_code == 200, response.text
    assert all(r["verdicts"] for r in response.json()["data"]["results"]), "precondition: scored"
    after = await _counts(test_session)
    assert {name: n for name, n in after.items() if n != before[name]} == {}, "scoring wrote a row"
    for name in ("probe_events", "probe_sessions", "probe_session_turns", "probe_transcripts",
                 "sensing_events", "circuit_edge_sensing_events", "batches", "batch_rows"):
        assert before[name] == 0, f"{name} counted"
    assert {name for name, n in after.items() if n} <= SEEDED, after
    assert "X-miLLM-Session-Id" not in response.headers
