"""G1 (043 FR-043.2 to FR-043.4): probe verdicts are provisional whenever the served weights are
not the base the probe was fitted on, and the `X-miLLM-Model` header agrees.

Driven through the REAL app on the CPU weights flow (`tests/unit/weights_e2e_fixtures.py`): a real
`WeightBarrier`, real commits through the holder routes, real saves and checkpoint loads, and one
REAL armed probe whose verdict is read from `X-miLLM-Probe-Verdicts` and from the stored
`probe_events` row of the same request.

The state table (FPRD FR-043.3) is the specification. Each served state is one row; every row is
one chat completion, and from that one response the test reads:

- the verdict's `provisional` (header) and the event row's `provisional` (database);
- `X-miLLM-Model`. FR-043.4: without an adapter, the weights fact is false EXACTLY when the header
  reads `origin=model;commits=0`. The header is computed by `_commits_since` and the fact by
  `served_weights`, independently, so drift in either turns a row red.

The probe's own window reasons never apply here (scope `all`, window `all`, calibrated and trained),
so a provisional verdict on a row without an adapter is the weights reason alone.

Red on the code before 043 (the proof that G1 existed): "after one commit", "a save with zero
commits", "after a save", "after a checkpoint load", "after one more commit".
"""

from __future__ import annotations

from typing import Any, Optional

import pytest
import torch
from sqlalchemy import event as sa_event
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from miserver.db.base import Base
from miserver.db.models.probe import ProbeEvent
from miserver.ml.model_loader import LoadedModelState
from miserver.ml.probe_head import ProbeHead
from miserver.services.probe_runtime import ArmedProbe, ProbeRuntimeState
from miserver.services import resident_identity
from tests.unit.api.test_weights_e2e_adapter_dropped_by_load import (  # noqa: F401 - fixture
    ADAPTER,
    _apply,
    adapters,
)
from tests.unit.api.test_weights_e2e_torn_recovery import (  # noqa: F401 - fixture
    _torn_commit,
    _until,
    short_watchdog,
)
from tests.unit.probe_fixtures import probe_definition
from tests.unit.weights_e2e_fixtures import (  # noqa: F401 - flow is a fixture
    OPS,
    ROW_ID,
    ROW_NAME,
    ROW_REPO,
    TRAINER,
    attach,
    chat,
    flow,
    holder,
    manifest,
)

BASE_HEADER = f'"{ROW_NAME}";origin=model;commits=0'


@pytest.fixture
async def probes(flow, monkeypatch):
    """A real probe row (so events persist against it) and its hook armed on the resident."""
    from miserver.api.dependencies import get_db
    from miserver.db.repositories.probe_repository import ProbeRepository
    from miserver.services.probe_service import ProbeService

    engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)

    @sa_event.listens_for(engine.sync_engine, "connect")
    def _fk(dbapi_connection, _record):  # noqa: ANN001
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    monkeypatch.setattr("miserver.db.base.async_session_factory", factory)

    async def _db():
        async with factory() as session:
            yield session

    flow.app.dependency_overrides[get_db] = _db
    ProbeRuntimeState.reset_for_tests()
    async with factory() as session:
        row = await ProbeService(ProbeRepository(session)).import_definition(
            probe_definition(), raw_bytes=1, on_conflict="rename", origin="file"
        )
    ns = _Probes(factory=factory, row=row)
    ns.arm()
    yield ns
    ProbeRuntimeState.reset_for_tests()
    await engine.dispose()


class _Probes:
    def __init__(self, factory: Any, row: Any) -> None:
        self.factory = factory
        self.row = row

    def arm(self) -> None:
        """Arm (or re-arm, after a FULL reload replaced the model object) on the resident."""
        state = ProbeRuntimeState()
        state.disarm(self.row.id)
        state.arm(ArmedProbe(
            probe_id=self.row.id, name=self.row.name,
            head=ProbeHead(weight=torch.ones(16), bias=0.0, layer=0),
            rule="mean", scope="all", layer=0, rung=2, rung_language="detects on unseen tasks",
            threshold=0.0, windows=("all",),
        ), LoadedModelState().current.model)

    async def events(self) -> list[ProbeEvent]:
        async with self.factory() as session:
            return list((await session.execute(select(ProbeEvent))).scalars().all())


def _provisional(response) -> bool:
    """The ONE verdict in the response's header: `provisional=?1` present or not."""
    assert response.status_code == 200, response.text
    header = response.headers["X-miLLM-Probe-Verdicts"]
    entries = [e for e in header.split(",") if e.strip()]
    assert len(entries) == 1, f"one armed probe x one window is one verdict: {header}"
    assert "not-scored" not in entries[0], header
    return "provisional=?1" in entries[0]


async def _row(client, probes, model: str = ROW_NAME) -> tuple[bool, str]:
    """One served request: (the verdict's provisional, X-miLLM-Model). Also checks the stored
    event of THIS request agrees with the header (one event per request)."""
    before = len(await probes.events())
    r = await chat(client, model)
    provisional = _provisional(r)
    events = await probes.events()
    assert len(events) == before + 1, "one event row per request, one probe, one window"
    mine = [e for e in events if e.request_id == r.json()["id"]]
    assert len(mine) == 1 and mine[0].provisional is provisional
    return provisional, r.headers["X-miLLM-Model"]


def _agrees(provisional: bool, model_header: str) -> None:
    """FR-043.4, for a row without an adapter: the fact is false exactly when the header is base."""
    assert provisional is (model_header != BASE_HEADER), (provisional, model_header)


async def _commit(client, attachment_id: str, model: Any, seed: int) -> dict:
    h = holder(attachment_id)
    assert (await client.post("/api/weights/quiesce", headers=h)).status_code == 200
    assert (await client.post("/api/weights/commit-begin", headers=h)).status_code == 200
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        p = next(model.parameters())
        p.add_(torch.randn(p.shape, generator=g).to(p.dtype) * 0.01)
    e = await client.post("/api/weights/commit-end", headers=h)
    assert e.status_code == 200, e.text
    assert (await client.post("/api/weights/resume", headers=h)).status_code == 200
    return e.json()["data"]


async def _detach(client, attachment_id: str) -> None:
    r = await client.post("/api/weights/detach", headers=holder(attachment_id))
    assert r.status_code == 200, r.text


async def _save(client, attachment_id: str, step: int, parent: Optional[str] = None) -> dict:
    r = await client.post("/api/weights/save", json={"manifest": manifest(step=step, parent=parent)},
                          headers=holder(attachment_id))
    assert r.status_code == 201, r.text
    return r.json()["data"]


async def test_state_table_through_the_real_flow(flow, probes):
    """FR-043.3 rows on one served model, in the FTID §7 order, each with FR-043.4's agreement."""
    table: list[tuple[str, bool, str]] = []
    async with flow.client() as client:
        provisional, header = await _row(client, probes)
        table.append(("plain row load", provisional, header))
        assert header == BASE_HEADER

        attachment_id = await attach(client)
        await _commit(client, attachment_id, flow.model, 1)
        provisional, header = await _row(client, probes)
        table.append(("after one commit", provisional, header))
        assert header == f'"{ROW_NAME}";origin=model;commits=1'

        saved = await _save(client, attachment_id, step=1)
        name1 = saved["name"]
        provisional, header = await _row(client, probes, name1)
        table.append(("after a save of those commits", provisional, header))
        await _detach(client, attachment_id)

        load = await client.post("/api/weights/load", json={"name": name1}, headers=TRAINER)
        assert load.status_code == 200, load.text
        assert load.json()["data"]["mode"] == "in_place"
        assert flow.barrier.snapshot().steps_since_load == 0, "the load's write resets the counter"
        provisional, header = await _row(client, probes, name1)
        table.append(("after POST /api/weights/load of a checkpoint", provisional, header))
        assert header == f'"{name1}";origin=checkpoint;commits=0'

        attachment_id = await attach(client)
        await _commit(client, attachment_id, flow.model, 2)
        await _detach(client, attachment_id)
        provisional, header = await _row(client, probes, name1)
        table.append(("after one more commit", provisional, header))

        load = await client.post("/api/weights/load", json={"model_id": ROW_ID}, headers=TRAINER)
        assert load.status_code == 200, load.text
        assert load.json()["data"]["mode"] == "full_reload"
        probes.arm()  # a full reload is a new model object; the operator re-arms
        provisional, header = await _row(client, probes)
        table.append(("after POST /api/weights/load of the base row", provisional, header))

        r = await client.post(f"/api/models/{ROW_ID}/unload")
        assert r.status_code == 200, r.text
        await _until(lambda: LoadedModelState().current is None)
        r = await client.post(f"/api/models/{ROW_ID}/load", json={})
        assert r.status_code in (200, 202), r.text
        await _until(lambda: LoadedModelState().current is not None, timeout=20.0)
        probes.arm()
        provisional, header = await _row(client, probes)
        table.append(("after POST /api/models/{id}/load", provisional, header))

    expected = {
        "plain row load": False,
        "after one commit": True,
        "after a save of those commits": True,
        "after POST /api/weights/load of a checkpoint": True,
        "after one more commit": True,
        "after POST /api/weights/load of the base row": False,
        "after POST /api/models/{id}/load": False,
    }
    got = {label: provisional for label, provisional, _ in table}
    assert got == expected
    for _label, provisional, header in table:
        _agrees(provisional, header)


async def test_a_save_with_zero_commits_is_provisional(flow, probes):
    """A conservative false positive (FPRD §9): the origin reads `checkpoint` though no value moved.
    This row kills M3 (a fact that ignores the origin)."""
    async with flow.client() as client:
        provisional, header = await _row(client, probes)
        assert provisional is False and header == BASE_HEADER
        attachment_id = await attach(client)
        saved = await _save(client, attachment_id, step=1)
        assert flow.barrier.snapshot().steps_since_load == 0
        provisional, header = await _row(client, probes, saved["name"])
        assert provisional is True
        assert header == f'"{saved["name"]}";origin=checkpoint;commits=0'
        _agrees(provisional, header)
        await _detach(client, attachment_id)


async def test_an_adapter_on_the_base_is_provisional_by_the_adapter_reason_only(
        flow, probes, adapters):  # noqa: F811
    async with flow.client() as client:
        await _apply(flow, client, base_name=ROW_REPO)
        assert resident_identity.served_weights().differs_from_base is False
        provisional, header = await _row(client, probes, ADAPTER)
        assert provisional is True
        assert header.startswith(f'"{ADAPTER}";origin=adapter')


async def test_an_adapter_over_committed_weights_holds_both_reasons(
        flow, probes, adapters):  # noqa: F811
    async with flow.client() as client:
        attachment_id = await attach(client)
        await _commit(client, attachment_id, flow.model, 3)
        await _detach(client, attachment_id)
        await _apply(flow, client, base_name=ROW_REPO)
        # The weights fact is true ON ITS OWN, so dropping the adapter reason keeps this red.
        assert resident_identity.served_weights().differs_from_base is True
        provisional, _ = await _row(client, probes, ADAPTER)
        assert provisional is True


async def test_a_torn_set_gives_no_verdict_and_no_event(flow, probes, short_watchdog):  # noqa: F811
    async with flow.client() as client:
        await _torn_commit(flow, client)
        await _until(lambda: flow.barrier.snapshot().weights_state == "torn")
        r = await chat(client)
        assert r.status_code == 503, r.text
        assert r.json()["error"]["code"] == "weights_torn"
        assert "X-miLLM-Probe-Verdicts" not in r.headers
        assert await probes.events() == []


async def test_surfaces_after_one_commit(flow, probes):
    """FR-043.5: `GET /api/probes/status` states the weights after a commit and after the base
    reload."""
    async with flow.client() as client:
        attachment_id = await attach(client)
        await _commit(client, attachment_id, flow.model, 4)
        status = (await client.get("/api/probes/status")).json()["data"]
        assert status["served_weights"] == {"name": ROW_NAME, "origin": "model",
                                            "commits_since_load": 1, "differs_from_base": True}
        await _detach(client, attachment_id)
        load = await client.post("/api/weights/load", json={"model_id": ROW_ID}, headers=TRAINER)
        assert load.status_code == 200, load.text
        status = (await client.get("/api/probes/status")).json()["data"]
        assert status["served_weights"] == {"name": ROW_NAME, "origin": "model",
                                            "commits_since_load": 0, "differs_from_base": False}



async def test_the_body_field_carries_the_contract_decision(flow, probes):
    """FR-043.2: `mi.probe.result/v1` `decision.provisional` in the non-streaming body field."""
    async with flow.client() as client:
        base = await chat(client)
        assert [e["decision"]["provisional"] for e in base.json()["millm_probe_verdicts"]] == [False]
        attachment_id = await attach(client)
        await _commit(client, attachment_id, flow.model, 5)
        r = await chat(client)
        assert r.status_code == 200, r.text
        entries = r.json()["millm_probe_verdicts"]
        assert len(entries) == 1
        assert entries[0]["decision"]["provisional"] is True
        assert entries[0]["provisional"] is True
        await _detach(client, attachment_id)


@pytest.fixture
async def scoring(flow, test_session, monkeypatch):
    """`POST /api/probes/score` on the flow's resident, with one imported probe."""
    from unittest.mock import AsyncMock

    from miserver.api.dependencies import get_db, get_probe_repository
    from miserver.db.repositories.probe_repository import ProbeRepository
    from miserver.services.probe_service import ProbeService
    from tests.unit.score_fixtures import tiny_definition, tiny_identity

    ProbeRuntimeState.reset_for_tests()
    monkeypatch.setattr(
        "miserver.services.probe_arm_bridge.loaded_identity",
        AsyncMock(side_effect=lambda _s: (tiny_identity(), flow.resident.model,
                                          flow.resident.tokenizer)),
    )
    repo = ProbeRepository(test_session)

    async def db():
        yield test_session

    flow.app.dependency_overrides[get_db] = db
    flow.app.dependency_overrides[get_probe_repository] = lambda: repo
    yield await ProbeService(repo).import_definition(tiny_definition())
    ProbeRuntimeState.reset_for_tests()


async def test_stateless_scoring_after_one_commit(flow, scoring):
    """FR-043.2 and FR-043.5 on `POST /api/probes/score`: every verdict provisional (one probe x one
    window x two inputs = two verdicts), and the `model` block states the weights."""
    body = {"inputs": [{"token_ids": [2, 6, 7, 8]}, {"token_ids": [2, 6, 9]}], "windows": ["all"]}
    async with flow.client() as client:
        base = (await client.post("/api/probes/score", json=body)).json()["data"]
        assert [v["provisional"] for r in base["results"] for v in r["verdicts"]] == [False, False]
        assert base["model"]["served_weights"]["differs_from_base"] is False
        attachment_id = await attach(client)
        await _commit(client, attachment_id, flow.model, 6)
        r = await client.post("/api/probes/score", json=body)
        assert r.status_code == 200, r.text
        data = r.json()["data"]
        verdicts = [v for result in data["results"] for v in result["verdicts"]]
        assert len(verdicts) == 2
        assert [v["provisional"] for v in verdicts] == [True, True]
        assert data["model"]["served_weights"] == {"name": ROW_NAME, "origin": "model",
                                                  "commits_since_load": 1,
                                                  "differs_from_base": True}
        await _detach(client, attachment_id)


async def test_a_behaviour_with_one_member_says_member_provisional(flow, probes):
    """FR-043.2: a behaviour's verdict over a member made provisional by the weights reads
    `provisional_reasons: ["member_provisional"]`. The member's verdict is the REAL one a chat after a
    real commit produced; the behaviour layer is fed it unchanged."""
    from types import SimpleNamespace

    from miserver.services import behaviour_verdict as bv
    from tests.support import verdict_factory as vf

    key, digest = "g1-behaviour", "d" * 64
    member = SimpleNamespace(probe_id=probes.row.id, descriptor=SimpleNamespace(
        behaviour_key=key, detector_digest=digest, decision=None))
    attached = bv.attached_from(key, vf.definition(key, [digest], candidate="any_of"))

    def behaviour(verdicts):
        items = bv.verdict_items(attached, verdicts, [member], rule_rows={},
                                 not_scored_reason=None, interrupted=None)
        assert len(items) == 1
        return items[0]

    state = ProbeRuntimeState()
    async with flow.client() as client:
        def one_verdict(request_id):
            state.begin_request(request_id)
            try:
                with torch.no_grad():
                    flow.model(torch.tensor([[0, 6, 7, 8]]))
            finally:
                context = state.end_request()
            (verdict,) = context.finish()
            return verdict

        base = one_verdict("base")
        assert base.scored and base.provisional is False
        assert behaviour([base])["provisional_reasons"] == []
        attachment_id = await attach(client)
        await _commit(client, attachment_id, flow.model, 7)
        after = one_verdict("after")
        assert after.scored and after.provisional is True
        assert behaviour([after])["provisional_reasons"] == ["member_provisional"]
        await _detach(client, attachment_id)


async def test_the_transition_is_logged_once_each_way(flow, probes, caplog, monkeypatch):
    """FTDD §11 / FTASKS 1.5: one `served_weights_differ` on the first context after a commit, none on
    the second, one `served_weights_base` after the base reload. No name at info level."""
    import logging

    from miserver.services import probe_runtime

    monkeypatch.setattr(probe_runtime, "_LAST_WEIGHTS", None)

    def lines(prefix):
        return [r for r in caplog.records
                if r.name == "miserver.services.probe_runtime" and r.getMessage().startswith(prefix)
                and r.levelno == logging.INFO]

    caplog.set_level(logging.INFO, logger="miserver.services.probe_runtime")
    async with flow.client() as client:
        await _row(client, probes)
        caplog.clear()
        attachment_id = await attach(client)
        await _commit(client, attachment_id, flow.model, 8)
        await _row(client, probes)
        assert len(lines("served_weights_differ")) == 1
        assert ROW_NAME not in lines("served_weights_differ")[0].getMessage()
        await _row(client, probes)
        assert len(lines("served_weights_differ")) == 1, "no line while the fact is unchanged"
        await _detach(client, attachment_id)
        load = await client.post("/api/weights/load", json={"model_id": ROW_ID}, headers=TRAINER)
        assert load.status_code == 200, load.text
        probes.arm()
        await _row(client, probes)
        assert len(lines("served_weights_base")) == 1
        assert len(lines("served_weights_differ")) == 1


async def test_no_info_record_carries_the_served_name(flow, probes, caplog, monkeypatch):
    """R3 review fix (OR3-043-4), 2026-11-16 (adopted from the review draft): the test above checks only the
    `served_weights_differ` line, so moving the separate `served_weights_name` line to INFO (review mutation X3)
    stayed green. The name can come from a foreign manifest (FTID §3.7, OVN-wsp-server-4): debug only."""
    import logging

    from miserver.services import probe_runtime

    monkeypatch.setattr(probe_runtime, "_LAST_WEIGHTS", None)
    caplog.set_level(logging.DEBUG, logger="miserver.services.probe_runtime")
    async with flow.client() as client:
        await _row(client, probes)
        attachment_id = await attach(client)
        await _commit(client, attachment_id, flow.model, 8)
        await _row(client, probes)
        await _detach(client, attachment_id)
        load = await client.post("/api/weights/load", json={"model_id": ROW_ID}, headers=TRAINER)
        assert load.status_code == 200, load.text
        probes.arm()
        await _row(client, probes)
    mine = [r for r in caplog.records if r.name == "miserver.services.probe_runtime"]
    assert any(r.getMessage().startswith("served_weights_differ") for r in mine), "precondition: a transition"
    assert any(ROW_NAME in r.getMessage() for r in mine if r.levelno == logging.DEBUG), "precondition: name at debug"
    leaked = [r.getMessage() for r in mine if r.levelno >= logging.INFO and ROW_NAME in r.getMessage()]
    assert leaked == [], leaked
