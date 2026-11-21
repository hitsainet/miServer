"""The context ceiling on the wire: the load route, the approval card, its digest, the PUT (2026-11-21).

An agent files loads through the approval flow (ENH-008, `action: model_load`). A ceiling the
agent names is in the stored payload, so the `request_digest` covers it and the card shows it:
the operator approves the window, not just the model. The model's own ceiling is written by
`PUT /api/models/{id}/context-ceiling`, an agent's write waiting as `model_context_write`.

MUTATION CONTROLS (each must turn this file red; run 2026-11-21, recorded in
0xcc/reviews/gguf_context_ceiling_2026-11-21.md):
  A1 the route drops request.context_ceiling                 -> test_the_route_forwards_*,
                                                                test_the_approved_load_runs_*
  A2 the load card leaves out "Context ceiling"              -> test_the_card_*
  A3 the describer skips validation                          -> test_a_refused_ceiling_files_nothing
  A4 ModelLoadRequest drops the field (extra ignored)        -> every test here
"""

from __future__ import annotations

import pytest

from miserver.services import approval_service
from tests.unit.approval_fixtures import (  # noqa: F401
    AGENT,
    harness,
    make_gguf_model,
    make_model,
    rows,
)
from tests.unit.f28_fixtures import db  # noqa: F401 - fixture


async def test_the_route_forwards_the_ceiling(harness):  # noqa: F811
    h = harness
    h.holder["row"] = row = await make_gguf_model(h.db)
    r = h.client.post(f"/api/models/{row.id}/load", json={"context_ceiling": "max"})
    assert r.status_code == 202, r.text
    h.service.load_model.assert_awaited_once_with(
        row.id, gpu=None, lease_id=None, context_ceiling=0)


async def test_the_card_shows_the_ceiling_and_the_digest_covers_it(harness):  # noqa: F811
    h = harness
    h.holder["row"] = row = await make_gguf_model(h.db)
    first = h.client.post(f"/api/models/{row.id}/load", json={"context_ceiling": 65536},
                          headers=AGENT).json()["data"]
    second = h.client.post(f"/api/models/{row.id}/load", json={"context_ceiling": "max"},
                           headers=AGENT).json()["data"]
    assert first["summary"]["Context ceiling"] == "65,536 tokens (named in this request)"
    assert second["summary"]["Context ceiling"].startswith("max: as large a window as fits")
    assert first["request_digest"] != second["request_digest"]
    stored = {s.id: s for s in await rows(h.db)}
    payload = stored[first["approval_id"]].payload
    assert payload["request"]["context_ceiling"] == 65536
    assert first["request_digest"] == "sha256:" + approval_service.digest(payload)
    assert h.service.load_model.await_count == 0


async def test_the_card_names_the_server_default_when_nothing_is_set(harness):  # noqa: F811
    h = harness
    h.holder["row"] = row = await make_gguf_model(h.db)
    data = h.client.post(f"/api/models/{row.id}/load", headers=AGENT).json()["data"]
    assert data["summary"]["Context ceiling"] == (
        "32,768 tokens (the server default, GGUF_CONTEXT_LENGTH)")


async def test_the_approved_load_runs_with_the_ceiling_it_showed(harness):  # noqa: F811
    h = harness
    h.holder["row"] = row = await make_gguf_model(h.db)
    data = h.client.post(f"/api/models/{row.id}/load", json={"context_ceiling": 65536},
                         headers=AGENT).json()["data"]
    assert h.client.post(f"/api/approvals/{data['approval_id']}/approve").status_code == 202
    await approval_service.wait_all()
    (after,) = await rows(h.db)
    assert after.status == "executed", (after.status, after.error)
    h.service.load_model.assert_awaited_once_with(
        row.id, gpu=None, lease_id=None, context_ceiling=65536)


@pytest.mark.parametrize("body", [{"context_ceiling": 65536}, {"context_ceiling": 1024}])
async def test_a_refused_ceiling_files_nothing(harness, body):  # noqa: F811
    """A transformers row, or a window under 2048: refused before anything is stored."""
    h = harness
    h.holder["row"] = row = await (
        make_model(h.db) if body["context_ceiling"] == 65536 else make_gguf_model(h.db))
    r = h.client.post(f"/api/models/{row.id}/load", json=body, headers=AGENT)
    assert r.status_code == 422, r.text
    assert await rows(h.db) == []


@pytest.mark.parametrize("value", ["32k", True, -5])
async def test_a_malformed_ceiling_is_a_422(harness, value):  # noqa: F811
    h = harness
    h.holder["row"] = row = await make_gguf_model(h.db)
    r = h.client.post(f"/api/models/{row.id}/load", json={"context_ceiling": value})
    assert r.status_code == 422, r.text
    assert h.service.load_model.await_count == 0


async def test_the_context_write_card(harness):  # noqa: F811
    h = harness
    h.holder["row"] = row = await make_gguf_model(h.db)
    data = h.client.put(f"/api/models/{row.id}/context-ceiling",
                        json={"context_ceiling": 131072}, headers=AGENT).json()["data"]
    assert data["action"] == "model_context_write"
    assert data["summary"]["Context ceiling now"] == (
        "32,768 tokens (the server default, GGUF_CONTEXT_LENGTH)")
    assert data["summary"]["After approval"] == "131,072 tokens (this model's setting)"
    assert h.service.set_context_ceiling.await_count == 0


async def test_the_operator_write_runs_with_its_value(harness):  # noqa: F811
    h = harness
    h.holder["row"] = row = await make_gguf_model(h.db)
    r = h.client.put(f"/api/models/{row.id}/context-ceiling", json={"context_ceiling": None})
    assert r.status_code == 200, r.text
    h.service.set_context_ceiling.assert_awaited_once_with(row.id, None)


async def test_the_context_write_is_an_approve_all_settings_change(harness):  # noqa: F811
    data = harness.client.get("/api/approvals/actions").json()["data"]
    listed = {a["action"]: a for a in data["actions"]}
    assert listed["model_context_write"]["routes"] == [
        "PUT /api/models/{model_id}/context-ceiling"]
    from miserver.api.approval_gate import settings_actions

    assert "model_context_write" in settings_actions()
