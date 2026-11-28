"""A probe weight-space programming derived from another shows its lineage before it is armed
(043 FR-043.16, FR-043.18; D43-13, TD43-6).

The projection tests store `provenance.derived_from` on an imported probe's definition, which is
exactly what the projection reads (the raw stored definition, never the mirror); the later tests
import the derived document through the real route, against the re-vendored schema (044-B).

- `GET /api/probes/{id}` and `GET /api/probes` carry `derived_from` verbatim; `null` for a plain
  probe (7.6).
- Arming a derived probe at rung 0 still needs the below-rung-2 acknowledgement (7.6b, FR-24.11).
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from miserver.core.errors import UnvalidatedProbeError
from miserver.db.repositories.probe_repository import ProbeRepository
from miserver.services.probe_arming import ProbeArmingService
from miserver.services.probe_runtime import ProbeRuntimeState
from miserver.services.probe_service import ProbeService
from tests.unit.probe_fixtures import acknowledged
from tests.unit.services.test_probe_arming import (
    TinyModel,
    definition,
    forward_at,
    loaded,
    run_inline,
)

DERIVED_FROM = {
    "parent_probe_id": "pr_0a1b2c3d4e5f",
    "parent_detector_digest": "a" * 64,
    "patch_sha256": "b" * 64,
    "provider": "weightspace",
    "provider_version": "0.3.0",
    "verdict": {"value": "improved", "run_sha256": "c" * 64},
}


@pytest.fixture(autouse=True)
def clean_state():
    ProbeRuntimeState.reset_for_tests()
    yield
    ProbeRuntimeState.reset_for_tests()


async def _derived(repo: ProbeRepository, *, rung: int = 2, name: str = "derived"):
    doc = definition()
    doc["name"] = name
    if rung < 2:
        doc["evidence"] = {"rung": rung, "rung_language": "trained",
                           "acknowledgement": acknowledged("WSP candidate"), "evaluations": []}
    row = await ProbeService(repo).import_definition(doc)
    stored = copy.deepcopy(row.definition)
    stored.setdefault("provenance", {})["derived_from"] = copy.deepcopy(DERIVED_FROM)
    await repo.update(row, definition=stored)
    return row


@pytest.fixture
def client(test_session):
    from miserver.api.dependencies import get_db, get_probe_repository
    from miserver.main import create_app

    repo = ProbeRepository(test_session)

    async def db():
        yield test_session

    app = create_app()
    app.dependency_overrides[get_db] = db
    app.dependency_overrides[get_probe_repository] = lambda: repo
    # No `with`: entering the client runs the app's lifespan.
    return TestClient(app), repo


async def test_the_detail_and_the_list_carry_derived_from(client):
    http, repo = client
    derived = await _derived(repo)
    plain = await ProbeService(repo).import_definition({**definition(), "name": "plain"})
    detail = http.get(f"/api/probes/{derived.id}")
    assert detail.status_code == 200, detail.text
    assert detail.json()["data"]["derived_from"] == DERIVED_FROM
    plain_detail = http.get(f"/api/probes/{plain.id}").json()["data"]
    assert "derived_from" in plain_detail and plain_detail["derived_from"] is None
    listed = {p["id"]: p for p in http.get("/api/probes").json()["data"]}
    assert listed[derived.id]["derived_from"] == DERIVED_FROM
    assert listed[plain.id]["derived_from"] is None


async def test_a_derived_rung_zero_probe_needs_the_acknowledgement(test_session):
    repo = ProbeRepository(test_session)
    probe = await _derived(repo, rung=0, name="derived-rung0")
    assert (probe.definition.get("provenance") or {}).get("derived_from") == DERIVED_FROM
    arming = ProbeArmingService(repo, ProbeRuntimeState())
    with pytest.raises(UnvalidatedProbeError) as exc:
        await arming.arm(probe, executor=run_inline, model=TinyModel(), loaded=loaded(),
                         forward=forward_at(0.5))
    assert exc.value.details["rung"] == 0
    assert ProbeRuntimeState().has_armed() is False
    await arming.arm(probe, executor=run_inline, model=TinyModel(), loaded=loaded(),
                     forward=forward_at(0.5), acknowledge_below_rung2=True,
                     reason="reviewing a WSP candidate")
    assert (await repo.get(probe.id)).armed is True


# ── after the re-vendor (044-B): the real import path ───────────────────────────────────────────

CASES = Path(__file__).resolve().parents[3] / "docs" / "schemas" / "probe-identity-cases.json"


def _case(name: str) -> dict:
    cases = json.loads(CASES.read_text())["cases"]
    (case,) = [c for c in cases if c["name"] == name]
    return case


async def test_derived_from_survives_the_import_route_and_the_mirror(client):
    """7.4 / FR-043.16: through `POST /api/probes/import`, `GET /api/probes/{id}`, the stored
    `provenance` column and a `model_dump` of the mirror."""
    from miserver.api.schemas.probe import ProbeDefinitionV1

    http, repo = client
    doc = definition()
    doc["name"] = "derived-import"
    doc["provenance"] = {"run_id": "pmr_x", "derived_from": copy.deepcopy(DERIVED_FROM)}
    r = http.post("/api/probes/import", json=doc)
    assert r.status_code == 200 and r.json()["success"], r.text
    probe_id = r.json()["data"]["id"]
    assert http.get(f"/api/probes/{probe_id}").json()["data"]["derived_from"] == DERIVED_FROM
    row = await repo.get(probe_id)
    assert row.provenance["derived_from"] == DERIVED_FROM
    assert row.definition["provenance"]["derived_from"] == DERIVED_FROM
    dumped = ProbeDefinitionV1.model_validate(row.definition).model_dump(mode="json")
    assert dumped["provenance"]["derived_from"] == DERIVED_FROM
    # A plain definition dumps NO `derived_from` key (not null): an older consumer refuses null.
    plain = ProbeDefinitionV1.model_validate(definition()).model_dump(mode="json")
    assert "derived_from" not in plain["provenance"]


async def test_the_derived_case_imports_with_its_identity_and_a_tampered_one_is_refused(client):
    """7.5 / FR-043.17 (P25): the case `excluded-provenance.derived_from`, carrying its identity
    block, imports; one hex digit changed in `detector_digest` is refused 409."""
    http, _ = client
    case = _case("excluded-provenance.derived_from")
    assert case["relation"] == {"to": "residual", "detector": "equal", "decision": "equal"}
    twin = _case("residual")
    assert (case["detector_digest"], case["decision_digest"]) == (
        twin["detector_digest"], twin["decision_digest"])
    doc = copy.deepcopy(case["document"])
    doc["identity"] = {"algorithm": "mistudio.probe-identity/1",
                       "detector_digest": case["detector_digest"],
                       "decision_digest": case["decision_digest"]}
    ok = http.post("/api/probes/import", json=doc)
    assert ok.status_code == 200 and ok.json()["success"], ok.text
    tampered = copy.deepcopy(doc)
    digest = tampered["identity"]["detector_digest"]
    tampered["identity"]["detector_digest"] = ("0" if digest[0] != "0" else "1") + digest[1:]
    tampered["name"] = "tampered"
    refused = http.post("/api/probes/import", json=tampered)
    assert refused.status_code == 409, refused.text
    assert refused.json()["error"]["code"] == "PROBE_IDENTITY_MISMATCH"
