"""A manual detach (`POST /api/saes/{id}/detach`): what it turns off, and what it says (2026-11-21).

Real `SAEService.detach_sae` over a real SQLite database (`f28_fixtures.db`), REAL `LoadedSAE`s hooked
on a tiny real Llama through the real `SAEHooker`, in the real `AttachedSAEState`.

* RR-12: a detached SAE's `is_steering_enabled` is False (it stayed True with its values cleared).
* RR-8 (operator decision, 2026-11-21): a detach first deactivates the circuits and the profile or
  cluster that steer through THAT SAE, through their own deactivate paths, and names them in the
  route's answer. A co-tenant steering another SAE keeps serving. The cluster's sensing is disarmed,
  and a later attach of the same SAE re-arms nothing and re-activates nothing (as after an unload).

Fixtures disagree with the defect: two SAEs on two layers; one circuit and the cluster on SAE A,
another circuit on SAE B; names differ from ids.

MUTATION CONTROLS (each must turn this file red; recorded in
0xcc/reviews/unload_detaches_saes_2026-11-21.md §11):
  R12 `enable_steering(False)` removed from the detach         -> test_a_detached_sae_is_not_steering
  R8a the deactivation call removed from detach_sae (wiring)    -> TestDetachDeactivatesItsOwners
  R8b the SAE filter dropped (every circuit deactivated)        -> test_a_co_tenant_on_another_*
  R8c the circuit filter ignores the owner map (layers only)    -> test_a_circuit_found_by_its_owner_map
  R8d the route answer drops the names                          -> test_the_route_answer_names_*
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import pytest

from miserver.db.models.sae import SAE, SAEStatus
from miserver.db.repositories.sae_repository import SAERepository
from miserver.services.sae_service import AttachedSAEState, SAEService
from tests.unit.checkpoint_service_fixtures import FakeInference
from tests.unit.f25_fixtures import word_model
from tests.unit.f28_fixtures import SAE_A, SAE_B, attach, clean_state, db  # noqa: F401

NAMES = {SAE_A: "alpha-resid-l0-8k", SAE_B: "beta-resid-l1-8k"}


async def seed_saes(factory: Any) -> None:
    async with factory() as session:
        for sae_id in (SAE_A, SAE_B):
            session.add(SAE(id=sae_id, repository_id=f"vendor/{sae_id}", name=NAMES[sae_id],
                            d_in=16, d_sae=8, cache_path=f"/data/saes/{sae_id}",
                            status=SAEStatus.ATTACHED))
        await session.commit()


@pytest.fixture
async def served(clean_state, db):  # noqa: F811
    await seed_saes(db)
    model = word_model()
    out = attach(model, [(SAE_A, 0, 11), (SAE_B, 1, 12)])
    for sae in out.saes.values():
        sae.set_steering_batch({3: 5.0})
        sae.enable_steering(True)
    yield out


def detach_service(session: Any, emitter: Any = None) -> SAEService:
    """The route's service: `get_sae_service` builds exactly this (repository on the request's
    session, the progress emitter, the inference service)."""
    import tempfile

    return SAEService(repository=SAERepository(session), cache_dir=tempfile.mkdtemp(),
                      emitter=emitter or AsyncMock(), inference_service=FakeInference())


async def test_a_detached_sae_is_not_steering(served, db):  # noqa: F811
    """RR-12: values cleared AND steering off on the SAE that was detached; the other untouched."""
    sae_a, sae_b = served.sae(SAE_A, 0), served.sae(SAE_B, 1)
    async with db() as session:
        await detach_service(session).detach_sae(SAE_A)
    assert sae_a.get_steering_values() == {} and sae_a.is_steering_enabled is False
    assert sae_b.get_steering_values() == {3: 5.0} and sae_b.is_steering_enabled is True
    assert [e.sae_id for e in AttachedSAEState().entries()] == [SAE_B]


# ── RR-8 ──────────────────────────────────────────────────────────────────────────────────────

CLUSTER_META = {"members": [{"feature_idx": 3, "label": "calm", "max_activation": 2.0}]}


async def seed_owners(factory: Any) -> None:
    """circ_a steers SAE A (layer 0) and the cluster is bound to A; circ_b steers SAE B (layer 1).
    circ_a's recorded `layers` say [5]: only its owner map ties it to A (R8c)."""
    from miserver.db.models.circuit import Circuit
    from miserver.db.models.circuit_layer_claim import CircuitLayerClaim
    from miserver.db.models.model import Model, ModelSource, ModelStatus, QuantizationType
    from miserver.db.models.profile import Profile

    async with factory() as session:
        session.add(Model(id=1, name="m1", source=ModelSource.HUGGINGFACE, repo_id="vendor/m1",
                          quantization=QuantizationType.FP16, cache_path="/data/models/m1",
                          status=ModelStatus.LOADED))
        for cid, name, layers in (("circ_a", "circuit on alpha", [5]),
                                  ("circ_b", "circuit on beta", [1])):
            session.add(Circuit(id=cid, name=name, description=None, rung=2, edge_count=1,
                                layers=layers, serveable=True, is_active=True,
                                serving_mode="full", intensity=1.0, per_sae_warnings=[],
                                circuit_meta={"budget": {}}))
        session.add(CircuitLayerClaim(circuit_id="circ_a", layer=0, steering_keys=[2]))
        session.add(CircuitLayerClaim(circuit_id="circ_b", layer=1, steering_keys=[2]))
        session.add(Profile(id="prof_calm", name="calm cluster", steering={"3": 5.0},
                            is_active=True, intensity=1.0, sae_id=SAE_A, layer=0,
                            source_kind="cluster", sensing_enabled=True,
                            cluster_meta=CLUSTER_META))
        await session.commit()


@pytest.fixture
async def owned(served, db):  # noqa: F811
    from datetime import datetime

    import miserver.api.dependencies as deps
    from miserver.db.models.profile import Profile
    from miserver.ml.model_loader import LoadedModel, LoadedModelState

    await seed_owners(db)
    state = AttachedSAEState()
    state.apply_owner("circuit:circ_a", {(SAE_A, 0): {2: 4.0}})
    state.apply_owner("circuit:circ_b", {(SAE_B, 1): {2: 4.0}})
    LoadedModelState().set(LoadedModel(model_id=1, model_name="m1", model=served.model,
                                       tokenizer=None, loaded_at=datetime(2026, 11, 21)))
    sensing = deps.get_sensing_service()
    async with db() as session:
        sensing.arm_for_profile(await session.get(Profile, "prof_calm"), served.sae(SAE_A, 0))
    assert sensing.is_armed
    yield served
    sensing.disarm(None)
    LoadedModelState()._loaded = None


async def rows(db: Any) -> dict:  # noqa: F811
    from sqlalchemy import select

    from miserver.db.models.circuit import Circuit
    from miserver.db.models.circuit_layer_claim import CircuitLayerClaim
    from miserver.db.models.profile import Profile

    async with db() as session:
        circuits = {c.id: c.is_active for c in (await session.execute(select(Circuit))).scalars()}
        claims = {c.circuit_id: c.released_at is not None
                  for c in (await session.execute(select(CircuitLayerClaim))).scalars()}
        profile = (await session.get(Profile, "prof_calm")).is_active
    return {"circuits": circuits, "released": claims, "cluster": profile}


class TestDetachDeactivatesItsOwners:
    async def test_the_circuit_and_cluster_on_it_go_and_its_sensing_is_disarmed(self, owned, db):  # noqa: F811
        import miserver.api.dependencies as deps

        async with db() as session:
            result = await detach_service(session).detach_sae(SAE_A)
        assert await rows(db) == {"circuits": {"circ_a": False, "circ_b": True},
                                  "released": {"circ_a": True, "circ_b": False},
                                  "cluster": False}
        assert deps.get_sensing_service().is_armed is False
        assert result["deactivated_circuits"] == [
            {"id": "circ_a", "name": "circuit on alpha", "kind": "circuit"}]
        assert result["deactivated_profiles"] == [
            {"id": "prof_calm", "name": "calm cluster", "kind": "cluster"}]
        assert "warnings" not in result

    async def test_a_co_tenant_on_another_sae_keeps_serving(self, owned, db):  # noqa: F811
        sae_b = owned.sae(SAE_B, 1)
        async with db() as session:
            await detach_service(session).detach_sae(SAE_A)
        assert sae_b.get_steering_values() == {2: 4.0} and sae_b.is_steering_enabled
        assert set(AttachedSAEState()._owners) == {"circuit:circ_b"}
        assert (await rows(db))["circuits"]["circ_b"] is True

    async def test_a_circuit_found_by_its_owner_map(self, owned, db):  # noqa: F811
        """circ_a's `layers` column says [5]; it steers SAE A only by its owner map entry."""
        async with db() as session:
            result = await detach_service(session).detach_sae(SAE_A)
        assert [c["id"] for c in result["deactivated_circuits"]] == ["circ_a"]

    async def test_detaching_the_other_sae_leaves_the_cluster(self, owned, db):  # noqa: F811
        async with db() as session:
            result = await detach_service(session).detach_sae(SAE_B)
        assert result["deactivated_profiles"] == []
        assert [c["id"] for c in result["deactivated_circuits"]] == ["circ_b"]
        assert await rows(db) == {"circuits": {"circ_a": True, "circ_b": False},
                                  "released": {"circ_a": False, "circ_b": True},
                                  "cluster": True}

    async def test_a_later_attach_re_arms_and_re_activates_nothing(self, owned, db, monkeypatch):  # noqa: F811
        """As after an unload: sensing was disarmed with the cluster, and the attach re-arms only
        for an ACTIVE cluster, so re-attaching SAE A brings back neither."""
        import miserver.api.dependencies as deps
        from miserver.services.sae_service import CompatibilityResult
        from tests.unit.f28_fixtures import make_sae

        async with db() as session:
            await detach_service(session).detach_sae(SAE_A)
        async with db() as session:
            service = detach_service(session)
            monkeypatch.setattr(service, "check_compatibility",
                                AsyncMock(return_value=CompatibilityResult(True, [], [])))
            service._loader = type("L", (), {"load": lambda self, **kw: make_sae(21)})()
            await service.attach_sae(SAE_A, 0)
        fresh = AttachedSAEState().get(SAE_A, 0).sae
        assert deps.get_sensing_service().is_armed is False
        assert fresh.get_steering_values() == {} and not fresh.is_steering_enabled
        assert (await rows(db))["cluster"] is False


async def test_the_route_answer_names_what_it_deactivated(owned, db, monkeypatch, tmp_path):  # noqa: F811
    """`POST /api/saes/{id}/detach` over the live app and the request's own session."""
    from fastapi.testclient import TestClient

    from miserver.api.dependencies import get_db, get_inference_service
    from miserver.core.config import settings
    from miserver.main import create_app

    monkeypatch.setattr(settings, "SAE_CACHE_DIR", str(tmp_path))

    app = create_app()

    async def _db():
        async with db() as session:
            yield session

    app.dependency_overrides[get_db] = _db
    app.dependency_overrides[get_inference_service] = lambda: FakeInference()
    r = TestClient(app).post(f"/api/saes/{SAE_A}/detach")
    assert r.status_code == 200, r.text
    data = r.json()["data"]
    assert data["deactivated_circuits"] == [
        {"id": "circ_a", "name": "circuit on alpha", "kind": "circuit"}]
    assert data["deactivated_profiles"] == [
        {"id": "prof_calm", "name": "calm cluster", "kind": "cluster"}]
    assert data["warnings"] == []
