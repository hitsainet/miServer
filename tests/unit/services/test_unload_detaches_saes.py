"""Every unload detaches the attached SAEs and deactivates what steered through them (RR-1, 2026-11-21).

The defect, measured with the real `ModelService.unload_model`: the `AttachedSAEState` entry
survived the unload. The SAE stayed listed as attached and kept its memory, its hook sat on the
freed model's modules and steered nothing, and circuits and clusters steering through it read
active. It held on every unload path: an explicit unload, a load that swaps the model, a `/v1`
auto-load swap, a `/v1` reload of a carrier, the resident reload, and the weights API's full
reload.

Now `ModelService.unload_model` calls ONE place, `unload_teardown.release_dependents`, after the
drain and before the weights move. These tests drive EVERY path for real and read the outcome, not
the source: real `ModelService` and `ModelLoader`, a TINY REAL Llama with REAL `LoadedSAE`s hooked
through the real `SAEHooker`, and a real SQLite database behind `async_session_factory` holding the
SAE rows, their attachment rows, a serving circuit with its layer claims and an active cluster.

Fixtures disagree with the defect: two SAEs on two layers from different seeds, display names that
differ from their ids (a card or log printing one for the other fails), a circuit steering one SAE
through the owner map and a cluster steering the other, both live before the unload.

MUTATION CONTROLS (each must turn this file red; run 2026-11-21, recorded in
0xcc/reviews/unload_detaches_saes_2026-11-21.md):
  U1 the detach call removed from the shared place             -> TestEveryPath (rows stay attached)
  U2 unload_model no longer calls the shared place (the wiring) -> every path
  U3 the load swap bypasses unload_model (loader.unload())      -> load_swap, v1_swap
  U3b the shared place skipped under the weight barrier         -> TestWeightsFullReload
  U4 the circuit deactivation skipped                           -> TestEveryPath (circuit active)
  U5 the profile deactivation skipped                           -> TestEveryPath (cluster active)
  U6 the teardown moved after the weights are released          -> test_the_teardown_runs_while_*
  U7 the safety sweep removed                                   -> test_a_detach_that_raises_*
  U8 the sae:detached emit removed                              -> TestEveryPath (no socket event)
"""

from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlalchemy import select

from miserver.db.models.circuit import Circuit
from miserver.db.models.circuit_layer_claim import CircuitLayerClaim
from miserver.db.models.model import Model, ModelSource, ModelStatus, QuantizationType
from miserver.db.models.profile import Profile
from miserver.db.models.sae import SAE, SAEAttachment, SAEStatus
from miserver.ml.model_loader import LoadedModel, LoadedModelState, ModelLoader
from miserver.ml.sae_wrapper import LoadedSAE
from miserver.services import unload_teardown
from miserver.services.sae_service import AttachedSAEState
from tests.support.factories import make_model
from tests.unit.f25_fixtures import word_model
from tests.unit.f28_fixtures import SAE_A, SAE_B, attach, clean_state, db  # noqa: F401
from tests.unit.checkpoint_service_fixtures import harness  # noqa: F401 - fixture
from tests.unit.lease_fixtures import FakeRepo, clear_resident, drain_executor, install_registry

NAMES = {SAE_A: "alpha-resid-l0-8k", SAE_B: "beta-resid-l1-8k"}
CIRCUIT = "circ_live"
CLUSTER = "prof_calm"


def hooks(model: Any) -> int:
    return sum(len(m._forward_hooks) + len(m._forward_pre_hooks) for m in model.modules())


async def seed(factory: Any) -> None:
    """The rows an attach, a circuit activation and a cluster activation leave behind."""
    async with factory() as session:
        for model_id, name in ((1, "m1"), (2, "m2")):
            session.add(Model(id=model_id, name=name, source=ModelSource.HUGGINGFACE,
                              repo_id=f"vendor/{name}", quantization=QuantizationType.FP16,
                              cache_path=f"/data/models/{name}", status=ModelStatus.READY))
        for sae_id, layer in ((SAE_A, 0), (SAE_B, 1)):
            session.add(SAE(id=sae_id, repository_id=f"vendor/{sae_id}", name=NAMES[sae_id],
                            d_in=16, d_sae=8, cache_path=f"/data/saes/{sae_id}",
                            status=SAEStatus.ATTACHED))
            session.add(SAEAttachment(sae_id=sae_id, model_id=1, layer=layer, is_active=True))
        session.add(Circuit(id=CIRCUIT, name="live circuit", description=None, rung=2,
                            edge_count=1, layers=[1], serveable=True, is_active=True,
                            serving_mode="full", intensity=1.0, per_sae_warnings=[],
                            circuit_meta={"budget": {}}))
        session.add(CircuitLayerClaim(circuit_id=CIRCUIT, layer=1, steering_keys=[2]))
        session.add(Profile(id=CLUSTER, name="calm cluster", steering={"3": 5.0},
                            is_active=True, intensity=1.0, sae_id=SAE_A, layer=0,
                            source_kind="cluster"))
        await session.commit()


class World(SimpleNamespace):
    service: Any
    repo: FakeRepo
    model: Any
    saes: dict
    emitter: AsyncMock
    to_cpu: list


@pytest.fixture
async def world(clean_state, db):  # noqa: F811
    install_registry()
    clear_resident()
    await seed(db)
    model = word_model()
    served = attach(model, [(SAE_A, 0, 11), (SAE_B, 1, 12)])
    sae_a, sae_b = served.sae(SAE_A, 0), served.sae(SAE_B, 1)
    # Live before the unload: the cluster on SAE_A, the circuit (owner map) on SAE_B.
    sae_a.set_steering_batch({3: 5.0})
    sae_a.enable_steering(True)
    AttachedSAEState().apply_owner(f"circuit:{CIRCUIT}", {(SAE_B, 1): {2: 4.0}})
    assert sae_b.is_steering_enabled and hooks(model) == 2
    LoadedModelState().set(LoadedModel(model_id=1, model_name="m1", model=model, tokenizer=None,
                                       loaded_at=datetime(2026, 11, 21), gpu_indices=[0]))
    from miserver.services.model_service import ModelService

    repo = FakeRepo(make_model(id=1, name="m1", status=ModelStatus.LOADED, locked=True),
                    make_model(id=2, name="m2", status=ModelStatus.READY))
    inference = MagicMock()
    inference.request_queue.pending_count = 0
    inference.request_queue.occupied_count = 0
    emitter = AsyncMock()
    service = ModelService(repository=repo, downloader=MagicMock(), loader=ModelLoader(),
                           emitter=emitter, inference_service=inference)
    service._precheck_placement = MagicMock()  # type: ignore[method-assign]
    service._load_worker = MagicMock()  # type: ignore[method-assign]
    service.wait_until_loaded = AsyncMock()  # type: ignore[method-assign]
    to_cpu: list = []
    real_to_cpu = LoadedSAE.to_cpu

    def spy(self: LoadedSAE) -> None:
        to_cpu.append(self)
        real_to_cpu(self)

    with patch.object(LoadedSAE, "to_cpu", spy):
        yield World(service=service, repo=repo, model=model, saes={SAE_A: sae_a, SAE_B: sae_b},
                    emitter=emitter, to_cpu=to_cpu, db=db)
    drain_executor(service)
    clear_resident()


async def unload(w: World) -> None:
    await w.service.unload_model(1)


async def load_swap(w: World) -> None:
    await w.service.load_model(2)


async def resident_reload(w: World) -> None:
    # The resident is on GPU 0; naming GPU 1 is a reload of the same model (`resident_reload`).
    await w.service.load_model(1, gpu=1)


async def v1_swap(w: World) -> None:
    # Attaching locks the model, and a locked model refuses a /v1 swap (`model_locked`); the
    # operator can unlock it by hand (POST /api/models/{id}/unlock) with the SAEs still attached,
    # and then a /v1 request naming another model swaps it out.
    w.repo.rows[1].locked = False
    await w.service.load_model_and_wait(2)


async def v1_reload(w: World) -> None:
    await w.service.load_model_and_wait(1, reload=True)


PATHS = [unload, load_swap, resident_reload, v1_swap, v1_reload]


async def assert_torn_down(w: World) -> None:
    # In memory: no hook on the model's modules, nothing listed, both SAEs moved off, no owner.
    assert hooks(w.model) == 0, "a hook outlived the unload, on the freed model's modules"
    assert AttachedSAEState().count == 0 and AttachedSAEState()._owners == {}
    assert sorted(id(s) for s in w.to_cpu) == sorted(id(s) for s in w.saes.values())
    for sae in w.saes.values():
        # RR-12: values cleared AND steering off on the detached SAE.
        assert sae.get_steering_values() == {} and sae.is_steering_enabled is False
    async with w.db() as session:
        saes = {s.id: s.status for s in (await session.execute(select(SAE))).scalars()}
        attachments = (await session.execute(select(SAEAttachment))).scalars().all()
        circuit = await session.get(Circuit, CIRCUIT)
        claims = (await session.execute(select(CircuitLayerClaim))).scalars().all()
        cluster = await session.get(Profile, CLUSTER)
    assert saes == {SAE_A: SAEStatus.CACHED, SAE_B: SAEStatus.CACHED}
    assert [a.is_active for a in attachments] == [False, False]
    assert all(a.detached_at is not None for a in attachments)
    assert circuit.is_active is False
    assert [c.released_at is not None for c in claims] == [True]
    assert cluster.is_active is False
    detached = [c.args[0] for c in w.emitter.emit_sae_detached.await_args_list]
    assert detached == [SAE_A, SAE_B]
    w.emitter.emit_steering_changed.assert_awaited_with(enabled=False, values={}, active_count=0)


class TestEveryPath:
    @pytest.mark.parametrize("path", PATHS, ids=lambda p: p.__name__)
    async def test_detaches_the_saes_and_deactivates_the_circuit_and_cluster(self, world, path):
        await path(world)
        await assert_torn_down(world)

    @pytest.mark.parametrize("path", PATHS, ids=lambda p: p.__name__)
    async def test_the_shared_place_is_called_once_with_the_resident_while_it_is_resident(
        self, world, path, monkeypatch
    ):
        """The wiring, by driving the path: one call, naming model 1, made while the weights are
        still resident (the hooks come off a live model, not a freed one). U2, U3, U6."""
        calls: list[tuple[int, Any, int]] = []
        real = unload_teardown.release_dependents

        async def spy(model_id: int, **kwargs: Any) -> Any:
            calls.append((model_id, LoadedModelState().loaded_model_id, AttachedSAEState().count))
            return await real(model_id, **kwargs)

        monkeypatch.setattr(unload_teardown, "release_dependents", spy)
        await path(world)
        assert calls == [(1, 1, 2)]
        await assert_torn_down(world)


async def test_nothing_is_attached_or_activated_again_by_a_reload(world):
    """The decision (2026-11-21): a reload of the same model leaves them detached and its card says
    so. Re-attaching needs room on the card after the new load, and a circuit below rung 2 must be
    acknowledged again at activation (the reason startup deactivates circuits too)."""
    await resident_reload(world)
    drain_executor(world.service)
    # The reload's load was started, on the card it named...
    assert world.service._load_worker.call_count == 1
    assert world.repo.status_writes[-1] == (1, ModelStatus.LOADING)
    # ...and nothing came back: no entry, no hook, the rows detached and inactive.
    await assert_torn_down(world)


async def test_a_detach_that_raises_still_takes_the_hook_off_the_model(world, monkeypatch):
    """A bookkeeping failure in one detach must not leave its hook on the freed model, nor stop the
    unload; it is reported and logged (U7)."""
    from miserver.services.sae_service import SAEService

    real = SAEService.detach_for_unload

    async def flaky(self: SAEService, sae_id: str) -> dict:
        if sae_id == SAE_A:
            raise RuntimeError("database went away")
        return await real(self, sae_id)

    monkeypatch.setattr(SAEService, "detach_for_unload", flaky)
    seen: list = []
    real_release = unload_teardown.release_dependents

    async def keep(model_id: int, **kwargs: Any) -> Any:
        report = await real_release(model_id, **kwargs)
        seen.append(report)
        return report

    monkeypatch.setattr(unload_teardown, "release_dependents", keep)
    await world.service.unload_model(1)
    assert hooks(world.model) == 0 and AttachedSAEState().count == 0
    assert LoadedModelState().current is None
    (report,) = seen
    assert report.detached_saes == [f"{NAMES[SAE_B]} (id {SAE_B}) at layer 1"]
    assert any(f.startswith(f"sae {SAE_A}") for f in report.failures)


async def test_the_report_names_what_went(world, monkeypatch):
    seen: list = []
    real = unload_teardown.release_dependents

    async def keep(model_id: int, **kwargs: Any) -> Any:
        seen.append(await real(model_id, **kwargs))
        return seen[-1]

    monkeypatch.setattr(unload_teardown, "release_dependents", keep)
    await world.service.unload_model(1)
    (report,) = seen
    assert report.failures == []
    assert report.detached_saes == [f"{NAMES[SAE_A]} (id {SAE_A}) at layer 0",
                                    f"{NAMES[SAE_B]} (id {SAE_B}) at layer 1"]
    assert report.deactivated_circuits == [f"live circuit (id {CIRCUIT})"]
    assert report.deactivated_profiles == [f"calm cluster (id {CLUSTER}, a cluster)"]


async def test_an_unload_with_nothing_attached_changes_nothing_else(clean_state, db):  # noqa: F811
    """No SAE, circuit or profile: the teardown reads, finds nothing and writes nothing."""
    report = await unload_teardown.release_dependents(1)
    assert (report.detached_saes, report.deactivated_circuits, report.deactivated_profiles,
            report.failures) == ([], [], [], [])


def tiny_sae(seed: int, d_in: int) -> LoadedSAE:
    import torch

    from miserver.ml.sae_config import SAEConfig

    torch.manual_seed(seed)
    return LoadedSAE(W_enc=torch.randn(d_in, 8), b_enc=torch.zeros(8),
                     W_dec=torch.randn(8, d_in), b_dec=torch.zeros(d_in),
                     config=SAEConfig(d_in=d_in, d_sae=8, model_name="t", hook_name="t",
                                      hook_layer=0))


async def attach_on_resident(factory: Any, sae_id: str, layer: int) -> Any:
    """A real SAE hooked on the resident's real module, with its rows, as attach leaves them."""
    from miserver.ml.sae_hooker import SAEHooker

    resident = LoadedModelState().current
    sae = tiny_sae(layer + 3, resident.model.config.hidden_size)
    AttachedSAEState().set(sae, sae_id, layer, SAEHooker().install(resident.model, layer, sae))
    async with factory() as session:
        if await session.get(SAE, sae_id) is None:
            session.add(SAE(id=sae_id, repository_id=f"vendor/{sae_id}", name=f"{sae_id}-name",
                            d_in=sae.d_in, d_sae=8, cache_path=f"/data/saes/{sae_id}",
                            status=SAEStatus.ATTACHED))
        else:
            (await session.get(SAE, sae_id)).status = SAEStatus.ATTACHED
        session.add(SAEAttachment(sae_id=sae_id, model_id=resident.model_id, layer=layer,
                                  is_active=True))
        await session.commit()
    return resident.model


async def assert_detached(factory: Any, model: Any, sae_id: str) -> None:
    assert hooks(model) == 0 and AttachedSAEState().count == 0
    async with factory() as session:
        assert (await session.get(SAE, sae_id)).status == SAEStatus.CACHED
        rows = (await session.execute(select(SAEAttachment))).scalars().all()
    assert rows and not any(r.is_active for r in rows)


class TestWeightsFullReload:
    """The weights API's full reload unloads under the barrier: through `load_model`'s auto-unload
    when the checkpoint is a new carrier, and directly (`unload_model(under_barrier=True)`) when
    the resident IS the carrier. Both are driven here for real (U3b)."""

    async def test_both_full_reload_unloads_detach_the_saes(
        self, harness, weights_db, clean_state  # noqa: F811
    ):
        import torch

        from miserver.api.schemas.weights_checkpoints import LoadRequest
        from miserver.services import resident_identity
        from miserver.services.weights_checkpoint_service import WeightsCheckpointService
        from tests.unit.checkpoint_service_fixtures import (
            TRAINER,
            resident_llama,
            write_foreign_checkpoint,
        )
        from tests.unit.weights_fixtures import tiny_llama

        h = harness
        resident_llama()
        first = await attach_on_resident(weights_db, "sae_w", 0)
        write_foreign_checkpoint(h.root, "big", 1,
                                 tiny_llama(vocab_size=320, dtype=torch.bfloat16, seed=7))
        service = WeightsCheckpointService(h.models)
        result = await service.load(TRAINER, LoadRequest(path="big/latest"))
        assert result.mode == "full_reload"
        assert h.names() == ["load_model", "unload_model", "wait_until_loaded"]
        await assert_detached(weights_db, first, "sae_w")

        # Again, with the carrier resident: the weights service unloads it itself.
        second = await attach_on_resident(weights_db, "sae_w", 1)
        loaded = LoadedModelState().current
        resident_identity.set_identity(loaded, resident_identity.ResidentIdentity(
            name="moved", base_model="elsewhere", origin="checkpoint", checkpoint_dir=None,
            manifest_sha256=None, expected_parent=None, commits_at_identity=0))
        h.calls.clear()
        await service.load(TRAINER, LoadRequest(path="big/latest"))
        assert h.names() == ["unload_model", "load_model", "wait_until_loaded"]
        assert h.calls[0][1]["under_barrier"] is True
        await assert_detached(weights_db, second, "sae_w")


@pytest.fixture
def weights_db(test_session, monkeypatch):
    """The teardown's session factory on the checkpoint harness's database."""
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

    factory = async_sessionmaker(test_session.bind, class_=AsyncSession, expire_on_commit=False)
    monkeypatch.setattr("miserver.db.base.async_session_factory", factory)
    return factory
