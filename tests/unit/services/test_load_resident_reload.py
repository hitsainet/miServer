"""A load that names the RESIDENT model: a reload, or a refusal that changes nothing (2026-11-21).

Live defect: model 17 (Qwen3.6-27B GGUF) was served at 32,768 tokens. The operator approved its
ceiling to max, an agent filed `POST /api/models/17/load`, the card promised "Unloads first:
nothing (this model is already loaded)" and "Context ceiling: max", and the approved load could
only fail: `_refuse_load_status` refused every LOADED row ("Model 17 is already loaded").

Now (`services/resident_reload.py`, applied by `ModelService.load_model`): a load of the resident
that changes its ceiling or its card unloads it (draining, ending its lease, as an unload does) and
loads it again, ONE request; one that changes nothing is refused 409 MODEL_ALREADY_LOADED with a
fix, before anything moves. These run the REAL ModelService over a fake repository
(`lease_fixtures`); one test runs the real load worker and loader down to `Llama(n_ctx=...)`.

Fixtures disagree with the defect: the resident is loaded at 32,768 with the server default at
32,768, and the reload asks for a DIFFERENT value (max, or 65,536), so a reload that kept the old
ceiling, or a comparison against the wrong field, cannot pass.

MUTATION CONTROLS (each must turn this file red; run 2026-11-21, recorded in
0xcc/reviews/load_resident_reload_2026-11-21.md §5):
  M1 the reload branch removed (`reloading = True` dropped)  -> TestReload (refused as already loaded)
  M2 the no-op check removed (`if not plan.reload: raise`)   -> TestNoChange (a reload is started)
  M3 compare() reads context_length, not the recorded ceiling -> test_a_ceiling_that_fitted_*
  M4 a placement request is never a change                    -> TestPlacement
"""

from __future__ import annotations

import asyncio
from contextlib import ExitStack
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from miserver.core.config import settings
from miserver.core.errors import ModelAlreadyLoadedError, ModelLeasedError
from miserver.db.models.model import ModelStatus
from miserver.ml.model_loader import LoadedModel, LoadedModelState, ModelLoader
from tests.support.factories import make_gguf_model, make_model
from tests.unit.lease_fixtures import (
    FakeRepo,
    clear_resident,
    drain_executor,
    install_registry,
    no_probe_rows,
)

SERVER_DEFAULT = 32_768


@pytest.fixture(autouse=True)
def _clean_loader():
    clear_resident()
    with patch.object(settings, "GGUF_CONTEXT_LENGTH", SERVER_DEFAULT):
        yield
    clear_resident()


def _resident(model_id: int = 1, *, ceiling=SERVER_DEFAULT, window=SERVER_DEFAULT,
              gpu_indices=(0,), requested=None) -> None:
    LoadedModelState().set(LoadedModel(
        model_id=model_id, model_name="Qwen3.6-27B-GGUF:Q4_K_M", model=SimpleNamespace(),
        tokenizer=None, loaded_at=datetime(2026, 11, 21), context_length=window,
        gpu_indices=list(gpu_indices), placement={"requested": requested},
        load_decisions={"context": {"ceiling": ceiling, "ceiling_source": "server",
                                    "loaded": window}},
    ))


def _service(*, row1=None, locked=False):
    from miserver.services.model_service import ModelService

    m1 = row1 or make_gguf_model(id=1, name="Qwen3.6-27B-GGUF:Q4_K_M",
                                 status=ModelStatus.LOADED, locked=locked)
    m2 = make_model(id=2, name="m2", status=ModelStatus.READY)
    repo = FakeRepo(m1, m2)
    inference = MagicMock()
    inference.request_queue.pending_count = 0
    # One request in flight: the reload's unload must wait for it, as an unload does.
    inference.request_queue.occupied_count = 1
    inference.request_queue.wait_idle = AsyncMock(return_value=True)
    service = ModelService(repository=repo, downloader=MagicMock(), loader=ModelLoader(),
                           inference_service=inference)
    service._precheck_placement = MagicMock()  # type: ignore[method-assign]
    service._load_worker = MagicMock()  # type: ignore[method-assign]
    return service, repo, inference


def _submitted(service) -> dict:
    drain_executor(service)
    assert service._load_worker.call_count == 1
    return service._load_worker.call_args.kwargs


class TestReload:
    async def test_a_new_ceiling_unloads_then_loads_with_it(self):
        install_registry()
        service, repo, inference = _service()
        _resident()
        with no_probe_rows():
            await service.load_model(1, context_ceiling=0)
        kwargs = _submitted(service)
        assert (kwargs["context_ceiling"], kwargs["ceiling_source"]) == (0, "request")
        # Unloaded FIRST (row READY, residency gone), then marked LOADING for the new load.
        assert repo.status_writes == [(1, ModelStatus.READY), (1, ModelStatus.LOADING)]
        assert LoadedModelState().current is None
        # ...after waiting for the request in flight, as an unload does.
        inference.request_queue.wait_idle.assert_awaited_once()

    async def test_the_models_own_new_ceiling_is_a_reload_too(self):
        """The live sequence: the ceiling was SET on the row (approved), then a bare load."""
        install_registry()
        row = make_gguf_model(id=1, name="Qwen3.6-27B-GGUF:Q4_K_M", status=ModelStatus.LOADED,
                              context_ceiling=0)
        service, repo, _ = _service(row1=row)
        _resident()
        with no_probe_rows():
            await service.load_model(1)
        kwargs = _submitted(service)
        assert (kwargs["context_ceiling"], kwargs["ceiling_source"]) == (0, "model")
        assert repo.status_writes[0] == (1, ModelStatus.READY)

    async def test_a_locked_resident_is_reloaded_and_its_lock_released_as_by_an_unload(self):
        """The steering lock keeps its rule: the management load does not read it (tracked
        debt in the manual), and the unload releases it."""
        install_registry()
        service, repo, _ = _service(locked=True)
        _resident()
        with no_probe_rows():
            await service.load_model(1, context_ceiling=65_536)
        _submitted(service)
        assert repo.rows[1].locked is False

    async def test_the_new_ceiling_reaches_n_ctx(self, tmp_path):
        """End to end: the real worker, the real loader, `Llama(n_ctx=...)` stood in."""
        from miserver.ml import model_loader
        from tests.unit.ml.test_gguf_context_ceiling import _Cards

        install_registry()
        (tmp_path / "zora-v1.13-Q5_K_M.gguf").write_bytes(b"x" * 32)
        row = make_gguf_model(id=1, status=ModelStatus.LOADED, cache_path=str(tmp_path))
        service, repo, _ = _service(row1=row)
        del service._load_worker  # the real one
        _resident()
        cards = _Cards()
        with ExitStack() as stack:
            for p in (patch.object(model_loader, "Llama", cards),
                      patch.object(model_loader, "declared_context", return_value=262_144),
                      patch.object(settings, "GGUF_ENABLE_EMBEDDINGS", False),
                      patch.object(settings, "GGUF_KV_CACHE_TYPE", "q8_0"),
                      patch.object(settings, "GGUF_FLASH_ATTENTION", True),
                      no_probe_rows()):
                stack.enter_context(p)
            await service.load_model(1, context_ceiling=65_536)
            for _ in range(200):
                if repo.rows[1].status in (ModelStatus.LOADED, ModelStatus.ERROR):
                    break
                await asyncio.sleep(0.02)
        assert repo.rows[1].status == ModelStatus.LOADED, repo.rows[1].error_message
        assert cards.calls == [65_536]
        loaded = LoadedModelState().current
        assert loaded.model_id == 1 and loaded.context_length == 65_536
        assert loaded.load_decisions["context"]["ceiling"] == 65_536


class TestNoChange:
    @pytest.mark.parametrize("kwargs", [{}, {"context_ceiling": SERVER_DEFAULT}, {"gpu": "auto"},
                                        {"gpu": 0}])
    async def test_a_load_that_changes_nothing_is_refused_and_moves_nothing(self, kwargs):
        install_registry()
        service, repo, inference = _service()
        _resident()
        with pytest.raises(ModelAlreadyLoadedError) as exc:
            await service.load_model(1, **kwargs)
        assert exc.value.status_code == 409
        details = exc.value.details
        assert details["loaded_ceiling"] == SERVER_DEFAULT
        assert "context-ceiling" in details["fix"] and '"max"' in details["fix"]
        assert "32,768-token window" in exc.value.message
        assert service._load_worker.call_count == 0
        assert repo.status_writes == []
        assert service._loading_model_id is None
        assert LoadedModelState().current.model_id == 1
        inference.request_queue.wait_idle.assert_not_awaited()

    async def test_max_asked_again_of_a_model_loaded_at_max_is_no_change(self):
        install_registry()
        service, _, _ = _service()
        _resident(ceiling="max", window=131_072)
        with pytest.raises(ModelAlreadyLoadedError):
            await service.load_model(1, context_ceiling=0)

    async def test_a_ceiling_that_fitted_smaller_asked_again_is_no_change(self):
        """65,536 asked, 40,960 fitted: the RECORDED ceiling is compared, not the window."""
        install_registry()
        service, _, _ = _service()
        _resident(ceiling=65_536, window=40_960)
        with pytest.raises(ModelAlreadyLoadedError):
            await service.load_model(1, context_ceiling=65_536)
        assert service._load_worker.call_count == 0


class TestPlacement:
    async def test_another_card_is_a_reload_on_it(self):
        install_registry()
        service, repo, _ = _service()
        _resident(gpu_indices=(0,))
        with no_probe_rows():
            await service.load_model(1, gpu=1)
        drain_executor(service)
        assert service._load_worker.call_count == 1
        assert service._load_worker.call_args.args[-1] == 1  # the card handed to the worker
        assert repo.status_writes[0] == (1, ModelStatus.READY)

    async def test_all_on_a_model_not_split_is_a_reload(self):
        install_registry()
        service, _, _ = _service()
        _resident(gpu_indices=(0,))
        with no_probe_rows():
            await service.load_model(1, gpu="all")
        drain_executor(service)
        assert service._load_worker.call_count == 1

    async def test_all_on_a_model_split_as_requested_is_no_change(self):
        install_registry()
        service, _, _ = _service()
        _resident(gpu_indices=(0, 1), requested="all")
        with pytest.raises(ModelAlreadyLoadedError):
            await service.load_model(1, gpu="all")


class TestTheOtherRulesStillApply:
    async def test_a_foreign_lease_refuses_the_reload_before_anything_moves(self):
        install_registry()
        service, repo, _ = _service()
        _resident()
        await service.acquire_lease(1, "midataworks", "label run 7", 120)
        with pytest.raises(ModelLeasedError):
            await service.load_model(1, context_ceiling=0)
        assert service._load_worker.call_count == 0
        assert repo.status_writes == []
        assert LoadedModelState().current.model_id == 1

    async def test_the_holders_reload_proceeds_and_ends_its_lease(self):
        registry, _ = install_registry()
        service, _, _ = _service()
        _resident()
        grant = await service.acquire_lease(1, "midataworks", "label run 7", 120)
        with no_probe_rows():
            await service.load_model(1, context_ceiling=0, lease_id=grant.lease_id)
        _submitted(service)
        assert registry.current(1) is None

    async def test_a_load_of_another_model_is_unchanged(self):
        install_registry()
        service, repo, _ = _service()
        _resident()
        with no_probe_rows():
            await service.load_model(2)
        drain_executor(service)
        assert service._load_worker.call_count == 1
        assert repo.status_writes == [(1, ModelStatus.READY), (2, ModelStatus.LOADING)]

    async def test_a_stale_loaded_row_not_resident_is_still_refused(self):
        """A LOADED row the loader does not hold is not the resident: the old refusal."""
        install_registry()
        service, _, _ = _service()
        _resident(model_id=2)
        with pytest.raises(ModelAlreadyLoadedError):
            await service.load_model(1, context_ceiling=0)
