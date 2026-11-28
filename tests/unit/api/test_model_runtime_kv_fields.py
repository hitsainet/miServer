"""The model record says which KV cache and window the loaded GGUF model really got (2026-11-21).

Model 17 served an f16 cache for days with nothing on the record to say so: the only trace was a
`gguf_retrying_with_f16_kv_cache` line in the backend log.

MUTATION CONTROLS (recorded in 0xcc/reviews/gguf_context_ceiling_2026-11-21.md):
  S2 `_with_runtime` drops kv_cache_type             -> test_the_record_carries_*
  S3 `get_loaded_model_info` drops gguf_load          -> test_the_service_reports_*
  S4 `_with_runtime` drops supports_embeddings        -> test_the_record_carries_*
     (the models API page promised this field on model records; no route returned it)
"""

from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock

from miserver.api.routes.management.models import _with_runtime
from miserver.api.schemas.model import ModelResponse
from miserver.ml.model_loader import ENGINE_LLAMACPP, LoadedModel
from miserver.services.model_service import ModelService
from tests.support.factories import make_gguf_model

DECISIONS = {"kv_cache_type": "q8_0", "kv_cache_reason": "q8_0: GGUF_KV_CACHE_TYPE",
             "embeddings": False, "context": {"ceiling": "max", "loaded": 155648}}


def _service_with_resident() -> ModelService:
    loader = MagicMock()
    loader.is_loaded = True
    loader.state.current = LoadedModel(
        1, "m", MagicMock(), None, datetime(2026, 11, 21), memory_used_mb=16038,
        device="cuda:0", gpu_indices=[0], memory_by_device_mb={"cuda:0": 19206},
        engine=ENGINE_LLAMACPP, context_length=155648, kv_cache_type="q8_0",
        load_decisions=DECISIONS)
    return ModelService(repository=MagicMock(), downloader=MagicMock(), loader=loader,
                        emitter=None)


def test_the_service_reports_the_window_the_cache_and_the_decisions():
    info = _service_with_resident().get_loaded_model_info()
    assert info["context_length"] == 155648
    assert info["kv_cache_type"] == "q8_0"
    assert info["gguf_load"] == DECISIONS
    assert info["supports_embeddings"] is False


def test_the_record_carries_them_for_the_loaded_model_only():
    info = _service_with_resident().get_loaded_model_info()
    row = make_gguf_model(id=1, context_ceiling=0)
    loaded = _with_runtime(ModelResponse.from_model(row), info)
    assert (loaded.context_ceiling, loaded.context_length, loaded.kv_cache_type) == (0, 155648, "q8_0")
    assert loaded.gguf_load == DECISIONS
    assert loaded.supports_embeddings is False
    other = _with_runtime(ModelResponse.from_model(make_gguf_model(id=2)), info)
    assert (other.context_length, other.kv_cache_type, other.gguf_load,
            other.supports_embeddings) == (None, None, None, None)
    assert other.context_ceiling is None
