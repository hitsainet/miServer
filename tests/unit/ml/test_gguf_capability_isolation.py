"""A ladder that fails at every length isolates its cause before it changes the KV cache.

Live, 2026-11-21: model 17 (Qwen3.6-27B Q4_K_M) on the RTX 3090. The q8_0 ladder WITH embeddings
failed at every length, was logged `gguf_retrying_with_f16_kv_cache` ("this build or
architecture refuses it"), the f16 ladder with embeddings failed too, and only then were
embeddings dropped. It loaded at 32768 with an f16 cache: twice the memory a token, because the
cause was embeddings (a hybrid model's recurrent state is held per sequence, and llama-cpp-python
sizes it for 256 sequences when `embedding=True`), never the cache type.

MUTATION CONTROLS (each must turn this file red; run 2026-11-21, recorded in
0xcc/reviews/gguf_context_ceiling_2026-11-21.md):
  I1 try the f16 candidate before the no-embeddings one         -> pooling-refused tests
  I2 drop BOTH capabilities on the first isolation probe         -> pooling-refused and cache-refused tests
  I3 skip the recurrent-state pre-decision                       -> TestEmbeddingsAreDecidedFirst
  I4 pre-decide on every model (no recurrent-state condition)    -> test_a_dense_model_keeps_embeddings
  I5 record GGUF_KV_CACHE_TYPE instead of the type that loaded   -> test_the_cache_type_that_loaded_is_recorded
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from miserver.core.config import settings
from miserver.core.errors import ModelLoadError
from miserver.ml import model_loader
from tests.support.fake_gpus import RTX_3090, fake_gpus
from tests.support.gguf_headers import GEMMA4_31B, QWEN36_27B, write_gguf


def _llama(calls, refuse):
    """A Llama stand-in: refuses (context failure) any construction `refuse(kwargs)` names."""

    def _factory(**kwargs):
        calls.append(dict(kwargs))
        if refuse(kwargs):
            raise ValueError("Failed to create llama_context")
        return MagicMock(close=MagicMock())

    return _factory


def _refuses_pooling(kwargs):
    return bool(kwargs.get("embedding")) or kwargs.get("pooling_type") is not None


def _refuses_q8(kwargs):
    return "type_k" in kwargs


def _load(tmp_path, refuse, calls, *, embeddings=True, ceiling=8192):
    (tmp_path / "m.gguf").write_bytes(b"x" * 32)  # no readable header: no pre-decision
    with patch.object(model_loader, "Llama", _llama(calls, refuse)), \
         patch.object(model_loader, "declared_context", return_value=8192), \
         patch.object(settings, "GGUF_ENABLE_EMBEDDINGS", embeddings), \
         patch.object(settings, "GGUF_KV_CACHE_TYPE", "q8_0"), \
         patch.object(settings, "GGUF_FLASH_ATTENTION", True), \
         patch.object(settings, "GGUF_CONTEXT_LENGTH", ceiling):
        return model_loader.load_gguf_model(1, "m", str(tmp_path), "m.gguf")


class TestThePoolingRefusalIsNotBlamedOnTheCache:
    def test_it_loads_q8_0_without_embeddings(self, tmp_path):
        calls: list[dict] = []
        loaded = _load(tmp_path, _refuses_pooling, calls)
        assert loaded.supports_embeddings is False
        assert loaded.kv_cache_type == "q8_0"
        assert "type_k" in calls[-1], "the served context dropped the quantized cache"
        assert loaded.context_length == 8192

    def test_no_f16_cache_is_ever_tried(self, tmp_path):
        """The old order walked a whole f16 ladder first; one probe now names the cause."""
        calls: list[dict] = []
        _load(tmp_path, _refuses_pooling, calls)
        assert all("type_k" in c for c in calls), [("type_k" in c, c["n_ctx"]) for c in calls]
        # 3 rungs with embeddings, 1 isolation probe at 2048, 1 load at 8192.
        assert [c["n_ctx"] for c in calls] == [8192, 4096, 2048, 2048, 8192]

    def test_it_says_why(self, tmp_path):
        loaded = _load(tmp_path, _refuses_pooling, [])
        decisions = loaded.load_decisions
        assert decisions["kv_cache_type"] == "q8_0"
        assert decisions["kv_cache_reason"] == "q8_0: GGUF_KV_CACHE_TYPE"
        assert decisions["embeddings"] is False
        assert "without them it loaded at 2048" in decisions["embeddings_reason"]
        assert decisions["isolation"] == [
            {"tried": "q8_0 cache, embeddings disabled", "n_ctx": 2048, "loaded": True}]


class TestARefusedCacheKeepsEmbeddings:
    def test_it_loads_f16_with_embeddings(self, tmp_path):
        calls: list[dict] = []
        loaded = _load(tmp_path, _refuses_q8, calls)
        assert loaded.kv_cache_type == "f16"
        assert loaded.supports_embeddings is True
        assert calls[-1].get("embedding") is True and "type_k" not in calls[-1]
        assert "an f16 cache loaded at 2048" in loaded.load_decisions["kv_cache_reason"]
        assert loaded.load_decisions["isolation"] == [
            {"tried": "q8_0 cache, embeddings disabled", "n_ctx": 2048, "loaded": False},
            {"tried": "f16 cache, embeddings on", "n_ctx": 2048, "loaded": True},
        ]

    def test_both_refused_drops_both(self, tmp_path):
        loaded = _load(tmp_path, lambda k: _refuses_q8(k) or _refuses_pooling(k), [])
        assert (loaded.kv_cache_type, loaded.supports_embeddings) == ("f16", False)

    def test_with_embeddings_off_a_refused_cache_goes_to_f16(self, tmp_path):
        loaded = _load(tmp_path, _refuses_q8, [], embeddings=False)
        assert loaded.kv_cache_type == "f16"
        assert loaded.load_decisions["embeddings_reason"] == "off: GGUF_ENABLE_EMBEDDINGS is false"

    def test_the_cache_type_that_loaded_is_recorded(self, tmp_path):
        loaded = _load(tmp_path, lambda k: False, [])
        assert loaded.kv_cache_type == "q8_0"
        loaded = _load(tmp_path, _refuses_q8, [])
        assert loaded.kv_cache_type == "f16"


class TestNothingLoads:
    def test_the_message_lists_every_isolation_probe(self, tmp_path):
        with pytest.raises(ModelLoadError) as raised:
            _load(tmp_path, lambda k: True, [])
        message = str(raised.value)
        assert "q8_0 cache, embeddings disabled at 2048: failed" in message
        assert "f16 cache, embeddings on at 2048: failed" in message
        assert "f16 cache, embeddings disabled at 2048: failed" in message
        assert "may not fit on this GPU" not in message


@pytest.fixture
def offload():
    with patch.object(model_loader, "llama_supports_gpu_offload", lambda: True):
        yield


class TestEmbeddingsAreDecidedFirst:
    """A hybrid model's embeddings need 256 sequences of recurrent state: decided from the header."""

    def _load(self, tmp_path, meta, calls, free=24_100, weights_mb=16_038):
        path = write_gguf(tmp_path / "m-Q4_K_M.gguf", meta, weights_mb=weights_mb)
        with fake_gpus((RTX_3090, free, 24_576)), \
             patch.object(model_loader, "Llama", _llama(calls, _refuses_pooling)), \
             patch.object(model_loader, "declared_context", return_value=262_144), \
             patch.object(settings, "GGUF_ENABLE_EMBEDDINGS", True), \
             patch.object(settings, "GGUF_KV_CACHE_TYPE", "q8_0"), \
             patch.object(settings, "GGUF_FLASH_ATTENTION", True), \
             patch.object(settings, "GGUF_CONTEXT_LENGTH", 32_768):
            return model_loader.load_gguf_model(17, "m", str(tmp_path), path.name)

    def test_model_17_loads_on_the_first_attempt_at_q8_0(self, tmp_path, offload):
        calls: list[dict] = []
        loaded = self._load(tmp_path, QWEN36_27B, calls)
        assert len(calls) == 1, [c["n_ctx"] for c in calls]
        assert calls[0]["n_ctx"] == 32_768, "the server default ceiling still governs"
        assert "embedding" not in calls[0] and "type_k" in calls[0]
        assert loaded.kv_cache_type == "q8_0"
        reason = loaded.load_decisions["embeddings_reason"]
        assert "256 parallel sequences (38,304 MiB)" in reason, reason

    def test_a_dense_model_keeps_embeddings(self, tmp_path, offload):
        """Specificity: gemma has no recurrent state; embeddings are tried as configured."""
        calls: list[dict] = []
        self._load(tmp_path, GEMMA4_31B, calls, weights_mb=4_000)
        assert calls[0].get("embedding") is True
