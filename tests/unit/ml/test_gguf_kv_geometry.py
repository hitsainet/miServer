"""The KV cache a GGUF model really costs, from the file's own header (2026-11-21).

Model 17, Qwen3.6-27B Q4_K_M (arch `qwen35`), loaded on the RTX 3090 with a prediction of 40,960
tokens at q8_0, and at f16 held 32,768 with 19,206 MiB in use. The predictor priced a token as

    2 x n_layer x n_head_kv x (n_embd // n_head) = 2 x 64 x 4 x 213 elements,

but `qwen35` is a hybrid: one layer in four (`full_attention_interval` 4) holds a KV cache, the
other 48 hold a fixed recurrent state, and the head width is `key_length` 256. The true figure is
16 x 4 x (256 + 256) = 32,768 elements, 3.33x less. Gemma-4 had the opposite error (28% low:
5376 // 32 = 168 where its layers are 256 and 512 wide), which the old 2048 MiB overhead absorbed.

MUTATION CONTROLS (each must turn this file red; run 2026-11-21, recorded in
0xcc/reviews/gguf_context_ceiling_2026-11-21.md):
  G1 ignore full_attention_interval (every layer attention)     -> the qwen35 geometry tests
  G2 ignore key_length (n_embd // n_head)                        -> the qwen35 and gemma geometry tests
  G3 ignore the _swa widths                                      -> test_gemma4_..._per_layer
  G4 gguf_kv_bytes_per_token skips the header (llama probe only) -> test_the_loader_prices_..._from_the_header
  G5 predicted_max_context ignores fixed_mb                      -> test_the_recurrent_state_is_budgeted
"""

from __future__ import annotations

import pytest

from miserver.ml import model_loader
from miserver.ml.gguf_metadata import (
    GgufMetadataError,
    gguf_kv_geometry,
    kv_geometry,
    parse_gguf_metadata,
    read_gguf_metadata,
)
from tests.support.gguf_headers import (
    GEMMA4_31B,
    QWEN36_27B,
    QWEN36_27B_F16_32K_IN_USE_MB,
    QWEN36_27B_WEIGHTS_MB,
    gguf_header_bytes,
    write_gguf,
)

CARD_MB = 24_576


class TestTheHeaderIsRead:
    def test_round_trip_including_per_layer_arrays(self, tmp_path):
        path = write_gguf(tmp_path / "g.gguf", GEMMA4_31B)
        meta = read_gguf_metadata(str(path))
        assert meta["gemma4.attention.head_count_kv"] == GEMMA4_31B["gemma4.attention.head_count_kv"]
        assert meta["gemma4.attention.sliding_window_pattern"][5] is False
        assert meta["general.architecture"] == "gemma4"

    def test_a_long_array_is_stepped_over_and_what_follows_still_reads(self):
        meta = {"general.architecture": "x", "tokenizer.ggml.tokens": ["t"] * 5000,
                "x.block_count": 3}
        parsed = parse_gguf_metadata(gguf_header_bytes(meta))
        assert parsed["tokenizer.ggml.tokens"] is None
        assert parsed["x.block_count"] == 3

    def test_not_a_gguf_file(self, tmp_path):
        (tmp_path / "m.gguf").write_bytes(b"x" * 32)
        with pytest.raises(GgufMetadataError):
            read_gguf_metadata(str(tmp_path / "m.gguf"))
        assert gguf_kv_geometry(str(tmp_path / "m.gguf")) is None

    def test_a_cut_header_is_refused_not_misread(self):
        raw = gguf_header_bytes(QWEN36_27B)
        with pytest.raises(GgufMetadataError):
            parse_gguf_metadata(raw[: len(raw) // 2])


class TestTheGeometry:
    def test_qwen35_caches_one_layer_in_four_at_its_real_head_width(self):
        g = kv_geometry(QWEN36_27B)
        assert (g.kv_layers, g.recurrent_layers) == (16, 48)
        assert g.kv_elements_per_token == 16 * 4 * (256 + 256)
        assert g.bytes_per_token("f16") == 65_536
        assert g.bytes_per_token("q8_0") == 34_816

    def test_qwen35_recurrent_state_per_sequence(self):
        """48 layers x ((4-1) x (6144 + 2 x 16 x 128) + 128 x 6144) f32 cells = 149.6 MiB."""
        g = kv_geometry(QWEN36_27B)
        assert g.recurrent_state_bytes_per_seq == 48 * (3 * 10_240 + 786_432) * 4
        assert round(g.recurrent_state_mb(1), 1) == 149.6

    def test_gemma4_sliding_and_global_layers_are_priced_per_layer(self):
        g = kv_geometry(GEMMA4_31B)
        assert g.kv_layers == 60 and g.recurrent_layers == 0
        assert g.kv_elements_per_token == 50 * 16 * 512 + 10 * 4 * 1024
        assert g.recurrent_state_bytes_per_seq == 0

    def test_a_dense_model_without_lengths_uses_llama_cpps_default_width(self):
        meta = {"general.architecture": "llama", "llama.block_count": 32,
                "llama.attention.head_count": 32, "llama.attention.head_count_kv": 8,
                "llama.embedding_length": 4096}
        assert kv_geometry(meta).kv_elements_per_token == 32 * 8 * (128 + 128)

    def test_zero_kv_heads_mark_the_recurrent_layers_of_an_array_hybrid(self):
        meta = {"general.architecture": "jamba", "jamba.block_count": 4,
                "jamba.attention.head_count": 8, "jamba.attention.head_count_kv": [0, 0, 0, 2],
                "jamba.embedding_length": 1024}
        g = kv_geometry(meta)
        assert (g.kv_layers, g.recurrent_layers) == (1, 3)
        assert g.kv_elements_per_token == 2 * (128 + 128)

    def test_metadata_that_does_not_give_the_terms_is_none(self):
        assert kv_geometry({}) is None
        assert kv_geometry({"general.architecture": "x", "x.block_count": 4,
                            "x.attention.head_count": 8,
                            "x.attention.head_count_kv": [1, 1]}) is None


class TestThePredictorOnModel17:
    """Hardware: f16 @ 32768 LOADED with 19,206 MiB in use (16,038 of it weights)."""

    def _predict(self, kv, free=CARD_MB):
        g = kv_geometry(QWEN36_27B)
        return model_loader.predicted_max_context(
            "unused.gguf", kv, free, QWEN36_27B_WEIGHTS_MB,
            bytes_per_token=g.bytes_per_token(kv), fixed_mb=g.recurrent_state_mb(1),
        )

    def test_it_predicts_the_window_that_demonstrably_loaded(self):
        assert self._predict("f16") >= 32_768, (
            "f16 at 32768 loaded on this card; the old formula predicted 21,504"
        )

    def test_the_plan_at_the_measured_point_covers_what_was_measured(self):
        """Planning must not claim the 32k load needed less than it used."""
        g = kv_geometry(QWEN36_27B)
        planned = (QWEN36_27B_WEIGHTS_MB + g.bytes_per_token("f16") * 32_768 / 2**20
                   + g.recurrent_state_mb(1) + model_loader._GGUF_RUNTIME_OVERHEAD_MB)
        assert QWEN36_27B_F16_32K_IN_USE_MB <= planned <= QWEN36_27B_F16_32K_IN_USE_MB * 1.05

    def test_q8_0_predicts_about_twice_f16(self):
        assert 1.8 < self._predict("q8_0") / self._predict("f16") < 2.0

    def test_the_recurrent_state_is_budgeted(self):
        g = kv_geometry(QWEN36_27B)
        without = model_loader.predicted_max_context(
            "unused.gguf", "f16", CARD_MB, QWEN36_27B_WEIGHTS_MB,
            bytes_per_token=g.bytes_per_token("f16"))
        assert self._predict("f16") < without


class TestTheLoaderPricesFromTheHeader:
    def test_the_loader_prices_a_token_from_the_header(self, tmp_path):
        """REACHABILITY: the function the loader calls reads the file, without llama.cpp."""
        path = write_gguf(tmp_path / "q.gguf", QWEN36_27B)
        assert model_loader.gguf_kv_bytes_per_token(str(path), "q8_0") == 34_816
        assert model_loader.gguf_kv_bytes_per_token(str(path), "f16") == 65_536

    def test_an_unreadable_header_falls_back_without_failing(self, tmp_path):
        from unittest.mock import patch

        (tmp_path / "m.gguf").write_bytes(b"x" * 32)
        with patch.object(model_loader, "llama_cpp_module", None):
            assert model_loader.gguf_kv_bytes_per_token(str(tmp_path / "m.gguf"), "q8_0") is None
