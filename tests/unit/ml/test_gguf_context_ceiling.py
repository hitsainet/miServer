"""A GGUF model's context ceiling: per load, per model, or the server default (2026-11-21).

GGUF_CONTEXT_LENGTH (32768) capped every GGUF model, whatever it declared and whatever the card
held: model 17 (Qwen3.6-27B, declares 262144) served 32768 on a 24 GiB card that holds several
times that at q8_0. The ceiling now comes from the load request, else the model row, else the
server (`services/context_ceiling.py`), and 0 / "max" is as large as fits up to the declared
length: start at the predicted fit, walk down if that fails, probe up if it loads.

MUTATION CONTROLS (each must turn this file red; run 2026-11-21, recorded in
0xcc/reviews/gguf_context_ceiling_2026-11-21.md):
  C1 the loader ignores context_ceiling (always GGUF_CONTEXT_LENGTH) -> TestTheCeilingReachesNCtx
  C2 the worker drops context_ceiling when calling the loader         -> test_the_worker_forwards_*
  C3 load_model submits the GGUF_CONTEXT_LENGTH default, not the row  -> test_the_row_ceiling_*
  C4 the request wins over nothing (row first)                        -> test_the_request_wins
  C5 no upward probe for max                                          -> test_max_probes_above_*
  C6 a probe made with the working instance still open                -> test_*_one_instance_at_a_time
  C7 the request body's context_ceiling dropped from the route call   -> test_the_route_forwards_*
"""

from __future__ import annotations

import asyncio
from contextlib import ExitStack
from concurrent.futures import Future
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from miserver.core.config import settings
from miserver.core.errors import ValidationError
from miserver.ml import model_loader
from miserver.services import context_ceiling
from tests.support.fake_gpus import RTX_3090, fake_gpus
from tests.support.gguf_headers import QWEN36_27B, write_gguf


class _Cards:
    """A Llama stand-in that holds weights like the real one: one open instance at a time fits."""

    def __init__(self, limit: int | None = None):
        self.limit = limit
        self.calls: list[int] = []
        self.trials: list[int] = []
        self.open = 0
        self.most_open = 0

    def __call__(self, **kwargs):
        self.calls.append(kwargs["n_ctx"])
        if self.open:
            # A second copy of 16 GiB of weights does not fit next to the first.
            raise ValueError("Failed to load model from file: /models/m.gguf")
        if self.limit is not None and kwargs["n_ctx"] > self.limit:
            raise ValueError("Failed to create llama_context")
        self.open += 1
        self.most_open = max(self.most_open, self.open)
        instance = MagicMock()

        def _close():
            self.open -= 1

        instance.close = MagicMock(side_effect=_close)
        return instance

    def decode_check(self, kwargs, n_tokens, measure, **_):
        """`gguf_decode_check.run_decode_check`'s stand-in: a child passes where a context fits.

        The server must hold NO instance meanwhile: the child loads its own copy of the weights.
        """
        from miserver.ml.gguf_decode_check import DecodeCheckResult

        n_ctx = kwargs["n_ctx"]
        self.trials.append(n_ctx)
        fits = not self.open and (self.limit is None or n_ctx <= self.limit)
        return DecodeCheckResult(n_ctx=n_ctx, tokens=n_tokens, created=fits, decode_ok=fits,
                                 exit_code=0, free_mb=dict(measure()) if fits else {})


def _load(tmp_path, cards, *, ceiling=None, server=32_768, declared=262_144, header=False,
          gpus=None, source="request"):
    if header:
        path = write_gguf(tmp_path / "m-Q4_K_M.gguf", QWEN36_27B, weights_mb=16_038)
    else:
        path = tmp_path / "m-Q4_K_M.gguf"
        path.write_bytes(b"x" * 32)
    patches = [
        patch.object(model_loader, "Llama", cards),
        patch.object(model_loader.gguf_decode_check, "run_decode_check", cards.decode_check),
        patch.object(model_loader, "declared_context", return_value=declared),
        patch.object(settings, "GGUF_ENABLE_EMBEDDINGS", False),
        patch.object(settings, "GGUF_KV_CACHE_TYPE", "q8_0"),
        patch.object(settings, "GGUF_FLASH_ATTENTION", True),
        patch.object(settings, "GGUF_CONTEXT_LENGTH", server),
    ]
    if gpus:
        patches += [fake_gpus(*gpus),
                    patch.object(model_loader, "llama_supports_gpu_offload", lambda: True)]
    with ExitStack() as stack:
        for p in patches:
            stack.enter_context(p)
        return model_loader.load_gguf_model(
            17, "m", str(tmp_path), path.name, context_ceiling=ceiling, ceiling_source=source)


class TestTheCeilingReachesNCtx:
    def test_a_named_ceiling_is_the_first_n_ctx(self, tmp_path):
        cards = _Cards()
        loaded = _load(tmp_path, cards, ceiling=65_536)
        assert cards.calls == [65_536]
        assert loaded.context_length == 65_536
        assert loaded.load_decisions["context"]["ceiling"] == 65_536
        assert loaded.load_decisions["context"]["ceiling_source"] == "request"

    def test_no_ceiling_is_the_server_default_unchanged(self, tmp_path):
        cards = _Cards()
        loaded = _load(tmp_path, cards, ceiling=None, source="server")
        assert cards.calls == [32_768]
        assert loaded.load_decisions["context"] == {
            "ceiling": 32_768, "ceiling_source": "server", "declared": 262_144,
            "predicted_fit": None, "started_at": 32_768, "loaded": 32_768}

    def test_a_ceiling_above_the_declared_length_is_bounded_by_it(self, tmp_path):
        cards = _Cards()
        _load(tmp_path, cards, ceiling=300_000, declared=131_072)
        assert cards.calls == [131_072]


class TestMax:
    GPU = ((RTX_3090, 24_100, 24_576),)

    def test_max_starts_at_the_predicted_fit_for_the_card(self, tmp_path):
        cards = _Cards(limit=142_336)
        loaded = _load(tmp_path, cards, ceiling=0, header=True, gpus=self.GPU)
        # 24,100 x 0.94 - 16,038 - 150 MiB of state = 6,466 MiB. The floor: 6,466 - 1,280 at
        # 34,816 B a token = 155,648. The window's own compute term (2026-11-21, after the
        # crash): 6,466 - 1,024 at 34,816 + 5,120 B a token = 142,336, the tighter of the two.
        assert cards.calls[0] == 142_336
        assert loaded.load_decisions["context"]["predicted_fit"] == 142_336
        assert loaded.load_decisions["context"]["ceiling"] == "max"
        assert loaded.kv_cache_type == "q8_0"

    def test_max_probes_above_a_prediction_that_loaded(self, tmp_path):
        cards = _Cards(limit=180_000)
        loaded = _load(tmp_path, cards, ceiling=0, header=True, gpus=self.GPU)
        assert 142_336 < loaded.context_length <= 180_000, cards.calls
        assert len(cards.calls) <= 5, f"{len(cards.calls)} loads: {cards.calls}"

    def test_max_never_exceeds_the_declared_length(self, tmp_path):
        cards = _Cards()
        loaded = _load(tmp_path, cards, ceiling=0, header=True, gpus=self.GPU, declared=196_608)
        assert loaded.context_length <= 196_608
        assert max(cards.calls) <= 196_608

    def test_the_upward_probe_holds_one_instance_at_a_time(self, tmp_path):
        cards = _Cards(limit=180_000)
        _load(tmp_path, cards, ceiling=0, header=True, gpus=self.GPU)
        assert cards.most_open == 1
        assert cards.open == 1, "exactly the served instance is left open"

    def test_max_falls_down_the_ladder_when_the_prediction_does_not_fit(self, tmp_path):
        cards = _Cards(limit=100_000)
        loaded = _load(tmp_path, cards, ceiling=0, header=True, gpus=self.GPU)
        assert 77_824 <= loaded.context_length <= 100_000, cards.calls


class TestTheDownwardBisectionReleasesFirst:
    def test_the_bisection_holds_one_instance_at_a_time(self, tmp_path):
        """32768 and 16384 fail, 8192 loads, 12288 is the truth: the probes above 8192 used to
        run with the 8192 instance still holding its weights."""
        cards = _Cards(limit=12_288)
        loaded = _load(tmp_path, cards, ceiling=32_768, declared=32_768)
        assert loaded.context_length == 12_288, cards.calls
        assert cards.most_open == 1 and cards.open == 1


class TestTheSourceRules:
    def test_the_request_wins(self):
        row = MagicMock(context_ceiling=65_536)
        assert context_ceiling.effective(row, 0) == (0, "request")
        assert context_ceiling.effective(row, None) == (65_536, "model")
        assert context_ceiling.effective(MagicMock(context_ceiling=None), None) == (None, "server")

    @pytest.mark.parametrize("raw,parsed", [("max", 0), ("MAX", 0), (0, 0), (65_536, 65_536),
                                            (None, None)])
    def test_parse(self, raw, parsed):
        assert context_ceiling.parse_ceiling(raw) == parsed

    @pytest.mark.parametrize("raw", [True, -1, "32k", 1.5, "65536"])
    def test_parse_refuses(self, raw):
        with pytest.raises(ValueError):
            context_ceiling.parse_ceiling(raw)

    def test_the_minimum_mirrors_the_loader(self):
        assert context_ceiling.MIN_CONTEXT == model_loader.GGUF_MIN_CONTEXT
        assert context_ceiling.CEILING_MAX == model_loader.CONTEXT_CEILING_MAX


class TestValidation:
    def _row(self, tmp_path, **overrides):
        from tests.support.factories import make_gguf_model

        write_gguf(tmp_path / "zora-v1.13-Q5_K_M.gguf", QWEN36_27B)
        return make_gguf_model(cache_path=str(tmp_path), **overrides)

    def test_above_the_declared_length_is_refused(self, tmp_path):
        with pytest.raises(ValidationError) as raised:
            context_ceiling.validate_for_model(self._row(tmp_path), 300_000)
        assert raised.value.details["declared"] == 262_144

    def test_the_declared_length_and_max_are_accepted(self, tmp_path):
        row = self._row(tmp_path)
        context_ceiling.validate_for_model(row, 262_144)
        context_ceiling.validate_for_model(row, 0)

    def test_below_the_smallest_window_is_refused(self, tmp_path):
        with pytest.raises(ValidationError):
            context_ceiling.validate_for_model(self._row(tmp_path), 1024)

    def test_a_transformers_row_is_refused(self):
        from tests.support.factories import make_model

        with pytest.raises(ValidationError):
            context_ceiling.validate_for_model(make_model(), 65_536)
        context_ceiling.validate_for_model(make_model(), None)


class _RecordingExecutor:
    def __init__(self):
        self.calls = []

    def submit(self, fn, *args):
        self.calls.append((fn, args))
        future = Future()
        future.set_result(None)
        return future


def _service(row):
    from miserver.services.model_service import ModelService

    repo = MagicMock()
    repo.get_by_id = AsyncMock(return_value=row)
    repo.update_status = AsyncMock(return_value=row)
    repo.update = AsyncMock(return_value=row)
    loader = MagicMock()
    loader.is_loaded = False
    loader.loaded_model_id = None
    svc = ModelService(repository=repo, downloader=MagicMock(), loader=loader, emitter=None)
    svc._executor = _RecordingExecutor()
    return svc


class TestTheServiceForwardsTheCeiling:
    def _submitted(self, row, **kwargs):
        svc = _service(row)
        with fake_gpus(), patch.object(model_loader, "llama_supports_gpu_offload", lambda: True):
            asyncio.run(svc.load_model(row.id, **kwargs))
        (fn, _args), = svc._executor.calls
        return fn.keywords

    def test_the_row_ceiling_is_submitted(self, tmp_path):
        from tests.support.factories import make_gguf_model

        row = make_gguf_model(cache_path=str(tmp_path), context_ceiling=0)
        keywords = self._submitted(row)
        assert (keywords["context_ceiling"], keywords["ceiling_source"]) == (0, "model")

    def test_the_request_ceiling_is_submitted_over_the_row(self, tmp_path):
        from tests.support.factories import make_gguf_model

        row = make_gguf_model(cache_path=str(tmp_path), context_ceiling=0)
        keywords = self._submitted(row, context_ceiling=65_536)
        assert (keywords["context_ceiling"], keywords["ceiling_source"]) == (65_536, "request")

    def test_no_ceiling_anywhere_is_the_server_default(self, tmp_path):
        from tests.support.factories import make_gguf_model

        keywords = self._submitted(make_gguf_model(cache_path=str(tmp_path)))
        assert (keywords["context_ceiling"], keywords["ceiling_source"]) == (None, "server")

    def test_a_refused_request_ceiling_submits_nothing(self):
        from tests.support.factories import make_model

        row = make_model()
        svc = _service(row)
        with fake_gpus(), pytest.raises(ValidationError):
            asyncio.run(svc.load_model(row.id, context_ceiling=65_536))
        assert svc._executor.calls == []
        assert svc._loading_model_id is None

    def test_the_worker_forwards_it_to_the_loader(self, tmp_path):
        from tests.support.factories import make_gguf_model

        row = make_gguf_model(cache_path=str(tmp_path))
        svc = _service(row)
        svc._run_async_from_thread = lambda coro: coro.close()
        svc._inference_service = MagicMock()
        svc._load_worker(row.id, row.name, str(tmp_path), "Q8", 0, False, "m.gguf", None,
                         context_ceiling=0, ceiling_source="model")
        (call,) = svc.loader.load.call_args_list
        assert call.kwargs["context_ceiling"] == 0
        assert call.kwargs["ceiling_source"] == "model"
        assert call.kwargs["gguf_file"] == "m.gguf"

    def test_set_context_ceiling_validates_and_stores(self, tmp_path):
        from tests.support.factories import make_gguf_model

        write_gguf(tmp_path / "zora-v1.13-Q5_K_M.gguf", QWEN36_27B)
        row = make_gguf_model(cache_path=str(tmp_path))
        svc = _service(row)
        asyncio.run(svc.set_context_ceiling(row.id, 131_072))
        svc.repository.update.assert_awaited_once_with(row.id, context_ceiling=131_072)
        with pytest.raises(ValidationError):
            asyncio.run(svc.set_context_ceiling(row.id, 300_000))
        assert svc.repository.update.await_count == 1


class TestTheModelLoaderForwards:
    def test_load_passes_the_ceiling_to_the_gguf_path(self):
        loader = model_loader.ModelLoader()
        with patch.object(model_loader, "load_gguf_model") as gguf:
            gguf.return_value = MagicMock(model_id=1)
            with patch.object(loader.state, "set"):
                loader.load(1, "m", "/x", "Q8", 0, gguf_file="m.gguf",
                            context_ceiling=65_536, ceiling_source="request")
        assert gguf.call_args.kwargs["context_ceiling"] == 65_536
        assert gguf.call_args.kwargs["ceiling_source"] == "request"


def test_a_transformers_load_is_called_without_the_ceiling(tmp_path):
    """The worker forwards the ceiling for a GGUF row only; a transformers load is unchanged."""
    from tests.support.factories import make_model

    row = make_model()
    svc = _service(row)
    svc._run_async_from_thread = lambda coro: coro.close()
    svc._inference_service = MagicMock()
    svc._load_worker(row.id, row.name, str(tmp_path), "FP16", 0, False, None, None,
                     context_ceiling=0, ceiling_source="model")
    (call,) = svc.loader.load.call_args_list
    assert "context_ceiling" not in call.kwargs and "ceiling_source" not in call.kwargs
