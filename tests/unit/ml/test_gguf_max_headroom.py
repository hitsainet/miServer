"""A GGUF window must be able to DECODE, not just be created (2026-11-21, model 17's crash).

Model 17 (Qwen3.6-27B Q4_K_M, q8_0) loaded with `context_ceiling: "max"` on the RTX 3090:
predicted 155,648, the upward probe created a context at 222,208, the load used 24,040 MiB of
~24,081 free, and the third request (a longer prompt) aborted the process inside `llama_decode`
(`ggml-cuda.cu:109: CUDA error`, exit 139). llama.cpp reserves its compute buffers when the
context is created, but still allocates while decoding, and a CUDA failure there aborts.

The fix, each part pinned here:
  * HEADROOM: a window counts as fitting only when every card keeps GGUF_CONTEXT_HEADROOM_MB
    free after the context is created (the ladder's rungs, the probes, the window served);
  * THE DECODE CHECK: a window above the prediction is tried in a CHILD process that decodes a
    full batch; a crash there costs a probe, not the server, and the load steps down;
  * nothing is served above the prediction without a check that passed;
  * the predictor budgets the compute buffer's growth with the window;
  * `gguf_load` records `headroom.free_mb_after_load` per card and `decode_check`.

`LlamaOnCards` takes memory from the fake cards as llama.cpp would, fitted to the two live readings
(17,112 MiB at 32,768, 24,040 MiB at 222,208): without the fix, the same fixture reaches the
crashing window (`test_the_fixture_reproduces_the_live_crash_without_the_fix`), so the other
tests are not green by construction.

MUTATION CONTROLS (run 2026-11-21, recorded in 0xcc/reviews/gguf_context_ceiling_2026-11-21.md
§13): H1 the margin removed (`_room` always True); H2 the margin read from the first card only;
D1 the decode check removed (`_needs_check` always False); D2 a failed check counted as passed;
D3 an extension constructed without the check (`_construct` guard removed); P1 the predictor
without the window's compute term; R1 `free_mb_after_load` not recorded.
"""

from __future__ import annotations

import json
import os
import signal
import sys
import textwrap
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import pytest

from miserver.core.config import settings
from miserver.core.errors import ModelLoadError
from miserver.ml import gguf_decode_check, model_loader
from miserver.ml.gguf_decode_check import child_command as real_child_command
from miserver.ml.gguf_metadata import kv_geometry
from tests.support.fake_gpus import RTX_3090, TI_3080, fake_gpus
from tests.support.fake_llama_card import (
    MODEL_17,
    MODEL_17_FREE_MB,
    CardCost,
    LlamaOnCards,
)
from tests.support.gguf_headers import QWEN36_27B, write_gguf

#: The window that crashed the pod.
LIVE_CRASH_WINDOW = 222_208
#: model 17's prediction on ~24,081 MiB free, with the window's compute term.
PREDICTED = 142_336


def _load(tmp_path, *, cards=((RTX_3090, MODEL_17_FREE_MB, 24_576),), ceiling=0, cost=MODEL_17,
          decode_mb=300, decode_fails_above=None, headroom=1024, mode="auto", gpu=None,
          checker=None, predict=True, tensor_split=""):
    path = write_gguf(tmp_path / "m-Q4_K_M.gguf", QWEN36_27B, weights_mb=16_038)
    with ExitStack() as stack:
        fake = stack.enter_context(fake_gpus(*cards))
        llama = LlamaOnCards(fake, cost, decode_mb=decode_mb,
                             decode_fails_above=decode_fails_above)
        for p in (
            patch.object(model_loader, "Llama", llama),
            patch.object(model_loader.gguf_decode_check, "run_decode_check",
                         checker or llama.decode_check),
            patch.object(model_loader, "llama_supports_gpu_offload", lambda: True),
            patch.object(model_loader, "declared_context", return_value=262_144),
            patch.object(settings, "GGUF_ENABLE_EMBEDDINGS", False),
            patch.object(settings, "GGUF_KV_CACHE_TYPE", "q8_0"),
            patch.object(settings, "GGUF_FLASH_ATTENTION", True),
            patch.object(settings, "GGUF_CONTEXT_LENGTH", 32_768),
            patch.object(settings, "GGUF_CONTEXT_HEADROOM_MB", headroom),
            patch.object(settings, "GGUF_DECODE_CHECK", mode),
            patch.object(settings, "GGUF_TENSOR_SPLIT", tensor_split),
        ):
            stack.enter_context(p)
        if not predict:
            stack.enter_context(
                patch.object(model_loader, "predicted_max_context", return_value=None))
        loaded = model_loader.load_gguf_model(
            17, "m", str(tmp_path), path.name, gpu=gpu, context_ceiling=ceiling)
        return loaded, llama, fake


class TestTheFixtureIsTheLiveCard:
    def test_the_fixture_reproduces_the_live_crash_without_the_fix(self, tmp_path):
        """No margin and a check that cannot fail: the old behaviour, which reached the window
        that crashed. If this went green at a small window, every test below would be too."""
        loaded, llama, fake = _load(tmp_path, headroom=0, decode_mb=0)
        assert loaded.context_length >= 215_000, llama.calls
        assert fake.free_mb[0] < 300, "the old window left the card all but full"

    def test_the_prediction_is_the_live_one_with_the_compute_term(self, tmp_path):
        loaded, _, _ = _load(tmp_path, mode="off")
        assert loaded.load_decisions["context"]["predicted_fit"] == PREDICTED


class TestMaxKeepsHeadroom:
    def test_the_upward_probe_stops_where_the_margin_ends(self, tmp_path):
        loaded, llama, fake = _load(tmp_path)
        n = loaded.context_length
        assert PREDICTED < n < LIVE_CRASH_WINDOW, (n, llama.trials)
        # What the served model leaves on the card: the margin, kept.
        free = loaded.load_decisions["headroom"]["free_mb_after_load"]
        assert free == {"cuda:0": fake.free_mb[0]}
        assert free["cuda:0"] >= 1024
        assert loaded.load_decisions["headroom"] == {
            "required_mb": 1024, "free_mb_after_load": free, "ok": True}
        # The probe above the one served did not keep it.
        trials = loaded.load_decisions["decode_check"]["trials"]
        assert any(not t["ok"] and t["headroom_ok"] is False and t["n_ctx"] > n for t in trials)

    def test_a_probe_short_of_the_margin_is_a_window_that_does_not_fit(self, tmp_path):
        loaded, llama, _ = _load(tmp_path)
        trials = loaded.load_decisions["decode_check"]["trials"]
        short = [t for t in trials if t["headroom_ok"] is False]
        assert short, trials
        assert all(not t["ok"] and t["n_ctx"] > loaded.context_length for t in short)
        assert all(t["free_mb"]["cuda:0"] < 1024 for t in short)

    def test_a_ladder_rung_short_of_the_margin_is_closed_and_the_ladder_steps_down(self, tmp_path):
        """A model that costs more than predicted: the predicted window is created with ~20 MiB
        free. It is not served; the ladder and its bisection find the largest that keeps 1 GiB."""
        heavy = CardCost(base_mb=MODEL_17.base_mb, per_token_bytes=60_000)
        loaded, llama, fake = _load(tmp_path, cost=heavy, ceiling=200_000, mode="off")
        attempts = loaded.load_decisions["attempts"]
        assert attempts[0]["n_ctx"] == PREDICTED and attempts[0]["loaded"] is True
        assert attempts[0]["headroom_ok"] is False
        assert attempts[0]["free_mb_after"]["cuda:0"] < 1024
        assert loaded.context_length < PREDICTED
        assert fake.free_mb[0] >= 1024
        assert llama.open == 1

    def test_no_margin_at_any_length_fails_honestly(self, tmp_path):
        loaded_ok = None
        with pytest.raises(ModelLoadError) as err:
            loaded_ok, _, _ = _load(tmp_path, cards=((RTX_3090, 17_000, 24_576),), ceiling=None)
        assert loaded_ok is None
        assert "GGUF_CONTEXT_HEADROOM_MB" in str(err.value)


class TestTheDecodeCheck:
    def test_a_failed_check_steps_the_load_down_and_is_recorded(self, tmp_path):
        """Contexts create and the memory reads fine above 170,000, but the decode aborts there,
        as model 17's did. The load serves below it, and says what happened."""
        loaded, llama, _ = _load(tmp_path, decode_fails_above=170_000)
        assert PREDICTED < loaded.context_length <= 170_000, llama.trials
        check = loaded.load_decisions["decode_check"]
        assert check["step_downs"] >= 1
        crashed = [t for t in check["trials"] if t["exit_code"] == 139]
        assert crashed and all(not t["ok"] and t["n_ctx"] > 170_000 for t in crashed)
        assert check["ok"] is True, "the served window above the prediction passed a check"
        assert check["tokens"] == 512 and check["mode"] == "auto"

    def test_the_server_holds_nothing_while_a_child_checks(self, tmp_path):
        _, llama, _ = _load(tmp_path)
        assert llama.trials, "max probed above the prediction without a decode check"
        assert llama.open_during_trial == [0] * len(llama.trials)
        assert llama.most_open == 1

    def test_a_checked_probe_is_not_also_constructed_in_the_server(self, tmp_path):
        """Prediction, then three child probes, then the winner once: two constructions here."""
        loaded, llama, _ = _load(tmp_path)
        assert llama.calls == [PREDICTED, loaded.context_length]
        assert len(llama.trials) == 3

    def test_always_also_checks_a_window_at_the_prediction(self, tmp_path):
        loaded, llama, _ = _load(tmp_path, ceiling=None, mode="always")
        assert loaded.context_length == 32_768
        assert llama.trials == [32_768]
        assert loaded.load_decisions["decode_check"]["ok"] is True

    def test_auto_does_not_check_the_server_default(self, tmp_path):
        loaded, llama, _ = _load(tmp_path, ceiling=None)
        assert loaded.context_length == 32_768
        assert llama.trials == []
        assert loaded.load_decisions["decode_check"]["ok"] is None


class TestNothingAboveThePredictionUnchecked:
    def test_with_every_check_failing_max_serves_the_prediction(self, tmp_path):
        loaded, llama, _ = _load(tmp_path, decode_fails_above=0)
        assert loaded.context_length == PREDICTED
        assert all(n <= PREDICTED for n in llama.calls), llama.calls
        assert loaded.load_decisions["decode_check"]["step_downs"] == len(llama.trials) == 3

    def test_off_means_no_window_above_the_prediction(self, tmp_path):
        loaded, llama, _ = _load(tmp_path, mode="off")
        assert loaded.context_length == PREDICTED
        assert llama.trials == []
        assert all(n <= PREDICTED for n in llama.calls), llama.calls

    def test_every_constructed_window_above_the_prediction_passed_a_check(self, tmp_path):
        loaded, llama, _ = _load(tmp_path, decode_fails_above=200_000)
        passed = {t["n_ctx"] for t in loaded.load_decisions["decode_check"]["trials"] if t["ok"]}
        above = [n for n in llama.calls if n > PREDICTED]
        assert above and all(n in passed for n in above), (llama.calls, passed)


class TestTheMarginIsPerCard:
    CARDS = ((RTX_3090, MODEL_17_FREE_MB, 24_576), (TI_3080, 11_500, 12_288))

    #: 131,072 tokens split evenly (GGUF_TENSOR_SPLIT=1,1) take 10,800 MiB a card: the 3090
    #: keeps ~13,280 MiB, the 3080 Ti ~700.
    HEAVY = CardCost(base_mb=MODEL_17.base_mb, per_token_bytes=45_488)

    def test_every_card_of_a_split_is_recorded(self, tmp_path):
        loaded, _, fake = _load(tmp_path, cards=self.CARDS, gpu="all")
        assert loaded.gpu_indices == [0, 1]
        free = loaded.load_decisions["headroom"]["free_mb_after_load"]
        assert free == {"cuda:0": fake.free_mb[0], "cuda:1": fake.free_mb[1]}
        assert all(mb >= 1024 for mb in free.values()), free

    def test_a_window_short_on_the_second_card_alone_does_not_fit(self, tmp_path):
        """Judging by the first card alone serves a window that leaves the 3080 Ti ~700 MiB."""
        loaded, _, fake = _load(tmp_path, cards=self.CARDS, gpu="all", cost=self.HEAVY,
                                ceiling=131_072, mode="off", predict=False, tensor_split="1,1")
        first = loaded.load_decisions["attempts"][0]
        assert first["n_ctx"] == 131_072 and first["loaded"] is True
        assert first["free_mb_after"]["cuda:0"] >= 1024 > first["free_mb_after"]["cuda:1"]
        assert first["headroom_ok"] is False
        assert loaded.context_length < 131_072
        assert min(fake.free_mb) >= 1024


class TestThePredictorBudgetsTheWindowsComputeBuffer:
    GEOMETRY = kv_geometry(QWEN36_27B)

    def test_the_compute_term_from_the_sources(self):
        # q8_0: the f16 copy of the one cache layer's K and V (4 heads x 512) + the f16 mask.
        assert self.GEOMETRY.widest_layer_kv_elements_per_token == 4 * (256 + 256)
        assert self.GEOMETRY.compute_bytes_per_token("q8_0", 512) == 2 * 2048 + 2 * 512
        assert self.GEOMETRY.compute_bytes_per_token("f16", 512) == 2 * 512

    @staticmethod
    def _plan_mb(n_ctx, kv, compute, state, cards=1):
        """The documented plan for ONE card: weights + state + cache + max(floor, base + window)."""
        rest = max(model_loader._GGUF_RUNTIME_OVERHEAD_MB,
                   model_loader._GGUF_BASE_OVERHEAD_MB + compute * n_ctx / 2**20)
        return 16_038 + state + kv * n_ctx / 2**20 + rest * cards

    def test_the_plan_covers_both_live_q8_0_readings(self):
        kv = self.GEOMETRY.bytes_per_token("q8_0")
        compute = self.GEOMETRY.compute_bytes_per_token("q8_0", 512)
        state = self.GEOMETRY.recurrent_state_mb(1)
        for n_ctx, measured in ((32_768, 17_112), (222_208, 24_040)):
            assert self._plan_mb(n_ctx, kv, compute, state) >= measured, n_ctx

    def test_the_prediction_is_the_largest_window_the_plan_allows(self):
        """The solver against the plan it solves: the answer fits, 1,024 tokens more does not."""
        kv = self.GEOMETRY.bytes_per_token("q8_0")
        compute = self.GEOMETRY.compute_bytes_per_token("q8_0", 512)
        state = self.GEOMETRY.recurrent_state_mb(1)
        room = MODEL_17_FREE_MB * model_loader._VRAM_PLANNING_FRACTION
        n = model_loader.predicted_max_context(
            "x", "q8_0", MODEL_17_FREE_MB, 16_038, bytes_per_token=kv, fixed_mb=state,
            compute_bytes_per_token=compute)
        assert self._plan_mb(n, kv, compute, state) <= room
        assert self._plan_mb(n + 1024, kv, compute, state) > room

    def test_the_live_card_predicts_below_the_old_constant_only_answer(self):
        kv = self.GEOMETRY.bytes_per_token("q8_0")
        common = {"bytes_per_token": kv, "fixed_mb": self.GEOMETRY.recurrent_state_mb(1)}
        old = model_loader.predicted_max_context("x", "q8_0", MODEL_17_FREE_MB, 16_038, **common)
        new = model_loader.predicted_max_context(
            "x", "q8_0", MODEL_17_FREE_MB, 16_038, **common,
            compute_bytes_per_token=self.GEOMETRY.compute_bytes_per_token("q8_0", 512))
        assert (old, new) == (155_648, PREDICTED)

    def test_a_short_window_is_governed_by_the_floor_unchanged(self):
        """gemma-4-31b at q8_0 on the 3090: the floor binds, so the measured bracket is kept."""
        from tests.support.gguf_headers import GEMMA4_31B

        g = kv_geometry(GEMMA4_31B)
        common = {"bytes_per_token": g.bytes_per_token("q8_0")}
        plain = model_loader.predicted_max_context("x", "q8_0", 24_576, 16_081, **common)
        with_term = model_loader.predicted_max_context(
            "x", "q8_0", 24_576, 16_081, **common,
            compute_bytes_per_token=g.compute_bytes_per_token("q8_0", 512))
        assert plain == with_term == 12_288


# =============================================================================
# The child process itself: real subprocesses, scripted.
# =============================================================================


def _script(tmp_path: Path, body: str) -> list[str]:
    script = tmp_path / "child.py"
    script.write_text(textwrap.dedent(body))
    return [sys.executable, "-I", str(script)]


class TestTheChildCannotTakeTheServerDown:
    def test_a_child_that_exits_139_is_a_failed_check(self, tmp_path):
        command = _script(tmp_path, """
            import os, sys
            sys.stdin.readline()
            sys.stderr.write("CUDA error: out of memory\\n  ggml-cuda.cu:109\\n")
            sys.stderr.flush()
            os._exit(139)
        """)
        reads = []
        result = gguf_decode_check.run_decode_check(
            {"n_ctx": 222_208}, 512, lambda: reads.append(1) or {0: 9_000}, command=command)
        assert (result.created, result.decode_ok, result.exit_code) == (False, False, 139)
        assert "exited 139" in result.error and "ggml-cuda.cu:109" in result.error
        assert result.free_mb == {}

    def test_a_child_killed_by_a_signal_is_a_failed_check(self, tmp_path):
        command = _script(tmp_path, """
            import os, signal, sys
            sys.stdin.readline()
            os.kill(os.getpid(), signal.SIGSEGV)
        """)
        result = gguf_decode_check.run_decode_check({"n_ctx": 8192}, 512, dict, command=command)
        assert result.exit_code == -signal.SIGSEGV
        assert not result.decode_ok and "signal" in result.error

    def test_the_card_is_read_while_the_child_holds_the_decoded_context(self, tmp_path):
        done = tmp_path / "exited"
        command = _script(tmp_path, f"""
            import json, sys
            request = json.loads(sys.stdin.readline())
            assert request["n_tokens"] == 512 and request["kwargs"]["n_ctx"] == 65536
            print("noise from llama.cpp")
            print({gguf_decode_check.MARKER!r} + json.dumps(
                {{"stage": "decoded", "created": True, "decode_ok": True, "tokens": 512}}))
            sys.stdout.flush()
            sys.stdin.readline()
            open({str(done)!r}, "w").write("bye")
        """)
        seen = []

        def measure():
            seen.append(done.exists())
            return {0: 4_321}

        result = gguf_decode_check.run_decode_check({"n_ctx": 65_536}, 512, measure,
                                                    command=command)
        assert result.decode_ok and result.created and result.exit_code == 0
        assert result.free_mb == {0: 4_321} and result.tokens == 512
        assert seen[:2] == [False, False], "the reading came after the child had let go"
        assert done.exists()

    def test_a_child_that_hangs_is_killed(self, tmp_path):
        command = _script(tmp_path, """
            import sys, time
            sys.stdin.readline()
            time.sleep(600)
        """)
        result = gguf_decode_check.run_decode_check({"n_ctx": 8192}, 512, dict, command=command,
                                                    timeout_s=1.0)
        assert not result.decode_ok and "did not finish" in result.error
        assert result.exit_code is not None and result.exit_code != 0

    def test_a_child_that_cannot_start_is_a_failed_check(self, tmp_path):
        result = gguf_decode_check.run_decode_check(
            {"n_ctx": 8192}, 512, dict, command=[str(tmp_path / "missing")])
        assert not result.created and "could not start" in result.error

    def test_the_loader_survives_a_crashing_child_and_serves_the_prediction(self, tmp_path):
        """End to end, with the REAL `run_decode_check` and a child that aborts like llama.cpp."""
        command = _script(tmp_path, """
            import os, sys
            sys.stdin.readline()
            os._exit(139)
        """)
        with patch.object(gguf_decode_check, "child_command", lambda: command):
            loaded, llama, _ = _load(tmp_path, checker=gguf_decode_check.run_decode_check)
        assert os.getpid() and loaded.context_length == PREDICTED
        check = loaded.load_decisions["decode_check"]
        assert [t["exit_code"] for t in check["trials"]] == [139, 139, 139]
        assert check["step_downs"] == 3 and check["ok"] is None
        assert llama.calls == [PREDICTED, PREDICTED]


class TestTheChildScript:
    """`gguf_decode_check.py` run as the child, against a stand-in `llama_cpp` package."""

    FAKE = """
        import json, os
        LOG = os.environ["FAKE_LLAMA_LOG"]
        MODE = os.environ.get("FAKE_LLAMA_MODE", "ok")

        def _log(entry):
            with open(LOG, "a") as handle:
                handle.write(json.dumps(entry) + "\\n")

        class Llama:
            def __init__(self, **kwargs):
                _log({"init": kwargs})
                if MODE == "no_context":
                    raise ValueError("Failed to create llama_context")
                self.n_batch = min(kwargs["n_ctx"], 512)

            def tokenize(self, text, add_bos=True, special=False):
                return list(range(1, len(text) // 4))

            def eval(self, tokens):
                _log({"eval": len(tokens)})
                if MODE == "decode_error":
                    raise RuntimeError("llama_decode returned -3")

            def reset(self):
                _log({"reset": True})
    """

    def _run(self, tmp_path, monkeypatch, mode="ok"):
        package = tmp_path / "site" / "llama_cpp"
        package.mkdir(parents=True)
        (package / "__init__.py").write_text(textwrap.dedent(self.FAKE))
        log = tmp_path / "llama.log"
        monkeypatch.setenv("PYTHONPATH", str(tmp_path / "site"))
        monkeypatch.setenv("FAKE_LLAMA_LOG", str(log))
        monkeypatch.setenv("FAKE_LLAMA_MODE", mode)
        # Not `-I`: the stand-in package is found through PYTHONPATH, which `-I` ignores.
        command = [sys.executable, gguf_decode_check.__file__]
        kwargs = {"model_path": "/models/m.gguf", "n_ctx": 196_608, "n_gpu_layers": -1,
                  "flash_attn": True, "type_k": 8, "type_v": 8, "split_mode": 0, "main_gpu": 0}
        self.reads: list[list[dict]] = []

        def measure():
            # What the child had done when the card was read (the first read is before it starts).
            self.reads.append([json.loads(line) for line in log.read_text().splitlines()]
                              if log.exists() else [])
            return {0: 2_000}

        result = gguf_decode_check.run_decode_check(kwargs, 512, measure, command=command)
        entries = [json.loads(line) for line in log.read_text().splitlines()]
        return result, entries, kwargs

    def test_it_builds_the_same_llama_and_decodes_a_full_batch_then_one_token(
            self, tmp_path, monkeypatch):
        result, entries, kwargs = self._run(tmp_path, monkeypatch)
        assert entries[0] == {"init": kwargs}
        assert entries[1:] == [{"eval": 512}, {"eval": 1}, {"reset": True}]
        # The card was read after both decodes and before the child let go of the context.
        assert self.reads[1] == entries[:3], self.reads
        assert result.decode_ok and result.tokens == 512 and result.free_mb == {0: 2_000}
        assert result.exit_code == 0

    def test_a_context_that_cannot_be_created_is_reported_not_crashed(self, tmp_path, monkeypatch):
        result, entries, _ = self._run(tmp_path, monkeypatch, mode="no_context")
        assert not result.created and "llama_context" in result.error
        assert result.exit_code == 0 and len(entries) == 1

    def test_a_decode_error_code_is_reported(self, tmp_path, monkeypatch):
        result, _, _ = self._run(tmp_path, monkeypatch, mode="decode_error")
        assert result.created and not result.decode_ok and "returned -3" in result.error

    def test_the_child_runs_isolated_from_this_interpreter(self, tmp_path):
        """The real command: `-I`, this file. It must start and report with nothing of miserver
        importable (whether llama_cpp is installed here or not, the bogus path cannot load)."""
        command = real_child_command()
        assert command[:2] == [sys.executable, "-I"]
        assert command[2] == os.path.abspath(gguf_decode_check.__file__)
        result = gguf_decode_check.run_decode_check(
            {"model_path": str(tmp_path / "none.gguf"), "n_ctx": 4096, "verbose": False},
            512, dict, command=command, timeout_s=120)
        assert result.exit_code == 0, result.error
        assert not result.created and result.error


def test_unit_tests_cannot_reach_the_real_child():
    """The conftest guard (`_no_real_decode_check_child`) is live: the real command refuses."""
    with pytest.raises(AssertionError, match="real GGUF decode-check child"):
        gguf_decode_check.child_command()
