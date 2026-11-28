"""The request's stop on a REAL tiny GGUF model through the real llama-cpp-python (2026-11-22).

Runs only where llama-cpp-python is importable AND `MISERVER_TINY_GGUF` names a GGUF file (CI
installs neither, so there it SKIPS, saying why). Reference file: ggml-org's `stories15M` (q4_0,
sha256 `66967fbe…1739`), on CPU.

    PYTHONPATH=<dir with llama_cpp 0.3.36> MISERVER_TINY_GGUF=<path>/stories15M-q4_0.gguf \\
        venv/bin/python -m pytest tests/unit/services/test_gguf_abandoned_generation_real.py

What only the wheel can prove: `Llama.create_chat_completion` takes no `stopping_criteria`
(0.3.36 raises TypeError), and its chat handler calls `llama.create_completion` on the instance, so
`generation_stop.completions_carry` is what stops a plain GGUF chat; and llama.cpp's own loop
checks the criterion per token and reports it as "stop", which miServer maps to "length" for a
deadline. `test_gguf_abandoned_generation.py` pins the same behaviour against a stand-in.
"""

from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock

import pytest

llama_cpp = pytest.importorskip(
    "llama_cpp", reason="llama-cpp-python is not installed (CI installs no GGUF engine)"
)
from pathlib import Path  # noqa: E402

from tests.support.env_flags import flag  # noqa: E402

_PATH = flag("TINY_GGUF")
if not _PATH or not Path(_PATH).is_file():
    pytest.skip("MISERVER_TINY_GGUF does not name a GGUF file", allow_module_level=True)

from miserver.ml.model_loader import ENGINE_LLAMACPP, LoadedModel  # noqa: E402
from miserver.services import generation_stop as gs  # noqa: E402

#: A one-line template (stories15M carries none), so the real chat handler renders something.
TEMPLATE = "{% for m in messages %}{{ m['content'] }}{% endfor %}"
MESSAGES = [{"role": "user", "content": "Once upon a time"}]
PARAMS = {"max_tokens": 200, "temperature": 0.0}


@pytest.fixture(scope="module")
def llm():
    from llama_cpp import Llama

    model = Llama(model_path=_PATH, n_ctx=512, n_threads=4, seed=0, verbose=False)
    model.metadata["tokenizer.chat_template"] = TEMPLATE
    return model


def _svc(llm):
    from miserver.services.inference_service import InferenceService

    svc = InferenceService.__new__(InferenceService)
    state = MagicMock()
    state.is_loaded = True
    state.current = LoadedModel(model_id=1, model_name="m", model=llm, tokenizer=None,
                                loaded_at=datetime.now(), engine=ENGINE_LLAMACPP)
    svc._model_state = state
    return svc


class _Ticks:
    """A clock that advances 0.1 s per reading, so a deadline falls after a known token count."""

    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        self.t += 0.1
        return self.t


def test_create_chat_completion_takes_no_stopping_criteria(llm):
    with pytest.raises(TypeError):
        llm.create_chat_completion(messages=MESSAGES, max_tokens=2, stopping_criteria=lambda *a: 1)


def test_the_plain_chat_is_stopped_through_the_handlers_inner_call(llm):
    """`_llamacpp_sync`'s plain branch: the deadline (10 readings) ends a 200-token request."""
    stop = gs.GenerationStop(1.0, clock=_Ticks())
    out = _svc(llm)._llamacpp_sync(MESSAGES, dict(PARAMS), stop)
    choice = out["choices"][0]
    assert choice["finish_reason"] == "length"
    assert 0 < out["usage"]["completion_tokens"] < 20
    assert stop.stopped_by == gs.TIME
    assert "create_completion" not in vars(llm), "the carry was left on the instance"


def test_an_unstopped_chat_runs_on(llm):
    """The control: the same request with no deadline generates far past the stopped one."""
    out = _svc(llm)._llamacpp_sync(MESSAGES, dict(PARAMS), gs.GenerationStop())
    assert out["usage"]["completion_tokens"] > 40


def test_abandonment_ends_a_raw_completion_within_a_token(llm):
    stop = gs.GenerationStop()
    criteria = stop.criteria()
    seen = {"n": 0}

    def after_five(ids, logits):
        seen["n"] += 1
        if seen["n"] == 5:
            stop.abandon()
        return criteria(ids, logits)

    raw = llm.create_completion(prompt="Once upon a time", stopping_criteria=after_five, **PARAMS)
    assert raw["usage"]["completion_tokens"] == 4  # the fifth token stopped it, unreturned
    assert criteria.fired == gs.ABANDONED
