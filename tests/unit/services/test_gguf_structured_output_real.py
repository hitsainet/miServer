"""Structured output on a REAL tiny GGUF model through miServer's own llama.cpp path (2026-11-20).

Runs only where llama-cpp-python is importable AND `MISERVER_TINY_GGUF` names a GGUF file; CI
installs neither, so there it SKIPS, saying why. The reference file is ggml-org's `stories15M`
(TinyStories, 15M parameters, q4_0, 19 MB, sha256 `66967fbe…1739`, from
`huggingface.co/ggml-org/models`, `tinyllamas/stories15M-q4_0.gguf`): a story model that writes
prose and never JSON, so a parsing, validating document is evidence the grammar ran (the control
below shows the same request unconstrained is not JSON) — the fixture cannot agree with the code
by construction.

    PYTHONPATH=<dir with llama_cpp 0.3.36> MISERVER_TINY_GGUF=<path>/stories15M-q4_0.gguf \\
        venv/bin/python -m pytest tests/unit/services/test_gguf_structured_output_real.py

Recorded run: `0xcc/reviews/gguf_structured_output_2026-11-20.md`. It proves enforcement by
llama-cpp-python 0.3.36 on CPU; the served Gemma GGUF on the GPU node is checked live separately
(`tests/hardware/feature25_acceptance.py gguf-structured`).
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from jsonschema import Draft202012Validator

llama_cpp = pytest.importorskip(
    "llama_cpp", reason="llama-cpp-python is not installed (CI installs no GGUF engine)"
)
from tests.support.env_flags import flag  # noqa: E402

_PATH = flag("TINY_GGUF")
if not _PATH or not Path(_PATH).is_file():
    pytest.skip("MISERVER_TINY_GGUF does not name a GGUF file", allow_module_level=True)

from miserver.api import json_schema_subset  # noqa: E402
from miserver.api.schemas.openai import ChatCompletionRequest  # noqa: E402
from miserver.core.errors import FieldNotHonouredError  # noqa: E402
from miserver.ml.model_loader import ENGINE_LLAMACPP, LoadedModel  # noqa: E402
from tests.unit.services import test_gguf_structured_output as _stubbed  # noqa: E402

#: Bounded everywhere, so a well-behaved generation ends inside the budget.
SCHEMA = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": ["inspect", "repair", "evaluate"]},
        "target": {"type": "string", "minLength": 1, "maxLength": 24},
        "rows": {"type": "array", "items": {"type": "integer"}, "minItems": 1, "maxItems": 4},
        "note": {"anyOf": [{"type": "string", "maxLength": 16}, {"type": "null"}]},
    },
    "required": ["action", "target", "rows"],
    "additionalProperties": False,
}


@pytest.fixture(scope="module")
def llm():
    from llama_cpp import Llama

    return Llama(model_path=_PATH, n_ctx=512, n_threads=4, seed=0, verbose=False)


@pytest.fixture
def svc(llm):
    from miserver.services.inference_service import InferenceService

    svc = InferenceService.__new__(InferenceService)
    state = MagicMock()
    state.is_loaded = True
    state.current = LoadedModel(model_id=1, model_name="m", model=llm, tokenizer=None,
                                loaded_at=datetime.utcnow(), engine=ENGINE_LLAMACPP)
    svc._model_state = state

    class _NullQueue:
        def acquire(self, **_kwargs):
            import contextlib

            @contextlib.asynccontextmanager
            async def _cm():
                yield

            return _cm()

    svc._request_queue = _NullQueue()
    return svc


def chat(**over) -> ChatCompletionRequest:
    body = {"model": "m", "messages": [{"role": "user", "content": "a"}], "max_tokens": 300,
            "temperature": 1.0,
            "response_format": {"type": "json_schema",
                                "json_schema": {"name": "wsp_action", "schema": SCHEMA}}}
    body.update(over)
    return ChatCompletionRequest(**body)


def _parses(text: str) -> bool:
    try:
        json.loads(text)
    except ValueError:
        return False
    return True


def test_the_reference_schema_is_one_the_route_serves():
    json_schema_subset.check(SCHEMA)
    json_schema_subset.check_llamacpp(SCHEMA)


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
async def test_json_schema_output_validates(svc, seed):
    resp = await svc._llamacpp_chat_completion(chat(seed=seed))
    choice = resp.choices[0]
    assert choice.finish_reason == "stop", choice.message.content
    Draft202012Validator(SCHEMA).validate(json.loads(choice.message.content))


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
async def test_the_same_request_unconstrained_is_not_json(svc, seed):
    """The control: without the grammar this model writes a story."""
    resp = await svc._llamacpp_chat_completion(chat(seed=seed, response_format=None))
    assert not _parses(resp.choices[0].message.content)


async def test_json_object_stops_only_on_an_object(svc):
    stops = 0
    for seed in range(8):
        resp = await svc._llamacpp_chat_completion(
            chat(seed=seed, max_tokens=400, response_format={"type": "json_object"}))
        choice = resp.choices[0]
        if choice.finish_reason == "stop":
            stops += 1
            assert isinstance(json.loads(choice.message.content), dict)
        else:
            assert choice.finish_reason == "length"
    assert stops >= 3


def test_llama_cpp_rejects_a_bad_grammar_natively(svc):
    """The native parse check is real: `\\d` is no GBNF escape, and llama.cpp returns no sampler
    (llama-cpp-python would have added that NULL to the chain unchecked)."""
    from llama_cpp.llama_grammar import LlamaGrammar

    assert not svc._llamacpp_grammar_parses(LlamaGrammar.from_string('root ::= "\\d"'))
    assert svc._llamacpp_grammar_parses(LlamaGrammar.from_string('root ::= "a"'))


def test_a_pattern_the_converter_mistranslates_is_refused_by_the_service(svc):
    """Why `pattern` is refused at the route on llama.cpp: the converter passes `\\d` through,
    and the grammar it builds does not parse. The service refuses it too, never generating."""
    fmt = {"type": "json_schema",
           "json_schema": {"name": "p", "schema": {"type": "string", "pattern": "^\\d+$"}}}
    with pytest.raises(FieldNotHonouredError) as exc:
        svc._llamacpp_grammar(ChatCompletionRequest(
            model="m", messages=[{"role": "user", "content": "a"}], response_format=fmt,
        ).response_format)
    assert "rejected the grammar" in exc.value.message


def _accepted_schemas() -> list:
    mark = next(m for m in _stubbed.TestTheLlamaCppSchemaCheck
                .test_a_schema_the_grammar_fully_enforces_passes.pytestmark
                if m.name == "parametrize")
    return list(mark.args[1])


@pytest.mark.parametrize("schema", _accepted_schemas())
def test_every_schema_the_route_accepts_compiles_and_parses(svc, schema):
    """The route's check and the real converter agree in the direction that matters: what the
    route lets through, llama-cpp-python converts and llama.cpp parses."""
    fmt = {"type": "json_schema", "json_schema": {"name": "s", "schema": schema}}
    request = ChatCompletionRequest(model="m", messages=[{"role": "user", "content": "a"}],
                                    response_format=fmt)
    assert svc._llamacpp_grammar(request.response_format) is not None


# =============================================================================
# Structured thinking (2026-11-21): the real Qwen3.6 template on the tiny model
# =============================================================================

QWEN36 = (Path(__file__).resolve().parents[2] / "fixtures" / "chat_templates"
          / "qwen3.6-27b-gguf.jinja").read_text()


@pytest.fixture(scope="module")
def thinking_svc():
    """A second instance whose metadata carries the REAL Qwen3.6 template (as the served file's
    does), so the request is rendered by miServer and the prompt opens `<think>` by default."""
    from llama_cpp import Llama

    from miserver.services.inference_service import InferenceService

    model = Llama(model_path=_PATH, n_ctx=2048, n_threads=4, seed=0, verbose=False)
    model.metadata["tokenizer.chat_template"] = QWEN36
    svc = InferenceService.__new__(InferenceService)
    state = MagicMock()
    state.is_loaded = True
    state.current = LoadedModel(model_id=1, model_name="m", model=model, tokenizer=None,
                                loaded_at=datetime.utcnow(), engine=ENGINE_LLAMACPP)
    svc._model_state = state

    class _NullQueue:
        def acquire(self, **_kwargs):
            import contextlib

            @contextlib.asynccontextmanager
            async def _cm():
                yield

            return _cm()

    svc._request_queue = _NullQueue()
    return svc


class TestThinkingOnARealModel:
    """A story model never writes `</think>`, so every thinking-on run here reaches the allowance:
    the block is closed by miServer and the document must still be generated and validate."""

    @pytest.mark.parametrize("seed", [0, 1, 2, 3])
    async def test_thinking_on_reasons_to_the_allowance_then_a_valid_document(
        self, thinking_svc, seed,
    ):
        resp = await thinking_svc._llamacpp_chat_completion(chat(
            seed=seed, max_tokens=200, max_reasoning_tokens=60,
            chat_template_kwargs={"enable_thinking": True}))
        choice = resp.choices[0]
        assert choice.finish_reason == "stop", choice.message.content
        Draft202012Validator(SCHEMA).validate(json.loads(choice.message.content))
        assert choice.message.reasoning_content  # prose, kept apart from the document
        assert not _parses(choice.message.reasoning_content)
        reasoning = resp.usage.completion_tokens_details.reasoning_tokens
        # The block, its closing tag included, counted in the model's own vocabulary.
        assert 40 < reasoning <= 60, reasoning
        assert resp.usage.completion_tokens <= 200

    @pytest.mark.parametrize("seed", [0, 1, 2])
    async def test_thinking_off_is_the_document_alone(self, thinking_svc, seed):
        resp = await thinking_svc._llamacpp_chat_completion(chat(
            seed=seed, chat_template_kwargs={"enable_thinking": False}))
        choice = resp.choices[0]
        assert choice.finish_reason == "stop", choice.message.content
        Draft202012Validator(SCHEMA).validate(json.loads(choice.message.content))
        assert choice.message.reasoning_content is None
        assert resp.usage.completion_tokens_details.reasoning_tokens == 0

    async def test_the_grammar_ran_after_the_block_not_inside_it(self, thinking_svc):
        """The prompts the real model received: phase 1 ends in the open block, the document's
        begins after miServer's `</think>` and the separator (read back from the run)."""
        from miserver.services import llamacpp_tools as lt

        request = chat(seed=5, max_tokens=120, max_reasoning_tokens=30)
        llm = thinking_svc._model
        grammar = thinking_svc._llamacpp_grammar(request.response_format)
        run = thinking_svc._llamacpp_tool_run(request, grammar)
        assert run.mode == "document" and run.opened_think and run.reasoning_allowance == 30
        prompts: list[str] = []
        real = llm.create_completion

        def spy(prompt, **kw):
            prompts.append(llm.detokenize(prompt, special=True).decode("utf-8", "replace"))
            return real(prompt=prompt, **kw)

        generation = lt.ToolGeneration(llm, run, {"max_tokens": 120, "temperature": 1.0,
                                                  "seed": 5})
        llm.create_completion = spy
        try:
            result = thinking_svc._llamacpp_tool_sync(generation)
        finally:
            del llm.create_completion
        assert prompts[0].endswith("<|im_start|>assistant\n<think>\n")
        assert prompts[1].endswith("\n</think>\n\n")
        assert result.reasoning_cut_short is True
        json.loads(result.text)
