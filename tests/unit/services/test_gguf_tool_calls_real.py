"""Function calling on a REAL tiny GGUF model through miServer's llama.cpp path (2026-11-21).

Runs only where llama-cpp-python is importable AND `MISERVER_TINY_GGUF` names a GGUF file; CI
installs neither, so there it SKIPS, saying why. The reference file is ggml-org's `stories15M`
(TinyStories, q4_0, sha256 `66967fbe…1739`): a story model that has never seen a tool call, so a
call that parses and validates is evidence the grammar ran — the control below shows the same
request with no grammar produces no call. The chat template is set to the REAL Qwen3.6 (or Qwen2.5)
one, as `tokenizer.chat_template` is in the served file's metadata.

    PYTHONPATH=<dir with llama_cpp 0.3.36> MISERVER_TINY_GGUF=<path>/stories15M-q4_0.gguf \\
        venv/bin/python -m pytest tests/unit/services/test_gguf_tool_calls_real.py

Also proved here, with the wheel: miServer's renderer is byte-identical to llama-cpp-python's
`Jinja2ChatFormatter`, and llama.cpp's OWN grammar engine accepts exactly the call strings it
should (`_Acceptor`: each character offered alone to the grammar sampler, then end of generation).
"""

from __future__ import annotations

import ctypes
import json
import math
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
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

from llama_cpp.llama_chat_format import Jinja2ChatFormatter  # noqa: E402
from llama_cpp.llama_grammar import PRIMITIVE_RULES, SchemaConverter  # noqa: E402

from miserver.api.schemas.openai import ChatCompletionRequest  # noqa: E402
from miserver.ml.model_loader import ENGINE_LLAMACPP, LoadedModel  # noqa: E402
from miserver.services import llamacpp_tools as lt  # noqa: E402
from miserver.services.tool_calls import HERMES, QWEN3_XML, parse_tool_calls, plan_tools  # noqa: E402

TEMPLATES = Path(__file__).resolve().parents[2] / "fixtures" / "chat_templates"
TEMPLATE = {QWEN3_XML: (TEMPLATES / "qwen3.6-27b-gguf.jinja").read_text(),
            HERMES: (TEMPLATES / "qwen2.5-7b-instruct.jinja").read_text()}

#: Bounded everywhere, so a grammar-forced call ends inside the budget on a model that never
#: chooses to end one.
TOOLS = [
    {"type": "function", "function": {"name": "describe_tensor", "description": "Statistics",
     "parameters": {"type": "object", "properties": {
         "name": {"type": "string", "maxLength": 12}}, "required": ["name"]}}},
    {"type": "function", "function": {"name": "scale_tensor", "description": "Scale",
     "parameters": {"type": "object", "properties": {
         "name": {"type": "string", "enum": ["weight", "bias"]},
         "factor": {"type": "integer"},
         "inplace": {"type": "boolean"},
         "layers": {"type": "array", "items": {"type": "integer"}, "maxItems": 2}},
      "required": ["name", "factor"]}}},
]
SCHEMAS = {t["function"]["name"]: t["function"]["parameters"] for t in TOOLS}
#: For the acceptance checks only (an unbounded raw string, which this model would never end).
RUN_CODE = {"type": "function", "function": {"name": "run_code", "parameters": {
    "type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"]}}}
HISTORY = [
    {"role": "system", "content": "You edit weights."},
    {"role": "user", "content": "Inspect weight."},
    {"role": "assistant", "content": None, "tool_calls": [{
        "id": "c1", "type": "function",
        "function": {"name": "describe_tensor", "arguments": '{"name": "weight"}'}}]},
    {"role": "tool", "tool_call_id": "c1", "content": "norm 5.21"},
]


@pytest.fixture(scope="module")
def llm():
    from llama_cpp import Llama

    return Llama(model_path=_PATH, n_ctx=2048, n_threads=4, seed=0, verbose=False)


def _svc(llm, fmt):
    from miserver.services.inference_service import InferenceService

    llm.metadata["tokenizer.chat_template"] = TEMPLATE[fmt]
    svc = InferenceService.__new__(InferenceService)
    state = MagicMock()
    state.is_loaded = True
    state.current = LoadedModel(model_id=1, model_name="m", model=llm, tokenizer=None,
                                loaded_at=datetime.now(), engine=ENGINE_LLAMACPP)
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
    body = {"model": "m", "messages": [{"role": "user", "content": "scale weight by 2"}],
            "tools": TOOLS, "max_tokens": 160, "temperature": 1.0,
            "chat_template_kwargs": {"enable_thinking": False}}
    body.update(over)
    return ChatCompletionRequest(**body)


# =============================================================================
# The renderer and the converter, against the wheel
# =============================================================================


@pytest.mark.parametrize("fmt", [QWEN3_XML, HERMES])
@pytest.mark.parametrize("case", ["tools", "history", "plain"])
def test_the_renderer_is_jinja2chatformatter(fmt, case):
    from miserver.services.tool_calls import template_messages

    messages = template_messages(ChatCompletionRequest(
        model="m", messages=HISTORY if case == "history" else HISTORY[:2]).messages)
    tools = TOOLS if case != "plain" else None
    kwargs = {"enable_thinking": False} if case == "tools" else {}
    theirs = Jinja2ChatFormatter(template=TEMPLATE[fmt], eos_token="<|im_end|>",
                                 bos_token="<|endoftext|>")(messages=messages, tools=tools,
                                                            **kwargs).prompt
    ours = lt.render_chat_prompt(TEMPLATE[fmt], messages, bos="<|endoftext|>", eos="<|im_end|>",
                                 tools=tools, template_kwargs=kwargs)
    assert ours == theirs


def test_the_rewritten_char_rule_is_the_wheels():
    assert lt.CONVERTER_CHAR_RULE == "char ::= " + PRIMITIVE_RULES["char"].content


class _Acceptor:
    """Does llama.cpp's own grammar engine accept `text`, whole, and then end of generation?"""

    def __init__(self, llm):
        self.vocab = llm._model.vocab
        self.by_char: dict[str, int] = {}
        buf = ctypes.create_string_buffer(64)
        for t in range(llama_cpp.llama_vocab_n_tokens(self.vocab)):
            k = llama_cpp.llama_token_to_piece(self.vocab, t, buf, 64, 0, False)
            try:
                piece = buf.raw[:k].decode("utf-8") if k > 0 else ""
            except UnicodeDecodeError:
                continue
            if len(piece) == 1:
                self.by_char.setdefault(piece, t)
        self.eos = llama_cpp.llama_vocab_eos(self.vocab)

    def _allows(self, sampler, token) -> bool:
        data = (llama_cpp.llama_token_data * 1)(llama_cpp.llama_token_data(token, 0.0, 0.0))
        array = llama_cpp.llama_token_data_array(data, 1, -1, False)
        llama_cpp.llama_sampler_apply(sampler, ctypes.byref(array))
        return not math.isinf(data[0].logit)

    def accepts(self, grammar: str, text: str) -> bool:
        sampler = llama_cpp.llama_sampler_init_grammar(self.vocab, grammar.encode(), b"root")
        assert sampler, "llama.cpp did not parse the grammar"
        try:
            for ch in text:
                token = self.by_char[ch]
                if not self._allows(sampler, token):
                    return False
                llama_cpp.llama_sampler_accept(sampler, token)
            return self._allows(sampler, self.eos)
        finally:
            llama_cpp.llama_sampler_free(sampler)


def _grammar(fmt, *, parallel=True):
    conv = SchemaConverter(prop_order={}, allow_fetch=False, dotall=False, raw_pattern=False)
    tools = TOOLS + [RUN_CODE]
    names = frozenset(t["function"]["name"] for t in tools)
    return lt.build_call_grammar(fmt, tools, allowed=names, parallel=parallel, converter=conv)


def _xml(name, params):
    inner = "".join(f"<parameter={k}>\n{v}\n</parameter>\n" for k, v in params)
    return f"\n<function={name}>\n{inner}</function>\n</tool_call>"


XML_GOOD = [
    _xml("describe_tensor", [("name", "w")]),
    _xml("run_code", [("code", "if a < b:\n    print('</p>', \"x\")\n")]),
    _xml("describe_tensor", [("name", "")]),
    _xml("scale_tensor", [("name", "bias"), ("factor", "-3")]),
    _xml("scale_tensor", [("name", "weight"), ("factor", "2"), ("inplace", "true"),
                          ("layers", "[1, 2]")]),
]
XML_BAD = [
    _xml("scale_tensor", [("factor", "2")]),                       # required name missing
    _xml("scale_tensor", [("name", "gamma"), ("factor", "2")]),    # not in the enum
    _xml("scale_tensor", [("name", "bias"), ("factor", "2.5")]),   # not an integer
    _xml("scale_tensor", [("name", "bias"), ("factor", "2"), ("layers", "[1, 2, 3]")]),
    _xml("run_code", [("code", "x</parameter>y")]),               # the raw text's one forbidden
    _xml("describe_tensor", [("name", "a<b")]),                    # bounded raw text: no `<`
    _xml("describe_tensor", [("name", "thirteen chars")]),         # over maxLength
    _xml("drop_tensor", [("name", "w")]),                          # not offered
    _xml("describe_tensor", [("name", "w")]) + " trailing",        # nothing after the call
]


def test_llama_cpp_accepts_exactly_the_xml_calls(llm):
    acceptor, grammar = _Acceptor(llm), _grammar(QWEN3_XML)
    for text in XML_GOOD:
        assert acceptor.accepts(grammar, text), text
    for text in XML_BAD:
        assert not acceptor.accepts(grammar, text), text
    two = XML_GOOD[0] + "\n<tool_call>" + XML_GOOD[3]
    assert acceptor.accepts(grammar, two)
    assert not acceptor.accepts(_grammar(QWEN3_XML, parallel=False), two)


def test_llama_cpp_accepts_exactly_the_hermes_calls(llm):
    acceptor, grammar = _Acceptor(llm), _grammar(HERMES)
    good = '\n{"name": "scale_tensor", "arguments": {"name": "bias", "factor": 2}}\n</tool_call>'
    assert acceptor.accepts(grammar, good)
    for bad in [
        '\n{"name": "scale_tensor", "arguments": {"factor": 2}}\n</tool_call>',
        '\n{"name": "nope", "arguments": {}}\n</tool_call>',
        '\n{"name": "describe_tensor", "arguments": {"name": "a\nb"}}\n</tool_call>',  # raw \n
    ]:
        assert not acceptor.accepts(grammar, bad), bad


# =============================================================================
# Generation on the tiny model
# =============================================================================


def _valid(tool_calls) -> bool:
    for tc in tool_calls:
        Draft202012Validator(SCHEMAS[tc["function"]["name"]]).validate(
            json.loads(tc["function"]["arguments"]))
    return True


@pytest.mark.parametrize("fmt", [QWEN3_XML, HERMES])
@pytest.mark.parametrize("seed", range(6))
async def test_a_forced_call_parses_and_validates(llm, fmt, seed):
    resp = await _svc(llm, fmt)._llamacpp_chat_completion(
        chat(tool_choice="required", parallel_tool_calls=False, seed=seed))
    choice = resp.choices[0]
    assert choice.finish_reason == "tool_calls", choice.message.content
    assert len(choice.message.tool_calls) == 1 and _valid(choice.message.tool_calls)


@pytest.mark.parametrize("fmt", [QWEN3_XML, HERMES])
async def test_a_named_call_names_that_function(llm, fmt):
    named = {"type": "function", "function": {"name": "describe_tensor"}}
    resp = await _svc(llm, fmt)._llamacpp_chat_completion(
        chat(tool_choice=named, parallel_tool_calls=False, seed=3, messages=HISTORY))
    (tc,) = resp.choices[0].message.tool_calls
    assert tc["function"]["name"] == "describe_tensor" and _valid([tc])


@pytest.mark.parametrize("fmt", [QWEN3_XML, HERMES])
def test_control_the_same_call_without_the_grammar_is_no_call(llm, fmt):
    """The fixture cannot agree with the code by construction: unconstrained, this model continues
    an opened call with story text, which does not parse as a call."""
    svc = _svc(llm, fmt)
    run = svc._llamacpp_tool_run(chat(tool_choice="required", parallel_tool_calls=False))
    plan = plan_tools(chat(), SimpleNamespace(chat_template=TEMPLATE[fmt]))
    misses = 0
    for seed in range(4):
        tail = llm.tokenize(b"<tool_call>", add_bos=False, special=True)
        raw = llm.create_completion(prompt=run.prompt_tokens + tail, max_tokens=60, seed=seed,
                                    temperature=1.0)
        misses += parse_tool_calls("<tool_call>" + raw["choices"][0]["text"], plan) is None
    assert misses == 4


async def test_reasoning_the_model_never_ends_leaves_too_little_is_length_never_a_marker(llm):
    """40 tokens: the allowance (20) closes the block, and the call cannot fit in what is left:
    `length`, the reasoning kept, no marker and no half call in `content` (2026-11-21)."""
    resp = await _svc(llm, QWEN3_XML)._llamacpp_chat_completion(
        chat(tool_choice="required", chat_template_kwargs=None, max_tokens=40, seed=1))
    choice = resp.choices[0]
    assert choice.finish_reason == "length" and not choice.message.tool_calls
    assert choice.message.reasoning_content
    assert "<tool_call>" not in (choice.message.content or "")
    assert "<function=" not in (choice.message.content or "")


@pytest.mark.parametrize("seed", [0, 1, 2])
async def test_reasoning_the_model_never_ends_is_cut_and_the_call_made(llm, seed):
    """The live failure (WSP run wsp_84dddc47ecaf) on the real engine: a story model never writes
    `</think>`; the allowance closes the block and the forced call is still generated, parsed and
    validated within max_tokens."""
    resp = await _svc(llm, QWEN3_XML)._llamacpp_chat_completion(
        chat(tool_choice="required", chat_template_kwargs=None, max_tokens=240,
             max_reasoning_tokens=80, parallel_tool_calls=False, seed=seed))
    choice = resp.choices[0]
    assert choice.finish_reason == "tool_calls", choice.message
    (call,) = choice.message.tool_calls
    assert call["function"]["name"] in {"describe_tensor", "scale_tensor", "run_code"}
    assert choice.message.reasoning_content
    reasoning = resp.usage.completion_tokens_details.reasoning_tokens
    assert 60 < reasoning <= 80, reasoning
    assert resp.usage.completion_tokens <= 240


@pytest.mark.parametrize("seed", [0, 1])
async def test_auto_reasoning_cut_then_an_answer_never_a_marker(llm, seed):
    resp = await _svc(llm, QWEN3_XML)._llamacpp_chat_completion(
        chat(chat_template_kwargs=None, max_tokens=160, seed=seed))
    choice = resp.choices[0]
    message = choice.message
    assert message.reasoning_content
    assert "<tool_call" not in (message.content or "")
    assert resp.usage.completion_tokens_details.reasoning_tokens <= 80
    assert resp.usage.completion_tokens <= 160


async def test_a_streamed_forced_call_is_one_tool_calls_delta(llm):
    events = []
    async for frame in _svc(llm, QWEN3_XML)._llamacpp_stream_chat_completion(
        chat(tool_choice="required", parallel_tool_calls=False, seed=2, stream=True)
    ):
        body = frame[len("data: "):].strip()
        if body != "[DONE]":
            events.append(json.loads(body))
    deltas = [c["delta"] for e in events for c in e.get("choices", [])]
    calls = [d["tool_calls"] for d in deltas if d.get("tool_calls")]
    assert len(calls) == 1 and _valid([{k: v for k, v in calls[0][0].items() if k != "index"}])
    assert [c.get("finish_reason") for e in events for c in e.get("choices", [])][-1] == \
        "tool_calls"


# =============================================================================
# weightspace's tool list (2026-11-21; live failure on WSP run wsp_c3e954d3f21f)
# =============================================================================


def test_llama_cpp_accepts_weightspace_write_tensor_values(llm):
    """`write_tensor`'s `values` is annotations alone, so ANY JSON value: llama.cpp's own grammar
    engine admits it as a nested list of numbers and as a scalar (and as anything else JSON), in
    both call formats, and still refuses text that is not JSON there."""
    import json as _json

    from tests.unit.services import test_gguf_weightspace_tools as ws

    acceptor = _Acceptor(llm)
    names = frozenset(t["function"]["name"] for t in ws.TOOLS)

    def grammar(fmt):
        conv = SchemaConverter(prop_order={}, allow_fetch=False, dotall=False, raw_pattern=False)
        return lt.build_call_grammar(fmt, ws.TOOLS, allowed=names, parallel=False, converter=conv)

    xml, hermes = grammar(QWEN3_XML), grammar(HERMES)
    for values in (ws.NESTED, ws.SCALAR, -7, "abc", None, True, {"k": [1, 2]}, []):
        text = ws.xml_write(values)
        assert acceptor.accepts(xml, text), text
        call = {"name": "write_tensor", "arguments": {"name": "w", "values": values}}
        assert acceptor.accepts(hermes, "\n" + _json.dumps(call) + "\n</tool_call>"), values
    for bad in ("[1, 2", "1 2", "nested", "[1,]"):
        text = ws.xml_write(0).replace("\n0\n", f"\n{bad}\n")
        assert not acceptor.accepts(xml, text), bad
    # The other tools of the list: a call with only the required arguments, and a forward_* one.
    assert acceptor.accepts(xml, _xml("diff", []))
    assert acceptor.accepts(xml, _xml("revert", []))
    assert acceptor.accepts(xml, _xml("forward_score", [("prompts", '["a"]'), ("mode", '"max"')]))
    assert not acceptor.accepts(xml, _xml("submit", []))  # answer is required


def test_a_response_format_with_an_annotation_only_value_compiles_and_accepts(llm):
    from miserver.api.json_schema_subset import for_llamacpp_converter

    schema = {"type": "object", "properties": {"values": {"description": "anything"}},
              "required": ["values"], "additionalProperties": False}
    from llama_cpp.llama_grammar import LlamaGrammar

    grammar = LlamaGrammar.from_json_schema(json.dumps(for_llamacpp_converter(schema)),
                                            verbose=False)._grammar
    grammar = lt.strict_json_strings(grammar)
    acceptor = _Acceptor(llm)
    for doc in ('{"values": [[1, 2.5], [3]]}', '{"values": 0.5}', '{"values": "x"}'):
        assert acceptor.accepts(grammar, doc), doc
    assert not acceptor.accepts(grammar, '{"values": [1, }')
    assert not acceptor.accepts(grammar, '{"other": 1}')
