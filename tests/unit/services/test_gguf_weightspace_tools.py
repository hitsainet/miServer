"""weightspace's exact tool list on a GGUF model (2026-11-21; live failure, WSP run `wsp_c3e954d3f21f`).

The first call of a weightspace run on Qwen3.6-27B-GGUF:Q4_K_M (main `76d374fd`) was refused:
`400 'tools' is not honoured ... (no type the llama.cpp converter can generate) at
/tools/4/function/parameters/properties/values`. `tools[4]` is `write_tensor`, whose `values` is
`{"description": "nested list of numbers, or a scalar"}`: annotations alone. In JSON Schema such a
schema constrains nothing and accepts ANY JSON value, so a grammar of any JSON value enforces all of
it; llama-cpp-python 0.3.36's converter has no spelling for that (a node with only annotations fails
its `assert`, `{}` becomes its object rule), but its primitive table holds the generic `value` rule,
which `json_schema_subset.for_llamacpp_converter` hands it. Both grammar paths (tools and
`response_format`) take it.

`TOOLS` below is weightspace 0.3.0's list as it reaches the wire (`weightspace/agent.py`
`_builtin_tools`/`_probe_tools`, lines 153-232, through `tools.tool()`, which turns
`{"optional": True}` into absence from `required` and REMOVES the key, and `probes.
_schema_from_signature` for a `forward_*` tool, which gives an unannotated parameter `{}` or
`{"default": ...}`, annotations alone too). The `optional` key therefore never reaches miServer from
weightspace; sent literally it is accepted as an annotation with no effect (`ANNOTATIONS`).

llama.cpp's OWN grammar engine is checked on the same list in
`test_gguf_tool_calls_real.py::test_llama_cpp_accepts_weightspace_write_tensor_values` (wheel +
tiny GGUF; skips in CI).

MUTATION CONTROLS (results in `0xcc/reviews/gguf_structured_thinking_2026-11-21.md`):
  W1 the pre-check's annotation-only branch removed (`_lc_walk`)
  W2 the converter rewrite's annotation-only branch removed (`for_llamacpp_converter`)
  W3 the rewrite not applied on the tool path (`build_call_grammar`)
  W4 the rewrite not applied on the response_format path (`_llamacpp_grammar`)
  W5 `optional` dropped from `ANNOTATIONS`
"""

from __future__ import annotations

import copy
import json
from typing import Any

import pytest

from miserver.api import json_schema_subset as jss
from miserver.api.schemas.openai import ChatCompletionRequest
from miserver.services import llamacpp_tools as lt
from miserver.services.tool_calls import HERMES, QWEN3_XML
from tests.unit.f25_fixtures import clear_loaded, make_client, model_row, unloaded_inference
from tests.unit.services.test_gguf_tool_calls import (
    QWEN36,
    FakeConverter,
    FakeLlama,
    FakeLlamaCpp,
    service,
)


def _ws_tool(name: str, description: str, /, **props: dict) -> dict:
    """weightspace 0.3.0 `tools.tool()` + `llm.openai.to_openai_tools`, as written there."""
    props = {k: dict(v) for k, v in props.items()}
    required = [k for k, v in props.items() if not v.pop("optional", False)]
    return {"type": "function", "function": {
        "name": name, "description": description,
        "parameters": {"type": "object", "properties": props, "required": required}}}


TOOLS: list[dict[str, Any]] = [
    _ws_tool("list_tensors", "List weight tensors (name, shape, dtype, numel). Optional "
             "glob/substring filter.", pattern={"type": "string", "optional": True}),
    _ws_tool("describe_tensor", "Statistics of one tensor: quantiles, row/col norms, singular "
             "values.", name={"type": "string"}),
    _ws_tool("read_tensor", "Read raw values of a tensor or a numpy-style slice of it, e.g. "
             "index='0:4, :16'. At most 4096 values per call.", name={"type": "string"},
             index={"type": "string", "optional": True},
             precision={"type": "integer", "optional": True}),
    _ws_tool("python", "Run Python in a persistent namespace with np, torch, R (read-only "
             "weights), W (writable weights: edit in place or assign W[name] = array), diff(), "
             "checkpoint(label), restore(label), revert(name=None). The last expression is "
             "printed. Use it for static analysis of weights and for writing weights. It cannot "
             "run the network.", code={"type": "string"}),
    _ws_tool("write_tensor", "Overwrite a tensor or a slice of it with values (nested list or "
             "scalar).", name={"type": "string"},
             values={"description": "nested list of numbers, or a scalar"},
             index={"type": "string", "optional": True}),
    _ws_tool("diff", "Summarise all edits made so far versus the original weights."),
    _ws_tool("checkpoint", "Save the current weights under a label.", label={"type": "string"}),
    _ws_tool("restore", "Restore weights saved under a label.", label={"type": "string"}),
    _ws_tool("revert", "Revert one tensor (or all, if name is omitted) to the original weights.",
             name={"type": "string", "optional": True}),
    _ws_tool("submit", "Finish: give the final answer (and for edits, a summary of what you "
             "changed and why).", answer={"type": "string"},
             confidence={"type": "number", "optional": True}),
    # `_probe_tools`: `Tool(f"forward_{p.name}", f"[forward pass, costs {p.cost}] ...",
    # p.parameters)`, the parameters a JSON-schema object (`_schema_from_signature` of
    # `def score(prompts: list, layer: int = 12, mode="mean", note=None)`).
    {"type": "function", "function": {
        "name": "forward_score", "description": "[forward pass, costs 2] Score the prompts.",
        "parameters": {"type": "object", "properties": {
            "prompts": {"type": "array"},
            "layer": {"type": "integer", "default": 12},
            "mode": {"default": "mean"},
            "note": {"default": None},
        }, "required": ["prompts"]}}},
]
WRITE = 4
assert TOOLS[WRITE]["function"]["name"] == "write_tensor"  # the live failure's /tools/4
VALUES = TOOLS[WRITE]["function"]["parameters"]["properties"]["values"]
USER = [{"role": "user", "content": "Zero the first row of layers.0.weight"}]
NESTED, SCALAR = [[1, 2.5], [-3e-2, 0]], 0.5


def chat(**over: Any) -> ChatCompletionRequest:
    body: dict[str, Any] = {"model": "m", "messages": USER, "tools": TOOLS, "max_tokens": 256,
                            "tool_choice": "required", "parallel_tool_calls": False,
                            "chat_template_kwargs": {"enable_thinking": False}}
    body.update(over)
    return ChatCompletionRequest(**body)


def xml_write(values: Any) -> str:
    return ("\n<function=write_tensor>\n<parameter=name>\nlayers.0.weight\n</parameter>\n"
            f"<parameter=values>\n{json.dumps(values)}\n</parameter>\n</function>\n</tool_call>")


@pytest.fixture(autouse=True)
def _clean():
    clear_loaded()
    yield
    clear_loaded()


class TestThePreCheck:
    def test_the_whole_list_passes(self):
        assert jss.llamacpp_tool_offending_keywords(TOOLS) == []
        jss.check_llamacpp_tools(chat().tools)

    def test_the_live_failure_is_the_values_schema(self):
        assert jss.annotation_only(VALUES)
        assert jss.llamacpp_offending_keywords(VALUES) == []

    @pytest.mark.parametrize("schema", [
        {}, {"description": "d"}, {"title": "t", "examples": [1, [2]], "default": 0},
        {"$comment": "c", "deprecated": True, "readOnly": True, "writeOnly": False},
        {"optional": True}, {"$defs": {"x": {"type": "string"}}, "description": "d"},
    ])
    def test_annotations_alone_are_any_value(self, schema):
        assert jss.annotation_only(schema)
        assert jss.llamacpp_offending_keywords({"type": "object", "properties": {"v": schema}}) == []

    @pytest.mark.parametrize("schema", [
        {"maxLength": 3}, {"description": "d", "minItems": 1}, {"type": "string"},
        {"description": "d", "format": "uuid"},
    ])
    def test_a_constraint_beside_them_is_not(self, schema):
        assert not jss.annotation_only(schema)

    def test_a_literal_optional_is_an_annotation(self):
        """Sent as written (weightspace removes it): accepted, no effect on `required`."""
        literal = copy.deepcopy(TOOLS)
        literal[0]["function"]["parameters"]["properties"]["pattern"]["optional"] = True
        jss.check(literal[0]["function"]["parameters"])
        assert jss.llamacpp_tool_offending_keywords(literal) == []

    def test_the_route_reaches_the_load_under_strict(self):
        client, svc = make_client(unloaded_inference(), model_row(gguf_files=["m.gguf"]))
        r = client.post("/v1/chat/completions", headers={"X-miLLM-Strict": "true"},
                        json={"model": "tiny", "messages": USER, "tools": TOOLS,
                              "tool_choice": "auto", "parallel_tool_calls": False,
                              "chat_template_kwargs": {"enable_thinking": True}})
        assert svc.load_model_and_wait.call_count == 1, r.text


class TestTheConverterInput:
    def test_annotation_only_subschemas_become_the_value_rule(self):
        out = jss.for_llamacpp_converter({
            "type": "object",
            "properties": {"a": {"description": "x"}, "b": {"type": "array", "items": {}},
                           "c": {"anyOf": [{"title": "t"}, {"type": "null"}]},
                           "d": {"type": "object", "additionalProperties": {"default": 1}},
                           "e": {"type": "string", "description": "kept"}},
            "$defs": {"any": {"examples": [1]}},
        })
        assert out["properties"]["a"] == jss.LLAMACPP_ANY_VALUE
        assert out["properties"]["b"]["items"] == jss.LLAMACPP_ANY_VALUE
        assert out["properties"]["c"]["anyOf"][0] == jss.LLAMACPP_ANY_VALUE
        assert out["properties"]["d"]["additionalProperties"] == jss.LLAMACPP_ANY_VALUE
        assert out["properties"]["e"] == {"type": "string", "description": "kept"}
        assert out["$defs"]["any"] == jss.LLAMACPP_ANY_VALUE
        assert jss.for_llamacpp_converter({}) == jss.LLAMACPP_ANY_VALUE
        assert jss.for_llamacpp_converter({}, rewrite_root=False) == {}

    def test_the_request_schema_itself_is_untouched(self):
        before = copy.deepcopy(TOOLS)
        jss.for_llamacpp_converter(TOOLS[WRITE]["function"]["parameters"])
        assert TOOLS == before

    @pytest.mark.parametrize("fmt", [QWEN3_XML, HERMES])
    def test_the_tool_grammar_gives_values_the_value_rule(self, fmt):
        conv = FakeConverter()
        lt.build_call_grammar(fmt, TOOLS, allowed=frozenset({"write_tensor"}), parallel=False,
                              converter=conv)
        visited = [s for s, _ in conv.visited]
        if fmt == QWEN3_XML:
            assert jss.LLAMACPP_ANY_VALUE in visited  # values (name/index are raw text)
        else:
            (call,) = visited
            args = call["properties"]["arguments"]
            assert args["properties"]["values"] == jss.LLAMACPP_ANY_VALUE
            assert args["required"] == ["name", "values"]

    def test_the_response_format_grammar_gets_the_value_rule(self, monkeypatch):
        seen: list[str] = []
        FakeLlamaCpp(monkeypatch)
        import sys

        grammar_mod = sys.modules["llama_cpp.llama_grammar"]
        original = grammar_mod.LlamaGrammar.from_json_schema

        def spy(text, verbose=True):
            seen.append(text)
            return original(text, verbose)

        monkeypatch.setattr(grammar_mod.LlamaGrammar, "from_json_schema", staticmethod(spy))
        svc = service(FakeLlama(None, []))
        fmt = {"type": "json_schema", "json_schema": {"name": "w", "schema": {
            "type": "object", "properties": {"values": VALUES}, "required": ["values"]}}}
        svc._llamacpp_grammar(ChatCompletionRequest(model="m", messages=USER,
                                                    response_format=fmt).response_format)
        (sent,) = seen
        assert json.loads(sent)["properties"]["values"] == jss.LLAMACPP_ANY_VALUE


class TestACallThroughTheService:
    @pytest.mark.parametrize("values", [NESTED, SCALAR, "a string", None, {"k": [1]}],
                             ids=["nested", "scalar", "string", "null", "object"])
    async def test_write_tensor_with_any_values_is_a_call(self, monkeypatch, values):
        FakeLlamaCpp(monkeypatch)
        llama = FakeLlama(QWEN36, [(xml_write(values), "stop")])
        resp = await service(llama)._llamacpp_chat_completion(
            chat(tool_choice={"type": "function", "function": {"name": "write_tensor"}}))
        (call,) = resp.choices[0].message.tool_calls
        assert call["function"]["name"] == "write_tensor"
        assert json.loads(call["function"]["arguments"]) == {"name": "layers.0.weight",
                                                             "values": values}
        assert resp.choices[0].finish_reason == "tool_calls"


class TestAnAnnotationOnlyParametersRoot:
    def test_it_is_an_object_with_no_arguments(self):
        """W6: a call's arguments are always an object, so annotations alone at the ROOT of a
        tool's parameters mean "no declared arguments", never "any JSON value"."""
        tool = {"type": "function", "function": {"name": "ping",
                                                 "parameters": {"description": "no arguments"}}}
        assert lt.tool_parameters(tool) == {"type": "object", "properties": {}}
        conv = FakeConverter()
        lt.build_call_grammar(HERMES, [tool], allowed=frozenset({"ping"}), parallel=False,
                              converter=conv)
        ((call, _),) = conv.visited
        assert call["properties"]["arguments"] == {"type": "object", "properties": {}}
        assert jss.llamacpp_tool_offending_keywords([tool]) == []
