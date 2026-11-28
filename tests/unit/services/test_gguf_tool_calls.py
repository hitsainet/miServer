"""Native function calling on the llama.cpp engine (GGUF), 2026-11-21.

`tools` on a GGUF model was refused before any load. miServer now renders the GGUF's own chat
template with the tools, the tool-call history and `chat_template_kwargs`, and generates a call under
a GBNF grammar after the template's call marker (`miserver/services/llamacpp_tools.py`). These tests
run with llama.cpp STUBBED (CI installs no llama-cpp-python): a fake `Llama` whose tokenizer is one
id per character, so the exact prompt each `create_completion` received is read back as text, and
whose replies are scripted. The template is the REAL one (`qwen3.6-27b-gguf.jinja`, read off
unsloth/Qwen3.6-27B-GGUF Q4_K_M's metadata; `qwen2.5-7b-instruct.jinja` for Hermes JSON).

`test_gguf_tool_calls_real.py` runs the same path on a REAL tiny GGUF model with the real wheel
(grammar acceptance by llama.cpp itself, the renderer against `Jinja2ChatFormatter`).

MUTATION CONTROLS (each turns this file red; all 24, with results: `0xcc/reviews/gguf_tool_calls_2026-11-21.md`):
  T1 the routing line in `_llamacpp_chat_completion` (`tool_run = ... if ... else None` -> None)
  T2 the routing line in `_llamacpp_stream_chat_completion`
  T3 `grammar=self.run.grammar` dropped from the call phase
  T4 the (CHAT, LC) `tools` cell back to refused
  T5 the route's `check_llamacpp_tools` call removed
  T6 `validate_arguments` not called (bounds unenforced)
  T7 the marker search ignores the reasoning block (a marker inside `<think>` triggers a call)
  T8 `tools=` not passed to the template render
  T9 `ENGINE_USED_WHEN` removed (chat_template_kwargs reported as ignored while used)
  T10 the stop-with-tools refusal removed
  T11 `parallel` ignored by the grammar (repetition always allowed)
  T12 `strict_json_strings` not applied (the converter's control-character rule kept)
"""

from __future__ import annotations

import json
import sys
import types
from datetime import datetime
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from miserver.api import json_schema_subset
from miserver.api.request_policy import (
    OUTPUT_CHANGING,
    Endpoint,
    Engine,
    Honoured,
    evaluate,
)
from miserver.api.schemas.openai import ChatCompletionRequest
from miserver.core.errors import ConstrainedOutputInvalidError, FieldNotHonouredError
from miserver.ml.model_loader import ENGINE_LLAMACPP, LoadedModel
from miserver.services import llamacpp_tools as lt
from miserver.services.tool_calls import HERMES, QWEN3_XML, detect_format, parse_qwen3_xml, plan_tools
from tests.unit.f25_fixtures import clear_loaded, make_client, model_row, unloaded_inference

TEMPLATES = Path(__file__).resolve().parents[2] / "fixtures" / "chat_templates"
QWEN36 = (TEMPLATES / "qwen3.6-27b-gguf.jinja").read_text()
QWEN25 = (TEMPLATES / "qwen2.5-7b-instruct.jinja").read_text()
LFM2 = (TEMPLATES / "lfm2.5-1.2b-instruct.jinja").read_text()
URL = "/v1/chat/completions"

DESCRIBE = {"type": "function", "function": {
    "name": "describe_tensor", "description": "Summary statistics of one tensor",
    "parameters": {"type": "object", "properties": {"name": {"type": "string"}},
                   "required": ["name"]}}}
SCALE = {"type": "function", "function": {
    "name": "scale_tensor", "description": "Multiply one tensor by a factor (é)",
    "parameters": {"type": "object", "properties": {
        "name": {"type": "string"},
        "factor": {"type": "number", "minimum": 0, "maximum": 10},
        "layers": {"type": "array", "items": {"type": "integer"}},
        "mode": {"type": "string", "enum": ["mul", "add"]},
    }, "required": ["name", "factor"]}}}
TOOLS = [DESCRIBE, SCALE]
USER = [{"role": "user", "content": "Scale weight by 2"}]
NO_THINK = {"enable_thinking": False}

XML_CALL = ("\n<function=scale_tensor>\n<parameter=name>\nweight\n</parameter>\n"
            "<parameter=factor>\n2\n</parameter>\n</function>\n</tool_call>")


def chat(**over: Any) -> ChatCompletionRequest:
    body: dict[str, Any] = {"model": "m", "messages": USER, "tools": TOOLS, "max_tokens": 64,
                            "chat_template_kwargs": NO_THINK}
    body.update(over)
    return ChatCompletionRequest(**body)


@pytest.fixture(autouse=True)
def _clean():
    clear_loaded()
    yield
    clear_loaded()


# =============================================================================
# Stand-ins: llama_cpp's grammar module, and a Llama with a character tokenizer
# =============================================================================


class _Grammar:
    def __init__(self, text: str):
        self._grammar = text
        self._root = "root"


class FakeConverter:
    """`SchemaConverter`'s surface: each visited schema becomes one named rule."""

    def __init__(self, **_kw: Any):
        self.visited: list[tuple[dict, str]] = []
        self._rules: dict[str, str] = {"space": '" "?'}

    def resolve_refs(self, schema: dict, url: str) -> dict:
        return schema

    def visit(self, schema: dict, name: str) -> str:
        self.visited.append((schema, name))
        rule = name.replace("_", "-") or "root"
        self._rules[rule] = f"json-{schema.get('type', 'any')}"
        return rule

    def format_grammar(self) -> str:
        return "\n".join(f"{k} ::= {v}" for k, v in sorted(self._rules.items()))


class FakeLlamaCpp:
    def __init__(self, monkeypatch, *, parses: bool = True):
        self.converters: list[FakeConverter] = []
        self.parsed: list[str] = []
        fake = self

        class LlamaGrammar:
            @classmethod
            def from_string(cls, text, verbose=True):
                return _Grammar(text)

            @classmethod
            def from_json_schema(cls, text, verbose=True):
                return _Grammar(lt.CONVERTER_CHAR_RULE + "\nroot ::= char*")

        def converter(**kw):
            c = FakeConverter(**kw)
            fake.converters.append(c)
            return c

        grammar_mod = types.ModuleType("llama_cpp.llama_grammar")
        grammar_mod.LlamaGrammar = LlamaGrammar
        grammar_mod.SchemaConverter = converter
        grammar_mod.JSON_GBNF = "root ::= object"
        top = types.ModuleType("llama_cpp")
        top.llama_grammar = grammar_mod

        def init(vocab, text, root):
            fake.parsed.append(text.decode())
            return object() if parses else None

        top.llama_sampler_init_grammar = init
        top.llama_sampler_free = lambda _s: None
        monkeypatch.setitem(sys.modules, "llama_cpp", top)
        monkeypatch.setitem(sys.modules, "llama_cpp.llama_grammar", grammar_mod)


class FakeLlama:
    """A `Llama` reduced to what the tool path calls. Token ids are code points, so a prompt reads
    back as text; `replies` are consumed one per `create_completion`, each `(text, finish)`."""

    def __init__(self, template: str | None, replies: list[tuple[str, str]]):
        self.metadata = {"tokenizer.chat_template": template} if template is not None else {}
        self.replies = list(replies)
        self.calls: list[dict[str, Any]] = []
        self.closed = 0
        inner = MagicMock()
        inner.token_get_text = lambda t: {1: "<|endoftext|>", 2: "<|im_end|>"}[t]
        inner.vocab = "vocab"
        self._model = inner

    def token_eos(self) -> int:
        return 2

    def token_bos(self) -> int:
        return 1

    def tokenize(self, text: bytes, add_bos: bool = True, special: bool = False) -> list[int]:
        assert add_bos is False and special is True  # as llama-cpp-python's chat handler does
        return [ord(c) for c in text.decode("utf-8")]

    @staticmethod
    def text_of(tokens: list[int]) -> str:
        return "".join(chr(t) for t in tokens)

    def create_completion(self, prompt, stream=False, **kw):
        text, finish = self.replies.pop(0)
        self.calls.append({"prompt": self.text_of(prompt), "stream": stream, **kw})
        if not stream:
            return {"choices": [{"text": text, "finish_reason": finish}],
                    "usage": {"completion_tokens": len(text)}}
        fake = self

        def chunks():
            try:
                for i in range(0, len(text), 3):
                    yield {"choices": [{"text": text[i:i + 3], "finish_reason": None}]}
                yield {"choices": [{"text": "", "finish_reason": finish}]}
            finally:
                fake.closed += 1

        return chunks()


def service(llama: FakeLlama):
    from miserver.services.inference_service import InferenceService

    svc = InferenceService.__new__(InferenceService)
    state = MagicMock()
    state.is_loaded = True
    state.current = LoadedModel(model_id=1, model_name="m", model=llama, tokenizer=None,
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


async def complete(llama: FakeLlama, request: ChatCompletionRequest):
    return await service(llama)._llamacpp_chat_completion(request)


async def stream(llama: FakeLlama, request: ChatCompletionRequest) -> list[dict]:
    out = []
    async for frame in service(llama)._llamacpp_stream_chat_completion(request):
        body = frame[len("data: "):].strip()
        if body != "[DONE]":
            out.append(json.loads(body))
    return out


# =============================================================================
# The policy and the route: everything decidable before the load
# =============================================================================


class TestThePolicy:
    @pytest.mark.parametrize("name", ["tools", "tool_choice", "parallel_tool_calls"])
    def test_the_llamacpp_chat_cell_is_honoured(self, name):
        assert isinstance(OUTPUT_CHANGING[name][(Endpoint.CHAT, Engine.LLAMACPP)], Honoured)
        evaluate(chat(tool_choice="required", parallel_tool_calls=False), Endpoint.CHAT,
                 Engine.LLAMACPP, strict=True)

    def test_a_partial_assistant_turn_with_tools_is_refused_on_llamacpp(self):
        messages = USER + [{"role": "assistant", "content": "Let me"}]
        with pytest.raises(FieldNotHonouredError) as exc:
            evaluate(chat(messages=messages), Endpoint.CHAT, Engine.LLAMACPP, strict=False)
        assert exc.value.details["param"] == "tools"
        assert "partial assistant turn" in exc.value.message
        # Without tools shown (tool_choice none) the continuation is served.
        evaluate(chat(messages=messages, tool_choice="none"), Endpoint.CHAT, Engine.LLAMACPP,
                 strict=False)

    def test_chat_template_kwargs_is_used_on_the_tool_path_only(self):
        """It reaches the template miServer renders; on llama-cpp-python's own path it does not."""
        used = evaluate(chat(), Endpoint.CHAT, Engine.LLAMACPP, strict=True)
        assert "chat_template_kwargs" not in used.unused
        history = USER + [
            {"role": "assistant", "content": None, "tool_calls": [{
                "id": "c1", "type": "function",
                "function": {"name": "describe_tensor", "arguments": '{"name": "w"}'}}]},
            {"role": "tool", "tool_call_id": "c1", "content": "norm 5"},
        ]
        assert evaluate(chat(tools=None, messages=history), Endpoint.CHAT, Engine.LLAMACPP,
                        strict=True).unused == []
        plain = evaluate(chat(tools=None), Endpoint.CHAT, Engine.LLAMACPP, strict=False)
        assert plain.unused == ["chat_template_kwargs"]


def _post(body: dict, *, headers: dict | None = None):
    client, svc = make_client(unloaded_inference(), model_row(gguf_files=["m.gguf"]))
    r = client.post(URL, json={"model": "tiny", "messages": USER, **body}, headers=headers or {})
    return r, svc


class TestTheRoute:
    @pytest.mark.parametrize("over", [
        {"tools": TOOLS},
        {"tools": TOOLS, "tool_choice": "required", "parallel_tool_calls": False},
        {"tools": TOOLS, "tool_choice": {"type": "function", "function": {"name": "scale_tensor"}}},
        {"tools": TOOLS, "stream": True, "chat_template_kwargs": NO_THINK},
    ])
    def test_a_servable_tool_request_reaches_the_load(self, over):
        r, svc = _post(over, headers={"X-miLLM-Strict": "true"})
        assert svc.load_model_and_wait.call_count == 1, r.text

    @pytest.mark.parametrize("schema, keyword, pointer", [
        ({"type": "object", "properties": {"x": {"type": "string", "pattern": "^a$"}}},
         "pattern", "/tools/0/function/parameters/properties/x/pattern"),
        ({"type": "object", "properties": {"x": {"type": "string", "format": "date"}}},
         "format", "/tools/0/function/parameters/properties/x/format"),
        ({"type": "object", "properties": {"x": {"type": "integer"}}, "required": ["y"]},
         "required 'y'", "/tools/0/function/parameters/required"),
    ])
    def test_a_parameter_keyword_the_grammar_would_drop_is_refused_before_the_load(
        self, schema, keyword, pointer,
    ):
        tool = {"type": "function", "function": {"name": "f", "parameters": schema}}
        r, svc = _post({"tools": [tool]})
        assert r.status_code == 400, r.text
        err = r.json()["error"]
        assert err["code"] == "field_not_honoured" and err["param"] == "tools"
        assert keyword in err["message"] and pointer in err["message"]
        assert svc.load_model_and_wait.call_count == 0

    def test_minimum_and_maximum_are_accepted(self):
        """Validated after generation instead (`TestBounds`); `response_format` still refuses
        them."""
        r, svc = _post({"tools": [SCALE]})
        assert svc.load_model_and_wait.call_count == 1, r.text
        assert json_schema_subset.llamacpp_tool_offending_keywords([SCALE]) == []

    def test_stop_with_tools_is_refused_before_the_load(self):
        r, svc = _post({"tools": TOOLS, "stop": ["\n"]})
        assert r.status_code == 400, r.text
        assert r.json()["error"]["param"] == "stop"
        assert svc.load_model_and_wait.call_count == 0
        # Without tools a stop string is served as before.
        r, svc = _post({"stop": ["\n"]})
        assert svc.load_model_and_wait.call_count == 1, r.text

    def test_a_transformers_row_is_unchanged(self):
        """The llama.cpp schema check is for GGUF rows: a pattern in a tool's parameters is passed
        to a transformers template as before."""
        tool = {"type": "function", "function": {"name": "f", "parameters": {
            "type": "object", "properties": {"x": {"type": "string", "pattern": "^a$"}}}}}
        client, svc = make_client(unloaded_inference(), model_row())
        r = client.post(URL, json={"model": "tiny", "messages": USER, "tools": [tool]})
        assert svc.load_model_and_wait.call_count == 1, r.text


# =============================================================================
# The format, the parser and the grammar
# =============================================================================


class TestTheFormat:
    def test_the_qwen36_template_is_the_xml_format_and_qwen25_is_hermes(self):
        assert detect_format(QWEN36) == QWEN3_XML
        assert detect_format(QWEN25) == HERMES

    def test_xml_values_raw_for_strings_json_otherwise(self):
        plan = plan_tools(chat(), types.SimpleNamespace(chat_template=QWEN36))
        text = ("Sure.\n\n<tool_call>" + XML_CALL.replace(
            "</function>", "<parameter=layers>\n[1, 2]\n</parameter>\n"
            "<parameter=mode>\nmul\n</parameter>\n</function>"))
        outside, calls = parse_qwen3_xml(text, plan, strict=True)
        assert outside.strip() == "Sure."
        assert calls == [("scale_tensor", {"name": "weight", "factor": 2, "layers": [1, 2],
                                           "mode": "mul"})]

    def test_a_raw_value_keeps_its_inner_newlines_and_angle_brackets(self):
        plan = plan_tools(chat(), types.SimpleNamespace(chat_template=QWEN36))
        text = ("<tool_call>\n<function=describe_tensor>\n<parameter=name>\nif a < b:\n  x\n\n"
                "</parameter>\n</function>\n</tool_call>")
        assert parse_qwen3_xml(text, plan, strict=True)[1] == [
            ("describe_tensor", {"name": "if a < b:\n  x\n"})]

    @pytest.mark.parametrize("text", [
        "<tool_call>\n<function=scale_tensor>\n<parameter=factor>\ntwo\n</parameter>\n"
        "</function>\n</tool_call>",  # a number that is not JSON
        "<tool_call>\n<function=scale_tensor>\n<parameter=name>\na\n</parameter>\n"
        "<parameter=name>\nb\n</parameter>\n</function>\n</tool_call>",  # a parameter twice
        "<tool_call>\n<function=scale_tensor>\nstray\n</function>\n</tool_call>",
        '<tool_call>\n{"name": "scale_tensor", "arguments": {}}\n</tool_call>',
    ])
    def test_strict_parsing_refuses_what_the_format_cannot_be(self, text):
        plan = plan_tools(chat(), types.SimpleNamespace(chat_template=QWEN36))
        assert parse_qwen3_xml(text, plan, strict=True) is None

    def test_the_transformers_parser_returns_xml_calls(self):
        """The transformers engine reads the same format (it was taken for Hermes JSON there, and
        a Qwen3.6 call came back as content)."""
        from miserver.services.tool_calls import parse_tool_calls

        plan = plan_tools(chat(), types.SimpleNamespace(chat_template=QWEN36))
        parsed = parse_tool_calls("<tool_call>" + XML_CALL, plan)
        assert parsed is not None and parsed.content is None
        (tc,) = parsed.tool_calls
        assert json.loads(tc["function"]["arguments"]) == {"name": "weight", "factor": 2}

    def test_the_transformers_prefill_opens_the_xml_call(self):
        tok = types.SimpleNamespace(chat_template=QWEN36)
        assert plan_tools(chat(tool_choice="required"), tok).prefill == "<tool_call>\n<function="
        named = {"type": "function", "function": {"name": "scale_tensor"}}
        assert plan_tools(chat(tool_choice=named), tok).prefill == \
            "<tool_call>\n<function=scale_tensor>\n"


def _grammar(fmt: str, tools=TOOLS, *, allowed=None, parallel=True):
    conv = FakeConverter()
    names = frozenset(t["function"]["name"] for t in tools)
    return lt.build_call_grammar(fmt, tools, allowed=allowed or names, parallel=parallel,
                                 converter=conv), conv


class TestTheGrammar:
    def test_xml_raw_strings_bounded_enum_and_json_values(self):
        text, conv = _grammar(QWEN3_XML)
        assert 'tc-f1 ::= "<function=scale_tensor>\\n" "<parameter=name>\\n" tc-xtext ' \
               '"\\n</parameter>\\n"' in text
        assert '( "mul" | "add" )' in text  # a string enum is raw literals
        # Only the JSON-typed parameters reach the converter; required ones are not optional.
        assert [name for _, name in conv.visited] == ["t1-factor", "t1-layers"]
        assert '( "<parameter=layers>\\n" t1-layers "\\n</parameter>\\n" )?' in text
        assert '"<parameter=factor>\\n" t1-factor' in text
        assert 'root ::= tc-call ( "\\n<tool_call>" tc-call )*' in text

    def test_parallel_false_admits_one_call(self):
        text, _ = _grammar(QWEN3_XML, parallel=False)
        assert "root ::= tc-call\n" in text and "tc-call )*" not in text

    def test_a_named_function_is_the_only_alternative(self):
        text, _ = _grammar(QWEN3_XML, allowed=frozenset({"describe_tensor"}))
        assert "tc-f0" in text and "tc-f1" not in text

    def test_hermes_visits_one_call_object_per_tool(self):
        text, conv = _grammar(HERMES)
        schema, name = conv.visited[1]
        assert name == "t1-call"
        assert schema["properties"]["name"] == {"const": "scale_tensor"}
        assert schema["properties"]["arguments"] == SCALE["function"]["parameters"]
        assert schema["required"] == ["name", "arguments"]
        assert 'tc-call ::= "\\n" ( t0-call | t1-call ) "\\n"? "</tool_call>"' in text

    def test_defs_are_prefixed_per_tool(self):
        tool = {"type": "function", "function": {"name": "f", "parameters": {
            "type": "object", "$defs": {"Item": {"type": "integer"}},
            "properties": {"x": {"$ref": "#/$defs/Item"}}}}}
        _, conv = _grammar(HERMES, [tool])
        schema = conv.visited[0][0]
        assert schema["$defs"] == {"t0-Item": {"type": "integer"}}
        assert schema["properties"]["arguments"]["properties"]["x"] == {"$ref": "#/$defs/t0-Item"}

    @pytest.mark.parametrize("parameters, fragment", [
        ({"anyOf": [{"type": "object"}]}, "object schema"),
        ({"type": "object", "properties": {"a<b": {"type": "string"}}}, "cannot be written"),
        ({"type": "object", "properties": {"x": {"type": "string",
                                                  "enum": ["ok", "</parameter>"]}}},
         "'</parameter'"),
    ])
    def test_what_the_xml_format_cannot_express_is_refused(self, parameters, fragment):
        tool = {"type": "function", "function": {"name": "f", "parameters": parameters}}
        with pytest.raises(FieldNotHonouredError) as exc:
            _grammar(QWEN3_XML, [tool])
        assert exc.value.details["param"] == "tools" and fragment in exc.value.message

    def test_the_converter_string_character_is_made_json(self):
        text = lt.strict_json_strings("a ::= b\n" + lt.CONVERTER_CHAR_RULE + "\n")
        assert lt.STRICT_CHAR_RULE in text.splitlines()
        assert lt.CONVERTER_CHAR_RULE not in text.splitlines()
        assert "\\x00-\\x1F" in lt.STRICT_CHAR_RULE

    def test_the_tool_grammar_carries_the_strict_rule_not_the_converters(self):
        """The rewrite is applied to the grammar llama.cpp receives (mutation T12a survived without
        this: the stand-in converter emitted no `char` rule)."""

        class WithChar(FakeConverter):
            def format_grammar(self) -> str:
                return super().format_grammar() + "\n" + lt.CONVERTER_CHAR_RULE

        names = frozenset(t["function"]["name"] for t in TOOLS)
        for fmt in (HERMES, QWEN3_XML):
            text = lt.build_call_grammar(fmt, TOOLS, allowed=names, parallel=True,
                                         converter=WithChar())
            assert lt.STRICT_CHAR_RULE in text.splitlines(), fmt
            assert lt.CONVERTER_CHAR_RULE not in text.splitlines(), fmt

    def test_the_real_converter_rule_text_is_the_one_rewritten(self):
        """Pins the literal against llama-cpp-python 0.3.36's `PRIMITIVE_RULES["char"]` (the real
        test file checks it against the installed wheel)."""
        assert lt.CONVERTER_CHAR_RULE == (
            'char ::= [^"\\\\] | "\\\\" (["\\\\/bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] '
            '[0-9a-fA-F])'
        )


class TestTheRender:
    def test_tools_render_as_the_template_writes_them_tojson_unsorted_unescaped(self):
        prompt = lt.render_chat_prompt(QWEN36, [{"role": "user", "content": "hi"}],
                                       bos="", eos="<|im_end|>", tools=TOOLS,
                                       template_kwargs=NO_THINK)
        assert "<tools>\n" + json.dumps(DESCRIBE, ensure_ascii=False) in prompt
        assert "(é)" in prompt and '"type": "function", "function"' in prompt
        assert prompt.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")
        assert not lt.prompt_opens_think(prompt)
        thinking = lt.render_chat_prompt(QWEN36, [{"role": "user", "content": "hi"}], bos="",
                                         eos="<|im_end|>", tools=TOOLS)
        assert lt.prompt_opens_think(thinking)

    @pytest.mark.parametrize("text, opened, start", [
        ("abc", False, 0), ("", False, None), ("<th", False, None), ("<to", False, 0),
        ("<think>x", False, None), ("<think>x</think>y", False, 16), ("r</think>a", True, 9),
        ("reasoning", True, None),
    ])
    def test_the_answer_starts_after_the_reasoning(self, text, opened, start):
        assert lt.answer_start(text, opened) == start


# =============================================================================
# The service: what reaches llama.cpp, and what comes back
# =============================================================================


class TestNamedAndRequired:
    async def test_a_named_call_renders_the_tools_and_is_generated_under_the_grammar(
        self, monkeypatch,
    ):
        fake = FakeLlamaCpp(monkeypatch)
        llama = FakeLlama(QWEN36, [(XML_CALL, "stop")])
        named = {"type": "function", "function": {"name": "scale_tensor"}}
        resp = await complete(llama, chat(tool_choice=named))
        (call,) = llama.calls  # no free phase: the template's thinking is off
        # The prompt: the model's own template, with the tools, then the opened call.
        prompt = call["prompt"]
        assert prompt.startswith("<|im_start|>system\n# Tools\n\nYou have access")
        assert json.dumps(SCALE, ensure_ascii=False) in prompt
        assert prompt.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n<tool_call>")
        # The grammar: the one built for this request, parsed natively, admitting only the named
        # function, passed to generation.
        grammar = call["grammar"]
        assert isinstance(grammar, _Grammar) and fake.parsed == [grammar._grammar]
        assert "<function=scale_tensor>" in grammar._grammar
        assert "<function=describe_tensor>" not in grammar._grammar
        assert call["stream"] is False and "stop" not in call
        choice = resp.choices[0]
        assert choice.finish_reason == "tool_calls"
        assert choice.message.content is None
        (tc,) = choice.message.tool_calls
        assert tc["type"] == "function" and tc["id"].startswith("call_")
        assert tc["function"]["name"] == "scale_tensor"
        assert json.loads(tc["function"]["arguments"]) == {"name": "weight", "factor": 2}
        assert resp.usage.prompt_tokens == len(prompt) - len("<tool_call>")
        assert resp.usage.completion_tokens == len(XML_CALL)

    async def test_required_with_thinking_reasons_freely_then_calls(self, monkeypatch):
        FakeLlamaCpp(monkeypatch)
        llama = FakeLlama(QWEN36, [("I should scale it.\n</think>\n\nignored", "stop"),
                                   (XML_CALL, "stop")])
        resp = await complete(llama, chat(tool_choice="required", chat_template_kwargs=None))
        free, call = llama.calls
        assert free["stream"] is True and "grammar" not in free
        assert free["prompt"].endswith("<|im_start|>assistant\n<think>\n")
        assert call["prompt"].endswith(
            "<think>\nI should scale it.\n</think>\n\n<tool_call>")
        assert call["grammar"] is not None
        message = resp.choices[0].message
        assert message.reasoning_content == "I should scale it."
        assert message.content is None and len(message.tool_calls) == 1
        assert llama.closed == 1  # the free stream was closed when the reasoning ended

    async def test_a_budget_ended_call_is_length_and_never_content(self, monkeypatch):
        """2026-11-21 (WSP run wsp_84dddc47ecaf): a call the budget cut short is NOT returned as
        text (it was, `<tool_call>\n<function=python`): `length`, no marker, no half call."""
        FakeLlamaCpp(monkeypatch)
        llama = FakeLlama(QWEN36, [("\n<function=scale_tensor>\n<parameter=na", "length")])
        resp = await complete(llama, chat(tool_choice="required"))
        choice = resp.choices[0]
        assert choice.finish_reason == "length" and not choice.message.tool_calls
        assert not choice.message.content
        assert "<tool_call>" not in (choice.message.content or "")


class TestAuto:
    async def test_text_then_a_marker_then_the_constrained_call(self, monkeypatch):
        FakeLlamaCpp(monkeypatch)
        llama = FakeLlama(QWEN36, [("Let me scale it.\n\n<tool_call>\n<function=junk", "stop"),
                                   (XML_CALL, "stop")])
        resp = await complete(llama, chat())
        free, call = llama.calls
        assert "grammar" not in free
        assert call["prompt"].endswith("</think>\n\nLet me scale it.\n\n<tool_call>")
        assert llama.closed == 1  # cut at the marker: what followed it was never used
        message = resp.choices[0].message
        assert message.content == "Let me scale it."
        assert message.tool_calls[0]["function"]["name"] == "scale_tensor"
        assert resp.choices[0].finish_reason == "tool_calls"

    async def test_a_plain_answer_is_one_free_generation(self, monkeypatch):
        FakeLlamaCpp(monkeypatch)
        llama = FakeLlama(QWEN36, [("The norm is 5 <b>exactly</b>.", "stop")])
        resp = await complete(llama, chat())
        assert len(llama.calls) == 1
        choice = resp.choices[0]
        assert choice.finish_reason == "stop" and not choice.message.tool_calls
        assert choice.message.content == "The norm is 5 <b>exactly</b>."

    async def test_a_marker_inside_the_reasoning_is_not_a_call(self, monkeypatch):
        FakeLlamaCpp(monkeypatch)
        llama = FakeLlama(QWEN36, [
            ("Maybe <tool_call> describe?\n</think>\n\nignored", "stop"),
            ("No call needed.", "stop")])
        resp = await complete(llama, chat(chat_template_kwargs=None))
        reason, answer = llama.calls  # the reasoning phase, then the answer after the block
        assert "grammar" not in reason and "grammar" not in answer
        assert answer["prompt"].endswith("<think>\nMaybe <tool_call> describe?\n</think>\n\n")
        message = resp.choices[0].message
        assert message.reasoning_content == "Maybe <tool_call> describe?"
        assert message.content == "No call needed." and not message.tool_calls

    async def test_hermes_json_calls(self, monkeypatch):
        FakeLlamaCpp(monkeypatch)
        body = '\n{"name": "describe_tensor", "arguments": {"name": "w"}}\n</tool_call>'
        llama = FakeLlama(QWEN25, [("<tool_call>", "stop"), (body, "stop")])
        resp = await complete(llama, chat(chat_template_kwargs=None))
        assert llama.calls[1]["prompt"].endswith("<|im_start|>assistant\n<tool_call>")
        (tc,) = resp.choices[0].message.tool_calls
        assert json.loads(tc["function"]["arguments"]) == {"name": "w"}


class TestBounds:
    async def test_a_call_outside_its_bounds_is_500_never_200(self, monkeypatch):
        FakeLlamaCpp(monkeypatch)
        llama = FakeLlama(QWEN36, [(XML_CALL.replace("\n2\n", "\n11\n"), "stop")])
        with pytest.raises(ConstrainedOutputInvalidError) as exc:
            await complete(llama, chat(tool_choice="required"))
        assert exc.value.details == {"tool": "scale_tensor", "keyword": "maximum",
                                     "pointer": "/factor"}
        assert exc.value.status_code == 500 and exc.value.code == "CONSTRAINED_OUTPUT_INVALID"
        # The client sees why (the schema's bound, never the model's value), so it can retry.
        from miserver.api.routes.openai.errors import client_safe_message

        message = client_safe_message(exc.value, 500, event="test")
        assert "'scale_tensor'" in message and "maximum 10 at /factor" in message
        assert "11" not in message

    async def test_streamed_it_is_an_error_event_not_a_call(self, monkeypatch):
        FakeLlamaCpp(monkeypatch)
        llama = FakeLlama(QWEN36, [(XML_CALL.replace("\n2\n", "\n-1\n"), "stop")])
        events = await stream(llama, chat(tool_choice="required", stream=True))
        errors = [e for e in events if "error" in e]
        assert errors and errors[0]["error"]["code"] == "constrained_output_invalid"
        assert "minimum 0 at /factor" in errors[0]["error"]["message"]
        assert not any(c.get("delta", {}).get("tool_calls")
                       for e in events for c in e.get("choices", []))


class TestHistory:
    HISTORY = USER + [
        {"role": "assistant", "content": None, "tool_calls": [{
            "id": "c1", "type": "function",
            "function": {"name": "describe_tensor", "arguments": '{"name": "weight"}'}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "shape (4096,), norm 5.21"},
    ]

    async def test_history_reaches_the_template_in_its_own_format(self, monkeypatch):
        FakeLlamaCpp(monkeypatch)
        llama = FakeLlama(QWEN36, [("The norm is 5.21.", "stop")])
        resp = await complete(llama, chat(messages=self.HISTORY))
        prompt = llama.calls[0]["prompt"]
        assert ("<tool_call>\n<function=describe_tensor>\n<parameter=name>\nweight\n"
                "</parameter>\n</function>\n</tool_call>") in prompt
        assert "<tool_response>\nshape (4096,), norm 5.21\n</tool_response>" in prompt
        assert resp.choices[0].message.content == "The norm is 5.21."

    async def test_history_alone_is_rendered_and_never_constrained(self, monkeypatch):
        fake = FakeLlamaCpp(monkeypatch)
        llama = FakeLlama(QWEN36, [("Done.", "stop")])
        resp = await complete(llama, chat(messages=self.HISTORY, tools=None))
        (call,) = llama.calls
        assert "<tool_response>" in call["prompt"] and "# Tools" not in call["prompt"]
        assert "grammar" not in call and fake.parsed == []
        assert resp.choices[0].message.content == "Done."


class TestStreaming:
    async def test_content_streams_then_one_tool_calls_delta(self, monkeypatch):
        FakeLlamaCpp(monkeypatch)
        llama = FakeLlama(QWEN36, [("Scaling now.\n<tool_call>", "stop"), (XML_CALL, "stop")])
        events = await stream(llama, chat(stream=True))
        deltas = [c["delta"] for e in events for c in e.get("choices", [])]
        content = "".join(d.get("content") or "" for d in deltas)
        assert content == "Scaling now.\n"  # never the marker, never the call
        calls = [d["tool_calls"] for d in deltas if d.get("tool_calls")]
        assert len(calls) == 1 and calls[0][0]["index"] == 0
        assert calls[0][0]["function"]["name"] == "scale_tensor"
        finishes = [c.get("finish_reason") for e in events for c in e.get("choices", [])]
        assert finishes[-1] == "tool_calls"
        final = [e for e in events if e.get("usage")][-1]
        assert final["usage"]["completion_tokens"] == len("Scaling now.\n<tool_call>") + len(
            XML_CALL)

    async def test_a_budget_cut_call_streams_no_marker_and_no_call(self, monkeypatch):
        """What the grammar phase wrote is withheld while generated; cut by the budget, it is never
        sent (2026-11-21: a half-opened call is not content) and no call is returned."""
        FakeLlamaCpp(monkeypatch)
        partial = "\n<function=scale_tensor>\n<parameter=na"
        llama = FakeLlama(QWEN36, [(partial, "length")])
        events = await stream(llama, chat(stream=True, tool_choice="required"))
        deltas = [c["delta"] for e in events for c in e.get("choices", [])]
        assert "".join(d.get("content") or "" for d in deltas) == ""
        assert not any(d.get("tool_calls") for d in deltas)
        assert [c.get("finish_reason") for e in events for c in e.get("choices", [])][-1] == \
            "length"

    async def test_streamed_reasoning_is_reasoning_content(self, monkeypatch):
        FakeLlamaCpp(monkeypatch)
        llama = FakeLlama(QWEN36, [("think hard</think>\n\nignored", "stop"), ("Answer.", "stop")])
        events = await stream(llama, chat(stream=True, chat_template_kwargs=None))
        deltas = [c["delta"] for e in events for c in e.get("choices", [])]
        assert "".join(d.get("reasoning_content") or "" for d in deltas) == "think hard"
        assert "".join(d.get("content") or "" for d in deltas).strip() == "Answer."


class TestRefusals:
    @pytest.mark.parametrize("template, fragment", [
        (LFM2, "'lfm2' format"),
        ("{% for m in messages %}{{ m.content }}{% endfor %}", "takes no tools"),
        (None, "carries no chat template"),
    ])
    async def test_refused_before_any_generation(self, monkeypatch, template, fragment):
        FakeLlamaCpp(monkeypatch)
        llama = FakeLlama(template, [])
        with pytest.raises(FieldNotHonouredError) as exc:
            await complete(llama, chat())
        assert exc.value.details["param"] == "tools" and fragment in exc.value.message
        assert llama.calls == []

    async def test_a_grammar_llama_cpp_rejects_refuses(self, monkeypatch):
        FakeLlamaCpp(monkeypatch, parses=False)
        llama = FakeLlama(QWEN36, [])
        with pytest.raises(FieldNotHonouredError) as exc:
            await complete(llama, chat())
        assert "rejected the grammar" in exc.value.message and llama.calls == []

    async def test_history_without_a_template_names_messages(self, monkeypatch):
        FakeLlamaCpp(monkeypatch)
        with pytest.raises(FieldNotHonouredError) as exc:
            await complete(FakeLlama(None, []), chat(tools=None, messages=TestHistory.HISTORY))
        assert exc.value.details["param"] == "messages"

    async def test_a_direct_caller_with_stop_is_refused_in_the_service(self, monkeypatch):
        FakeLlamaCpp(monkeypatch)
        llama = FakeLlama(QWEN36, [])
        with pytest.raises(FieldNotHonouredError) as exc:
            await complete(llama, chat(stop=["x"]))
        assert exc.value.details["param"] == "stop" and llama.calls == []


class TestResponseFormatStrings:
    async def test_the_response_format_grammar_gets_json_string_characters(self, monkeypatch):
        """The same converter defect on the structured-output path (2026-11-20): a raw newline in
        a string failed validation as a 500. The grammar llama.cpp receives is the strict one."""
        FakeLlamaCpp(monkeypatch)
        llama = MagicMock()
        llama.metadata = {}
        llama.create_chat_completion.return_value = {
            "choices": [{"message": {"content": '"ok"'}, "finish_reason": "stop"}],
            "usage": {}}
        svc = service(llama)
        await svc._llamacpp_chat_completion(ChatCompletionRequest(
            model="m", messages=USER, response_format={
                "type": "json_schema", "json_schema": {"name": "s", "schema": {"type": "string"}}}))
        grammar = llama.create_chat_completion.call_args.kwargs["grammar"]._grammar
        assert lt.STRICT_CHAR_RULE in grammar.splitlines()
        assert lt.CONVERTER_CHAR_RULE not in grammar.splitlines()
