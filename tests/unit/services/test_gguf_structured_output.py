"""Structured output on the llama.cpp engine, and the empty inline steering set as no change
(2026-11-20; FR-25.10/FR-25.11 on GGUF, FR-25.3.4, FR-28.2.2).

miForge's WSP editor sends `response_format: {"type": "json_schema", ...}` and relies on miServer
to ENFORCE it. On a GGUF model the enforcement is a GBNF grammar miServer builds with
llama-cpp-python's own converter and hands to `create_chat_completion(grammar=...)`. These tests
pin, with llama.cpp stubbed (CI installs no llama-cpp-python):

* the policy cell, the route's llama.cpp schema check and the steering NEUTRAL entry;
* that the request's schema reaches the converter UNCHANGED and the grammar it returns reaches
  generation, and that llama-cpp-python's own `response_format` argument is never used (in
  0.3.36 it ignores OpenAI's `json_schema` type: measured on a real model, it generated
  unconstrained text — `0xcc/reviews/gguf_structured_output_2026-11-20.md`);
* that a schema that does not convert, or a grammar llama.cpp rejects, refuses and never
  generates; that a complete output is validated and a budget-ended one is "length";
* that streaming, a partial assistant turn, completions and n > 1 stay refused.

`test_gguf_structured_output_real.py` runs the same path against a REAL tiny GGUF model when
llama-cpp-python and the file are present.

MUTATION CONTROLS (each turns this file red; recorded in the review note):
  M1 the (CHAT, LC) `response_format` cell back to refused
  M2 the `steering` NEUTRAL entry removed
  M3 `params["grammar"] = grammar` removed (no grammar reaches llama.cpp)
  M4 the converter's exception swallowed (compile-failure refusal removed)
  M5 the native-parse refusal removed
  M6 the route's `check_llamacpp` call removed
  M7 the output validation skipped
  M8 `_refuse_llamacpp_steering` back to refusing the empty set
"""

from __future__ import annotations

import json
import sys
import types
from datetime import datetime
from unittest.mock import MagicMock

import pytest

from miserver.api import json_schema_subset
from miserver.api.request_policy import (
    NEUTRAL,
    OUTPUT_CHANGING,
    Endpoint,
    Engine,
    Honoured,
    evaluate,
)
from miserver.api.schemas.openai import (
    ChatCompletionRequest,
    EmbeddingRequest,
    TextCompletionRequest,
    is_empty_inline_set,
)
from miserver.core.errors import (
    ConstrainedOutputInvalidError,
    EngineUnsupportedError,
    FieldNotHonouredError,
    ResponseFormatUnsupportedError,
)
from miserver.ml.model_loader import ENGINE_LLAMACPP, LoadedModel
from tests.unit.f25_fixtures import clear_loaded, make_client, model_row, unloaded_inference

URL = "/v1/chat/completions"
MESSAGES = [{"role": "user", "content": "a"}]

#: A schema every keyword of which the llama.cpp grammar enforces (no minimum/maximum/pattern).
SCHEMA = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": ["inspect", "repair", "evaluate"]},
        "target": {"type": "string", "minLength": 1, "maxLength": 64},
        "rows": {"type": "array", "items": {"type": "integer"}, "minItems": 1, "maxItems": 4},
        "note": {"anyOf": [{"type": "string"}, {"type": "null"}]},
    },
    "required": ["action", "target", "rows"],
    "additionalProperties": False,
}
VALID = {"action": "repair", "target": "probe_7", "rows": [1, 2], "note": None}


def schema_format(schema=SCHEMA, name="wsp_action"):
    return {"type": "json_schema", "json_schema": {"name": name, "schema": schema}}


def chat(**over) -> ChatCompletionRequest:
    body = {"model": "m", "messages": MESSAGES, "max_tokens": 64,
            "response_format": schema_format()}
    body.update(over)
    return ChatCompletionRequest(**body)


@pytest.fixture(autouse=True)
def _clean():
    clear_loaded()
    yield
    clear_loaded()


# =============================================================================
# The policy table
# =============================================================================


class TestThePolicyCell:
    def test_response_format_is_honoured_on_llamacpp_chat(self):
        assert isinstance(OUTPUT_CHANGING["response_format"][(Endpoint.CHAT, Engine.LLAMACPP)],
                          Honoured)
        evaluate(chat(), Endpoint.CHAT, Engine.LLAMACPP, strict=True)

    def test_json_object_is_honoured_on_llamacpp_chat(self):
        evaluate(chat(response_format={"type": "json_object"}), Endpoint.CHAT,
                 Engine.LLAMACPP, strict=True)

    def test_a_partial_assistant_turn_is_refused_on_llamacpp_only(self):
        messages = MESSAGES + [{"role": "assistant", "content": '{"action": '}]
        with pytest.raises(FieldNotHonouredError) as exc:
            evaluate(chat(messages=messages), Endpoint.CHAT, Engine.LLAMACPP, strict=False)
        assert exc.value.details["param"] == "response_format"
        assert "partial assistant turn" in exc.value.message
        # Transformers renders the trailing turn as a fresh one; its cell is unchanged.
        evaluate(chat(messages=messages), Endpoint.CHAT, Engine.TRANSFORMERS, strict=False)
        # An EMPTY trailing assistant turn asks for a fresh answer: not a continuation.
        evaluate(chat(messages=MESSAGES + [{"role": "assistant", "content": " "}]),
                 Endpoint.CHAT, Engine.LLAMACPP, strict=False)

    def test_completions_and_n_stay_refused_on_llamacpp(self):
        req = TextCompletionRequest(model="m", prompt="a", response_format={"type": "json_object"})
        with pytest.raises(FieldNotHonouredError) as exc:
            evaluate(req, Endpoint.COMPLETIONS, Engine.LLAMACPP, strict=False)
        assert exc.value.details["param"] == "response_format"
        with pytest.raises(FieldNotHonouredError) as exc:
            evaluate(chat(n=2), Endpoint.CHAT, Engine.LLAMACPP, strict=False)
        assert exc.value.details["param"] == "n"


class TestTheEmptySteeringSetIsNeutral:
    EMPTY = {"features": []}
    ONE = {"features": [{"index": 1, "strength": 2.0}]}

    def test_the_predicate_reads_only_the_exact_form(self):
        assert NEUTRAL["steering"] is is_empty_inline_set
        assert is_empty_inline_set(chat(steering=self.EMPTY).steering)
        assert is_empty_inline_set({"features": []})
        assert not is_empty_inline_set(chat(steering=self.ONE).steering)
        assert not is_empty_inline_set(chat(steering={"features": [], "sae_id": None}).steering)
        assert not is_empty_inline_set({"features": [], "sae_id": None})
        assert not is_empty_inline_set({"features": [], "other": 1})
        assert not is_empty_inline_set({"features": {}})
        assert not is_empty_inline_set({})
        assert not is_empty_inline_set([])

    @pytest.mark.parametrize("endpoint", [Endpoint.CHAT, Endpoint.COMPLETIONS])
    def test_honoured_on_llamacpp_and_a_set_still_refused(self, endpoint):
        make = (lambda **kw: chat(response_format=None, **kw)) if endpoint is Endpoint.CHAT \
            else (lambda **kw: TextCompletionRequest(model="m", prompt="a", **kw))
        result = evaluate(make(steering=self.EMPTY), endpoint, Engine.LLAMACPP, strict=True)
        assert result.unused == []
        for refused_body in ({"steering": self.ONE},
                             {"steering": {"features": [], "sae_id": None}},
                             {"profile": "p"}, {"steering_intensity": 0.0}):
            with pytest.raises(FieldNotHonouredError) as exc:
                evaluate(make(**refused_body), endpoint, Engine.LLAMACPP, strict=False)
            assert exc.value.details["param"] == next(iter(refused_body)), refused_body

    def test_honoured_on_embeddings_and_anything_else_refused(self):
        ok = EmbeddingRequest(model="m", input="a", steering={"features": []})
        assert evaluate(ok, Endpoint.EMBEDDINGS, Engine.TRANSFORMERS, strict=True).unused == []
        assert evaluate(ok, Endpoint.EMBEDDINGS, Engine.LLAMACPP, strict=True).unused == []
        for value in (self.ONE, {"features": [], "x": 1}, {"features": [], "sae_id": "s"}):
            req = EmbeddingRequest(model="m", input="a", steering=value)
            with pytest.raises(FieldNotHonouredError):
                evaluate(req, Endpoint.EMBEDDINGS, Engine.TRANSFORMERS, strict=False)

    def test_scoring_accepts_the_empty_set_and_still_refuses_a_set(self):
        """Scoring is always unsteered (X-09): the empty set asks for exactly that. This reverses
        028 FTASKS 2.5's "refuse even the no-op" (2026-11-20; recorded in the review note)."""
        scoring = dict(response_format=None, logprobs=True, max_tokens=1)
        evaluate(chat(steering=self.EMPTY, **scoring), Endpoint.CHAT, Engine.TRANSFORMERS,
                 strict=True)
        with pytest.raises(FieldNotHonouredError) as exc:
            evaluate(chat(steering=self.ONE, **scoring), Endpoint.CHAT, Engine.TRANSFORMERS,
                     strict=False)
        assert "X-09" in exc.value.message

    def test_transformers_generation_is_unchanged(self):
        """Transformers generation honoured both forms before and does now, and the policy does
        not touch the value: the empty set still reaches the steering code, which turns every
        live source off for the request (FR-28.2.2; `test_inline_steering.py` pins that)."""
        for value in (self.EMPTY, self.ONE):
            req = chat(response_format=None, steering=value)
            before = req.model_dump()
            assert evaluate(req, Endpoint.CHAT, Engine.TRANSFORMERS, strict=True).unused == []
            assert req.model_dump() == before
        assert chat(response_format=None, steering=self.EMPTY).steering.features == []


# =============================================================================
# The llama.cpp schema check (pure)
# =============================================================================


class TestTheLlamaCppSchemaCheck:
    @pytest.mark.parametrize("schema", [
        SCHEMA,
        {},
        {"type": "object"},
        {"type": "string", "enum": ["a", "b"]},
        {"const": 3, "type": "integer"},
        {"type": ["string", "null"]},
        {"type": "object", "additionalProperties": {"type": "number"}},
        {"type": "object", "properties": {"a": {"type": "boolean"}}, "additionalProperties": True},
        {"$defs": {"x": {"type": "string"}}, "type": "object",
         "properties": {"a": {"$ref": "#/$defs/x"}}, "required": ["a"]},
        {"type": "array", "items": {"type": "array", "items": {"type": "number"}}, "maxItems": 2},
        {"anyOf": [{"type": "integer"}, {"type": "string", "maxLength": 3}],
         "description": "annotations are inert"},
        # 2026-11-21: annotations alone constrain nothing, so any JSON value enforces them all
        # (was refused "no type"; weightspace's write_tensor `values`).
        {"type": "object", "properties": {"a": {"description": "only"}}},
        {"description": "any JSON value"},
        {"type": "array", "items": {"title": "t", "$comment": "c", "deprecated": True}},
        {"type": "object", "properties": {"a": {"type": "string", "optional": True,
                                                "readOnly": True, "writeOnly": False}}},
    ])
    def test_a_schema_the_grammar_fully_enforces_passes(self, schema):
        json_schema_subset.check(schema)
        json_schema_subset.check_llamacpp(schema)

    @pytest.mark.parametrize("schema, fragment, pointer", [
        ({"type": "integer", "minimum": 0}, "minimum", "/minimum"),
        ({"type": "number", "maximum": 1}, "maximum", "/maximum"),
        ({"type": "string", "pattern": "^[a-z]+$"}, "pattern", "/pattern"),
        ({"maxLength": 3}, "no type", ""),
        ({"type": ["string", "null"], "maxLength": 3}, "maxLength", "/maxLength"),
        ({"type": "array", "minItems": 1}, "minItems", "/minItems"),
        ({"type": "object", "required": ["a"]}, "required", "/required"),
        ({"type": "object", "properties": {"a": {"type": "string"}}, "required": ["a", "b"]},
         "required 'b'", "/required"),
        ({"type": "object", "anyOf": [{"type": "object"}]}, "type", "/type"),
        ({"$defs": {"x": {"type": "string"}}, "$ref": "#/$defs/x", "type": "string"},
         "type", "/type"),
        ({"$ref": "#/$defs/missing"}, "target not found", "/$ref"),
        ({"type": "string", "enum": ["a"], "maxLength": 3}, "maxLength", "/maxLength"),
        ({"type": "integer", "enum": [1, "two"]}, "not of this type", "/type"),
        ({"type": "array", "items": True}, "boolean schema", "/items"),
        ({"type": "object", "properties": {"a": {"type": "object", "properties": {
            "b": {"type": "integer", "minimum": 1}}}}}, "minimum",
         "/properties/a/properties/b/minimum"),
    ])
    def test_a_keyword_the_grammar_would_drop_is_refused(self, schema, fragment, pointer):
        json_schema_subset.check(schema)  # inside the shared subset: transformers serves it
        with pytest.raises(FieldNotHonouredError) as exc:
            json_schema_subset.check_llamacpp(schema, model_name="g:Q4_K_M")
        assert exc.value.code == "FIELD_NOT_HONOURED"
        assert exc.value.details["param"] == "response_format"
        assert exc.value.details["engine"] == "llamacpp"
        assert "for model 'g:Q4_K_M'" in exc.value.message
        pairs = [(k["keyword"], k["pointer"]) for k in exc.value.details["keywords"]]
        assert any(fragment in k and p == pointer for k, p in pairs), pairs


# =============================================================================
# The route: every refusal before the load, and no refusal for a servable request
# =============================================================================


def _post(body, *, gguf=True, cbm=False, headers=None):
    inference = unloaded_inference()
    inference.cbm_enabled = lambda: cbm
    client, svc = make_client(inference, model_row(gguf_files=["m.gguf"] if gguf else None))
    r = client.post(URL, json={"model": "tiny", "messages": MESSAGES, **body},
                    headers=headers or {})
    return r, svc


class TestTheRoute:
    @pytest.mark.parametrize("fmt", [schema_format(), {"type": "json_object"}])
    def test_a_servable_gguf_request_reaches_the_load(self, fmt):
        """Past every pre-load refusal: the stand-in service then records the auto-load."""
        r, svc = _post({"response_format": fmt}, headers={"X-miLLM-Strict": "true"})
        assert svc.load_model_and_wait.call_count == 1, r.text

    def test_continuous_batching_does_not_refuse_a_gguf_request(self):
        """The manager never serves a GGUF model (`create_chat_completion` takes llama.cpp
        first), so its refusal is transformers-only; a transformers row is still refused."""
        r, svc = _post({"response_format": schema_format()}, cbm=True)
        assert svc.load_model_and_wait.call_count == 1, r.text
        r, svc = _post({"response_format": schema_format()}, gguf=False, cbm=True)
        assert r.status_code == 400 and "continuous batching" in r.json()["error"]["message"]
        assert svc.load_model_and_wait.call_count == 0

    @pytest.mark.parametrize("body, code, fragment", [
        ({"response_format": schema_format({"type": "integer", "minimum": 0})},
         "field_not_honoured", "minimum"),
        ({"response_format": schema_format({"type": "string", "pattern": "^a$"})},
         "field_not_honoured", "pattern"),
        ({"response_format": schema_format(), "stream": True}, None, "stream"),
        ({"response_format": schema_format(),
          "messages": MESSAGES + [{"role": "assistant", "content": "{"}]},
         "field_not_honoured", "partial assistant turn"),
        ({"response_format": schema_format({"type": "integer", "multipleOf": 2})},
         "response_format_unsupported", "multipleOf"),
    ])
    def test_refused_naming_response_format_before_the_load(self, body, code, fragment):
        r, svc = _post(body)
        assert r.status_code == 400, r.text
        err = r.json()["error"]
        assert err["param"] == "response_format"
        if code:
            assert err["code"] == code
        assert fragment in err["message"]
        assert svc.load_model_and_wait.call_count == 0

    def test_the_empty_steering_set_reaches_the_load_and_a_set_is_refused(self):
        r, svc = _post({"steering": {"features": []}}, headers={"X-miLLM-Strict": "true"})
        assert svc.load_model_and_wait.call_count == 1, r.text
        r, svc = _post({"steering": {"features": [{"index": 1, "strength": 2.0}]}})
        assert r.status_code == 400 and r.json()["error"]["param"] == "steering"
        assert svc.load_model_and_wait.call_count == 0


# =============================================================================
# The service, llama.cpp stubbed
# =============================================================================


class _Grammar:
    def __init__(self, text):
        self._grammar = text
        self._root = "root"


class _FakeLlamaCpp:
    """`llama_cpp` and `llama_cpp.llama_grammar` reduced to what `_llamacpp_grammar` calls."""

    JSON_GBNF = "root ::= object"

    def __init__(self, monkeypatch, *, parses=True, convert_error=None):
        self.schemas: list[str] = []
        self.inits: list[tuple] = []
        self.freed: list = []
        self.parses = parses
        fake = self

        class LlamaGrammar:
            @classmethod
            def from_json_schema(cls, text, verbose=True):
                fake.schemas.append(text)
                if convert_error is not None:
                    raise convert_error
                return _Grammar(f"schema:{text}")

            @classmethod
            def from_string(cls, text, verbose=True):
                return _Grammar(text)

        grammar_mod = types.ModuleType("llama_cpp.llama_grammar")
        grammar_mod.LlamaGrammar = LlamaGrammar
        grammar_mod.JSON_GBNF = self.JSON_GBNF
        top = types.ModuleType("llama_cpp")
        top.llama_grammar = grammar_mod

        def init(vocab, text, root):
            fake.inits.append((vocab, text, root))
            return object() if fake.parses else None

        top.llama_sampler_init_grammar = init
        top.llama_sampler_free = fake.freed.append
        monkeypatch.setitem(sys.modules, "llama_cpp", top)
        monkeypatch.setitem(sys.modules, "llama_cpp.llama_grammar", grammar_mod)


def _service(content='{"action": "repair", "target": "probe_7", "rows": [1, 2], "note": null}',
             finish="stop"):
    from miserver.services.inference_service import InferenceService

    svc = InferenceService.__new__(InferenceService)
    state = MagicMock()
    state.is_loaded = True
    current = LoadedModel(model_id=1, model_name="m", model=MagicMock(), tokenizer=None,
                          loaded_at=datetime.utcnow(), engine=ENGINE_LLAMACPP)
    current.model.metadata = {}
    current.model.create_chat_completion.return_value = {
        "choices": [{"message": {"content": content}, "finish_reason": finish}],
        "usage": {"prompt_tokens": 3, "completion_tokens": 9, "total_tokens": 12},
    }
    state.current = current
    svc._model_state = state

    class _NullQueue:
        def acquire(self, **_kwargs):
            import contextlib

            @contextlib.asynccontextmanager
            async def _cm():
                yield

            return _cm()

    svc._request_queue = _NullQueue()
    return svc, current.model


class TestTheServiceEnforces:
    async def test_the_schema_reaches_the_converter_unchanged_and_its_grammar_generation(
        self, monkeypatch,
    ):
        from miserver.services.inference_service import get_request_outcome, reset_request_outcome

        fake = _FakeLlamaCpp(monkeypatch)
        svc, llama = _service()
        reset_request_outcome()
        resp = await svc._llamacpp_chat_completion(chat())
        assert len(fake.schemas) == 1 and json.loads(fake.schemas[0]) == SCHEMA
        kwargs = llama.create_chat_completion.call_args.kwargs
        assert kwargs["grammar"]._grammar == f"schema:{fake.schemas[0]}"
        # llama-cpp-python's own response_format would IGNORE json_schema (0.3.36): never sent.
        assert "response_format" not in kwargs
        assert llama.create_chat_completion.call_count == 1
        # The grammar was parsed natively against the resident vocabulary, then freed.
        assert fake.inits == [(llama._model.vocab, kwargs["grammar"]._grammar.encode(), b"root")]
        assert len(fake.freed) == 1
        assert resp.choices[0].finish_reason == "stop"
        assert json.loads(resp.choices[0].message.content) == VALID
        assert get_request_outcome()["constrained"] == 'json_schema;name="wsp_action"'

    async def test_json_object_gets_the_json_object_grammar(self, monkeypatch):
        fake = _FakeLlamaCpp(monkeypatch)
        svc, llama = _service(content='{"a": 1}')
        resp = await svc._llamacpp_chat_completion(chat(response_format={"type": "json_object"}))
        assert llama.create_chat_completion.call_args.kwargs["grammar"]._grammar == fake.JSON_GBNF
        assert fake.schemas == []
        assert resp.choices[0].finish_reason == "stop"

    @pytest.mark.parametrize("over", [{"response_format": None},
                                      {"response_format": {"type": "text"}}])
    async def test_no_constraint_sends_no_grammar(self, monkeypatch, over):
        fake = _FakeLlamaCpp(monkeypatch)
        svc, llama = _service(content="plain words")
        resp = await svc._llamacpp_chat_completion(chat(**over))
        assert "grammar" not in llama.create_chat_completion.call_args.kwargs
        assert fake.inits == [] and resp.choices[0].message.content == "plain words"

    async def test_a_schema_that_does_not_convert_refuses_and_never_generates(self, monkeypatch):
        _FakeLlamaCpp(monkeypatch, convert_error=AssertionError("Unrecognized schema"))
        svc, llama = _service()
        with pytest.raises(FieldNotHonouredError) as exc:
            await svc._llamacpp_chat_completion(chat())
        assert exc.value.details["param"] == "response_format"
        assert "could not be converted" in exc.value.message
        assert "Unrecognized schema" in exc.value.message
        assert llama.create_chat_completion.call_count == 0

    async def test_a_grammar_llama_cpp_rejects_refuses_and_never_generates(self, monkeypatch):
        fake = _FakeLlamaCpp(monkeypatch, parses=False)
        svc, llama = _service()
        with pytest.raises(FieldNotHonouredError) as exc:
            await svc._llamacpp_chat_completion(chat())
        assert "rejected the grammar" in exc.value.message
        assert len(fake.inits) == 1 and fake.freed == []
        assert llama.create_chat_completion.call_count == 0

    async def test_without_llama_cpp_it_refuses_rather_than_generating(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "llama_cpp", None)
        monkeypatch.setitem(sys.modules, "llama_cpp.llama_grammar", None)
        svc, llama = _service()
        with pytest.raises(FieldNotHonouredError):
            await svc._llamacpp_chat_completion(chat())
        assert llama.create_chat_completion.call_count == 0

    async def test_a_complete_output_that_fails_the_schema_is_500_never_200(self, monkeypatch):
        _FakeLlamaCpp(monkeypatch)
        svc, _ = _service(content='{"action": "delete", "target": "x", "rows": [1]}')
        with pytest.raises(ConstrainedOutputInvalidError):
            await svc._llamacpp_chat_completion(chat())

    async def test_a_complete_json_object_output_that_is_not_an_object_is_500(self, monkeypatch):
        _FakeLlamaCpp(monkeypatch)
        svc, _ = _service(content="[1, 2]")
        with pytest.raises(ConstrainedOutputInvalidError):
            await svc._llamacpp_chat_completion(chat(response_format={"type": "json_object"}))

    async def test_a_budget_ended_output_is_length_and_not_validated(self, monkeypatch):
        _FakeLlamaCpp(monkeypatch)
        svc, _ = _service(content='{"action": "rep', finish="length")
        resp = await svc._llamacpp_chat_completion(chat())
        assert resp.choices[0].finish_reason == "length"


class TestTheServiceStillRefuses:
    async def test_a_direct_streaming_call_is_refused(self, monkeypatch):
        """`stream_chat_completion` refuses first; this is the llama.cpp generator's own guard."""
        _FakeLlamaCpp(monkeypatch)
        svc, llama = _service()
        with pytest.raises(ResponseFormatUnsupportedError):
            async for _ in svc._llamacpp_stream_chat_completion(chat()):
                pass
        assert llama.create_chat_completion.call_count == 0

    async def test_a_partial_assistant_turn_is_refused_in_the_service(self, monkeypatch):
        _FakeLlamaCpp(monkeypatch)
        svc, llama = _service()
        req = chat(messages=MESSAGES + [{"role": "assistant", "content": '{"action"'}])
        with pytest.raises(FieldNotHonouredError):
            await svc._llamacpp_chat_completion(req)
        assert llama.create_chat_completion.call_count == 0
        assert llama.create_completion.call_count == 0

    async def test_n_2_is_still_refused(self, monkeypatch):
        _FakeLlamaCpp(monkeypatch)
        svc, llama = _service()
        with pytest.raises(EngineUnsupportedError):
            await svc._llamacpp_chat_completion(chat(n=2))
        assert llama.create_chat_completion.call_count == 0


class TestTheServiceSteeringGuard:
    async def test_the_empty_set_generates_and_a_set_is_refused(self):
        from miserver.services.inference_service import InferenceService

        svc, llama = _service(content="hi")
        resp = await svc._llamacpp_chat_completion(
            chat(response_format=None, steering={"features": []}))
        assert resp.choices[0].message.content == "hi"
        assert llama.create_chat_completion.call_count == 1
        for body in ({"steering": {"features": [{"index": 1, "strength": 2.0}]}},
                     {"profile": "p"}, {"steering_intensity": 0.0}):
            with pytest.raises(EngineUnsupportedError):
                InferenceService._refuse_llamacpp_steering(chat(response_format=None, **body))
        with pytest.raises(EngineUnsupportedError):
            await svc._llamacpp_chat_completion(chat(
                response_format=None, steering={"features": [{"index": 1, "strength": 2.0}]}))
        assert llama.create_chat_completion.call_count == 1
