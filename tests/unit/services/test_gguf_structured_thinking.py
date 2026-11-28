"""Structured output that may THINK first, on the llama.cpp engine (GGUF), 2026-11-21.

`response_format` json_schema / json_object on a GGUF model applied its grammar from the first
generated token, through llama-cpp-python's own chat handler, which never passes
`chat_template_kwargs`. On Qwen3.6, whose template opens `<think>` in the generation prompt, that
forced the JSON inside the reasoning block, and `enable_thinking: false` could not reach the
template. miServer now renders such a request itself ("document" mode,
`miserver/services/llamacpp_tools.py`): the reasoning block is generated freely within the
reasoning allowance (`max_reasoning_tokens`, else half of `max_tokens`) and closed by miServer if
the allowance ends it; the document follows under the grammar, after `</think>\\n\\n`.

llama.cpp is STUBBED (CI installs no llama-cpp-python): `FakeLlama` from the tool-call tests (one
token id per character, so every `create_completion` prompt reads back as text and every count is
exact), here made to honour `max_tokens` the way llama.cpp does (cut, `finish_reason: "length"`).
The template is the REAL Qwen3.6 GGUF one.

`test_gguf_structured_output_real.py::TestThinkingOnARealModel` runs the same path on a REAL tiny
GGUF with the real wheel and grammar engine.

MUTATION CONTROLS (each turns this file red; results in
`0xcc/reviews/gguf_structured_thinking_2026-11-21.md`):
  S1 the grammar deferred past `</think>`: `continuation = cut + "\\n\\n"` -> `""`
  S2 the allowance cut-off: the reasoning phase's budget -> the whole `budget`
  S3 the `enable_thinking` pass-through: `template_kwargs=request.chat_template_kwargs` -> None
  S4 the routing: `constrains_a_document(request)` dropped from `llamacpp_renders_template_itself`
  S5 the closing tag not counted against the allowance
  S6 the document's grammar handed to the free phase too (`params["grammar"]` set unconditionally)
  S6b the free phase's own `grammar` drop removed
  S7 the (CHAT, LC) `max_reasoning_tokens` refuse_if removed
  S8 `completion_tokens_details` not filled
  S9 the service's defence-in-depth refusal removed
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from miserver.api.request_policy import OUTPUT_CHANGING, Endpoint, Engine, Honoured, evaluate
from miserver.api.schemas.openai import DEFAULT_MAX_TOKENS, ChatCompletionRequest
from miserver.core.errors import ConstrainedOutputInvalidError, FieldNotHonouredError
from miserver.ml.generation_config import GenerationConfig
from miserver.services import llamacpp_tools as lt
from tests.unit.f25_fixtures import clear_loaded, make_client, model_row, unloaded_inference
from tests.unit.services.test_gguf_tool_calls import (
    QWEN25,
    QWEN36,
    FakeLlama,
    FakeLlamaCpp,
    service,
)

URL = "/v1/chat/completions"
USER = [{"role": "user", "content": "Pick the next action"}]
SCHEMA = {
    "type": "object",
    "properties": {"action": {"type": "string", "enum": ["inspect", "repair"]},
                   "target": {"type": "string"}},
    "required": ["action", "target"],
    "additionalProperties": False,
}
FORMAT = {"type": "json_schema", "json_schema": {"name": "wsp_action", "schema": SCHEMA}}
DOC = '{"action": "repair", "target": "probe_7"}'
THINK_ON, THINK_OFF = {"enable_thinking": True}, {"enable_thinking": False}
CLOSE = "\n</think>"  # what miServer appends when it closes the block: 9 tokens here
GENERATION_PROMPT = "<|im_start|>assistant\n"


def chat(**over: Any) -> ChatCompletionRequest:
    body: dict[str, Any] = {"model": "m", "messages": USER, "max_tokens": 200,
                            "response_format": FORMAT}
    body.update(over)
    return ChatCompletionRequest(**body)


class BudgetLlama(FakeLlama):
    """`FakeLlama` that honours `max_tokens` as llama.cpp does: a reply longer than the budget is
    cut to it and reported "length" (one token per character)."""

    def create_completion(self, prompt, stream=False, **kw):
        text, finish = self.replies[0]
        budget = kw.get("max_tokens")
        if budget is not None and len(text) > budget:
            self.replies[0] = (text[:budget], "length")
        return super().create_completion(prompt, stream=stream, **kw)


def llama(*replies: tuple[str, str], template: str | None = QWEN36) -> BudgetLlama:
    return BudgetLlama(template, list(replies))


async def complete(fake: FakeLlama, request: ChatCompletionRequest):
    return await service(fake)._llamacpp_chat_completion(request)


@pytest.fixture(autouse=True)
def _clean():
    clear_loaded()
    yield
    clear_loaded()


# =============================================================================
# The policy: max_reasoning_tokens and chat_template_kwargs, decided before the load
# =============================================================================


class TestThePolicy:
    def test_the_allowance_is_honoured_on_llamacpp_structured_output(self):
        assert isinstance(OUTPUT_CHANGING["max_reasoning_tokens"][(Endpoint.CHAT, Engine.LLAMACPP)],
                          Honoured)
        result = evaluate(chat(max_reasoning_tokens=120, chat_template_kwargs=THINK_ON),
                          Endpoint.CHAT, Engine.LLAMACPP, strict=True)
        assert result.unused == []
        evaluate(chat(max_reasoning_tokens=5, response_format={"type": "json_object"}),
                 Endpoint.CHAT, Engine.LLAMACPP, strict=True)

    @pytest.mark.parametrize("over, fragment", [
        ({"response_format": None}, "applies to response_format"),
        ({"response_format": {"type": "text"}}, "applies to response_format"),
        ({"max_tokens": 120}, "must be smaller than max_tokens (120)"),
        ({"max_tokens": None, "max_reasoning_tokens": DEFAULT_MAX_TOKENS},
         f"max_tokens ({DEFAULT_MAX_TOKENS}, the default)"),
        ({"max_tokens": None, "max_completion_tokens": 100}, "max_tokens (100)"),
    ])
    def test_refused_off_its_path_or_at_max_tokens(self, over, fragment):
        body = {"max_reasoning_tokens": 120, **over}
        with pytest.raises(FieldNotHonouredError) as exc:
            evaluate(chat(**body), Endpoint.CHAT, Engine.LLAMACPP, strict=False)
        assert exc.value.details["param"] == "max_reasoning_tokens"
        assert fragment in exc.value.message

    @pytest.mark.parametrize("endpoint, engine", [
        (Endpoint.CHAT, Engine.TRANSFORMERS),
        (Endpoint.COMPLETIONS, Engine.LLAMACPP),
        (Endpoint.EMBEDDINGS, Engine.LLAMACPP),
    ])
    def test_refused_on_every_other_cell(self, endpoint, engine):
        assert not isinstance(OUTPUT_CHANGING["max_reasoning_tokens"][(endpoint, engine)],
                              Honoured)

    def test_chat_template_kwargs_is_used_with_a_constrained_document(self):
        """It now reaches the template miServer renders; without a constraint (llama-cpp-python's
        own handler) it is still reported unused."""
        used = evaluate(chat(chat_template_kwargs=THINK_OFF), Endpoint.CHAT, Engine.LLAMACPP,
                        strict=True)
        assert "chat_template_kwargs" not in used.unused
        plain = evaluate(chat(response_format=None, chat_template_kwargs=THINK_OFF),
                         Endpoint.CHAT, Engine.LLAMACPP, strict=False)
        assert plain.unused == ["chat_template_kwargs"]

    def test_the_default_max_tokens_is_the_one_generation_uses(self):
        """The policy compares the allowance with DEFAULT_MAX_TOKENS; generation must use it."""
        request = ChatCompletionRequest(model="m", messages=USER)
        assert GenerationConfig.from_request(request).max_new_tokens == DEFAULT_MAX_TOKENS


def _post(body: dict, *, headers: dict | None = None):
    client, svc = make_client(unloaded_inference(), model_row(gguf_files=["m.gguf"]))
    r = client.post(URL, json={"model": "tiny", "messages": USER, **body}, headers=headers or {})
    return r, svc


class TestTheRoute:
    @pytest.mark.parametrize("over", [
        {"chat_template_kwargs": THINK_ON, "max_reasoning_tokens": 150},
        {"chat_template_kwargs": THINK_OFF},
        {},
    ])
    def test_a_servable_request_reaches_the_load_under_strict(self, over):
        r, svc = _post({"response_format": FORMAT, "max_tokens": 400, **over},
                       headers={"X-miLLM-Strict": "true"})
        assert svc.load_model_and_wait.call_count == 1, r.text

    @pytest.mark.parametrize("body", [
        {"max_reasoning_tokens": 10},
        {"max_reasoning_tokens": 400, "max_tokens": 400, "response_format": FORMAT},
    ])
    def test_refused_naming_the_field_before_the_load(self, body):
        r, svc = _post(body)
        assert r.status_code == 400, r.text
        assert r.json()["error"]["param"] == "max_reasoning_tokens"
        assert svc.load_model_and_wait.call_count == 0

    def test_refused_on_a_transformers_row_before_the_load(self):
        client, svc = make_client(unloaded_inference(), model_row(gguf_files=None))
        r = client.post(URL, json={"model": "tiny", "messages": USER, "response_format": FORMAT,
                                   "max_reasoning_tokens": 10})
        assert r.status_code == 400 and r.json()["error"]["param"] == "max_reasoning_tokens"
        assert svc.load_model_and_wait.call_count == 0


# =============================================================================
# The template: what enable_thinking renders on Qwen3.6
# =============================================================================


class TestTheTemplate:
    @pytest.mark.parametrize("kwargs, tail, opened", [
        (None, "<think>\n", True),  # the template's default: it thinks
        (THINK_ON, "<think>\n", True),
        (THINK_OFF, "<think>\n\n</think>\n\n", False),  # a closed, empty block
    ])
    def test_qwen36_renders(self, kwargs, tail, opened):
        prompt = lt.render_chat_prompt(QWEN36, USER, bos="", eos="<|im_end|>",
                                       template_kwargs=kwargs)
        assert prompt.endswith(GENERATION_PROMPT + tail)
        assert lt.prompt_opens_think(prompt) is opened

    def test_the_default_allowance_is_half(self):
        assert lt.reasoning_allowance(200, None) == 100
        assert lt.reasoning_allowance(201, None) == 100
        assert lt.reasoning_allowance(200, 30) == 30


# =============================================================================
# The service: thinking on
# =============================================================================


class TestThinkingOn:
    @pytest.mark.parametrize("kwargs", [THINK_ON, None], ids=["enable_thinking", "unset"])
    async def test_reasoning_freely_then_the_document_under_the_grammar(self, monkeypatch, kwargs):
        FakeLlamaCpp(monkeypatch)
        fake = llama(("Repair it.\n</think>\n\nignored", "stop"), (DOC, "stop"))
        resp = await complete(fake, chat(chat_template_kwargs=kwargs, max_reasoning_tokens=60))
        free, doc = fake.calls
        # Phase 1: the exact prompt the template rendered, free (no grammar), streamed, bounded
        # by the allowance less the closing tag.
        assert free["prompt"] == lt.render_chat_prompt(
            QWEN36, USER, bos="<|endoftext|>", eos="<|im_end|>", template_kwargs=kwargs)
        assert free["prompt"].endswith(GENERATION_PROMPT + "<think>\n")
        assert free["stream"] is True and "grammar" not in free
        assert free["max_tokens"] == 60 - len(CLOSE)
        # Phase 2: the grammar begins AFTER `</think>` and the template's separator.
        assert doc["prompt"] == free["prompt"] + "Repair it.\n</think>\n\n"
        assert doc["stream"] is False
        assert lt.STRICT_CHAR_RULE in doc["grammar"]._grammar.splitlines()
        reasoning_tokens = len("Repair it.\n</think>")
        assert doc["max_tokens"] == 200 - reasoning_tokens
        # The reply: reasoning apart, the content exactly the document.
        choice = resp.choices[0]
        assert choice.finish_reason == "stop"
        assert choice.message.reasoning_content == "Repair it."
        assert choice.message.content == DOC
        assert resp.usage.completion_tokens == reasoning_tokens + len(DOC)
        assert resp.usage.completion_tokens_details.reasoning_tokens == reasoning_tokens
        assert resp.usage.prompt_tokens == len(free["prompt"])
        assert fake.closed == 1  # the free stream closed at `</think>`

    async def test_the_allowance_closes_the_block_and_the_document_follows(self, monkeypatch):
        FakeLlamaCpp(monkeypatch)
        fake = llama(("x" * 500, "stop"), (DOC, "stop"))
        resp = await complete(fake, chat(max_tokens=100, max_reasoning_tokens=40))
        free, doc = fake.calls
        assert free["max_tokens"] == 40 - len(CLOSE) == 31
        reasoning = "x" * 31
        assert doc["prompt"] == free["prompt"] + reasoning + CLOSE + "\n\n"
        assert doc["max_tokens"] == 100 - 40 == 60
        choice = resp.choices[0]
        assert choice.finish_reason == "stop"  # never "length" because the reasoning ran long
        assert choice.message.reasoning_content == reasoning
        assert choice.message.content == DOC
        assert resp.usage.completion_tokens_details.reasoning_tokens == 40
        assert resp.usage.completion_tokens == 40 + len(DOC) <= 100

    async def test_without_the_field_the_allowance_is_half_of_max_tokens(self, monkeypatch):
        FakeLlamaCpp(monkeypatch)
        fake = llama(("x" * 500, "stop"), (DOC, "stop"))
        resp = await complete(fake, chat(max_tokens=100))
        free, doc = fake.calls
        assert free["max_tokens"] == 50 - len(CLOSE)
        assert doc["max_tokens"] == 50
        assert resp.usage.completion_tokens_details.reasoning_tokens == 50

    async def test_an_allowance_smaller_than_the_tag_reasons_not_at_all(self, monkeypatch):
        FakeLlamaCpp(monkeypatch)
        fake = llama((DOC, "stop"))
        resp = await complete(fake, chat(max_tokens=100, max_reasoning_tokens=3))
        (doc,) = fake.calls  # no free phase: the block is closed at once
        assert doc["prompt"].endswith(GENERATION_PROMPT + "<think>\n" + CLOSE + "\n\n")
        assert resp.choices[0].message.reasoning_content is None
        assert resp.choices[0].message.content == DOC

    async def test_an_end_of_turn_inside_the_reasoning_is_closed_too(self, monkeypatch):
        FakeLlamaCpp(monkeypatch)
        fake = llama(("Done thinking", "stop"), (DOC, "stop"))
        resp = await complete(fake, chat())
        _, doc = fake.calls
        assert doc["prompt"].endswith("<think>\nDone thinking" + CLOSE + "\n\n")
        assert resp.choices[0].message.content == DOC
        assert resp.choices[0].message.reasoning_content == "Done thinking"

    async def test_reasoning_that_mentions_json_or_tags_stays_out_of_content(self, monkeypatch):
        FakeLlamaCpp(monkeypatch)
        fake = llama(('Maybe {"action": "inspect"}? or <think> again\n</think>', "stop"),
                     (DOC, "stop"))
        resp = await complete(fake, chat())
        message = resp.choices[0].message
        assert message.content == DOC
        assert message.reasoning_content == 'Maybe {"action": "inspect"}? or <think> again'

    async def test_a_document_the_budget_ends_is_length_and_not_validated(self, monkeypatch):
        FakeLlamaCpp(monkeypatch)
        fake = llama(("ok\n</think>", "stop"), ('{"action": "repair", "target": "' + "p" * 90,
                                                  "stop"))
        resp = await complete(fake, chat(max_tokens=60))
        choice = resp.choices[0]
        assert fake.calls[1]["max_tokens"] == 60 - len("ok\n</think>")
        assert choice.finish_reason == "length"
        assert resp.usage.completion_tokens == 60

    async def test_a_complete_document_that_fails_the_schema_is_500(self, monkeypatch):
        FakeLlamaCpp(monkeypatch)
        fake = llama(("ok\n</think>", "stop"), ('{"action": "delete", "target": "x"}', "stop"))
        with pytest.raises(ConstrainedOutputInvalidError):
            await complete(fake, chat())

    async def test_the_documents_grammar_travels_in_the_run_not_the_parameters(
        self, monkeypatch,
    ):
        """S6: the service hands the grammar to the run only. (The free phase also drops any
        `grammar` parameter, S6b below; each layer is pinned on its own.)"""
        FakeLlamaCpp(monkeypatch)
        seen: list[dict] = []
        real = lt.ToolGeneration

        class Spy(real):
            def __init__(self, model, run, params, stop=None):
                seen.append(dict(params))
                super().__init__(model, run, params, stop)

        monkeypatch.setattr(lt, "ToolGeneration", Spy)
        fake = llama(("ok\n</think>", "stop"), (DOC, "stop"))
        await complete(fake, chat())
        (params,) = seen
        assert "grammar" not in params and params["max_tokens"] == 200

    def test_the_free_phase_never_takes_a_grammar_parameter(self):
        """S6b: a direct caller that passes `grammar` still gets free reasoning."""
        fake = llama(("ok\n</think>", "stop"), (DOC, "stop"))
        prompt = lt.render_chat_prompt(QWEN36, USER, bos="", eos="<|im_end|>")
        run = lt.LlamaToolRun(prompt=prompt, prompt_tokens=[ord(c) for c in prompt],
                              opened_think=True, mode="document", grammar="G",
                              reasoning_allowance=50)
        generation = lt.ToolGeneration(fake, run, {"max_tokens": 100, "grammar": "G"})
        list(generation.events())
        free, doc = fake.calls
        assert "grammar" not in free and doc["grammar"] == "G"
        assert generation.result.text == DOC and generation.result.reasoning == "ok"

    async def test_json_object_takes_the_same_path(self, monkeypatch):
        FakeLlamaCpp(monkeypatch)
        fake = llama(("hm\n</think>", "stop"), ('{"a": 1}', "stop"))
        resp = await complete(fake, chat(response_format={"type": "json_object"}))
        _, doc = fake.calls
        assert doc["grammar"]._grammar == "root ::= object"
        assert resp.choices[0].message.content == '{"a": 1}'
        assert resp.choices[0].message.reasoning_content == "hm"

    async def test_tool_call_history_with_a_document_reasons_first_too(self, monkeypatch):
        FakeLlamaCpp(monkeypatch)
        history = USER + [
            {"role": "assistant", "content": None, "tool_calls": [{
                "id": "c1", "type": "function",
                "function": {"name": "describe_tensor", "arguments": '{"name": "w"}'}}]},
            {"role": "tool", "tool_call_id": "c1", "content": "norm 5"},
        ]
        fake = llama(("fine\n</think>", "stop"), (DOC, "stop"))
        resp = await complete(fake, chat(messages=history))
        free, doc = fake.calls
        assert "<tool_response>\nnorm 5\n</tool_response>" in free["prompt"]
        assert "grammar" not in free and doc["grammar"] is not None
        assert resp.choices[0].message.content == DOC


# =============================================================================
# The service: thinking off, or no thinking in the template
# =============================================================================


class TestThinkingOff:
    async def test_enable_thinking_false_is_one_phase_after_the_empty_block(self, monkeypatch):
        FakeLlamaCpp(monkeypatch)
        fake = llama((DOC, "stop"))
        resp = await complete(fake, chat(chat_template_kwargs=THINK_OFF))
        (doc,) = fake.calls
        assert doc["prompt"].endswith(GENERATION_PROMPT + "<think>\n\n</think>\n\n")
        assert doc["prompt"] == lt.render_chat_prompt(
            QWEN36, USER, bos="<|endoftext|>", eos="<|im_end|>", template_kwargs=THINK_OFF)
        assert doc["stream"] is False and doc["grammar"] is not None
        assert doc["max_tokens"] == 200
        choice = resp.choices[0]
        assert choice.message.content == DOC and choice.message.reasoning_content is None
        assert resp.usage.completion_tokens_details.reasoning_tokens == 0
        assert resp.usage.completion_tokens == len(DOC)

    async def test_a_template_without_thinking_is_one_phase(self, monkeypatch):
        FakeLlamaCpp(monkeypatch)
        fake = llama((DOC, "stop"), template=QWEN25)
        resp = await complete(fake, chat(max_reasoning_tokens=50))
        (doc,) = fake.calls
        assert doc["prompt"].endswith(GENERATION_PROMPT)
        assert doc["grammar"] is not None and doc["max_tokens"] == 200
        assert resp.choices[0].message.content == DOC

    async def test_a_file_without_a_template_takes_the_old_path(self, monkeypatch):
        """Nothing to render: llama-cpp-python's handler with the grammar, as before."""
        from unittest.mock import MagicMock

        FakeLlamaCpp(monkeypatch)
        fake = MagicMock()
        fake.metadata = {}
        fake.create_chat_completion.return_value = {
            "choices": [{"message": {"content": DOC}, "finish_reason": "stop"}], "usage": {}}
        resp = await service(fake)._llamacpp_chat_completion(chat(chat_template_kwargs=THINK_ON))
        assert fake.create_chat_completion.call_args.kwargs["grammar"] is not None
        assert fake.create_completion.call_count == 0
        assert resp.choices[0].message.content == DOC
        assert resp.usage.completion_tokens_details is None


# =============================================================================
# Direct callers (Feature 26) and the body shape
# =============================================================================


class TestTheService:
    @pytest.mark.parametrize("over", [
        {"max_reasoning_tokens": 200},
        {"max_reasoning_tokens": 10, "response_format": None},
    ])
    async def test_a_direct_caller_is_refused_before_any_token(self, monkeypatch, over):
        FakeLlamaCpp(monkeypatch)
        fake = llama((DOC, "stop"))
        with pytest.raises(FieldNotHonouredError) as exc:
            await complete(fake, chat(**over))
        assert exc.value.details["param"] == "max_reasoning_tokens"
        assert fake.calls == []

    async def test_the_usage_details_are_absent_elsewhere(self, monkeypatch):
        """Only where miServer counted the reasoning: a plain GGUF chat body is unchanged."""
        FakeLlamaCpp(monkeypatch)
        from unittest.mock import MagicMock

        fake = MagicMock()
        fake.metadata = {"tokenizer.chat_template": QWEN36}
        fake.create_chat_completion.return_value = {
            "choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}
        resp = await service(fake)._llamacpp_chat_completion(chat(response_format=None))
        assert "completion_tokens_details" not in json.loads(resp.model_dump_json())["usage"]

    async def test_the_body_carries_openais_field(self, monkeypatch):
        FakeLlamaCpp(monkeypatch)
        fake = llama(("ok\n</think>", "stop"), (DOC, "stop"))
        body = json.loads((await complete(fake, chat())).model_dump_json())
        assert body["usage"]["completion_tokens_details"] == {"reasoning_tokens": 11}
        assert body["choices"][0]["message"]["content"] == DOC
        assert body["choices"][0]["message"]["reasoning_content"] == "ok"

    def test_a_direct_allowance_at_max_tokens_still_leaves_the_document_a_token(self):
        """S10: the service refuses it, and the generation clamps it for a direct caller."""
        fake = llama(("x" * 500, "stop"), ("{", "stop"))
        prompt = lt.render_chat_prompt(QWEN36, USER, bos="", eos="<|im_end|>")
        run = lt.LlamaToolRun(prompt=prompt, prompt_tokens=[ord(c) for c in prompt],
                              opened_think=True, mode="document", grammar="G",
                              reasoning_allowance=100)
        generation = lt.ToolGeneration(fake, run, {"max_tokens": 100})
        list(generation.events())
        free, doc = fake.calls
        assert free["max_tokens"] == 99 - len(CLOSE)
        assert doc["max_tokens"] == 1
        assert generation.result.reasoning_tokens == 99

    def test_every_other_chat_path_may_still_build_a_plain_usage(self):
        """S15: `ChatCompletionResponse.usage` is a `ChatUsage`; the paths that build `Usage`
        are accepted, and their bodies carry no details."""
        from miserver.api.schemas.openai import (
            ChatCompletionChoice,
            ChatCompletionResponse,
            ChatMessage,
            ChatUsage,
            Usage,
        )

        resp = ChatCompletionResponse(
            id="c", created=0, model="m",
            choices=[ChatCompletionChoice(index=0, message=ChatMessage(role="assistant",
                                                                       content="hi"),
                                          finish_reason="stop")],
            usage=Usage(prompt_tokens=2, completion_tokens=3))
        assert isinstance(resp.usage, ChatUsage) and resp.usage.total_tokens == 5
        assert json.loads(resp.model_dump_json())["usage"] == {
            "prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5}
