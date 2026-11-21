"""The reasoning allowance on the GGUF TOOL path (2026-11-21; live failure, WSP run `wsp_84dddc47ecaf`).

weightspace on Qwen3.6-27B-GGUF (`tool_choice: auto`, thinking at the template default, `max_tokens`
2048): calls 14 and 15 came back `finish_reason: length` with all 2048 tokens spent on reasoning and
`content` `<tool_call>\\n<function=python`; no call was made. The rule of the document path now holds
here too (`llamacpp_tools.ToolGeneration.events`):

* a prompt-opened `<think>` block takes at most the allowance (`max_reasoning_tokens`, else half of
  `max_tokens`; the closing tag included), closed by miServer (`\\n</think>\\n\\n`) when it runs out;
* `required` / a named function: straight to the call grammar after the block;
* `auto`: the answer is generated freely after the block, holding back a call's worth
  (`call_reserve`: the shortest allowed call plus `CALL_MARGIN`, at most half of what is left), so a
  call opened late still fits;
* a call the budget cuts short is never content: `length`, the text ending before the marker, the
  reasoning kept; a marker prefix left dangling by the budget is stripped too;
* `usage.completion_tokens_details.reasoning_tokens` is reported, streamed and not.

Stubbed llama.cpp (`BudgetLlama`: one token per character, `max_tokens` honoured as llama.cpp does).

MUTATION CONTROLS (results in `0xcc/reviews/gguf_structured_thinking_2026-11-21.md`):
  R1 the allowance cut-off on the tool path (`_reason` given the whole budget)
  R2 the auto answer phase not continued after a cut-off block (the old single phase)
  R3 the call reserve not held back
  R4 a budget-cut call returned as text again (`finish_without_a_call` given `marker + body`)
  R5 the dangling marker prefix not stripped
  R6 `reasoning_tokens` not reported on the tool path (non-stream / stream)
  R7 the policy's tools branch removed (`max_reasoning_tokens` refused with tools)
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from miserver.api.request_policy import Endpoint, Engine, evaluate
from miserver.api.schemas.openai import ChatCompletionRequest
from miserver.core.errors import FieldNotHonouredError
from miserver.services import llamacpp_tools as lt
from miserver.services.tool_calls import QWEN3_XML
from tests.unit.f25_fixtures import clear_loaded, make_client, model_row, unloaded_inference
from tests.unit.services.test_gguf_structured_thinking import BudgetLlama
from tests.unit.services.test_gguf_tool_calls import (
    QWEN36,
    TOOLS,
    USER,
    XML_CALL,
    FakeLlamaCpp,
    service,
    stream,
)

CLOSE = "\n</think>"
SEP = "\n\n"
SHORTEST = lt.shortest_call(QWEN3_XML, TOOLS, frozenset({"describe_tensor", "scale_tensor"}))
RESERVE = len(SHORTEST) + lt.CALL_MARGIN  # one token per character here


def chat(**over: Any) -> ChatCompletionRequest:
    body: dict[str, Any] = {"model": "m", "messages": USER, "tools": TOOLS, "max_tokens": 2048}
    body.update(over)
    return ChatCompletionRequest(**body)


def llama(*replies: tuple[str, str]) -> BudgetLlama:
    return BudgetLlama(QWEN36, list(replies))


async def complete(fake, request):
    return await service(fake)._llamacpp_chat_completion(request)


@pytest.fixture(autouse=True)
def _clean():
    clear_loaded()
    yield
    clear_loaded()


class TestThePolicy:
    @pytest.mark.parametrize("over", [{}, {"tool_choice": "required"},
                                      {"tool_choice": {"type": "function",
                                                       "function": {"name": "scale_tensor"}}}])
    def test_honoured_with_tools(self, over):
        result = evaluate(chat(max_reasoning_tokens=512, **over), Endpoint.CHAT, Engine.LLAMACPP,
                          strict=True)
        assert result.unused == []

    @pytest.mark.parametrize("over, fragment", [
        ({"max_reasoning_tokens": 2048}, "must be smaller than max_tokens (2048)"),
        ({"max_reasoning_tokens": 5, "tool_choice": "none"}, "or to tools"),
        ({"max_reasoning_tokens": 5, "tools": None}, "or to tools"),
    ])
    def test_refused_at_max_tokens_or_without_tools_shown(self, over, fragment):
        with pytest.raises(FieldNotHonouredError) as exc:
            evaluate(chat(**over), Endpoint.CHAT, Engine.LLAMACPP, strict=False)
        assert exc.value.details["param"] == "max_reasoning_tokens"
        assert fragment in exc.value.message

    def test_the_route_reaches_the_load(self):
        client, svc = make_client(unloaded_inference(), model_row(gguf_files=["m.gguf"]))
        r = client.post("/v1/chat/completions", headers={"X-miLLM-Strict": "true"},
                        json={"model": "tiny", "messages": USER, "tools": TOOLS,
                              "max_tokens": 2048, "max_reasoning_tokens": 1024})
        assert svc.load_model_and_wait.call_count == 1, r.text


class TestTheReserve:
    def test_the_shortest_call_is_the_cheapest_allowed_one(self):
        assert SHORTEST == ("<tool_call>\n<function=describe_tensor>\n<parameter=name>\n0\n"
                            "</parameter>\n</function>\n</tool_call>")
        named = lt.shortest_call(QWEN3_XML, TOOLS, frozenset({"scale_tensor"}))
        assert "<parameter=factor>" in named and "describe_tensor" not in named


class TestTheLiveFailure:
    async def test_auto_reasoning_that_never_ends_is_cut_and_the_call_still_made(self, monkeypatch):
        """wsp_84dddc47ecaf calls 14/15: `auto`, template-default thinking, 2048 tokens."""
        FakeLlamaCpp(monkeypatch)
        fake = llama(("x" * 5000, "stop"),
                     ("Let me run it.\n<tool_call>\n<function=junk", "stop"),
                     (XML_CALL, "stop"))
        resp = await complete(fake, chat())
        reason, answer, call = fake.calls
        # The reasoning: free, bounded by half of max_tokens less the closing tag.
        assert reason["stream"] is True and "grammar" not in reason
        assert reason["max_tokens"] == 1024 - len(CLOSE)
        block = "x" * (1024 - len(CLOSE)) + CLOSE
        # The answer: free, after the closed block and its separator, holding a call back.
        assert answer["prompt"] == reason["prompt"] + block + SEP
        assert "grammar" not in answer
        assert answer["max_tokens"] == (2048 - 1024) - RESERVE
        # The call: under the grammar, after the marker, with what is left.
        assert call["prompt"] == answer["prompt"] + "Let me run it.\n<tool_call>"
        assert call["grammar"] is not None
        assert call["max_tokens"] == 2048 - 1024 - len("Let me run it.\n<tool_call>")
        choice = resp.choices[0]
        assert choice.finish_reason == "tool_calls"
        assert choice.message.tool_calls[0]["function"]["name"] == "scale_tensor"
        assert choice.message.reasoning_content == "x" * (1024 - len(CLOSE))
        assert choice.message.content == "Let me run it."
        assert resp.usage.completion_tokens_details.reasoning_tokens == 1024
        assert resp.usage.completion_tokens <= 2048

    async def test_an_explicit_allowance_is_honoured(self, monkeypatch):
        FakeLlamaCpp(monkeypatch)
        fake = llama(("x" * 5000, "stop"), ("<tool_call>", "stop"), (XML_CALL, "stop"))
        resp = await complete(fake, chat(max_reasoning_tokens=300))
        reason, answer, _ = fake.calls
        assert reason["max_tokens"] == 300 - len(CLOSE)
        assert answer["max_tokens"] == (2048 - 300) - RESERVE
        assert resp.usage.completion_tokens_details.reasoning_tokens == 300
        assert resp.choices[0].finish_reason == "tool_calls"

    async def test_a_naturally_closed_block_is_not_cut(self, monkeypatch):
        FakeLlamaCpp(monkeypatch)
        fake = llama(("Scale it.\n</think>\n\nignored", "stop"), ("<tool_call>", "stop"),
                     (XML_CALL, "stop"))
        resp = await complete(fake, chat())
        _, answer, _ = fake.calls
        assert answer["prompt"].endswith("<think>\nScale it.\n</think>\n\n")
        assert resp.usage.completion_tokens_details.reasoning_tokens == len("Scale it.\n</think>")
        assert resp.choices[0].message.reasoning_content == "Scale it."

    @pytest.mark.parametrize("choice", ["required",
                                        {"type": "function", "function": {"name": "scale_tensor"}}])
    async def test_forced_goes_straight_to_the_call_after_the_cut(self, monkeypatch, choice):
        FakeLlamaCpp(monkeypatch)
        fake = llama(("x" * 5000, "stop"), (XML_CALL, "stop"))
        resp = await complete(fake, chat(tool_choice=choice))
        reason, call = fake.calls  # no free answer phase
        assert call["prompt"] == reason["prompt"] + "x" * (1024 - len(CLOSE)) + CLOSE + SEP + \
            "<tool_call>"
        assert call["max_tokens"] == 2048 - 1024
        assert resp.choices[0].finish_reason == "tool_calls"  # it was "length" before the fix
        assert resp.usage.completion_tokens_details.reasoning_tokens == 1024


class TestNeverAHalfOpenedMarker:
    async def test_a_call_the_budget_cuts_is_length_with_the_marker_stripped(self, monkeypatch):
        FakeLlamaCpp(monkeypatch)
        fake = llama(("ok\n</think>", "stop"), ("Run it.\n<tool_call>", "stop"),
                     ("\n<function=python" + "y" * 5000, "stop"))
        resp = await complete(fake, chat(max_tokens=400))
        choice = resp.choices[0]
        assert choice.finish_reason == "length" and not choice.message.tool_calls
        assert choice.message.content == "Run it."
        assert "<tool_call>" not in choice.message.content
        assert choice.message.reasoning_content == "ok"
        assert resp.usage.completion_tokens == 400

    async def test_a_marker_prefix_the_budget_leaves_is_stripped(self, monkeypatch):
        FakeLlamaCpp(monkeypatch)
        answer_budget = (400 - len("ok\n</think>")) - min(RESERVE, (400 - 11) // 2)
        text = "a" * (answer_budget - len("<tool_")) + "<tool_"
        fake = llama(("ok\n</think>", "stop"), (text + "call>", "stop"))
        resp = await complete(fake, chat(max_tokens=400))
        assert fake.calls[1]["max_tokens"] == answer_budget
        choice = resp.choices[0]
        assert choice.finish_reason == "length"
        assert choice.message.content == "a" * (answer_budget - len("<tool_"))

    async def test_the_reserve_is_never_more_than_half_of_what_is_left(self, monkeypatch):
        FakeLlamaCpp(monkeypatch)
        fake = llama(("Plain answer.", "stop"))
        resp = await complete(fake, chat(max_tokens=100, chat_template_kwargs={
            "enable_thinking": False}))
        (answer,) = fake.calls
        assert answer["max_tokens"] == 100 - 50  # RESERVE (> 50) capped at half
        assert resp.choices[0].message.content == "Plain answer."
        assert resp.usage.completion_tokens_details.reasoning_tokens == 0

    async def test_an_end_of_turn_inside_auto_reasoning_is_the_whole_reply(self, monkeypatch):
        FakeLlamaCpp(monkeypatch)
        fake = llama(("I give up", "stop"))
        resp = await complete(fake, chat())
        assert len(fake.calls) == 1
        choice = resp.choices[0]
        assert choice.finish_reason == "stop" and choice.message.reasoning_content == "I give up"


class TestStreaming:
    async def test_reasoning_then_content_then_one_call_with_reasoning_tokens(self, monkeypatch):
        FakeLlamaCpp(monkeypatch)
        fake = llama(("x" * 5000, "stop"), ("Running.\n<tool_call>", "stop"), (XML_CALL, "stop"))
        events = await stream(fake, chat(stream=True))
        deltas = [c["delta"] for e in events for c in e.get("choices", [])]
        kinds = [k for d in deltas for k in ("reasoning_content", "content", "tool_calls")
                 if d.get(k)]
        # Reasoning first, then content, then the call: no reasoning after content.
        assert kinds.index("content") > max(i for i, k in enumerate(kinds)
                                            if k == "reasoning_content")
        assert kinds[-1] == "tool_calls"
        # (the newline miServer writes before its `</think>` streams as reasoning, as a model's does)
        assert "".join(d.get("reasoning_content") or "" for d in deltas).rstrip("\n") == \
            "x" * (1024 - len(CLOSE))
        assert "".join(d.get("content") or "" for d in deltas).strip() == "Running."
        final = [e for e in events if e.get("usage")][-1]
        assert final["usage"]["completion_tokens_details"] == {"reasoning_tokens": 1024}
        assert [c.get("finish_reason") for e in events for c in e.get("choices", [])][-1] == \
            "tool_calls"

    async def test_a_budget_cut_call_in_a_stream_sends_no_marker(self, monkeypatch):
        FakeLlamaCpp(monkeypatch)
        fake = llama(("ok\n</think>", "stop"), ("Run.\n<tool_call>", "stop"),
                     ("\n<function=python" + "y" * 5000, "stop"))
        events = await stream(fake, chat(stream=True, max_tokens=400))
        content = "".join(c["delta"].get("content") or "" for e in events
                          for c in e.get("choices", []))
        assert "<tool" not in content and content.strip() == "Run."
        assert [c.get("finish_reason") for e in events for c in e.get("choices", [])][-1] == \
            "length"


async def test_the_body_carries_reasoning_tokens_without_thinking_too(monkeypatch):
    """`0` when the prompt opened no block (the field is present on the tool path)."""
    FakeLlamaCpp(monkeypatch)
    fake = llama(("<tool_call>", "stop"), (XML_CALL, "stop"))
    resp = await complete(fake, chat(chat_template_kwargs={"enable_thinking": False}))
    body = json.loads(resp.model_dump_json())
    assert body["usage"]["completion_tokens_details"] == {"reasoning_tokens": 0}
    assert body["choices"][0]["finish_reason"] == "tool_calls"
