"""A turn that ends inside its reasoning on a complete call (live check of `dfa78213`, 2026-11-22).

Re-sending WSP run `wsp_7e06a7ab41cd`'s turn 8 on the misclosed-tag fix, one reply still lost its
call: Qwen3.6 wrote no closing tag at all, then `<tool_call>...</tool_call>`, and ended its turn.
A marker in the middle of reasoning is still not a call. A well-formed call of an offered function,
on its own line, that ENDS the turn is one: the block is closed before it and the call is
generated under the grammar. Nothing is parsed out of free text into a returned call.
"""

from __future__ import annotations

import json

import pytest

from tests.unit.services.test_gguf_tool_calls import (

    QWEN36,
    XML_CALL,
    FakeLlama,
    FakeLlamaCpp,
    chat,
    clear_loaded,
    complete,
    stream,
)

CALL = f"<tool_call>{XML_CALL}"
OTHER = CALL.replace("\n2\n", "\n3\n")
LIVE = f"One thing I notice is that the bias is... let me check.\n\n{CALL}"


@pytest.fixture(autouse=True)
def _clean():
    clear_loaded()
    yield
    clear_loaded()


def thinking(**over):
    over.setdefault("max_tokens", 4096)
    return chat(chat_template_kwargs=None, **over)


def streamed(events):
    choices = [c for e in events for c in e.get("choices", [])]
    text = {k: "".join(c["delta"].get(k) or "" for c in choices)
            for k in ("reasoning_content", "content")}
    calls = [c["delta"]["tool_calls"] for c in choices if c["delta"].get("tool_calls")]
    return text["reasoning_content"], text["content"], calls


@pytest.mark.parametrize("end", ["", "\n", " \n"])
async def test_the_call_that_ends_the_turn_is_made_under_the_grammar(monkeypatch, end):
    FakeLlamaCpp(monkeypatch)
    # Regenerated: the returned arguments are the grammar phase's, never the free text's.
    llama = FakeLlama(QWEN36, [(LIVE + end, "stop"), (XML_CALL.replace("\n2\n", "\n4\n"), "stop")])
    resp = await complete(llama, thinking())
    reason, call = llama.calls
    assert "grammar" not in reason and call.get("grammar") is not None
    assert call["prompt"].endswith("let me check.\n\n</think>\n\n<tool_call>")
    choice = resp.choices[0]
    assert choice.finish_reason == "tool_calls"
    (made,) = choice.message.tool_calls
    assert json.loads(made["function"]["arguments"]) == {"name": "weight", "factor": 4}
    assert choice.message.reasoning_content.strip().endswith("let me check.")
    assert not (choice.message.content or "").strip()


async def test_several_calls_are_cut_before_the_first(monkeypatch):
    FakeLlamaCpp(monkeypatch)
    llama = FakeLlama(QWEN36, [(f"Both.\n{CALL}\n{OTHER}", "stop"), (XML_CALL, "stop")])
    await complete(llama, thinking())
    assert llama.calls[1]["prompt"].endswith("Both.\n</think>\n\n<tool_call>")


async def test_only_the_run_that_ends_the_turn_is_cut(monkeypatch):
    FakeLlamaCpp(monkeypatch)
    text = f"An example:\n{CALL}\nNo, rather this.\n{OTHER}"
    llama = FakeLlama(QWEN36, [(text, "stop"), (XML_CALL, "stop")])
    await complete(llama, thinking())
    assert llama.calls[1]["prompt"].endswith("No, rather this.\n</think>\n\n<tool_call>")


async def test_a_turn_that_is_only_the_call(monkeypatch):
    FakeLlamaCpp(monkeypatch)
    llama = FakeLlama(QWEN36, [(CALL, "stop"), (XML_CALL, "stop")])
    resp = await complete(llama, thinking())
    assert resp.choices[0].finish_reason == "tool_calls"


async def test_the_discarded_free_call_is_still_counted(monkeypatch):
    FakeLlamaCpp(monkeypatch)
    llama = FakeLlama(QWEN36, [(LIVE, "stop"), (XML_CALL, "stop")])
    resp = await complete(llama, thinking())
    # The fake tokenizer is one token a character: what was generated was paid for.
    assert resp.usage.completion_tokens >= len(LIVE) + len(XML_CALL)
    assert llama.calls[1]["max_tokens"] <= 4096 - len(LIVE)


async def test_streamed_the_free_call_is_never_sent(monkeypatch):
    FakeLlamaCpp(monkeypatch)
    llama = FakeLlama(QWEN36, [(LIVE, "stop"), (XML_CALL, "stop")])
    reasoning, content, calls = streamed(await stream(llama, thinking(stream=True)))
    assert "<tool" not in reasoning and "<function" not in reasoning and not content.strip()
    assert reasoning.rstrip().endswith("let me check.") and calls


@pytest.mark.parametrize("text", [
    f"An example is\n{CALL}\nbut I will not make it.",
    f"I could write {CALL}",  # not on a line of its own
    "Maybe\n<tool_call>\n<function=scale_tensor>\n<parameter=name>\nweight",  # unfinished
    f"Not offered\n{CALL.replace('scale_tensor', 'delete_everything')}",
    "Not a call\n<tool_call>\nsomething\n</tool_call>",
])
async def test_anything_else_is_still_reasoning_and_is_not_lost(monkeypatch, text):
    FakeLlamaCpp(monkeypatch)
    llama = FakeLlama(QWEN36, [(text, "stop")])
    resp = await complete(llama, thinking())
    assert len(llama.calls) == 1
    assert resp.choices[0].finish_reason == "stop" and not resp.choices[0].message.tool_calls
    clear_loaded()
    again = FakeLlama(QWEN36, [(text, "stop")])
    reasoning, _, calls = streamed(await stream(again, thinking(stream=True)))
    assert reasoning == text and not calls


async def test_a_mention_does_not_hold_the_stream_back():
    from miserver.services.llamacpp_tools import _call_hold
    from miserver.services.tool_calls import QWEN3_XML

    after_prose = f"An example\n{CALL}\nbut no, and more thought"
    assert _call_hold(after_prose, "<tool_call>", QWEN3_XML) == len(after_prose)
    not_a_call = "The tag\n<tool_call> opens a call, as the template says"
    assert _call_hold(not_a_call, "<tool_call>", QWEN3_XML) == len(not_a_call)


async def test_a_length_stop_is_not_the_end_of_a_turn(monkeypatch):
    FakeLlamaCpp(monkeypatch)
    llama = FakeLlama(QWEN36, [(LIVE, "length"), ("No call after all.", "stop")])
    resp = await complete(llama, thinking())
    assert all("grammar" not in c for c in llama.calls)
    assert not resp.choices[0].message.tool_calls


async def test_a_real_close_after_a_drafted_call_is_untouched(monkeypatch):
    FakeLlamaCpp(monkeypatch)
    llama = FakeLlama(QWEN36, [
        (f"Draft:\n{CALL}\n</think>\n\n", "stop"), ("I decided against it.", "stop")])
    resp = await complete(llama, thinking())
    assert all("grammar" not in c for c in llama.calls)
    assert resp.choices[0].message.content == "I decided against it."


async def test_it_is_logged_with_the_misclosed_tag(monkeypatch):
    from structlog.testing import capture_logs

    FakeLlamaCpp(monkeypatch)
    llama = FakeLlama(QWEN36, [(LIVE, "stop"), (XML_CALL, "stop")])
    with capture_logs() as logs:
        await complete(llama, thinking())
    (line,) = [e for e in logs if e.get("event") == "llamacpp_tool_generation"]
    assert line["think_misclosed"] is True


def test_the_hermes_format_is_found_too():
    from miserver.services.llamacpp_tools import _trailing_call_at
    from miserver.services.tool_calls import HERMES

    call = '<tool_call>\n{"name": "describe_tensor", "arguments": {"name": "w"}}\n</tool_call>'

    class Plan:
        fmt = HERMES
        names = ("describe_tensor",)

    assert _trailing_call_at(f"Look.\n{call}", "<tool_call>", Plan()) == len("Look.\n")


def test_the_run_holds_calls_only_and_is_closed():
    """Asked of the detector itself: through the stream, what was already sent bounds the cut and
    would hide both rules."""
    from miserver.services.llamacpp_tools import _trailing_call_at
    from miserver.services.tool_calls import QWEN3_XML, ToolPlan
    from tests.unit.services.test_gguf_tool_calls import TOOLS

    plan = ToolPlan(tools=TOOLS, fmt=QWEN3_XML,
                    names=frozenset(t["function"]["name"] for t in TOOLS))
    text = f"An example:\n{CALL}\nNo, rather this.\n{OTHER}"
    assert _trailing_call_at(text, "<tool_call>", plan) == text.rindex("<tool_call>")
    unclosed = LIVE.removesuffix("</tool_call>").rstrip()
    assert _trailing_call_at(unclosed, "<tool_call>", plan) is None
