"""A tool call after a misclosed reasoning block (WSP run `wsp_7e06a7ab41cd`, 2026-11-22).

Qwen3.6-27B-GGUF ended three turns of that run with `</thinking>\\n\\n<tool_call>...</tool_call>`:
the wrong closing tag, then a complete call. miServer looks for `</think>`, so the whole reply was
reasoning, `finish_reason` was `stop`, and the client got no call (three such turns ended the run).
Replaying the three requests reproduced it in 4 of 6 tries.

`</thinking>` with nothing but whitespace before the call marker now closes the block, and the
call is generated under the grammar exactly as after `</think>`. A marker, or that tag, anywhere
else in the reasoning is still reasoning.
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


@pytest.fixture(autouse=True)
def _clean():
    clear_loaded()
    yield
    clear_loaded()


def thinking(**over):
    return chat(chat_template_kwargs=None, max_tokens=4096, **over)


@pytest.mark.parametrize("gap", ["\n\n", "\n", "", " \n \n"])
async def test_the_call_is_made_under_the_grammar(monkeypatch, gap):
    FakeLlamaCpp(monkeypatch)
    llama = FakeLlama(QWEN36, [
        (f"Let me scale it.\n</thinking>{gap}<tool_call>{XML_CALL}", "stop"),
        (XML_CALL, "stop")])
    resp = await complete(llama, thinking())
    reason, call = llama.calls
    assert "grammar" not in reason and call.get("grammar") is not None
    assert call["prompt"].endswith("<think>\nLet me scale it.\n</think>\n\n<tool_call>")
    choice = resp.choices[0]
    assert choice.finish_reason == "tool_calls"
    (made,) = choice.message.tool_calls
    assert made["function"]["name"] == "scale_tensor"
    assert json.loads(made["function"]["arguments"]) == {"name": "weight", "factor": 2}
    assert choice.message.reasoning_content.strip() == "Let me scale it."
    assert not (choice.message.content or "").strip()


async def test_a_marker_mentioned_earlier_does_not_hide_the_misclose(monkeypatch):
    FakeLlamaCpp(monkeypatch)
    llama = FakeLlama(QWEN36, [
        ("I could use <tool_call> here. I will.\n</thinking>\n\n<tool_call>", "stop"),
        (XML_CALL, "stop")])
    resp = await complete(llama, thinking())
    assert resp.choices[0].finish_reason == "tool_calls"
    assert llama.calls[1]["prompt"].endswith("here. I will.\n</think>\n\n<tool_call>")


async def test_the_block_is_closed_on_its_own_line(monkeypatch):
    FakeLlamaCpp(monkeypatch)
    llama = FakeLlama(QWEN36, [("Scale it.</thinking>\n<tool_call>", "stop"), (XML_CALL, "stop")])
    await complete(llama, thinking())
    assert llama.calls[1]["prompt"].endswith("Scale it.\n</think>\n\n<tool_call>")


async def test_streamed_neither_the_tag_nor_the_marker_is_sent(monkeypatch):
    FakeLlamaCpp(monkeypatch)
    llama = FakeLlama(QWEN36, [
        (f"Let me scale it.\n</thinking>\n\n<tool_call>{XML_CALL}", "stop"), (XML_CALL, "stop")])
    events = await stream(llama, thinking(stream=True))
    choices = [c for e in events for c in e.get("choices", [])]
    reasoning = "".join(c["delta"].get("reasoning_content") or "" for c in choices)
    content = "".join(c["delta"].get("content") or "" for c in choices)
    # Exactly the block: had the real close not streamed, its separator would be reasoning too.
    assert reasoning == "Let me scale it.\n" and not content.strip()
    assert [c["delta"]["tool_calls"] for c in choices if c["delta"].get("tool_calls")]
    assert [c.get("finish_reason") for c in choices][-1] == "tool_calls"


@pytest.mark.parametrize("text", [
    "Do I write </thinking> or </think>? I give up",
    "The tag </thinking> comes first.\nThen <tool_call> would follow, but no",
    "A bare <tool_call> in my reasoning, then nothing",
])
async def test_the_tag_or_the_marker_elsewhere_is_still_reasoning(monkeypatch, text):
    FakeLlamaCpp(monkeypatch)
    # "</think>" inside the first text closes the block, as it always did.
    replies = [(text, "stop"), ("No call needed.", "stop")]
    llama = FakeLlama(QWEN36, replies)
    resp = await complete(llama, thinking())
    assert all("grammar" not in c for c in llama.calls)
    assert resp.choices[0].finish_reason == "stop" and not resp.choices[0].message.tool_calls


@pytest.mark.parametrize("text", [
    "I was going to close with </thinking>",
    "I was going to close with </thinking>\n\n<tool",
    "ends on a partial </think",
])
async def test_text_held_back_for_the_tag_is_not_lost(monkeypatch, text):
    FakeLlamaCpp(monkeypatch)
    events = await stream(FakeLlama(QWEN36, [(text, "stop")]), thinking(stream=True))
    sent = "".join(c["delta"].get("reasoning_content") or ""
                   for e in events for c in e.get("choices", []))
    assert sent == text


async def test_a_real_close_is_untouched(monkeypatch):
    FakeLlamaCpp(monkeypatch)
    llama = FakeLlama(QWEN36, [
        ("Think.\n</think>\n\n", "stop"), ("<tool_call>", "stop"), (XML_CALL, "stop")])
    resp = await complete(llama, thinking())
    assert len(llama.calls) == 3 and resp.choices[0].finish_reason == "tool_calls"


async def test_the_call_is_still_validated(monkeypatch):
    from tests.unit.services.test_gguf_tool_calls import ConstrainedOutputInvalidError

    FakeLlamaCpp(monkeypatch)
    llama = FakeLlama(QWEN36, [
        ("Scale.\n</thinking>\n\n<tool_call>", "stop"),
        (XML_CALL.replace("\n2\n", "\n11\n"), "stop")])
    with pytest.raises(ConstrainedOutputInvalidError):
        await complete(llama, thinking())


async def test_it_is_logged_so_the_rate_can_be_seen(monkeypatch):
    from structlog.testing import capture_logs

    FakeLlamaCpp(monkeypatch)
    llama = FakeLlama(QWEN36, [("Scale.\n</thinking>\n\n<tool_call>", "stop"), (XML_CALL, "stop")])
    with capture_logs() as logs:
        await complete(llama, thinking())
    (line,) = [e for e in logs if e.get("event") == "llamacpp_tool_generation"]
    assert line["think_misclosed"] is True
