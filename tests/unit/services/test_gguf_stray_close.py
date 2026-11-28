"""A turn that ends inside its reasoning on a call's closing tags alone (live check, 2026-11-22).

The third shape found re-sending WSP run `wsp_7e06a7ab41cd`'s turn 19: "Let me try this.", then
`</parameter></function></tool_call>` and end of turn, with no call opened. There is no call to
recover. Closing the block and letting the model continue made a real call in 5 of 5 tries on
that reply, so miServer does that; whether to call stays the model's choice.
"""

from __future__ import annotations

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

STRAY = "Let me try this.\n</parameter>\n</function>\n</tool_call>"


@pytest.fixture(autouse=True)
def _clean():
    clear_loaded()
    yield
    clear_loaded()


def thinking(**over):
    return chat(chat_template_kwargs=None, max_tokens=4096, **over)


async def test_the_block_is_closed_and_the_model_goes_on_to_its_call(monkeypatch):
    FakeLlamaCpp(monkeypatch)
    llama = FakeLlama(QWEN36, [(STRAY, "stop"), ("<tool_call>", "stop"), (XML_CALL, "stop")])
    resp = await complete(llama, thinking())
    reason, answer, call = llama.calls
    assert "grammar" not in reason and "grammar" not in answer
    assert call.get("grammar") is not None
    assert answer["prompt"].endswith(f"{STRAY}\n</think>\n\n")
    assert resp.choices[0].finish_reason == "tool_calls"


async def test_an_answer_stays_the_models_choice(monkeypatch):
    FakeLlamaCpp(monkeypatch)
    llama = FakeLlama(QWEN36, [(STRAY, "stop"), ("No edit is needed.", "stop")])
    resp = await complete(llama, thinking())
    assert len(llama.calls) == 2
    choice = resp.choices[0]
    assert choice.finish_reason == "stop" and not choice.message.tool_calls
    assert choice.message.content == "No edit is needed."


async def test_an_earlier_drafted_call_does_not_hide_it(monkeypatch):
    FakeLlamaCpp(monkeypatch)
    text = f"Draft:\n<tool_call>{XML_CALL}\nNo. {STRAY}"
    llama = FakeLlama(QWEN36, [(text, "stop"), ("Done.", "stop")])
    await complete(llama, thinking())
    assert len(llama.calls) == 2


async def test_streamed_what_follows_is_not_reasoning(monkeypatch):
    FakeLlamaCpp(monkeypatch)
    llama = FakeLlama(QWEN36, [(STRAY, "stop"), ("No edit is needed.", "stop")])
    events = await stream(llama, thinking(stream=True))
    choices = [c for e in events for c in e.get("choices", [])]
    assert "".join(c["delta"].get("content") or "" for c in choices).strip() == "No edit is needed."


@pytest.mark.parametrize("text", [
    "I give up",
    "The closing tag is </tool_call>, I think, but no",
    f"I could write <tool_call>{XML_CALL}",
])
async def test_any_other_end_of_turn_is_the_whole_reply(monkeypatch, text):
    FakeLlamaCpp(monkeypatch)
    llama = FakeLlama(QWEN36, [(text, "stop")])
    resp = await complete(llama, thinking())
    assert len(llama.calls) == 1 and resp.choices[0].finish_reason == "stop"


async def test_it_is_logged(monkeypatch):
    from structlog.testing import capture_logs

    FakeLlamaCpp(monkeypatch)
    llama = FakeLlama(QWEN36, [(STRAY, "stop"), ("<tool_call>", "stop"), (XML_CALL, "stop")])
    with capture_logs() as logs:
        await complete(llama, thinking())
    (line,) = [e for e in logs if e.get("event") == "llamacpp_tool_generation"]
    assert line["think_misclosed"] is True


async def test_it_is_logged_for_an_answer_too(monkeypatch):
    from structlog.testing import capture_logs

    FakeLlamaCpp(monkeypatch)
    llama = FakeLlama(QWEN36, [(STRAY, "stop"), ("No edit is needed.", "stop")])
    with capture_logs() as logs:
        await complete(llama, thinking())
    (line,) = [e for e in logs if e.get("event") == "llamacpp_tool_generation"]
    assert line["think_misclosed"] is True
