"""Stateless rule scoring runs its validator scans off the event loop (DEBT_TASKS 5.3; 039 R1-F6).

`stateless_rule_results` scores rule signals for `POST /api/probes/score` and `/score-chat`. The scan over a long
text is unbounded inside one validator, so it must not run on the loop that streams every other request: it runs in
`asyncio.to_thread(evaluate_work, …)`, while `tokenizer.decode` stays on the loop (an HF fast tokenizer is not safe
across threads while inference runs). The fix was on `main` (`efe88f48`, 2026-10-24) but no test held it: moving the
scan back onto the loop left 839 tests green (2026-11-26). This test fails then.

The spy records where `evaluate_work` ran, how often, and with what (the payload, not only the call).
"""

from __future__ import annotations

import asyncio
import threading
from types import SimpleNamespace
from typing import Any

import pytest


class _Tokenizer:
    def __init__(self) -> None:
        self.decoded_on: list[int] = []

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        self.decoded_on.append(threading.get_ident())
        return "ZQ-1234 " + " ".join(str(i) for i in ids)


class _Code:
    async def resolve(self, work: Any, texts: Any, items: Any) -> Any:
        return items


@pytest.mark.asyncio
async def test_the_stateless_scan_runs_off_the_loop_and_decode_stays_on_it(monkeypatch):
    from miserver.rules import registry as rule_registry
    from miserver.rules import serving
    from miserver.services.probe_scoring import stateless_rule_results

    attached = SimpleNamespace(name="sentinel set")
    monkeypatch.setattr(rule_registry, "current",
                        lambda: SimpleNamespace(for_behaviour=lambda key: (attached,) if key == "b" else ()))
    calls: list[dict[str, Any]] = []

    def spy(work, texts, *, clock, deadline, return_positions=False):
        try:
            asyncio.get_running_loop()
            on_loop = True  # called from a coroutine: the loop is blocked meanwhile
        except RuntimeError:
            on_loop = False  # a worker thread has no running loop
        calls.append({"on_loop": on_loop, "work": work, "prompt": texts.texts.get("prompt"),
                      "thread": threading.get_ident()})
        return []

    monkeypatch.setattr(serving, "evaluate_work", spy)
    probes = [SimpleNamespace(descriptor=SimpleNamespace(behaviour_key="b"), windows=("prompt",)),
              SimpleNamespace(descriptor=SimpleNamespace(behaviour_key="other"), windows=("prompt",))]
    request = SimpleNamespace(behaviours=["b"], return_rule_positions=False)
    item = SimpleNamespace(messages=None, text=None, token_ids=[5, 7])
    tokenizer = _Tokenizer()
    loop_thread = threading.get_ident()

    out, decoded = await stateless_rule_results(request, item, probes, tokenizer, code=_Code())

    assert decoded is True and out == []
    assert len(calls) == 1, calls
    call = calls[0]
    assert call["on_loop"] is False, "the validator scan ran on the event loop"
    assert call["thread"] != loop_thread
    assert call["work"] == [("b", ("prompt",), attached)]
    assert call["prompt"] == "ZQ-1234 5 7"
    assert tokenizer.decoded_on == [loop_thread], "tokenizer.decode left the event loop's thread"
