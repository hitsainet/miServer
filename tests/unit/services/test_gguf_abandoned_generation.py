"""Abandoned GGUF generations stop, and `max_time_s` bounds one (2026-11-22).

THE DEFECT (production, 2026-11-21). A non-streaming GGUF chat request (Qwen3.6-27B-GGUF, tools, a
two-phase qwen3_xml tool run, `max_tokens` 32000) was abandoned by its client at 23:37:49 after a
600 s timeout. miServer generated on until 00:13:31 (all 32,000 tokens, `llamacpp_tool_generation`
finish_reason length), holding the model slot for 46 minutes; two requests queued behind it failed
their callers at 900 s. Every llama.cpp call ran in `asyncio.to_thread` with no stop hook, and
nothing watched the connection.

What is pinned here, with llama.cpp STUBBED by `ForeverLlama` (generates until stopped; no GPU,
no model):
* a disconnect stops generation (non-streaming chat, streaming chat, text completions), the slot
  is released only after the thread has left llama.cpp, and a queued request is then served;
* the stop reaches EVERY phase: the free/reasoning phase, the auto answer, the call grammar phase,
  the document grammar phase, the continuation's raw completion, the plain chat (through the chat
  handler's inner `create_completion`), the text completion;
* `max_time_s` ends free generation as a length stop with usage counted, and a tool run whose
  reasoning the deadline cut still returns a well-formed call (the grammar phase runs to the
  grace); a grammar phase that overruns the grace ends `length` with no call;
* the route builds the watch from the request (raw ASGI with a receive that disconnects), and the
  policy honours `max_time_s` on llama.cpp only.

MUTATION CONTROLS (each turns this file red; all 30, with results, in
`0xcc/reviews/gguf_abandoned_generation_2026-11-22.md`):
  A1 `_free`'s criteria removed             A2 `_call`'s removed (call and document grammar)
  A3 `_llamacpp_sync` continuation's         A4 `_llamacpp_sync`'s `completions_carry` removed
  A5 the stream's continuation criteria      A6 the stream's `completions_carry` removed
  A7 the text completion's criteria          A8/A9/A9b `watch_client` given no client (chat, stream, text)
  A10/A10b/A10c the routes' `client=` not passed (chat, chat stream, completions)
  A11 `BlockingPull.aclose` skips the drain  A11b the stream's `finally` does not set the stop
  A11c `run_in_thread` re-raises a cancel without abandoning and draining
  A12a/b/c `max_time_s` not read (chat, stream, text)
  A13 a deadline in `_free` not mapped to "length"   A14 no grace for the grammar phase
  A15 the (CHAT, LC) `max_time_s` cell refused       A16 transformers chat's `_carry_abandonment` removed
  A17 `stream_chat_completion` does not close the delegated generator
  A18 `_llamacpp_sync`'s deadline not mapped to "length"   A19 `_call`'s not mapped
  A20 no `raise_if_abandoned` after a free phase     A21 the stream's pull loop ignores the stop
  A22 the criteria's per-length latch removed        A23 transformers text completion's check removed
  A24 the transformers `max_time_s` refusal for a direct caller removed
A11, A11c, A12b and A23 survived the first batch; the prefill, stream-deadline and text-completion tests
below were written for them, and each then turned red.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
from typing import Any

import pytest
import structlog

from miserver.api.request_policy import OUTPUT_CHANGING, Endpoint, Engine, Honoured, evaluate
from miserver.api.schemas.openai import ChatCompletionRequest, TextCompletionRequest
from miserver.core.errors import FieldNotHonouredError, GenerationAbandonedError
from miserver.ml.model_loader import ENGINE_LLAMACPP, LoadedModelState
from miserver.services import generation_stop as gs
from miserver.services.tool_calls import parse_qwen3_xml  # noqa: F401 - the format under test
from tests.unit.f25_fixtures import clear_loaded, make_client, make_service, model_row
from tests.unit.services.test_gguf_tool_calls import (
    QWEN36,
    TOOLS,
    USER,
    XML_CALL,
    FakeLlama,
    FakeLlamaCpp,
)

#: Seconds per generated token, and the most tokens a call may take before it is declared
#: unstoppable (a red test, never a hung one).
TOKEN_S = 0.002
CAP = 2500
DOC_SCHEMA = {"type": "json_schema", "json_schema": {
    "name": "answer", "schema": {"type": "object", "properties": {"a": {"type": "string"}},
                                 "required": ["a"]}}}


@pytest.fixture(autouse=True)
def _clean():
    clear_loaded()
    yield
    clear_loaded()


# =============================================================================
# The stand-in: a Llama that generates until it is stopped
# =============================================================================


class ForeverLlama(FakeLlama):
    """`FakeLlama` (character tokenizer, real Qwen3.6 template) whose `create_completion` runs as
    llama-cpp-python 0.3.36's does: one token per step, the stopping criteria checked after each
    sampled token (the stopping token is not returned, finish "stop"), and once more after the loop
    with the same ids; `max_tokens` ends it "length".

    `script`: one behaviour per call, in order: "forever" (generate until stopped), or a text the
    call produces (ten times faster) and then ends "stop". With no script left, "forever". A call
    nothing stops runs
    to `CAP` tokens and is recorded as `capped`, so a missing stop turns a test red, not hung.

    `create_chat_completion` calls `self.create_completion` as llama-cpp-python's chat handler does
    (an attribute lookup on the instance), with the handler's own `stopping_criteria=None`.
    """

    def __init__(self, template: str | None = QWEN36, script: list[str] | None = None,
                 prefill_s: float = 0.0):
        super().__init__(template, [])
        self.script = list(script or [])
        #: The FIRST call's first step takes this long: a prompt prefill, which llama.cpp cannot
        #: interrupt (the criteria run only after a token is sampled). `in_prefill` is set in it.
        self.prefill_s = prefill_s
        self.in_prefill = threading.Event()
        self.records: list[dict[str, Any]] = []
        self.started = threading.Event()
        self._lock = threading.Lock()
        self._busy = 0
        self.max_busy = 0

    # -- concurrency: never two decodes at once on one context --------------------------------

    def _step(self, seconds: float) -> None:
        with self._lock:
            self._busy += 1
            self.max_busy = max(self.max_busy, self._busy)
        time.sleep(seconds)
        with self._lock:
            self._busy -= 1

    def create_completion(self, prompt, stream=False, stopping_criteria=None, max_tokens=None,
                          grammar=None, **kw):
        behaviour = self.script.pop(0) if self.script else "forever"
        text = None if behaviour == "forever" else behaviour
        ids = list(prompt) if isinstance(prompt, list) else [ord(c) for c in prompt]
        record: dict[str, Any] = {
            "prompt": self.text_of(ids), "stream": stream, "grammar": grammar is not None,
            "criteria": stopping_criteria is not None, "max_tokens": max_tokens,
            "tokens": 0, "yielded": 0, "stopped": False, "capped": False, "closed": False,
            "finish": None, "ended_at": None,
        }
        self.records.append(record)
        limit = max_tokens if max_tokens is not None else 10 ** 9

        prefill = self.prefill_s if len(self.records) == 1 else 0.0

        def tokens():
            i = 0
            while True:
                if text is not None and i >= len(text):
                    record["finish"] = "stop"
                    return
                if i == 0 and prefill:
                    self.in_prefill.set()
                    self._step(prefill)
                # "forever" decodes at TOKEN_S a token; a scripted text at a tenth of that.
                self._step(TOKEN_S if text is None else TOKEN_S / 10)
                piece = text[i] if text is not None else "x"
                ids.append(ord(piece))
                i += 1
                record["tokens"] += 1
                if text is None and record["tokens"] >= 3:
                    self.started.set()
                if stopping_criteria is not None and stopping_criteria(ids, None):
                    # As `Llama.generate`: the stopping token is never returned.
                    record["stopped"] = True
                    record["finish"] = "stop"
                    return
                record["yielded"] += 1
                yield piece
                if record["tokens"] >= limit:
                    record["finish"] = "length"
                    return
                if record["tokens"] >= CAP:
                    record["capped"] = True
                    record["finish"] = "length"
                    return

        def finish():
            # llama-cpp-python checks the criteria once more after its loop, same ids.
            if stopping_criteria is not None and stopping_criteria(ids, None):
                record["finish"] = "stop"
            record["ended_at"] = time.monotonic()

        if not stream:
            out = "".join(tokens())
            finish()
            return {"choices": [{"text": out, "finish_reason": record["finish"]}],
                    "usage": {"prompt_tokens": len(prompt), "completion_tokens": record["yielded"],
                              "total_tokens": len(prompt) + record["yielded"]}}

        def chunks():
            try:
                for piece in tokens():
                    yield {"choices": [{"text": piece, "finish_reason": None}]}
                finish()
                yield {"choices": [{"text": "", "finish_reason": record["finish"]}]}
            finally:
                if record["ended_at"] is None:  # closed before it finished: sampling ends here
                    record["closed"] = True
                    record["ended_at"] = time.monotonic()

        return chunks()

    def create_chat_completion(self, messages, stream=False, **params):
        """llama-cpp-python's chat handler, reduced: it renders, then calls
        `llama.create_completion(..., stopping_criteria=<template stop ids or None>)`."""
        prompt = [ord(c) for c in "".join(m.get("content") or "" for m in messages)]
        raw = self.create_completion(prompt=prompt, stream=stream, stopping_criteria=None,
                                     **params)
        if not stream:
            choice = raw["choices"][0]
            return {"choices": [{"index": 0, "message": {"role": "assistant",
                                                         "content": choice["text"]},
                                 "finish_reason": choice["finish_reason"]}],
                    "usage": raw["usage"]}

        def chat_chunks(chunks):
            # The inner stream is an ARGUMENT, as in `_convert_text_completion_chunks_to_chat`:
            # closing this generator releases it, which closes it.
            yield {"choices": [{"index": 0, "delta": {"role": "assistant"},
                                "finish_reason": None}]}
            for chunk in chunks:
                c = chunk["choices"][0]
                yield {"choices": [{"index": 0, "delta": ({"content": c["text"]} if c["text"]
                                                          else {}),
                                    "finish_reason": c["finish_reason"]}]}

        return chat_chunks(raw)


def gguf_service(fake: ForeverLlama):
    """A REAL InferenceService (its real request queue: one slot) serving `fake` as llama.cpp."""
    svc = make_service(fake, None)
    LoadedModelState().current.engine = ENGINE_LLAMACPP
    return svc


def chat(**over: Any) -> ChatCompletionRequest:
    body: dict[str, Any] = {"model": "tiny", "messages": [{"role": "user", "content": "hi"}],
                            "max_tokens": 100000}
    body.update(over)
    return ChatCompletionRequest(**body)


def tool_chat(**over: Any) -> ChatCompletionRequest:
    return chat(messages=USER, tools=TOOLS, **over)


class Gone:
    """A `ClientWatch` the test disconnects: `wait()` returns once `leave()` is called."""

    def __init__(self) -> None:
        self._event = asyncio.Event()
        self.watch = gs.ClientWatch(self._event.wait)

    def leave(self) -> None:
        self._event.set()


async def started(fake: ForeverLlama) -> None:
    assert await asyncio.to_thread(fake.started.wait, 10), "generation never started"


async def disconnect_when_started(fake: ForeverLlama, client: Gone) -> float:
    await started(fake)
    at = time.monotonic()
    client.leave()
    return at


def assert_stopped_promptly(record: dict, left_at: float) -> None:
    assert record["criteria"], "the call carried no stopping criteria"
    assert not record["capped"], "nothing stopped the call: it ran to the cap"
    # Stopped by its criterion, or (a stream cancelled between tokens) closed after the token in
    # flight; either way within about one token.
    assert record["stopped"] or record["closed"], record
    assert record["ended_at"] - left_at < 1.0, "it stopped, but not promptly"


# =============================================================================
# A disconnect stops generation and frees the slot
# =============================================================================


class TestADisconnectStopsIt:
    async def test_non_streaming_chat_stops_and_the_next_request_is_served(self):
        fake = ForeverLlama(script=["forever", "second answer"])
        svc = gguf_service(fake)
        client = Gone()
        with structlog.testing.capture_logs() as logs:
            first = asyncio.create_task(svc.create_chat_completion(chat(), client=client.watch))
            await started(fake)
            # Queued behind the abandoned one: admitted only when its slot is free.
            second = asyncio.create_task(svc.create_chat_completion(chat(max_tokens=50)))
            await asyncio.sleep(0.05)
            assert not second.done(), "the second request was not queued behind the first"
            left = time.monotonic()
            client.leave()
            with pytest.raises(GenerationAbandonedError) as exc:
                await asyncio.wait_for(first, 5)
            served = await asyncio.wait_for(second, 5)
        assert exc.value.status_code == 499 and exc.value.code == "GENERATION_ABANDONED"
        assert_stopped_promptly(fake.records[0], left)
        assert served.choices[0].message.content == "second answer"
        assert fake.max_busy == 1, "two decodes overlapped on one llama.cpp context"
        (event,) = [e for e in logs if e["event"] == "generation_abandoned"]
        assert event["reason"] == "client_disconnected" and event["endpoint"] == "chat"
        assert event["tokens_generated"] == fake.records[0]["yielded"]  # the stopping one dropped
        assert event["elapsed_s"] > 0

    async def test_a_tool_run_stops_in_its_reasoning_phase_and_starts_no_call(self, monkeypatch):
        """The incident's shape: tools, qwen3_xml, the template's default thinking."""
        FakeLlamaCpp(monkeypatch)
        fake = ForeverLlama()
        svc = gguf_service(fake)
        client = Gone()
        task = asyncio.create_task(svc.create_chat_completion(tool_chat(), client=client.watch))
        left = await disconnect_when_started(fake, client)
        with pytest.raises(GenerationAbandonedError):
            await asyncio.wait_for(task, 5)
        assert len(fake.records) == 1, "a later phase started after the abandonment"
        assert_stopped_promptly(fake.records[0], left)

    async def test_streaming_chat_stops_on_the_watch_and_frees_the_slot(self):
        """ASGI spec 2.4 servers give Starlette no disconnect listener: the watch is what stops."""
        fake = ForeverLlama(script=["forever", "next"])
        svc = gguf_service(fake)
        client = Gone()
        frames: list[str] = []

        async def consume():
            async for frame in svc.stream_chat_completion(chat(stream=True), client=client.watch):
                frames.append(frame)

        task = asyncio.create_task(consume())
        left = await disconnect_when_started(fake, client)
        await asyncio.wait_for(task, 5)
        assert_stopped_promptly(fake.records[0], left)
        assert "data: [DONE]\n\n" not in frames, "an abandoned stream was finished as if served"
        served = await asyncio.wait_for(svc.create_chat_completion(chat(max_tokens=10)), 5)
        assert served.choices[0].message.content == "next"

    async def test_a_cancelled_stream_waits_for_the_grammar_phase_before_the_slot_is_freed(
        self, monkeypatch,
    ):
        """Starlette (ASGI 2.3) CANCELS the stream on a disconnect. One pull of a forced tool run
        is the whole grammar phase; the slot must not be released while it still decodes."""
        FakeLlamaCpp(monkeypatch)
        fake = ForeverLlama(script=["forever", "after"])
        svc = gguf_service(fake)
        request = tool_chat(stream=True, tool_choice="required",
                            chat_template_kwargs={"enable_thinking": False})

        async def consume():
            async for _ in svc.stream_chat_completion(request):
                pass

        task = asyncio.create_task(consume())
        await started(fake)
        left = time.monotonic()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        record = fake.records[0]
        assert record["grammar"], "the forced run's first call was not the grammar phase"
        assert_stopped_promptly(record, left)
        served = await asyncio.wait_for(svc.create_chat_completion(chat(max_tokens=10)), 5)
        assert served.choices[0].message.content == "after"
        assert fake.max_busy == 1, "the slot was released while the grammar phase decoded"

    @pytest.mark.parametrize("stream", [False, True], ids=["non-streaming", "streaming"])
    async def test_a_cancel_during_prefill_holds_the_slot_until_the_thread_leaves(self, stream):
        """A prompt prefill cannot be interrupted (the criteria run after a token is sampled).
        A request cancelled in one must keep the slot until its thread is out of llama.cpp, and
        then generate nothing more; the next request must never decode beside it."""
        fake = ForeverLlama(script=["forever", "next"], prefill_s=0.4)
        svc = gguf_service(fake)

        async def consume():
            if stream:
                async for _ in svc.stream_chat_completion(chat(stream=True)):
                    pass
            else:
                await svc.create_chat_completion(chat())

        task = asyncio.create_task(consume())
        assert await asyncio.to_thread(fake.in_prefill.wait, 5)
        task.cancel()
        nxt = asyncio.create_task(svc.create_chat_completion(chat(max_tokens=10)))
        with pytest.raises(asyncio.CancelledError):
            await task
        served = await asyncio.wait_for(nxt, 5)
        assert served.choices[0].message.content == "next"
        assert fake.max_busy == 1, "the next request decoded while the cancelled one prefilled"
        assert fake.records[0]["tokens"] <= 1, "the cancelled request generated after its prefill"

    async def test_a_stream_closed_at_a_yield_stops_generation(self):
        """The other disconnect shape: the generator closed (GeneratorExit) between chunks."""
        fake = ForeverLlama()
        svc = gguf_service(fake)
        gen = svc.stream_chat_completion(chat(stream=True))
        await gen.__anext__()  # the role chunk
        await gen.__anext__()  # a content chunk: generation is under way
        await gen.aclose()
        record = fake.records[0]
        assert record["closed"] and not record["capped"]
        tokens = record["tokens"]
        await asyncio.sleep(0.05)
        assert record["tokens"] == tokens, "sampling went on after the stream was closed"

    async def test_text_completion_stops(self):
        fake = ForeverLlama(script=["forever", "more"])
        svc = gguf_service(fake)
        client = Gone()
        task = asyncio.create_task(svc.create_text_completion(
            TextCompletionRequest(model="tiny", prompt=["one", "two"], max_tokens=100000),
            client=client.watch))
        left = await disconnect_when_started(fake, client)
        with pytest.raises(GenerationAbandonedError):
            await asyncio.wait_for(task, 5)
        assert len(fake.records) == 1, "the second prompt was generated after the abandonment"
        assert_stopped_promptly(fake.records[0], left)

    async def test_a_request_abandoned_while_queued_generates_nothing(self):
        fake = ForeverLlama(script=["forever"])
        svc = gguf_service(fake)
        blocker = Gone()
        first = asyncio.create_task(svc.create_chat_completion(chat(), client=blocker.watch))
        await started(fake)
        queued = Gone()
        second = asyncio.create_task(svc.create_chat_completion(chat(), client=queued.watch))
        await asyncio.sleep(0.02)
        queued.leave()
        await asyncio.sleep(0.02)
        blocker.leave()
        for task in (first, second):
            with pytest.raises(GenerationAbandonedError):
                await asyncio.wait_for(task, 5)
        assert len(fake.records) == 1, "the abandoned queued request still generated"


# =============================================================================
# The stop reaches every phase
# =============================================================================


async def _abandon(fake: ForeverLlama, coro_factory) -> dict:
    client = Gone()
    task = asyncio.create_task(coro_factory(client.watch))
    left = await disconnect_when_started(fake, client)
    with pytest.raises(GenerationAbandonedError):
        await asyncio.wait_for(task, 5)
    record = fake.records[-1]
    assert_stopped_promptly(record, left)
    return record


class TestEveryPhaseCarriesTheStop:
    async def test_the_reasoning_phase(self, monkeypatch):
        FakeLlamaCpp(monkeypatch)
        fake = ForeverLlama()
        svc = gguf_service(fake)
        record = await _abandon(fake, lambda w: svc.create_chat_completion(
            tool_chat(tool_choice="required"), client=w))
        assert record["stream"] and not record["grammar"]  # the free phase

    async def test_the_auto_answer_phase(self, monkeypatch):
        FakeLlamaCpp(monkeypatch)
        fake = ForeverLlama(script=["brief\n</think>"])
        svc = gguf_service(fake)
        record = await _abandon(fake, lambda w: svc.create_chat_completion(tool_chat(), client=w))
        assert record is fake.records[1] and not record["grammar"]

    async def test_the_call_grammar_phase(self, monkeypatch):
        FakeLlamaCpp(monkeypatch)
        fake = ForeverLlama()
        svc = gguf_service(fake)
        record = await _abandon(fake, lambda w: svc.create_chat_completion(
            tool_chat(tool_choice="required", chat_template_kwargs={"enable_thinking": False}),
            client=w))
        assert record["grammar"] and not record["stream"]

    async def test_the_document_grammar_phase(self, monkeypatch):
        FakeLlamaCpp(monkeypatch)
        fake = ForeverLlama()
        svc = gguf_service(fake)
        record = await _abandon(fake, lambda w: svc.create_chat_completion(
            chat(response_format=DOC_SCHEMA, chat_template_kwargs={"enable_thinking": False}),
            client=w))
        assert record["grammar"] and not record["stream"]

    async def test_the_continuation_raw_completion(self):
        fake = ForeverLlama()
        svc = gguf_service(fake)
        messages = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "Once"}]
        record = await _abandon(fake, lambda w: svc.create_chat_completion(
            chat(messages=messages), client=w))
        assert record["prompt"].endswith("Once")

    async def test_the_plain_chat_through_the_handlers_inner_call(self):
        fake = ForeverLlama()
        svc = gguf_service(fake)
        record = await _abandon(fake, lambda w: svc.create_chat_completion(chat(), client=w))
        assert record["prompt"] == "hi"
        assert "create_completion" not in vars(fake), "the carry was left installed"

    async def test_the_streamed_continuation(self):
        fake = ForeverLlama()
        svc = gguf_service(fake)
        messages = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "Once"}]
        client = Gone()

        async def consume():
            async for _ in svc.stream_chat_completion(chat(messages=messages, stream=True),
                                                      client=client.watch):
                pass

        task = asyncio.create_task(consume())
        left = await disconnect_when_started(fake, client)
        await asyncio.wait_for(task, 5)
        assert_stopped_promptly(fake.records[0], left)

    async def test_the_text_completion(self):
        fake = ForeverLlama()
        svc = gguf_service(fake)
        await _abandon(fake, lambda w: svc.create_text_completion(
            TextCompletionRequest(model="tiny", prompt="go", max_tokens=100000), client=w))


# =============================================================================
# max_time_s
# =============================================================================


class TestMaxTime:
    async def test_free_generation_ends_as_a_length_stop_with_usage(self):
        fake = ForeverLlama()
        svc = gguf_service(fake)
        with structlog.testing.capture_logs() as logs:
            t0 = time.monotonic()
            resp = await asyncio.wait_for(svc.create_chat_completion(chat(max_time_s=0.3)), 5)
            took = time.monotonic() - t0
        record = fake.records[0]
        assert not record["capped"] and record["stopped"]
        assert 0.25 < took < 1.5
        choice = resp.choices[0]
        assert choice.finish_reason == "length"
        assert resp.usage.completion_tokens == record["yielded"] > 0
        assert len(choice.message.content) == resp.usage.completion_tokens
        (event,) = [e for e in logs if e["event"] == "generation_stopped"]
        assert event["stopped_by"] == "time" and event["max_time_s"] == 0.3

    async def test_absent_means_no_limit(self):
        fake = ForeverLlama(script=["a" * 400])
        svc = gguf_service(fake)
        resp = await svc.create_chat_completion(chat())
        assert resp.choices[0].finish_reason == "stop" and len(resp.choices[0].message.content) == 400

    async def test_a_tool_run_cut_in_its_reasoning_still_returns_a_well_formed_call(
        self, monkeypatch,
    ):
        """The deadline arrives in the reasoning: the block is closed as the allowance closes one,
        and the call grammar phase still runs (to the grace), so the caller gets a call."""
        FakeLlamaCpp(monkeypatch)
        fake = ForeverLlama(script=["forever", XML_CALL])
        svc = gguf_service(fake)
        with structlog.testing.capture_logs() as logs:
            resp = await asyncio.wait_for(svc.create_chat_completion(
                tool_chat(tool_choice="required", max_time_s=1.0)), 5)
        reason, call = fake.records
        assert reason["stopped"] and not reason["grammar"]
        assert call["grammar"] and call["prompt"].startswith(reason["prompt"])
        assert "\n</think>\n\n<tool_call>" in call["prompt"], "the reasoning was not closed"
        choice = resp.choices[0]
        assert choice.finish_reason == "tool_calls"
        (tc,) = choice.message.tool_calls
        assert tc["function"]["name"] == "scale_tensor"
        assert json.loads(tc["function"]["arguments"]) == {"name": "weight", "factor": 2}
        line = next(e for e in logs if e["event"] == "llamacpp_tool_generation")
        assert line["stopped_by"] == "time" and line["reasoning_cut_short"] is True
        assert resp.usage.completion_tokens_details.reasoning_tokens > 0

    async def test_auto_mode_after_a_cut_reasoning_may_still_open_a_call(self, monkeypatch):
        FakeLlamaCpp(monkeypatch)
        fake = ForeverLlama(script=["forever", "Done.\n<tool_call>", XML_CALL])
        svc = gguf_service(fake)
        resp = await asyncio.wait_for(svc.create_chat_completion(tool_chat(max_time_s=1.0)), 5)
        assert resp.choices[0].finish_reason == "tool_calls"

    async def test_a_grammar_phase_past_the_grace_ends_length_with_no_call(self, monkeypatch):
        FakeLlamaCpp(monkeypatch)
        fake = ForeverLlama(script=["forever", "forever"])
        svc = gguf_service(fake)
        t0 = time.monotonic()
        resp = await asyncio.wait_for(svc.create_chat_completion(
            tool_chat(tool_choice="required", max_time_s=0.3)), 5)
        reason, call = fake.records
        assert call["grammar"] and call["stopped"] and not call["capped"]
        # The grace: min(30 s, 10 %) = 0.03 s past the deadline, never the whole budget.
        assert call["ended_at"] - t0 < 0.3 + 0.03 + 0.5
        assert call["ended_at"] - t0 >= 0.3 + 0.03
        choice = resp.choices[0]
        assert choice.finish_reason == "length" and not choice.message.tool_calls

    async def test_the_grammar_phase_gets_the_grace_not_the_bare_deadline(self, monkeypatch):
        """A call phase that starts AFTER the deadline (the reasoning ran up to it) must not be
        stopped at its first token: it runs to the grace."""
        FakeLlamaCpp(monkeypatch)
        fake = ForeverLlama(script=["forever", XML_CALL])
        svc = gguf_service(fake)
        resp = await asyncio.wait_for(svc.create_chat_completion(
            tool_chat(tool_choice="required", max_time_s=1.0)), 10)
        # XML_CALL (120 characters at TOKEN_S / 10) takes ~0.03 s: inside the 0.1 s grace (10 % of
        # 1 s), and past the bare deadline from its first token.
        assert fake.records[1]["tokens"] == len(XML_CALL)
        assert resp.choices[0].finish_reason == "tool_calls"

    async def test_streaming_ends_as_a_length_stop(self):
        fake = ForeverLlama()
        svc = gguf_service(fake)
        t0 = time.monotonic()
        frames = [f async for f in svc.stream_chat_completion(chat(stream=True, max_time_s=0.3))]
        # Stopped by the deadline, not by the stand-in's cap (which also ends "length").
        assert fake.records[0]["stopped"] and not fake.records[0]["capped"]
        assert time.monotonic() - t0 < 1.5
        final = json.loads(next(f for f in reversed(frames) if '"finish_reason":"' in f)
                           .removeprefix("data: "))
        assert final["choices"][0]["finish_reason"] == "length"
        assert frames[-1] == "data: [DONE]\n\n"

    async def test_text_completion_ends_as_a_length_stop(self):
        fake = ForeverLlama()
        svc = gguf_service(fake)
        resp = await asyncio.wait_for(svc.create_text_completion(TextCompletionRequest(
            model="tiny", prompt="go", max_tokens=100000, max_time_s=0.3)), 5)
        assert resp.choices[0].finish_reason == "length"
        assert resp.usage.completion_tokens == fake.records[0]["yielded"] > 0

    async def test_the_grace_is_min_30s_or_ten_percent(self):
        assert gs.grace_s(10) == pytest.approx(1.0)
        assert gs.grace_s(3600) == 30.0


class TestMaxTimePolicy:
    @pytest.mark.parametrize("endpoint", [Endpoint.CHAT, Endpoint.COMPLETIONS])
    def test_honoured_on_llamacpp_refused_on_transformers(self, endpoint):
        cells = OUTPUT_CHANGING["max_time_s"]
        assert isinstance(cells[(endpoint, Engine.LLAMACPP)], Honoured)
        request = (chat(max_time_s=5) if endpoint is Endpoint.CHAT else
                   TextCompletionRequest(model="tiny", prompt="x", max_time_s=5))
        assert evaluate(request, endpoint, Engine.LLAMACPP, strict=True).unused == []
        with pytest.raises(FieldNotHonouredError) as exc:
            evaluate(request, endpoint, Engine.TRANSFORMERS, strict=False)
        assert exc.value.details["param"] == "max_time_s"

    @pytest.mark.parametrize("value", [0, -1, 3601, True, "soon"])
    def test_out_of_bounds_is_refused_by_the_schema(self, value):
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            chat(max_time_s=value)

    async def test_a_direct_caller_on_transformers_is_refused(self):
        from tests.unit.f25_fixtures import word_model, word_tokenizer

        svc = make_service(word_model(), word_tokenizer())
        with pytest.raises(FieldNotHonouredError):
            await svc.create_chat_completion(chat(max_time_s=5, max_tokens=2))


# =============================================================================
# The route: the watch is built from the request
# =============================================================================


class _Asgi:
    """Drive the app at the ASGI level with a receive that can disconnect."""

    def __init__(self, body: dict, spec: str = "2.3"):
        self.body = json.dumps(body).encode()
        self.gone = asyncio.Event()
        self.done = asyncio.Event()
        self.sent: list[dict] = []
        self._delivered = False
        self.spec = spec

    async def receive(self) -> dict:
        if not self._delivered:
            self._delivered = True
            return {"type": "http.request", "body": self.body, "more_body": False}
        gone = asyncio.ensure_future(self.gone.wait())
        done = asyncio.ensure_future(self.done.wait())
        await asyncio.wait({gone, done}, return_when=asyncio.FIRST_COMPLETED)
        gone.cancel()
        done.cancel()
        return {"type": "http.disconnect"}

    async def send(self, message: dict) -> None:
        self.sent.append(message)
        if message["type"] == "http.response.body" and not message.get("more_body"):
            self.done.set()

    def scope(self, path: str) -> dict:
        return {"type": "http", "asgi": {"version": "3.0", "spec_version": self.spec},
                "http_version": "1.1", "method": "POST", "scheme": "http", "path": path,
                "raw_path": path.encode(), "root_path": "", "query_string": b"",
                "headers": [(b"host", b"testserver"), (b"content-type", b"application/json")],
                "client": ("127.0.0.1", 1), "server": ("testserver", 80), "state": {}}

    @property
    def status(self) -> int | None:
        start = next((m for m in self.sent if m["type"] == "http.response.start"), None)
        return start["status"] if start else None


class TestTheRouteWatchesTheClient:
    @pytest.mark.parametrize("path, body, spec", [
        ("/v1/chat/completions", {"model": "tiny", "messages": [{"role": "user",
                                                                 "content": "hi"}],
                                  "max_tokens": 100000}, "2.3"),
        ("/v1/chat/completions", {"model": "tiny", "messages": [{"role": "user",
                                                                 "content": "hi"}],
                                  "max_tokens": 100000, "stream": True}, "2.3"),
        ("/v1/chat/completions", {"model": "tiny", "messages": [{"role": "user",
                                                                 "content": "hi"}],
                                  "max_tokens": 100000, "stream": True}, "2.4"),
        ("/v1/completions", {"model": "tiny", "prompt": "go", "max_tokens": 100000}, "2.3"),
    ], ids=["chat", "chat-stream-asgi2.3", "chat-stream-asgi2.4", "completions"])
    async def test_a_disconnect_through_the_route_stops_generation(self, path, body, spec):
        fake = ForeverLlama(script=["forever", "served"])
        svc = gguf_service(fake)
        client, _ = make_client(svc, model_row(gguf_files=["m.gguf"]))
        app = client.app
        conn = _Asgi(body, spec)
        task = asyncio.create_task(app(conn.scope(path), conn.receive, conn.send))
        await started(fake)
        left = time.monotonic()
        conn.gone.set()
        await asyncio.wait_for(task, 5)
        assert_stopped_promptly(fake.records[0], left)
        # The slot is free: the next request through the route is served.
        nxt = _Asgi({**body, "max_tokens": 20, "stream": False})
        await asyncio.wait_for(app(nxt.scope(path), nxt.receive, nxt.send), 5)
        assert nxt.status == 200
        assert b"served" in b"".join(m.get("body", b"") for m in nxt.sent)

    async def test_client_watch_from_request_returns_on_disconnect_only(self):
        messages = [{"type": "http.request", "body": b"", "more_body": False},
                    {"type": "http.disconnect"}]

        class _Req:
            async def receive(self):
                return messages.pop(0)

        await asyncio.wait_for(gs.ClientWatch.from_request(_Req()).wait(), 1)
        assert messages == []

    async def test_the_watcher_is_cancelled_on_every_exit(self):
        stop = gs.GenerationStop()
        client = Gone()
        with pytest.raises(RuntimeError):
            async with gs.watch_client(stop, client.watch) as task:
                raise RuntimeError("boom")
        await asyncio.sleep(0)
        assert task.cancelled()
        async with gs.watch_client(stop, client.watch) as task:
            pass
        await asyncio.sleep(0)
        assert task.cancelled() and not stop.abandoned


# =============================================================================
# The transformers serial path (the same defect, fixed the same way)
# =============================================================================


class TestTransformersSerial:
    async def test_a_watched_generate_carries_the_abandonment(self):
        from tests.unit.f25_fixtures import word_model, word_tokenizer

        svc = make_service(word_model(), word_tokenizer())
        client = Gone()
        seen: list[dict] = []
        real = svc._generate_sync
        loop = asyncio.get_running_loop()

        def spy(kwargs, *a, **k):
            seen.append(kwargs)
            loop.call_soon_threadsafe(client.leave)  # the client goes while generate() runs
            # The criteria the request carries now answer "stop".
            time.sleep(0.05)
            return real(kwargs, *a, **k)

        svc._generate_sync = spy
        with pytest.raises(GenerationAbandonedError):
            await svc.create_chat_completion(chat(max_tokens=4), client=client.watch)
        (kwargs,) = seen
        criteria = kwargs["stopping_criteria"]
        assert any(c(None, None) for c in criteria), "generate() would not stop on abandonment"

    async def test_a_watched_text_completion_stops_and_raises(self):
        from tests.unit.f25_fixtures import word_model, word_tokenizer

        svc = make_service(word_model(), word_tokenizer())
        client = Gone()
        seen: list[dict] = []
        real = svc._generate_sync
        loop = asyncio.get_running_loop()

        def spy(kwargs, *a, **k):
            seen.append(kwargs)
            loop.call_soon_threadsafe(client.leave)
            time.sleep(0.05)
            return real(kwargs, *a, **k)

        svc._generate_sync = spy
        with pytest.raises(GenerationAbandonedError):
            await svc.create_text_completion(
                TextCompletionRequest(model="tiny", prompt=["w1", "w2"], max_tokens=4),
                client=client.watch)
        assert len(seen) == 1, "the second prompt was generated after the abandonment"
        assert any(c(None, None) for c in seen[0]["stopping_criteria"])

    async def test_unwatched_kwargs_are_unchanged(self):
        from tests.unit.f25_fixtures import word_model, word_tokenizer

        svc = make_service(word_model(), word_tokenizer())
        seen: list[dict] = []
        real = svc._generate_sync
        svc._generate_sync = lambda kwargs, *a, **k: seen.append(kwargs) or real(kwargs, *a, **k)
        await svc.create_chat_completion(chat(max_tokens=2))
        assert "stopping_criteria" not in seen[0]


# =============================================================================
# The pieces, directly
# =============================================================================


class _Clock:
    def __init__(self) -> None:
        self.t = 100.0

    def __call__(self) -> float:
        return self.t


class TestStopCriteria:
    def test_the_post_loop_call_repeats_the_in_loop_answer(self):
        """llama-cpp-python calls the criteria once more after its loop with the SAME ids. A model
        that ended by itself just before the deadline must not be re-labelled by that call (and
        a stop sequence's trimmed text replaced by the untrimmed one)."""
        clock = _Clock()
        stop = gs.GenerationStop(1.0, clock=clock)
        criteria = stop.criteria()
        ids = [1, 2, 3]
        assert criteria(ids, None) is False  # the last in-loop call: before the deadline
        clock.t += 5  # the deadline passes before the post-loop call
        assert criteria(ids, None) is False, "the post-loop call re-labelled a finished call"
        assert criteria(ids + [4], None) is True and criteria.fired == gs.TIME
        assert stop.tokens == 2 and stop.stopped_by == gs.TIME

    def test_abandonment_is_never_latched(self):
        stop = gs.GenerationStop()
        criteria = stop.criteria()
        assert criteria([1], None) is False
        stop.abandon()
        assert criteria([1], None) is True and criteria.fired == gs.ABANDONED

    def test_after_the_deadline_every_later_phase_gets_the_grace(self):
        clock = _Clock()
        stop = gs.GenerationStop(10.0, clock=clock)
        assert stop.criteria()._deadline == 110.0
        assert stop.criteria(grammar=True)._deadline == 111.0
        stop.note_time()
        assert stop.criteria()._deadline == 111.0


class TestCompletionsCarry:
    def test_merges_with_the_handlers_own_criteria_and_restores(self):
        seen: list[Any] = []

        class _Llama:
            def create_completion(self, **kw):
                seen.append(kw["stopping_criteria"])
                return "ok"

        llama = _Llama()
        stop = gs.GenerationStop()
        ours = stop.criteria()
        template_stop = lambda ids, logits: ids[-1] == 7  # noqa: E731
        with gs.completions_carry(llama, ours):
            assert llama.create_completion(stopping_criteria=None) == "ok"
            assert llama.create_completion(stopping_criteria=[template_stop]) == "ok"
        assert seen[0] is ours
        assert isinstance(seen[1], gs.AnyOf) and seen[1][0] is template_stop and seen[1][1] is ours
        assert seen[1]([1, 7], None) is True  # the template's own stop still ends it
        assert "create_completion" not in vars(llama)

    def test_restored_when_the_call_raises(self):
        class _Llama:
            def create_completion(self, **kw):
                raise RuntimeError("decode failed")

        llama = _Llama()
        with pytest.raises(RuntimeError), gs.completions_carry(llama, gs.GenerationStop().criteria()):
            llama.create_completion(stopping_criteria=None)
        assert "create_completion" not in vars(llama)


class TestMaxTimeOnADocument:
    async def test_reasoning_cut_by_the_deadline_still_yields_the_document(self, monkeypatch):
        """Structured output that thinks first: the deadline closes the block and the document
        is generated under its grammar within the grace, validated, finish "stop"."""
        FakeLlamaCpp(monkeypatch)
        fake = ForeverLlama(script=["forever", '{"a": "b"}'])
        svc = gguf_service(fake)
        resp = await asyncio.wait_for(svc.create_chat_completion(
            chat(response_format=DOC_SCHEMA, max_time_s=1.0)), 5)
        reason, doc = fake.records
        assert reason["stopped"] and not reason["grammar"] and doc["grammar"]
        assert "\n</think>\n\n" in doc["prompt"]
        choice = resp.choices[0]
        assert choice.finish_reason == "stop" and json.loads(choice.message.content) == {"a": "b"}
        assert choice.message.reasoning_content
