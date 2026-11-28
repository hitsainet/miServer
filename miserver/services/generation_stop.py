"""Stopping a generation whose client has gone, or whose time is up (2026-11-22).

THE DEFECT IT REMOVES (production, 2026-11-21 23:27 UTC). A non-streaming GGUF chat request
(Qwen3.6-27B-GGUF, tools, a two-phase qwen3_xml tool run, `max_tokens` 32000) was abandoned by its
client (an OpenAI SDK with a 600 s timeout) at 23:37:49. miServer generated on until 00:13:31, all
32,000 tokens, holding the model slot for 46 minutes; two later requests queued behind it and their
callers timed out at 900 s. Every llama.cpp call ran in `asyncio.to_thread` with nothing that could
stop it, and nothing watched the connection.

WHAT THIS MODULE GIVES A REQUEST:

* `GenerationStop`: one per request. It is set ABANDONED when the client disconnects (the watcher
  below) or the request's task is cancelled, and it carries the request's optional `max_time_s`
  DEADLINE, measured from when the request entered the service (its queueing included, model
  loads excluded).
* `GenerationStop.criteria(...)`: a callable `(input_ids, logits) -> bool` for ONE llama.cpp
  generation call, passed as `stopping_criteria=`. llama-cpp-python 0.3.36 checks it after every
  sampled token (`Llama.generate`), so a set stop ends that call within one token. ⚠
  `Llama.create_chat_completion` does NOT take `stopping_criteria` (a TypeError); its chat handler
  calls `llama.create_completion` itself, so `completions_carry` installs the criterion on that
  inner call for the duration of the outer one.
* `watch_client`: a task that waits for the ASGI `http.disconnect` and sets the stop. Cancelled on
  every exit path.
* `run_in_thread`: `asyncio.to_thread` that, when its awaiting task is cancelled, sets the stop and
  WAITS (bounded) for the thread to leave llama.cpp before re-raising, so the model slot is never
  released while a decode is still running on the single, non-thread-safe `Llama` context.

The DEADLINE has two values. Free generation (a reasoning block, an `auto` answer, an ordinary
reply, a text completion) stops at `max_time_s` and ends as a LENGTH stop. A grammar-constrained
phase (a tool call, a structured-output document), and every phase after the deadline was reached
once, may run to the GRACE deadline, `max_time_s + min(30 s, 10 % of max_time_s)`, so that a tool
run whose reasoning the deadline cut still returns a well-formed call.
"""

from __future__ import annotations

import asyncio
import contextlib
import contextvars
import threading
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator
from typing import Any, TypeVar

from miserver.core.logging import get_logger

logger = get_logger(__name__)

_T = TypeVar("_T")

#: The longest grace a grammar-constrained phase gets past `max_time_s`, and its share of it.
GRACE_MAX_S = 30.0
GRACE_SHARE = 0.10
#: The most `max_time_s` may be (the schema's bound).
MAX_TIME_S_LIMIT = 3600.0
#: How long a cancelled request waits for its generation thread to leave llama.cpp before the slot
#: is released anyway (a set stop ends a call within one token; this bounds a stuck decode).
DRAIN_TIMEOUT_S = 30.0

ABANDONED = "abandoned"
TIME = "time"


def grace_s(max_time_s: float) -> float:
    """The grace a grammar-constrained phase gets past the deadline: min(30 s, 10 %)."""
    return min(GRACE_MAX_S, GRACE_SHARE * max_time_s)


class GenerationAbandoned(Exception):  # noqa: N818 - a control-flow signal, not an error
    """Raised in the generation thread when the stop was abandoned, so no later phase starts."""


class StopCriteria:
    """The stop of ONE llama.cpp generation call (`stopping_criteria=`).

    True once the request is abandoned, or once the call's deadline has passed. The deadline is
    latched per sequence length: llama-cpp-python calls the criterion again AFTER its loop with
    the same `input_ids` (`_create_completion`), and that call must repeat the in-loop answer,
    or a call the model ended exactly at the deadline would be re-labelled (and a stop sequence's
    trimmed text replaced by the untrimmed one).
    """

    def __init__(self, stop: GenerationStop, deadline: float | None) -> None:
        self._stop = stop
        self._deadline = deadline
        self._last_len = -1
        self._last = False
        #: Why this call was stopped: ABANDONED, TIME, or None (it ended by itself).
        self.fired: str | None = None
        #: Tokens this call sampled (one criterion call per sampled token).
        self.tokens = 0

    def __call__(self, input_ids: Any, logits: Any = None) -> bool:
        stop = self._stop
        if stop.abandoned:
            self.fired = self.fired or ABANDONED
            return True
        try:
            n = len(input_ids)
        except TypeError:
            n = -2
        if n == self._last_len:
            return self._last
        self._last_len = n
        self.tokens += 1
        stop.tokens += 1
        if self._deadline is not None and stop.clock() >= self._deadline:
            self.fired = TIME
            stop.note_time()
            self._last = True
        else:
            self._last = False
        return self._last


class GenerationStop:
    """One request's stop: its client's disconnect and its optional `max_time_s` deadline."""

    def __init__(self, max_time_s: float | None = None, *,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self.clock = clock
        self.started = clock()
        self.max_time_s = float(max_time_s) if max_time_s else None
        self.deadline = (self.started + self.max_time_s) if self.max_time_s else None
        self.grace_deadline = (self.deadline + grace_s(self.max_time_s)
                               if self.deadline is not None and self.max_time_s else None)
        self._abandoned = threading.Event()
        #: Why it was abandoned ("client_disconnected", "cancelled"), or None.
        self.abandon_reason: str | None = None
        #: TIME once a deadline stopped any phase of this request.
        self.stopped_by: str | None = None
        #: Tokens sampled so far across every call of the request.
        self.tokens = 0
        self._logged = False

    # -- the two signals ---------------------------------------------------------------------

    def abandon(self, reason: str = "client_disconnected") -> None:
        if not self._abandoned.is_set():
            self.abandon_reason = reason
            self._abandoned.set()

    @property
    def abandoned(self) -> bool:
        return self._abandoned.is_set()

    @property
    def event(self) -> threading.Event:
        """The abandonment as a `threading.Event` (the transformers path's stop event)."""
        return self._abandoned

    def note_time(self) -> None:
        self.stopped_by = TIME

    @property
    def time_reached(self) -> bool:
        return self.stopped_by == TIME

    # -- what a generation call carries -------------------------------------------------------

    def criteria(self, *, grammar: bool = False) -> StopCriteria:
        """The criterion for one call. Free generation stops at the deadline; a grammar phase,
        and any phase after the deadline was reached once, at the grace deadline."""
        if self.deadline is None:
            deadline = None
        elif grammar or self.time_reached:
            deadline = self.grace_deadline
        else:
            deadline = self.deadline
        return StopCriteria(self, deadline)

    def raise_if_abandoned(self) -> None:
        if self.abandoned:
            raise GenerationAbandoned(self.abandon_reason or "abandoned")

    def elapsed_s(self) -> float:
        return round(self.clock() - self.started, 3)

    # -- the log lines ------------------------------------------------------------------------

    def log_abandoned(self, **fields: Any) -> None:
        """`generation_abandoned`, once per request."""
        if self._logged:
            return
        self._logged = True
        logger.warning(
            "generation_abandoned",
            reason=self.abandon_reason,
            tokens_generated=self.tokens,
            elapsed_s=self.elapsed_s(),
            **fields,
        )

    def log_time_stop(self, **fields: Any) -> None:
        """`generation_stopped` with `stopped_by: "time"`, when the deadline cut a phase."""
        if self.stopped_by != TIME:
            return
        logger.info(
            "generation_stopped",
            stopped_by=TIME,
            max_time_s=self.max_time_s,
            tokens_generated=self.tokens,
            elapsed_s=self.elapsed_s(),
            **fields,
        )


class AnyOf(list):  # type: ignore[type-arg]
    """llama-cpp-python's `StoppingCriteriaList` without importing it: true if any member is."""

    def __call__(self, input_ids: Any, logits: Any) -> bool:
        # A list, as llama-cpp-python's: every member is called (each counts its tokens).
        return any([c(input_ids, logits) for c in self])  # noqa: C419


@contextlib.contextmanager
def completions_carry(llama: Any, criteria: StopCriteria) -> Iterator[None]:
    """For the duration of the block, every `llama.create_completion` call carries `criteria`
    (merged with any criteria the caller passes).

    For `Llama.create_chat_completion`, which takes no `stopping_criteria` in 0.3.36: its chat
    handler calls `llama.create_completion(..., stopping_criteria=<the template's stop ids>)`, an
    attribute lookup on the instance, so an instance attribute set here is what it calls. The model
    slot is held, so nothing else uses this `Llama` meanwhile; the attribute is removed on exit,
    even when the call raises. A streamed completion binds the criteria when the call is made, so
    the block need only span the call, not the stream."""
    had_own = "create_completion" in getattr(llama, "__dict__", {})
    inner = getattr(llama, "create_completion", None)
    if inner is None:  # a stand-in with no inner call: nothing to carry the criteria on
        yield
        return

    def create_completion(*args: Any, **kwargs: Any) -> Any:
        theirs = kwargs.get("stopping_criteria")
        kwargs["stopping_criteria"] = (criteria if theirs is None
                                       else AnyOf([*(theirs if isinstance(theirs, list)
                                                     else [theirs]), criteria]))
        return inner(*args, **kwargs)

    llama.create_completion = create_completion
    try:
        yield
    finally:
        if had_own:
            llama.create_completion = inner
        else:
            # Off the instance dict, never `del` (a test double records a deletion as permanent).
            vars(llama).pop("create_completion", None)


# =============================================================================
# The client watch
# =============================================================================


class ClientWatch:
    """Something that resolves when the HTTP client has gone: `wait()` returns on disconnect."""

    def __init__(self, wait: Callable[[], Awaitable[None]]) -> None:
        self.wait = wait

    @classmethod
    def from_request(cls, request: Any) -> ClientWatch:
        """Wait on the request's ASGI `receive` for `http.disconnect`. The JSON body has been read
        by the time a route runs, so the server's next message is the disconnect (uvicorn answers
        `http.disconnect` once the connection is lost; it blocks until then)."""
        receive = request.receive

        async def wait() -> None:
            while True:
                message = await receive()
                if message.get("type") == "http.disconnect":
                    return

        return cls(wait)


@contextlib.asynccontextmanager
async def watch_client(stop: GenerationStop,
                       client: ClientWatch | None) -> AsyncIterator[asyncio.Task[None] | None]:
    """Set `stop` abandoned when `client` disconnects, while the block runs. The watcher task is
    cancelled on EVERY exit path (return, exception, cancellation). With no client (a direct
    caller, the Batch API) nothing is watched and the stop is never abandoned by a disconnect."""
    if client is None:
        yield None
        return

    async def _watch() -> None:
        try:
            await client.wait()
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 - a broken receive must never fail the request
            logger.warning("client_watch_failed", exc_info=True)
            return
        stop.abandon("client_disconnected")

    task = asyncio.get_running_loop().create_task(_watch())
    try:
        yield task
    finally:
        task.cancel()


# =============================================================================
# Running a generation off the loop
# =============================================================================


async def _drain(done: threading.Event, timeout: float) -> bool:
    """Wait for `done` through a cancellation (a shielded scope, bounded)."""
    if done.is_set():
        return True
    import anyio

    with anyio.CancelScope(shield=True), anyio.move_on_after(timeout):
        while not done.is_set():
            await asyncio.sleep(0.005)
    return done.is_set()


class InFlight:
    """Tracks whether a generation thread is inside llama.cpp, so an exit can wait for it."""

    def __init__(self) -> None:
        self.idle = threading.Event()
        self.idle.set()

    def wrap(self, fn: Callable[[], _T]) -> Callable[[], _T]:
        def run() -> _T:
            self.idle.clear()
            try:
                return fn()
            finally:
                self.idle.set()

        return run

    async def drain(self, timeout: float = DRAIN_TIMEOUT_S) -> bool:
        drained = await _drain(self.idle, timeout)
        if not drained:
            logger.error("generation_thread_still_running", waited_s=timeout,
                         detail="the slot is released with a decode still in flight")
        return drained


async def run_in_thread(stop: GenerationStop, fn: Callable[..., _T], *args: Any) -> _T:
    """`asyncio.to_thread(fn, *args)` (the context copied, as to_thread copies it), except that a
    cancellation of the awaiting task sets `stop` abandoned and waits for the thread to leave
    llama.cpp before re-raising: the caller's `finally` (the slot release) never runs while a
    decode is still in flight."""
    loop = asyncio.get_running_loop()
    ctx = contextvars.copy_context()
    inflight = InFlight()
    call = inflight.wrap(lambda: ctx.run(fn, *args))
    future = loop.run_in_executor(None, call)
    try:
        return await future
    except asyncio.CancelledError:
        stop.abandon("cancelled")
        await inflight.drain()
        raise


_END = object()


class BlockingPull:
    """Async iteration of a blocking iterator (a llama.cpp stream), one `next()` per executor
    call, with the in-flight call tracked: `aclose()` sets nothing itself, but waits for a pending
    `next()` to return before closing the iterator, so a stream is never closed (or the slot
    released) while another thread is still executing it."""

    def __init__(self, iterable: Any) -> None:
        self._source = iterable
        self._it = iter(iterable)
        self._inflight = InFlight()
        self._ctx = contextvars.copy_context()

    def __aiter__(self) -> BlockingPull:
        return self

    async def __anext__(self) -> Any:
        loop = asyncio.get_running_loop()
        item = await loop.run_in_executor(
            None, self._inflight.wrap(lambda: self._ctx.run(next, self._it, _END))
        )
        if item is _END:
            raise StopAsyncIteration
        return item

    async def aclose(self) -> None:
        """Wait (bounded, through a cancellation) for an in-flight `next()`, then close."""
        drained = await self._inflight.drain()
        # The object handed in (llama.cpp's generator), whose iterator may be another object.
        close = getattr(self._source, "close", None) or getattr(self._it, "close", None)
        if drained and callable(close):
            try:
                close()
            except Exception:  # noqa: BLE001
                logger.warning("llamacpp_stream_close_failed", exc_info=True)
