"""A GGUF window's decode check, run in a short-lived child process (2026-11-21).

WHY A CHILD. Model 17 (Qwen3.6-27B Q4_K_M, q8_0 cache) was loaded with `context_ceiling: "max"`
on the RTX 3090 on 2026-11-21. The upward probe reached 222,208 tokens because a context could be
CREATED there; the load used 24,040 MiB of the ~24,080 MiB that were free. Two short requests
served; the third, a longer prompt, aborted the whole process inside `llama_decode`
(`ggml-cuda.cu:109: CUDA error`, exit 139), and the model was lost with the pod.

Creating a context is not the same as being able to decode in it. Read from the llama.cpp
sources llama-cpp-python 0.3.36 vendors (`vendor/llama.cpp`, commit 0c1e5709):

* `llama_context::sched_reserve` RESERVES the compute buffers at creation, for the worst case:
  `n_ubatch` tokens against the FULL memory module (`memory->init_full()`). With a quantized cache
  under flash attention, that reservation includes an f16 copy of one layer's whole K and V
  (`ggml_cuda_flash_attn_ext_get_alloc_size`), so it grows with the window.
* What is NOT reserved is allocated while decoding, and an allocation failure there is fatal:
  ggml-cuda's per-device pool (`ggml_cuda_pool_vmm::alloc` grows with `CU_CHECK(cuMemCreate)`, the
  legacy pool with `CUDA_CHECK(cudaMalloc)`) holds every op's temporaries (the q8_1 copy of the
  activations each quantized matmul makes, MMQ's stream-k fix-up, flash attention's
  parallel-block scratch); cuBLAS creates its handle and workspace on first use; CUDA loads
  kernels lazily on their first launch. Each of those failures goes through `ggml_cuda_error`,
  which ends in `GGML_ABORT` (ggml-cuda.cu line 109 in that tree, the line in the crash): the
  process aborts. There is no error to catch, in Python or anywhere else in the process.

So the only place a decode can be TRIED without risking the server is another process. The child
builds the same `Llama(**kwargs)` the loader would, decodes one full `n_batch` of throwaway tokens
(the prompt-processing graph at its largest) and one more token (the generation graph), says so
on stdout, and waits while the parent reads the card's free memory, which now includes what the
decode allocated. The parent then tells it to exit. A child that aborts, segfaults or hangs costs
one probe, never the server: its exit code and the tail of its stderr are recorded.

THIS FILE IS ALSO THE CHILD'S SCRIPT (`python -I gguf_decode_check.py`): it imports nothing from
miserver and nothing outside the standard library at module level, so it runs isolated (`-I`: no
working directory, no PYTHONPATH, no user site on `sys.path`) with only `llama_cpp` imported, in
the child alone.
"""

from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, TextIO

#: Every line the child writes for the parent starts with this; anything else on stdout is noise.
MARKER = "MISERVER_GGUF_DECODE_CHECK "

#: The whole check: a full model load in the child, the decode, the measurement. Model 17's 16 GiB
#: construct in seconds from the page cache; a cold read from disk takes longer.
DEFAULT_TIMEOUT_S = 900.0

#: How long the parent waits after the child exits for the card's memory to come back.
_SETTLE_S = 15.0
#: A card counts as recovered when it is within this of what it had free before the child.
_SETTLE_SLACK_MB = 256

#: The text the throwaway prompt is made of. Its content does not matter; its length does.
_FILLER = (
    "The decode check fills one batch with ordinary text so the graph llama.cpp runs is the "
    "largest a prompt will make. "
)


@dataclass
class DecodeCheckResult:
    """What one child reported. `ok` is the loader's judgement and is set by it."""

    n_ctx: int
    tokens: int = 0
    #: The child created the context.
    created: bool = False
    #: The child decoded a full batch and one more token, and said so.
    decode_ok: bool = False
    #: The child's exit code: negative for a signal (subprocess's convention), None if it was
    #: still running when the parent gave up on it.
    exit_code: int | None = None
    #: Free memory per card (torch index) while the child held the decoded context.
    free_mb: dict[int, int] = field(default_factory=dict)
    error: str | None = None
    seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_ctx": self.n_ctx,
            "tokens": self.tokens,
            "created": self.created,
            "decode_ok": self.decode_ok,
            "exit_code": self.exit_code,
            "free_mb": {f"cuda:{index}": mb for index, mb in sorted(self.free_mb.items())},
            "error": self.error,
            "seconds": round(self.seconds, 1),
        }


def child_command() -> list[str]:
    """The child: this interpreter, isolated, running this file."""
    return [sys.executable, "-I", os.path.abspath(__file__)]


def _tail(handle: Any, limit: int = 2000) -> str:
    try:
        handle.seek(0, os.SEEK_END)
        size = handle.tell()
        handle.seek(max(0, size - limit))
        return str(handle.read().decode("utf-8", "replace")).strip()
    except Exception:  # noqa: BLE001 - diagnostics only
        return ""


def _reap(proc: subprocess.Popen[str], grace_s: float) -> int | None:
    """The child's exit code, after `grace_s` to exit by itself and then a kill."""
    if grace_s > 0:
        try:
            return proc.wait(timeout=grace_s)
        except subprocess.TimeoutExpired:
            pass
    proc.kill()
    try:
        return proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        return None


def _settle(measure: Callable[[], dict[int, int]], before: dict[int, int]) -> None:
    """Wait (briefly) until the child's memory is back on every card it was measured on."""
    if not before:
        return
    deadline = time.monotonic() + _SETTLE_S
    while time.monotonic() < deadline:
        try:
            now = measure()
        except Exception:  # noqa: BLE001 - a reading, never a gate
            return
        if all(now.get(index, 0) >= mb - _SETTLE_SLACK_MB for index, mb in before.items()):
            return
        time.sleep(0.25)


def run_decode_check(
    kwargs: dict[str, Any],
    n_tokens: int,
    measure: Callable[[], dict[int, int]],
    *,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    command: list[str] | None = None,
) -> DecodeCheckResult:
    """Construct `Llama(**kwargs)` in a child, decode `n_tokens` and one more, measure, exit.

    `measure` reads free memory per card (the loader passes `reported_free_mb_by_index` for the
    model's cards); it is called while the child still holds the decoded context, and again after
    it exits until the memory is back. Never raises: a child that cannot start, crashes or times
    out is a failed check, with the reason in `error`.
    """
    started = time.monotonic()
    result = DecodeCheckResult(n_ctx=int(kwargs.get("n_ctx", 0)))
    try:
        before = measure()
    except Exception:  # noqa: BLE001
        before = {}
    stderr = tempfile.TemporaryFile()
    try:
        proc = subprocess.Popen(
            command or child_command(),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=stderr,
            text=True,
            env=os.environ.copy(),
        )
    except OSError as exc:
        stderr.close()
        result.error = f"the decode-check process could not start: {exc}"
        result.seconds = time.monotonic() - started
        return result

    lines: queue.Queue[str | None] = queue.Queue()

    def _read() -> None:
        assert proc.stdout is not None
        for line in proc.stdout:
            lines.put(line)
        lines.put(None)

    reader = threading.Thread(target=_read, name="gguf-decode-check-reader", daemon=True)
    reader.start()
    report: dict[str, Any] | None = None
    closed = False
    try:
        assert proc.stdin is not None
        proc.stdin.write(json.dumps({"kwargs": kwargs, "n_tokens": int(n_tokens)}) + "\n")
        proc.stdin.flush()
        deadline = started + timeout_s
        while report is None:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                result.error = f"the decode check did not finish within {timeout_s:.0f} s"
                break
            try:
                line = lines.get(timeout=remaining)
            except queue.Empty:
                continue
            if line is None:
                closed = True  # stdout closed: the child is ending
                break
            if line.startswith(MARKER):
                try:
                    report = json.loads(line[len(MARKER):])
                except ValueError:
                    report = {"stage": "garbled", "error": line[:300]}
        if report is not None:
            result.created = bool(report.get("created"))
            result.decode_ok = bool(report.get("decode_ok"))
            result.tokens = int(report.get("tokens") or 0)
            if report.get("error"):
                result.error = str(report["error"])[:500]
            if result.decode_ok:
                # The child holds the decoded context now: what is free IS what serving would
                # have, with whatever the decode allocated already taken.
                try:
                    result.free_mb = dict(measure())
                except Exception as exc:  # noqa: BLE001
                    result.error = f"free memory could not be read: {exc}"[:500]
        if report is not None:
            # Tell it to go; a child that already died has nothing to read.
            try:
                proc.stdin.write("exit\n")
                proc.stdin.flush()
            except (BrokenPipeError, OSError, ValueError):
                pass
        # A child that reported, or closed its output, is let exit by itself (its exit code is
        # the evidence: 139, -6); one that timed out holding the card is killed at once.
        grace = 60 if report is not None else (10 if closed else 0)
        result.exit_code = _reap(proc, grace)
    except (BrokenPipeError, OSError) as exc:
        result.error = result.error or f"the decode-check process went away: {exc}"
        result.exit_code = _reap(proc, 0)
    finally:
        if proc.poll() is None:
            result.exit_code = _reap(proc, 0)
        reader.join(timeout=10)
        for stream in (proc.stdin, proc.stdout):
            try:
                if stream is not None:
                    stream.close()
            except (BrokenPipeError, OSError, ValueError):
                pass  # a child that died left "exit" unsent in the buffer
        if report is None or (result.exit_code not in (0, None) and not result.error):
            tail = _tail(stderr)
            code = result.exit_code
            what = (
                f"killed by signal {-code}" if isinstance(code, int) and code < 0
                else f"exited {code}"
            )
            result.error = (result.error or f"the decode-check process {what} before reporting") + (
                f"; stderr: {tail[-800:]}" if tail else "")
            result.created = result.created and report is not None
            result.decode_ok = result.decode_ok and report is not None
        stderr.close()
    _settle(measure, before)
    result.seconds = time.monotonic() - started
    return result


# =============================================================================
# The child
# =============================================================================


def _say(out: TextIO, message: dict[str, Any]) -> None:
    out.write(MARKER + json.dumps(message) + "\n")
    out.flush()


def _throwaway_tokens(llm: Any, n_tokens: int) -> list[int]:
    text = _FILLER * (n_tokens // 8 + 4)
    tokens = list(llm.tokenize(text.encode("utf-8"), add_bos=True, special=False))
    while tokens and len(tokens) < n_tokens:
        tokens += tokens[1:] or tokens
    return tokens[:n_tokens]


def child_main(stdin: TextIO = sys.stdin, stdout: TextIO = sys.stdout) -> int:
    """The child's whole life: one construction, one full batch, one token, one report."""
    request = json.loads(stdin.readline())
    kwargs = dict(request["kwargs"])
    try:
        import llama_cpp  # the child's only non-stdlib import

        llm = llama_cpp.Llama(**kwargs)
    except Exception as exc:  # noqa: BLE001 - reported, the parent decides
        _say(stdout, {"stage": "construct", "created": False, "decode_ok": False,
                      "error": f"{type(exc).__name__}: {exc}"[:500]})
        return 0
    n_tokens = max(1, min(int(request.get("n_tokens") or 1), int(llm.n_batch)))
    try:
        tokens = _throwaway_tokens(llm, n_tokens)
        if not tokens:
            raise RuntimeError("the model tokenized the filler to nothing")
        # One full batch: the prompt-processing graph at its largest (llama-cpp-python's `eval`
        # decodes in `n_batch` chunks), then one token: the generation graph.
        llm.eval(tokens)
        llm.eval(tokens[:1])
    except Exception as exc:  # noqa: BLE001 - llama_decode returned an error code
        _say(stdout, {"stage": "decode", "created": True, "decode_ok": False,
                      "tokens": n_tokens, "error": f"{type(exc).__name__}: {exc}"[:500]})
        return 0
    _say(stdout, {"stage": "decoded", "created": True, "decode_ok": True, "tokens": len(tokens)})
    # Hold the context while the parent reads the card, then go. The throwaway prompt lives only
    # in this process: the served instance never sees it.
    stdin.readline()
    try:
        llm.reset()
    except Exception:  # noqa: BLE001
        pass
    return 0


if __name__ == "__main__":
    code = child_main()
    sys.stdout.flush()
    # Skip interpreter and CUDA teardown: the driver frees the card when the process ends.
    os._exit(code)
