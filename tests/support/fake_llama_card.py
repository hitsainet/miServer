"""A llama.cpp stand-in that takes memory from fake cards the way llama.cpp does (2026-11-21).

`fake_gpus` gives the loader per-card free memory it reads through nvidia-smi; a bare MagicMock
Llama leaves that memory untouched, so every window "fits" with the whole card free and a
headroom check can never fail. `LlamaOnCards` makes a construction TAKE what llama.cpp would
(weights, cache, compute buffers: `used_mb(n_ctx)`), split over the cards its kwargs name, and
give it back on `close()`. Creating a context fails, as in llama.cpp, when a card cannot hold its
share. `decode_check` stands in for `gguf_decode_check.run_decode_check`: a child that creates the
same context, then needs `decode_mb` more on each card to decode, and aborts (exit 139, as the
pod did on 2026-11-21) when a card cannot give it.

Model 17's live figures (2026-11-21, RTX 3090, q8_0): the card used 17,112 MiB at 32,768 tokens
and 24,040 MiB at 222,208, i.e. 15,914 MiB + 38,347 B a token (`MODEL_17`), from ~24,081 MiB free.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock

from miserver.ml.gguf_decode_check import DecodeCheckResult
from tests.support.fake_gpus import FakeCards

MIB = 1024 * 1024


@dataclass(frozen=True)
class CardCost:
    base_mb: float
    per_token_bytes: float

    def used_mb(self, n_ctx: int) -> float:
        return self.base_mb + n_ctx * self.per_token_bytes / MIB


#: Model 17 at q8_0, fitted to the two live readings (17,112 @ 32,768; 24,040 @ 222,208).
MODEL_17 = CardCost(base_mb=17_112 - 32_768 * 38_347 / MIB, per_token_bytes=38_347)
#: ~24,081 MiB: the free memory the live prediction of 155,648 implies.
MODEL_17_FREE_MB = 24_081


class LlamaOnCards:
    def __init__(
        self,
        fake: FakeCards,
        cost: CardCost = MODEL_17,
        *,
        decode_mb: float = 300,
        decode_fails_above: int | None = None,
    ):
        self.fake = fake
        self.cost = cost
        self.decode_mb = decode_mb
        self.decode_fails_above = decode_fails_above
        self.calls: list[int] = []
        self.trials: list[int] = []
        self.open = 0
        self.most_open = 0
        self.open_during_trial: list[int] = []

    def shares(self, kwargs: dict[str, Any]) -> dict[int, float]:
        split = kwargs.get("tensor_split")
        if split:
            total = sum(split)
            return {i: v / total for i, v in enumerate(split) if v > 0}
        return {int(kwargs.get("main_gpu", 0)): 1.0}

    def _take_mb(self, need: dict[int, float]) -> bool:
        if any(mb > self.fake.free_mb[i] for i, mb in need.items()):
            return False
        for i, mb in need.items():
            self.fake.free_mb[i] -= int(round(mb))
        return True

    def _take(self, kwargs: dict[str, Any]) -> dict[int, float] | None:
        used = self.cost.used_mb(int(kwargs["n_ctx"]))
        need = {i: share * used for i, share in self.shares(kwargs).items()}
        return need if self._take_mb(need) else None

    def _give(self, need: dict[int, float]) -> None:
        for i, mb in need.items():
            self.fake.free_mb[i] += int(round(mb))

    def __call__(self, **kwargs: Any) -> Any:
        self.calls.append(int(kwargs["n_ctx"]))
        if self.open:
            raise ValueError("Failed to load model from file: a second copy of the weights")
        need = self._take(kwargs)
        if need is None:
            raise ValueError("Failed to create llama_context")
        self.open += 1
        self.most_open = max(self.most_open, self.open)
        instance = MagicMock()

        def _close() -> None:
            self.open -= 1
            self._give(need)

        instance.close = MagicMock(side_effect=_close)
        return instance

    def decode_check(
        self,
        kwargs: dict[str, Any],
        n_tokens: int,
        measure: Callable[[], dict[int, int]],
        **_: Any,
    ) -> DecodeCheckResult:
        n_ctx = int(kwargs["n_ctx"])
        self.trials.append(n_ctx)
        self.open_during_trial.append(self.open)
        result = DecodeCheckResult(n_ctx=n_ctx)
        need = self._take(kwargs)
        if need is None:
            result.error = "ValueError: Failed to create llama_context"
            result.exit_code = 0
            return result
        try:
            decode = {i: share * self.decode_mb for i, share in self.shares(kwargs).items()}
            crash = self.decode_fails_above is not None and n_ctx > self.decode_fails_above
            if crash or not self._take_mb(decode):
                # ggml_cuda_error -> GGML_ABORT: the child dies before it can report.
                result.exit_code = 139
                result.error = ("the decode-check process exited 139 before reporting; stderr: "
                                "CUDA error: out of memory ... ggml-cuda.cu:109")
                return result
            result.created = result.decode_ok = True
            result.tokens = n_tokens
            result.exit_code = 0
            result.free_mb = dict(measure())
            self._give(decode)
            return result
        finally:
            self._give(need)
