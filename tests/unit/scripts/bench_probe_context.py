"""Benchmark of the path 043 changed: opening a probe request context (FTID §13, FTDD §9).

Not collected (no `test_` prefix). Builds `ProbeRequestContext("bench", [armed])` N times with a
real `WeightBarrier` and a resident model stand-in, and prints the mean and p99 in microseconds.
043 added `served_weights()` (one barrier snapshot under its lock) to every context. Run it on
`main` and on the branch; no threshold is asserted (memory `check-ci-not-just-local-green`).

    venv/bin/python tests/unit/scripts/bench_probe_context.py [N]
"""

from __future__ import annotations

import statistics
import sys
import time
from datetime import datetime
from types import SimpleNamespace

import torch


def main(n: int = 10_000) -> None:
    from miserver.ml.model_loader import LoadedModelState
    from miserver.ml.probe_head import ProbeHead
    from miserver.services.probe_runtime import ArmedProbe, ProbeRequestContext
    from miserver.services.weight_barrier import WeightBarrier, set_weight_barrier

    set_weight_barrier(WeightBarrier())
    state = LoadedModelState()
    state._loaded = SimpleNamespace(model_id=1, model_name="bench", weights_identity=None,
                                    loaded_at=datetime(2026, 11, 16))
    armed = ArmedProbe(
        probe_id="pr_bench", name="bench", head=ProbeHead(weight=torch.ones(16), bias=0.0, layer=0),
        rule="mean", scope="all", layer=0, rung=2, rung_language="bench", threshold=0.0,
        windows=("all",),
    )
    for _ in range(200):  # warm-up
        ProbeRequestContext("bench", [armed])
    samples = []
    for _ in range(n):
        started = time.perf_counter()
        ProbeRequestContext("bench", [armed])
        samples.append((time.perf_counter() - started) * 1e6)
    samples.sort()
    p99 = samples[int(0.99 * (len(samples) - 1))]
    print(f"n={n} mean_us={statistics.fmean(samples):.2f} p99_us={p99:.2f}")
    state._loaded = None
    set_weight_barrier(None)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 10_000)
