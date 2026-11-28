"""A batch `probe` line after a commit is provisional (043 FR-043.2, G1).

A real BatchRunner over the real app and a tiny real Llama (`BatchHarness`), one imported probe,
and a REAL `WeightBarrier` taking one real commit (quiesce, commit-begin, commit-end, resume) before
the batch. The line's verdicts come from the same `ProbeScoringService` the route runs, read from
the batch output file.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import miserver.services.weight_barrier as barrier_module
from miserver.ml.model_loader import LoadedModelState
from miserver.services.weight_attachment import WeightsIdentity
from miserver.services.weight_barrier import WeightBarrier, set_weight_barrier
from tests.unit.batch_fixtures import (  # noqa: F401
    batch_db,
    batch_dir,
    client_for,
    harness,
)
from tests.unit.score_fixtures import tiny_definition, tiny_identity

OWNER = "e" * 64


async def _probe_line_verdicts(harness, custom_id: str) -> list[dict]:  # noqa: F811
    line = {"custom_id": custom_id, "method": "POST", "url": "/api/probes/score",
            "body": {"inputs": [{"token_ids": [2, 6, 7, 8]}], "windows": ["all"]}}
    async with client_for(harness.app()) as client:
        created = await harness.create(client, [line], endpoint="/api/probes/score", pack=False)
    assert created.status_code == 200, created.text
    await harness.drain()
    async with client_for(harness.app()) as client:
        batch = await harness.batch(created.json()["id"])
        out = await harness.lines(client, batch.output_file_id)
    assert batch.status == "completed", batch.errors
    (only,) = out
    data = only["response"]["body"]["data"]
    return [v for result in data["results"] for v in result["verdicts"]]


async def test_a_probe_line_after_one_commit_is_provisional(harness, monkeypatch):  # noqa: F811
    from miserver.db.repositories.probe_repository import ProbeRepository
    from miserver.services.probe_service import ProbeService

    current = LoadedModelState().current
    monkeypatch.setattr("miserver.services.probe_arm_bridge.loaded_identity",
                        AsyncMock(return_value=(tiny_identity(), current.model, current.tokenizer)))
    async with harness.factory() as session:
        await ProbeService(ProbeRepository(session)).import_definition(tiny_definition())
        await session.commit()
    monkeypatch.setattr(barrier_module, "_synchronize_resident_devices", lambda: None)
    barrier = WeightBarrier(inference=lambda: harness.inference)
    set_weight_barrier(barrier)
    try:
        base = await _probe_line_verdicts(harness, "before")
        assert [v["provisional"] for v in base] == [False]

        await barrier.quiesce(OWNER, WeightsIdentity("forge", "trainer"))
        barrier.commit_begin(OWNER)
        barrier.commit_end(OWNER)
        barrier.resume(OWNER)
        assert barrier.snapshot().steps_since_load == 1

        after = await _probe_line_verdicts(harness, "after")
        assert [v["provisional"] for v in after] == [True]
    finally:
        set_weight_barrier(None)
