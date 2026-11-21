"""Parity on changed weights runs and says so (043 FR-043.5, FR-043.6; D43-4).

On the CPU weights flow, a probe reading decoder layer 0 has its test vector recorded from the BASE
(the vector holds exactly what this build scores on the base, so the base run passes at zero
difference). Then one real commit through the holder routes writes:

- layer 0's MLP (the read layer): parity FAILS, and the report carries `weights_note`;
- layer 1's MLP (above the read layer, so the residual read at layer 0 is unchanged): parity
  PASSES with no note, and the probe's next live verdict is still provisional.

Every report's `model` block carries `served_weights`, read where the parity forward runs.
"""

from __future__ import annotations

import torch

from miserver.ml.probe_head import ProbeHead
from miserver.services.probe_arm_bridge import build_parity_forward
from miserver.services.probe_parity import WEIGHTS_NOTE, ProbeParityEngine, model_summary
from miserver.services.probe_runtime import ArmedProbe, ProbeRequestContext
from miserver.services.resident_identity import served_weights
from tests.unit.api.test_g1_weights_provisional import _detach, _row, probes  # noqa: F401
from tests.unit.weights_e2e_fixtures import attach, flow, holder  # noqa: F401 - fixtures

LAYER = 0
IDS = [0, 6, 7, 8, 9]


def _probe() -> ArmedProbe:
    torch.manual_seed(11)
    return ArmedProbe(
        probe_id="pr_parity", name="parity", head=ProbeHead(weight=torch.randn(16) * 20.0, bias=0.0,
                                                            layer=LAYER),
        rule="mean", scope="all", layer=LAYER, rung=2, rung_language="detects on unseen tasks",
        threshold=0.0, windows=("all",),
    )


def _recorded(model, probe) -> dict:
    """The test vector as this build scores it on the base: the definition miWorkbench would ship."""
    forward = build_parity_forward(model, LAYER)
    context = ProbeRequestContext("record", [probe])
    forward(torch.tensor([IDS], dtype=torch.long), context)
    (verdict,) = [v for v in context.finish() if v.window == "all"]
    tokens = context.token_scores_for(probe.probe_id)
    return {"test_vectors": {"tolerance": 0.001, "vectors": [
        {"token_ids": IDS, "token_scores": [float(t) for t in tokens], "score": verdict.score}]}}


def _parity(flow, probe, definition):
    model = flow.resident.model
    return ProbeParityEngine(build_parity_forward(model, LAYER)).run(
        probe, definition, tolerance=0.001,
        model=model_summary(None, weights=served_weights()),
    )


async def _commit_layer(client, attachment_id, model, layer: int) -> None:
    h = holder(attachment_id)
    assert (await client.post("/api/weights/quiesce", headers=h)).status_code == 200
    assert (await client.post("/api/weights/commit-begin", headers=h)).status_code == 200
    with torch.no_grad():
        model.model.layers[layer].mlp.down_proj.weight.add_(100.0)
    assert (await client.post("/api/weights/commit-end", headers=h)).status_code == 200
    assert (await client.post("/api/weights/resume", headers=h)).status_code == 200


async def test_a_commit_to_the_read_layer_fails_with_the_note(flow):  # noqa: F811
    probe = _probe()
    definition = _recorded(flow.model, probe)
    base = _parity(flow, probe, definition)
    assert base.passed and base.weights_note is None
    assert base.as_details()["model"]["served_weights"]["differs_from_base"] is False
    assert base.as_details()["weights_note"] is None
    async with flow.client() as client:
        attachment_id = await attach(client)
        await _commit_layer(client, attachment_id, flow.model, LAYER)
        report = _parity(flow, probe, definition)
        assert report.passed is False
        assert report.weights_note == WEIGHTS_NOTE
        details = report.as_details()
        assert details["weights_note"] == WEIGHTS_NOTE
        assert details["model"]["served_weights"] == {
            "name": "tiny", "origin": "model", "commits_since_load": 1, "differs_from_base": True}
        await _detach(client, attachment_id)


async def test_a_commit_above_the_read_layer_passes_but_verdicts_stay_provisional(
        flow, probes):  # noqa: F811
    probe = _probe()
    definition = _recorded(flow.model, probe)
    async with flow.client() as client:
        attachment_id = await attach(client)
        await _commit_layer(client, attachment_id, flow.model, LAYER + 1)
        report = _parity(flow, probe, definition)
        assert report.passed is True, report.as_details()
        assert report.weights_note is None, "a pass carries no note"
        assert report.as_details()["model"]["served_weights"]["differs_from_base"] is True
        await _detach(client, attachment_id)
        # A pass does not clear `provisional`: an armed probe's next verdict still is.
        provisional, _ = await _row(client, probes)
        assert provisional is True


async def test_a_failure_on_the_base_carries_no_note(flow):  # noqa: F811
    probe = _probe()
    definition = _recorded(flow.model, probe)
    definition["test_vectors"]["vectors"][0]["score"] += 1.0  # a real mismatch, on the base
    report = _parity(flow, probe, definition)
    assert report.passed is False and report.weights_note is None
