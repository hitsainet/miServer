"""The model pin counts commits (043 FR-043.7; pre-existing defect fixed under D43-6).

`probe_scoring.model_pin` is what "the same model" means for stateless scoring (`MODEL_CHANGED`)
and for arming (`MODEL_CHANGED` too, 409; R3 review fix (OR3-043-1): this said `PROBE_MODEL_CHANGED`). Before 043 it held the row, the load, the base epoch and the
adapter; a commit moves none of them, so an input scored after a commit counted as the same model
as the input before it, and a probe checked on one parameter set was armed on another.

Both cases run on the CPU weights flow with REAL commits through the holder routes
(`tests/unit/weights_e2e_fixtures.py`), the commit landing exactly between the two moments the pin
is compared.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import torch

from miserver.core.errors import ProbeModelChangedError
from miserver.services.probe_runtime import ProbeRuntimeState
from tests.unit.score_fixtures import tiny_definition, tiny_identity
from tests.unit.weights_e2e_fixtures import attach, flow, holder  # noqa: F401 - flow is a fixture


async def _commit(client, attachment_id: str, model) -> None:
    """One real bracketed commit: quiesce, commit-begin, an in-place write, commit-end, resume."""
    h = holder(attachment_id)
    assert (await client.post("/api/weights/quiesce", headers=h)).status_code == 200
    assert (await client.post("/api/weights/commit-begin", headers=h)).status_code == 200
    with torch.no_grad():
        next(model.parameters()).add_(0.01)
    end = await client.post("/api/weights/commit-end", headers=h)
    assert end.status_code == 200, end.text
    assert (await client.post("/api/weights/resume", headers=h)).status_code == 200


@pytest.fixture
async def scoring(flow, test_session, monkeypatch):  # noqa: F811
    """`POST /api/probes/score` on the flow's resident, with one imported probe."""
    from miserver.api.dependencies import get_db, get_probe_repository
    from miserver.db.repositories.probe_repository import ProbeRepository
    from miserver.services.probe_service import ProbeService

    ProbeRuntimeState.reset_for_tests()
    monkeypatch.setattr(
        "miserver.services.probe_arm_bridge.loaded_identity",
        AsyncMock(side_effect=lambda _s: (tiny_identity(), flow.resident.model,
                                          flow.resident.tokenizer)),
    )
    repo = ProbeRepository(test_session)

    async def db():
        yield test_session

    flow.app.dependency_overrides[get_db] = db
    flow.app.dependency_overrides[get_probe_repository] = lambda: repo
    row = await ProbeService(repo).import_definition(tiny_definition())
    yield SimpleNamespace(repo=repo, row=row)
    ProbeRuntimeState.reset_for_tests()


async def test_a_commit_between_two_inputs_refuses_the_second(flow, scoring, monkeypatch):  # noqa: F811
    async with flow.client() as client:
        attachment_id = await attach(client)
        real = flow.inference.run_model_work
        calls = {"n": 0}

        async def commit_after_the_first(fn, *args, **kwargs):
            out = await real(fn, *args, **kwargs)
            calls["n"] += 1
            if calls["n"] == 1:
                await _commit(client, attachment_id, flow.model)
            return out

        monkeypatch.setattr(flow.inference, "run_model_work", commit_after_the_first)
        r = await client.post("/api/probes/score", json={
            "inputs": [{"token_ids": [2, 6, 7, 8]}, {"token_ids": [2, 6, 7, 9]}],
            "windows": ["all"],
        })
        assert r.status_code == 200, r.text
        results = r.json()["data"]["results"]
        assert flow.barrier.snapshot().steps_since_load == 1, "the commit really landed"
        assert results[0]["error"] is None and len(results[0]["verdicts"]) == 1
        assert results[1]["error"] is not None, "input 1 was scored on different weights"
        assert results[1]["error"]["code"] == "MODEL_CHANGED"
        assert results[1]["verdicts"] == []


async def test_a_commit_between_check_and_commit_arms_nothing(flow, scoring, monkeypatch):  # noqa: F811
    from miserver.services.probe_arming import ProbeArmingService

    async with flow.client() as client:
        attachment_id = await attach(client)

        async def check_then_commit(self, probe, **_kwargs):
            # The gates passed on the weights as they were; then the trainer commits.
            await _commit(client, attachment_id, flow.model)
            return SimpleNamespace(probe=probe, armed=None, parity=None)

        monkeypatch.setattr(ProbeArmingService, "check", check_then_commit)
        service = ProbeArmingService(scoring.repo)
        with pytest.raises(ProbeModelChangedError) as exc:
            await service.arm(
                scoring.row, model=flow.resident.model, loaded=tiny_identity(),
                forward=lambda *_: None, executor=flow.inference.run_model_work,
            )
        assert flow.barrier.snapshot().steps_since_load == 1, "the commit really landed"
        before, after = exc.value.details["pin_before"], exc.value.details["pin_after"]
        assert len(before) == len(after) == 5, "_pin_text formats the five-element pin"
        assert before[:4] == after[:4] and (before[4], after[4]) == ("0", "1")
        assert ProbeRuntimeState().armed() == [], "nothing is armed"
        _assert_names_the_commit(exc.value)


# R2 review fix (OR2-043-2), 2026-11-16: training commits MANY times. The two tests above take the first
# commit after load (0 -> 1), so a pin that keys on "any commit yet" (review mutation F7, the count capped at
# one) passed both. These take one commit first and compare across the second (1 -> 2).


async def test_a_second_commit_between_two_inputs_refuses_the_second(flow, scoring, monkeypatch):  # noqa: F811
    async with flow.client() as client:
        attachment_id = await attach(client)
        await _commit(client, attachment_id, flow.model)
        assert flow.barrier.snapshot().steps_since_load == 1
        real = flow.inference.run_model_work
        calls = {"n": 0}

        async def commit_after_the_first(fn, *args, **kwargs):
            out = await real(fn, *args, **kwargs)
            calls["n"] += 1
            if calls["n"] == 1:
                await _commit(client, attachment_id, flow.model)
            return out

        monkeypatch.setattr(flow.inference, "run_model_work", commit_after_the_first)
        r = await client.post("/api/probes/score", json={
            "inputs": [{"token_ids": [2, 6, 7, 8]}, {"token_ids": [2, 6, 7, 9]}],
            "windows": ["all"],
        })
        assert r.status_code == 200, r.text
        results = r.json()["data"]["results"]
        assert flow.barrier.snapshot().steps_since_load == 2, "the second commit really landed"
        assert results[0]["error"] is None and len(results[0]["verdicts"]) == 1
        assert results[1]["error"] is not None, "input 1 was scored on different weights"
        assert results[1]["error"]["code"] == "MODEL_CHANGED"
        assert results[1]["verdicts"] == []


async def test_a_second_commit_between_check_and_commit_arms_nothing(flow, scoring, monkeypatch):  # noqa: F811
    from miserver.services.probe_arming import ProbeArmingService

    async with flow.client() as client:
        attachment_id = await attach(client)
        await _commit(client, attachment_id, flow.model)
        assert flow.barrier.snapshot().steps_since_load == 1

        async def check_then_commit(self, probe, **_kwargs):
            await _commit(client, attachment_id, flow.model)
            return SimpleNamespace(probe=probe, armed=None, parity=None)

        monkeypatch.setattr(ProbeArmingService, "check", check_then_commit)
        service = ProbeArmingService(scoring.repo)
        with pytest.raises(ProbeModelChangedError) as exc:
            await service.arm(
                scoring.row, model=flow.resident.model, loaded=tiny_identity(),
                forward=lambda *_: None, executor=flow.inference.run_model_work,
            )
        assert flow.barrier.snapshot().steps_since_load == 2, "the second commit really landed"
        before, after = exc.value.details["pin_before"], exc.value.details["pin_after"]
        assert before[:4] == after[:4] and (before[4], after[4]) == ("1", "2")
        assert ProbeRuntimeState().armed() == [], "nothing is armed"
        # R3 review fix (OR3-043-6), 2026-11-16: the refusal names the cause and the single-probe tool.
        _assert_names_the_commit(exc.value)


# R3 review fix (OR3-043-6), 2026-11-16: the arming refusal says a training commit changed the weights when
# that is the cause, labels the pin, and names `miserver_arm_probe` for a single probe. Code and status stay.


def _assert_names_the_commit(error) -> None:
    from miserver.services.probe_arming import PIN_FIELDS

    assert error.code == "MODEL_CHANGED" and error.status_code == 409
    assert error.next_tool == "miserver_arm_probe", error.next_tool
    assert error.details["cause"] == "weights_commit"
    assert error.details["changed"] == ["commits_since_load"]
    assert error.details["pin_fields"] == list(PIN_FIELDS)
    assert "weights commit landed" in error.message and "trainer" in error.message


def test_a_moved_model_is_still_reported_as_a_model_change():
    from miserver.services.probe_arming import _model_changed

    error = _model_changed(("m1", "t0", 0, None, 0), ("m2", "t1", 0, None, 0), ["pr_1"], single=False)
    assert error.code == "MODEL_CHANGED"
    assert error.details["cause"] == "model_changed"
    assert error.details["changed"] == ["model_id", "loaded_at"]
    assert error.next_tool == "miserver_arm_behaviour", "a set keeps the class's hint"
    assert "loaded model changed" in error.message
    nothing = _model_changed(("m1", "t0", 0, None, 0), None, ["pr_1"], single=True)
    assert nothing.details["cause"] == "model_changed" and nothing.next_tool == "miserver_arm_probe"
