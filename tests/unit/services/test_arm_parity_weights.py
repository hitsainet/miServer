"""The ARM path's parity report states the weights fact read inside its slot (043 FR-043.5, FR-043.6).

R1 review fix (OR1-043-1), 2026-11-16. `test_parity_on_changed_weights.py` drives `ProbeParityEngine`
with a `model_summary` it builds itself, so it never reaches the arm path's own caller
(`probe_arming.py`: `model=model_summary(loaded, weights=served_weights())`). Replacing that argument
with a constant base fact stayed green across the 043 set and the arming suites (review mutation R9).

These tests go through the real `ProbeArmingService.arm`, with `served_weights` patched where the arm
path reads it, and assert the call (once, no arguments) and the stored payload:
- parity passes on differing weights: the probe arms, the stored report states the fact, no note;
- parity fails on differing weights: the stored report states the fact and carries `weights_note`.
"""

from __future__ import annotations

import pytest

from miserver.core.errors import ProbeParityFailedError
from miserver.db.repositories.probe_repository import ProbeRepository
from miserver.services import probe_arming
from miserver.services.probe_arming import ProbeArmingService
from miserver.services.probe_parity import WEIGHTS_NOTE
from miserver.services.probe_runtime import ProbeRuntimeState
from miserver.services.probe_service import ProbeService
from miserver.services.resident_identity import ServedWeights
from tests.unit.services.test_probe_arming import TinyModel, definition, forward_at, loaded, run_inline

pytestmark = pytest.mark.asyncio

EDITED = ServedWeights("tiny", "model", 3, True)


@pytest.fixture(autouse=True)
def clean_state():
    ProbeRuntimeState.reset_for_tests()
    yield
    ProbeRuntimeState.reset_for_tests()


@pytest.fixture
def weights_reads(monkeypatch):
    calls: list[tuple] = []

    def _served_weights(*args):
        calls.append(args)
        return EDITED

    monkeypatch.setattr(probe_arming, "served_weights", _served_weights)
    return calls


async def _arm(test_session, value: float):
    repo = ProbeRepository(test_session)
    probe = await ProbeService(repo).import_definition(definition())
    arming = ProbeArmingService(repo, ProbeRuntimeState())
    return repo, probe, arming.arm(
        probe, executor=run_inline, model=TinyModel(), loaded=loaded(), forward=forward_at(value))


async def test_a_pass_on_differing_weights_arms_and_states_the_fact(test_session, weights_reads):
    repo, probe, arm = await _arm(test_session, 0.5)
    armed = await arm
    assert armed.probe_id == probe.id
    stored = await repo.get(probe.id)
    assert stored.armed is True
    assert weights_reads == [()], weights_reads
    assert stored.parity["passed"] is True
    assert stored.parity["model"]["served_weights"] == EDITED.as_dict()
    assert stored.parity["weights_note"] is None, "a pass carries no note"


async def test_a_failure_on_differing_weights_carries_the_note(test_session, weights_reads):
    repo, probe, arm = await _arm(test_session, 0.9)  # 7.2 per token against the recorded 4.0
    with pytest.raises(ProbeParityFailedError):
        await arm
    stored = await repo.get(probe.id)
    assert stored.armed is False
    assert weights_reads == [()], weights_reads
    assert stored.parity["passed"] is False
    assert stored.parity["model"]["served_weights"] == EDITED.as_dict()
    assert stored.parity["weights_note"] == WEIGHTS_NOTE


async def test_the_fact_is_read_inside_the_slot(test_session, monkeypatch):
    """R2 review fix (OR2-043-3), 2026-11-16: with an inline executor and a constant fact, a read moved
    before the executor (outside the slot) states the same fact (review mutation S1 stayed green). Here the
    fact changes when the slot is entered, as a commit landing while the arm waited for the slot would."""
    base = ServedWeights("tiny", "model", 0, False)
    state = {"in_slot": False}
    monkeypatch.setattr(probe_arming, "served_weights", lambda *a: EDITED if state["in_slot"] else base)

    async def slot(fn):
        state["in_slot"] = True
        try:
            return fn()
        finally:
            state["in_slot"] = False

    repo = ProbeRepository(test_session)
    probe = await ProbeService(repo).import_definition(definition())
    await ProbeArmingService(repo, ProbeRuntimeState()).arm(
        probe, executor=slot, model=TinyModel(), loaded=loaded(), forward=forward_at(0.5))
    stored = await repo.get(probe.id)
    assert stored.parity["model"]["served_weights"] == EDITED.as_dict()
