"""The two fields the response models dropped are declared (044 FR-044.12, SH9).

`CircuitActivationResponse` lacked `applied_epoch` (sent by `circuit_service.py:977`) and `ModelPlacement`
lacked `working_mb_by_device` (sent by `Placement.to_dict`, `ml/gpu_placement.py:521`): FastAPI dropped both
from every answer, silently. Both routes are enforced, so the field must be on the model for it to reach the
wire; the contract then describes it.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from miserver.api.schemas.model import ModelPlacement
from miserver.main import create_app
from miserver.ml.gpu_placement import Placement
from tests.unit.api.test_circuit_composed_activation_route import _COMPOSED
from tests.unit.services.test_circuit_service import clean_registry, service  # noqa: F401


def test_an_activation_carries_the_epoch_the_service_returned():
    from miserver.api.dependencies import get_circuit_service

    service = MagicMock()
    service.activate = AsyncMock(return_value={**_COMPOSED, "applied_epoch": 41})
    app = create_app()
    app.dependency_overrides[get_circuit_service] = lambda: service
    response = TestClient(app).post(
        "/api/circuits/circ_f309009bca3e/activate",
        params={"acknowledge_unvalidated": "true", "allow_layer_overlap": "true"},
    )
    assert response.status_code == 200, response.text
    assert response.json()["data"]["applied_epoch"] == 41


def test_a_placement_keeps_its_working_memory_by_device():
    placement = Placement(
        mode="shard",
        reason="no_single_card_fits",
        required_mb=100,
        working_mb_by_device=(("cuda:0", 1200), ("cuda:1", 900)),
    )
    sent = placement.to_dict()
    assert sent["working_mb_by_device"] == {"cuda:0": 1200, "cuda:1": 900}
    assert (
        ModelPlacement(**sent).model_dump()["working_mb_by_device"] == sent["working_mb_by_device"]
    )
    # A placement recorded before the field existed still reads, as an empty map.
    older = {k: v for k, v in sent.items() if k != "working_mb_by_device"}
    assert ModelPlacement(**older).working_mb_by_device == {}


def test_the_contract_declares_both():
    from tests.support.answer_validator import _contract

    defs = _contract()["$defs"]
    activation = defs["CircuitActivationResponse"]
    # R2-1: every serve reports it, so it is an integer, always sent.
    assert activation["properties"]["applied_epoch"]["type"] == "integer"
    assert "applied_epoch" in activation["required"]
    placement = defs["ModelPlacement"]
    assert placement["properties"]["working_mb_by_device"]["additionalProperties"] == {
        "type": "integer"
    }
    assert "working_mb_by_device" in placement["required"]


@pytest.mark.parametrize("layers", [(10,), (10, 13)], ids=["slice_fallback", "full"])
async def test_a_real_activation_reports_the_live_epoch(service, layers):  # noqa: F811
    """044 R1-2 and R2-1, through the REAL service (not a stand-in that agrees with the model by
    construction): a slice-fallback serve sent no epoch (R1-2: a 500 once the field was required),
    and a full serve sent the epoch read BEFORE activation's own bump (R2-1: one behind the live
    epoch at once). Every activation now reports the epoch its write took effect under."""
    from miserver.api.schemas.circuit import CircuitActivationResponse
    from miserver.services.sae_service import AttachedSAEState
    from tests.unit.services.test_circuit_service import attach, make_doc

    for layer in layers:
        attach(f"sae-{layer}", layer)
    circuit = await service.import_definition(make_doc())
    result = await service.activate(circuit.id)
    assert result["serving_mode"] == ("slice_fallback" if len(layers) == 1 else "full")
    assert result["applied_epoch"] == AttachedSAEState().steering_epoch
    # As the route builds its answer (routes/management/circuits.py: the map keys become strings).
    applied = result.get("applied_per_layer")
    if applied:
        result["applied_per_layer"] = {
            str(layer): {str(i): v for i, v in steering.items()} for layer, steering in applied.items()
        }
    assert CircuitActivationResponse(**result).applied_epoch == AttachedSAEState().steering_epoch
