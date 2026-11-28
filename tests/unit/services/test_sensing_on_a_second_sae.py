"""Sensing armed on a cluster whose SAE is not the first attached entry SENSES (044 debt, 2026-11-25).

028 FU-1 made every arming path arm the cluster's OWN SAE. The serving side was not changed with it:
`_sensing_begin`, the serial-routing clause and the two batch skip lines read
`AttachedSAEState().attached_sae`, the FIRST entry. With the cluster on the second SAE, begin found
the first SAE unarmed and returned None: every request went unsensed while `/api/sensing/status`
reported `armed: true`. Found by 044's relayed-route module, which had to bind its cluster to layer 0.

Two REAL `LoadedSAE`s from different seeds, hooked into a tiny real Llama and registered in the real
`AttachedSAEState`; a real `SensingService` arms the second; one real chat request is served on the
serial path. Only the database write (`record`) is replaced, by a spy that keeps its arguments.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest

import miserver.api.dependencies as deps
from miserver.services.sae_service import AttachedSAEState
from miserver.services.sensing_service import SensingService
from tests.unit.f28_fixtures import (
    SAE_A,
    SAE_B,
    build,
    chat,
    clean_state,  # noqa: F401 - fixture
    no_cuda,
)


def _cluster(profile_id: str, sae_id: str, layer: int) -> SimpleNamespace:
    """A cluster profile on `(sae_id, layer)`, sensing with a quorum of one: four members with a small
    activation scale, so the random seeded SAE fires on the prompt."""
    return SimpleNamespace(
        id=profile_id, name=f"cluster {profile_id}", source_kind="cluster", sae_id=sae_id,
        layer=layer, is_active=True, sensing_enabled=True,
        cluster_meta={
            "sae": {"n_features": 8},
            "members": [{"feature_idx": i, "max_activation": 0.05} for i in range(4)],
            "sensing": {"min_k": 1},
        },
    )


class _Recorded:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def __call__(self, request_id, hits, truncated, full_ids, tokenizer, **kw):
        self.calls.append({"request_id": request_id, "hits": list(hits),
                           "profile_id": kw.get("profile_id")})
        return []


@pytest.fixture
def two_saes(clean_state, monkeypatch):  # noqa: F811 - fixture
    served = build([(SAE_A, 0, 11), (SAE_B, 1, 12)])
    service = SensingService()
    recorded = _Recorded()
    service.record = recorded
    monkeypatch.setattr(deps, "_sensing_service", service)
    yield served, service, recorded
    for handle in served.handles:
        handle.remove()


async def _serve(served) -> str:
    with no_cuda():
        response = await served.svc.create_chat_completion(chat())
    return response.id


@pytest.mark.asyncio
async def test_a_cluster_on_the_second_sae_records_its_moments(two_saes):
    served, service, recorded = two_saes
    first, second = served.sae(SAE_A, 0), served.sae(SAE_B, 1)
    service.arm_for_profile(_cluster("prof_b", SAE_B, 1), second)
    assert second.is_sensing_armed and not first.is_sensing_armed, "fixture: the SECOND entry is armed"
    assert AttachedSAEState().attached_sae is first, "fixture: the armed SAE is not the first entry"

    request_id = await _serve(served)

    assert len(recorded.calls) == 1, (
        "a request served with sensing armed on the cluster's own (second) SAE recorded nothing: "
        "the serving path read the first entry"
    )
    call = recorded.calls[0]
    assert call["request_id"] == request_id
    assert call["profile_id"] == "prof_b"
    assert call["hits"], "the armed members fired on the prompt and no hit was kept"


@pytest.mark.asyncio
async def test_a_cluster_on_the_first_sae_still_records(two_saes):
    """The control on the other entry: the fix must not have moved sensing off the first SAE."""
    served, service, recorded = two_saes
    service.arm_for_profile(_cluster("prof_a", SAE_A, 0), served.sae(SAE_A, 0))

    request_id = await _serve(served)

    assert [c["request_id"] for c in recorded.calls] == [request_id]
    assert recorded.calls[0]["profile_id"] == "prof_a"


def test_begin_opens_the_boundary_on_the_armed_entry(two_saes):
    served, service, _ = two_saes
    first, second = served.sae(SAE_A, 0), served.sae(SAE_B, 1)
    service.arm_for_profile(_cluster("prof_b", SAE_B, 1), second)

    ctx = served.svc._sensing_begin("req-1")

    assert ctx is not None and ctx.sae is second and ctx.profile_id == "prof_b"
    assert second._sensing_began and not first._sensing_began


def test_a_switch_to_a_cluster_on_another_sae_leaves_one_sae_armed(two_saes):
    """Arming cluster B on the second SAE while cluster A is armed on the first left A's SAE armed with
    A's config: begin (first entry) then sensed for the INACTIVE cluster A, and B went unsensed."""
    served, service, _ = two_saes
    first, second = served.sae(SAE_A, 0), served.sae(SAE_B, 1)
    service.arm_for_profile(_cluster("prof_a", SAE_A, 0), first)
    service.arm_for_profile(_cluster("prof_b", SAE_B, 1), second)

    assert not first.is_sensing_armed, "the previous cluster's SAE kept its arm"
    assert second.is_sensing_armed and service.armed_profile_id == "prof_b"
    ctx = served.svc._sensing_begin("req-1")
    assert ctx is not None and ctx.sae is second and ctx.profile_id == "prof_b"


def test_begin_prefers_the_services_profile_over_a_stale_arm(two_saes):
    """Should two entries ever be armed (the invariant above broken), begin takes the SAE armed for the
    profile the service reports, never whichever is first."""
    served, service, _ = two_saes
    first, second = served.sae(SAE_A, 0), served.sae(SAE_B, 1)
    first.arm_sensing(service.build_config(_cluster("prof_stale", SAE_A, 0)))
    service.arm_for_profile(_cluster("prof_b", SAE_B, 1), second)
    first.arm_sensing(service.build_config(_cluster("prof_stale", SAE_A, 0)))  # re-planted after arm

    ctx = served.svc._sensing_begin("req-1")

    assert ctx is not None and ctx.sae is second and ctx.profile_id == "prof_b"


class _RunningCBM:
    is_running = True

    def sampling_params_match(self, temperature, top_p) -> bool:
        return True


def test_armed_sensing_on_the_second_sae_forces_the_serial_path(two_saes):
    """CBM rows cannot be attributed to a request, so armed sensing forces serial (SEN-S1). The clause
    read the first entry, so a second-SAE cluster was served by CBM and never sensed."""
    served, service, _ = two_saes
    served.svc._cbm_backend = _RunningCBM()
    with patch("miserver.core.config.settings.SENSING_FORCE_SERIAL", True), \
         patch("miserver.services.weight_attachment.get_weight_service") as weights:
        weights.return_value.is_attached.return_value = False
        assert served.svc._use_cbm_for_request() is True, "control: nothing armed routes to CBM"
        service.arm_for_profile(_cluster("prof_b", SAE_B, 1), served.sae(SAE_B, 1))
        assert served.svc._use_cbm_for_request() is False


@pytest.mark.asyncio
async def test_detaching_an_unarmed_sae_keeps_the_service_armed(two_saes):
    """Detach disarmed the SERVICE for any SAE it removed, armed or not: removing the first (unarmed)
    SAE left the second armed and sensing while `/api/sensing/status` said `armed: false`."""
    from unittest.mock import AsyncMock

    from miserver.services.sae_service import SAEService

    served, service, recorded = two_saes
    second = served.sae(SAE_B, 1)
    service.arm_for_profile(_cluster("prof_b", SAE_B, 1), second)

    sae_service = SAEService.__new__(SAEService)
    sae_service._sae_state = AttachedSAEState()
    sae_service._hooker = SimpleNamespace(remove=lambda handle: handle.remove())
    sae_service._reset_dynamo_for_hook_change = lambda: None
    sae_service.repository = SimpleNamespace(update_status=AsyncMock(), deactivate_attachment=AsyncMock())
    with no_cuda():
        await sae_service._detach_sae_admitted(SAE_A)

    assert [e.sae_id for e in AttachedSAEState().entries()] == [SAE_B], "fixture: only the first was detached"
    assert second.is_sensing_armed
    assert service.is_armed and service.armed_profile_id == "prof_b", (
        "detaching an SAE sensing was not armed on disarmed the service"
    )
    assert service.status()["armed"] is True
    request_id = await _serve(served)
    assert [c["request_id"] for c in recorded.calls] == [request_id]


@pytest.mark.asyncio
async def test_detaching_the_armed_sae_disarms_the_service(two_saes):
    """The control for the test above: detaching the SAE sensing IS armed on still disarms."""
    from unittest.mock import AsyncMock

    from miserver.services.sae_service import SAEService

    served, service, _ = two_saes
    second = served.sae(SAE_B, 1)
    service.arm_for_profile(_cluster("prof_b", SAE_B, 1), second)

    sae_service = SAEService.__new__(SAEService)
    sae_service._sae_state = AttachedSAEState()
    sae_service._hooker = SimpleNamespace(remove=lambda handle: handle.remove())
    sae_service._reset_dynamo_for_hook_change = lambda: None
    sae_service.repository = SimpleNamespace(update_status=AsyncMock(), deactivate_attachment=AsyncMock())
    with no_cuda():
        await sae_service._detach_sae_admitted(SAE_B)

    assert not second.is_sensing_armed
    assert not service.is_armed


def _skip_reasons(log) -> list:
    return [c.kwargs.get("reason") for c in log.info.call_args_list if c.args[:1] == ("sensing_skipped",)]


@pytest.mark.asyncio
async def test_a_batch_says_it_went_unsensed_for_a_cluster_on_the_second_sae(two_saes):
    """A batch is never sensed, and must say so (a quietly dark sensing path is the failure this project
    has shipped before). The line asked the first entry whether sensing was armed."""
    from miserver.api.schemas.openai import ChatMessage

    served, service, recorded = two_saes
    service.arm_for_profile(_cluster("prof_b", SAE_B, 1), served.sae(SAE_B, 1))
    request = chat(extra_messages=[[ChatMessage(role="user", content="w2 w3")]])
    with no_cuda(), patch("miserver.services.inference_service.logger") as log:
        await served.svc.create_chat_completion(request)

    assert "batched_request" in _skip_reasons(log)
    assert recorded.calls == []


@pytest.mark.asyncio
async def test_n_above_one_says_it_went_unsensed_for_a_cluster_on_the_second_sae(two_saes):
    served, service, recorded = two_saes
    service.arm_for_profile(_cluster("prof_b", SAE_B, 1), served.sae(SAE_B, 1))
    with no_cuda(), patch("miserver.services.inference_service.logger") as log:
        await served.svc.create_chat_completion(chat(n=2))

    assert "n_gt_1" in _skip_reasons(log)
    assert recorded.calls == []
