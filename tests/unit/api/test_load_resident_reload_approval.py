"""Approval cards that promise only what the action does, and filings that could only fail (2026-11-21).

The live defect: model 17 was served at 32,768 tokens, its ceiling was approved to max, and an
agent's `POST /api/models/17/load` was filed with the card "Unloads first: nothing (this model is
already loaded)" and "Context ceiling: max"; the approved load could only be refused "already
loaded". Now the load of the resident is a RELOAD when it changes the ceiling or the card (one
approval; the card says so), and REFUSED AT FILING when it changes nothing. The same check, made
for every gated model action:

* an unload of a model that is not loaded: refused at filing (the card said "unloading it changes
  nothing" and filed it);
* a delete of the loaded model: refused at filing (the delete refuses it);
* a capacity measurement of a model that is not resident: refused at filing (the card said "the
  measurement needs it loaded and will refuse" and filed it);
* a load of a weights-origin row: refused at filing (every management load refuses it).

The live app (`create_app`) over SQLite with the model service mocked (`approval_fixtures`): the
describers run for real against `LoadedModelState`.

MUTATION CONTROLS (each must turn this file red; run 2026-11-21, recorded in
0xcc/reviews/load_resident_reload_2026-11-21.md §5):
  A1 the describer's no-op refusal removed          -> test_a_load_that_changes_nothing_files_nothing
  A2 "Unloads first" reverted to "nothing (this model is already loaded)"
                                                    -> test_the_reload_card_*
  A3 the unload describer's residency refusal removed -> test_an_unload_of_a_model_not_loaded_*
  A4 the delete describer's loaded refusal removed  -> test_a_delete_of_the_loaded_model_*
  A5 refuse_unmeasurable() dropped from the card    -> test_a_measurement_of_a_model_not_resident_*
  A6 the SAE line reverted to "Attached SAEs it drops" -> test_the_card_says_attached_saes_stay
     (renamed test_no_card_says_the_unload_leaves_saes_attached on fix/unload-detaches-saes, when
     RR-1 was fixed; the named card lines are pinned in test_unload_detaches_saes_cards.py)
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from miserver.core.config import settings
from miserver.services import approval_service
from tests.unit.approval_fixtures import (  # noqa: F401
    AGENT,
    harness,
    make_gguf_model,
    make_model,
    rows,
)
from tests.unit.f28_fixtures import db  # noqa: F401 - fixture

LOADED_32K = {"context_length": 32_768, "gpu_indices": [0],
              "load_decisions": {"context": {"ceiling": 32_768, "ceiling_source": "server",
                                             "loaded": 32_768}}}


@pytest.fixture(autouse=True)
def _server_default():
    with patch.object(settings, "GGUF_CONTEXT_LENGTH", 32_768):
        yield


async def _set_ceiling(h, row, value):
    from miserver.db.models.model import Model

    async with h.db() as session:
        stored = await session.get(Model, row.id)
        stored.context_ceiling = value
        await session.commit()
        return stored


async def test_the_reload_card_says_it_unloads_and_reloads_at_the_new_ceiling(harness):  # noqa: F811
    h = harness
    h.holder["row"] = row = await make_gguf_model(h.db)
    h.make_resident(row, **LOADED_32K)
    r = h.client.post(f"/api/models/{row.id}/load", json={"context_ceiling": "max"}, headers=AGENT)
    assert r.status_code == 202, r.text
    summary = r.json()["data"]["summary"]
    assert summary["Unloads first"] == f"this model, {row.name} (id {row.id}): it is reloaded"
    assert not summary["Unloads first"].startswith("nothing")
    assert summary["Reloads"] == (
        "this model is loaded with a 32,768-token window (ceiling 32,768 tokens) on GPU 0; after "
        "approval it is unloaded (requests already running are given time to finish) and loaded "
        "again with max: as large a window as fits on the card, up to the model's declared "
        "length (named in this request)")
    assert summary["SAEs it detaches"] == "none"
    assert summary["Circuits it deactivates"] == "none"
    assert summary["Profiles it deactivates"] == "none"
    assert summary["Armed probes it disarms"] == "none"
    assert summary["Steering lock it releases"] == "none"
    # ...and the approved request runs the load with that ceiling (the service reloads it).
    data = r.json()["data"]
    assert h.client.post(f"/api/approvals/{data['approval_id']}/approve").status_code == 202
    await approval_service.wait_all()
    (after,) = await rows(h.db)
    assert after.status == "executed", (after.status, after.error)
    h.service.load_model.assert_awaited_once_with(row.id, gpu=None, lease_id=None,
                                                  context_ceiling=0)


async def test_the_live_sequence_ceiling_set_then_a_bare_load_is_one_reload(harness):  # noqa: F811
    h = harness
    row = await make_gguf_model(h.db)
    h.holder["row"] = row = await _set_ceiling(h, row, 0)
    h.make_resident(row, **LOADED_32K)
    r = h.client.post(f"/api/models/{row.id}/load", headers=AGENT)
    assert r.status_code == 202, r.text
    summary = r.json()["data"]["summary"]
    assert summary["Reloads"].endswith("loaded again with max: as large a window as fits on the "
                                       "card, up to the model's declared length (this model's "
                                       "setting)")
    assert summary["Context ceiling"].startswith("max:")


async def test_a_reload_on_another_card_says_so(harness):  # noqa: F811
    h = harness
    h.holder["row"] = row = await make_gguf_model(h.db)
    h.make_resident(row, **LOADED_32K)
    r = h.client.post(f"/api/models/{row.id}/load", json={"gpu": 1}, headers=AGENT)
    assert r.status_code == 202, r.text
    assert r.json()["data"]["summary"]["Reloads"].endswith("and loaded again on 1")


@pytest.mark.parametrize("body", [None, {"context_ceiling": 32_768}, {"gpu": "auto"},
                                  {"gpu": 0, "context_ceiling": 32_768}])
async def test_a_load_that_changes_nothing_files_nothing(harness, body):  # noqa: F811
    h = harness
    h.holder["row"] = row = await make_gguf_model(h.db)
    h.make_resident(row, **LOADED_32K)
    r = h.client.post(f"/api/models/{row.id}/load", json=body, headers=AGENT)
    assert r.status_code == 409, r.text
    error = r.json()["error"]
    assert error["code"] == "MODEL_ALREADY_LOADED"
    assert "would change nothing" in error["message"]
    assert "would change nothing" in error["details"]["technical_message"]
    assert "context-ceiling" in error["details"]["fix"]
    assert await rows(h.db) == []
    assert h.service.load_model.await_count == 0


async def test_a_load_of_another_model_is_filed_as_before_and_names_what_it_drops(harness):  # noqa: F811
    from miserver.services.probe_runtime import ProbeRuntimeState

    h = harness
    resident = await make_model(h.db, name="served-now")
    h.holder["row"] = row = await make_gguf_model(h.db)
    h.make_resident(resident)
    armed = [SimpleNamespace(probe_id="p1", name="ssn-detector")]
    with patch.object(ProbeRuntimeState, "armed", lambda self: armed):
        r = h.client.post(f"/api/models/{row.id}/load", headers=AGENT)
    assert r.status_code == 202, r.text
    summary = r.json()["data"]["summary"]
    assert summary["Unloads first"] == f"served-now (id {resident.id})"
    assert "Reloads" not in summary
    assert summary["Armed probes it disarms"] == "ssn-detector"


async def test_nothing_loaded_is_unchanged(harness):  # noqa: F811
    h = harness
    h.holder["row"] = row = await make_gguf_model(h.db)
    summary = h.client.post(f"/api/models/{row.id}/load", headers=AGENT).json()["data"]["summary"]
    assert summary["Unloads first"] == "nothing (no model is loaded)"
    assert "SAEs it detaches" not in summary and "Reloads" not in summary


async def test_no_card_says_the_unload_leaves_saes_attached(harness):  # noqa: F811
    """RR-1 is fixed (2026-11-21): the unload detaches them, so the interim "the unload does NOT
    detach them" line is gone, as is the older false "Attached SAEs it drops". The named lines
    are pinned in `test_unload_detaches_saes_cards.py`."""
    from miserver.services.sae_service import AttachedSAEState

    h = harness
    h.holder["row"] = row = await make_gguf_model(h.db)
    h.make_resident(row, **LOADED_32K)
    entry = SimpleNamespace(sae_id="sae-x", layer=12,
                            sae=SimpleNamespace(is_steering_enabled=True))
    with patch.object(AttachedSAEState, "entries", lambda self: [entry]):
        load = h.client.post(f"/api/models/{row.id}/load", json={"context_ceiling": "max"},
                             headers=AGENT).json()["data"]["summary"]
        unload = h.client.post(f"/api/models/{row.id}/unload",
                               headers=AGENT).json()["data"]["summary"]
    for summary in (load, unload):
        assert "Attached SAEs it drops" not in summary and "Attached SAEs" not in summary
        assert "does NOT detach" not in summary["SAEs it detaches"]
        assert summary["SAEs it detaches"].startswith("sae-x (id sae-x) at layer 12; detached ")


async def test_a_reload_of_a_locked_model_says_it_releases_the_lock(harness):  # noqa: F811
    from miserver.db.models.model import Model

    h = harness
    row = await make_gguf_model(h.db)
    async with h.db() as session:
        stored = await session.get(Model, row.id)
        stored.locked = True
        await session.commit()
    h.holder["row"] = stored
    h.make_resident(stored, **LOADED_32K)
    summary = h.client.post(f"/api/models/{row.id}/load", json={"context_ceiling": 65_536},
                            headers=AGENT).json()["data"]["summary"]
    assert summary["Steering lock it releases"] == "yes (the model is locked for steering now)"


# --- the other gated actions: refused at filing when they could only fail ---------------------


async def test_an_unload_of_a_model_not_loaded_files_nothing(harness):  # noqa: F811
    h = harness
    h.holder["row"] = row = await make_model(h.db)
    r = h.client.post(f"/api/models/{row.id}/unload", headers=AGENT)
    assert r.status_code == 400, r.text
    error = r.json()["error"]
    assert error["code"] == "MODEL_NOT_LOADED" and "fix" in error["details"]
    assert await rows(h.db) == []
    # ...and the loaded one is still filed.
    h.make_resident(row)
    r = h.client.post(f"/api/models/{row.id}/unload", headers=AGENT)
    assert r.status_code == 202, r.text
    assert r.json()["data"]["summary"]["Loaded now"] == "yes"


async def test_a_delete_of_the_loaded_model_files_nothing(harness):  # noqa: F811
    h = harness
    h.holder["row"] = row = await make_model(h.db)
    h.make_resident(row)
    r = h.client.delete(f"/api/models/{row.id}", headers=AGENT)
    assert r.status_code == 409, r.text
    assert r.json()["error"]["code"] == "MODEL_BUSY"
    assert await rows(h.db) == []
    assert h.service.delete_model.await_count == 0


async def test_a_measurement_of_a_model_not_resident_files_nothing(harness):  # noqa: F811
    h = harness
    h.holder["row"] = row = await make_model(h.db)
    r = h.client.post(f"/api/models/{row.id}/probe-capacity/measure", headers=AGENT)
    assert r.status_code == 409, r.text
    error = r.json()["error"]
    assert error["code"] == "PROBE_CAPACITY_REFUSED"
    assert error["details"]["reason"] == "not_resident"
    assert await rows(h.db) == []
    h.make_resident(row)
    r = h.client.post(f"/api/models/{row.id}/probe-capacity/measure", headers=AGENT)
    assert r.status_code == 202, r.text


async def test_a_load_of_a_weights_origin_row_files_nothing(harness):  # noqa: F811
    from miserver.db.models.model import Model

    h = harness
    row = await make_model(h.db)
    async with h.db() as session:
        stored = await session.get(Model, row.id)
        stored.weights_origin = "checkpoint"
        await session.commit()
    h.holder["row"] = stored
    r = h.client.post(f"/api/models/{row.id}/load", headers=AGENT)
    assert r.status_code == 409, r.text
    assert r.json()["error"]["code"] == "LOAD_THROUGH_WEIGHTS_API"
    assert await rows(h.db) == []


async def test_the_context_write_card_says_one_load_applies_it_now(harness):  # noqa: F811
    h = harness
    h.holder["row"] = row = await make_gguf_model(h.db)
    h.make_resident(row, **LOADED_32K)
    summary = h.client.put(f"/api/models/{row.id}/context-ceiling", json={"context_ceiling": "max"},
                           headers=AGENT).json()["data"]["summary"]
    assert summary["Loaded now"] == (
        "yes, with a 32,768-token window (ceiling 32,768 tokens); to apply the new ceiling now, "
        f"load it again (POST /api/models/{row.id}/load): that one approval reloads it")

