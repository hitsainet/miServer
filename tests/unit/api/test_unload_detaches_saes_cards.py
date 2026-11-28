"""The cards that unload a model name what the unload detaches and deactivates (RR-1, 2026-11-21).

An unload (and a load that swaps the model out, and a reload of the resident) detaches every
attached SAE and deactivates every active circuit and profile before the weights go
(`unload_teardown`). Its approval card names each one, by display name AND id, and says that the
load that follows attaches and activates nothing again. The card reads the same function the unload
acts on (`unload_teardown.read_dependents`), so the last test runs the real teardown against the
same database and requires the names it reports to be the names the card listed.

The live app (`create_app`) over SQLite, the model service mocked (`approval_fixtures`); REAL SAEs
hooked on a tiny real model in the real `AttachedSAEState`. Names differ from ids, so a card printing
one for the other fails.

MUTATION CONTROLS (each must turn this file red; recorded in
0xcc/reviews/unload_detaches_saes_2026-11-21.md):
  C1 the SAE line drops the names (ids only)         -> every card test
  C2 the circuit line reads "none"                    -> every card test
  C3 the reload card says the swap's words             -> test_the_reload_card_*
"""

from __future__ import annotations

from typing import Any

from miserver.db.models.circuit import Circuit
from miserver.db.models.profile import Profile
from miserver.db.models.sae import SAE, SAEStatus
from tests.unit.approval_fixtures import AGENT, harness, make_model  # noqa: F401
from tests.unit.f25_fixtures import word_model
from tests.unit.f28_fixtures import SAE_A, SAE_B, attach, clean_state, db  # noqa: F401

NAMES = {SAE_A: "alpha-resid-l0-8k", SAE_B: "beta-resid-l1-8k"}
SAE_LINE = (f"{NAMES[SAE_A]} (id {SAE_A}) at layer 0, {NAMES[SAE_B]} (id {SAE_B}) at layer 1; "
            "detached before the weights are released (hooks removed, GPU memory freed); ")
ATTACH_HINT = " (POST /api/saes/{sae_id}/attach attaches one)"
CIRCUIT_WORDS = "live circuit (id circ_live)"
CLUSTER_WORDS = "calm cluster (id prof_calm, a cluster)"


async def _seed(factory: Any) -> None:
    async with factory() as session:
        for sae_id in (SAE_A, SAE_B):
            session.add(SAE(id=sae_id, repository_id=f"vendor/{sae_id}", name=NAMES[sae_id],
                            d_in=16, d_sae=8, cache_path=f"/data/saes/{sae_id}",
                            status=SAEStatus.ATTACHED))
        session.add(Circuit(id="circ_live", name="live circuit", description=None, rung=2,
                            edge_count=1, layers=[1], serveable=True, is_active=True,
                            serving_mode="full", intensity=1.0, per_sae_warnings=[],
                            circuit_meta={"budget": {}}))
        session.add(Circuit(id="circ_idle", name="idle circuit", description=None, rung=2,
                            edge_count=1, layers=[0], serveable=True, is_active=False,
                            serving_mode=None, intensity=1.0, per_sae_warnings=[],
                            circuit_meta={"budget": {}}))
        session.add(Profile(id="prof_calm", name="calm cluster", steering={"3": 5.0},
                            is_active=True, intensity=1.0, sae_id=SAE_A, layer=0,
                            source_kind="cluster"))
        session.add(Profile(id="prof_idle", name="idle profile", steering={"1": 2.0},
                            is_active=False, intensity=1.0))
        await session.commit()


async def _served(h: Any, **fields: Any) -> Any:
    row = await make_model(h.db, name="served-now")
    h.make_resident(row, gpu_indices=[0], **fields)
    attach(word_model(), [(SAE_A, 0, 11), (SAE_B, 1, 12)])
    await _seed(h.db)
    return row


def _summary(h: Any, path: str, **kwargs: Any) -> dict:
    r = h.client.post(path, headers=AGENT, **kwargs)
    assert r.status_code == 202, r.text
    return r.json()["data"]["summary"]


async def test_the_unload_card_names_what_it_detaches_and_deactivates(harness, clean_state):  # noqa: F811
    h = harness
    h.holder["row"] = row = await _served(h)
    summary = _summary(h, f"/api/models/{row.id}/unload")
    after = "a later load does not attach or activate them again"
    assert summary["SAEs it detaches"] == SAE_LINE + after + ATTACH_HINT
    assert summary["Circuits it deactivates"] == f"{CIRCUIT_WORDS}; {after}"
    assert summary["Profiles it deactivates"] == f"{CLUSTER_WORDS}; {after}"
    # Inactive rows are not named: the unload does not touch them.
    assert "idle" not in str(summary)


async def test_the_swap_card_says_nothing_comes_across(harness, clean_state):  # noqa: F811
    h = harness
    await _served(h)
    h.holder["row"] = target = await make_model(h.db, name="next-model")
    summary = _summary(h, f"/api/models/{target.id}/load")
    after = f"they are not attached to or activated on next-model (id {target.id})"
    assert summary["Unloads first"].startswith("served-now")
    assert summary["SAEs it detaches"] == SAE_LINE + after + ATTACH_HINT
    assert summary["Circuits it deactivates"] == f"{CIRCUIT_WORDS}; {after}"
    assert summary["Profiles it deactivates"] == f"{CLUSTER_WORDS}; {after}"


async def test_the_reload_card_says_the_reload_attaches_nothing_again(harness, clean_state):  # noqa: F811
    """A reload of the SAME model leaves them detached (decided 2026-11-21) and says so."""
    h = harness
    h.holder["row"] = row = await _served(h)
    summary = _summary(h, f"/api/models/{row.id}/load", json={"gpu": 1})
    assert summary["Unloads first"] == f"this model, served-now (id {row.id}): it is reloaded"
    after = "the reload does not attach or activate them again"
    assert summary["SAEs it detaches"] == SAE_LINE + after + ATTACH_HINT
    assert summary["Circuits it deactivates"] == f"{CIRCUIT_WORDS}; {after}"
    assert summary["Profiles it deactivates"] == f"{CLUSTER_WORDS}; {after}"


async def test_what_the_card_names_is_what_the_unload_takes(harness, clean_state):  # noqa: F811
    """The card and the teardown read one function; run the real teardown and compare."""
    from miserver.services import unload_teardown

    h = harness
    h.holder["row"] = row = await _served(h)
    summary = _summary(h, f"/api/models/{row.id}/unload")
    report = await unload_teardown.release_dependents(row.id)
    assert report.failures == []
    named = summary["SAEs it detaches"].split("; detached ")[0]
    assert named == ", ".join(report.detached_saes)
    assert summary["Circuits it deactivates"].split("; ")[0] == ", ".join(
        report.deactivated_circuits)
    assert summary["Profiles it deactivates"].split("; ")[0] == ", ".join(
        report.deactivated_profiles)


async def test_nothing_attached_says_none(harness, clean_state):  # noqa: F811
    h = harness
    h.holder["row"] = row = await make_model(h.db, name="served-now")
    h.make_resident(row)
    summary = _summary(h, f"/api/models/{row.id}/unload")
    for key in ("SAEs it detaches", "Circuits it deactivates", "Profiles it deactivates"):
        assert summary[key] == "none"
