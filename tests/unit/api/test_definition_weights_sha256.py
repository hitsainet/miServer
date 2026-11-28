"""ENH-011 X-2 (miServer half, #12): a circuit or cluster definition's `saes[]` / `sae` entry carries the SAE weights
hash miWorkbench wrote (`weights_sha256`, sha256 of the weights file, additive and optional), and miServer keeps it.

What is pinned here:

* **AC-1, the round trip.** A circuit and a cluster carrying the field go in through the LIVE import routes and come
  back out of the live export routes unchanged, byte for byte as JSON. The routes store the raw document, so this
  half held before the mirror knew the field; the mirror is where it was lost: a circuit served as a per-layer
  slice is re-projected from the PARSED definition (`CircuitService.to_layer_slice`), and the slice's `sae` block is
  the mirror's `DefinitionSAERef.model_dump`, which dropped the hash. The slice goes in through the live cluster
  import route too, and its re-export must still name the hash.
* **AC-2, the bind.** When both hashes are known (the definition's, and the attached SAE's, which the loader reads
  from the weights file it loaded), a difference is reported: a `warn` verdict in the circuit's `per_sae_warnings`,
  a line in the cluster's import warnings. It is not a refusal in this release (E11-6): the circuit stays
  serveable and the cluster still binds. An equal hash, or one side unknown, says nothing.

Controls (record: `0xcc/reviews/ENH-011_X-2_controls_2026-11-28.md`): C1 drops the mirror field, C2 drops the
bind comparison, C3 makes it a refusal, C4 drops the loader's hash.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from miserver.api.schemas.circuit import CircuitDefinitionV1
from miserver.api.schemas.cluster import ClusterDefinitionV1, DefinitionSAERef
from miserver.services.circuit_service import CircuitService
from miserver.services.cluster_service import ClusterService
from miserver.services.sae_service import AttachedSAEState
from tests.unit.api.test_probe_event_filters import client  # noqa: F401 - fixture
from tests.unit.f28_fixtures import db  # noqa: F401 - fixture

REPO = Path(__file__).resolve().parents[3]
FIXTURES = REPO / "tests" / "fixtures"

H_A = hashlib.sha256(b"sae weights, run A").hexdigest()
H_B = hashlib.sha256(b"sae weights, run B (same shape)").hexdigest()


def _doc(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text())


def _circuit_with_hash(digest: str = H_A) -> dict:
    doc = _doc("real_circuit_definition.json")
    for ref in doc["saes"]:
        ref["weights_sha256"] = digest
    return doc


def _cluster_with_hash(digest: str = H_A) -> dict:
    doc = _doc("miworkbench_export.cluster.json")
    doc["sae"]["weights_sha256"] = digest
    return doc


@pytest.fixture(autouse=True)
def _registry():
    state = AttachedSAEState()
    state.reset_for_tests()
    yield state
    state.reset_for_tests()


def _attach(sae_id: str, layer: int, d_sae: int, weights_sha256: str | None):
    sae = MagicMock()
    sae.d_sae = d_sae
    sae.weights_sha256 = weights_sha256
    AttachedSAEState().set(sae, sae_id, layer, None)
    return sae


# ----------------------------------------------------------------------------- AC-1: the round trip


def test_a_circuit_carrying_the_hash_reexports_it_through_the_live_routes(client, monkeypatch, tmp_path):  # noqa: F811
    from miserver.core.config import settings

    monkeypatch.setattr(settings, "SAE_CACHE_DIR", str(tmp_path))
    doc = _circuit_with_hash()
    r = client.post("/api/circuits/import", json=doc)
    assert r.status_code == 200 and r.json()["success"] is True, r.text
    circuit_id = r.json()["data"]["id"]
    exported = client.get(f"/api/circuits/{circuit_id}/export")
    assert exported.status_code == 200, exported.text
    assert exported.json() == doc
    assert [s["weights_sha256"] for s in exported.json()["saes"]] == [H_A] * len(doc["saes"])


def test_a_cluster_carrying_the_hash_reexports_it_through_the_live_routes(client, monkeypatch, tmp_path):  # noqa: F811
    from miserver.core.config import settings

    monkeypatch.setattr(settings, "SAE_CACHE_DIR", str(tmp_path))
    doc = _cluster_with_hash()
    r = client.post("/api/clusters/import", json=doc)
    assert r.status_code == 200 and r.json()["success"] is True, r.text
    (item,) = r.json()["data"]["results"]
    exported = client.get(f"/api/clusters/{item['profile_id']}/export")
    assert exported.status_code == 200, exported.text
    assert exported.json() == doc
    assert exported.json()["sae"]["weights_sha256"] == H_A


def test_a_circuit_served_as_a_slice_keeps_the_hash_through_import_and_reexport(client, monkeypatch, tmp_path):  # noqa: F811
    """The loss path: the slice is re-projected from the PARSED circuit, so its `sae` block is the mirror's dump."""
    from miserver.core.config import settings

    monkeypatch.setattr(settings, "SAE_CACHE_DIR", str(tmp_path))
    definition = CircuitDefinitionV1.model_validate(_circuit_with_hash())
    layer = definition.saes[0].layer
    slice_doc = CircuitService(repository=MagicMock()).to_layer_slice(definition, layer, circuit_rung_value=2)
    assert slice_doc["sae"]["weights_sha256"] == H_A
    r = client.post("/api/clusters/import", json=slice_doc)
    assert r.status_code == 200 and r.json()["success"] is True, r.text
    (item,) = r.json()["data"]["results"]
    exported = client.get(f"/api/clusters/{item['profile_id']}/export").json()
    assert exported["sae"]["weights_sha256"] == H_A
    assert exported == json.loads(json.dumps(slice_doc))


@pytest.mark.parametrize(
    "model, make",
    [(CircuitDefinitionV1, _circuit_with_hash), (ClusterDefinitionV1, _cluster_with_hash)],
    ids=["circuit", "cluster"],
)
def test_the_hand_mirror_keeps_the_hash(model, make):
    doc = make()
    dumped = model.model_validate(doc).model_dump(mode="json", exclude_none=True)
    refs = dumped["saes"] if "saes" in dumped else [dumped["sae"]]
    assert [r["weights_sha256"] for r in refs] == [H_A] * len(refs)


def test_the_mirror_refuses_a_hash_that_is_not_lowercase_hex():
    """The vendored pattern `^[0-9a-f]{64}$`, as the mirror validates it (the schema-sync tests compare the rest)."""
    for bad in ("A" * 64, "0" * 63, "g" * 64):
        with pytest.raises(ValueError):
            DefinitionSAERef(layer=1, weights_sha256=bad)
    assert DefinitionSAERef(layer=1).weights_sha256 is None


# ----------------------------------------------------------------------------- AC-2: the bind


def _two_layer_circuit(
    h_up: str | None, h_down: str | None, *, layers: tuple[int, int] = (10, 13), n_features: int = 8192
) -> CircuitDefinitionV1:
    up, down = layers
    saes = [
        {"layer": up, "n_features": n_features, "mistudio_sae_id": f"sae-{up}"},
        {"layer": down, "n_features": n_features, "mistudio_sae_id": f"sae-{down}"},
    ]
    for ref, digest in zip(saes, (h_up, h_down), strict=True):
        if digest is not None:
            ref["weights_sha256"] = digest
    return CircuitDefinitionV1.model_validate(
        {
            "kind": "mistudio.circuit-definition",
            "schema_version": "1",
            "name": "x2",
            "saes": saes,
            "members": [
                {"layer": up, "feature": {"feature_idx": 1, "strength": 40.0}},
                {"layer": down, "feature": {"feature_idx": 2, "strength": 30.0}},
            ],
            "edges": [{"up": {"layer": up, "feature_idx": 1}, "down": {"layer": down, "feature_idx": 2}, "rung": 2}],
        }
    )


def test_a_circuit_bound_to_an_sae_with_other_weights_warns_and_stays_serveable():
    _attach("sae-10", 10, 8192, H_B)  # same id, same shape, other weights
    _attach("sae-13", 13, 8192, H_A)
    verdicts = CircuitService(repository=MagicMock()).assess_compatibility(_two_layer_circuit(H_A, H_A))
    by_layer = {v["layer"]: v for v in verdicts}
    assert by_layer[10]["verdict"] == "warn", by_layer[10]
    assert H_A[:12] in by_layer[10]["reason"] and H_B[:12] in by_layer[10]["reason"]
    assert "weights" in by_layer[10]["reason"]
    assert by_layer[13] == {"layer": 13, "sae_id": "sae-13", "verdict": "bind"}
    assert len(verdicts) == 2


@pytest.mark.parametrize(
    "declared, attached",
    [(H_A, H_A), (None, H_A), (H_A, None), (None, None)],
    ids=["equal", "definition-silent", "sae-unknown", "both-unknown"],
)
def test_an_equal_or_unknown_hash_binds_silently(declared, attached):
    _attach("sae-10", 10, 8192, attached)
    _attach("sae-13", 13, 8192, attached)
    verdicts = CircuitService(repository=MagicMock()).assess_compatibility(_two_layer_circuit(declared, declared))
    assert verdicts == [
        {"layer": 10, "sae_id": "sae-10", "verdict": "bind"},
        {"layer": 13, "sae_id": "sae-13", "verdict": "bind"},
    ]


def test_a_circuit_imported_against_other_weights_is_stored_serveable_with_the_warning(client, monkeypatch, tmp_path):  # noqa: F811
    """The wiring through the live import route: the verdict reaches the stored row, and it still serves."""
    from miserver.core.config import settings

    monkeypatch.setattr(settings, "SAE_CACHE_DIR", str(tmp_path))
    doc = _circuit_with_hash(H_A)
    for ref in doc["saes"]:
        _attach(ref.get("mistudio_sae_id") or "sae", ref["layer"], ref["n_features"], H_B)
    r = client.post("/api/circuits/import", json=doc)
    assert r.status_code == 200 and r.json()["success"] is True, r.text
    row = r.json()["data"]
    assert row["serveable"] is True
    warns = row["per_sae_warnings"]
    assert len(warns) == len(doc["saes"])
    assert {w["verdict"] for w in warns} == {"warn"}
    assert all(H_B[:12] in w["reason"] for w in warns)


def _cluster_definition(digest: str | None) -> ClusterDefinitionV1:
    doc = _doc("miworkbench_export.cluster.json")
    doc["sae"].pop("weights_sha256", None)
    if digest is not None:
        doc["sae"]["weights_sha256"] = digest
    return ClusterDefinitionV1.model_validate(doc)


def _cluster_service() -> ClusterService:
    return ClusterService(repository=MagicMock(), profile_service=MagicMock(), sae_service=MagicMock())


def test_a_cluster_bound_to_an_sae_with_other_weights_warns_and_still_binds():
    definition = _cluster_definition(H_A)
    _attach("sae-x", int(definition.sae.layer), int(definition.sae.n_features), H_B)
    sae_id, warnings = _cluster_service()._assess_compatibility(definition)
    assert sae_id == "sae-x"
    hash_lines = [w for w in warnings if "weights" in w]
    assert len(hash_lines) == 1, warnings
    assert H_A[:12] in hash_lines[0] and H_B[:12] in hash_lines[0]


@pytest.mark.parametrize(
    "declared, attached",
    [(H_A, H_A), (None, H_A), (H_A, None)],
    ids=["equal", "definition-silent", "sae-unknown"],
)
def test_a_cluster_with_an_equal_or_unknown_hash_binds_without_a_hash_warning(declared, attached):
    definition = _cluster_definition(declared)
    _attach("sae-x", int(definition.sae.layer), int(definition.sae.n_features), attached)
    sae_id, warnings = _cluster_service()._assess_compatibility(definition)
    assert sae_id == "sae-x"
    assert [w for w in warnings if "weights" in w] == []


# ----------------------------------------------------------------------------- the attached SAE's hash


def test_the_loader_records_the_sha256_of_the_weights_file_it_loaded(tmp_path):
    """The attached side of the comparison: the hash of the very file the loader read (miWorkbench hashes the
    same weights file at export), so `assess_compatibility` compares like with like."""
    import torch
    from safetensors.torch import save_file

    from miserver.ml.sae_loader import SAELoader

    d_in, d_sae = 8, 16
    sae_dir = tmp_path / "sae"
    sae_dir.mkdir()
    (sae_dir / "cfg.json").write_text(
        json.dumps({"d_in": d_in, "d_sae": d_sae, "model_name": "t/m", "hook_name": "blocks.1.hook_resid_post",
                    "hook_layer": 1, "dtype": "float32"})
    )
    save_file(
        {"W_enc": torch.randn(d_in, d_sae), "b_enc": torch.randn(d_sae), "W_dec": torch.randn(d_sae, d_in),
         "b_dec": torch.randn(d_in)},
        str(sae_dir / "sae_weights.safetensors"),
    )
    expected = hashlib.sha256((sae_dir / "sae_weights.safetensors").read_bytes()).hexdigest()
    loaded = SAELoader().load(sae_dir)
    assert loaded.weights_sha256 == expected

    # ... and that is the value the bind compares: the loaded SAE attached at layer 1.
    AttachedSAEState().set(loaded, "sae-1", 1, None)
    definition = _two_layer_circuit(H_B, None, layers=(1, 2), n_features=d_sae)
    verdicts = CircuitService(repository=MagicMock()).assess_compatibility(definition)
    first = [v for v in verdicts if v["layer"] == 1]
    assert len(first) == 1 and first[0]["verdict"] == "warn", verdicts
    assert expected[:12] in first[0]["reason"] and H_B[:12] in first[0]["reason"]
