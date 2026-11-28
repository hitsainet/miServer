"""The export routes reference the document schemas they answer (044 FR-044.23, SH8).

`GET /api/circuits/{id}/export` and `GET /api/clusters/{id}/export` return the stored document losslessly, with
no envelope. The contract does not re-model it: each entry is a `$ref` to the `$id` of the vendored schema, and
`external` lists each referenced file with its sha256, so a consumer resolves the reference offline.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from miserver.api.schemas import contract_tools as tools
from miserver.api.schemas.circuit_answers import CIRCUIT_DEFINITION_ID
from miserver.api.schemas.cluster_answers import CLUSTER_DEFINITION_ID
from tests.support import answer_validator
from tests.unit.api.test_probe_event_filters import client  # noqa: F401 - fixture
from tests.unit.f28_fixtures import db  # noqa: F401 - fixture

ROOT = Path(__file__).resolve().parents[3]
EXPORTS = {
    "GET /api/circuits/{circuit_id}/export": ("circuit-definition-v1.json", CIRCUIT_DEFINITION_ID),
    "GET /api/clusters/{cluster_id}/export": ("cluster-definition-v1.json", CLUSTER_DEFINITION_ID),
}


@pytest.mark.parametrize("key", sorted(EXPORTS))
def test_each_export_references_its_vendored_schema(key: str):
    contract = answer_validator._contract()
    file, sid = EXPORTS[key]
    vendored = json.loads((ROOT / "docs" / "schemas" / file).read_text())
    assert contract["responses"][key]["200"] == {"$ref": sid}
    assert vendored["$id"] == sid


def _mismatches(contract: dict, root: Path) -> list[str]:
    """Each `external` row whose file under `root` is missing, names another `$id`, or hashes differently."""
    out = []
    for sid, row in contract["external"].items():
        path = root / row["file"]
        if not path.is_file():
            out.append(f"{row['file']}: missing")
            continue
        raw = path.read_bytes()
        if json.loads(raw).get("$id") != sid:
            out.append(f"{row['file']}: $id is not {sid}")
        if hashlib.sha256(raw).hexdigest() != row["sha256"]:
            out.append(f"{row['file']}: sha256 differs")
    return out


def test_external_lists_every_referenced_file_with_its_sha256():
    contract = answer_validator._contract()
    referenced = {
        s["$ref"]
        for s in tools._walk(contract)
        if isinstance(s.get("$ref"), str) and s["$ref"].startswith("https://")
    }
    assert set(contract["external"]) == referenced
    assert _mismatches(contract, ROOT) == []


def test_a_changed_vendored_byte_is_caught(tmp_path):
    """FR-044.23 NC: one byte of the vendored cluster schema changed in a temporary tree, and the same
    check the test above runs reports that file."""
    contract = answer_validator._contract()
    copy = tmp_path / "docs" / "schemas"
    copy.mkdir(parents=True)
    for row in contract["external"].values():
        (tmp_path / row["file"]).write_bytes((ROOT / row["file"]).read_bytes())
    target = tmp_path / contract["external"][CLUSTER_DEFINITION_ID]["file"]
    target.write_bytes(target.read_bytes().replace(b'"description"', b'"Description"', 1))
    assert _mismatches(contract, tmp_path) == [
        f"{contract['external'][CLUSTER_DEFINITION_ID]['file']}: sha256 differs"
    ]


def _doc(name: str) -> dict:
    return json.loads((ROOT / "tests" / "fixtures" / name).read_text())


def test_real_exports_validate_against_the_vendored_schemas(client, monkeypatch, tmp_path):  # noqa: F811
    from miserver.core.config import settings

    monkeypatch.setattr(settings, "SAE_CACHE_DIR", str(tmp_path))
    circuit = client.post("/api/circuits/import", json=_doc("real_circuit_definition.json"))
    assert circuit.status_code == 200, circuit.text
    cluster = client.post("/api/clusters/import", json=_doc("miworkbench_export.cluster.json"))
    assert cluster.status_code == 200, cluster.text
    circuit_id = circuit.json()["data"]["id"]
    cluster_id = cluster.json()["data"]["results"][0]["profile_id"]
    for path, sid in (
        (f"/api/circuits/{circuit_id}/export", CIRCUIT_DEFINITION_ID),
        (f"/api/clusters/{cluster_id}/export", CLUSTER_DEFINITION_ID),
    ):
        response = client.get(path)  # the answer validator checks it against the contract too
        assert response.status_code == 200, response.text
        validator = Draft202012Validator({"$ref": sid}, registry=answer_validator._registry())
        assert list(validator.iter_errors(response.json())) == [], path
