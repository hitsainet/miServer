"""Exercises for the entries P9 wave 1 registered in miServer (042 FTASKS 2.3; suite
`specs/ENH-009-rebrand/p9/exercises-2026-11-14.md` §2): DR-58 (the document field
`mistudio_version`, GAP-03) and DR-68 (the published component name `MillmExtension`, GAP-16).

Each drives the real reader or generator, not the source text, and names the line it guards.
"""

from __future__ import annotations

import json
from pathlib import Path

import jsonschema

from miserver.api.schemas.cluster import ClusterDefinitionV1
from miserver.api.schemas.scoring_contract import CONTRACT_FILENAME, scoring_api_contract
from miserver.api.schemas.weights_contract import render
from tests.unit.api.test_probe_event_filters import client  # noqa: F401 - fixture
from tests.unit.f28_fixtures import db  # noqa: F401 - fixture

REPO = Path(__file__).resolve().parents[4]
FIXTURES = REPO / "tests" / "fixtures"
SCHEMAS = REPO / "docs" / "schemas"
FIELD = "mistudio_version"


def test_cluster_import_keeps_mistudio_version(client, monkeypatch, tmp_path):  # noqa: F811  # p9: keep, registered exercise id
    """DR-58: miWorkbench writes `provenance.mistudio_version` into every probe, cluster and circuit
    definition. miServer's cluster mirror reads it (`miserver/api/schemas/cluster.py`,
    `DefinitionProvenance`), a cluster imported through the live route re-exports it unchanged, and
    the vendored probe schema, whose `Provenance` allows no other key, refuses the field renamed."""
    from miserver.core.config import settings

    monkeypatch.setattr(settings, "SAE_CACHE_DIR", str(tmp_path))  # the real SAE service, no /app
    raw = json.loads((FIXTURES / "miworkbench_export.cluster.json").read_text())
    version = raw["provenance"][FIELD]
    assert version, "the fixture is a real export that carries the field"

    # The reader: the mirror model validates the field by this name.
    assert ClusterDefinitionV1.model_validate(raw).provenance.model_dump()[FIELD] == version

    # The mirror's generated schema names it, as the frozen contract miWorkbench publishes does.
    generated = ClusterDefinitionV1.model_json_schema(ref_template="#/$defs/{model}")
    assert FIELD in generated["$defs"]["DefinitionProvenance"]["properties"]
    vendored = json.loads((SCHEMAS / "cluster-definition-v1.json").read_text())
    assert FIELD in vendored["$defs"]["DefinitionProvenance"]["properties"]

    # The round trip through the live routes.
    r = client.post("/api/clusters/import", json=raw)
    assert r.status_code == 200 and r.json()["success"] is True, r.text
    cluster_id = r.json()["data"]["results"][0]["profile_id"]
    exported = client.get(f"/api/clusters/{cluster_id}/export").json()
    assert exported["provenance"][FIELD] == version

    # A renamed key is not the same document: the probe contract's Provenance refuses it.
    probe_schema = json.loads((SCHEMAS / "probe-definition-v1.json").read_text())
    provenance = probe_schema["$defs"]["Provenance"]
    assert provenance["additionalProperties"] is False and FIELD in provenance["properties"]
    doc = {FIELD: "0.5.0"}
    jsonschema.validate(doc, {**provenance, "$defs": probe_schema["$defs"]})
    renamed = {"miworkbench_version": "0.5.0"}
    try:
        jsonschema.validate(renamed, {**provenance, "$defs": probe_schema["$defs"]})
    except jsonschema.ValidationError:
        pass
    else:
        raise AssertionError("the probe contract accepted a renamed provenance field")


def test_scoring_contract_names_millm_extension():  # p9: keep, registered exercise id
    """DR-68: the scoring contract miForge vendors byte for byte names the response extension's
    component after the CLASS `MillmExtension` (`miserver/api/schemas/miserver_extension.py`). The
    module's name reaches no reader; renaming the class renames a published component."""
    doc = scoring_api_contract()
    assert "MillmExtension" in doc["$defs"], sorted(doc["$defs"])
    text = render(doc)
    assert '"$ref": "#/$defs/MillmExtension"' in text
    assert (SCHEMAS / CONTRACT_FILENAME).read_text(encoding="utf-8") == text, (
        "the committed scoring contract differs from the generated one")
