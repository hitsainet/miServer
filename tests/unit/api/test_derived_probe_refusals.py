"""A malformed `provenance.derived_from` is refused at import; both legal unset forms import
(043 FR-043.15, FR-043.17; D43-12).

R1 review fix (OR1-043-2), 2026-11-16, adopted from the review draft. The sync test proves the mirror's
`Provenance` class has a `derived_from` field, not that `ProbeDefinitionV1` uses the class: reverting
`provenance` to `dict[str, Any]`, typing `derived_from` as a plain dict, or dropping either hash pattern
all stayed green (review mutations R16, R13, R17, R18). Through the real `POST /api/probes/import`:
every malformed form answers 422 located under `provenance.derived_from` and stores nothing; absent and
`null` both answer 200 with the summary reading `None`; a well-formed lineage is shown verbatim.
"""
from __future__ import annotations

import pytest
from tests.unit.api.test_derived_probe_lineage import DERIVED_FROM, client  # noqa: F401
from tests.unit.services.test_probe_arming import definition

BAD = {
    "patch_not_hex": {**DERIVED_FROM, "patch_sha256": "B" * 64},
    "parent_digest_not_hex": {**DERIVED_FROM, "parent_detector_digest": "z" * 64},
    "run_not_hex": {**DERIVED_FROM, "verdict": {"value": "improved", "run_sha256": "x"}},
    "unknown_verdict": {**DERIVED_FROM, "verdict": {"value": "great", "run_sha256": "c" * 64}},
    "no_provider": {k: v for k, v in DERIVED_FROM.items() if k != "provider"},
    "no_verdict": {k: v for k, v in DERIVED_FROM.items() if k != "verdict"},
    "not_an_object": "pr_parent",
}


def _doc(name, **prov):
    d = definition()
    d["name"] = name
    d["provenance"] = {"run_id": "pmr_x", **prov}
    return d


@pytest.mark.parametrize("case", list(BAD))
async def test_a_malformed_derived_from_is_refused(client, case):  # noqa: F811
    http, _ = client
    r = http.post("/api/probes/import", json=_doc(f"bad-{case}", derived_from=BAD[case]))
    assert r.status_code == 422, r.text
    locs = [e["loc"][:2] for e in r.json()["error"]["details"]["errors"]]
    assert ["provenance", "derived_from"] in locs, locs
    listed = http.get("/api/probes").json()["data"]
    assert [p for p in listed if p["name"] == f"bad-{case}"] == [], "a refused document was stored"


@pytest.mark.parametrize("form", ["absent", "null"])
async def test_both_unset_forms_import(client, form):  # noqa: F811
    http, _ = client
    prov = {} if form == "absent" else {"derived_from": None}
    r = http.post("/api/probes/import", json=_doc(f"unset-{form}", **prov))
    assert r.status_code == 200, r.text
    assert http.get(f"/api/probes/{r.json()['data']['id']}").json()["data"]["derived_from"] is None


async def test_a_well_formed_derived_from_imports_verbatim(client):  # noqa: F811
    http, _ = client
    r = http.post("/api/probes/import", json=_doc("derived-ok", derived_from=DERIVED_FROM))
    assert r.status_code == 200, r.text
    assert http.get(f"/api/probes/{r.json()['data']['id']}").json()["data"]["derived_from"] == DERIVED_FROM


# R3 review fix (OR3-043-5), 2026-11-16 (adopted from the review draft): the contract's `maxLength: 200` on
# `parent_probe_id`, `provider` and `provider_version`. Dropping `max_length=200` from `parent_probe_id` (review
# mutation X6) left every test green: the cases above cover patterns, the enum and missing fields, never a length.
LONG = {f: {**DERIVED_FROM, f: "p" * 201} for f in ("parent_probe_id", "provider", "provider_version")}


@pytest.mark.parametrize("field", list(LONG))
async def test_an_over_long_field_is_refused(client, field):  # noqa: F811
    http, _ = client
    r = http.post("/api/probes/import", json=_doc(f"long-{field}", derived_from=LONG[field]))
    assert r.status_code == 422, r.text
    locs = [list(e["loc"][:3]) for e in r.json()["error"]["details"]["errors"]]
    assert ["provenance", "derived_from", field] in locs, locs


@pytest.mark.parametrize("field", list(LONG))
async def test_exactly_200_characters_import(client, field):  # noqa: F811
    http, _ = client
    r = http.post("/api/probes/import", json=_doc(f"len-200-{field}", derived_from={**DERIVED_FROM, field: "p" * 200}))
    assert r.status_code == 200, r.text
