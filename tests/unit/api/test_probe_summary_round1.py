"""Two probe-summary findings of 044 review round 1, through the live routes (R1-1, R1-3)."""

from __future__ import annotations

import pytest

from tests.support import answer_validator
from tests.unit.api.test_probe_event_filters import client  # noqa: F401 - fixture
from tests.unit.f28_fixtures import db  # noqa: F401 - fixture
from tests.unit.probe_fixtures import probe_definition


@pytest.mark.parametrize("version", [True, "2"])  # 0 is refused at import (ge=1)
def test_a_version_that_is_not_a_positive_integer_is_not_sent_as_one(client, version):  # noqa: F811
    """R1-1, R2-2: `true` and `"2"` import, the raw document keeps them, and the row, the
    behaviours list and probe results all treat them as uncatalogued (`behaviour_version_of`); the
    tile sent them as they were. One rule now: a positive integer, else null."""
    doc = probe_definition(name="bool-version")
    doc["behaviour"] = {"key": "pii-leak", "name": "PII leak", "version": version}
    response = client.post("/api/probes/import", json=doc)
    assert response.status_code == 200, response.text
    assert response.json()["data"]["behaviour"] == {
        "key": "pii-leak",
        "name": "PII leak",
        "version": None,
    }


def test_a_list_row_without_its_totals_is_refused(client):  # noqa: F811
    """R1-3: `GET /api/probes` always sends the three totals; its rows are `ProbeListItem`, where
    they are required (as `ProbeSummary` they were may-be-absent, so a row without them passed)."""
    assert (
        client.post("/api/probes/import", json=probe_definition(name="listed")).status_code == 200
    )
    body = client.get("/api/probes").json()
    assert {"results_total", "results_fired", "last_result_at"} <= set(body["data"][0])
    stripped = {
        **body,
        "data": [{k: v for k, v in row.items() if k != "results_total"} for row in body["data"]],
    }
    with pytest.raises(answer_validator.AnswerContractError, match="results_total"):
        answer_validator.check_body("GET /api/probes", 200, stripped)
    answer_validator.FAILURES.clear()


@pytest.mark.parametrize("version", [True, False, "2", 0, -1, 2, None, 2.0])
def test_the_tile_and_the_row_read_a_version_by_one_rule(version):
    """R2-2, R3-4: `behaviour_of` (the tile) and `behaviour_version_of` (the row, selection) agree on
    every value a raw document can hold, including those the import refuses (0, -1)."""
    from miserver.core.probe_labels import behaviour_of, behaviour_version_of

    definition = {"behaviour": {"key": "k", "version": version}}
    assert behaviour_of(definition)["version"] == behaviour_version_of(definition)


def test_a_single_probe_answer_carrying_totals_is_refused(client):  # noqa: F811
    """R2-5, R3-3: the routes answering with one changed probe never send the result totals, and
    `ProbeSummary` does not declare them, so a planted total is refused."""
    response = client.post("/api/probes/import", json=probe_definition(name="no-totals"))
    assert response.status_code == 200, response.text
    body = response.json()
    assert "results_total" not in body["data"]
    body["data"]["results_total"] = 0
    with pytest.raises(answer_validator.AnswerContractError, match="results_total"):
        answer_validator.check_body("POST /api/probes/import", 200, body)
    answer_validator.FAILURES.clear()
