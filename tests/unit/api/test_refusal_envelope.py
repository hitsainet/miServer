"""The contract's error envelopes describe real refusals (044 FR-044.13, SH10, SH13).

`errors.management` is the management envelope with `RefusalDetails` (`ErrorDetails` plus `next`, the MCP
tool that resolves the refusal, added by `exception_handlers.py` only when the exception names one).
`errors.openai` is `OpenAIErrorResponse`. Both are checked in the validator's strict form, so a key the
refusal sends that the model lacks fails; `details` is a declared map, so its keys are free.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from jsonschema import Draft202012Validator

from miserver.main import create_app
from tests.support import answer_validator
from tests.unit.f28_fixtures import db  # noqa: F401 - fixture

ROOT = Path(__file__).resolve().parents[3]
HINTS = json.loads((ROOT / "docs" / "schemas" / "next-tool-hints.json").read_text())["next_tools"]


def _errors(name: str) -> Draft202012Validator:
    uri = answer_validator._contract()["$id"] + "#" + answer_validator._pointer("errors", name)
    return Draft202012Validator({"$ref": uri}, registry=answer_validator._registry())


@pytest.fixture
def client(db) -> TestClient:  # noqa: F811
    """The live app over the test database (no `/app` data directory is touched)."""
    from miserver.api.dependencies import get_db

    app = create_app()

    async def _db():
        async with db() as session:
            yield session

    app.dependency_overrides[get_db] = _db
    return TestClient(app, raise_server_exceptions=False)


def test_a_management_refusal_with_a_next_tool_validates(client: TestClient):
    """`rule_sets.py` refuses an evaluate that sends both `messages` and `text`, naming the tool to call."""
    response = client.post(
        "/api/rule-sets/rs_000000000000/evaluate",
        json={"text": "a", "messages": [{"role": "user", "content": "b"}]},
    )
    assert 400 <= response.status_code < 500, response.text
    body = response.json()
    assert body["error"]["next"] == "miserver_evaluate_rule_set"
    assert body["error"]["next"] in HINTS
    assert list(_errors("management").iter_errors(body)) == []
    # The keys are exactly the envelope's and `RefusalDetails`'.
    assert set(body) == {"success", "data", "error"}
    assert set(body["error"]) == {"code", "message", "details", "next"}


def test_a_refusal_without_a_next_tool_validates_too(client: TestClient):
    response = client.get("/api/probes/sessions/ses_000000000000")
    assert response.status_code == 404, response.text
    body = response.json()
    assert "next" not in body["error"]
    assert list(_errors("management").iter_errors(body)) == []


def test_an_undeclared_refusal_key_fails_the_strict_form():
    body = {
        "success": False,
        "data": None,
        "error": {"code": "X", "message": "m", "details": {"free": 1}, "surprise": True},
    }
    assert list(_errors("management").iter_errors(body))
    body["error"].pop("surprise")
    assert list(_errors("management").iter_errors(body)) == []


def test_an_openai_refusal_validates(client: TestClient):
    response = client.get("/v1/batches/batch_000000000000")
    assert 400 <= response.status_code < 500, response.text
    body = response.json()
    assert "error" in body and "success" not in body
    assert list(_errors("openai").iter_errors(body)) == []
