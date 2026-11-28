"""The other unions of the contract (044 FR-044.8; TD8, TD14).

* `GET /api/circuits/active` answers a list of rows, one row (`?single=true`) or null: all three validate.
* An approval's stored `result` is checked by its `action` (an `if`/`then` per action, built from the live
  gate table), in its HISTORY FORM: each variant keeps its model's properties and types and drops `required`,
  because a stored row can predate a field. That is the one stated exception to SH7; the live answers keep
  their `required`.
* Chat's `stream: true` Server-Sent Events are listed beside the typed JSON answer.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient
from jsonschema import Draft202012Validator

from miserver.api.schemas.answers_contract import HISTORY_EXCEPTION
from miserver.api.schemas.circuit import CircuitSummary
from tests.support import answer_validator
from tests.unit.api.test_circuit_composed_activation_route import _COMPOSED
from tests.unit.approval_fixtures import AGENT, harness, make_model  # noqa: F401
from tests.unit.f28_fixtures import db  # noqa: F401 - fixture

ACTIVE = "GET /api/circuits/active"
ROW = {k: v for k, v in _COMPOSED.items() if k in CircuitSummary.model_fields}


def _active_client(rows: list[dict]) -> TestClient:
    from miserver.api.dependencies import get_circuit_service
    from miserver.main import create_app

    service = MagicMock()
    service.list_active = AsyncMock(return_value=[dict(r) for r in rows])
    app = create_app()
    app.dependency_overrides[get_circuit_service] = lambda: service
    return TestClient(app)


@pytest.mark.parametrize(
    ("rows", "single", "shape"),
    [
        ([ROW], False, list),
        ([ROW], True, dict),
        ([], True, type(None)),
        ([], False, list),
    ],
)
def test_every_form_of_active_validates(monkeypatch, rows, single, shape):
    seen = []
    real = answer_validator.check_body
    monkeypatch.setattr(
        answer_validator, "check_body", lambda k, s, b: (seen.append(k), real(k, s, b))
    )
    response = _active_client(rows).get(
        "/api/circuits/active", params={"single": str(single).lower()}
    )
    assert response.status_code == 200, response.text
    assert isinstance(response.json()["data"], shape)
    assert seen == [ACTIVE]


def test_active_declared_as_the_list_alone_refuses_the_single_row():
    """The control's reading: the row form is a member of the union, not of its list member."""
    contract = answer_validator._contract()
    union = contract["$defs"]["ActiveCircuits"]
    list_only = next(m for m in union["anyOf"] if m.get("type") == "array")
    v = Draft202012Validator({**list_only, "$defs": {}}, registry=answer_validator._registry())
    assert list(v.iter_errors({"id": "x"}))


def _read_validator() -> Draft202012Validator:
    contract = answer_validator._contract()
    return Draft202012Validator(
        {"$ref": contract["$id"] + "#/$defs/ApprovalRead"}, registry=answer_validator._registry()
    )


async def _approval_and_model(h) -> tuple[dict, dict]:
    """A real stored approval row as the read route answers it, and a real model answer."""
    h.holder["row"] = row = await make_model(h.db)
    h.make_resident(row)
    filed = h.client.post(f"/api/models/{row.id}/unload", headers=AGENT)
    assert filed.status_code == 202, filed.text
    read = h.client.get(f"/api/approvals/{filed.json()['data']['approval_id']}").json()["data"]
    model = h.client.get(f"/api/models/{row.id}")
    assert model.status_code == 200, model.text
    return read, model.json()["data"]


async def test_an_executed_unload_result_validates_by_its_action(harness):  # noqa: F811
    read, model = await _approval_and_model(harness)
    executed = {**read, "status": "executed", "result": model}
    assert list(_read_validator().iter_errors(executed)) == []
    # The same result under another action's variants fails (EnvironmentSaved | EnvironmentRow).
    assert list(_read_validator().iter_errors({**executed, "action": "telemetry_write"}))


async def test_a_delete_result_is_the_stored_null_value(harness):  # noqa: F811
    read, _ = await _approval_and_model(harness)
    executed = {**read, "action": "model_delete", "status": "executed", "result": {"value": None}}
    assert list(_read_validator().iter_errors(executed)) == []


async def test_an_older_stored_row_missing_a_field_reads_and_a_wrong_type_does_not(harness):  # noqa: F811
    read, model = await _approval_and_model(harness)
    older = {k: v for k, v in model.items() if k != "context_ceiling"}
    assert "context_ceiling" in model
    executed = {**read, "action": "model_load", "status": "executed", "result": older}
    assert list(_read_validator().iter_errors(executed)) == []
    wrong = {**executed, "result": {**model, "context_length": "8192"}}
    assert list(_read_validator().iter_errors(wrong))


def test_the_live_load_answer_still_requires_every_field():
    contract = answer_validator._contract()
    assert "context_ceiling" in contract["$defs"]["ModelResponse"]["required"]
    assert "required" not in contract["$defs"]["ModelResponse__history"]


def test_the_contract_states_the_one_exception():
    contract = answer_validator._contract()
    assert HISTORY_EXCEPTION in contract["description"]
    assert HISTORY_EXCEPTION in contract["$defs"]["ApprovalRead"]["description"]
    assert "one exception" in HISTORY_EXCEPTION and "SH7" in HISTORY_EXCEPTION


async def test_an_unlisted_action_leaves_the_result_unconstrained(harness):  # noqa: F811
    """A stored row of an action no longer in `APPROVAL_ACTIONS` keeps whatever result it stored."""
    from miserver.api.approval_gate import APPROVAL_ACTIONS

    contract = answer_validator._contract()
    branches = [
        b["if"]["properties"]["action"]["const"] for b in contract["$defs"]["ApprovalRead"]["allOf"]
    ]
    assert sorted(branches) == sorted(APPROVAL_ACTIONS)
    read, _ = await _approval_and_model(harness)
    retired = {
        **read,
        "action": "a_retired_action",
        "status": "executed",
        "result": {"anything": [1, "two"], "at": {"all": None}},
    }
    assert list(_read_validator().iter_errors(retired)) == []
    # ...while the same result under a listed action is checked.
    assert list(_read_validator().iter_errors({**retired, "action": "model_unload"}))


def test_chat_lists_its_stream_variant():
    entry = answer_validator._contract()["responses"]["POST /v1/chat/completions"]
    assert "200" in entry and "text/event-stream" in entry["x-variants"]
    assert "Server-Sent Events" in entry["x-variants"]["text/event-stream"]


async def test_a_new_card_line_in_an_approval_summary_is_not_an_undeclared_key(harness):  # noqa: F811
    """FR-044.6 AC (SH13): `summary` is a declared map, so a card line no earlier answer used passes."""
    read, _ = await _approval_and_model(harness)
    assert isinstance(read["summary"], dict)
    novel = {**read, "summary": {**read["summary"], "A card line nobody has sent before": "x"}}
    assert list(_read_validator().iter_errors(novel)) == []


def test_no_switch_turns_the_undeclared_key_check_off():
    """FR-044.6 AC: the validator reads no environment value and no marker, so nothing can make it
    lenient: its module imports no `os` and reads no flag, and strict form is applied unconditionally
    when the registry is built."""
    import ast
    import inspect

    tree = ast.parse(inspect.getsource(answer_validator))
    imported = {
        a.name
        for n in ast.walk(tree)
        if isinstance(n, (ast.Import, ast.ImportFrom))
        for a in n.names
    } | {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
    assert not {"os", "tests.support.env_flags", "env_flags", "flag"} & imported, imported
    registry = answer_validator._registry()
    contract = answer_validator._contract()
    held = registry[contract["$id"]].contents
    from miserver.api.schemas.contract_tools import strict_form

    assert held == strict_form(contract)
