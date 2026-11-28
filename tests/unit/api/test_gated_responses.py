"""Every gated route declares the pending approval it answers an agent (044 FR-044.7, TD1).

`gated_responses(...)` (`approval_gate.py`) is the one place the pending 202 is declared. The guard reads the
LIVE gate table (`gated_routes(app)`): the set of operations whose 202 declares the pending answer must EQUAL
it, and the status each route passed must equal its own. Then each of the 10 routes is called twice through
the live app: as an agent (202, `X-miLLM-Approval: pending`, a body the validator accepts against the 202
entry) and as the operator (its own entry). The answer validator checks both bodies; the test asserts it saw
them, with the payload, so a validator that skipped gated endpoints would go red here.
"""

from __future__ import annotations

import re

import pytest
from fastapi import FastAPI
from fastapi.routing import APIRoute, iter_route_contexts
from jsonschema import Draft202012Validator

from miserver.api.approval_gate import APPROVAL_HEADER, gated_routes
from miserver.api.routes import register_routes
from tests.support import answer_validator
from tests.unit.api.test_approval_gate_coverage import (
    CALLS,
    GGUF_ACTIONS,
    RESIDENT_ACTIONS,
)
from tests.unit.approval_fixtures import AGENT, harness, make_gguf_model, make_model  # noqa: F401
from tests.unit.f28_fixtures import db  # noqa: F401 - fixture


def _key(method: str, path: str) -> str:
    return f"{method} {re.sub(r'\{(\w+):\w+\}', r'{\1}', path)}"


@pytest.fixture(scope="module")
def app() -> FastAPI:
    app = FastAPI()
    register_routes(app)
    return app


def _pending_ref(app: FastAPI) -> str:
    """The component of the pending envelope, read off a route whose own status is not 202."""
    spec = app.openapi()
    ref = spec["paths"]["/api/models/{model_id}/unload"]["post"]["responses"]["202"]["content"][
        "application/json"
    ]["schema"]["$ref"]
    assert ref.endswith("ApiResponse_ApprovalPending_"), ref
    return ref


def _declares_pending(spec: dict, path: str, method: str, pending: str) -> bool:
    op = spec["paths"][path][method.lower()]
    schema = (
        op.get("responses", {})
        .get("202", {})
        .get("content", {})
        .get("application/json", {})
        .get("schema", {})
    )
    if schema.get("$ref") == pending:
        return True
    name = schema.get("$ref", "").removeprefix("#/components/schemas/")
    union = spec["components"]["schemas"].get(name, {})
    return pending in [m.get("$ref") for m in union.get("anyOf", [])]


def test_the_routes_declaring_the_pending_202_are_the_live_gated_set(app: FastAPI):
    spec = app.openapi()
    pending = _pending_ref(app)
    declaring = {
        _key(m.upper(), p)
        for p, ops in spec["paths"].items()
        for m in ops
        if _declares_pending(spec, p, m, pending)
    }
    gated = {_key(m, p) for m, p, _ in gated_routes(app)}
    assert len(gated) == 10
    assert declaring == gated, {
        "undeclared": sorted(gated - declaring),
        "extra": sorted(declaring - gated),
    }


def test_each_route_passed_its_own_status(app: FastAPI):
    """A route whose own status is 202 gets the union; any other gets the plain pending 202."""
    spec = app.openapi()
    pending = _pending_ref(app)
    for ctx in iter_route_contexts(app.routes):
        if not isinstance(ctx.original_route, APIRoute):
            continue
        if not getattr(ctx.endpoint, "__miserver_approval_action__", None):
            continue
        path = re.sub(r"\{(\w+):\w+\}", r"{\1}", str(ctx.path))
        for method in ctx.methods or ():
            schema = spec["paths"][path][method.lower()]["responses"]["202"]["content"][
                "application/json"
            ]["schema"]
            own = ctx.original_route.status_code or 200
            if own == 202:
                assert schema["$ref"] != pending and _declares_pending(
                    spec, path, method, pending
                ), path
            else:
                assert schema["$ref"] == pending, (method, path, own)


#: The live gate table's routes, each with its call: (action, METHOD path) -> call(h, headers).
def _environment(method: str):
    def call(h, headers):
        if method == "PUT /api/environment":
            return h.client.put(
                "/api/environment",
                json={"values": {"TELEMETRY_TRACES_SAMPLER_RATIO": 0.5}, "reason": "pilot"},
                headers=headers,
            )
        # A delete needs an override to delete: the operator sets one first (not an agent: no row).
        set_first = h.client.put(
            "/api/environment/TELEMETRY_TRACES_SAMPLER_RATIO",
            json={"value": 0.5, "reason": "pilot"},
        )
        assert set_first.status_code == 200, set_first.text
        return h.client.delete(
            "/api/environment/TELEMETRY_TRACES_SAMPLER_RATIO",
            params={"reason": "pilot"},
            headers=headers,
        )

    return call


ROUTES = {
    "POST /api/models/{model_id}/load": ("model_load", CALLS["model_load"][0]),
    "POST /api/models/{model_id}/unload": ("model_unload", CALLS["model_unload"][0]),
    "DELETE /api/models/{model_id}": ("model_delete", CALLS["model_delete"][0]),
    "PUT /api/models/{model_id}/probe-limits": (
        "probe_limits_write",
        CALLS["probe_limits_write"][0],
    ),
    "POST /api/models/{model_id}/probe-capacity/measure": (
        "capacity_measure",
        CALLS["capacity_measure"][0],
    ),
    "POST /api/rule-sets/import": ("code_rule_import", CALLS["code_rule_import"][0]),
    "PUT /api/models/{model_id}/context-ceiling": (
        "model_context_write",
        CALLS["model_context_write"][0],
    ),
    "PUT /api/environment/{key}": ("telemetry_write", CALLS["telemetry_write"][0]),
    "PUT /api/environment": ("telemetry_write", _environment("PUT /api/environment")),
    "DELETE /api/environment/{key}": (
        "telemetry_write",
        _environment("DELETE /api/environment/{key}"),
    ),
}


def test_every_gated_route_has_a_call_here(app: FastAPI):
    assert set(ROUTES) == {_key(m, p) for m, p, _ in gated_routes(app)}


class _Seen:
    """Records what the answer validator checked (the payload and the count), around the real check."""

    def __init__(self, monkeypatch) -> None:
        self.calls: list[tuple[str, str, dict]] = []
        real = answer_validator.check_body

        def check_body(key, status, body):
            self.calls.append((key, str(status), body))
            return real(key, status, body)

        monkeypatch.setattr(answer_validator, "check_body", check_body)


async def _arrange(h, action: str) -> None:
    h.holder["row"] = await (make_gguf_model if action in GGUF_ACTIONS else make_model)(h.db)
    if action in RESIDENT_ACTIONS:
        h.make_resident(h.holder["row"])


@pytest.mark.parametrize("key", sorted(ROUTES))
async def test_an_agent_call_answers_the_declared_pending_202(harness, monkeypatch, key):  # noqa: F811
    action, call = ROUTES[key]
    await _arrange(harness, action)
    seen = _Seen(monkeypatch)
    response = call(harness, AGENT)
    assert response.status_code == 202, response.text
    assert response.headers[APPROVAL_HEADER] == "pending"
    body = response.json()
    assert [c for c in seen.calls if c[0] == key] == [(key, "202", body)], seen.calls
    assert body["data"]["status"] == "pending" and body["data"]["action"] == action


@pytest.mark.parametrize("key", sorted(ROUTES))
async def test_the_operator_call_answers_its_own_entry(harness, monkeypatch, key):  # noqa: F811
    action, call = ROUTES[key]
    harness.holder["row"] = await (make_gguf_model if action in GGUF_ACTIONS else make_model)(
        harness.db
    )
    seen = _Seen(monkeypatch)
    response = call(harness, {})
    assert response.status_code in (200, 202), response.text
    assert APPROVAL_HEADER not in response.headers
    assert [(k, s) for k, s, _ in seen.calls if k == key] == [(key, str(response.status_code))]


def test_the_load_union_is_exclusive():
    """FR-044.7: each sampled 202 answer matches exactly ONE member of the load union (anyOf, told apart by
    the pending answer's required `approval_id` and `status: pending`, which `ModelStatus` never holds)."""
    contract = answer_validator._contract()
    entry = contract["responses"]["POST /api/models/{model_id}/load"]["202"]
    union = contract["$defs"][entry["$ref"].removeprefix("#/$defs/")]
    members = union["anyOf"]
    assert len(members) == 2
    registry = answer_validator._registry()
    validators = [
        Draft202012Validator(
            {"$ref": contract["$id"] + "#/$defs/" + m["$ref"].removeprefix("#/$defs/")},
            registry=registry,
        )
        for m in members
    ]
    pending = {
        "success": True,
        "error": None,
        "data": {
            "approval_id": "apr_" + "0" * 32,
            "status": "pending",
            "action": "model_load",
            "summary": {"Model": "x"},
            "request_digest": "sha256:" + "0" * 64,
            "expires_at": "2026-11-26T00:00:00+00:00",
            "hint": "h",
        },
    }
    matches = [not list(v.iter_errors(pending)) for v in validators]
    assert matches.count(True) == 1, matches
