"""The 70 T1 routes are typed in documentation only, and serve through the model they served before 044
(044 FR-044.5, SH6; ENH-013 operator decision, 2026-11-26).

Each route that answered untyped JSON on 2026-11-25 (FPRD §6, table T1) documents its answer model in
`responses=`. Its `response_model` is the one it served before 044 (`SERVED`, read from the live route table
of `44d1ee6a`), because the response model decides FastAPI's serializer, not only its filtering:
- **with a response model**, pydantic-core writes the bytes. A UTC datetime is `...Z`, a small float
  `0.0000286102294921875`, NaN `null`.
- **with `response_model=None`**, `jsonable_encoder` and `json.dumps` write them. A datetime in a dict is
  `...+00:00`, the float `2.86102294921875e-05`, and NaN a 500.

044-1 shipped `response_model=None` on 69 of these routes. That changed the live bytes of `GET /api/probes`'s
small scores (measured 2026-11-26). The decision: documentation-only typing must not change a route's
serializer. So:
- `test_each_t1_route_serves_its_pre_044_model` pins each route's served model;
- `test_each_t1_route_writes_the_risky_values_as_before` pins the bytes over the risky types;
- `test_a_typed_route_writes_the_risky_values_as_before_over_http` follows one route over HTTP.
Control: switch one route back to `response_model=None` and these go red.
"""

from __future__ import annotations

import asyncio
import math
import re
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.routing import APIRoute, iter_route_contexts, serialize_response

from miserver.api.routes import register_routes

#: FPRD §6 table T1 -> the response model each route served on origin/main 44d1ee6a (before 044), as
#: `repr`-style text; None: the route served no model before 044 either.
SERVED: dict[str, str | None] = {
    "DELETE /api/circuit-sensing/events": 'ApiResponse[dict]',
    "DELETE /api/circuits/{circuit_id}": 'ApiResponse[dict]',
    "DELETE /api/clusters/{cluster_id}": 'ApiResponse[dict[str, Any]]',
    "DELETE /api/probes/behaviours/{key}/rule-signals": 'ApiResponse',
    "DELETE /api/probes/behaviours/{key}/verdict": 'ApiResponse',
    "DELETE /api/probes/events": 'ApiResponse',
    "DELETE /api/probes/sessions/{session_id}": 'ApiResponse',
    "DELETE /api/probes/{probe_id}": 'ApiResponse',
    "DELETE /api/rule-sets/{rule_set_id}": 'ApiResponse',
    "DELETE /api/sensing/events": 'ApiResponse[dict]',
    "DELETE /v1/files/{file_id}": 'dict[str, Any]',
    "GET /api/approvals": 'ApiResponse',
    "GET /api/approvals/actions": 'ApiResponse',
    "GET /api/approvals/{approval_id}": 'ApiResponse',
    "GET /api/circuits/active": 'ApiResponse[Any]',
    "GET /api/circuits/claims": 'ApiResponse[list[dict]]',
    "GET /api/circuits/{circuit_id}/export": 'dict[str, Any]',
    "GET /api/clusters/{cluster_id}/export": 'dict[str, Any]',
    "GET /api/health/inference": 'dict[str, Any]',
    "GET /api/health/version": None,
    "GET /api/models/{model_id}/probe-capacity": 'ApiResponse',
    "GET /api/models/{model_id}/probe-limits/recommendation": 'ApiResponse',
    "GET /api/probes": 'PageResponse',
    "GET /api/probes/behaviours": 'PageResponse',
    "GET /api/probes/behaviours/{key}/rule-signals": 'ApiResponse',
    "GET /api/probes/behaviours/{key}/verdict": 'ApiResponse',
    "GET /api/probes/events": 'PageResponse',
    "GET /api/probes/events/{event_id}": 'ApiResponse',
    "GET /api/probes/hub/search": 'ApiResponse',
    "GET /api/probes/hub/{repo_id}/definitions": 'ApiResponse',
    "GET /api/probes/sessions": 'PageResponse',
    "GET /api/probes/sessions/{session_id}": 'ApiResponse',
    "GET /api/probes/sessions/{session_id}/transcript": 'ApiResponse',
    "GET /api/probes/status": 'ApiResponse',
    "GET /api/rule-sets": 'PageResponse',
    "GET /api/rule-sets/validators": 'ApiResponse',
    "GET /api/rule-sets/{rule_set_id}": 'ApiResponse',
    "GET /v1/batches": 'dict[str, Any]',
    "GET /v1/batches/{batch_id}": 'dict[str, Any]',
    "GET /v1/files": 'dict[str, Any]',
    "GET /v1/files/{file_id}": 'dict[str, Any]',
    "POST /api/approvals/approve-batch": 'ApiResponse',
    "POST /api/approvals/{approval_id}/approve": 'ApiResponse',
    "POST /api/approvals/{approval_id}/reject": 'ApiResponse',
    "POST /api/circuits/claims/release": 'ApiResponse[dict]',
    "POST /api/clusters/{cluster_id}/activate": 'ApiResponse[dict[str, Any]]',
    "POST /api/clusters/{cluster_id}/deactivate": 'ApiResponse[dict[str, Any]]',
    "POST /api/models/{model_id}/probe-capacity/measure": 'ApiResponse',
    "POST /api/probes/behaviours/{key}/arm": 'ApiResponse',
    "POST /api/probes/behaviours/{key}/disarm": 'ApiResponse',
    "POST /api/probes/hub/import": 'ApiResponse',
    "POST /api/probes/import": 'ApiResponse',
    "POST /api/probes/score": 'ApiResponse',
    "POST /api/probes/sessions/sample": 'ApiResponse',
    "POST /api/probes/{probe_id}/arm": 'ApiResponse',
    "POST /api/probes/{probe_id}/disarm": 'ApiResponse',
    "POST /api/probes/{probe_id}/parity": 'ApiResponse',
    "POST /api/probes/{probe_id}/recalibrate": 'ApiResponse',
    "POST /api/rule-sets/import": 'ApiResponse',
    "POST /api/rule-sets/validate": 'ApiResponse',
    "POST /api/rule-sets/{rule_set_id}/evaluate": 'ApiResponse',
    "POST /v1/batches": 'dict[str, Any]',
    "POST /v1/batches/{batch_id}/cancel": 'dict[str, Any]',
    "POST /v1/batches/{batch_id}/lease": 'dict[str, Any]',
    "POST /v1/files": 'dict[str, Any]',
    "PUT /api/clusters/active/intensity": 'ApiResponse[dict[str, Any]]',
    "PUT /api/clusters/{cluster_id}/intensity": 'ApiResponse[dict[str, Any]]',
    "PUT /api/models/{model_id}/probe-limits": 'ApiResponse',
    "PUT /api/probes/behaviours/{key}/rule-signals": 'ApiResponse',
    "PUT /api/probes/behaviours/{key}/verdict": 'ApiResponse',
}
T1 = frozenset(SERVED)


@pytest.fixture(scope="module")
def routes() -> dict:
    app = FastAPI()
    register_routes(app)
    out = {}
    for ctx in iter_route_contexts(app.routes):
        if isinstance(ctx.original_route, APIRoute):
            for method in ctx.methods or ():
                path = re.sub(r"\{(\w+):\w+\}", r"{\1}", str(ctx.path))
                out[f"{method} {path}"] = ctx
    return out


def _served(ctx) -> Any:
    """The route's effective (served) route: the one FastAPI serializes through."""
    return getattr(ctx, "_effective_route", None) or ctx


def _text(model: Any) -> str | None:
    if model is None:
        return None
    text = repr(model)
    match = re.fullmatch(r"<class 'miserver\.api\.schemas\.common\.(.+)'>", text)
    return match.group(1) if match else text.replace("typing.", "")


def test_the_t1_set_is_the_one_recorded():
    assert len(T1) == 70
    assert sum(1 for v in SERVED.values() if v is None) == 1  # GET /api/health/version, unchanged


@pytest.mark.parametrize("key", sorted(T1))
def test_each_t1_route_serves_its_pre_044_model(routes: dict, key: str):
    ctx = routes[key]
    served = _served(ctx)
    assert _text(served.response_model) == SERVED[key], (
        f"{key} serves {served.response_model!r}; before 044 it served {SERVED[key]}: its serializer changed"
    )
    assert (served.response_field is None) == (SERVED[key] is None)
    status = ctx.original_route.status_code or 200
    declared = ctx.original_route.responses.get(status) or ctx.original_route.responses.get(str(status)) or {}
    assert declared.get("model") is not None or (declared.get("content") or {}).get(
        "application/json"
    ), f"{key} declares no {status} answer model"


#: The values whose bytes depend on the serializer, and the bytes pydantic-core writes for each.
RISKY = {
    "t": datetime(2026, 11, 26, 1, 2, 3, tzinfo=timezone.utc),
    "small": 2.86102294921875e-05,
    "nan": math.nan,
    "dec": Decimal("1.50"),
    "span": timedelta(seconds=90),
}
EXPECTED = (b'"t":"2026-11-26T01:02:03Z"', b'"small":0.0000286102294921875', b'"nan":null',
            b'"dec":"1.50"', b'"span":"PT1M30S"')


def _payload(text: str) -> Any:
    risky = dict(RISKY)
    if text.startswith("dict"):
        return risky
    data: Any = [risky] if text == "ApiResponse[list[dict]]" else risky
    body: dict[str, Any] = {"success": True, "data": data, "error": None}
    if text == "PageResponse":
        body["meta"] = {"next": None, "limit": 1}
    return body


@pytest.mark.parametrize("key", sorted(k for k, v in SERVED.items() if v is not None))
def test_each_t1_route_writes_the_risky_values_as_before(routes: dict, key: str):
    """The route's own served field writes the risky values as pydantic-core does (FastAPI's fast path)."""
    field = _served(routes[key]).response_field
    assert field is not None, f"{key} has no served field: jsonable_encoder would write its answer"
    body = asyncio.run(serialize_response(field=field, response_content=_payload(SERVED[key]),
                                          dump_json=True))
    for expected in EXPECTED:
        assert expected in body, f"{key}: {expected!r} not in {body!r}"


def test_a_typed_route_writes_the_risky_values_as_before_over_http(monkeypatch):
    """One route end to end through the real app: `GET /api/health/inference` (served `dict[str, Any]`)."""
    from fastapi.testclient import TestClient

    from miserver.api.dependencies import get_inference_service
    from miserver.main import create_app

    from unittest.mock import MagicMock

    from miserver.services.inference_service import InferenceService

    real = InferenceService(model_service=MagicMock()).get_backend_info()  # a valid answer, on its contract

    class _Stub:
        def get_backend_info(self) -> dict[str, Any]:
            # `gguf_load` is the model's free-form map: the risky values ride it on the contract.
            return {**real, "gguf_load": dict(RISKY)}

    app = create_app()
    app.dependency_overrides[get_inference_service] = lambda: _Stub()
    response = TestClient(app).get("/api/health/inference")
    assert response.status_code == 200, response.text
    for expected in EXPECTED:
        assert expected in response.content, f"{expected!r} not in {response.content!r}"
