"""The answer contract covers every route and types every JSON answer (044 FR-044.3, .4, .9, .14, .15, .19; TD10).

Every check reads the GENERATED contract (`answers_contract()`), so a generator defect shows here even when the
committed file is stale; `test_answers_contract.py` separately pins the committed file to it.
"""

from __future__ import annotations

import re

import pytest
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.routing import APIRoute, iter_route_contexts

from miserver.api.routes import register_routes
from miserver.api.schemas import contract_tools as tools
from miserver.api.schemas.answers_contract import answers_contract, contract_from

#: TD10: the module-qualified component names FastAPI writes when two schema modules share a class name, as of
#: 2026-11-25. A new model whose name collides (an `AttachResponse` in an answers module) renames components the
#: frozen weights contract embeds; this set then changes and the test goes red.
QUALIFIED_2026_11_25 = frozenset(
    {
        "miserver__api__schemas__common__ApiResponse_AttachResponse___1",
        "miserver__api__schemas__common__ApiResponse_AttachResponse___2",
        "miserver__api__schemas__common__ApiResponse_DetachResponse___1",
        "miserver__api__schemas__common__ApiResponse_DetachResponse___2",
        "miserver__api__schemas__sae__AttachResponse",
        "miserver__api__schemas__sae__AttachmentStatus",
        "miserver__api__schemas__sae__DetachResponse",
        "miserver__api__schemas__weights__AttachResponse",
        "miserver__api__schemas__weights__AttachmentStatus",
        "miserver__api__schemas__weights__DetachResponse",
    }
)
_EXCLUDE_FLAGS = (
    "response_model_exclude_none",
    "response_model_exclude_unset",
    "response_model_exclude_defaults",
)


@pytest.fixture(scope="module")
def app() -> FastAPI:
    app = FastAPI()
    register_routes(app)
    return app


@pytest.fixture(scope="module")
def contract() -> dict:
    return answers_contract()


def _path(path: str) -> str:
    return re.sub(r"\{(\w+):\w+\}", r"{\1}", path)


def _live_operations(app: FastAPI) -> dict[str, APIRoute]:
    out = {}
    for ctx in iter_route_contexts(app.routes):
        if isinstance(ctx.original_route, APIRoute) and ctx.include_in_schema:
            for method in ctx.methods or ():
                out[f"{method} {_path(str(ctx.path))}"] = ctx
    return out


def test_responses_and_excluded_together_equal_the_live_table(app: FastAPI, contract: dict):
    """FR-044.3: set EQUALITY with `iter_route_contexts` (184 operations on 2026-11-25); a subset does not pass."""
    live = set(_live_operations(app))
    declared = set(contract["responses"]) | set(contract["excluded"])
    assert not set(contract["responses"]) & set(contract["excluded"])
    assert declared == live, {"missing": sorted(live - declared), "extra": sorted(declared - live)}
    assert len(live) >= 184


def test_each_excluded_route_names_the_media_type_it_declares(app: FastAPI, contract: dict):
    """FR-044.3 / .9: read from the route's own declaration (`response_class`, or `responses` content)."""
    live = _live_operations(app)
    for key, row in contract["excluded"].items():
        ctx = live[key]
        declared = set()
        response_class = getattr(
            ctx.original_route.response_class, "value", ctx.original_route.response_class
        )
        if isinstance(response_class, type) and response_class is not JSONResponse:
            media = getattr(response_class, "media_type", None)
            if media:
                declared.add(media)
        for status, response in (ctx.original_route.responses or {}).items():
            if str(status).startswith("2"):
                declared |= set(response.get("content") or {})
        assert row["media_type"] in declared, (key, row, declared)
        assert row["media_type"] != "application/json", key
        assert row["reason"], key


def _untyped(contract: dict) -> set[str]:
    ext = frozenset(contract["external"])
    return {
        key
        for key, entry in contract["responses"].items()
        if not all(
            tools.is_typed(s, contract["$defs"], ext) for k, s in entry.items() if k[:1].isdigit()
        )
    }


def test_every_json_answer_is_typed(contract: dict):
    """FR-044.4: every `responses` entry is typed (task 4.12: the shrinking allowance of task 1.4, the 70 T1
    routes and the two FR-044.9 routes of 2026-11-25, reached empty and was removed)."""
    assert _untyped(contract) == set()
    assert contract["coverage"]["untyped"] == 0


def test_no_schema_closes_its_objects(contract: dict):
    """FR-044.14 (SH7): no `additionalProperties: false` anywhere in the published file."""
    closed = [s for s in tools._walk(contract) if s.get("additionalProperties") is False]
    assert closed == []


def test_the_envelope_requires_what_it_always_sends(contract: dict):
    """FR-044.14 AC: the management envelope's `required` is every field it always sends."""
    envelopes = {
        name: d
        for name, d in contract["$defs"].items()
        if {"success", "data", "error"} <= set(d.get("properties", {}))
        and "meta" not in d["properties"]
    }
    assert len(envelopes) > 50, sorted(envelopes)
    for name, body in envelopes.items():
        assert body.get("required") == ["data", "error", "success"], (name, body.get("required"))


def _referenced(node) -> set[str]:
    return {
        s["$ref"].removeprefix("#/$defs/")
        for s in tools._walk(node)
        if isinstance(s.get("$ref"), str) and s["$ref"].startswith("#/$defs/")
    }


def test_every_definition_is_reached(contract: dict):
    """FR-044.15: `$defs` holds exactly what `responses` and `errors` reach (an independent walk)."""
    seen: set[str] = set()
    frontier = _referenced([contract["responses"], contract["errors"]])
    while frontier:
        name = frontier.pop()
        if name not in seen:
            seen.add(name)
            frontier |= _referenced(contract["$defs"][name]) - seen
    assert set(contract["$defs"]) == seen, sorted(set(contract["$defs"]) - seen)


def _resolve(node: dict, defs: dict) -> dict:
    while "$ref" in node and node["$ref"].startswith("#/$defs/"):
        node = defs[node["$ref"].removeprefix("#/$defs/")]
    return node


def _alternatives(node: dict, defs: dict) -> int:
    node = _resolve(node, defs)
    if "anyOf" in node or "oneOf" in node:
        return sum(_alternatives(m, defs) for m in node.get("anyOf", node.get("oneOf")))
    return 0 if node.get("type") == "null" else 1


def test_the_coverage_counts_are_a_recount(contract: dict):
    """FR-044.19: `coverage` equals a recount of `responses` and `excluded`."""
    defs = contract["$defs"]
    untyped = len(_untyped(contract))
    unions = 0
    for entry in contract["responses"].values():
        for status, schema in entry.items():
            if not status[:1].isdigit():
                continue
            body = _resolve(schema, defs)
            data = (
                body.get("properties", {}).get("data")
                if "success" in body.get("properties", {})
                else None
            )
            unions += _alternatives(schema, defs) > 1 or (
                data is not None and _alternatives(data, defs) > 1
            )
    assert contract["coverage"] == {
        "typed": len(contract["responses"]) - untyped,
        "unions": unions,
        "excluded": len(contract["excluded"]),
        "untyped": untyped,
    }


def test_the_module_qualified_component_names_are_those_of_2026_11_25(app: FastAPI):
    """TD10: no new answer model takes a name an existing schema module already uses."""
    names = set(app.openapi()["components"]["schemas"])
    assert {n for n in names if "__api__schemas__" in n} == QUALIFIED_2026_11_25


def test_no_route_drops_fields_at_serialization(app: FastAPI):
    """FR-044.14: `response_model_exclude_*` would make "always sent" false."""
    offenders = [
        key
        for key, ctx in _live_operations(app).items()
        if any(getattr(ctx.original_route, f, False) for f in _EXCLUDE_FLAGS)
    ]
    assert offenders == []


def test_the_generator_refuses_a_route_that_drops_fields():
    from pydantic import BaseModel

    class Thing(BaseModel):
        x: int | None = None

    probe = FastAPI()

    @probe.get("/thing", response_model=Thing, response_model_exclude_none=True)
    async def thing() -> Thing:
        return Thing()

    with pytest.raises(ValueError, match="response_model_exclude_none"):
        contract_from(probe, probe.openapi())


#: SH13: the strict validator refuses undeclared keys EXCEPT where a model declares extras
#: (`extra="allow"`, rendered `additionalProperties: true`). The models that do, on 2026-11-25, all
#: older than 044: the weights handles (PyTorch's reducer fields) and the chat message.
DECLARE_EXTRAS_2026_11_25 = frozenset({"ChatMessage", "ParameterHandle", "StorageHandle"})


def test_only_the_known_models_declare_extras(contract: dict):
    """A new answer model must not escape the undeclared-key rule by allowing extras."""
    declaring = {
        name
        for name, body in contract["$defs"].items()
        if body.get("additionalProperties") is True and body.get("properties")
    }
    assert declaring == DECLARE_EXTRAS_2026_11_25


def test_the_generator_fails_closed_on_what_it_cannot_describe(monkeypatch):
    """044 R1-10: the generator's refusals, each on a planted app."""
    from pydantic import BaseModel, Field


    class Stray(BaseModel):
        x: int | None = Field(default=None, exclude_if=lambda v: v is None)

    probe = FastAPI()

    @probe.get("/stray", response_model=Stray)
    async def stray() -> Stray:
        return Stray()

    spec = probe.openapi()
    # an exclude_if model whose component cannot be found
    renamed = {
        **spec,
        "components": {"schemas": {"Elsewhere": spec["components"]["schemas"]["Stray"]}},
    }
    with pytest.raises(ValueError, match="no served component"):
        contract_from(probe, renamed)
    # an external reference that names no vendored file
    ext = FastAPI()

    @ext.get(
        "/doc",
        response_model=None,
        responses={
            200: {"content": {"application/json": {"schema": {"$ref": "https://x/none.json"}}}}
        },
    )
    async def doc() -> dict:
        return {}

    with pytest.raises(ValueError, match="unknown external"):
        contract_from(ext, ext.openapi())
    # a route with no 2xx body at all
    nobody = FastAPI()

    @nobody.get("/none", status_code=204, response_model=None, responses={204: {}})
    async def none() -> None:
        return None

    spec = nobody.openapi()
    spec["paths"]["/none"]["get"]["responses"] = {"204": {"description": "nothing"}}
    with pytest.raises(ValueError, match="no 2xx body"):
        contract_from(nobody, spec)


def test_a_history_marker_outside_an_approval_is_refused():
    from miserver.api.schemas.answers_contract import _history_result
    from miserver.api.schemas.contract_tools import HISTORY_FORM

    defs = {"Thing": {"type": "object", "properties": {"result": {HISTORY_FORM: True}}}}
    with pytest.raises(ValueError, match="history form"):
        _history_result(defs, {})
