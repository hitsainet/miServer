"""The answer validator (044 FR-044.10; TD2, TD3, TD5; SH13): every 2xx JSON answer any test produces is checked
against the committed answer contract, in its STRICT form.

Installed when imported, by wrapping `fastapi.routing.APIRoute.get_route_handler`. FastAPI builds a route's handler
when the route is created and again for each inclusion of its router, so a router built BEFORE this import keeps an
unwrapped handler, silently. `tests/conftest.py` therefore imports this module as its first statement, and
`tests/unit/support/test_answer_validator_wiring.py` proves every route of a fresh `create_app()` is wrapped.

What a check does, for an answer whose status is 2xx and whose media type is `application/json` (not a stream):
1. find the route's contract key from its ENDPOINT (TD3: a router-local `self.path` has no `/v1`), a gated
   endpoint through `EXECUTORS`; an endpoint outside `miserver.` (a test double) is not checked, and a `miserver`
   endpoint the contract does not hold FAILS;
2. take the entry for the answer's status; an undeclared status FAILS;
3. validate the body against the entry's strict form: every object with named properties refuses keys it does not
   declare (SH13, permanent; maps accept any key). The published file stays permissive (SH7).

A failure is raised as `AnswerContractError` AND recorded in `FAILURES` (IC-4): the generic exception handler
(`miserver/main.py`) can turn a raise into a 500 that a test never asserts on, so the autouse fixture
`_answers_match_the_contract` in `tests/conftest.py` fails the test from the record.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from fastapi.routing import APIRoute

#: The original, kept for the wiring test and the control M3.
ORIGINAL_GET_ROUTE_HANDLER = APIRoute.get_route_handler
MARK = "__answer_checked__"


class AnswerContractError(AssertionError):
    """An answer that its contract entry does not describe."""

    def __init__(self, route: str, status: int | str, json_path: str, message: str) -> None:
        self.route, self.status, self.json_path, self.message = (
            route,
            str(status),
            json_path,
            message,
        )
        super().__init__(
            f"{route} answered {status} off its contract at {json_path or '(body)'}: {message}"
        )


#: Every failure of the session's current test (IC-4); the conftest fixture clears it before each test.
FAILURES: list[AnswerContractError] = []


def _wrapped_get_route_handler(self: APIRoute) -> Callable[..., Any]:
    inner = ORIGINAL_GET_ROUTE_HANDLER(self)

    async def handler(request: Any) -> Any:
        response = await inner(request)
        _check(self, request, response)
        return response

    setattr(handler, MARK, True)
    return handler


APIRoute.get_route_handler = _wrapped_get_route_handler  # type: ignore[method-assign]


def handler_of(asgi_app: Any) -> Any:
    """The request handler a route's ASGI app runs (Starlette's `request_response` closes over it), or None."""
    for cell in getattr(asgi_app, "__closure__", None) or ():
        try:
            value = cell.cell_contents
        except ValueError:
            continue
        if callable(value) and getattr(value, MARK, False):
            return value
    return None


# ── state built once per session ─────────────────────────────────────────────────────────────────────

_STATE: dict[str, Any] = {}


def _contract() -> dict[str, Any]:
    if "contract" not in _STATE:
        from miserver.api.schemas.answers_contract import CONTRACT_FILENAME, SCHEMAS_DIR

        _STATE["contract"] = json.loads(
            (SCHEMAS_DIR / CONTRACT_FILENAME).read_text(encoding="utf-8")
        )
    return _STATE["contract"]


def endpoint_name(endpoint: Callable[..., Any]) -> str:
    """`module.qualname` of the function a route runs; a gated route's through `EXECUTORS` (TD3)."""
    from miserver.api.approval_gate import EXECUTORS

    key = getattr(endpoint, "__millm_executor_key__", None)
    if key is not None:
        endpoint = EXECUTORS[key].fn
    return f"{endpoint.__module__}.{endpoint.__qualname__}"


def endpoint_map() -> dict[str, dict[str, str]]:
    """`module.qualname` -> {METHOD: contract key}, from a fresh `register_routes` app."""
    if "endpoints" not in _STATE:
        from fastapi import FastAPI

        from miserver.api.routes import register_routes

        app = FastAPI()
        register_routes(app)
        _STATE["endpoints"] = map_routes(app)
    return _STATE["endpoints"]


def map_routes(app: Any) -> dict[str, dict[str, str]]:
    """`module.qualname` -> {METHOD: contract key} for `app`'s route table."""
    import re

    from fastapi.routing import iter_route_contexts

    out: dict[str, dict[str, str]] = {}
    for ctx in iter_route_contexts(app.routes):
        if not isinstance(ctx.original_route, APIRoute) or not ctx.include_in_schema:
            continue
        path = re.sub(r"\{(\w+):\w+\}", r"{\1}", str(ctx.path))
        for method in ctx.methods or ():
            by_method = out.setdefault(endpoint_name(ctx.endpoint), {})
            key = f"{method} {path}"
            # 044 R1-5: one endpoint serving two routes with one method would be ambiguous here.
            if by_method.get(method, key) != key:
                raise AssertionError(
                    f"{endpoint_name(ctx.endpoint)} serves {by_method[method]} and {key}: the "
                    "validator cannot tell its answers apart (TD3)"
                )
            by_method[method] = key
    return out


def _registry() -> Any:
    if "registry" not in _STATE:
        from referencing import Registry, Resource
        from referencing.jsonschema import DRAFT202012

        from miserver.api.schemas.answers_contract import SCHEMAS_DIR
        from miserver.api.schemas.contract_tools import strict_form

        contract = _contract()
        resources = [
            (
                contract["$id"],
                Resource.from_contents(strict_form(contract), default_specification=DRAFT202012),
            )
        ]
        for sid, row in contract.get("external", {}).items():
            doc = json.loads((SCHEMAS_DIR.parent.parent / row["file"]).read_text(encoding="utf-8"))
            resources.append((sid, Resource.from_contents(doc, default_specification=DRAFT202012)))
        _STATE["registry"] = Registry().with_resources(resources)
    return _STATE["registry"]


def _pointer(*parts: str) -> str:
    from urllib.parse import quote

    return "/" + "/".join(quote(p.replace("~", "~0").replace("/", "~1"), safe="~") for p in parts)


def validator_for(key: str, status: str) -> Any:
    """The compiled validator of one entry's strict form, cached on (key, status)."""
    cache = _STATE.setdefault("validators", {})
    if (key, status) not in cache:
        from jsonschema import Draft202012Validator

        uri = _contract()["$id"] + "#" + _pointer("responses", key, status)
        cache[(key, status)] = Draft202012Validator({"$ref": uri}, registry=_registry())
    return cache[(key, status)]


def _json_path(error: Any) -> str:
    out = ""
    for part in error.absolute_path:
        out += f"[{part}]" if isinstance(part, int) else (f".{part}" if out else str(part))
    return out


def _message(error: Any) -> str:
    if error.validator == "additionalProperties" and isinstance(error.instance, dict):
        declared = set((error.schema or {}).get("properties", {}))
        extra = sorted(set(error.instance) - declared)
        model = (error.schema or {}).get("title", "?")
        return f"undeclared key {', '.join(map(repr, extra))} (model {model})"
    return error.message


def check_body(key: str, status: int | str, body: Any) -> None:
    """Validate one parsed body against `key`'s entry for `status`; raise and record on a mismatch."""
    entry = _contract()["responses"].get(key)
    if entry is None:
        _fail(key, status, "", "the route is not in the contract's `responses`")
    if str(status) not in entry:
        _fail(
            key,
            status,
            "",
            f"status {status} is not declared (declared: {sorted(k for k in entry if k[:1].isdigit())})",
        )
    errors = list(validator_for(key, str(status)).iter_errors(body))
    if errors:
        from jsonschema.exceptions import best_match

        # A union (`anyOf`, e.g. a nullable `data`) reports only that no branch matched; name the
        # deepest error inside it for the reader, the one a fix must address.
        def leaves(error: Any) -> list[Any]:
            inner = [leaf for sub in (error.context or []) for leaf in leaves(sub)]
            return inner or [error]

        candidates = [
            leaf
            for error in errors
            for leaf in leaves(error)
            if not (
                leaf.validator == "type"
                and leaf.instance is not None
                and leaf.validator_value == "null"
            )
        ]
        chosen = max(candidates or errors, key=lambda e: len(list(e.absolute_path)))
        if not candidates:
            chosen = best_match(errors)
        _fail(key, status, _json_path(chosen), _message(chosen))


def _fail(key: str, status: int | str, json_path: str, message: str) -> None:
    error = AnswerContractError(key, status, json_path, message)
    FAILURES.append(error)
    raise error


def _check(route: APIRoute, request: Any, response: Any) -> None:
    from starlette.responses import StreamingResponse

    status = getattr(response, "status_code", 0)
    if not 200 <= status < 300 or isinstance(response, StreamingResponse):
        return
    media = (response.headers.get("content-type") or "").split(";")[0].strip()
    if media != "application/json":
        return
    endpoint = route.endpoint
    if not (getattr(endpoint, "__module__", "") or "").startswith("miserver."):
        return
    name = endpoint_name(endpoint)
    by_method = endpoint_map().get(name)
    if not by_method or request.method not in by_method:
        _fail(
            f"{request.method} {route.path} ({name})",
            status,
            "",
            "the endpoint is not in the contract",
        )
    body = json.loads(bytes(response.body)) if response.body else None
    check_body(by_method[request.method], status, body)
