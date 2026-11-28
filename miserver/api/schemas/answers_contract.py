"""The published answer contract, DERIVED from the live route table (044, BRD-MILLM-SHAPES-001).

`docs/schemas/miserver-answers-v1.json` describes every answer miServer serves: per route and 2xx status, its
JSON body schema; the routes that answer no JSON (`excluded`, with the media type their declaration names); the two
error envelopes; the vendored document schemas the answers reference (`external`); and coverage counts. It is
generated from `register_routes` -> FastAPI's OpenAPI document, never written by hand, and pinned by
`tests/unit/api/test_answers_contract.py`. Regenerate with `MISERVER_WRITE_CONTRACTS=1` on that test; the writer
(`write_contract`) refuses a breaking change (FR-044.16), which needs `miserver-answers-v2.json` beside v1.

Transforms, in order (FTDD §6.2): references rewritten to `#/$defs/` (`weights_contract._rewrite`, TD12); every
always-sent field `required` (SH7, TD4); the approval `result` in its history form (TD14); no
`additionalProperties: false` (SH7); external references resolved to vendored files (SH8); only reachable
definitions kept (FR-044.15); coverage counted (FR-044.19); the digest computed (TD11).
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from miserver.api.schemas import contract_tools as tools
from miserver.api.schemas.weights_contract import _rewrite, render

CONTRACT_FILENAME = "miserver-answers-v1.json"
KIND = "miserver.answer-contract/v1"
CONTRACT_ID = "https://hitsai.net/schemas/" + CONTRACT_FILENAME
SCHEMAS_DIR = Path(__file__).resolve().parents[3] / "docs" / "schemas"
_OPENAPI_REF = "#/components/schemas/"
_JSON = "application/json"
_DEFAULT_DESCRIPTION = "Successful Response"
_EXCLUDE_FLAGS = (
    "response_model_exclude_none",
    "response_model_exclude_unset",
    "response_model_exclude_defaults",
)
#: The one exception to SH7, stated in the file and on the approval read (TD14, FR-044.8).
HISTORY_EXCEPTION = (
    "The one exception to 'required means always sent' (SH7): an approval's stored `result` is checked "
    "in its HISTORY FORM. Each variant keeps its model's properties and types but has no `required` at "
    "any depth, because a stored row can predate a field (TD14). The variant is chosen by `action`; an "
    "action that is no longer listed leaves `result` unconstrained."
)
_DESCRIPTION = (
    "GENERATED from miServer's registered routes (their response declarations, via FastAPI's OpenAPI "
    "document); do not edit by hand. Regenerate with MISERVER_WRITE_CONTRACTS=1 on "
    "tests/unit/api/test_answers_contract.py; the writer refuses a breaking change, which needs "
    "miserver-answers-v2.json beside this file. Keys of `responses` are '<METHOD> <path>'; each value "
    "maps a 2xx status to that answer's JSON body schema. `required` lists every field a route always "
    "sends; extra keys are allowed (SH7). " + HISTORY_EXCEPTION
)


class BreakingAnswerChange(Exception):
    """The writer met a change FR-044.16 classifies as breaking (or one no row names)."""

    def __init__(self, changes: list[tools.Change]) -> None:
        self.changes = changes
        lines = "\n".join(f"  {c}" for c in changes)
        super().__init__(
            f"{len(changes)} breaking change(s) to {CONTRACT_FILENAME}:\n{lines}\n"
            "a breaking change needs miserver-answers-v2.json beside v1 (deferred-renames procedure C)"
        )


def _path_key(path: str) -> str:
    """The OpenAPI form of a route path: converter suffixes dropped (`{repo_id:path}` -> `{repo_id}`)."""
    return re.sub(r"\{(\w+):\w+\}", r"{\1}", path)


def _deref(schema: Any, defs: dict[str, Any]) -> Any:
    return tools._deref(schema, defs)


def _members(schema: Any, defs: dict[str, Any]) -> list[Any]:
    """The non-null alternatives of a schema: itself, or its `anyOf`/`oneOf` members, flattened
    through references (a `RootModel` union behind an optional field is one level down)."""
    resolved = _deref(schema, defs)
    for key in ("anyOf", "oneOf"):
        if isinstance(resolved, dict) and key in resolved:
            return [m for member in resolved[key] for m in _members(member, defs)]
    return [] if isinstance(resolved, dict) and resolved.get("type") == "null" else [schema]


def _is_union(entry: Any, defs: dict[str, Any]) -> bool:
    if len(_members(entry, defs)) > 1:
        return True
    resolved = _deref(entry, defs)
    props = resolved.get("properties") if isinstance(resolved, dict) else None
    if isinstance(props, dict) and {"success", "data", "error"} <= set(props):
        return len(_members(props["data"], defs)) > 1
    return False


def _is_pending(member: Any, defs: dict[str, Any]) -> bool:
    """An envelope whose `data` is the pending approval (`ApprovalPending`: `status` is the constant `pending`)."""
    props = _deref(member, defs).get("properties") or {}
    for data in _members(props.get("data", {}), defs):
        status = (_deref(data, defs).get("properties") or {}).get("status") or {}
        if status.get("const") == "pending":
            return True
    return False


def _external_files() -> dict[str, tuple[str, str]]:
    """Every vendored document schema by its `$id`: (file, sha256)."""
    out: dict[str, tuple[str, str]] = {}
    for path in sorted(SCHEMAS_DIR.glob("*.json")):
        if path.name == CONTRACT_FILENAME:
            continue
        raw = path.read_bytes()
        try:
            doc = json.loads(raw)
        except ValueError:
            continue
        sid = doc.get("$id") if isinstance(doc, dict) else None
        if isinstance(sid, str):
            out[sid] = (f"docs/schemas/{path.name}", hashlib.sha256(raw).hexdigest())
    return out


def _error_defs(components: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """`errors.management` (the envelope with `RefusalDetails`, SH10) and `errors.openai`, through Pydantic's
    serialization schema as FastAPI renders answers. Neither is declared by a route, so the served components (and
    the weights contract that embeds `ErrorDetails`) do not move."""
    from pydantic.json_schema import models_json_schema

    from miserver.api.schemas.common import RefusalEnvelope
    from miserver.api.schemas.openai import OpenAIErrorResponse

    _, schema = models_json_schema(
        [(RefusalEnvelope, "serialization"), (OpenAIErrorResponse, "serialization")],
        ref_template=_OPENAPI_REF + "{model}",
    )
    # A name the served document already has (`OpenAIError`, `OpenAIErrorResponse`) is kept as FastAPI rendered
    # it; only the two refusal models, which no route declares, come from Pydantic directly.
    extra = {name: body for name, body in schema.get("$defs", {}).items() if name not in components}
    errors = {
        "management": {"$ref": _OPENAPI_REF + "RefusalEnvelope"},
        "openai": {"$ref": _OPENAPI_REF + "OpenAIErrorResponse"},
    }
    return errors, extra


def _media_reason(response: dict[str, Any], media: str, content: dict[str, Any]) -> str:
    reason = content.get("x-reason")
    if isinstance(reason, str) and reason:
        return reason
    text = response.get("description")
    if isinstance(text, str) and text and text != _DEFAULT_DESCRIPTION:
        return text
    return f"the route answers {media}, not JSON"


def _history_result(defs: dict[str, Any], own_data: dict[str, list[Any]]) -> None:
    """TD8 / TD14: the approval read's `result`, tagged by `action`. For each action, `result` is one of the
    history forms of the `data` schemas of that action's gated routes, or null; a route whose `data` is null is
    stored as `{"value": null}` (`approval_service._result_of`). An action with no branch is unconstrained."""
    for name, body in list(defs.items()):
        props = body.get("properties") if isinstance(body, dict) else None
        if not isinstance(props, dict):
            continue
        marked = [
            p for p, s in props.items() if isinstance(s, dict) and s.pop(tools.HISTORY_FORM, False)
        ]
        if not marked:
            continue
        if "action" not in props or marked != ["result"]:
            raise ValueError(
                f"{name}: the history form is defined for an approval's `result` beside its `action`"
            )
        props["result"] = {k: v for k, v in props["result"].items() if k == "description"}
        branches = []
        for action in sorted(own_data):
            variants: list[Any] = []
            for schema in own_data[action]:
                variant = tools.history_form(schema, defs)
                if variant not in variants:
                    variants.append(variant)
            variants.append({"type": "null"})
            branches.append(
                {
                    "if": {"properties": {"action": {"const": action}}, "required": ["action"]},
                    "then": {"properties": {"result": {"anyOf": variants}}},
                }
            )
        body["allOf"] = [*body.get("allOf", []), *branches]
        body["description"] = (
            body.get("description", "").rstrip() + "\n\n" + HISTORY_EXCEPTION
        ).strip()


def _answer_models(app: Any) -> set[type]:
    """Every Pydantic model an answer of `app` can contain: the routes' `response_model` and `responses`
    models, followed through field annotations and generic arguments."""
    import typing

    from fastapi.routing import APIRoute, iter_route_contexts
    from pydantic import BaseModel

    found: set[type] = set()

    def visit(tp: Any) -> None:
        origin = typing.get_origin(tp)
        if origin is not None:
            for arg in typing.get_args(tp):
                visit(arg)
            return
        if isinstance(tp, type) and issubclass(tp, BaseModel) and tp not in found:
            found.add(tp)
            meta = getattr(tp, "__pydantic_generic_metadata__", None) or {}
            for arg in meta.get("args", ()):
                visit(arg)
            for field in tp.model_fields.values():
                visit(field.annotation)

    for ctx in iter_route_contexts(app.routes):
        route = ctx.original_route
        if isinstance(route, APIRoute):
            visit(route.response_model)
            for response in (route.responses or {}).values():
                visit(response.get("model"))
    return found


def _mark_exclude_if(app: Any, components: dict[str, Any]) -> None:
    """A field Pydantic leaves out of the serialized answer when its value is None (`exclude_if`) is a
    field some answer does not carry, exactly as `may_be_absent()` declares (TD4); mark it so
    `always_sent_required` does not require it. Read from the models, so the served components keep
    their bytes (the scoring contract embeds some of them). A model that cannot be matched to its
    component fails closed."""
    for model in sorted(_answer_models(app), key=lambda m: (m.__module__, m.__qualname__)):
        absent = [
            field.serialization_alias or field.alias or name
            for name, field in model.model_fields.items()
            if getattr(field, "exclude_if", None) is not None
        ]
        if not absent:
            continue
        qualified = model.__module__.replace(".", "__") + "__" + model.__name__
        names = [
            n for n in components
            if n in (model.__name__, qualified)
            or n in (f"{model.__name__}-Output", f"{qualified}-Output")
        ]
        if not names:
            raise ValueError(f"{model.__module__}.{model.__qualname__} has exclude_if fields "
                             "but no served component of its name")
        for name in names:
            props = components[name].get("properties", {})
            for prop in absent:
                if prop not in props:
                    raise ValueError(f"{name}: exclude_if field {prop!r} is not in its component")
                props[prop][tools.MAY_BE_ABSENT] = True


#: What FastAPI merges beside a `responses={…}` model's `$ref` when the route also serves a `response_model`
#: (e.g. `dict[str, Any]`): the served model's own `type`, `additionalProperties` and generated `title`.
_SERVED_MODEL_SIBLINGS = ("type", "additionalProperties", "title")


def _documented_only(schema: Any) -> Any:
    """A 2xx answer schema as documented: the `responses={…}` model's `$ref` without the siblings FastAPI merges
    in from the served `response_model`.

    ENH-013 decision (operator, 2026-11-26): documentation-only typing must not change a route's serializer, so
    a typed route keeps the `response_model` it served before (FastAPI serializes through it) and documents its
    answer in `responses={…}`. FastAPI then writes the documented `$ref` and merges the served model's schema
    keys beside it; the contract publishes the documented answer only.
    """
    if isinstance(schema, dict) and "$ref" in schema:
        return {k: v for k, v in schema.items() if k not in _SERVED_MODEL_SIBLINGS}
    return schema


def contract_from(app: Any, spec: dict[str, Any]) -> dict[str, Any]:
    """The contract for `app`'s live route table, read from the OpenAPI document `spec` (normally `app.openapi()`;
    the live check of FR-044.24 passes the pod's)."""
    from fastapi.routing import APIRoute, iter_route_contexts

    from miserver.api import approval_gate

    components: dict[str, Any] = copy.deepcopy(spec.get("components", {}).get("schemas", {}))
    responses: dict[str, Any] = {}
    excluded: dict[str, Any] = {}
    reached: set[str] = set()
    for ctx in iter_route_contexts(app.routes):
        route = ctx.original_route
        if not isinstance(route, APIRoute) or not ctx.include_in_schema:
            continue
        path = _path_key(str(ctx.path))
        flags = [f for f in _EXCLUDE_FLAGS if getattr(route, f, False)]
        if flags:
            raise ValueError(
                f"{sorted(ctx.methods or ())} {path} sets {flags}: 'always sent' would be false (FR-044.14)"
            )
        for method in sorted(ctx.methods or ()):
            key = f"{method} {path}"
            operation = spec["paths"][path][method.lower()]
            entry: dict[str, Any] = {}
            other: dict[str, str] = {}
            for status, response in sorted(operation.get("responses", {}).items()):
                if not status.startswith("2"):
                    continue
                content = response.get("content") or {}
                for media, body in sorted(content.items()):
                    if media == _JSON:
                        entry[status] = _rewrite(_documented_only(body.get("schema", {})), reached)
                    else:
                        other[media] = _media_reason(response, media, body)
            if entry:
                if other:
                    entry["x-variants"] = dict(sorted(other.items()))
                action = operation.get(approval_gate.X_APPROVAL_ACTION)
                if action:
                    entry["x-approval-action"] = action
                responses[key] = entry
            elif other:
                media = sorted(other)[0]
                excluded[key] = {"media_type": media, "reason": other[media]}
            else:
                raise ValueError(f"{key} declares no 2xx body at all")

    _mark_exclude_if(app, components)
    errors, error_defs = _error_defs(components)
    defs = {name: _rewrite(body, reached) for name, body in {**components, **error_defs}.items()}
    errors = _rewrite(errors, reached)

    # SH7 (TD4): `required` is every field a model always sends. Before the history copies are made.
    tools.always_sent_required(defs)
    tools.always_sent_required(responses)

    # TD8 / TD14: the gated routes' own `data`, by action, read from the contract entries just built.
    own_data: dict[str, list[Any]] = {}
    for method, raw_path, action in approval_gate.gated_routes(app):
        key = f"{method} {_path_key(raw_path)}"
        route = next(
            c.original_route
            for c in iter_route_contexts(app.routes)
            if isinstance(c.original_route, APIRoute)
            and _path_key(str(c.path)) == _path_key(raw_path)
            and method in (c.methods or ())
        )
        status = str(route.status_code or 200)
        envelopes = [m for m in _members(responses[key][status], defs) if not _is_pending(m, defs)]
        if len(envelopes) != 1:
            raise ValueError(
                f"{key}: cannot tell its own answer from the pending approval ({envelopes})"
            )
        data = (_deref(envelopes[0], defs).get("properties") or {}).get("data", {})
        members = _members(data, defs)
        own_data.setdefault(action, []).extend(
            members
            or [
                {"type": "object", "properties": {"value": {"type": "null"}}, "required": ["value"]}
            ]
        )
    _history_result(defs, own_data)

    # SH7: extra keys are allowed for consumers.
    tools.open_objects(defs)
    tools.open_objects(responses)

    # SH8: an external reference must name a vendored file by its `$id`; the generator fails closed.
    known = _external_files()
    external: dict[str, Any] = {}
    for schema in tools._walk([responses, defs]):
        ref = schema.get("$ref")
        if isinstance(ref, str) and ref.startswith("https://"):
            if ref not in known:
                raise ValueError(
                    f"unknown external $ref {ref!r}: no docs/schemas/*.json has that $id"
                )
            for sibling in ("type", "additionalProperties", "title"):
                schema.pop(sibling, None)
            file, sha = known[ref]
            external[ref] = {"file": file, "sha256": sha}

    kept = tools.reachable_defs([responses, errors], defs)
    ext = frozenset(external)
    typed = sum(
        1
        for entry in responses.values()
        if all(tools.is_typed(s, kept, ext) for k, s in entry.items() if k[:1].isdigit())
    )
    unions = sum(
        1
        for entry in responses.values()
        for k, s in entry.items()
        if k[:1].isdigit() and _is_union(s, kept)
    )
    coverage = {
        "typed": typed,
        "unions": unions,
        "excluded": len(excluded),
        "untyped": len(responses) - typed,
    }

    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": CONTRACT_ID,
        "kind": KIND,
        "title": "miServer answers",
        "description": _DESCRIPTION,
        "responses": dict(sorted(responses.items())),
        "excluded": dict(sorted(excluded.items())),
        "errors": errors,
        "external": dict(sorted(external.items())),
        "coverage": coverage,
        "$defs": kept,
    }


def answers_contract() -> dict[str, Any]:
    """The contract body (no `revision`, no `content_sha256`: the writer sets both) of this process's routes."""
    from fastapi import FastAPI

    from miserver.api.routes import register_routes

    app = FastAPI()
    register_routes(app)
    return contract_from(app, app.openapi())


def body_of(doc: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in doc.items() if k not in tools.UNDIGESTED}


def finished(body: dict[str, Any], revision: int) -> dict[str, Any]:
    """`body` with its `revision` and `content_sha256` (TD11)."""
    doc = dict(body)
    doc["revision"] = revision
    doc["content_sha256"] = tools.content_digest(body)
    return doc


def resolver(*docs: dict[str, Any]) -> tools.Resolver:
    """Reads each vendored schema the documents' `external` names, by sha256, from this checkout's
    working tree and git history (FR-044.16's re-vendor row)."""
    files = {sid: row["file"] for doc in docs for sid, row in (doc.get("external") or {}).items()}
    return tools.git_resolver(SCHEMAS_DIR.parents[1], files)


def write_contract(path: Path, body: dict[str, Any] | None = None) -> dict[str, Any]:
    """Write `path` from `body` (default: this process's contract), refusing a breaking change (FR-044.16).

    Equal bodies: nothing is written. All changes additive: `revision` + 1. No committed file: revision 1. Any
    breaking change: `BreakingAnswerChange`, and the committed bytes are left as they were."""
    body = answers_contract() if body is None else body_of(body)
    if path.is_file():
        old = json.loads(path.read_text(encoding="utf-8"))
        if body_of(old) == body:
            return old
        bad = tools.breaking(tools.classify(body_of(old), body, resolver(old, body)))
        if bad:
            raise BreakingAnswerChange(bad)
        doc = finished(body, int(old.get("revision", 0)) + 1)
    else:
        doc = finished(body, 1)
    path.write_text(render(doc), encoding="utf-8")
    return doc


def problems(doc: dict[str, Any]) -> list[str]:
    """What is wrong with a committed contract document on its own (FR-044.18): a hand edit leaves
    `content_sha256` stale, because only the writer sets it."""
    out = []
    revision = doc.get("revision")
    if not isinstance(revision, int) or isinstance(revision, bool) or revision < 1:
        out.append(f"revision {revision!r} is not a positive integer")
    if doc.get("content_sha256") != tools.content_digest(doc):
        out.append(
            "content_sha256 does not match the content: was it edited by hand? Regenerate it "
            "with the writer (MISERVER_WRITE_CONTRACTS=1)"
        )
    return out
