"""The JSON Schema subset `response_format: json_schema` accepts (Feature 25, FR-25.10.9, FR-25.11.2).

THE GUARANTEE: a schema is either inside a subset whose every keyword the installed constrained
decoder ENFORCES, or it is refused before any model load, naming the keyword and its JSON pointer.

THE DEFECT IT PREVENTS: xgrammar 0.2.8 compiles `multipleOf`, `not` and `uniqueItems` and does
NOT enforce them (measured, 025_FTDD §3.1) — a violating string is accepted. A schema using one
would produce output that "conforms" to a constraint nobody applied. The subset is therefore
miServer's own allowlist, not "whatever the library compiles", and
`tests/unit/api/test_json_schema_subset.py` probes the INSTALLED library for every allowlisted
keyword: a keyword joins `ALLOWED_KEYWORDS` only with a violating example the library rejects. A
second line of defence validates every complete output with `jsonschema`
(`miserver/ml/constrained_decoding.py::validate_output`).

Pure, no imports from `services` or `ml`; runs in the route between the request policy and the
auto-load.

THE LLAMA.CPP ENGINE (2026-11-20). A GGUF model is constrained by a GBNF grammar that
llama-cpp-python's own converter (`llama_cpp.llama_grammar.SchemaConverter`, 0.3.36) builds from
the schema. That converter takes ONE branch per schema node and silently drops every keyword the
branch does not read: `minimum`/`maximum` are a TODO, `pattern` is a lossy regex translation that
can produce a grammar llama.cpp rejects natively, `minLength` without `type: "string"` and
`minItems` without `items` are dropped, `$ref`/`anyOf`/`enum`/`const` win over their siblings,
and a `required` name absent from `properties` vanishes. `check_llamacpp` mirrors that branch
order and refuses (`field_not_honoured`, naming each keyword and pointer) any keyword the taken
branch would drop, so a GGUF request either gets every keyword enforced or a 400 before the load.
The `jsonschema` validation of every complete output still runs as the second line of defence.
"""

from __future__ import annotations

import json
from typing import Any

from miserver.core.errors import FieldNotHonouredError, ResponseFormatUnsupportedError

#: Constraint keywords (025_FTDD §4.3). Each has an enforcement probe in the test suite.
ALLOWED_KEYWORDS: frozenset[str] = frozenset({
    "type", "properties", "required", "additionalProperties", "items", "minItems", "maxItems",
    "enum", "const", "minimum", "maximum", "minLength", "maxLength", "pattern", "anyOf",
    "$ref", "$defs",
})
#: Annotations: no constraint, so nothing to enforce; their values are not walked. `$comment`,
#: `deprecated`, `readOnly` and `writeOnly` (2026-11-21) are JSON Schema 2020-12 annotations as well
#: (xgrammar compiles them as nothing, measured). `optional` is NOT JSON Schema: weightspace's
#: `tool()` helper uses `{"optional": True}` to mean "leave out of `required`" and removes it before
#: the schema is sent; a client that sends it literally has it treated as an annotation, with no
#: effect on the grammar or on validation (`required` alone decides what must be present).
ANNOTATIONS: frozenset[str] = frozenset({
    "title", "description", "default", "examples", "$comment", "deprecated", "readOnly",
    "writeOnly", "optional",
})

JSON_TYPES = frozenset({"string", "number", "integer", "boolean", "object", "array", "null"})

MAX_SCHEMA_BYTES = 64 * 1024
MAX_DEPTH = 32
MAX_PROPERTIES = 256
MAX_REPORTED = 20


def _pointer(path: list[str]) -> str:
    return "/" + "/".join(p.replace("~", "~0").replace("/", "~1") for p in path) if path else ""


def _walk(node: Any, path: list[str], depth: int, out: list[tuple[str, str]]) -> None:
    if len(out) >= MAX_REPORTED:
        return
    if depth > MAX_DEPTH:
        out.append(("(depth)", _pointer(path)))
        return
    if isinstance(node, bool):
        return  # `true` / `false` schemas
    if not isinstance(node, dict):
        out.append(("(not a schema)", _pointer(path)))
        return
    for key, value in node.items():
        here = path + [key]
        if key in ANNOTATIONS:
            continue
        if key not in ALLOWED_KEYWORDS:
            out.append((key, _pointer(here)))
            continue
        if key == "type":
            kinds = value if isinstance(value, list) else [value]
            if not kinds or any(not isinstance(k, str) or k not in JSON_TYPES for k in kinds):
                out.append(("type", _pointer(here)))
        elif key in ("properties", "$defs"):
            if not isinstance(value, dict):
                out.append((key, _pointer(here)))
                continue
            if key == "properties" and len(value) > MAX_PROPERTIES:
                out.append(("properties (more than 256)", _pointer(here)))
                continue
            for name, sub in value.items():
                _walk(sub, here + [name], depth + 1, out)
        elif key in ("items", "additionalProperties"):
            # `items` as a LIST is the draft-4 tuple form, which the subset does not include.
            if key == "items" and isinstance(value, list):
                out.append(("items (tuple form)", _pointer(here)))
            else:
                _walk(value, here, depth + 1, out)
        elif key == "anyOf":
            if not isinstance(value, list) or not value:
                out.append(("anyOf", _pointer(here)))
                continue
            for i, sub in enumerate(value):
                _walk(sub, here + [str(i)], depth + 1, out)
        elif key == "$ref":
            if not isinstance(value, str) or not value.startswith("#/$defs/"):
                out.append(("$ref (only local #/$defs/ references)", _pointer(here)))
        elif key == "required":
            if not isinstance(value, list) or any(not isinstance(v, str) for v in value):
                out.append(("required", _pointer(here)))
        # enum, const, minimum, maximum, minLength, maxLength, minItems, maxItems, pattern:
        # values, not schemas — nothing to walk.


def offending_keywords(schema: Any) -> list[tuple[str, str]]:
    """Every (keyword, JSON pointer) outside the subset, up to 20."""
    out: list[tuple[str, str]] = []
    _walk(schema, [], 0, out)
    return out


def check(schema: Any) -> None:
    """Refuse a schema outside the subset or over the size caps.

    Raises:
        ResponseFormatUnsupportedError: naming the first keyword and listing up to 20, each with
            its JSON pointer (`details["keywords"]`).
    """
    try:
        size = len(json.dumps(schema, separators=(",", ":")).encode("utf-8"))
    except (TypeError, ValueError) as exc:
        raise ResponseFormatUnsupportedError(
            f"response_format schema is not serialisable JSON: {exc}",
            details={"param": "response_format"},
        ) from exc
    if size > MAX_SCHEMA_BYTES:
        raise ResponseFormatUnsupportedError(
            f"response_format schema is {size} bytes; the limit is {MAX_SCHEMA_BYTES}",
            details={"param": "response_format", "schema_bytes": size},
        )
    offenders = offending_keywords(schema)
    if offenders:
        listed = ", ".join(f"{k} at {p or '/'}" for k, p in offenders)
        raise ResponseFormatUnsupportedError(
            f"response_format schema uses what the supported JSON Schema subset does not "
            f"include: {listed}. Supported keywords: {', '.join(sorted(ALLOWED_KEYWORDS))}; "
            f"annotations: {', '.join(sorted(ANNOTATIONS))}.",
            details={
                "param": "response_format",
                "keywords": [{"keyword": k, "pointer": p} for k, p in offenders],
            },
        )


# =============================================================================
# The llama.cpp engine (2026-11-20): what llama-cpp-python's grammar converter ENFORCES
# =============================================================================

#: In the shared subset but never enforced by llama-cpp-python 0.3.36's converter: `minimum` and
#: `maximum` are a TODO there (the number rule is unbounded), and `pattern` is translated by a
#: partial regex-to-GBNF port that passes `\d`-style escapes through to a grammar llama.cpp then
#: rejects natively.
LLAMACPP_UNENFORCED: frozenset[str] = frozenset({"minimum", "maximum", "pattern"})

#: The converter's primitive rules (`PRIMITIVE_RULES`): a bare `type` it can generate.
_LC_PRIMITIVES = frozenset({"string", "number", "integer", "boolean", "object", "array", "null"})

#: Keys that are never a constraint: annotations, and `$defs` (storage `$ref` reads).
_LC_INERT = ANNOTATIONS | {"$defs"}


def annotation_only(node: Any) -> bool:
    """A schema object of annotations alone (`{}` included; `$defs` is storage): in JSON Schema it
    constrains nothing and accepts ANY JSON value, so a grammar of any JSON value enforces all of it
    (weightspace's `write_tensor` sends `values: {"description": "nested list of numbers, or a
    scalar"}`)."""
    return isinstance(node, dict) and not (set(node) - _LC_INERT)


#: llama-cpp-python 0.3.36's converter has no JSON Schema spelling for "any value": `{}` becomes
#: its `object` rule, and a node with only annotations fails its `assert schema_type in
#: PRIMITIVE_RULES`. Its primitive table does hold the generic `value` rule (object | array |
#: string | number | boolean | null), reached by this `type`, so that is what such a node becomes.
LLAMACPP_ANY_VALUE: dict[str, Any] = {"type": "value"}


def for_llamacpp_converter(schema: Any, *, rewrite_root: bool = True) -> Any:
    """A copy of `schema` for the converter only: every annotation-only subschema (`{}` included)
    is `LLAMACPP_ANY_VALUE`, its `$defs` kept. The request's own schema is what the model is shown
    and what output is validated against; this copy only builds the grammar. `rewrite_root=False`
    keeps the root itself (a tool's parameters, which must stay an object)."""

    def sub(node: Any) -> Any:
        if not isinstance(node, dict):
            return node
        if annotation_only(node):
            out: dict[str, Any] = dict(LLAMACPP_ANY_VALUE)
            if isinstance(node.get("$defs"), dict):
                out["$defs"] = {k: sub(v) for k, v in node["$defs"].items()}
            return out
        return children(node)

    def children(node: dict[str, Any]) -> dict[str, Any]:
        out = dict(node)
        for key in ("properties", "$defs"):
            if isinstance(out.get(key), dict):
                out[key] = {k: sub(v) for k, v in out[key].items()}
        for key in ("items", "additionalProperties"):
            if isinstance(out.get(key), dict):
                out[key] = sub(out[key])
        for key in ("anyOf", "oneOf"):
            if isinstance(out.get(key), list):
                out[key] = [sub(v) for v in out[key]]
        return out

    if not isinstance(schema, dict):
        return schema
    return sub(schema) if rewrite_root else children(schema)


def _lc_type_admits(kind: str, value: Any) -> bool:
    """Whether a JSON value is of JSON Schema type `kind` (for `type` beside `enum`/`const`)."""
    if kind == "null":
        return value is None
    if kind == "boolean":
        return isinstance(value, bool)
    if kind == "integer":
        return (isinstance(value, int) and not isinstance(value, bool)) or (
            isinstance(value, float) and value.is_integer()
        )
    if kind == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if kind == "string":
        return isinstance(value, str)
    if kind == "array":
        return isinstance(value, list)
    if kind == "object":
        return isinstance(value, dict)
    return False


def _lc_resolves(root: Any, ref: str) -> bool:
    node = root
    for part in ref[2:].split("/"):
        part = part.replace("~1", "/").replace("~0", "~")
        if not isinstance(node, dict) or part not in node:
            return False
        node = node[part]
    return True


def _lc_walk(
    node: Any, path: list[str], root: Any, out: list[tuple[str, str]], depth: int = 0,
    validated: frozenset[str] = frozenset(),
) -> None:
    """Mirror `SchemaConverter.visit`'s branch order; record every keyword the branch drops.

    `validated`: unenforced keywords the caller enforces by validating the output instead (tool
    parameters' `minimum`/`maximum`, `TOOL_VALIDATED`), so they are not reported.
    """
    if len(out) >= MAX_REPORTED or depth > MAX_DEPTH:
        return
    if isinstance(node, bool) or not isinstance(node, dict):
        # The converter calls `.get` on every node: a boolean schema raises.
        out.append(("boolean schema (the llama.cpp converter needs an object)", _pointer(path)))
        return
    keys = set(node) - _LC_INERT
    for key in sorted(keys & (LLAMACPP_UNENFORCED - validated)):
        out.append((f"{key} (not enforced by the llama.cpp grammar)", _pointer(path + [key])))
    keys -= LLAMACPP_UNENFORCED
    kind = node.get("type")

    def dropped(consumed: set[str], branch: str) -> None:
        for key in sorted(keys - consumed):
            out.append((f"{key} (dropped beside {branch} by the llama.cpp grammar)",
                        _pointer(path + [key])))

    def walk(sub: Any, *more: str) -> None:
        _lc_walk(sub, path + list(more), root, out, depth + 1, validated)

    if "$ref" in node:
        ref = node["$ref"]
        if isinstance(ref, str) and ref.startswith("#/") and not _lc_resolves(root, ref):
            out.append(("$ref (target not found)", _pointer(path + ["$ref"])))
        dropped({"$ref"}, "$ref")
    elif "anyOf" in node:
        dropped({"anyOf"}, "anyOf")
        for i, sub in enumerate(node["anyOf"] if isinstance(node["anyOf"], list) else []):
            walk(sub, "anyOf", str(i))
    elif isinstance(kind, list):
        dropped({"type"}, "a list of types")
    elif "const" in node or "enum" in node:
        which = "const" if "const" in node else "enum"
        values = [node["const"]] if which == "const" else (
            node["enum"] if isinstance(node["enum"], list) else []
        )
        if "type" in node and any(not _lc_type_admits(kind, v) for v in values):
            out.append((f"type (the llama.cpp grammar generates the {which} values, and one is "
                        "not of this type)", _pointer(path + ["type"])))
        dropped({which, "type"}, which)
    elif kind in (None, "object") and (
        "properties" in node
        or ("additionalProperties" in node and node["additionalProperties"] is not True)
    ):
        properties = node.get("properties") if isinstance(node.get("properties"), dict) else {}
        for name in node.get("required") or []:
            if name not in properties:
                out.append((f"required '{name}' (not in properties, so the llama.cpp grammar "
                            "drops it)", _pointer(path + ["required"])))
        dropped({"type", "properties", "required", "additionalProperties"}, "properties")
        for name, sub in properties.items():
            walk(sub, "properties", name)
        extra = node.get("additionalProperties")
        if isinstance(extra, dict):
            walk(extra, "additionalProperties")
    elif kind in (None, "array") and "items" in node:
        dropped({"type", "items", "minItems", "maxItems"}, "items")
        walk(node["items"], "items")
    elif kind == "string" and ("minLength" in node or "maxLength" in node):
        dropped({"type", "minLength", "maxLength"}, "minLength/maxLength")
    elif annotation_only(node):
        # Annotations alone (or `{}`): any JSON value, generated as the converter's `value` rule
        # (`for_llamacpp_converter`). Nothing is left unenforced: the schema enforces nothing.
        pass
    elif kind == "object":
        # A generic object.
        dropped({"type", "additionalProperties"}, "type: object")
    elif isinstance(kind, str) and kind in _LC_PRIMITIVES:
        dropped({"type"}, f"type: {kind}")
    else:
        out.append(("(no type the llama.cpp converter can generate)", _pointer(path)))
    defs = node.get("$defs")
    for name, sub in (defs.items() if isinstance(defs, dict) else ()):
        walk(sub, "$defs", name)


def llamacpp_offending_keywords(schema: Any) -> list[tuple[str, str]]:
    """Every (keyword, JSON pointer) the llama.cpp grammar would not enforce, up to 20."""
    out: list[tuple[str, str]] = []
    _lc_walk(schema, [], schema, out)
    return out


def check_llamacpp(schema: Any, *, model_name: str | None = None) -> None:
    """Refuse a schema (already inside the shared subset) that llama.cpp would not fully enforce.

    Raises:
        FieldNotHonouredError: `param: response_format`, naming every keyword and its pointer
            (`details["keywords"]`).
    """
    offenders = llamacpp_offending_keywords(schema)
    if not offenders:
        return
    listed = "; ".join(f"{k} at {p or '/'}" for k, p in offenders)
    on_model = f" for model '{model_name}'" if model_name else ""
    raise FieldNotHonouredError(
        f"'response_format' is not honoured on /v1/chat/completions{on_model} with the llamacpp "
        f"engine: its grammar would not enforce every keyword of this schema: {listed}. "
        "Drop or restate these keywords, or use a transformers-served model.",
        details={
            "param": "response_format",
            "fields": ["response_format"],
            "endpoint": "chat",
            "engine": "llamacpp",
            "keywords": [{"keyword": k, "pointer": p} for k, p in offenders],
        },
    )


# =============================================================================
# Tool parameters on the llama.cpp engine (2026-11-21)
# =============================================================================

#: Unenforced by the grammar, yet ACCEPTED in a tool's parameters: the rendered tools show them to
#: the model and the parsed arguments are validated against them with `jsonschema` (a violating
#: call is a 500 `constrained_output_invalid`, never a 200). `response_format` still refuses them:
#: a document there has no second reader, while a tool call's bounds are also its executor's.
TOOL_VALIDATED: frozenset[str] = frozenset({"minimum", "maximum"})


def llamacpp_tool_offending_keywords(tools: Any) -> list[tuple[str, str]]:
    """Every (keyword, JSON pointer from the request root) in the tools' parameter schemas that the
    llama.cpp tool-call grammar would not enforce, up to 20: the shared subset first, then the
    converter's dropped keywords, `minimum`/`maximum` excepted."""
    out: list[tuple[str, str]] = []
    for i, tool in enumerate(tools or []):
        function = getattr(tool, "function", None)
        parameters = getattr(function, "parameters", None) if function is not None else None
        if parameters is None and isinstance(tool, dict):
            parameters = (tool.get("function") or {}).get("parameters")
        if parameters is None:
            continue
        base = ["tools", str(i), "function", "parameters"]
        found: list[tuple[str, str]] = []
        _walk(parameters, [], 0, found)
        if not found:
            _lc_walk(parameters, [], parameters, found, validated=TOOL_VALIDATED)
        for keyword, pointer in found:
            out.append((keyword, _pointer(base) + pointer))
            if len(out) >= MAX_REPORTED:
                return out
    return out


def check_llamacpp_tools(tools: Any, *, model_name: str | None = None) -> None:
    """Refuse tools whose parameter schemas the llama.cpp call grammar would not fully enforce.

    Raises:
        FieldNotHonouredError: `param: tools`, naming every keyword and its JSON pointer
            (`details["keywords"]`).
    """
    for i, tool in enumerate(tools or []):
        function = getattr(tool, "function", None)
        parameters = getattr(function, "parameters", None) if function is not None else None
        if parameters is None:
            continue
        try:
            size = len(json.dumps(parameters, separators=(",", ":")).encode("utf-8"))
        except (TypeError, ValueError):
            size = MAX_SCHEMA_BYTES + 1
        if size > MAX_SCHEMA_BYTES:
            raise FieldNotHonouredError(
                f"'tools' is not honoured on /v1/chat/completions with the llamacpp engine: "
                f"tools[{i}]'s parameters are {size} bytes; the limit is {MAX_SCHEMA_BYTES}",
                details={"param": "tools", "fields": ["tools"], "endpoint": "chat",
                         "engine": "llamacpp", "schema_bytes": size},
            )
    offenders = llamacpp_tool_offending_keywords(tools)
    if not offenders:
        return
    listed = "; ".join(f"{k} at {p}" for k, p in offenders)
    on_model = f" for model '{model_name}'" if model_name else ""
    raise FieldNotHonouredError(
        f"'tools' is not honoured on /v1/chat/completions{on_model} with the llamacpp engine: the "
        f"grammar that constrains a tool call would not enforce every keyword of these parameter "
        f"schemas: {listed}. Drop or restate these keywords, or use a transformers-served model "
        "(minimum and maximum are accepted: the arguments are validated against them).",
        details={
            "param": "tools",
            "fields": ["tools"],
            "endpoint": "chat",
            "engine": "llamacpp",
            "keywords": [{"keyword": k, "pointer": p} for k, p in offenders],
        },
    )
