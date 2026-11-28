"""Pure transforms for the published answer contract (044, FTID §3.1). STANDARD LIBRARY ONLY.

`answers_contract.py` builds the published answer contract from the live OpenAPI document with these
functions, the test validator derives its strict form with them, and the cross-repo consumer test
(`tests/unit/test_answers_contract_consumers.py`) classifies a vendored copy with `classify`. That last test runs in
the `cross-repo-contract` CI job, which installs only pytest, mcp, pydantic, pydantic-settings, sqlalchemy, httpx and
rfc8785, so nothing here may import outside the standard library (IC-2; held by
`tests/unit/test_answers_contract_consumers_need_only_job_deps.py`).
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

OPEN_KEY = "additionalProperties"
#: TD4: a field a route sometimes leaves out (`common.may_be_absent()`); never `required` in the contract.
MAY_BE_ABSENT = "x-may-be-absent"
#: TD14 (IC-3): a field whose stored value may predate its model; its schema keeps types, drops every `required`.
HISTORY_FORM = "x-history-form"
DEFS_REF = "#/$defs/"
HISTORY_SUFFIX = "__history"
#: The two top-level keys a digest leaves out (TD11): the writer sets both.
UNDIGESTED = ("revision", "content_sha256")


def render(doc: dict[str, Any]) -> str:
    """The committed byte form. Identical to `weights_contract.render`, which `answers_contract` uses (TD12); the copy
    here exists only so the digest can be recomputed where `miserver` is not importable."""
    return json.dumps(doc, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def _walk(node: Any):
    """Every dict below `node`, depth first (the node itself included)."""
    if isinstance(node, dict):
        yield node
        for value in node.values():
            yield from _walk(value)
    elif isinstance(node, list):
        for item in node:
            yield from _walk(item)


def always_sent_required(node: Any) -> None:
    """SH7 / FR-044.14, in place: every object schema with `properties` requires every property not marked
    `x-may-be-absent`, and the marker is removed. An object whose list comes out empty carries no `required`."""
    for schema in list(_walk(node)):
        props = schema.get("properties")
        if not isinstance(props, dict):
            continue
        required = []
        for name, prop in props.items():
            if isinstance(prop, dict) and prop.pop(MAY_BE_ABSENT, False):
                continue
            required.append(name)
        if required:
            schema["required"] = sorted(required)
        else:
            schema.pop("required", None)


def open_objects(node: Any) -> Any:
    """SH7, in place: delete every `additionalProperties: false`. A schema-valued `additionalProperties` (a map) stays."""
    for schema in _walk(node):
        if schema.get(OPEN_KEY) is False:
            del schema[OPEN_KEY]
    return node


def strict_form(node: Any) -> Any:
    """SH13 / TD5, on a COPY: every object with `properties` refuses keys it does not declare. Maps (only
    `additionalProperties`) are left alone, so a new card line in an approval `summary` is still accepted."""
    out = copy.deepcopy(node)

    def close(value: Any, conditional: bool) -> None:
        if isinstance(value, dict):
            # A condition (`if`/`then`/`else`) is a partial view of its object: closing it would refuse
            # every key the view does not name. Its own nested objects are closed as usual. A model
            # that DECLARES extras (`extra="allow"`, rendered `additionalProperties: true`) keeps
            # them: that is a declaration, not an omission (the weights handles carry PyTorch's
            # reducer fields); `test_answers_contract_coverage.py` pins which models may do it.
            if (
                not conditional
                and isinstance(value.get("properties"), dict)
                and OPEN_KEY not in value
            ):
                value[OPEN_KEY] = False
            for key, sub in value.items():
                if key == "properties" and isinstance(sub, dict):
                    for prop in sub.values():
                        close(prop, False)
                elif key == "allOf" and isinstance(sub, list):
                    # 044 R2-4: `allOf` members are views of ONE object, like a condition: a closed
                    # member would refuse the keys its sibling members declare.
                    for member in sub:
                        close(member, True)
                else:
                    close(sub, key in ("if", "then", "else"))
        elif isinstance(value, list):
            for item in value:
                close(item, False)

    close(out, False)
    return out


def _refs(node: Any) -> list[str]:
    return [
        s["$ref"][len(DEFS_REF) :]
        for s in _walk(node)
        if isinstance(s.get("$ref"), str) and s["$ref"].startswith(DEFS_REF)
    ]


def reachable_defs(roots: list[Any], defs: dict[str, Any]) -> dict[str, Any]:
    """FR-044.15: only the definitions a `#/$defs/<name>` reference reaches from `roots`, sorted by name."""
    seen: dict[str, Any] = {}
    pending = sorted({name for root in roots for name in _refs(root)})
    while pending:
        name = pending.pop()
        if name in seen:
            continue
        seen[name] = defs[name]
        pending.extend(sorted(set(_refs(defs[name])) - set(seen)))
    return dict(sorted(seen.items()))


def history_form(node: Any, defs: dict[str, Any]) -> Any:
    """TD14: a copy of `node` whose every reference points at a `<Name>__history` copy of its definition, made (in
    `defs`) with no `required` at any depth. The live definitions keep theirs. `__history` is not a suffix Pydantic
    produces, so a copy's name cannot collide with a component's."""
    out = copy.deepcopy(node)

    def strip(schema: Any) -> None:
        for s in _walk(schema):
            s.pop("required", None)
            ref = s.get("$ref")
            if isinstance(ref, str) and ref.startswith(DEFS_REF):
                name = ref[len(DEFS_REF) :]
                if not name.endswith(HISTORY_SUFFIX):
                    target = name + HISTORY_SUFFIX
                    s["$ref"] = DEFS_REF + target
                    if target not in defs:
                        defs[target] = copy.deepcopy(defs[name])
                        strip(defs[target])

    strip(out)
    return out


def _deref(schema: Any, defs: dict[str, Any]) -> Any:
    seen = 0
    while (
        isinstance(schema, dict)
        and isinstance(schema.get("$ref"), str)
        and schema["$ref"].startswith(DEFS_REF)
    ):
        schema = defs.get(schema["$ref"][len(DEFS_REF) :], {})
        seen += 1
        if seen > 50:
            return {}
    return schema


def is_typed(
    schema: Any,
    defs: dict[str, Any],
    external: frozenset[str] | set[str] = frozenset(),
    _depth: int = 0,
) -> bool:
    """FR-044.4. Typed: an object with named properties; an array of typed items; an `anyOf`/`oneOf` whose non-null
    members are all typed; the envelope whose `data` is typed or null-only; an external `$ref` listed in `external`.
    Not typed: `{}`, an object without properties (a map), an array of untyped items, anything else."""
    if _depth > 40 or not isinstance(schema, dict) or not schema:
        return False
    ref = schema.get("$ref")
    if isinstance(ref, str) and not ref.startswith(DEFS_REF):
        return ref in external
    schema = _deref(schema, defs)
    if not isinstance(schema, dict) or not schema:
        return False
    for key in ("anyOf", "oneOf"):
        if key in schema:
            members = [m for m in schema[key] if _deref(m, defs).get("type") != "null"]
            return bool(members) and all(is_typed(m, defs, external, _depth + 1) for m in members)
    props = schema.get("properties")
    if isinstance(props, dict) and {"success", "data", "error"} <= set(props):
        data = props["data"]
        if _deref(data, defs).get("type") == "null":
            return True
        members = [m for m in data.get("anyOf", [data]) if _deref(m, defs).get("type") != "null"]
        return not members or all(is_typed(m, defs, external, _depth + 1) for m in members)
    if isinstance(props, dict) and props:
        return True
    if schema.get("type") == "array":
        return is_typed(schema.get("items") or {}, defs, external, _depth + 1)
    return False


def content_digest(doc: dict[str, Any]) -> str:
    """TD11 / FR-044.18: sha256 of the rendered document with `revision` and `content_sha256` left out."""
    body = {k: v for k, v in doc.items() if k not in UNDIGESTED}
    return hashlib.sha256(render(body).encode("utf-8")).hexdigest()


# ── the classifier (FR-044.16; D6 plus SH12) ─────────────────────────────────────────────────────────────────

#: FR-044.16's table, by row name. Every row not marked additive is breaking, and so is anything no row names.
ADDITIVE_ROWS = frozenset(
    {
        "new route",
        "new status code",
        "new optional field",
        "new always-sent field",
        "descriptive",
        "vendored schema revised (additive)",
    }
)
UNLISTED = "unlisted change"
# `$comment` is JSON Schema's annotation for comments; like a description it validates nothing
# (044, the re-vendor row: the rule-set schema's last re-vendor changed only its `$comment`).
_DESCRIPTIVE = ("description", "title", "$comment")
_RETYPING = ("type", "format")
#: Keys that derive from others and are not compared on their own: `coverage` (from `responses`/`excluded`), and the
#: two the writer sets.
_DERIVED_TOP = ("coverage", "revision", "content_sha256")


@dataclass(frozen=True)
class Change:
    path: str
    row: str

    @property
    def breaking(self) -> bool:
        return self.row not in ADDITIVE_ROWS

    def __str__(self) -> str:
        return f"{self.row}: {self.path}"


#: `resolve(sid, sha256)` -> the vendored document with that `$id` whose bytes have that sha256, or None.
Resolver = Callable[[str, str], "dict[str, Any] | None"]


class _Classifier:
    def __init__(
        self, old: dict[str, Any], new: dict[str, Any], resolve: Resolver | None = None
    ) -> None:
        self.old_defs: dict[str, Any] = old.get("$defs", {}) or {}
        self.new_defs: dict[str, Any] = new.get("$defs", {}) or {}
        self.changes: list[Change] = []
        self.compared: set[tuple[str, str]] = set()
        self.resolve = resolve

    def external(self, a: dict[str, Any], b: dict[str, Any]) -> None:
        """FR-044.16's row for `external` (suite decision, 2026-11-25): an entry removed is breaking, a
        new one additive (its `$ref` site is classified where it appears), a moved file unlisted. The
        SAME `$id` and file with a NEW sha256 (a re-vendor) is classified by running this classifier on
        the old and the new vendored schema themselves: additive passes as one
        `vendored schema revised (additive)` change; anything else is breaking. A copy that cannot be
        read (no resolver, or its bytes not found by sha256) fails closed."""
        for sid in sorted(set(a) | set(b)):
            path = f"external.{sid}"
            if sid not in b:
                self.add(path, "removed")
            elif sid not in a:
                self.add(path, "descriptive")
            elif a[sid] != b[sid]:
                if a[sid].get("file") != b[sid].get("file"):
                    self.add(f"{path}.file", UNLISTED)
                    continue
                old_doc = self.resolve(sid, a[sid]["sha256"]) if self.resolve else None
                new_doc = self.resolve(sid, b[sid]["sha256"]) if self.resolve else None
                if old_doc is None or new_doc is None:
                    self.add(f"{path}.sha256", UNLISTED)
                    continue
                inner = _Classifier(old_doc, new_doc)
                inner.schema(_without_defs(old_doc), _without_defs(new_doc), path)
                inner.sweep()
                bad = [c for c in inner.changes if c.breaking]
                if bad:
                    self.changes.extend(bad)
                else:
                    self.add(f"{path}.sha256", "vendored schema revised (additive)")

    def sweep(self) -> None:
        """Compare every definition both sides hold that nothing compared yet (044 R1-4)."""
        for name in sorted(set(self.old_defs) & set(self.new_defs)):
            if (name, name) not in self.compared:
                self.compared.add((name, name))
                self.schema(self.old_defs[name], self.new_defs[name], f"$defs.{name}")

    def add(self, path: str, row: str) -> None:
        self.changes.append(Change(path, row))

    def schema(self, old: Any, new: Any, path: str) -> None:
        if old == new:
            # Equal text can still reference definitions whose CONTENT changed: follow its references.
            self._follow(new)
            return
        if not isinstance(old, dict) or not isinstance(new, dict):
            if old != new:
                self.add(path, UNLISTED)
            return
        old_ref, new_ref = old.get("$ref"), new.get("$ref")
        if old_ref != new_ref:
            self.add(f"{path}.$ref", "retyped")
        elif isinstance(old_ref, str) and old_ref.startswith(DEFS_REF):
            name = old_ref[len(DEFS_REF) :]
            if (name, name) not in self.compared:
                self.compared.add((name, name))
                if name not in self.new_defs:
                    self.add(f"$defs.{name}", "removed")
                elif name in self.old_defs:
                    self.schema(self.old_defs[name], self.new_defs[name], f"$defs.{name}")
        for key in sorted(set(old) | set(new)):
            if key == "$ref":
                continue
            a, b = old.get(key), new.get(key)
            sub = f"{path}.{key}"
            if a == b:
                self._follow(b)
                continue
            if key in _DESCRIPTIVE:
                self.add(sub, "descriptive")
            elif key in _RETYPING:
                self.add(sub, "retyped")
            elif key == "properties":
                self.properties(a or {}, b or {}, new, sub)
            elif key == "required":
                self.required(
                    set(a or ()),
                    set(b or ()),
                    old.get("properties") or {},
                    new.get("properties") or {},
                    path,
                )
            elif key == "enum":
                self.add(
                    sub,
                    "enum widened"
                    if set(map(_key, b or ())) - set(map(_key, a or ()))
                    else UNLISTED,
                )
            elif key == "const":
                self.add(sub, "enum widened")
            elif (
                key == "items"
                or (key == OPEN_KEY and isinstance(a, dict) and isinstance(b, dict))
                or (key in ("if", "then", "else") and isinstance(a, dict) and isinstance(b, dict))
            ):
                self.schema(a, b, sub)
            elif (
                key in ("anyOf", "oneOf", "allOf")
                and isinstance(a, list)
                and isinstance(b, list)
                and len(a) == len(b)
            ):
                for i, (x, y) in enumerate(zip(a, b, strict=True)):
                    self.schema(x, y, f"{sub}[{i}]")
            else:
                self.add(sub, UNLISTED)

    def _follow(self, node: Any) -> None:
        """Compare, once each, the definitions `node` references by name, old against new: equal text
        can still reference a definition whose content changed."""
        for name in sorted(set(_refs(node))):
            if (name, name) in self.compared:
                continue
            self.compared.add((name, name))
            if name not in self.new_defs:
                self.add(f"$defs.{name}", "removed")
            elif name in self.old_defs:
                self.schema(self.old_defs[name], self.new_defs[name], f"$defs.{name}")

    def properties(
        self,
        a: dict[str, Any],
        b: dict[str, Any],
        new: dict[str, Any],
        path: str,
    ) -> None:
        new_required = set(new.get("required") or ())
        for name in sorted(set(a) | set(b)):
            sub = f"{path}.{name}"
            if name not in b:
                self.add(sub, "removed")
            elif name not in a:
                self.add(
                    sub, "new always-sent field" if name in new_required else "new optional field"
                )
                self._follow(b[name])
            else:
                self.schema(a[name], b[name], sub)

    def required(
        self,
        a: set[str],
        b: set[str],
        old_props: dict[str, Any],
        new_props: dict[str, Any],
        path: str,
    ) -> None:
        for name in sorted(b - a):
            if name in old_props:
                self.add(f"{path}.required.{name}", "made required")
        for name in sorted(a - b):
            if name in new_props:
                self.add(f"{path}.required.{name}", "sometimes absent")

    def routes(self, old: dict[str, Any], new: dict[str, Any]) -> None:
        old_r, new_r = old.get("responses", {}) or {}, new.get("responses", {}) or {}
        old_x, new_x = old.get("excluded", {}) or {}, new.get("excluded", {}) or {}
        for key in sorted(set(old_r) | set(new_r) | set(old_x) | set(new_x)):
            was = "responses" if key in old_r else "excluded" if key in old_x else None
            now = "responses" if key in new_r else "excluded" if key in new_x else None
            if was and now and was != now:
                self.add(f"{now}.{key}", "moved between responses and excluded")
            elif was is None:
                self.add(f"{now}.{key}", "new route")
            elif now is None:
                self.add(f"{was}.{key}", "removed")
            elif now == "excluded":
                if old_x[key] != new_x[key]:
                    self.add(f"excluded.{key}", UNLISTED)
            else:
                self.statuses(old_r[key], new_r[key], f"responses.{key}")

    def statuses(self, a: dict[str, Any], b: dict[str, Any], path: str) -> None:
        for status in sorted(set(a) | set(b)):
            sub = f"{path}.{status}"
            if not status[:1].isdigit():
                if a.get(status) != b.get(status):
                    self.add(sub, UNLISTED)
                continue
            if status not in b:
                self.add(sub, "removed")
            elif status not in a:
                self.add(sub, "new status code")
                self._follow(b[status])
            else:
                self.schema(a[status], b[status], sub)

    def top(self, old: dict[str, Any], new: dict[str, Any]) -> None:
        self.routes(old, new)
        for key in sorted(set(old) | set(new)):
            if key in ("responses", "excluded", "$defs") or key in _DERIVED_TOP:
                continue
            a, b = old.get(key), new.get(key)
            if a == b:
                if key == "errors":
                    self._follow(b)
                continue
            if key in _DESCRIPTIVE:
                self.add(key, "descriptive")
            elif key == "external" and isinstance(a, dict) and isinstance(b, dict):
                self.external(a, b)
            elif key == "errors" and isinstance(a, dict) and isinstance(b, dict):
                for name in sorted(set(a) | set(b)):
                    if name not in b:
                        self.add(f"errors.{name}", "removed")
                    elif name not in a:
                        self.add(f"errors.{name}", UNLISTED)
                    else:
                        self.schema(a[name], b[name], f"errors.{name}")
            else:
                self.add(key, UNLISTED)
        # 044 R1-4: every definition both files hold is compared, even one reached only through a path
        # the walk above does not follow (an `excluded` entry, an `x-` key): nothing is left unseen.
        self.sweep()
        # A definition only the new file has, and that nothing compared above reached, is a new definition.
        for name in sorted(set(self.new_defs) - set(self.old_defs)):
            self.add(f"$defs.{name}", "descriptive")
        for name in sorted(set(self.old_defs) - set(self.new_defs)):
            if not any(c.path == f"$defs.{name}" for c in self.changes):
                self.add(f"$defs.{name}", "removed")


def _without_defs(doc: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in doc.items() if k != "$defs"}


def git_resolver(root: Any, files: dict[str, str]) -> Resolver:
    """A `Resolver` over a git checkout: `files` maps each `$id` to its path under `root`; the bytes
    are looked for in the working tree, then in every commit that touched the file, by sha256. Uses
    the `git` command only (standard library), so it also runs in the cross-repo CI job."""
    import subprocess
    from pathlib import Path

    def resolve(sid: str, sha: str) -> "dict[str, Any] | None":
        rel = files.get(sid)
        if rel is None:
            return None
        here = Path(root) / rel
        candidates: list[bytes] = [here.read_bytes()] if here.is_file() else []
        try:
            revs = subprocess.run(
                ["git", "rev-list", "--all", "--", rel],
                cwd=root,
                capture_output=True,
                text=True,
                check=True,
                timeout=60,
            ).stdout.split()
        except (OSError, subprocess.SubprocessError):
            revs = []
        for raw in candidates:
            if hashlib.sha256(raw).hexdigest() == sha:
                return json.loads(raw)
        for rev in revs:
            shown = subprocess.run(
                ["git", "show", f"{rev}:{rel}"], cwd=root, capture_output=True, timeout=60
            )
            if shown.returncode == 0 and hashlib.sha256(shown.stdout).hexdigest() == sha:
                return json.loads(shown.stdout)
        return None

    return resolve


def _key(value: Any) -> str:
    return json.dumps(value, sort_keys=True)


def classify(
    old: dict[str, Any], new: dict[str, Any], resolve: Resolver | None = None
) -> list[Change]:
    """Every difference between two contract documents, each mapped onto a row of FR-044.16's table. A difference
    no row names is `unlisted change`, which is breaking (fails closed). A rename is a removal plus an addition.
    `resolve` reads a vendored schema by `$id` and sha256, for a re-vendored `external` entry."""
    run = _Classifier(old, new, resolve)
    run.top(old, new)
    return run.changes


def breaking(changes: list[Change]) -> list[Change]:
    return [c for c in changes if c.breaking]
