"""Every dotted patch target a test names exists (miServer 042 FTASKS 5.3, FR-042.15).

A rename moves a function, class or module; a `mock.patch("miserver.x.Y")` or
`monkeypatch.setattr("miserver.x.Y", …)` that still names the old spelling either raises (loud) or,
when the old name survives as an alias somewhere, patches an object nothing calls any more (silent).
This reads every such string target in the test tree, by its call, and resolves it the way
`unittest.mock` does: import the longest importable module prefix, then walk the attributes.

ENH-011 SV-2 (#16, 2026-11-26): `create=True` and `raising=False` are resolved too. They used to be
skipped as "the target may not exist on purpose", and two such patches patched nothing for months:
`probe_hub_service.settings` (the module has no `settings`) and `inference_service.AutoModelForCausalLM`
(imported inside the function, so the module attribute is never read). A patch that creates the
name it patches is inert unless the code under test reads that exact attribute, so each one now
resolves like any other target unless `ABSENT_OK` names it with the reason it legitimately does
not exist. The object form (`monkeypatch.setattr(mod, "name", ..., raising=False)`,
`patch.object(mod, "name", create=True)`) is resolved when `mod` is a name the file imports from
`miserver`; an object the reader cannot name statically (a local, a fixture) is out of its reach.
"""

from __future__ import annotations

import ast
import importlib
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PATCHERS = frozenset({"patch", "setattr", "delattr"})
#: The object form, `patch.object(obj, "name", ...)` / `monkeypatch.setattr(obj, "name", ...)`.
OBJECT_PATCHERS = frozenset({"object", "setattr", "delattr"})

#: `create=True` / `raising=False` patches whose target legitimately does not exist until patched:
#: (test file, resolved target) -> why. Every entry needs a reason, must match a patch in that
#: file, and must still name nothing (an entry whose target now exists is stale).
ABSENT_OK: dict[tuple[str, str], str] = {
    ("tests/unit/ml/test_weights_io.py", "miserver.ml.weights_io.open"): (
        "shadows the builtin open() for the module's own global lookups: the module has no `open` "
        "attribute, and its read_header resolves open through module globals first"
    ),
}


def _test_files() -> list[str]:
    listed = subprocess.run(["git", "ls-files", "tests"], cwd=ROOT, capture_output=True, text=True)
    files = [f for f in listed.stdout.split() if f.endswith(".py")] if listed.returncode == 0 else [
        p.relative_to(ROOT).as_posix() for p in (ROOT / "tests").rglob("*.py")]
    return [f for f in files if not f.startswith("tests/fixtures/")]


def _imported_names(tree: ast.AST) -> dict[str, str]:
    """Local name -> dotted `miserver` path, for every import in the file (any scope). A name bound
    to two different paths in one file is dropped: the reader cannot tell which one a call sees."""
    bound: dict[str, set[str]] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name == "miserver" or a.name.startswith("miserver."):
                    local = a.asname or a.name.split(".")[0]
                    bound.setdefault(local, set()).add(a.name if a.asname else local)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module and (
                node.module == "miserver" or node.module.startswith("miserver.")):
            for a in node.names:
                bound.setdefault(a.asname or a.name, set()).add(f"{node.module}.{a.name}")
    return {k: next(iter(v)) for k, v in bound.items() if len(v) == 1}


def _dotted(expr: ast.expr, names: dict[str, str]) -> str | None:
    attrs = []
    while isinstance(expr, ast.Attribute):
        attrs.append(expr.attr)
        expr = expr.value
    if isinstance(expr, ast.Name) and expr.id in names:
        return ".".join([names[expr.id], *reversed(attrs)])
    return None


def _may_be_absent(node: ast.Call) -> bool:
    """`create=True` / `raising=False`: the patch tolerates a target that does not exist."""
    return any(k.arg in ("create", "raising") and isinstance(k.value, ast.Constant)
               and k.value.value is (k.arg == "create") for k in node.keywords)


def _targets_in(rel: str, source: str) -> list[tuple[str, int, str, bool]]:
    """(file, line, dotted target, may_be_absent) for every patch the reader can name."""
    tree = ast.parse(source)
    names = _imported_names(tree)
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not node.args:
            continue
        name = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", "")
        first = node.args[0]
        absent = _may_be_absent(node)
        if name in PATCHERS and isinstance(first, ast.Constant) and isinstance(first.value, str) \
                and first.value.startswith("miserver."):
            found.append((rel, node.lineno, first.value, absent))
        elif absent and name in OBJECT_PATCHERS and len(node.args) > 1 \
                and isinstance(node.args[1], ast.Constant) and isinstance(node.args[1].value, str):
            # Without create/raising a missing attribute raises at patch time (loud); only the
            # tolerant object form can patch nothing in silence.
            owner = _dotted(first, names)
            if owner is not None:
                found.append((rel, node.lineno, f"{owner}.{node.args[1].value}", absent))
    return sorted(found, key=lambda t: t[1])


def _targets() -> list[tuple[str, int, str, bool]]:
    found = []
    for rel in _test_files():
        found += _targets_in(rel, (ROOT / rel).read_text(encoding="utf-8"))
    return found


def _has_reason(why: object) -> bool:
    return isinstance(why, str) and bool(why.strip())


def _allowlist_problems(allow: dict[tuple[str, str], str]) -> list[str]:
    return [f"{rel}: {target}: an ABSENT_OK entry needs a reason"
            for (rel, target), why in allow.items() if not _has_reason(why)]


def _broken(targets, allow: dict[tuple[str, str], str]) -> list[str]:
    return [f"{rel}:{line}: {t} ({why})" + (" [create=True/raising=False]" if absent else "")
            for rel, line, t, absent in targets
            if not (absent and _has_reason(allow.get((rel, t)))) and (why := _resolves(t))]


def _resolves(target: str) -> str | None:
    parts = target.split(".")
    for cut in range(len(parts), 0, -1):
        try:
            obj = importlib.import_module(".".join(parts[:cut]))
        except ImportError:
            continue
        for attr in parts[cut:]:
            if not hasattr(obj, attr):
                return f"{'.'.join(parts[:cut])} has no {attr!r}"
            obj = getattr(obj, attr)
        return None
    return "no importable prefix"


def test_every_string_patch_target_resolves():
    targets = _targets()
    assert len(targets) > 200, f"found only {len(targets)} patch targets: the reader is failing open"
    broken = _broken(targets, ABSENT_OK)
    assert not broken, "patch targets that name nothing:\n" + "\n".join(broken)


def test_create_and_raising_targets_are_read_not_skipped():
    tolerant = [t for t in _targets() if t[3]]
    # 2026-11-26: 35 create=True / raising=False targets the reader can name (string and object form).
    assert len(tolerant) >= 30, f"only {len(tolerant)} tolerant targets read: the reader is failing open"
    assert any(not t[2].startswith(("miserver.main.", "miserver.core.")) for t in tolerant)


def test_every_absent_ok_entry_has_a_reason_is_used_and_still_names_nothing():
    assert not _allowlist_problems(ABSENT_OK), _allowlist_problems(ABSENT_OK)
    tolerant = {(rel, t) for rel, _line, t, absent in _targets() if absent}
    for key in ABSENT_OK:
        assert key in tolerant, f"{key}: no create=True/raising=False patch matches this entry"
        assert _resolves(key[1]) is not None, f"{key}: the target exists now; drop the entry"


def test_an_absent_ok_entry_without_a_reason_is_refused():
    key = ("tests/x.py", "miserver.core.errors.NoSuchName")
    for empty in ("", "   ", None):
        assert _allowlist_problems({key: empty}), f"accepted the reason {empty!r}"
        targets = [("tests/x.py", 1, key[1], True)]
        assert _broken(targets, {key: empty}), f"a reason {empty!r} exempted a missing target"


def test_a_tolerant_patch_on_a_missing_name_is_caught():
    src = (
        "from unittest.mock import patch\n"
        "from miserver.core import errors\n"
        "def test_x(monkeypatch):\n"
        "    with patch('miserver.core.errors.NoSuchName', create=True):\n"
        "        pass\n"
        "    monkeypatch.setattr(errors, 'AlsoMissing', 1, raising=False)\n"
        "    with patch.object(errors, 'MiServerError', create=True):\n"
        "        pass\n"
    )
    targets = _targets_in("tests/x.py", src)
    assert [(t[2], t[3]) for t in targets] == [
        ("miserver.core.errors.NoSuchName", True),
        ("miserver.core.errors.AlsoMissing", True),
        ("miserver.core.errors.MiServerError", True),
    ], targets
    broken = _broken(targets, {})
    assert len(broken) == 2 and "NoSuchName" in broken[0] and "AlsoMissing" in broken[1], broken


def test_a_stale_target_is_caught():
    assert _resolves("miserver.core.errors.MiServerError") is None
    assert _resolves("miserver.core.errors.MiLLMError") is not None
    assert _resolves("miserver.no_such_module.x") is not None
