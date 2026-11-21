"""Every dotted patch target a test names exists (miServer 042 FTASKS 5.3, FR-042.15).

A rename moves a function, class or module; a `mock.patch("miserver.x.Y")` or
`monkeypatch.setattr("miserver.x.Y", …)` that still names the old spelling either raises (loud) or,
when the old name survives as an alias somewhere, patches an object nothing calls any more (silent).
This reads every such string target in the test tree, by its call, and resolves it the way
`unittest.mock` does: import the longest importable module prefix, then walk the attributes.
"""

from __future__ import annotations

import ast
import importlib
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PATCHERS = frozenset({"patch", "setattr", "delattr"})


def _targets() -> list[tuple[str, int, str]]:
    listed = subprocess.run(["git", "ls-files", "tests"], cwd=ROOT, capture_output=True, text=True)
    files = [f for f in listed.stdout.split() if f.endswith(".py")] if listed.returncode == 0 else [
        p.relative_to(ROOT).as_posix() for p in (ROOT / "tests").rglob("*.py")]
    found = []
    for rel in files:
        if rel.startswith("tests/fixtures/"):
            continue
        tree = ast.parse((ROOT / rel).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not node.args:
                continue
            name = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", "")
            first = node.args[0]
            # `create=True` / `raising=False` say on purpose that the target may not exist.
            if any(k.arg in ("create", "raising") and isinstance(k.value, ast.Constant)
                   and k.value.value is (k.arg == "create") for k in node.keywords):
                continue
            if name in PATCHERS and isinstance(first, ast.Constant) and isinstance(first.value, str) \
                    and first.value.startswith("miserver."):
                found.append((rel, node.lineno, first.value))
    return found


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
    broken = [f"{rel}:{line}: {t} ({why})" for rel, line, t in targets if (why := _resolves(t))]
    assert not broken, "patch targets that name nothing:\n" + "\n".join(broken)


def test_a_stale_target_is_caught():
    assert _resolves("miserver.core.errors.MiServerError") is None
    assert _resolves("miserver.core.errors.MiLLMError") is not None
    assert _resolves("miserver.no_such_module.x") is not None
