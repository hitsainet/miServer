"""No weight-space-programming library in miServer (043 FR-043.13; suite rule: WSP goes through
miForge's provider, never a library directly; ENH-010 D4).

Four checks:
- the library is not importable in this environment;
- no Python file in the repository imports `weightspace`, `wsplab` or `dspy_wsp`: every tracked `*.py`
  outside `0xcc/` (the package, `tests/`, `scripts/`, `docker/`, `integrations/`, `.github/`) plus every
  file under the package root. By AST, so a comment or a string that only names them is not an import;
  an `import`/`from` statement at any depth, and ANY call whose first argument is a string literal naming
  one (`importlib.import_module(...)`, an aliased `import_module`, `__import__(...)`), all count;
- no dependency file names one;
- the walk visited enough files that a wrong root cannot pass empty.

R1 review fix (OR1-043-4), 2026-11-16: the walk first covered only the package and `tests/`, and saw only
import statements; `importlib.import_module("weightspace")` and `import weightspace` under `scripts/`
both passed.

R2 review fix (OR2-043-4), 2026-11-16: a scanner that returns nothing passes `offenders == []`, and breaking
its own lines (the forbidden-name match, the `import` branch, the call branch) left every test green
(review mutations GW1-GW3). `test_the_scanner_finds_each_form` feeds it every form through the same
`_offenders` helper the tree scan uses. A call is now flagged by its literal argument whatever the callee,
so `from importlib import import_module as _load; _load("wsplab")` (review N3) is caught.
"""

from __future__ import annotations

import ast
import importlib.util
import re
import subprocess
import warnings
from pathlib import Path

import pytest

from tests.support.package_root import package_root

ROOT = Path(__file__).resolve().parents[2]
FORBIDDEN = ("weightspace", "wsplab", "dspy_wsp")
#: Dependency files that would carry the library as a requirement (those present are read).
DEPENDENCY_FILES = ("pyproject.toml", "constraints.txt", "requirements.txt", "requirements-dev.txt")
#: Keyword arguments that name the module a call imports (`import_module`, `__import__`).
IMPORT_KEYWORDS = frozenset({"name", "package"})
#: The walk's floor: miServer has well over this many `.py` files.
MIN_FILES = 500


def _imports(path: Path) -> list[str]:
    with warnings.catch_warnings():  # an invalid escape in a scanned file is that file's business
        warnings.simplefilter("ignore", SyntaxWarning)
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found += [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            found.append(node.module)
        elif isinstance(node, ast.Call):
            # Any callee: `import_module`, an alias of it, `__import__`. A call whose module argument is a
            # literal naming a forbidden module is treated as importing it: the first or second positional
            # argument (`import_module(".api", "weightspace")` names the package second), or the keyword
            # `name` or `package` (R3 review fix OR3-043-8). A string elsewhere, a comment or a docstring is not.
            candidates = [*node.args[:2], *(k.value for k in node.keywords if k.arg in IMPORT_KEYWORDS)]
            found += [c.value.lstrip(".") for c in candidates
                      if isinstance(c, ast.Constant) and isinstance(c.value, str)]
    return found


def _offenders(path: Path) -> list[str]:
    """The forbidden names `path` imports. The tree scan and the scanner's self-check both use this."""
    return [name for name in _imports(path) if name.split(".")[0] in FORBIDDEN]


def _tracked_python() -> set[Path]:
    out = subprocess.run(["git", "ls-files", "-z", "--", "*.py"], cwd=ROOT, capture_output=True,
                         check=True).stdout.decode().split("\0")
    return {ROOT / f for f in out if f and not f.startswith("0xcc/") and (ROOT / f).is_file()}


def test_the_library_is_not_importable():
    for name in FORBIDDEN:
        assert importlib.util.find_spec(name) is None, f"{name} is installed in this environment"


def test_no_module_imports_a_wsp_library():
    # The package through `package_root()`, which raises when it is not there (P1: a walk that finds
    # no package must fail, never pass over nothing; `test_package_walks_fail_closed.py`).
    files = {*package_root().rglob("*.py"), *_tracked_python()}
    assert len(files) >= MIN_FILES, f"walked only {len(files)} files: wrong root?"
    offenders = sorted(
        f"{path.relative_to(ROOT)}: {name}"
        for path in files
        for name in _offenders(path)
    )
    assert offenders == []


def test_no_dependency_file_names_a_wsp_library():
    pattern = re.compile(r"(?<![a-z0-9])(weightspace|wsplab|dspy[-_.]wsp)(?![a-z0-9])", re.IGNORECASE)
    present = [ROOT / name for name in DEPENDENCY_FILES if (ROOT / name).is_file()]
    assert ROOT / "pyproject.toml" in present
    offenders = [f"{p.name}: {m.group(0)}" for p in present for m in pattern.finditer(p.read_text(encoding="utf-8"))]
    assert offenders == []


def test_the_walk_reaches_the_folders_outside_the_package():
    """The walk's reach: a narrowed walk cannot pass quietly. (The call forms are the scanner's business,
    checked by `test_the_scanner_finds_each_form`; R2 review fix (OR2-043-4) corrected this docstring, which
    claimed the call forms too while checking only a call-name helper.)"""
    tracked = {p.relative_to(ROOT).parts[0] for p in _tracked_python()}
    for folder in ("tests", "scripts", "docker", "integrations"):
        if any((ROOT / folder).rglob("*.py")):
            assert folder in tracked, f"{folder}/ holds Python the walk does not visit"


#: Every import form the scanner claims to see (adopted from the round-2 review draft, plus the alias).
FORMS = {
    "plain": "import weightspace\n",
    "dotted_alias": "import os, wsplab.core as w\n",
    "from": "from dspy_wsp.programs import thing\n",
    "nested": "def f():\n    import weightspace\n",
    "import_module": "import importlib\nimportlib.import_module('weightspace.api')\n",
    "aliased_import_module": "from importlib import import_module as _load\n_load('wsplab')\n",
    "dunder": "__import__('wsplab')\n",
    # R3 review fix (OR3-043-8), 2026-11-16: the keyword and package-argument forms.
    "import_module_name_kw": "import importlib\nimportlib.import_module(name='weightspace')\n",
    "dunder_name_kw": "__import__(name='wsplab')\n",
    "package_positional": "from importlib import import_module\nimport_module('.api', 'weightspace')\n",
    "package_kw": "from importlib import import_module\nimport_module('.api', package='dspy_wsp')\n",
}


@pytest.mark.parametrize("form", list(FORMS))
def test_the_scanner_finds_each_form(tmp_path, form):
    path = tmp_path / f"{form}.py"
    path.write_text(FORMS[form])
    assert _offenders(path), f"{form}: the scanner saw no forbidden import in {FORMS[form]!r}"


def test_a_file_that_only_mentions_one_is_not_an_import(tmp_path):
    path = tmp_path / "mention.py"
    path.write_text("# weightspace is never imported here\nNAME = 'weightspace'\nlog = 'wsplab'\n")
    assert _offenders(path) == []
