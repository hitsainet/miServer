"""The answer-contract consumer test runs with the `cross-repo-contract` job's packages alone (044 IC-2, M27).

That job installs pytest, mcp, pydantic, pydantic-settings, sqlalchemy, httpx and rfc8785 (read from its
`pip install` line here), not the app's runtime (torch, fastapi, structlog; jsonschema arrives only through mcp). The consumer test and
`contract_tools.py` must therefore import nothing else. This runs the test in a subprocess where any import
outside the standard library, pytest, those packages and their own installed requirements raises: the
conditions of that runner. Precedent: `test_cross_repo_sync_files_need_only_pytest.py`.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import yaml

from tests.support import env_flags

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github" / "workflows" / "backend-tests.yml"
CROSS_REPO_SWITCHES = (
    env_flags.new_name("REQUIRE_CROSS_REPO_CHECKS"),
    "MISTUDIO_REQUIRE_CROSS_REPO_CHECKS",
)
CONSUMER_TEST = "tests/unit/test_answers_contract_consumers.py"


def job_packages() -> list[str]:
    """The distributions the job's `pip install` line names (`"$spec"` is miWorkbench's mcp pin)."""
    job = yaml.safe_load(WORKFLOW.read_text())["jobs"]["cross-repo-contract"]
    line = next(
        ln
        for s in job["steps"]
        for ln in s.get("run", "").splitlines()
        if ln.strip().startswith("pip install") and "pytest" in ln
    )
    words = [w for w in line.split()[2:] if not w.startswith("-") and "constraints" not in w]
    names = ["mcp" if w.strip('"') == "$spec" else re.split(r"[<>=!\[]", w)[0] for w in words]
    assert {"pytest", "pydantic", "mcp"} <= set(names), names
    return names


_RUNNER = r"""
import importlib.abc, importlib.metadata as md, re, sys
def deps(name, seen):
    key = re.sub(r"[-_.]+", "-", name).lower()
    if key in seen:
        return
    seen.add(key)
    try:
        dist = md.distribution(name)
    except md.PackageNotFoundError:
        return
    for top in (dist.read_text("top_level.txt") or "").split():
        TOP.add(top)
    for f in dist.files or ():
        part = str(f).split("/")[0]
        if part.endswith(".py"):
            TOP.add(part[:-3])
        elif "." not in part and not part.endswith("__pycache__"):
            TOP.add(part)
    for req in dist.requires or ():
        if "extra ==" in req:
            continue
        deps(re.split(r"[ ;<>=!\[(~]", req, maxsplit=1)[0], seen)
TOP = set()
seen = set()
for name in sys.argv[1].split(","):
    deps(name, seen)
ALLOWED = set(sys.stdlib_module_names) | TOP | {"miserver", "tests", "_pytest", "py"}
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        top = name.split(".")[0]
        # `_sysconfigdata_*` is the interpreter's own build record (stdlib, but not in the list).
        if top not in ALLOWED and not top.startswith("_sysconfigdata"):
            raise ImportError(f"BLOCKED: {name} is not installed by the cross-repo-contract job")
        return None
sys.meta_path.insert(0, Block())
import pytest
sys.exit(pytest.main(["--noconftest", "-p", "no:cacheprovider", "-o", "addopts=", "-q", *sys.argv[2:]]))
"""


def _run(*args: str) -> subprocess.CompletedProcess:
    env = {k: v for k, v in os.environ.items() if k not in CROSS_REPO_SWITCHES}
    env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    return subprocess.run(
        [sys.executable, "-c", _RUNNER, ",".join(job_packages()), *args],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )


def test_the_consumer_test_runs_on_the_jobs_packages_alone():
    proc = _run(CONSUMER_TEST)
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-2000:]
    assert "BLOCKED" not in proc.stdout + proc.stderr


def test_the_runner_does_block_a_package_the_job_lacks(tmp_path):
    """The block is real: a planted test importing fastapi (not installed by the job) is refused.

    jsonschema would NOT do here: mcp requires it, so the job has it (measured 2026-11-25)."""
    probe = tmp_path / "test_probe_fastapi.py"
    probe.write_text("def test_it():\n    import fastapi  # noqa: F401\n")
    proc = _run(str(probe))
    assert proc.returncode != 0 and "BLOCKED: fastapi" in proc.stdout + proc.stderr
