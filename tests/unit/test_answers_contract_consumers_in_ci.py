"""The answer-contract consumer test runs where miWorkbench is checked out and the check is required (044 M26).

The unit job has no sibling, so there the consumer test SKIPS. Only the `cross-repo-contract` job checks
miWorkbench out and sets `MISERVER_REQUIRE_CROSS_REPO_CHECKS: '1'`, so the test must be in that job's run step,
read from the parsed workflow (not a source scrape).
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml

from tests.support import env_flags

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github" / "workflows" / "backend-tests.yml"
CONSUMER_TEST = "tests/unit/test_answers_contract_consumers.py"


def _job() -> dict:
    return yaml.safe_load(WORKFLOW.read_text())["jobs"]["cross-repo-contract"]


def test_the_consumer_test_is_in_the_jobs_run_step():
    run = next(s["run"] for s in _job()["steps"] if "--noconftest" in s.get("run", ""))
    code = "\n".join(line for line in run.splitlines() if not line.lstrip().startswith("#"))
    assert CONSUMER_TEST in re.findall(r"tests/unit/[\w/]+\.py", code)


def test_the_job_requires_the_cross_repo_check():
    job = _job()
    assert job["env"][env_flags.new_name("REQUIRE_CROSS_REPO_CHECKS")] == "1"
    assert job["env"]["MIWORKBENCH_REPO"].endswith("/miWorkbench")


def test_the_job_checks_out_miservers_history():
    """FR-044.16's re-vendor row reads the old vendored bytes from history; a depth-1 checkout has none."""
    own = next(
        s
        for s in _job()["steps"]
        if s.get("uses", "").startswith("actions/checkout")
        and s.get("with", {}).get("path") == "miServer"
    )
    assert own["with"]["fetch-depth"] == 0
