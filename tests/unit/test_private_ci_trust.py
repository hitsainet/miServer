"""The mirror's image build trusts THIS repository's CI (operator directive, 2026-10-20).

Before: every push ran the suite here, the sync published a snapshot straight away, and the
public mirror's image build waited for the mirror to run the same suite again (or, in some
repos, for nothing at all). After:

1. PRIVATE: `sync-to-clean.yml`'s `gate` job runs `.github/scripts/private_ci_gate.py`, which
   refuses unless every required CI workflow passed for the exact commit (Backend Tests
   including its `all-shards` aggregator). The `sync` job `needs` the gate, checks out the
   commit the gate passed, and stamps the snapshot with `Source-Commit:` and
   `Private-CI: passed` trailers.
2. MIRROR: `docker-images.yml`'s `build` needs a `private-ci` job that requires those
   trailers on the snapshot, and nothing in it waits on the mirror's own test runs.
3. ORDER: the sync job is serialised, and a branch sync never moves the mirror back to an
   older commit than the one it already holds.

Every assertion reads the parsed YAML or RUNS the step's script; none greps the text.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

yaml = pytest.importorskip("yaml")

# ── per-repository facts ─────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[2]
PRIVATE = "Onegaishimas/miServer"
# Workflows that run as CI here but deliberately do not gate the mirror, with the reason.
EXEMPT: dict[str, str] = {}
# ─────────────────────────────────────────────────────────────────────────────────────

WORKFLOWS = REPO_ROOT / ".github" / "workflows"
GATE_SCRIPT = REPO_ROOT / ".github" / "scripts" / "private_ci_gate.py"


def _load(name: str) -> dict:
    doc = yaml.safe_load((WORKFLOWS / name).read_text())
    doc["on"] = doc.pop(True, doc.get("on"))  # PyYAML reads the bare key `on` as boolean True
    return doc


def _runs_here(job: dict) -> bool:
    """False for a job that can only run on the public mirror."""
    cond = str(job.get("if", ""))
    return not (
        "github.repository" in cond and "hitsainet/" in cond and "Onegaishimas/" not in cond
    )


def _private_ci_workflows() -> set[str]:
    """Every workflow that is CI for main here: runs on a push to main AND on pull requests,
    with at least one job that runs in this repository."""
    out = set()
    for path in sorted(WORKFLOWS.glob("*.yml")):
        doc = _load(path.name)
        on = doc["on"] if isinstance(doc["on"], dict) else dict.fromkeys(doc["on"])
        if "pull_request" not in on or "push" not in on:
            continue
        branches = (on["push"] or {}).get("branches")
        if branches is not None and "main" not in branches:
            continue
        if any(_runs_here(j) for j in doc["jobs"].values()):
            out.add(path.name)
    return out - set(EXEMPT)


def _step(job: dict, name: str) -> dict:
    return next(s for s in job["steps"] if s.get("name") == name)


def _bash(script: str, cwd: Path, env: dict[str, str]) -> subprocess.CompletedProcess:
    full = {**os.environ, **env}
    return subprocess.run(
        ["bash", "-e", "-c", script], cwd=cwd, env=full, capture_output=True, text=True, check=False
    )


def _git(cwd: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=cwd, check=True, capture_output=True, text=True
    ).stdout.strip()


# ── 1. the private gate ──────────────────────────────────────────────────────────────


def test_the_gate_requires_every_ci_workflow_that_runs_here():
    sync = _load("sync-to-clean.yml")
    gate = sync["jobs"]["gate"]
    required = {w.strip() for w in gate["env"]["REQUIRED_WORKFLOWS"].split() if w.strip()}
    assert required == _private_ci_workflows(), (
        "REQUIRED_WORKFLOWS must name every CI workflow that runs on main here (or EXEMPT it "
        "with a reason): a CI workflow left out can be red while its code reaches the mirror"
    )
    assert "backend-tests.yml" in required
    assert gate["permissions"].get("actions") == "read", "the gate lists runs and jobs"
    run = _step(gate, "Require private CI to have passed for this commit")
    assert run["run"].strip() == "python .github/scripts/private_ci_gate.py"
    assert run["env"]["SHA"] == "${{ github.sha }}"
    assert gate["outputs"]["sha"] == "${{ steps.gate.outputs.sha }}" and run["id"] == "gate"


def test_nothing_reaches_the_mirror_except_through_the_gate():
    jobs = _load("sync-to-clean.yml")["jobs"]
    pushers = {k for k, j in jobs.items() if "HITSAINET_SYNC" in json.dumps(j)}
    assert pushers == {"sync"}, f"only the gated job may hold the mirror token: {pushers}"
    needs = jobs["sync"]["needs"]
    assert needs == "gate" or "gate" in needs
    assert "always()" not in str(
        jobs["sync"].get("if", "")
    ), "always() would run the sync over a failed gate"


def test_the_sync_publishes_the_commit_the_gate_passed():
    sync = _load("sync-to-clean.yml")["jobs"]["sync"]
    checkout = next(
        s for s in sync["steps"] if str(s.get("uses", "")).startswith("actions/checkout")
    )
    assert checkout["with"]["ref"] == "${{ needs.gate.outputs.sha }}"
    snap = _step(sync, "Build the orphan snapshot")
    assert snap["env"]["SOURCE_SHA"] == "${{ needs.gate.outputs.sha }}"
    for step in sync["steps"]:
        assert "GITHUB_SHA" not in step.get("run", ""), (
            f"{step.get('name')}: GITHUB_SHA names the triggering commit, "
            "not necessarily the gated one"
        )


def test_the_sync_is_serialised_and_never_moves_the_mirror_backwards():
    sync = _load("sync-to-clean.yml")["jobs"]["sync"]
    conc = sync["concurrency"]
    assert conc["group"] and conc["cancel-in-progress"] is False
    names = [s.get("name") for s in sync["steps"]]
    order = names.index("Never move the mirror backwards")
    assert sync["steps"][order]["id"] == "order"
    for step in sync["steps"][order + 1 :]:
        assert "steps.order.outputs.publish != 'false'" in str(step.get("if", "")), (
            f"{step.get('name')} runs even when the ordering check said "
            "an older commit must not publish"
        )


# ── 2. the snapshot's trailers and the mirror's trust check, end to end ─────────────


@pytest.fixture
def source_repo(tmp_path):
    repo = tmp_path / "src"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    (repo / "VERSION").write_text("0.5.0\n")
    (repo / "pyproject.toml").write_text('[project]\nname = "x"\nversion = "0.5.0"\n')
    (repo / "README.md").write_text("x\n")
    _git(repo, "add", "-A")
    _git(repo, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "source")
    return repo


def _mirror_check() -> str:
    job = _load("docker-images.yml")["jobs"]["private-ci"]
    return _step(job, "Require the private repo's CI verdict")["run"]


def test_a_snapshot_the_sync_builds_passes_the_mirror_check(source_repo):
    source_sha = _git(source_repo, "rev-parse", "HEAD")
    snap = _step(_load("sync-to-clean.yml")["jobs"]["sync"], "Build the orphan snapshot")
    built = _bash(snap["run"], source_repo, {"SOURCE_SHA": source_sha})
    assert built.returncode == 0, built.stdout + built.stderr
    head = _git(source_repo, "rev-parse", "HEAD")
    trailers = _git(source_repo, "log", "-1", "--format=%(trailers:only,unfold)")
    assert f"Source-Commit: {source_sha}" in trailers and "Private-CI: passed" in trailers

    ok = _bash(_mirror_check(), source_repo, {"GITHUB_SHA": head})
    assert ok.returncode == 0, ok.stdout + ok.stderr


@pytest.mark.parametrize(
    "message",
    [
        "snapshot with no trailers",
        "snapshot\n\nSource-Commit: " + "a" * 40 + "\nPrivate-CI: failed",
        "snapshot\n\nPrivate-CI: passed",
        "snapshot\n\nSource-Commit: abc1234\nPrivate-CI: passed",
    ],
)
def test_the_mirror_refuses_a_commit_without_the_verdict(source_repo, message):
    _git(
        source_repo,
        "-c",
        "user.name=t",
        "-c",
        "user.email=t@t",
        "commit",
        "-q",
        "--allow-empty",
        "-m",
        message,
    )
    head = _git(source_repo, "rev-parse", "HEAD")
    res = _bash(_mirror_check(), source_repo, {"GITHUB_SHA": head})
    assert res.returncode != 0, f"accepted {message!r}: {res.stdout}"


def test_the_mirror_build_trusts_the_trailer_not_its_own_test_runs():
    doc = _load("docker-images.yml")
    jobs = doc["jobs"]
    trust = jobs["private-ci"]
    assert not _runs_here(trust), "the trust check belongs to the mirror"
    build = jobs["build"]
    needs = build["needs"] if isinstance(build["needs"], list) else [build["needs"]]
    assert "private-ci" in needs
    assert "always()" not in str(
        build.get("if", "")
    ), "always() would build over a failed trust check"
    for name, job in jobs.items():
        perms = job.get("permissions") or {}
        assert (
            perms.get("actions") is None
        ), f"{name} can read workflow runs - nothing here should wait on tests"
        for step in job.get("steps", []):
            run = step.get("run", "")
            assert (
                "/actions/runs" not in run and "Backend Tests" not in run
            ), f"{name}/{step.get('name')} still waits on the mirror's own test runs"


# ── 3. the shards and their aggregator ───────────────────────────────────────────────


def _shard_jobs(doc: dict) -> dict[str, dict]:
    return {
        k: j
        for k, j in doc["jobs"].items()
        if "shard" in ((j.get("strategy") or {}).get("matrix") or {})
    }


def test_every_shard_job_runs_one_slice_of_n_and_the_aggregator_needs_them_all():
    doc = _load("backend-tests.yml")
    shards = _shard_jobs(doc)
    assert shards, "Backend Tests is not sharded"
    for name, job in shards.items():
        values = job["strategy"]["matrix"]["shard"]
        n = len(values)
        assert values == list(range(1, n + 1)), f"{name}: shards must be 1..{n}, got {values}"
        assert (
            job["env"]["CI_SHARD"] == f"${{{{ matrix.shard }}}}/{n}"
        ), f"{name}: CI_SHARD disagrees with the matrix"
        assert job["strategy"].get("fail-fast") is False
    agg = doc["jobs"]["all-shards"]
    needs = agg["needs"] if isinstance(agg["needs"], list) else [agg["needs"]]
    assert set(needs) == set(shards), "the aggregator must need every shard job"
    assert "always()" in str(
        agg.get("if", "")
    ), "a skipped aggregator reads as neither pass nor fail"


@pytest.mark.parametrize(
    "results, ok",
    [
        ({"a": "success", "b": "success"}, True),
        ({"a": "success", "b": "failure"}, False),
        ({"a": "success", "b": "cancelled"}, False),
        ({"a": "skipped"}, False),
        ({}, False),
    ],
)
def test_the_aggregator_passes_only_when_every_shard_passed(tmp_path, results, ok):
    agg = _load("backend-tests.yml")["jobs"]["all-shards"]
    step = _step(agg, "Require every shard to have passed")
    needs = json.dumps({k: {"result": v, "outputs": {}} for k, v in results.items()})
    res = _bash(step["run"], tmp_path, {"RESULTS": needs})
    assert (res.returncode == 0) is ok, res.stdout + res.stderr


# ── 4. the gate script's decisions ───────────────────────────────────────────────────


def _gate_module():
    spec = importlib.util.spec_from_file_location("private_ci_gate", GATE_SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _Api:
    """A fake of the three GitHub endpoints the gate reads."""

    def __init__(self, by_sha=None, history=None, jobs=None):
        self.by_sha = by_sha or []  # list of successive answers for head_sha=<sha>
        self.history = history or []
        self.jobs = jobs or {}
        self.calls = 0

    def __call__(self, path: str):
        if "/jobs?" in path:
            run_id = int(path.split("/runs/")[1].split("/")[0])
            return {"jobs": self.jobs.get(run_id, [])}
        if "head_sha=" in path:
            answer = self.by_sha[min(self.calls, len(self.by_sha) - 1)] if self.by_sha else []
            self.calls += 1
            return {"workflow_runs": answer}
        return {"workflow_runs": self.history}


def _run(i, sha, status="completed", conclusion="success", event="push"):
    return {
        "id": i,
        "head_sha": sha,
        "status": status,
        "conclusion": conclusion,
        "event": event,
        "created_at": f"2026-10-20T00:00:{i:02d}Z",
        "run_attempt": 1,
    }


@pytest.fixture
def history_repo(tmp_path):
    """main: A (tests passed) -> B (docs only) -> C (touches the tested path)."""
    repo = tmp_path / "repo"
    (repo / ".github" / "workflows").mkdir(parents=True)
    _git(repo, "init", "-q", "-b", "main")
    (repo / ".github" / "workflows" / "filtered.yml").write_text(
        "name: F\non:\n  push:\n    branches: [main]\n    paths: ['src/**']\n"
        "jobs:\n  t: {runs-on: x, steps: []}\n"
    )
    (repo / ".github" / "workflows" / "sharded.yml").write_text(
        "name: S\non:\n  push:\n    branches: [main]\njobs:\n  t: {runs-on: x, steps: []}\n"
        "  all-shards:\n    name: all shards\n    needs: [t]\n    runs-on: x\n    steps: []\n"
    )
    (repo / "src").mkdir()
    shas = []
    for name, path in (("A", "src/a.py"), ("B", "docs.md"), ("C", "src/c.py")):
        (repo / path).write_text(name)
        _git(repo, "add", "-A")
        _git(repo, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", name)
        shas.append(_git(repo, "rev-parse", "HEAD"))
    return SimpleNamespace(repo=repo, a=shas[0], b=shas[1], c=shas[2])


def _gate(mod, repo, sha, api):
    return mod.Gate(
        "o/r",
        sha,
        repo,
        get=api,
        sleep=lambda s: None,
        clock=iter(range(0, 10**6, 60)).__next__,
        timeout=600,
        grace=120,
        interval=30,
    )


def test_gate_passes_a_green_run_for_the_commit(history_repo):
    mod = _gate_module()
    h = history_repo
    ok, msg = _gate(mod, h.repo, h.c, _Api(by_sha=[[_run(1, h.c)]])).check("filtered.yml")
    assert ok, msg


def test_gate_waits_for_a_running_run_then_passes(history_repo):
    mod = _gate_module()
    h = history_repo
    api = _Api(
        by_sha=[
            [_run(1, h.c, "in_progress", None)],
            [_run(1, h.c, "in_progress", None)],
            [_run(1, h.c)],
        ]
    )
    ok, msg = _gate(mod, h.repo, h.c, api).check("filtered.yml")
    assert ok and api.calls == 3, msg


@pytest.mark.parametrize("conclusion", ["failure", "cancelled", "timed_out"])
def test_gate_refuses_a_red_run_for_the_commit(history_repo, conclusion):
    mod = _gate_module()
    h = history_repo
    ok, msg = _gate(mod, h.repo, h.c, _Api(by_sha=[[_run(1, h.c, conclusion=conclusion)]])).check(
        "filtered.yml"
    )
    assert not ok and conclusion in msg


def test_gate_ignores_a_pull_request_run(history_repo):
    mod = _gate_module()
    h = history_repo
    ok, _ = _gate(mod, h.repo, h.c, _Api(by_sha=[[_run(1, h.c, event="pull_request")]])).check(
        "filtered.yml"
    )
    assert not ok


def test_gate_requires_the_all_shards_job(history_repo):
    mod = _gate_module()
    h = history_repo
    run = _run(7, h.c)
    no_agg = _Api(by_sha=[[run]], jobs={7: [{"name": "t", "conclusion": "success"}]})
    assert not _gate(mod, h.repo, h.c, no_agg).check("sharded.yml")[0]
    red_agg = _Api(by_sha=[[run]], jobs={7: [{"name": "all shards", "conclusion": "failure"}]})
    assert not _gate(mod, h.repo, h.c, red_agg).check("sharded.yml")[0]
    green = _Api(by_sha=[[run]], jobs={7: [{"name": "all shards", "conclusion": "success"}]})
    assert _gate(mod, h.repo, h.c, green).check("sharded.yml")[0]


def test_gate_falls_back_only_when_the_tested_paths_are_unchanged(history_repo):
    mod = _gate_module()
    h = history_repo
    # B touched only docs: no run of the filtered workflow, and A passed with identical src/.
    ok, msg = _gate(mod, h.repo, h.b, _Api(history=[_run(1, h.a)])).check("filtered.yml")
    assert ok and h.a in msg, msg
    # C changed src/ since A: A's verdict does not cover it.
    ok, _ = _gate(mod, h.repo, h.c, _Api(history=[_run(1, h.a)])).check("filtered.yml")
    assert not ok
    # A newer commit that is NOT an ancestor never vouches for an older one - even when, as
    # B here, its tested paths are identical: only the ancestry rule refuses this one.
    ok, _ = _gate(mod, h.repo, h.a, _Api(history=[_run(2, h.b)])).check("filtered.yml")
    assert not ok


def test_gate_refuses_a_missing_run_for_an_unfiltered_workflow(history_repo):
    mod = _gate_module()
    h = history_repo
    api = _Api(history=[_run(1, h.b)], jobs={1: [{"name": "all shards", "conclusion": "success"}]})
    ok, msg = _gate(mod, h.repo, h.c, api).check("sharded.yml")
    assert not ok and "no path filter" in msg


def test_gate_refuses_an_empty_requirement(monkeypatch):
    mod = _gate_module()
    monkeypatch.setenv("GITHUB_REPOSITORY", "o/r")
    monkeypatch.setenv("SHA", "a" * 40)
    monkeypatch.setenv("REQUIRED_WORKFLOWS", "  \n")
    assert mod.main() == 1


# ── 5. the gitops sync is gated too (ENH-011 SV-1, #13) ───────────────────────────────
#
# `sync-main-to-gitops-pilot.yml` merges main into the branch ArgoCD deploys from. Before
# ENH-011 it ran on every push with no `needs:` and merged whatever `origin/main` was, so a
# manifest change reached the cluster while Backend Tests was red. It now waits for the same
# gate as the mirror and merges exactly the commit the gate passed.

GITOPS = "sync-main-to-gitops-pilot.yml"
GATE_RUN = "python .github/scripts/private_ci_gate.py"


def _gate_job_names(jobs: dict) -> set[str]:
    """Jobs that run the private CI gate script."""
    return {
        k
        for k, j in jobs.items()
        if any((s.get("run") or "").strip() == GATE_RUN for s in j.get("steps", []))
    }


def _needs(job: dict) -> list[str]:
    needs = job.get("needs") or []
    return [needs] if isinstance(needs, str) else list(needs)


def _gitops_gate_name() -> str:
    jobs = _load(GITOPS)["jobs"]
    gates = _gate_job_names(jobs)
    assert len(gates) == 1, f"the gitops sync must have exactly one private-CI gate job: {gates}"
    return next(iter(gates))


def test_the_gitops_sync_needs_a_gate_that_requires_private_ci():
    jobs = _load(GITOPS)["jobs"]
    name = _gitops_gate_name()
    gate = jobs[name]
    required = {w.strip() for w in gate["env"]["REQUIRED_WORKFLOWS"].split() if w.strip()}
    assert "backend-tests.yml" in required
    mirror = _load("sync-to-clean.yml")["jobs"]["gate"]
    mirror_required = {w.strip() for w in mirror["env"]["REQUIRED_WORKFLOWS"].split() if w.strip()}
    assert required == mirror_required == _private_ci_workflows(), (
        "the gitops gate requires the same CI as the mirror's gate: what may not reach the "
        "mirror may not reach the cluster either"
    )
    assert gate["permissions"].get("actions") == "read", "the gate lists runs and jobs"
    assert gate["permissions"].get("contents") != "write", "the gate only reads"
    run = _step(gate, "Require private CI to have passed for this commit")
    assert run["run"].strip() == GATE_RUN and run["id"] == "gate"
    assert gate["outputs"]["sha"] == "${{ steps.gate.outputs.sha }}"
    # The only job that writes is `sync`, and it cannot start without the gate.
    writers = {k for k, j in jobs.items() if (j.get("permissions") or {}).get("contents") == "write"}
    assert writers == {"sync"}, writers
    assert name in _needs(jobs["sync"]), "sync must `needs:` the gate"
    cond = str(jobs["sync"].get("if", ""))
    assert not any(f in cond for f in ("always()", "failure()", "cancelled()")), (
        f"sync would run over a failed gate: {cond}"
    )


def test_the_gitops_gate_gates_the_pushed_commit_or_mains_head_on_a_dispatch(tmp_path):
    gate = _load(GITOPS)["jobs"][_gitops_gate_name()]
    checkout = next(s for s in gate["steps"] if str(s.get("uses", "")).startswith("actions/checkout"))
    # A push gates the pushed commit; a dispatch gates main's head, whatever ref started it.
    assert checkout["with"]["ref"] == "${{ github.event_name == 'push' && github.sha || 'main' }}"
    assert checkout["with"]["fetch-depth"] == 0, "the path-filter fallback reads ancestry"
    run = _step(gate, "Require private CI to have passed for this commit")
    target_ref = run["env"]["SHA"]
    assert target_ref == "${{ steps.target.outputs.sha }}", target_ref
    target = next(s for s in gate["steps"] if s.get("id") == "target")
    repo = tmp_path / "r"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "--allow-empty", "-m", "x")
    out = tmp_path / "out"
    out.write_text("")
    res = _bash(target["run"], repo, {"GITHUB_OUTPUT": str(out)})
    assert res.returncode == 0, res.stdout + res.stderr
    assert out.read_text().strip() == f"sha={_git(repo, 'rev-parse', 'HEAD')}"


def _gated_ref_forms(step: dict, gate: str) -> set[str]:
    """The spellings by which a step's script can name the gated commit."""
    expr = f"${{{{ needs.{gate}.outputs.sha }}}}"
    forms = {expr}
    for var, value in (step.get("env") or {}).items():
        if value == expr:
            forms |= {f"${var}", f"${{{var}}}"}
    return forms


def _commands(script: str) -> list[str]:
    """The script's commands, with backslash continuations joined."""
    return [c.strip() for c in script.replace("\\\n", " ").splitlines() if c.strip()]


def test_every_merge_and_converge_in_the_gitops_sync_reads_the_gated_sha():
    jobs = _load(GITOPS)["jobs"]
    gate = _gitops_gate_name()
    sync = jobs["sync"]
    seen = {"merge": 0, "converge": 0, "merge-base": 0}
    for step in sync["steps"]:
        forms = _gated_ref_forms(step, gate)
        for cmd in _commands(step.get("run") or ""):
            if cmd.startswith("#"):
                continue
            kind = None
            if "git merge-base" in cmd:
                kind = "merge-base"
            elif "git merge " in cmd:
                kind = "merge"
            elif "git checkout " in cmd and " -- ." in cmd:
                kind = "converge"
            if kind is None:
                continue
            seen[kind] += 1
            assert "origin/main" not in cmd, f"{step.get('name')}: reads origin/main: {cmd}"
            assert any(f in cmd for f in forms), (
                f"{step.get('name')}: {kind} does not read needs.{gate}.outputs.sha: {cmd}"
            )
        assert "GITHUB_SHA" not in (step.get("run") or "") and "github.sha" not in json.dumps(
            step
        ), f"{step.get('name')}: the triggering commit is not necessarily the gated one"
    assert seen["merge"] >= 1 and seen["converge"] >= 1 and seen["merge-base"] >= 1, seen


# ── the gitops sync, run end to end against a local origin ───────────────────────────

PIN = "k8s/base/.argocd-source-miserver.yaml"


def _commit(repo: Path, msg: str, files: dict[str, str]) -> str:
    for rel, text in files.items():
        (repo / rel).parent.mkdir(parents=True, exist_ok=True)
        (repo / rel).write_text(text)
    _git(repo, "add", "-A")
    _git(repo, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", msg)
    return _git(repo, "rev-parse", "HEAD")


@pytest.fixture
def gitops_origin(tmp_path):
    """origin: main A -> B -> C (C deletes a file); gitops/pilot = A + an image-pin commit."""
    origin = tmp_path / "origin.git"
    _git(tmp_path, "init", "-q", "--bare", "-b", "main", str(origin))
    dev = tmp_path / "dev"
    _git(tmp_path, "clone", "-q", str(origin), str(dev))
    _git(dev, "checkout", "-q", "-b", "main")
    a = _commit(dev, "A", {"k8s/base/app.yaml": "image: v1\n", "old.txt": "x\n"})
    _git(dev, "checkout", "-q", "-b", "gitops/pilot")
    _commit(dev, "pin", {PIN: "pinned: digest-1\n"})
    _git(dev, "checkout", "-q", "main")
    b = _commit(dev, "B", {"k8s/base/app.yaml": "image: v2\n"})
    (dev / "old.txt").unlink()
    c = _commit(dev, "C", {"k8s/base/app.yaml": "image: v3\n"})
    _git(dev, "push", "-q", "origin", "main", "gitops/pilot")
    return SimpleNamespace(origin=origin, tmp=tmp_path, a=a, b=b, c=c)


def _eval_if(cond: str, outputs: dict[str, dict[str, str]]) -> bool:
    """The few step conditions the gitops sync may use: `steps.X.outputs.Y != 'Z'` joined by &&."""
    import re

    if not cond:
        return True
    ok = True
    for part in str(cond).replace("${{", "").replace("}}", "").split("&&"):
        m = re.fullmatch(r"\s*steps\.(\w+)\.outputs\.(\w+)\s*(!=|==)\s*'([^']*)'\s*", part)
        assert m, f"unsupported step condition in the gitops sync: {cond!r}"
        sid, key, op, val = m.groups()
        got = outputs.get(sid, {}).get(key, "")
        ok &= (got != val) if op == "!=" else (got == val)
    return ok


def _run_gitops_sync(env, sha: str, label: str):
    """Run every `run:` step of the sync job, in order, the way Actions would."""
    jobs = _load(GITOPS)["jobs"]
    gate = _gitops_gate_name()
    sync = jobs["sync"]
    work = env.tmp / f"run-{label}"
    _git(env.tmp, "clone", "-q", str(env.origin), str(work))
    _git(work, "checkout", "-q", "gitops/pilot")
    expr = f"${{{{ needs.{gate}.outputs.sha }}}}"
    outputs: dict[str, dict[str, str]] = {}
    log = []
    for step in sync["steps"]:
        if "run" not in step:
            continue
        cond = str(step.get("if", ""))
        assert not any(f in cond for f in ("always()", "failure()", "cancelled()")), (
            f"{step.get('name')} would run after a failed step"
        )
        if not _eval_if(cond, outputs):
            log.append((step.get("name"), "skipped"))
            continue
        out = env.tmp / f"out-{label}-{len(log)}"
        out.write_text("")
        step_env = {
            k: (sha if v == expr else str(v)) for k, v in (step.get("env") or {}).items()
        }
        step_env["GITHUB_OUTPUT"] = str(out)
        res = _bash(step["run"].replace(expr, sha), work, step_env)
        if step.get("id"):
            outputs[step["id"]] = dict(
                line.split("=", 1) for line in out.read_text().splitlines() if "=" in line
            )
        log.append((step.get("name"), res.returncode, res.stdout + res.stderr))
        if res.returncode != 0:
            return SimpleNamespace(ok=False, log=log, work=work)
    return SimpleNamespace(ok=True, log=log, work=work)


def _pilot_tree(env) -> dict[str, str]:
    check = env.tmp / f"check-{len(list(env.tmp.iterdir()))}"
    _git(env.tmp, "clone", "-q", "-b", "gitops/pilot", str(env.origin), str(check))
    return {
        str(p.relative_to(check)): p.read_text()
        for p in check.rglob("*")
        if p.is_file() and ".git" not in p.relative_to(check).parts
    }


def test_the_gitops_sync_merges_the_gated_commit_not_mains_head(gitops_origin):
    env = gitops_origin
    # main's head is C, but the gate passed B: gitops/pilot must hold B's manifests, not C's.
    res = _run_gitops_sync(env, env.b, "b")
    assert res.ok, res.log
    tree = _pilot_tree(env)
    assert tree["k8s/base/app.yaml"] == "image: v2\n", "merged something other than the gated B"
    assert tree["old.txt"] == "x\n", "C's deletion arrived although only B was gated"
    assert tree[PIN] == "pinned: digest-1\n", "the image pin file is the pinner's, not main's"
    # Then C is gated: forward moves work, and converge drops what C deleted.
    res = _run_gitops_sync(env, env.c, "c")
    assert res.ok, res.log
    tree = _pilot_tree(env)
    assert tree["k8s/base/app.yaml"] == "image: v3\n" and "old.txt" not in tree
    assert tree[PIN] == "pinned: digest-1\n"


def test_the_gitops_sync_never_moves_gitops_pilot_backwards(gitops_origin):
    env = gitops_origin
    assert _run_gitops_sync(env, env.c, "c").ok
    before = _git(env.tmp, "--git-dir", str(env.origin), "rev-parse", "gitops/pilot")
    # A gate for the older A finishes after C was merged (out of order): refuse, loudly.
    res = _run_gitops_sync(env, env.a, "a")
    assert not res.ok, f"the sync accepted an older commit than one it already merged: {res.log}"
    failed = res.log[-1]
    assert "backwards" in failed[2], failed
    names = [s.get("name") for s in _load(GITOPS)["jobs"]["sync"]["steps"]]
    assert names.index(failed[0]) < names.index(
        next(n for n in names if n and n.startswith("Merge"))
    ), "the ordering check must run before the merge"
    after = _git(env.tmp, "--git-dir", str(env.origin), "rev-parse", "gitops/pilot")
    assert before == after, "gitops/pilot moved"
    tree = _pilot_tree(env)
    assert tree["k8s/base/app.yaml"] == "image: v3\n" and "old.txt" not in tree
    # Re-running the commit already merged is a no-op, not a failure.
    res = _run_gitops_sync(env, env.c, "c-again")
    assert res.ok, res.log
    assert _git(env.tmp, "--git-dir", str(env.origin), "rev-parse", "gitops/pilot") == before


# ── 6. the mirror sync dispatches the public manual (ENH-011 SV-1, #17) ─────────────
#
# The mirror receives a one-commit orphan, so Deploy Manual's `paths: manual/**` push filter
# never matches there. The sync dispatches it, and, unlike the informational Backend Tests
# dispatch, a refused dispatch FAILS the sync: the manual is the published product.

MANUAL_DISPATCH = "/actions/workflows/deploy-manual.yml/dispatches"


def _manual_dispatch_steps(sync: dict) -> list[dict]:
    return [s for s in sync["steps"] if MANUAL_DISPATCH in (s.get("run") or "")]


def _fake_curl(tmp_path: Path, code: str) -> dict[str, str]:
    bindir = tmp_path / "bin"
    bindir.mkdir(exist_ok=True)
    log = tmp_path / "curl.log"
    curl = bindir / "curl"
    curl.write_text(
        "#!/usr/bin/env bash\n"
        f'printf "%s\\n" "$*" >> "{log}"\n'
        f'printf "%s" "{code}"\n'
    )
    curl.chmod(0o755)
    return {"PATH": f"{bindir}:{os.environ['PATH']}", "LOG": str(log)}


def test_the_mirror_sync_dispatches_the_manual_under_the_backend_tests_condition():
    doc = _load("deploy-manual.yml")
    assert "workflow_dispatch" in doc["on"], "Deploy Manual must accept a dispatch"
    sync = _load("sync-to-clean.yml")["jobs"]["sync"]
    steps = _manual_dispatch_steps(sync)
    assert len(steps) == 1, f"exactly one manual dispatch step: {[s.get('name') for s in steps]}"
    (step,) = steps
    backend = _step(sync, "Dispatch Backend Tests on the mirror")
    assert step["if"] == backend["if"], (step["if"], backend["if"])
    assert step["env"]["SYNC_TOKEN"] == "${{ secrets.HITSAINET_SYNC }}"
    names = [s.get("name") for s in sync["steps"]]
    assert names.index(step["name"]) > names.index(backend["name"])
    # A refused dispatch must not skip the leak check: verification runs first.
    assert names.index(step["name"]) > names.index("Verify the published mirror")


@pytest.mark.parametrize("code, ok", [("204", True), ("403", False), ("404", False),
                                      ("422", False), ("500", False), ("", False)])
def test_the_manual_dispatch_fails_on_anything_but_204(tmp_path, code, ok):
    (step,) = _manual_dispatch_steps(_load("sync-to-clean.yml")["jobs"]["sync"])
    env = _fake_curl(tmp_path, code)
    env |= {"SYNC_TOKEN": "tok-123", "MIRROR": "hitsainet/miServer"}
    res = _bash(step["run"], tmp_path, env)
    assert (res.returncode == 0) is ok, f"HTTP {code!r}: {res.stdout}{res.stderr}"
    calls = Path(env["LOG"]).read_text().splitlines()
    assert len(calls) == 1, f"one dispatch per sync: {calls}"
    (call,) = calls
    assert "-X POST" in call
    assert f"https://api.github.com/repos/hitsainet/miServer{MANUAL_DISPATCH}" in call
    assert "Authorization: Bearer tok-123" in call
    assert "Accept: application/vnd.github+json" in call
    assert "X-GitHub-Api-Version: 2022-11-28" in call
    assert '{"ref":"main"}' in call
