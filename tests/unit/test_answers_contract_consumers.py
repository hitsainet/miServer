"""Each consumer's vendored copy of the answer contract is the committed file, or one additive revision
behind it (044 FR-044.17, SH3).

`CONSUMERS` names each app that vendors `docs/schemas/miserver-answers-v1.json` and its state:
* `pending`: the consumer has not vendored it yet; an absent copy passes (push 044-1);
* `vendored`: an absent copy fails (push 044-2, once miWorkbench's copy is on its `main`).

A present copy is compared with the committed contract through `contract_tools.classify` (the writer's own
table): equal passes; an additive difference passes and says the consumer must re-vendor; a breaking difference
under the same major fails, because a breaking change needs `miserver-answers-v2.json`.

Runs in the `cross-repo-contract` CI job, which checks miWorkbench out and sets
`MISERVER_REQUIRE_CROSS_REPO_CHECKS=1`; that job installs only pytest, mcp, pydantic, pydantic-settings,
sqlalchemy, httpx and rfc8785, so nothing here imports outside them
(`test_answers_contract_consumers_need_only_job_deps.py`). Without the sibling it skips loudly; under the
require switch it fails instead.

miAssist (006, task 0.6) vendors it at `docs/schemas/miserver-answers-v1.json` in its own repository. That job
cannot check miAssist out (this repository has no deploy key for `Onegaishimas/miAssist`), so
`NOT_CHECKED_OUT_IN_CI` names it: while it is `pending` and `$MIASSIST_REPO` is unset, an absent checkout skips
loudly even under the require switch. The exemption ends by itself: an exempt consumer must stay `pending`
(a `vendored` one needs the checkout, hence the key), and once its path is set an absent checkout fails. The
suite's wave-wide cross-repo pass sets every `_REPO` path, so there the copy is compared.
"""

from __future__ import annotations

import json
import os
import warnings
from pathlib import Path

import pytest

from miserver.api.schemas import contract_tools
from miserver.api.schemas.answers_contract import CONTRACT_FILENAME
from tests.support.sibling_repos import assist_repo, require_or_skip, workbench_repo

ROOT = Path(__file__).resolve().parents[2]
COMMITTED = ROOT / "docs" / "schemas" / CONTRACT_FILENAME

#: app -> (the vendored path in its repository (TD13), state).
CONSUMERS = {
    # 044 R2-3: the path follows `CONTRACT_FILENAME`, so a v2 file is compared with the v2 copy.
    "miWorkbench": ("backend/src/mcp_server/vendored/" + CONTRACT_FILENAME, "pending"),
    # miAssist 006 (FTID §2, FTASKS 0.6, 3.1): copied to its `docs/schemas/` by push 006-1.
    "miAssist": ("docs/schemas/" + CONTRACT_FILENAME, "pending"),
}
_REPOS = {"miWorkbench": workbench_repo, "miAssist": assist_repo}
#: app -> (its `_REPO` variable, why the `cross-repo-contract` job cannot check it out).
NOT_CHECKED_OUT_IN_CI = {
    "miAssist": ("MIASSIST_REPO", "no deploy key for Onegaishimas/miAssist in this repository's CI"),
}


def compare(committed: dict, copy: dict | None, state: str) -> tuple[bool, str]:
    """(passes, message) for one consumer's copy."""
    if copy is None:
        if state == "pending":
            return True, "not vendored yet (pending)"
        return False, "the consumer's copy is missing, and it is vendored"
    if copy.get("$id") != committed["$id"]:
        # 044 R1-6: the vendored file is THIS major's name, so its `$id` must be this major's too; a
        # copy under another `$id` is a corrupted or mis-vendored file, not one to skip.
        return False, f"the copy's $id is {copy.get('$id')!r}, not {committed['$id']!r}"
    files = {sid: row["file"] for doc in (copy, committed) for sid, row in (doc.get("external") or {}).items()}
    changes = contract_tools.classify(_body(copy), _body(committed),
                                      contract_tools.git_resolver(ROOT, files))
    if not changes:
        return True, "equal"
    bad = contract_tools.breaking(changes)
    if bad:
        return False, (
            "breaking under the same major (a breaking change needs miserver-answers-v2.json):\n"
            + "\n".join(f"  {c}" for c in bad)
        )
    return True, (
        f"additive: the consumer must re-vendor (its revision {copy.get('revision')}, "
        f"committed {committed.get('revision')})"
    )


def _body(doc: dict) -> dict:
    return {k: v for k, v in doc.items() if k not in contract_tools.UNDIGESTED}


@pytest.mark.parametrize("app", sorted(CONSUMERS))
def test_each_consumers_copy_is_the_committed_contract_or_behind_it_additively(app: str):
    path, state = CONSUMERS[app]
    repo = _REPOS[app]()
    if not repo.is_dir():
        exempt = NOT_CHECKED_OUT_IN_CI.get(app)
        if exempt and state == "pending" and not os.environ.get(exempt[0]):
            pytest.skip(f"{app} is not checked out ({repo}) and is still pending: {exempt[1]}; set "
                        f"{exempt[0]} to compare its copy (the suite's cross-repo pass does)")
        require_or_skip(
            f"{app} is not checked out beside this repository ({repo}); set its _REPO path"
        )
    committed = json.loads(COMMITTED.read_text(encoding="utf-8"))
    vendored = repo / path
    copy = json.loads(vendored.read_text(encoding="utf-8")) if vendored.is_file() else None
    ok, message = compare(committed, copy, state)
    assert ok, f"{app} ({vendored}): {message}"
    if message.startswith("additive"):
        warnings.warn(f"{app}: {message}", stacklevel=1)


# ── the comparison itself, on planted copies (tmp_path; no sibling needed) ───────────────────────────


@pytest.fixture(scope="module")
def committed() -> dict:
    return json.loads(COMMITTED.read_text(encoding="utf-8"))


def _planted(tmp_path, doc: dict | None) -> dict | None:
    if doc is None:
        return None
    path = tmp_path / CONTRACT_FILENAME
    path.write_text(contract_tools.render(doc), encoding="utf-8")
    return json.loads(path.read_text(encoding="utf-8"))


def test_an_equal_copy_passes(committed, tmp_path):
    assert compare(committed, _planted(tmp_path, committed), "vendored") == (True, "equal")


def test_a_copy_one_additive_revision_behind_passes_and_says_re_vendor(committed, tmp_path):
    behind = json.loads(json.dumps(committed))
    behind["responses"].pop(sorted(behind["responses"])[0])  # the committed file ADDS this route
    behind["revision"] = committed["revision"] - 1 if committed["revision"] > 1 else 0
    ok, message = compare(committed, _planted(tmp_path, behind), "vendored")
    assert ok and "re-vendor" in message, message


def test_a_breaking_copy_under_the_same_major_fails(committed, tmp_path):
    ahead = json.loads(json.dumps(committed))
    ahead["responses"]["GET /api/a-route-the-server-dropped"] = {"200": {"type": "object"}}
    ok, message = compare(committed, _planted(tmp_path, ahead), "vendored")
    assert not ok and "breaking" in message, message


def test_a_copy_under_another_id_fails(committed, tmp_path):
    other = {
        **json.loads(json.dumps(committed)),
        "$id": "https://hitsai.net/schemas/elsewhere.json",
    }
    ok, message = compare(committed, _planted(tmp_path, other), "vendored")
    assert not ok and "$id" in message, message


def test_an_absent_copy_passes_only_while_pending(committed):
    assert compare(committed, None, "pending")[0] is True
    assert compare(committed, None, "vendored")[0] is False


def test_no_sibling_skips_loudly_or_fails_under_the_switch(monkeypatch, tmp_path):
    import tests.unit.test_answers_contract_consumers as me

    monkeypatch.setitem(me._REPOS, "miWorkbench", lambda: tmp_path / "absent")
    for switch in ("MISERVER_REQUIRE_CROSS_REPO_CHECKS", "MISTUDIO_REQUIRE_CROSS_REPO_CHECKS"):
        monkeypatch.delenv(switch, raising=False)
    with pytest.raises(pytest.skip.Exception):
        me.test_each_consumers_copy_is_the_committed_contract_or_behind_it_additively("miWorkbench")
    monkeypatch.setenv("MISTUDIO_REQUIRE_CROSS_REPO_CHECKS", "1")
    # Either outcome is caught, then the type checked: a skip escaping `pytest.raises(fail)` reports this test
    # as skipped instead of failing it.
    with pytest.raises((pytest.fail.Exception, pytest.skip.Exception)) as outcome:
        me.test_each_consumers_copy_is_the_committed_contract_or_behind_it_additively("miWorkbench")
    assert outcome.type is pytest.fail.Exception


def test_each_consumer_path_names_this_majors_file():
    """R2-3: a consumer's copy is the file of the major this repository publishes."""
    for app, (path, _state) in CONSUMERS.items():
        assert Path(path).name == CONTRACT_FILENAME, app


# ── miAssist as a consumer (006 task 0.6) ─────────────────────────────────────────────────────────────


def test_the_consumers_and_their_paths():
    """The consumer set is exactly these apps at these paths: dropping miAssist's entry, or moving its
    path off 006's `docs/schemas/<file>`, fails here (miAssist 006 task 0.6 reads this entry)."""
    assert {app: path for app, (path, _state) in CONSUMERS.items()} == {
        "miWorkbench": "backend/src/mcp_server/vendored/" + CONTRACT_FILENAME,
        "miAssist": "docs/schemas/" + CONTRACT_FILENAME,
    }
    assert set(_REPOS) == set(CONSUMERS)
    from tests.support import sibling_repos

    assert _REPOS == {"miWorkbench": sibling_repos.workbench_repo, "miAssist": sibling_repos.assist_repo}


def test_an_exempt_consumer_is_still_pending():
    """The CI exemption lasts only while the consumer is pending: a vendored consumer must be checked out."""
    for app in NOT_CHECKED_OUT_IN_CI:
        assert CONSUMERS[app][1] == "pending", (
            f"{app} is vendored but the cross-repo-contract job cannot check it out: add its deploy key "
            f"and checkout, then drop it from NOT_CHECKED_OUT_IN_CI"
        )


def _assist_layout(monkeypatch, tmp_path, copy: dict | None):
    import tests.unit.test_answers_contract_consumers as me

    repo = tmp_path / "miAssist"
    if copy is not None:
        (repo / "docs" / "schemas").mkdir(parents=True)
        (repo / "docs" / "schemas" / CONTRACT_FILENAME).write_text(contract_tools.render(copy), encoding="utf-8")
    monkeypatch.setitem(me._REPOS, "miAssist", lambda: repo)
    return me


def test_miassist_absent_skips_loudly_under_the_switch_only_while_its_path_is_unset(monkeypatch, tmp_path):
    me = _assist_layout(monkeypatch, tmp_path, None)
    monkeypatch.setenv("MISTUDIO_REQUIRE_CROSS_REPO_CHECKS", "1")
    monkeypatch.delenv("MIASSIST_REPO", raising=False)
    with pytest.raises(pytest.skip.Exception, match="still pending"):
        me.test_each_consumers_copy_is_the_committed_contract_or_behind_it_additively("miAssist")
    monkeypatch.setenv("MIASSIST_REPO", str(tmp_path / "miAssist"))
    # Caught as either outcome, then checked: a skip raised inside `pytest.raises(fail)` would escape and
    # report this test as skipped, not failed (the S2 control N4 survived exactly that way).
    with pytest.raises((pytest.fail.Exception, pytest.skip.Exception)) as outcome:
        me.test_each_consumers_copy_is_the_committed_contract_or_behind_it_additively("miAssist")
    assert outcome.type is pytest.fail.Exception, "a set MIASSIST_REPO that is absent must fail, not skip"


def test_a_vendored_consumer_absent_under_the_switch_fails_even_if_exempt(monkeypatch, tmp_path):
    """The exemption is for a pending consumer only: once vendored, an absent checkout fails."""
    me = _assist_layout(monkeypatch, tmp_path, None)
    monkeypatch.setitem(me.CONSUMERS, "miAssist", (me.CONSUMERS["miAssist"][0], "vendored"))
    monkeypatch.setenv("MISTUDIO_REQUIRE_CROSS_REPO_CHECKS", "1")
    monkeypatch.delenv("MIASSIST_REPO", raising=False)
    with pytest.raises((pytest.fail.Exception, pytest.skip.Exception)) as outcome:
        me.test_each_consumers_copy_is_the_committed_contract_or_behind_it_additively("miAssist")
    assert outcome.type is pytest.fail.Exception, "a vendored consumer was exempted"


def test_miassists_copy_is_compared_when_it_is_checked_out(monkeypatch, tmp_path, committed):
    ahead = json.loads(json.dumps(committed))
    ahead["responses"]["GET /api/a-route-the-server-dropped"] = {"200": {"type": "object"}}
    me = _assist_layout(monkeypatch, tmp_path, ahead)
    with pytest.raises(AssertionError, match="breaking"):
        me.test_each_consumers_copy_is_the_committed_contract_or_behind_it_additively("miAssist")


def test_miassists_equal_copy_passes(monkeypatch, tmp_path, committed):
    me = _assist_layout(monkeypatch, tmp_path, committed)
    me.test_each_consumers_copy_is_the_committed_contract_or_behind_it_additively("miAssist")
