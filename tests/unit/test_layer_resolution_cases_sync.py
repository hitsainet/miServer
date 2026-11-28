"""miWorkbench's layer-resolution cases (ENH-011 X-1): the vendored copy must stay byte-identical.

miWorkbench publishes `docs/schemas/layer-resolution-cases.json` (its X-1, `src/ml/layer_discovery.py`): for each
model family, the dotted path of the text decoder's layer list, its depth on a tiny config, and the vision stack a
first-match search used to take. miServer resolves layers with its own code (`miserver/ml/layer_resolution.py`) and
tests it on the same cases (`tests/unit/ml/test_layer_resolution_cases.py`), so the two resolvers cannot disagree
about which stack an index names without one of the two suites going red.

**Byte identity.** Skipped loudly when miWorkbench is not checked out (the public mirror), REQUIRED under
`MISERVER_REQUIRE_CROSS_REPO_CHECKS=1`, which the `cross-repo-contract` CI job sets with miWorkbench checked out.
Re-vendor by copying; never hand-edit the copy. Change a case in miWorkbench first.

**The version this repo has read.** A version bump is a new rule to read here deliberately, not a silent re-copy.

This file imports nothing from `miserver` and needs pytest alone, so the `cross-repo-contract` job can run it without
the ML stack (`test_cross_repo_sync_files_need_only_pytest.py` holds that).
"""

from __future__ import annotations

import json

import pytest

from tests.support.sibling_repos import REPO_ROOT, cross_repo_checks_required, workbench_repo

NAME = "layer-resolution-cases.json"
VENDORED = REPO_ROOT / "docs" / "schemas" / NAME
REQUIRED = cross_repo_checks_required()
#: The cases-file version this repo's resolver and tests were written against.
READ_VERSION = 1


def test_the_vendored_file_is_byte_identical_to_miworkbenchs():
    source = workbench_repo() / "docs" / "schemas" / NAME
    if not source.exists():
        message = f"{source} is not checked out — the cross-repo byte pin did NOT run"
        if REQUIRED:
            pytest.fail(message + " and MISERVER_REQUIRE_CROSS_REPO_CHECKS=1")
        pytest.skip(message)
    assert VENDORED.read_bytes() == source.read_bytes(), (
        "the vendored layer-resolution cases differ from miWorkbench's; re-copy them, never edit the copy"
    )


def test_the_vendored_version_is_the_one_this_repo_has_read():
    document = json.loads(VENDORED.read_text())
    assert document["version"] == READ_VERSION
    assert document["cases"], "an empty cases file pins nothing"
