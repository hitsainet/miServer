"""`docs/schemas/miserver-answers-v1.json` is what the live routes declare (044 FR-044.1, .2, .18).

The contract is GENERATED (`miserver/api/schemas/answers_contract.py`) from `register_routes`' OpenAPI document,
so it moves when a route or an answer model moves, and this test goes red until the committed file is regenerated
(`MISERVER_WRITE_CONTRACTS=1 pytest tests/unit/api/test_answers_contract.py`). The writer refuses a breaking change
(FR-044.16): that needs `miserver-answers-v2.json` beside v1. Regeneration is the moment a consumer's vendored copy
(miWorkbench, `tests/unit/test_answers_contract_consumers.py`) must move too.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from miserver.api.schemas import contract_tools
from miserver.api.schemas.answers_contract import (
    CONTRACT_FILENAME,
    CONTRACT_ID,
    KIND,
    answers_contract,
    body_of,
    problems,
    render,
    write_contract,
)
from tests.support.env_flags import flag

ROOT = Path(__file__).resolve().parents[3]
CONTRACT = ROOT / "docs" / "schemas" / CONTRACT_FILENAME
REGENERATE = "MISERVER_WRITE_CONTRACTS=1 pytest tests/unit/api/test_answers_contract.py"


@pytest.fixture(scope="module")
def generated() -> dict:
    body = answers_contract()
    if flag("WRITE_CONTRACTS") == "1":
        write_contract(CONTRACT, body)
    return body


@pytest.fixture(scope="module")
def committed(generated: dict) -> dict:  # noqa: ARG001 (written first, when asked)
    assert CONTRACT.is_file(), f"{CONTRACT} is missing; run {REGENERATE}"
    return json.loads(CONTRACT.read_text(encoding="utf-8"))


def test_the_committed_contract_is_the_generated_one(generated: dict, committed: dict):
    assert body_of(committed) == generated, (
        f"{CONTRACT.name} no longer matches the registered routes. Regenerate it ({REGENERATE}); "
        "then miWorkbench re-vendors it."
    )


def test_the_committed_bytes_are_the_rendered_form(committed: dict):
    assert CONTRACT.read_text(encoding="utf-8") == render(committed), (
        f"{CONTRACT.name} is not in its rendered form (sorted keys, two-space indent, newline): {REGENERATE}"
    )


def test_its_digest_and_revision_hold(committed: dict):
    """FR-044.18 (TD11): the embedded digest is recomputed; a hand edit without the writer goes red."""
    assert problems(committed) == []


def test_a_hand_edit_without_a_revision_change_is_caught(committed: dict):
    """FR-044.18 AC: one schema edited by hand, `revision` and `content_sha256` left as they were."""
    edited = json.loads(json.dumps(committed))
    name = sorted(edited["$defs"])[0]
    edited["$defs"][name]["description"] = "edited by hand"
    assert any("content_sha256" in p for p in problems(edited)), problems(edited)


def test_rendering_is_deterministic_across_hash_seeds():
    """FR-044.1 AC: two processes with different hash seeds render byte-identical text."""
    code = (
        "import hashlib, sys\n"
        "from miserver.api.schemas.answers_contract import answers_contract, render\n"
        "sys.stdout.write('DIGEST=' + hashlib.sha256(render(answers_contract()).encode()).hexdigest())\n"
    )
    digests = []
    for seed in ("1", "2"):
        env = {**os.environ, "PYTHONHASHSEED": seed}
        proc = subprocess.run(
            [sys.executable, "-c", code],
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=300,
        )
        assert proc.returncode == 0, proc.stderr[-3000:]
        digests.append(proc.stdout.rsplit("DIGEST=", 1)[1].strip())
    assert digests[0] == digests[1]
    assert len(digests[0]) == 64


def test_its_identity(committed: dict):
    """FR-044.1 AC (SH1): the `$id`, `kind` and file name exactly; no new name carries the package's old name."""
    assert CONTRACT_FILENAME == "miserver-answers-v1.json"
    assert committed["$id"] == CONTRACT_ID == "https://hitsai.net/schemas/miserver-answers-v1.json"
    assert committed["kind"] == KIND == "miserver.answer-contract/v1"
    assert isinstance(committed["revision"], int) and committed["revision"] >= 1
    names = [CONTRACT_FILENAME, KIND, CONTRACT_ID, *committed["$defs"], *committed["errors"]]
    # DR-68: a generated name in a published contract, frozen since before 044 (docs/legacy-identifiers.md).
    frozen = {"MillmExtension"}
    assert not [n for n in names if "millm" in n.lower() and n not in frozen]


def test_the_file_name_is_written_in_the_generator_only():
    """FR-044.1: `CONTRACT_FILENAME` is the one place the name is written in the package."""
    from tests.support.package_root import package_rel, package_root

    hits = [
        p
        for p in package_root().rglob("*.py")
        if CONTRACT_FILENAME in p.read_text(encoding="utf-8")
    ]
    assert [p.relative_to(ROOT).as_posix() for p in hits] == [
        package_rel("api", "schemas", "answers_contract.py")
    ]


def test_the_digest_is_the_one_contract_tools_computes(committed: dict):
    body = {k: v for k, v in committed.items() if k not in ("revision", "content_sha256")}
    expected = hashlib.sha256(render(body).encode("utf-8")).hexdigest()
    assert committed["content_sha256"] == expected == contract_tools.content_digest(committed)
