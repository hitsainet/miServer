"""`docs/mcp-contract.md` records the answer contract's publication and its current revision (044 FR-044.22).

The changelog must name the contract file, its `$id` and its CURRENT `revision`: a regeneration that raises the
revision without a changelog line goes red. §2 points readers at the file for answer shapes. On the public
mirror, where `docs/mcp-contract.md` is stripped, this skips loudly.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from miserver.api.schemas.answers_contract import CONTRACT_FILENAME, CONTRACT_ID

ROOT = Path(__file__).resolve().parents[2]
DOC = ROOT / "docs" / "mcp-contract.md"


def _doc() -> str:
    if not DOC.is_file():
        pytest.skip(f"{DOC.name} is not in this checkout (the public mirror strips docs/)")
    return DOC.read_text(encoding="utf-8")


def _revision() -> int:
    return json.loads((ROOT / "docs" / "schemas" / CONTRACT_FILENAME).read_text())["revision"]


def test_the_changelog_cites_the_current_revision():
    text = _doc()
    changelog = text[text.index("## 1. Versioning rule") : text.index("## 2. ")]
    # A changelog entry names the file, then (within the entry's sentence) "revision N".
    cited = {
        int(n)
        for n in re.findall(
            rf"{re.escape(CONTRACT_FILENAME)}`[^*]{{0,400}}?revision (\d+)", changelog
        )
    }
    assert _revision() in cited, (
        f"no changelog line cites {CONTRACT_FILENAME} revision {_revision()}"
    )


def test_section_2_points_at_the_file():
    text = _doc()
    section = text[text.index("## 2. ") : text.index("## 3. ")]
    assert CONTRACT_FILENAME in section and CONTRACT_ID in section
