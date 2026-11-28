"""Every refusal code the MCP contract and the manual name is one miServer can send (043; R3 review fix
(OR3-043-1), 2026-11-16).

The contract and `manual/docs/api/weights.md` named `PROBE_MODEL_CHANGED` for arming during training, while
`ProbeModelChangedError.code` is `MODEL_CHANGED`: a gateway written from the normative contract would key on a
code that never arrives, and no test noticed.

What counts as a documented refusal code, in `docs/mcp-contract.md` and every page under `manual/docs/`
(except `getting-started/`, whose install guides name environment variables in the same sentence shapes):
- a backticked `UPPER_SNAKE` token after "refused", "refuses", "answers", "rejects" (within the sentence), or
  right after an HTTP status (`409 MODEL_CHANGED`);
- the first column of a table row whose second column is an HTTP status (`manual/docs/reference/error-codes.md`).

What a code miServer sends is: an error class's `code` (`miserver/core/errors.py`), or any string literal in the
package (codes built as literals: `QUEUE_FULL`, `MODEL_LOADING`, ...), read by AST through `package_root()`.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

from miserver.core import errors
from tests.support.package_root import package_root

ROOT = Path(__file__).resolve().parents[2]
CODE = r"`(?:[45]\d\d\s+)?([A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+)`"
IN_PROSE = re.compile(r"(?:(?i:refus\w*|answer\w*|reject\w*)[^`.]{0,30}?|\b[45]\d\d\s+\*{0,2})" + CODE)
IN_TABLE = re.compile(r"^\|\s*" + CODE + r"\s*\|\s*[45]\d\d\b", re.MULTILINE)
#: The floor: the documents name well over this many codes, so a broken pattern cannot pass empty.
MIN_CODES = 60


#: SOURCE-ONLY. `sync-to-clean` strips `docs/` except `schemas/`, and the public mirror is where the image is
#: built: this file first read the contract unconditionally and failed the mirror's Backend Tests (043-1,
#: 2026-11-17). The manual ships to the mirror and is checked everywhere; the contract is checked in the source
#: repo and SKIPPED LOUDLY where absent (the split `test_batch_contract_rows_are_served.py` uses).
CONTRACT = ROOT / "docs" / "mcp-contract.md"


def _manual_pages() -> list[Path]:
    return [p for p in (ROOT / "manual" / "docs").rglob("*.md") if "getting-started" not in p.parts]


def _documents() -> list[Path]:
    return [CONTRACT, *_manual_pages()] if CONTRACT.exists() else _manual_pages()


def documented_codes(text: str) -> set[str]:
    return {m.group(1) for m in IN_PROSE.finditer(text)} | {m.group(1) for m in IN_TABLE.finditer(text)}


def _subclasses(cls):
    for sub in cls.__subclasses__():
        yield sub
        yield from _subclasses(sub)


def sent_codes() -> set[str]:
    codes = {c.code for c in _subclasses(errors.MiServerError) if isinstance(getattr(c, "code", None), str)}
    for path in package_root().rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                codes.add(node.value)
    return codes


def test_every_documented_refusal_code_is_sent():
    named: dict[str, list[str]] = {}
    for doc in _documents():
        for code in documented_codes(doc.read_text(encoding="utf-8")):
            named.setdefault(code, []).append(str(doc.relative_to(ROOT)))
    assert len(named) >= MIN_CODES, f"found only {len(named)} documented codes: the patterns are failing open"
    assert "MODEL_CHANGED" in named, "precondition: the arming and scoring refusal is documented"
    sent = sent_codes()
    unknown = {code: where for code, where in sorted(named.items()) if code not in sent}
    assert unknown == {}, f"documented refusal codes miServer never sends: {unknown}"


@pytest.mark.skipif(
    not CONTRACT.exists(),
    reason="docs/mcp-contract.md is not in this checkout: the public mirror strips docs/ except schemas/. "
    "Checked in the source repo.",
)
def test_the_contract_is_read_and_names_the_arming_code():
    """In the source repo the contract is one of the documents read, so its codes are held to the rule too."""
    assert CONTRACT in _documents()
    assert "MODEL_CHANGED" in documented_codes(CONTRACT.read_text(encoding="utf-8"))


def test_the_patterns_see_both_forms():
    """The scanner's own reach: a code in prose and in an error table are both read."""
    assert documented_codes("arming refuses `409 MODEL_CHANGED` if a commit lands") == {"MODEL_CHANGED"}
    assert documented_codes("is refused\n  `PROBE_MODEL_CHANGED`.") == {"PROBE_MODEL_CHANGED"}
    assert documented_codes("| `MODEL_NOT_FOUND` | 404 | No model |\n") == {"MODEL_NOT_FOUND"}
    assert documented_codes("set `PROBE_TRANSCRIPTS` to on") == set()
