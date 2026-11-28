"""The miServer routes the MCP relays call, read from `docs/mcp-contract.md` §4 (044 FTID §8.2).

§4 is the one hand-written list of relayed routes; `tests/unit/test_mcp_contract_consistency.py` already checks its
tool names two-way against miWorkbench's live registry. This reads its ENDPOINTS: every backticked `METHOD /path`
in a §4 table row whose first cell names a `miserver_` tool. Path parameters are normalised to `{}` (the document
writes `{id}` where a route says `{profile_id}`) and query strings are dropped. Callers match the result against
the contract's keys with `normalise`.

On the public mirror `docs/mcp-contract.md` is stripped; `endpoints()` then skips the calling test loudly, as
`tests/unit/test_mcp_contract_consistency.py` does.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

CONTRACT = Path(__file__).resolve().parents[2] / "docs" / "mcp-contract.md"
#: The §4 heading `tests/unit/test_mcp_contract_consistency.py` reads (`INVENTORY_HEADING`).
INVENTORY_HEADING = "## 4. Endpoint inventory"
_ENDPOINT = re.compile(r"`(GET|POST|PUT|PATCH|DELETE) (/[^`\s?]*)[^`]*`")
#: A parse that matched almost nothing must fail, not pass empty.
MINIMUM = 90


def normalise(path: str) -> str:
    return re.sub(r"\{[^}]*\}", "{}", path.split("?")[0]).rstrip("/") or "/"


def endpoints() -> set[tuple[str, str]]:
    if not CONTRACT.is_file():
        pytest.skip(
            f"{CONTRACT.name} is not in this checkout (the public mirror strips docs/): the §4 routes "
            "cannot be read here; they are checked in the private repository's CI"
        )
    lines = CONTRACT.read_text(encoding="utf-8").splitlines()
    start = next((i for i, line in enumerate(lines) if line.startswith(INVENTORY_HEADING)), None)
    assert start is not None, f"no {INVENTORY_HEADING!r} heading in {CONTRACT.name}"
    end = next(
        (i for i, line in enumerate(lines[start + 1 :], start + 1) if line.startswith("## ")),
        len(lines),
    )
    out: set[tuple[str, str]] = set()
    for line in lines[start:end]:
        if not line.startswith("|") or line.startswith("| _("):
            continue
        cells = line.strip().strip("|").split("|")
        if not re.search(r"`miserver_[a-z_]+`", cells[0]):
            continue
        for method, path in _ENDPOINT.findall("|".join(cells[1:])):
            out.add((method, normalise(path)))
    assert len(out) >= MINIMUM, (
        f"read only {len(out)} endpoints from §4: the parse no longer matches the table"
    )
    return out
