"""Every route the MCP relays call is in the answer contract (044 FR-044.21, parent BR-1).

The relayed set is `docs/mcp-contract.md` §4, which `test_mcp_contract_consistency.py` already checks two-way
against miWorkbench's live registry; a scan of relay source missed `GET /api/approvals/actions` on
2026-11-25, so the document is the source here. Each endpoint must be a TYPED `responses` key, or an `excluded`
key with a reason. On the public mirror, where `docs/mcp-contract.md` is stripped, this skips loudly.
"""

from __future__ import annotations

import json
from pathlib import Path

from miserver.api.schemas import contract_tools as tools
from miserver.api.schemas.answers_contract import CONTRACT_FILENAME
from tests.support.mcp_contract_rows import endpoints, normalise

ROOT = Path(__file__).resolve().parents[2]


def _keys(contract: dict) -> dict:
    out = {}
    for section in ("responses", "excluded"):
        for key in contract[section]:
            method, path = key.split(" ", 1)
            out[(method, normalise(path))] = (section, key)
    return out


def test_every_relayed_route_is_typed_or_excluded_with_a_reason():
    contract = json.loads((ROOT / "docs" / "schemas" / CONTRACT_FILENAME).read_text())
    keys = _keys(contract)
    ext = frozenset(contract["external"])
    missing, untyped = [], []
    for endpoint in sorted(endpoints()):
        if endpoint not in keys:
            missing.append(endpoint)
            continue
        section, key = keys[endpoint]
        if section == "excluded":
            assert contract["excluded"][key]["reason"], key
            continue
        entry = contract["responses"][key]
        if not all(
            tools.is_typed(s, contract["$defs"], ext) for k, s in entry.items() if k[:1].isdigit()
        ):
            untyped.append(key)
    assert missing == [] and untyped == [], {"not in the contract": missing, "untyped": untyped}
