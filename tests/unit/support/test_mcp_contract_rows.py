"""`tests/support/mcp_contract_rows.endpoints()` reads §4 of `docs/mcp-contract.md` (044 task 2.4).

A parser that matched nothing would make every test parametrized from it pass empty, so its count is pinned: §4 named
100 endpoints on 2026-11-25, and 101 once 044 adds the `GET /api/health/inference` that `miserver_list_models`
already calls (FR-044.22). A row added to §4 moves this number on purpose.
"""

from __future__ import annotations

from tests.support.mcp_contract_rows import MINIMUM, endpoints, normalise

#: §4's endpoint count: 100 on 2026-11-25, 101 with `GET /api/health/inference` (FR-044.22).
SECTION_4_ENDPOINTS = 101


def test_it_reads_every_section_4_endpoint():
    found = endpoints()
    assert len(found) >= MINIMUM
    assert len(found) == SECTION_4_ENDPOINTS


def test_it_reads_rows_of_every_shape():
    found = endpoints()
    # a plain row, a row with a body note, a query string, and a parameter named differently from the route's
    for row in [
        ("GET", "/api/sensing/status"),
        ("PUT", "/api/clusters/active/intensity"),
        ("GET", "/api/clusters/hub/search"),
        ("POST", "/api/profiles/{}/activate"),
        ("GET", "/api/approvals/actions"),
    ]:
        assert row in found, row


def test_normalise():
    assert normalise("/api/x/{id}/y?q=1") == "/api/x/{}/y"
    assert normalise("/api/x/{repo_id:path}") == "/api/x/{}"
