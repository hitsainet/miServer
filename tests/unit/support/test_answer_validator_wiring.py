"""The answer validator is WIRED, and it FIRES on a real route (044 FR-044.10; FTDD §10).

Two assertions, the wiring and the payload, not that the module exists:
1. every route of a freshly built `create_app()` runs a handler the validator built (marked `__answer_checked__`);
2. a real route answering off its contract fails, naming the route and the key, both through `TestClient` (the
   raise) and through the conftest fixture's record (IC-4), even when the app turns the raise into a 500.

The strict-form case (FR-044.10 AC) is here too: the validator's form of a typed object refuses an extra key that
the published form accepts.
"""

from __future__ import annotations

import pytest
from fastapi.routing import APIRoute, iter_route_contexts
from fastapi.testclient import TestClient

from tests.support import answer_validator
from tests.unit.api.test_probe_event_filters import client  # noqa: F401 - fixture
from tests.unit.f28_fixtures import db  # noqa: F401 - fixture
from tests.unit.probe_fixtures import probe_definition


def test_every_route_of_a_fresh_app_runs_a_checked_handler():
    from miserver.main import create_app

    app = create_app()
    routes = [
        ctx for ctx in iter_route_contexts(app.routes) if isinstance(ctx.original_route, APIRoute)
    ]
    assert len(routes) >= 184
    unchecked = [
        f"{sorted(ctx.methods or ())} {ctx.path}"
        for ctx in routes
        if answer_validator.handler_of(ctx.app) is None
    ]
    assert unchecked == []


def _probe_id(client) -> str:  # noqa: F811
    response = client.post("/api/probes/import", json=probe_definition(name="canary"))
    assert response.status_code == 200, response.text
    return response.json()["data"]["id"]


def _retype(monkeypatch, field: str, value) -> None:
    """The stored row answers one identity field with the wrong JSON type; the route does not validate it
    (documented-only), so only the validator can see it."""
    from miserver.db.repositories.probe_repository import ProbeRepository

    real = ProbeRepository.get

    async def get(self, probe_id):
        probe = await real(self, probe_id)
        if probe is not None:
            probe.__dict__[field] = value
        return probe

    monkeypatch.setattr(ProbeRepository, "get", get)


def test_an_off_contract_answer_raises_through_testclient(client, monkeypatch):  # noqa: F811
    probe_id = _probe_id(client)
    _retype(monkeypatch, "decision_revision", "seven")
    with pytest.raises(answer_validator.AnswerContractError) as raised:
        client.get(f"/api/probes/{probe_id}")
    assert raised.value.route == "GET /api/probes/{probe_id}"
    assert raised.value.json_path == "data.decision_revision"
    assert raised.value.status == "200"
    assert answer_validator.FAILURES, "the failure was not recorded for the conftest fixture"
    answer_validator.FAILURES.clear()


def test_an_off_contract_answer_is_recorded_when_the_app_answers_500(client, monkeypatch):  # noqa: F811
    from fastapi.testclient import TestClient

    probe_id = _probe_id(client)
    _retype(monkeypatch, "decision_revision", "seven")
    quiet = TestClient(client.app, raise_server_exceptions=False)
    response = quiet.get(f"/api/probes/{probe_id}")
    assert response.status_code == 500
    assert len(answer_validator.FAILURES) == 1
    failure = answer_validator.FAILURES[0]
    assert (
        failure.route == "GET /api/probes/{probe_id}" and "decision_revision" in failure.json_path
    )
    answer_validator.FAILURES.clear()


def test_the_strict_form_refuses_an_extra_key_the_published_form_accepts(client):  # noqa: F811
    """FR-044.10 AC (SH13 vs SH7), on the probe summary: a real import answer, then one extra key."""
    from jsonschema import Draft202012Validator
    from referencing import Registry, Resource
    from referencing.jsonschema import DRAFT202012

    contract = answer_validator._contract()
    key = "POST /api/probes/import"
    response = client.post("/api/probes/import", json=probe_definition(name="strict-form"))
    assert response.status_code == 200, response.text
    body = response.json()
    published = Registry().with_resource(
        contract["$id"], Resource.from_contents(contract, default_specification=DRAFT202012)
    )
    uri = contract["$id"] + "#" + answer_validator._pointer("responses", key, "200")
    assert not list(Draft202012Validator({"$ref": uri}, registry=published).iter_errors(body))
    assert not list(answer_validator.validator_for(key, "200").iter_errors(body))
    extra = {**body, "data": {**body["data"], "surprise": 1}}
    assert not list(Draft202012Validator({"$ref": uri}, registry=published).iter_errors(extra))
    errors = list(answer_validator.validator_for(key, "200").iter_errors(extra))
    assert errors, "the strict form accepted a key ProbeSummary does not declare"


def test_the_import_time_app_is_wrapped_too(tmp_path):
    """044 R1-8: `miserver.main.app` is built when `miserver.main` is first imported, which a test
    module may do at its top, during collection; some tests serve through that app. It is wrapped
    only because `tests/conftest.py` imports the validator FIRST. Run in a subprocess with the real
    conftest as the only plugin and a planted module that imports `miserver.main` before anything
    else, so the order cannot be rescued by this module's own imports."""
    import os
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[3]
    planted = tmp_path / "test_planted_import_order.py"
    planted.write_text(
        "import miserver.main as main\n"
        "from fastapi.routing import APIRoute, iter_route_contexts\n"
        "def test_it():\n"
        "    from tests.support import answer_validator\n"
        "    app = main.fastapi_app_of(main.app)\n"
        "    routes = [c for c in iter_route_contexts(app.routes)\n"
        "              if isinstance(c.original_route, APIRoute)]\n"
        "    assert len(routes) >= 184\n"
        "    assert [str(c.path) for c in routes if answer_validator.handler_of(c.app) is None] == []\n"
    )
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-o",
            "addopts=",
            "-p",
            "no:cacheprovider",
            "-p",
            "tests.conftest",
            "--rootdir",
            str(root),
            str(planted),
        ],
        cwd=root,
        env={**os.environ, "PYTHONPATH": str(root)},
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert proc.returncode == 0, (proc.stdout + proc.stderr)[-3000:]


def test_an_undeclared_status_fails():
    with pytest.raises(answer_validator.AnswerContractError, match="status 299 is not declared"):
        answer_validator.check_body("GET /api/sensing/status", 299, {})
    answer_validator.FAILURES.clear()


def test_a_route_the_contract_does_not_hold_fails():
    with pytest.raises(answer_validator.AnswerContractError, match="not in the contract"):
        answer_validator.check_body("GET /api/no-such-route", 200, {})
    answer_validator.FAILURES.clear()


def test_a_miserver_endpoint_outside_the_map_fails():
    """An endpoint under `miserver.` that the endpoint map does not know is a route the contract
    does not cover: it fails, it is not skipped (a test double outside `miserver.` is skipped)."""
    from types import SimpleNamespace

    from fastapi.responses import JSONResponse

    def stray() -> None: ...

    stray.__module__ = "miserver.api.routes.management.stray"
    route = SimpleNamespace(endpoint=stray, path="/stray")
    request = SimpleNamespace(method="GET")
    with pytest.raises(answer_validator.AnswerContractError, match="not in the contract"):
        answer_validator._check(route, request, JSONResponse({"x": 1}))
    answer_validator.FAILURES.clear()
    stray.__module__ = "tests.doubles"
    answer_validator._check(route, request, JSONResponse({"x": 1}))  # skipped: a test double


def test_an_unasserted_500_still_fails_the_test(tmp_path):
    """IC-4: the conftest fixture fails a test whose answer the app turned into a 500 it never
    asserted on. A planted test file runs under the real conftest, in a subprocess."""
    import os
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[3]
    planted = tmp_path / "test_planted_500.py"
    planted.write_text(
        "from fastapi.testclient import TestClient\n"
        "from tests.support import answer_validator\n"
        "def test_it(monkeypatch):\n"
        "    from miserver.main import create_app\n"
        "    real = answer_validator.check_body\n"
        "    def bad(key, status, body):\n"
        "        return real(key, status, {**body, 'surprise': 1})\n"
        "    monkeypatch.setattr(answer_validator, 'check_body', bad)\n"
        "    TestClient(create_app(), raise_server_exceptions=False).get('/api/health/version')\n"
    )
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-o",
            "addopts=",
            "-p",
            "no:cacheprovider",
            "-p",
            "tests.conftest",
            "--rootdir",
            str(root),
            str(planted),
        ],
        cwd=root,
        env={**os.environ, "PYTHONPATH": str(root)},
        capture_output=True,
        text=True,
        timeout=300,
    )
    out = proc.stdout + proc.stderr
    assert proc.returncode != 0, out[-2000:]
    assert "answer(s) off the contract" in out, out[-2000:]


def test_an_endpoint_serving_two_routes_is_refused():
    """R1-5: the map is keyed by endpoint; one endpoint on two paths with one method is ambiguous."""
    from fastapi import FastAPI

    app = FastAPI()

    async def twice() -> dict:
        return {}

    app.add_api_route("/one", twice, methods=["GET"])
    app.add_api_route("/two", twice, methods=["GET"])
    with pytest.raises(AssertionError, match="cannot tell its answers apart"):
        answer_validator.map_routes(app)


@pytest.mark.parametrize(
    ("method", "path", "key"),
    [
        ("get", "/api/health/version", "GET /api/health/version"),
        ("get", "/api/approvals/actions", "GET /api/approvals/actions"),
        ("get", "/api/probes", "GET /api/probes"),
        ("get", "/api/sensing/status", "GET /api/sensing/status"),
    ],
)
def test_an_extra_key_on_a_real_answer_is_refused_whatever_the_route(
    client, monkeypatch, method, path, key  # noqa: F811
):
    """FR-044.6 AC, by behaviour (044 R2-6): real answers of four kinds (bare, envelope, page, an
    enforced route) each pass, and each with one planted key is refused. No skip list, marker or
    switch lets any of them through."""
    response = getattr(client, method)(path)
    assert response.status_code == 200, response.text
    body = response.json()
    answer_validator.check_body(key, 200, body)
    # R3-2: the LIVE checker too, through the route wrapper: the body it parses gains the key, and the
    # recorded failure names it (a skip list in `_check` would let this through).
    import json as real_json

    class Planting:
        @staticmethod
        def loads(raw):
            parsed = real_json.loads(raw)
            spot = parsed["data"] if isinstance(parsed.get("data"), dict) else parsed
            spot["planted_by_the_test"] = 1
            return parsed

        def __getattr__(self, name):
            return getattr(real_json, name)

    monkeypatch.setattr(answer_validator, "json", Planting())
    quiet = TestClient(client.app, raise_server_exceptions=False)
    assert getattr(quiet, method)(path).status_code == 500
    assert [f.route for f in answer_validator.FAILURES] == [key]
    assert "planted_by_the_test" in answer_validator.FAILURES[0].message
    answer_validator.FAILURES.clear()
    monkeypatch.undo()
    target = body["data"] if isinstance(body.get("data"), dict) else body
    target["planted_by_the_test"] = 1
    with pytest.raises(answer_validator.AnswerContractError, match="planted_by_the_test"):
        answer_validator.check_body(key, 200, body)
    answer_validator.FAILURES.clear()
