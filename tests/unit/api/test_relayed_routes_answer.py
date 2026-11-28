"""Every route an MCP relay calls ANSWERS, and its answer is the one the contract describes (044 FR-044.11, TD6).

`docs/mcp-contract.md` §4 names each miServer route a relay calls. For each one this module holds one case that
calls the route over HTTP through the LIVE app (`create_app()`), as the operator, and requires a 2xx answer. The
session's answer validator (`tests/support/answer_validator.py`, installed by `tests/conftest.py`) checks every 2xx
JSON answer against `docs/schemas/miserver-answers-v1.json` in its strict form; the test also wraps `check_body` and
requires exactly ONE check of the endpoint's own contract key, on a body equal to the response it got (payload and
count, so a route that escaped the validator, or was checked against another key, fails here).

The test is parametrized over `endpoints()` (§4), NOT over `CASES`: a §4 row without a case fails rather than
vanishing, and `test_the_cases_are_exactly_the_section_4_endpoints` fails on a case that §4 does not name.

Setup rules (the brief, TD6): real services over the test database; seeding through the real routes; mocks only at
external boundaries (the GPU, model weights, the Hugging Face hub, the rule runner process), and a mock whose return
becomes the answer is built from the real code's types.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import subprocess
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from tests.support import answer_validator
from tests.support import behaviour_factory as bf
from tests.support import capture_pre038_goldens as cap
from tests.support import rule_set_factory as rf
from tests.support import verdict_factory as vf
from tests.support.mcp_contract_rows import CONTRACT as MCP_CONTRACT
from tests.support.mcp_contract_rows import endpoints, normalise

# On the public mirror `docs/mcp-contract.md` is stripped: §4 cannot be read, so the parameters cannot
# be built. Skip the module loudly, at collection (a skip inside the parametrize call would be an error
# that interrupts the whole shard's collection; found by the mirror simulation, 2026-11-25).
if not MCP_CONTRACT.is_file():
    pytest.skip(f"{MCP_CONTRACT.name} is not in this checkout (the public mirror strips docs/): the §4 "
                "routes are answered in the private repository's CI", allow_module_level=True)

ROOT = Path(__file__).resolve().parents[3]
FIXTURES = ROOT / "tests" / "fixtures"
ANSWERS = ROOT / "docs" / "schemas" / "miserver-answers-v1.json"
#: The one §4 endpoint the contract excludes (JSON Lines, streamed): the validator does not check it.
EXCLUDED_MEDIA = {("GET", "/v1/files/{}/content"): "application/jsonl"}

# ── recording (the answers siblings vendor, e.g. miWorkbench 046's `miserver_answers_recorded_044.json`) ───────
#: A file path: write one scrubbed answer per §4 route there at the end of this module's run.
RECORD_ENV = "MISERVER_RECORD_ANSWERS"
#: What a workstation path in an answer becomes. The routes answer their cache and batch folders, which this module
#: points at pytest's `tmp_path` (`/tmp/pytest-of-<user>/…`), and a recording can ship to a public mirror.
PLACEHOLDER = "/tmp/recorded"
_PYTEST_TEMP = re.compile(r"/tmp/pytest-of-[^/\s\"']+(?:/pytest-\d+)?(?:/[^/\s\"']+)?")
_HOME = str(Path.home())
RECORDED: dict[str, dict[str, Any]] = {}


def scrub(value: Any, tmp_path: Path) -> Any:
    """`value` with this test's `tmp_path`, any pytest temp folder and the home folder replaced by placeholders."""
    if isinstance(value, dict):
        return {scrub(k, tmp_path): scrub(v, tmp_path) for k, v in value.items()}
    if isinstance(value, list):
        return [scrub(v, tmp_path) for v in value]
    if isinstance(value, str):
        out = value.replace(str(tmp_path), PLACEHOLDER)
        out = _PYTEST_TEMP.sub(PLACEHOLDER, out)
        return out.replace(_HOME, "~") if _HOME not in ("", "/") else out
    return value


def leaks(value: Any) -> list[str]:
    """Every string in `value` (keys included) holding a pytest temp path or the home path."""
    if isinstance(value, dict):
        return [x for k, v in value.items() for x in leaks(k) + leaks(v)]
    if isinstance(value, list):
        return [x for v in value for x in leaks(v)]
    if isinstance(value, str) and ("/pytest-of-" in value or (_HOME not in ("", "/") and _HOME in value)):
        return [value]
    return []


@pytest.fixture(scope="module", autouse=True)
def _recording_file():
    """With `MISERVER_RECORD_ANSWERS` set, write the scrubbed answers once this module has run; refuse a leak."""
    RECORDED.clear()
    yield
    target = os.environ.get(RECORD_ENV)
    if not target or not RECORDED:
        return
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    doc = {"source": {"repo": "miServer", "commit": commit or None,
                      "recorded_by": "tests/unit/api/test_relayed_routes_answer.py"},
           "answers": dict(sorted(RECORDED.items()))}
    assert leaks(doc) == [], f"refusing to record a workstation path: {leaks(doc)[:3]}"
    Path(target).write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n", encoding="utf-8")


def _contract_keys() -> dict[tuple[str, str], tuple[str, str]]:
    contract = json.loads(ANSWERS.read_text(encoding="utf-8"))
    out: dict[tuple[str, str], tuple[str, str]] = {}
    for section in ("responses", "excluded"):
        for key in contract[section]:
            method, path = key.split(" ", 1)
            out[(method, normalise(path))] = (section, key)
    return out


def _doc(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


# ── the context every case receives ──────────────────────────────────────────────────────────────────


@dataclass
class Ctx:
    """`client` over the live app, `db` the session factory, the served tiny model `h`, and the test's tools."""

    h: Any
    monkeypatch: pytest.MonkeyPatch
    tmp_path: Path
    extra: dict[str, Any] = field(default_factory=dict)
    #: `(key, status, body)` of every validator check in this test; a case that seeds through its OWN route clears
    #: it (`fresh()`) before the call it returns.
    checked: list[tuple[str, str, Any]] = field(default_factory=list)

    def fresh(self) -> None:
        self.checked.clear()

    @property
    def client(self) -> Any:
        return self.h.client

    @property
    def app(self) -> Any:
        return self.h.client.app

    @property
    def db(self) -> Any:
        return self.h.factory


@pytest.fixture
async def ctx(monkeypatch, tmp_path):
    import sqlalchemy.ext.asyncio as sa_async

    from miserver.api.dependencies import get_model_downloader
    from miserver.core.config import settings
    from tests.support.capture_pre038_goldens import Harness

    monkeypatch.setattr(settings, "SAE_CACHE_DIR", str(tmp_path / "saes"))  # the real SAE service, no /app
    # The real model downloader is a process singleton built from MODEL_CACHE_DIR: rebuilt here over tmp_path
    # and dropped again afterwards, so no later test inherits this directory.
    monkeypatch.setattr(settings, "MODEL_CACHE_DIR", str(tmp_path / "models"))
    get_model_downloader.cache_clear()
    # ⚠ A FILE-backed SQLite database, not the Harness's `:memory:` one. An in-memory aiosqlite engine shares ONE
    # connection between every session, so a nested session's close rolls back the outer session's uncommitted
    # writes: the batch runner's `_finish_validation` inserted its rows, `_model_gate` opened and closed a second
    # session, and the rows vanished while the batch went `in_progress` (and then `completed` with 0 of 2 rows
    # run). PostgreSQL gives each session its own connection; a file database does too, as
    # `tests/unit/batch_fixtures.batch_db` says.
    real_engine = sa_async.create_async_engine

    def file_engine(url: str, **kw: Any) -> Any:
        from sqlalchemy import event

        if url != "sqlite+aiosqlite:///:memory:":
            return real_engine(url, **kw)
        engine = real_engine(f"sqlite+aiosqlite:///{tmp_path / 'miserver.db'}", **kw)

        @event.listens_for(engine.sync_engine, "connect")
        def _fast(dbapi_connection: Any, _record: Any) -> None:
            # No fsync per statement (a throwaway database); each connection still its own.
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA synchronous=OFF")
            cursor.execute("PRAGMA journal_mode=MEMORY")
            cursor.close()

        return engine

    with monkeypatch.context() as m:
        m.setattr(sa_async, "create_async_engine", file_engine)
        harness = Harness()
        h = await harness.__aenter__()
    assert h.engine.url.database == str(tmp_path / "miserver.db"), h.engine.url
    try:
        import miserver.api.routes.management.probes as probe_routes
        from miserver.services import probe_arm_bridge

        # The Harness serves the tiny model's identity through `probe_arm_bridge.loaded_identity`; the
        # single-probe routes bound the name at import, so they get the same one (a real `LoadedIdentity`).
        monkeypatch.setattr(probe_routes, "loaded_identity", probe_arm_bridge.loaded_identity)
        context = Ctx(h=h, monkeypatch=monkeypatch, tmp_path=tmp_path)
        yield context
        if "aclient" in context.extra:
            await context.extra["aclient"].aclose()
    finally:
        await harness.__aexit__(None, None, None)
        get_model_downloader.cache_clear()


Case = Callable[[Ctx], Awaitable[Any]]
CASES: dict[tuple[str, str], Case] = {}
#: Cases that fail for a defect in miServer (U5), not in their setup: `strict=True` xfails with the root cause.
DEFECTS: dict[tuple[str, str], str] = {}


def case(method: str, path: str) -> Callable[[Case], Case]:
    def register(fn: Case) -> Case:
        key = (method, normalise(path))
        assert key not in CASES, f"two cases for {key}"
        CASES[key] = fn
        return fn

    return register


# ── health, environment, approvals ───────────────────────────────────────────────────────────────────


@case("GET", "/api/health/detailed")
async def _(c: Ctx) -> Any:
    return c.client.get("/api/health/detailed")


@case("GET", "/api/health/inference")
async def _(c: Ctx) -> Any:
    return c.client.get("/api/health/inference")


@case("GET", "/api/environment")
async def _(c: Ctx) -> Any:
    return c.client.get("/api/environment")


@case("GET", "/api/environment/history")
async def _(c: Ctx) -> Any:
    r = c.client.put("/api/environment", json={"values": {"TELEMETRY_EXPORT_TIMEOUT_S": 7}, "reason": "relay"})
    assert r.status_code == 200, r.text
    return c.client.get("/api/environment/history")


@case("GET", "/api/approvals")
async def _(c: Ctx) -> Any:
    await _models(c)
    _filed(c)
    return c.client.get("/api/approvals")


@case("GET", "/api/approvals/actions")
async def _(c: Ctx) -> Any:
    return c.client.get("/api/approvals/actions")


# ── probes and behaviours (the tiny real Llama: real import, arming, parity, scoring, serving) ───────

_NAMES = iter(range(10**6))


def _probe_doc(key: str | None = None, *, layer: int = 1, threshold: float = 0.0, **over: Any) -> dict:
    """A definition for the tiny model whose test vector reproduces on it (parity passes for real)."""
    doc = bf.definition(f"relay-p{next(_NAMES)}", key=key, layer=layer, threshold=threshold,
                        scale=0.2 + 0.05 * layer)
    doc["provenance"] = {"probe_id": f"pm_relay{next(_NAMES)}", "run_id": "pmr_relay",
                         "built_at": "2026-11-25T00:00:00Z"}
    doc.update(over)
    return doc


def _import_probe(c: Ctx, doc: dict | None = None) -> dict:
    r = c.client.post("/api/probes/import", json=doc or _probe_doc())
    assert r.status_code == 200, r.text
    return r.json()["data"]


def _arm_probe(c: Ctx, probe_id: str) -> None:
    r = c.client.post(f"/api/probes/{probe_id}/arm", json={})
    assert r.status_code == 200, r.text


async def _behaviour(c: Ctx) -> tuple[str, list[Any], dict]:
    """A behaviour of two armed members, a vetoing rule set attached, and a verdict definition attached:
    each through its real route (`test_behaviour_verdict_serving.setup`)."""
    from tests.unit.api.test_behaviour_verdict_serving import setup

    return await setup(c.h)


async def _served(c: Ctx) -> tuple[str, Any]:
    """A behaviour served on one real chat request: probe events and a behaviour verdict are stored."""
    key, _members, _definition = await _behaviour(c)
    r = c.h.post_chat({**cap.BODY, "messages": [{"role": "user", "content": "nothing here w1 w2"}]}, {})
    assert r.status_code == 200, r.text
    return key, r


@case("POST", "/api/probes/import")
async def _(c: Ctx) -> Any:
    return c.client.post("/api/probes/import", json=_probe_doc(bf.random_key()))


@case("GET", "/api/probes")
async def _(c: Ctx) -> Any:
    _import_probe(c, _probe_doc(bf.random_key()))
    _arm_probe(c, _import_probe(c)["id"])
    return c.client.get("/api/probes")


@case("GET", "/api/probes/{probe_id}")
async def _(c: Ctx) -> Any:
    await _served(c)
    probe_id = c.client.get("/api/probes").json()["data"][0]["id"]
    return c.client.get(f"/api/probes/{probe_id}")


@case("DELETE", "/api/probes/{probe_id}")
async def _(c: Ctx) -> Any:
    return c.client.delete(f"/api/probes/{_import_probe(c)['id']}")


@case("POST", "/api/probes/{probe_id}/arm")
async def _(c: Ctx) -> Any:
    probe = _import_probe(c, _probe_doc(bf.random_key()))
    return c.client.post(f"/api/probes/{probe['id']}/arm", json={"windows": ["prompt", "response"]})


@case("POST", "/api/probes/{probe_id}/disarm")
async def _(c: Ctx) -> Any:
    probe = _import_probe(c)
    _arm_probe(c, probe["id"])
    return c.client.post(f"/api/probes/{probe['id']}/disarm")


@case("POST", "/api/probes/{probe_id}/parity")
async def _(c: Ctx) -> Any:
    return c.client.post(f"/api/probes/{_import_probe(c)['id']}/parity")


@case("POST", "/api/probes/{probe_id}/recalibrate")
async def _(c: Ctx) -> Any:
    doc = _probe_doc()
    probe = _import_probe(c, doc)
    _arm_probe(c, probe["id"])
    return c.client.post(f"/api/probes/{probe['id']}/recalibrate", json={
        "decision": {"threshold": 0.5, "target_fpr": 0.005, "realised_fpr": 0.005,
                     "threshold_source": "calibration_set"},
        "mistudio_probe_id": doc["provenance"]["probe_id"], "mistudio_run_id": "pmr_relay",
        "reason": "a re-cut on a fresh calibration set"})


@case("POST", "/api/probes/score")
async def _(c: Ctx) -> Any:
    key = bf.random_key()
    ids = [_import_probe(c, _probe_doc(key, layer=layer))["id"] for layer in (0, 1)]
    return c.client.post("/api/probes/score", json={"probe_ids": ids, "inputs": [
        {"token_ids": [0, 2, 6, 7, 8], "prompt_tokens": 3},
        {"messages": [{"role": "user", "content": "w1 w2"}]}]})


@case("GET", "/api/probes/status")
async def _(c: Ctx) -> Any:
    await _served(c)
    return c.client.get("/api/probes/status")


@case("GET", "/api/probes/events")
async def _(c: Ctx) -> Any:
    await _served(c)
    return c.client.get("/api/probes/events")


@case("GET", "/api/probes/events/{event_id}")
async def _(c: Ctx) -> Any:
    await _served(c)
    event_id = c.client.get("/api/probes/events").json()["data"][0]["id"]
    return c.client.get(f"/api/probes/events/{event_id}")


@case("DELETE", "/api/probes/events")
async def _(c: Ctx) -> Any:
    await _served(c)
    return c.client.delete("/api/probes/events")


@case("GET", "/api/probes/behaviours")
async def _(c: Ctx) -> Any:
    await _served(c)
    return c.client.get("/api/probes/behaviours")


@case("POST", "/api/probes/behaviours/{key}/arm")
async def _(c: Ctx) -> Any:
    key = bf.random_key()
    for layer in (0, 1):
        _import_probe(c, _probe_doc(key, layer=layer))
    return c.client.post(f"/api/probes/behaviours/{key}/arm", json={})


@case("POST", "/api/probes/behaviours/{key}/disarm")
async def _(c: Ctx) -> Any:
    key, _members, _definition = await _behaviour(c)
    return c.client.post(f"/api/probes/behaviours/{key}/disarm", json={})


@case("GET", "/api/probes/behaviours/{key}/rule-signals")
async def _(c: Ctx) -> Any:
    key, _response = await _served(c)
    return c.client.get(f"/api/probes/behaviours/{key}/rule-signals")


@case("PUT", "/api/probes/behaviours/{key}/rule-signals")
async def _(c: Ctx) -> Any:
    key, _members, _definition = await _behaviour(c)
    rule_set_id = c.client.get(f"/api/probes/behaviours/{key}/rule-signals").json()["data"]["rule_sets"][0]["id"]
    c.fresh()
    return c.client.put(f"/api/probes/behaviours/{key}/rule-signals",
                        json={"rule_set_ids": [rule_set_id], "reason": "re-attached"})


@case("DELETE", "/api/probes/behaviours/{key}/rule-signals")
async def _(c: Ctx) -> Any:
    key = bf.random_key()
    for layer in (0, 1):
        _import_probe(c, _probe_doc(key, layer=layer))
    r = c.client.post("/api/rule-sets/import", content=rf.raw(vf.masked_document(key)),
                      headers={"content-type": "application/json"})
    assert r.status_code == 200, r.text
    r = c.client.put(f"/api/probes/behaviours/{key}/rule-signals", json={"rule_set_ids": [r.json()["data"]["id"]]})
    assert r.status_code == 200, r.text
    return c.client.delete(f"/api/probes/behaviours/{key}/rule-signals", params={"reason": "detached"})


@case("GET", "/api/probes/behaviours/{key}/verdict")
async def _(c: Ctx) -> Any:
    key, _response = await _served(c)
    return c.client.get(f"/api/probes/behaviours/{key}/verdict")


@case("PUT", "/api/probes/behaviours/{key}/verdict")
async def _(c: Ctx) -> Any:
    key, _members, definition = await _behaviour(c)
    c.fresh()
    return c.client.put(f"/api/probes/behaviours/{key}/verdict",
                        json={"definition": definition, "reason": "re-attached"})


@case("DELETE", "/api/probes/behaviours/{key}/verdict")
async def _(c: Ctx) -> Any:
    key, _members, _definition = await _behaviour(c)
    return c.client.delete(f"/api/probes/behaviours/{key}/verdict", params={"reason": "detached"})


# ── rule sets ────────────────────────────────────────────────────────────────────────────────────────


@case("GET", "/api/rule-sets")
async def _(c: Ctx) -> Any:
    _import_rule_set(c)
    return c.client.get("/api/rule-sets")


@case("GET", "/api/rule-sets/validators")
async def _(c: Ctx) -> Any:
    return c.client.get("/api/rule-sets/validators")


# ── the Hugging Face hub (the one network boundary of the hub routes) ─────────────────────────────────


class FakeHub:
    """`HfApi` and `hf_hub_download` as `cluster_hub_service` calls them, over files held here. `list_models`
    answers real `huggingface_hub.hf_api.ModelInfo` objects; `hf_hub_download` writes a real file."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.repos: dict[str, dict[str, bytes]] = {}
        self.tags: dict[str, list[str]] = {}

    def add(self, repo_id: str, tag: str, files: dict[str, Any]) -> None:
        self.repos[repo_id] = {name: (body if isinstance(body, bytes) else json.dumps(body).encode())
                               for name, body in files.items()}
        self.tags[repo_id] = [tag, "base_model:tiny/llama"]

    def api(self) -> Any:
        from huggingface_hub.hf_api import ModelInfo

        hub = self

        class Api:
            def list_models(self, *, filter: list[str], search: Any = None, limit: int = 30, token: Any = None):
                return [ModelInfo(id=repo, likes=3, downloads=41, lastModified="2026-11-01T00:00:00.000Z",
                                  tags=tags)
                        for repo, tags in hub.tags.items() if filter[0] in tags][:limit]

            def list_repo_files(self, repo_id: str, *, revision: Any = None, token: Any = None):
                return sorted(hub.repos[repo_id])

        return Api

    def download(self, *, repo_id: str, filename: str, revision: Any, cache_dir: str, token: Any) -> str:
        path = self.root / repo_id.replace("/", "--") / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(self.repos[repo_id][filename])
        return str(path)


def _hub(c: Ctx) -> FakeHub:
    from miserver.api import dependencies
    from miserver.services import cluster_hub_service

    if "hub" not in c.extra:
        hub = FakeHub(c.tmp_path / "hub")
        c.monkeypatch.setattr(cluster_hub_service, "HfApi", hub.api())
        c.monkeypatch.setattr(cluster_hub_service, "hf_hub_download", hub.download)
        # Fresh process-wide hub services: no listing cached by another test.
        c.monkeypatch.setattr(dependencies, "_probe_hub_service", None)
        c.monkeypatch.setattr(dependencies, "_cluster_hub_service", None)
        c.extra["hub"] = hub
    return c.extra["hub"]


@case("GET", "/api/probes/hub/search")
async def _(c: Ctx) -> Any:
    _hub(c).add("relay/probes", "mistudio-probe-definition", {"a.probe.json": _probe_doc()})
    return c.client.get("/api/probes/hub/search", params={"q": "relay"})


@case("GET", "/api/probes/hub/{repo_id}/definitions")
async def _(c: Ctx) -> Any:
    _hub(c).add("relay/probes", "mistudio-probe-definition", {
        "a.probe.json": _probe_doc(), "manifest.jsonl": b'{"file": "a.probe.json", "name": "a"}\n'})
    return c.client.get("/api/probes/hub/relay/probes/definitions")


@case("POST", "/api/probes/hub/import")
async def _(c: Ctx) -> Any:
    _hub(c).add("relay/probes", "mistudio-probe-definition", {"a.probe.json": _probe_doc(bf.random_key())})
    return c.client.post("/api/probes/hub/import", json={"repo_id": "relay/probes", "filename": "a.probe.json"})


# ── steering: SAEs, clusters, profiles, circuits, sensing ────────────────────────────────────────────

SAE_A, SAE_B = "sae_alpha", "sae_beta"


def _saes(c: Ctx) -> None:
    """Two REAL `LoadedSAE`s hooked into the served tiny model at layers 0 and 1 (`f28_fixtures.attach`), and
    fresh process-wide sensing services."""
    from miserver.api import dependencies
    from tests.unit.f28_fixtures import attach

    if "saes" not in c.extra:
        c.monkeypatch.setattr(dependencies, "_sensing_service", None)
        c.monkeypatch.setattr(dependencies, "_circuit_sensing_service", None)
        c.extra["saes"] = attach(c.h.model, [(SAE_A, 0, 11), (SAE_B, 1, 12)])


def _cluster_doc(**over: Any) -> dict:
    """The committed miWorkbench export, re-pointed at the attached SAE at layer 0."""
    doc = _doc("miworkbench_export.cluster.json")
    doc["name"] = f"relay cluster {next(_NAMES)}"
    doc["sae"] = {**doc["sae"], "mistudio_sae_id": SAE_A, "layer": 0, "n_features": 8, "d_model": 16}
    doc["members"] = [{**m, "feature_idx": i, "max_activation": 0.05} for i, m in enumerate(doc["members"][:4])]
    doc.update(over)
    return doc


def _import_cluster(c: Ctx, *, activate: bool = False) -> str:
    _saes(c)
    r = c.client.post("/api/clusters/import", json=_cluster_doc(),
                      params={"activate": str(activate).lower()})
    assert r.status_code == 200 and r.json()["success"], r.text
    return r.json()["data"]["results"][0]["profile_id"]


@case("POST", "/api/clusters/import")
async def _(c: Ctx) -> Any:
    _saes(c)
    return c.client.post("/api/clusters/import", json=_cluster_doc())


@case("GET", "/api/clusters")
async def _(c: Ctx) -> Any:
    _import_cluster(c, activate=True)
    _import_cluster(c)
    return c.client.get("/api/clusters")


@case("GET", "/api/clusters/{cluster_id}/export")
async def _(c: Ctx) -> Any:
    return c.client.get(f"/api/clusters/{_import_cluster(c)}/export")


@case("POST", "/api/clusters/{cluster_id}/activate")
async def _(c: Ctx) -> Any:
    return c.client.post(f"/api/clusters/{_import_cluster(c)}/activate")


@case("POST", "/api/clusters/{cluster_id}/deactivate")
async def _(c: Ctx) -> Any:
    return c.client.post(f"/api/clusters/{_import_cluster(c, activate=True)}/deactivate")


@case("PUT", "/api/clusters/active/intensity")
async def _(c: Ctx) -> Any:
    _import_cluster(c, activate=True)
    return c.client.put("/api/clusters/active/intensity", json={"intensity": 0.5})


@case("GET", "/api/clusters/hub/search")
async def _(c: Ctx) -> Any:
    _hub(c).add("relay/clusters", "mistudio-cluster-definition", {"a.cluster.json": _cluster_doc()})
    return c.client.get("/api/clusters/hub/search")


@case("GET", "/api/clusters/hub/{repo_id}/definitions")
async def _(c: Ctx) -> Any:
    _hub(c).add("relay/clusters", "mistudio-cluster-definition", {
        "a.cluster.json": _cluster_doc(),
        "manifest.jsonl": b'{"file": "a.cluster.json", "name": "a", "member_count": 4, "base_model": "tiny/llama"}\n'})
    return c.client.get("/api/clusters/hub/relay/clusters/definitions", params={"revision": "main"})


@case("POST", "/api/clusters/hub/import")
async def _(c: Ctx) -> Any:
    _saes(c)
    _hub(c).add("relay/clusters", "mistudio-cluster-definition", {"a.cluster.json": _cluster_doc()})
    return c.client.post("/api/clusters/hub/import", json={"repo_id": "relay/clusters", "filename": "a.cluster.json"})


@case("GET", "/api/profiles")
async def _(c: Ctx) -> Any:
    _import_cluster(c, activate=True)
    return c.client.get("/api/profiles")


@case("POST", "/api/profiles/{profile_id}/activate")
async def _(c: Ctx) -> Any:
    return c.client.post(f"/api/profiles/{_import_cluster(c)}/activate", json={})


@case("POST", "/api/profiles/{profile_id}/deactivate")
async def _(c: Ctx) -> Any:
    return c.client.post(f"/api/profiles/{_import_cluster(c, activate=True)}/deactivate")


def _sensing(c: Ctx) -> str:
    """An active cluster with sensing enabled and armed on its own SAE (a quorum of one member), served on one
    chat request."""
    profile_id = _import_cluster(c, activate=True)
    r = c.client.post(f"/api/sensing/{profile_id}/enable")
    assert r.status_code == 200 and r.json()["data"]["armed"], r.text
    r = c.client.put(f"/api/sensing/{profile_id}/config", json={"min_k": 1})
    assert r.status_code == 200 and r.json()["data"]["armed"], r.text
    r = c.h.post_chat({**cap.BODY, "max_tokens": 4}, {})
    assert r.status_code == 200, r.text
    return profile_id


@case("GET", "/api/sensing/status")
async def _(c: Ctx) -> Any:
    _sensing(c)
    return c.client.get("/api/sensing/status")


@case("GET", "/api/sensing/events")
async def _(c: Ctx) -> Any:
    _sensing(c)
    return c.client.get("/api/sensing/events")


@case("GET", "/api/sensing/events/{event_id}")
async def _(c: Ctx) -> Any:
    _sensing(c)
    events = c.client.get("/api/sensing/events").json()["data"]["events"]
    assert events, "the served request recorded no co-activation event"
    return c.client.get(f"/api/sensing/events/{events[0]['id']}")


@case("DELETE", "/api/sensing/events")
async def _(c: Ctx) -> Any:
    _sensing(c)
    return c.client.delete("/api/sensing/events")


@case("POST", "/api/sensing/{profile_id}/enable")
async def _(c: Ctx) -> Any:
    return c.client.post(f"/api/sensing/{_import_cluster(c, activate=True)}/enable")


@case("POST", "/api/sensing/{profile_id}/disable")
async def _(c: Ctx) -> Any:
    return c.client.post(f"/api/sensing/{_sensing(c)}/disable")


@case("PUT", "/api/sensing/{profile_id}/config")
async def _(c: Ctx) -> Any:
    profile_id = _sensing(c)
    c.fresh()
    return c.client.put(f"/api/sensing/{profile_id}/config", json={"min_k": 2})


def _circuit_doc() -> dict:
    """The committed circuit definition re-pointed at the two attached SAEs
    (`test_circuit_applied_intensity_header.calibrated_doc`)."""
    from tests.unit.api.test_circuit_applied_intensity_header import calibrated_doc

    doc = calibrated_doc()
    doc["name"] = f"relay circuit {next(_NAMES)}"
    for member in doc["members"]:
        member["feature"]["max_activation"] = 0.05  # a usable edge threshold, so edge sensing arms
    return doc


def _import_circuit(c: Ctx, *, activate: bool = False) -> str:
    _saes(c)
    r = c.client.post("/api/circuits/import", json=_circuit_doc())
    assert r.status_code == 200 and r.json()["success"], r.text
    circuit_id = r.json()["data"]["id"]
    if activate:
        r = c.client.post(f"/api/circuits/{circuit_id}/activate")
        assert r.status_code == 200 and r.json()["success"], r.text
    return circuit_id


@case("POST", "/api/circuits/import")
async def _(c: Ctx) -> Any:
    _saes(c)
    return c.client.post("/api/circuits/import", json=_circuit_doc())


@case("GET", "/api/circuits")
async def _(c: Ctx) -> Any:
    _import_circuit(c, activate=True)
    _import_circuit(c)
    return c.client.get("/api/circuits")


@case("GET", "/api/circuits/active")
async def _(c: Ctx) -> Any:
    _import_circuit(c, activate=True)
    return c.client.get("/api/circuits/active")


@case("GET", "/api/circuits/claims")
async def _(c: Ctx) -> Any:
    _import_circuit(c, activate=True)
    return c.client.get("/api/circuits/claims")


@case("POST", "/api/circuits/claims/release")
async def _(c: Ctx) -> Any:
    return c.client.post("/api/circuits/claims/release", params={"circuit_id": _import_circuit(c, activate=True)})


@case("GET", "/api/circuits/{circuit_id}/export")
async def _(c: Ctx) -> Any:
    return c.client.get(f"/api/circuits/{_import_circuit(c)}/export")


@case("DELETE", "/api/circuits/{circuit_id}")
async def _(c: Ctx) -> Any:
    return c.client.delete(f"/api/circuits/{_import_circuit(c)}")


@case("POST", "/api/circuits/{circuit_id}/activate")
async def _(c: Ctx) -> Any:
    return c.client.post(f"/api/circuits/{_import_circuit(c)}/activate")


@case("POST", "/api/circuits/{circuit_id}/deactivate")
async def _(c: Ctx) -> Any:
    return c.client.post(f"/api/circuits/{_import_circuit(c, activate=True)}/deactivate")


@case("PUT", "/api/circuits/active/intensity")
async def _(c: Ctx) -> Any:
    _import_circuit(c, activate=True)
    return c.client.put("/api/circuits/active/intensity", json={"intensity": 0.6})


def _circuit_sensing(c: Ctx) -> str:
    """An active circuit with edge sensing enabled and armed, served on one chat request."""
    circuit_id = _import_circuit(c, activate=True)
    r = c.client.post(f"/api/circuit-sensing/{circuit_id}/enable")
    assert r.status_code == 200 and r.json()["data"]["armed"], r.text
    r = c.h.post_chat({**cap.BODY, "max_tokens": 4}, {})
    assert r.status_code == 200, r.text
    return circuit_id


@case("GET", "/api/circuit-sensing/status")
async def _(c: Ctx) -> Any:
    _circuit_sensing(c)
    return c.client.get("/api/circuit-sensing/status")


@case("GET", "/api/circuit-sensing/events")
async def _(c: Ctx) -> Any:
    _circuit_sensing(c)
    return c.client.get("/api/circuit-sensing/events")


@case("GET", "/api/circuit-sensing/events/{event_id}")
async def _(c: Ctx) -> Any:
    _circuit_sensing(c)
    events = c.client.get("/api/circuit-sensing/events").json()["data"]["events"]
    assert events, "the served request recorded no edge event"
    return c.client.get(f"/api/circuit-sensing/events/{events[0]['id']}")


@case("DELETE", "/api/circuit-sensing/events")
async def _(c: Ctx) -> Any:
    _circuit_sensing(c)
    return c.client.delete("/api/circuit-sensing/events")


@case("POST", "/api/circuit-sensing/{circuit_id}/enable")
async def _(c: Ctx) -> Any:
    return c.client.post(f"/api/circuit-sensing/{_import_circuit(c, activate=True)}/enable")


@case("POST", "/api/circuit-sensing/{circuit_id}/disable")
async def _(c: Ctx) -> Any:
    return c.client.post(f"/api/circuit-sensing/{_circuit_sensing(c)}/disable")


# ── models (a REAL `ModelService` over the test database; the GPU load, nvidia-smi and the Hub mocked) ──

AGENT = {"X-miLLM-Agent": "agent:relay-test"}
LEASE = "X-miLLM-Lease"


async def _models(c: Ctx) -> tuple[Any, Any]:
    """Rows 1 ("tiny", LOADED: the served tiny model, resident as model 1) and 2 (READY), served by the REAL
    model service. Mocked: the background load worker (it loads weights onto a GPU), the placement pre-check
    (it reads nvidia-smi) and the capacity timer (a wall clock)."""
    from miserver.api.dependencies import get_model_service
    from miserver.db.models.model import ModelStatus
    from miserver.services import probe_capacity_measure
    from miserver.services.model_service import ModelService
    from tests.unit.approval_fixtures import make_model

    if "models" not in c.extra:
        c.app.dependency_overrides.pop(get_model_service, None)
        c.monkeypatch.setattr(ModelService, "_precheck_placement", lambda self, model, wanted: None)
        c.monkeypatch.setattr(ModelService, "_load_worker", lambda self, *a, **k: None)
        c.monkeypatch.setattr(probe_capacity_measure, "_clock", lambda fn, devices: (fn(), 3.0)[1])
        resident = await make_model(c.db, name="tiny")
        other = await make_model(c.db, name="LFM2.5-1.2B-Instruct")
        assert resident.id == 1, resident.id  # the id `f25_fixtures.make_service` made resident
        async with c.db() as session:
            row = await session.get(type(resident), 1)
            row.status = ModelStatus.LOADED
            await session.commit()
        c.extra["models"] = (resident, other)
    return c.extra["models"]


def _lease(c: Ctx) -> str:
    r = c.client.post("/api/models/1/lease", json={"holder": "midataworks", "reason": "label run 7",
                                                   "ttl_seconds": 600})
    assert r.status_code == 201, r.text
    return r.json()["data"]["lease_id"]


@case("GET", "/api/models")
async def _(c: Ctx) -> Any:
    await _models(c)
    return c.client.get("/api/models")


@case("GET", "/api/models/{model_id}")
async def _(c: Ctx) -> Any:
    await _models(c)
    _lease(c)
    return c.client.get("/api/models/1")


def _hub_model_info(c: Ctx) -> None:
    """The Hub's model-info answer, as `huggingface_hub` parses it: a real `ModelInfo` built from the keys
    the Hub's `/api/models/{repo}` JSON carries. It keeps the legacy `modelId` key the full answer still sends; the
    downloader reads the declared `id` (`test_model_info_reads_the_declared_id.py` builds one without `modelId`)."""
    from huggingface_hub.hf_api import ModelInfo

    from miserver.api.dependencies import get_model_downloader

    def model_info(repo_id: str, **_kw: Any) -> Any:
        return ModelInfo(id=repo_id, modelId=repo_id, sha="0" * 40, gated=False,
                         pipeline_tag="text-generation", library_name="transformers",
                         downloads=10, likes=2, tags=["text-generation", "safetensors"],
                         config={"model_type": "llama", "architectures": ["LlamaForCausalLM"]},
                         cardData={"license": "apache-2.0", "language": ["en"]},
                         safetensors={"parameters": {"BF16": 1_200_000_000}, "total": 1_200_000_000},
                         siblings=[{"rfilename": "model.safetensors", "size": 2_400_000_000},
                                   {"rfilename": "config.json", "size": 700}])

    c.monkeypatch.setattr(get_model_downloader().hf_api, "model_info", model_info)


@case("POST", "/api/models")
async def _(c: Ctx) -> Any:
    from miserver.services.model_service import ModelService

    await _models(c)
    _hub_model_info(c)
    # The download itself (network) never starts: the row is registered DOWNLOADING, as the route answers.
    c.monkeypatch.setattr(ModelService, "_download_worker", lambda self, *a, **k: None)
    return c.client.post("/api/models", json={"source": "huggingface", "repo_id": "relay/tiny-llama",
                                              "quantization": "FP16"})


@case("POST", "/api/models/preview")
async def _(c: Ctx) -> Any:
    await _models(c)
    _hub_model_info(c)
    return c.client.post("/api/models/preview", json={"repo_id": "relay/tiny-llama"})


@case("POST", "/api/models/{model_id}/cancel")
async def _(c: Ctx) -> Any:
    from miserver.db.models.model import ModelStatus

    _resident, other = await _models(c)
    async with c.db() as session:
        row = await session.get(type(other), other.id)
        row.status = ModelStatus.DOWNLOADING
        await session.commit()
    return c.client.post(f"/api/models/{other.id}/cancel")


@case("POST", "/api/models/{model_id}/load")
async def _(c: Ctx) -> Any:
    _resident, other = await _models(c)
    return c.client.post(f"/api/models/{other.id}/load", json={"gpu": "auto"})


@case("POST", "/api/models/{model_id}/unload")
async def _(c: Ctx) -> Any:
    await _models(c)
    return c.client.post("/api/models/1/unload")


@case("DELETE", "/api/models/{model_id}")
async def _(c: Ctx) -> Any:
    _resident, other = await _models(c)
    return c.client.delete(f"/api/models/{other.id}")


@case("PUT", "/api/models/{model_id}/context-ceiling")
async def _(c: Ctx) -> Any:
    _resident, other = await _models(c)
    return c.client.put(f"/api/models/{other.id}/context-ceiling", json={"context_ceiling": None})


@case("POST", "/api/models/{model_id}/lease")
async def _(c: Ctx) -> Any:
    await _models(c)
    return c.client.post("/api/models/1/lease", json={"holder": "midataworks", "reason": "label run 7",
                                                      "ttl_seconds": 600})


@case("GET", "/api/models/{model_id}/lease")
async def _(c: Ctx) -> Any:
    await _models(c)
    lease_id = _lease(c)
    assert c.client.delete("/api/models/1/lease", headers={LEASE: lease_id}).status_code == 200
    _lease(c)
    return c.client.get("/api/models/1/lease")


@case("POST", "/api/models/{model_id}/lease/renew")
async def _(c: Ctx) -> Any:
    await _models(c)
    return c.client.post("/api/models/1/lease/renew", headers={LEASE: _lease(c)}, json={"ttl_seconds": 900})


@case("DELETE", "/api/models/{model_id}/lease")
async def _(c: Ctx) -> Any:
    await _models(c)
    return c.client.delete("/api/models/1/lease", headers={LEASE: _lease(c)})


@case("POST", "/api/models/{model_id}/probe-capacity/measure")
async def _(c: Ctx) -> Any:
    await _models(c)
    return c.client.post("/api/models/1/probe-capacity/measure")


@case("GET", "/api/models/{model_id}/probe-capacity")
async def _(c: Ctx) -> Any:
    await _models(c)
    assert c.client.post("/api/models/1/probe-capacity/measure").status_code == 200
    return c.client.get("/api/models/1/probe-capacity")


@case("GET", "/api/models/{model_id}/probe-limits/recommendation")
async def _(c: Ctx) -> Any:
    await _models(c)
    assert c.client.post("/api/models/1/probe-capacity/measure").status_code == 200
    return c.client.get("/api/models/1/probe-limits/recommendation")


@case("PUT", "/api/models/{model_id}/probe-limits")
async def _(c: Ctx) -> Any:
    await _models(c)
    assert c.client.post("/api/models/1/probe-capacity/measure").status_code == 200
    return c.client.put("/api/models/1/probe-limits", json={
        "max_armed": 12, "decode_budget": {"value": 2, "unit": "pct"},
        "prefill_budget": {"value": 10, "unit": "pct"}, "reason": "the tiny model is small"})


# ── approvals and environment ────────────────────────────────────────────────────────────────────────


def _filed(c: Ctx) -> str:
    """An approval an agent filed (its probe-limit write waits for the operator)."""
    r = c.client.put("/api/models/2/probe-limits", json={"max_armed": 4, "reason": "agent asks"},
                     headers=AGENT)
    assert r.status_code == 202, r.text
    return r.json()["data"]["approval_id"]


@case("GET", "/api/approvals/{approval_id}")
async def _(c: Ctx) -> Any:
    await _models(c)
    return c.client.get(f"/api/approvals/{_filed(c)}")


@case("PUT", "/api/environment")
async def _(c: Ctx) -> Any:
    return c.client.put("/api/environment", json={"values": {"TELEMETRY_EXPORT_TIMEOUT_S": 7},
                                                  "reason": "a shorter export timeout"})


# ── rule sets ────────────────────────────────────────────────────────────────────────────────────────


def _import_rule_set(c: Ctx, key: str | None = None) -> dict:
    r = c.client.post("/api/rule-sets/import", content=rf.raw(vf.masked_document(key or bf.random_key())),
                      headers={"content-type": "application/json"})
    assert r.status_code == 200, r.text
    return r.json()["data"]


@case("POST", "/api/rule-sets/import")
async def _(c: Ctx) -> Any:
    return c.client.post("/api/rule-sets/import", content=rf.raw(vf.masked_document(bf.random_key())),
                         headers={"content-type": "application/json"})


@case("POST", "/api/rule-sets/validate")
async def _(c: Ctx) -> Any:
    return c.client.post("/api/rule-sets/validate", content=rf.raw(vf.masked_document(bf.random_key())),
                         headers={"content-type": "application/json"})


@case("GET", "/api/rule-sets/{rule_set_id}")
async def _(c: Ctx) -> Any:
    return c.client.get(f"/api/rule-sets/{_import_rule_set(c)['id']}")


@case("DELETE", "/api/rule-sets/{rule_set_id}")
async def _(c: Ctx) -> Any:
    return c.client.delete(f"/api/rule-sets/{_import_rule_set(c)['id']}")


@case("POST", "/api/rule-sets/{rule_set_id}/evaluate")
async def _(c: Ctx) -> Any:
    return c.client.post(f"/api/rule-sets/{_import_rule_set(c)['id']}/evaluate",
                         json={"text": "marker ZQ-1234 and ZQ-9999", "return_rule_positions": True})


# ── the OpenAI surface (the served tiny model) and the Batch API (a real runner) ──────────────────────


@case("POST", "/v1/chat/completions")
async def _(c: Ctx) -> Any:
    await _behaviour(c)  # served with probe and behaviour results on the answer
    _import_cluster(c, activate=True)  # and steered
    return c.h.post_chat({**cap.BODY, "messages": [{"role": "user", "content": "marker ZQ-1234 w1 w2"}]}, {})


@case("POST", "/v1/completions")
async def _(c: Ctx) -> Any:
    await _behaviour(c)
    return c.client.post("/v1/completions", json={"model": "tiny", "prompt": "w1 w2 w3", "max_tokens": 3,
                                                  "temperature": 0.0})


@case("POST", "/v1/embeddings")
async def _(c: Ctx) -> Any:
    return c.client.post("/v1/embeddings", json={"model": "tiny", "input": ["w1 w2", "w3"]})


async def _batches(c: Ctx) -> Any:
    """A REAL `BatchRunner` over the test database and the served tiny model (`batch_fixtures.BatchHarness`),
    and an in-loop HTTP client over the live app. The batch routes start background work on the running
    loop (`submit_validation`), so they are called through `httpx.ASGITransport` in THIS loop, as the batch
    tests do: a `TestClient` request runs in a loop that ends with the request, which would drop that work."""
    from miserver.api.routes.openai.batches import runner_dependency
    from miserver.core.config import settings
    from miserver.db.repositories.model_repository import ModelRepository
    from miserver.ml.model_loader import ModelLoader
    from miserver.services.batch.runner import BatchRunner
    from miserver.services.model_service import ModelService
    from tests.unit.batch_fixtures import SpyEmitter, client_for

    if "runner" not in c.extra:
        await _models(c)
        c.monkeypatch.setattr(settings, "BATCH_FILES_DIR", str(c.tmp_path / "batch_files"))
        loader = ModelLoader()

        def model_service(session: Any) -> Any:
            return ModelService(repository=ModelRepository(session), downloader=None, loader=loader,
                                emitter=None, inference_service=c.h.svc)

        runner = BatchRunner(c.db, inference_provider=lambda: c.h.svc, model_service_factory=model_service,
                             emitter=SpyEmitter())
        c.app.dependency_overrides[runner_dependency] = lambda: runner
        c.extra["runner"] = runner
        c.extra["aclient"] = client_for(c.app)
    return c.extra["runner"], c.extra["aclient"]


async def _upload(c: Ctx) -> str:
    from tests.unit.batch_fixtures import completion_line, jsonl, upload

    _runner, client = await _batches(c)
    r = await upload(client, jsonl([completion_line("c0"), completion_line("c1", "w3 w4")]))
    assert r.status_code == 200, r.text
    return r.json()["id"]


async def _batch(c: Ctx, *, run: bool = False) -> str:
    runner, client = await _batches(c)
    r = await client.post("/v1/batches", json={"input_file_id": await _upload(c), "endpoint": "/v1/completions",
                                              "completion_window": "24h", "metadata": {"job": "relay"}})
    assert r.status_code == 200, r.text
    while runner._validations:
        await asyncio.gather(*list(runner._validations))
    if run:
        for _ in range(100):
            if not await runner.step():
                break
    return r.json()["id"]


@case("POST", "/v1/files")
async def _(c: Ctx) -> Any:
    from tests.unit.batch_fixtures import completion_line, jsonl, upload

    _runner, client = await _batches(c)
    return await upload(client, jsonl([completion_line("c0")]))


@case("GET", "/v1/files/{file_id}/content")
async def _(c: Ctx) -> Any:
    _runner, client = await _batches(c)
    return await client.get(f"/v1/files/{await _upload(c)}/content")


@case("POST", "/v1/batches")
async def _(c: Ctx) -> Any:
    _runner, client = await _batches(c)
    file_id = await _upload(c)
    response = await client.post("/v1/batches", json={"input_file_id": file_id, "endpoint": "/v1/completions",
                                                     "completion_window": "24h"})
    runner = c.extra["runner"]
    while runner._validations:
        await asyncio.gather(*list(runner._validations))
    return response


@case("GET", "/v1/batches/{batch_id}")
async def _(c: Ctx) -> Any:
    batch_id = await _batch(c, run=True)
    _runner, client = await _batches(c)
    response = await client.get(f"/v1/batches/{batch_id}")
    # The batch really ran: both rows answered by the tiny model, an output file written.
    body = response.json()
    assert (body["status"], body["request_counts"]) == ("completed", {"total": 2, "completed": 2, "failed": 0}), body
    assert body["output_file_id"], body
    return response


@case("POST", "/v1/batches/{batch_id}/cancel")
async def _(c: Ctx) -> Any:
    batch_id = await _batch(c)
    _runner, client = await _batches(c)
    return await client.post(f"/v1/batches/{batch_id}/cancel")


@case("POST", "/v1/batches/{batch_id}/lease")
async def _(c: Ctx) -> Any:
    batch_id = await _batch(c)
    _runner, client = await _batches(c)
    return await client.post(f"/v1/batches/{batch_id}/lease", headers={LEASE: _lease(c)})


# ── the tests ────────────────────────────────────────────────────────────────────────────────────────


def test_the_cases_are_exactly_the_section_4_endpoints():
    wanted = endpoints()
    assert set(CASES) == wanted, {
        "a §4 endpoint without a case": sorted(wanted - set(CASES)),
        "a case §4 does not name": sorted(set(CASES) - wanted),
    }
    assert set(DEFECTS) <= set(CASES), sorted(set(DEFECTS) - set(CASES))


def _params() -> list[Any]:
    out = []
    for endpoint in sorted(endpoints()):
        marks = []
        if endpoint in DEFECTS:
            marks.append(pytest.mark.xfail(strict=True, reason=DEFECTS[endpoint]))
        out.append(pytest.param(endpoint, id=f"{endpoint[0]} {endpoint[1]}", marks=marks))
    return out


@pytest.mark.parametrize("endpoint", _params())
async def test_each_relayed_route_answers(endpoint, ctx, monkeypatch):
    assert endpoint in CASES, f"§4 names {endpoint} and this module has no case for it"
    section, key = _contract_keys()[endpoint]
    checked = ctx.checked
    real_check = answer_validator.check_body

    def recording(key_: str, status: int | str, body: Any) -> None:
        checked.append((key_, str(status), body))
        real_check(key_, status, body)

    monkeypatch.setattr(answer_validator, "check_body", recording)
    response = await CASES[endpoint](ctx)
    assert 200 <= response.status_code < 300, (endpoint, response.status_code, response.text[:2000])
    assert answer_validator.FAILURES == [], answer_validator.FAILURES
    media = (response.headers.get("content-type") or "").split(";")[0].strip()
    if section == "excluded":
        assert media == EXCLUDED_MEDIA[endpoint], (endpoint, media)
        assert [k for k, _, _ in checked if k == key] == []
        return
    assert media == "application/json", (endpoint, media)
    mine = [(s, body) for k, s, body in checked if k == key]
    assert mine == [(str(response.status_code), response.json())], {
        "endpoint": endpoint, "key": key, "checked": [(k, s) for k, s, _ in checked]}
    # The recording: every answer scrubbed, every run (not only when recording), and refused if a path survives.
    recorded = scrub(response.json(), ctx.tmp_path)
    assert leaks(recorded) == [], f"{endpoint}: a workstation path would be recorded: {leaks(recorded)[:3]}"
    RECORDED[key] = {"status": str(response.status_code), "body": recorded}


def test_the_scrub_replaces_temp_and_home_paths(tmp_path):
    planted = {"cache": f"{tmp_path}/saes/x", "other": "/tmp/pytest-of-someone/pytest-12/test_x0/models",
               "home": f"{_HOME}/app/thing", "list": [f"{tmp_path}"], f"{tmp_path}/key": 1, "plain": "/data/models"}
    assert leaks(planted), "fixture: the planted values must hold paths"
    clean = scrub(planted, tmp_path)
    assert leaks(clean) == []
    assert clean["cache"] == f"{PLACEHOLDER}/saes/x" and clean["other"] == f"{PLACEHOLDER}/models"
    assert clean["home"] == "~/app/thing" and clean["plain"] == "/data/models"

