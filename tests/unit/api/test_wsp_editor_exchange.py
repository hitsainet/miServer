"""miServer as the WSP editor endpoint (043 FR-043.10 to FR-043.13; D43-7, D43-8, D43-9, P47).

miForge's gateway forwards a `weightspace` editor session to `/v1/chat/completions` with two
headers (`X-miLLM-Origin: judge`, `X-miLLM-Load-Policy: refuse`) and one body field
(`steering: {"features": []}`). The session's requests are in `tests/fixtures/wsp_editor_exchange.json`,
written by hand from the library's 0.3.0 source (never installed or imported here).

Every case runs the REAL route over a REAL chat template (`tool_call_fixtures`), one per call
format (LFM2, Hermes, Llama 3.1 JSON), with the stand-in model's generation scripted to emit the
fixture's expected call. So the parse, the `tool_calls` shape, the provenance headers and the
field policy are the production code's.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest
import torch

from miserver.api.request_policy import CHAT, OUTPUT_CHANGING, TF, Refused
from tests.unit.f25_fixtures import clear_loaded, make_client, make_service, model_row
from tests.unit.api.test_circuit_applied_intensity_header import (  # noqa: F401 - fixture
    calibrated_doc,
    served,
)
from tests.unit.f28_fixtures import (  # noqa: F401 - clean_state and db are fixtures
    SAE_A,
    add_profile,
    attach,
    clean_state,
    db,
    no_cuda,
)
from tests.unit.tool_call_fixtures import family_tokenizer, scripted_model

URL = "/v1/chat/completions"
FIXTURE = json.loads(
    (Path(__file__).resolve().parents[2] / "fixtures" / "wsp_editor_exchange.json").read_text()
)
FORMATS = ("lfm2", "hermes", "llama3_json")
NAME = {"lfm2": "LFM2.5-1.2B-Instruct", "hermes": "Qwen2.5-7B-Instruct",
        "llama3_json": "Llama-3.1-8B-Instruct"}
CALL_ID = re.compile(r"^call_[0-9a-f]{24}$")


@pytest.fixture(autouse=True)
def _clean():
    clear_loaded()
    yield
    clear_loaded()


def reply_for(fmt: str, call: dict[str, Any]) -> str:
    """The model's raw text for one call, in the format's own syntax."""
    name, args = call["name"], call["arguments"]
    if fmt == "lfm2":
        inner = ", ".join(f"{k}={v!r}" for k, v in args.items())
        return f"<|tool_call_start|>[{name}({inner})]<|tool_call_end|>"
    if fmt == "hermes":
        return f'<tool_call>\n{json.dumps({"name": name, "arguments": args})}\n</tool_call>'
    return json.dumps({"name": name, "parameters": args})


def client_for(fmt: str, reply: str):
    tok = family_tokenizer(fmt)
    model = scripted_model(tok, reply)
    client, svc = make_client(make_service(model, tok, name=NAME[fmt]),
                              model_row(name=NAME[fmt]))
    svc.loader.loaded_model_id = 1  # the row is resident: `refuse` lets the request continue
    return client, svc, model


def turn_bodies(fmt: str, call_ids: list[str] | None = None) -> list[dict[str, Any]]:
    """Each turn's request body as the library builds it (`to_openai_messages`): the system
    message, the user turn, then for each earlier turn its assistant `tool_calls` message
    (`content: null`) and its `tool` result, plus the gateway's `steering` and the client fields."""
    messages: list[dict[str, Any]] = [{"role": "system", "content": FIXTURE["system"]}]
    bodies = []
    for index, turn in enumerate(FIXTURE["turns"]):
        messages = messages + turn["new_messages"]
        bodies.append({
            "model": NAME[fmt], "messages": list(messages), "tools": FIXTURE["tools"],
            **FIXTURE["client_fields"], **FIXTURE["body_added_by_gateway"],
        })
        (call,) = turn["expect"]["tool_calls"]
        call_id = (call_ids or [])[index] if call_ids and index < len(call_ids) else f"call_{index}"
        messages = messages + [
            {"role": "assistant", "content": None, "tool_calls": [{
                "id": call_id, "type": "function",
                "function": {"name": call["name"], "arguments": json.dumps(call["arguments"])}}]},
            {"role": "tool", "tool_call_id": call_id, "content": turn["tool_result"]},
        ]
    return bodies


def ignored(response) -> list[str]:
    value = response.headers.get("X-miLLM-Ignored-Fields", "")
    return sorted(part.strip().strip('"') for part in value.split(",") if part.strip())


def test_the_fixture_names_its_source_and_sends_no_refused_field():
    """5.1: the fixture cites the library source it was written from, and none of its extra
    fields is one the request policy refuses on transformers chat."""
    source = FIXTURE["source"]
    assert source["library"] == "weightspace" and source["version"] == "0.3.0"
    assert source["file"] == "src/weightspace/llm/openai.py" and source["read"]
    assert FIXTURE["headers"] == {"X-miLLM-Origin": "judge", "X-miLLM-Load-Policy": "refuse"}
    assert FIXTURE["body_added_by_gateway"] == {"steering": {"features": []}}
    extras = set(FIXTURE["client_fields"])
    assert extras == set(FIXTURE["client_extras"]["ignored"]) | set(
        FIXTURE["client_extras"]["honoured"])
    for field in sorted(extras | {"steering", "tools"}):
        outcome = OUTPUT_CHANGING.get(field, {}).get((CHAT, TF))
        assert not isinstance(outcome, Refused), f"{field} is refused on transformers chat"
    # The one field a gateway must never forward: it is refused on every cell.
    assert isinstance(OUTPUT_CHANGING["logit_bias"][(CHAT, TF)], Refused)
    assert "logit_bias" not in extras
    assert len(FIXTURE["turns"]) == 3


@pytest.mark.parametrize("fmt", FORMATS)
def test_each_turn_returns_the_expected_call(fmt):
    """5.2 and 5.7, non-streaming: the call COUNT, name and decoded arguments EQUAL the fixture's;
    `finish_reason`, `usage`, the ignored-field report and the provenance headers."""
    seen_ids: list[str] = []
    for index, turn in enumerate(FIXTURE["turns"]):
        (expected,) = turn["expect"]["tool_calls"]
        client, svc, model = client_for(fmt, reply_for(fmt, expected))
        body = turn_bodies(fmt, seen_ids)[index]
        r = client.post(URL, json=body, headers=FIXTURE["headers"])
        assert r.status_code == 200, (fmt, index, r.text)
        data = r.json()
        (choice,) = data["choices"]
        assert choice["finish_reason"] == "tool_calls"
        message = choice["message"]
        calls = message["tool_calls"]
        assert len(calls) == len(turn["expect"]["tool_calls"]) == 1
        assert CALL_ID.match(calls[0]["id"]), calls[0]["id"]
        assert calls[0]["type"] == "function"
        assert calls[0]["function"]["name"] == expected["name"]
        assert json.loads(calls[0]["function"]["arguments"]) == expected["arguments"]
        assert set(data["usage"]) >= {"prompt_tokens", "completion_tokens", "total_tokens"}
        assert data["usage"]["completion_tokens"] > 0
        # The three client extras are REPORTED, never refused; `chat_template_kwargs` is honoured.
        assert ignored(r) == FIXTURE["client_extras"]["ignored"], r.headers.get(
            "X-miLLM-Ignored-Fields")
        assert "chat_template_kwargs" not in r.headers.get("X-miLLM-Ignored-Fields", "")
        assert r.headers["X-miLLM-Model"].startswith(f'"{NAME[fmt]}"')
        assert r.headers["X-miLLM-Steering"] == "none"
        # The history turns were accepted and rendered: the model ran on a prompt that grew.
        assert len(model.generate_calls) == 1
        seen_ids.append(calls[0]["id"])
        svc.load_model_and_wait.assert_not_called()


@pytest.mark.parametrize("fmt", FORMATS)
def test_the_streamed_turn_carries_one_tool_calls_delta(fmt):
    """5.3 and 5.7: the call arrives as one `tool_calls` delta; the terminal chunks carry
    `millm_model` and `millm_steering: none`."""
    turn = FIXTURE["turns"][0]
    (expected,) = turn["expect"]["tool_calls"]
    client, _, _ = client_for(fmt, reply_for(fmt, expected))
    r = client.post(URL, json={**turn_bodies(fmt)[0], "stream": True}, headers=FIXTURE["headers"])
    assert r.status_code == 200, r.text
    events = [json.loads(line[6:]) for line in r.text.splitlines()
              if line.startswith("data: ") and line != "data: [DONE]"]
    deltas = [e["choices"][0] for e in events if e.get("choices")]
    with_calls = [c["delta"]["tool_calls"] for c in deltas if c["delta"].get("tool_calls")]
    assert len(with_calls) == 1 and len(with_calls[0]) == 1
    call = with_calls[0][0]
    assert call["function"]["name"] == expected["name"]
    assert json.loads(call["function"]["arguments"]) == expected["arguments"]
    assert [c["finish_reason"] for c in deltas if c.get("finish_reason")] == ["tool_calls"]
    terminal = {k: v for e in events for k, v in e.items() if k.startswith("millm_")}
    assert terminal["millm_model"].startswith(f'"{NAME[fmt]}"')
    assert terminal["millm_steering"] == "none"


# ── 5.6 refusals in the fixture's shape ─────────────────────────────────────────────────────────


def test_a_prompt_past_the_context_is_refused_before_generation():
    fmt = "lfm2"
    (expected,) = FIXTURE["turns"][0]["expect"]["tool_calls"]
    client, _, model = client_for(fmt, reply_for(fmt, expected))
    body = turn_bodies(fmt)[0]
    window = model.config.max_position_embeddings
    body["messages"] = body["messages"] + [{"role": "user", "content": "x" * (window + 10)}]
    r = client.post(URL, json=body, headers=FIXTURE["headers"])
    assert r.status_code == 400, r.text
    assert r.json()["error"]["code"] == "context_length_exceeded"
    assert model.generate_calls == []


def test_a_gguf_model_passes_every_field_check():
    """The gateway's full body on a GGUF row (2026-11-21; until then it was refused naming `tools`,
    "llama.cpp has no tool parser"): no field is refused, the three client extras are the only
    unused fields (`chat_template_kwargs` is USED: miServer renders that template itself), and the
    request reaches the load (sent without the gateway's `X-miLLM-Load-Policy: refuse`, which
    would answer first for a row that is not resident). The generation itself is
    `tests/unit/services/test_gguf_tool_calls.py`."""
    from miserver.api.request_policy import LC, evaluate
    from miserver.api.schemas.openai import ChatCompletionRequest
    from tests.unit.f25_fixtures import unloaded_inference

    body = {**turn_bodies("lfm2")[1], "model": "tiny"}
    assert body["steering"] == {"features": []}
    result = evaluate(ChatCompletionRequest(**body), CHAT, LC, strict=False)
    assert sorted(result.unused) == sorted(FIXTURE["client_extras"]["ignored"])

    client, svc = make_client(unloaded_inference(), model_row(gguf_files=["m.gguf"]))
    headers = {k: v for k, v in FIXTURE["headers"].items() if k != "X-miLLM-Load-Policy"}
    r = client.post(URL, json=body, headers=headers)
    assert svc.load_model_and_wait.call_count == 1, r.text


def test_a_template_with_an_unrecognised_call_format_is_refused_naming_tools():
    from tests.unit.f25_fixtures import word_model, word_tokenizer

    client, svc = make_client(make_service(word_model(), word_tokenizer()), model_row())
    svc.loader.loaded_model_id = 1
    body = {**turn_bodies("lfm2")[0], "model": "tiny"}
    r = client.post(URL, json=body, headers=FIXTURE["headers"])
    assert r.status_code == 400, r.text
    error = r.json()["error"]
    assert error["code"] == "field_not_honoured" and error["param"] == "tools"


# ── 5.4 judge origin: scored nothing, recorded no text ─────────────────────────────────────────


@pytest.fixture
async def judged(clean_state, db, monkeypatch):  # noqa: F811
    """The LFM2 stand-in with one REAL probe armed and transcripts ON (the worst case)."""
    from miserver.api.dependencies import get_db
    from miserver.core.config import settings
    from miserver.db.repositories.probe_repository import ProbeRepository
    from miserver.ml.probe_head import ProbeHead
    from miserver.services import probe_sessions
    from miserver.services.probe_runtime import ArmedProbe, ProbeRuntimeState
    from miserver.services.probe_service import ProbeService
    from tests.unit.probe_fixtures import probe_definition

    monkeypatch.setattr(settings, "PROBE_TRANSCRIPTS", "on")
    probe_sessions._forget_for_tests()
    ProbeRuntimeState.reset_for_tests()
    (expected,) = FIXTURE["turns"][0]["expect"]["tool_calls"]
    client, svc, model = client_for("lfm2", reply_for("lfm2", expected))
    async with db() as session:
        row = await ProbeService(ProbeRepository(session)).import_definition(
            probe_definition(), raw_bytes=1, on_conflict="rename", origin="file")
    ProbeRuntimeState().arm(ArmedProbe(
        probe_id=row.id, name=row.name, head=ProbeHead(weight=torch.ones(16), bias=0.0, layer=0),
        rule="mean", scope="all", layer=0, rung=2, rung_language="detects on unseen tasks",
        threshold=0.0, windows=("all",)), model)

    async def _db():
        async with db() as session:
            yield session

    client.app.dependency_overrides[get_db] = _db
    yield client, db, row
    ProbeRuntimeState.reset_for_tests()
    probe_sessions._forget_for_tests()


async def _all(factory, model_cls):
    from sqlalchemy import select

    async with factory() as session:
        return (await session.execute(select(model_cls))).scalars().all()


async def test_editor_traffic_is_judge_traffic_and_records_no_text(judged):
    from miserver.db.models.probe import ProbeEvent
    from miserver.db.models.probe_session import ProbeSession, ProbeSessionTurn, ProbeTranscript

    client, factory, row = judged
    body = turn_bodies("lfm2")[0]
    with no_cuda():
        r = client.post(URL, json=body, headers={**FIXTURE["headers"], "X-miLLM-Session": "s1"})
    assert r.status_code == 200, r.text
    assert 'not-scored;reason="judge_traffic"' in r.headers["X-miLLM-Probe-Verdicts"]
    events = await _all(factory, ProbeEvent)
    assert [(e.probe_id, e.scored, e.not_scored_reason, e.session_id) for e in events] == [
        (row.id, False, "judge_traffic", None)]
    assert await _all(factory, ProbeSession) == []
    assert await _all(factory, ProbeSessionTurn) == []
    assert await _all(factory, ProbeTranscript) == []


# ── 5.4b unsteered under an active profile (P47) ───────────────────────────────────────────────


async def test_the_empty_set_unsteers_under_an_active_profile(clean_state, db):  # noqa: F811
    """`steering: {"features": []}` reads `none` under an ACTIVE profile on an attached SAE. The
    control (M13b's shape) is the same turn without the field, which the profile steers."""
    (expected,) = FIXTURE["turns"][0]["expect"]["tool_calls"]
    client, _, model = client_for("lfm2", reply_for("lfm2", expected))
    served = attach(model, [(SAE_A, 0, 11)])
    try:
        await add_profile(db, "editor-guard", {1: 2.0}, active=True, sae_id=SAE_A, layer=0)
        # What activating the profile does to the served SAE: its values are live and enabled.
        sae = served.sae(SAE_A, 0)
        sae.set_steering_batch({1: 2.0})
        sae.enable_steering(True)
        body = turn_bodies("lfm2")[0]
        without = {k: v for k, v in body.items() if k != "steering"}
        with no_cuda():
            control = client.post(URL, json=without, headers=FIXTURE["headers"])
            unsteered = client.post(URL, json=body, headers=FIXTURE["headers"])
        assert control.status_code == 200, control.text
        assert control.headers["X-miLLM-Steering"] != "none", "the control must be steered"
        assert unsteered.status_code == 200, unsteered.text
        assert unsteered.headers["X-miLLM-Steering"] == "none"
        with no_cuda():
            streamed = client.post(URL, json={**body, "stream": True}, headers=FIXTURE["headers"])
        terminal = {k: v for line in streamed.text.splitlines()
                    if line.startswith("data: ") and line != "data: [DONE]"
                    for k, v in json.loads(line[6:]).items() if k.startswith("millm_")}
        assert terminal["millm_steering"] == "none"
    finally:
        for handle in served.handles:
            handle.remove()


# ── 5.5 load policy: the load spy is the primary check ─────────────────────────────────────────


@pytest.fixture
def policy_env():
    from unittest.mock import AsyncMock

    from miserver.api.dependencies import get_inference_service, get_model_service
    from miserver.main import create_app
    from fastapi.testclient import TestClient
    from tests.unit.lease_fixtures import (
        build_service,
        clear_resident,
        install_registry,
        resident_inference,
    )

    clear_resident()
    install_registry()
    service, repo = build_service()
    service.load_model_and_wait = AsyncMock(wraps=service.load_model_and_wait)
    app = create_app()
    app.dependency_overrides[get_model_service] = lambda: service
    app.dependency_overrides[get_inference_service] = resident_inference
    yield TestClient(app), service, repo
    clear_resident()


def test_a_non_resident_model_is_refused_and_nothing_loads(policy_env):
    client, service, repo = policy_env
    body = {**turn_bodies("lfm2")[0], "model": "m2"}  # m1 is resident
    r = client.post(URL, json=body, headers=FIXTURE["headers"])
    assert r.status_code == 409, r.text
    assert r.json()["error"]["code"] == "model_not_resident"
    assert service.load_model_and_wait.await_count == 0
    assert service._load_worker.call_count == 0
    assert repo.status_writes == []


def test_a_model_being_loaded_is_503_with_retry_after_and_nothing_loads(policy_env):
    from miserver.db.models.model import ModelStatus

    client, service, repo = policy_env
    repo.rows[2].status = ModelStatus.LOADING
    body = {**turn_bodies("lfm2")[0], "model": "m2"}
    r = client.post(URL, json=body, headers=FIXTURE["headers"])
    assert r.status_code == 503, r.text
    assert r.json()["error"]["code"] == "model_loading"
    assert r.headers["Retry-After"]
    assert service.load_model_and_wait.await_count == 0
    assert service._load_worker.call_count == 0


# ── 5.4b unsteered under serving circuits (P47; the Feature 19 harness) ────────────────────────


def _second_circuit(doc: dict) -> dict:
    """A second circuit over the same two SAEs, on other features, so both can serve together."""
    import copy

    second = copy.deepcopy(doc)
    second["name"] = "second"
    second["members"][0]["feature"].update(feature_idx=1, strength=1.5)
    second["members"][1]["feature"].update(feature_idx=2, strength=2.5)
    second["edges"][0]["up"].update(feature_idx=1)
    second["edges"][0]["down"].update(feature_idx=2)
    return second


def test_the_empty_set_unsteers_under_serving_circuits(served):  # noqa: F811
    """The editor's turn (its messages; the word model's template takes no tools) with the empty
    set reads `none` while one circuit serves and again while two do; the control without the
    field reads the circuit(s) (M13b's shape)."""
    from miserver.api.dependencies import get_model_service

    _s, client = served
    client.app.dependency_overrides[get_model_service]().loader.loaded_model_id = 1  # resident
    body = {"model": "tiny", "messages": turn_bodies("lfm2")[0]["messages"], "max_tokens": 3,
            **FIXTURE["body_added_by_gateway"]}
    without = {k: v for k, v in body.items() if k != "steering"}
    first = client.post("/api/circuits/import", json=calibrated_doc()).json()
    assert first["success"], first
    act = client.post(f"/api/circuits/{first['data']['id']}/activate").json()
    assert act["success"], act
    serving = [first["data"]["id"]]
    second = client.post("/api/circuits/import", json=_second_circuit(calibrated_doc())).json()
    assert second["success"], second
    # Two circuits composing on the same layers (the compounding override): the case where the
    # intensity dial is ignored, so only the empty set can unsteer the editor (TD43-7, P47).
    both = client.post(f"/api/circuits/{second['data']['id']}/activate",
                       params={"allow_layer_overlap": "true"}).json()
    if both["success"]:
        serving.append(second["data"]["id"])
    for _ in serving:
        with no_cuda():
            control = client.post(URL, json=without, headers=FIXTURE["headers"])
            unsteered = client.post(URL, json=body, headers=FIXTURE["headers"])
        assert control.status_code == 200, control.text
        assert control.headers["X-miLLM-Steering"].startswith("circuit"), control.headers[
            "X-miLLM-Steering"]
        assert unsteered.status_code == 200, unsteered.text
        assert unsteered.headers["X-miLLM-Steering"] == "none"
    assert len(serving) == 2, f"a second circuit did not serve beside the first: {both}"
