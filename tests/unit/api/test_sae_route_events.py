"""The attach and detach routes emit `sae:attached` / `sae:detached`, once each (RR-9, 2026-11-21).

`ProgressEmitter.emit_sae_attached` and `emit_sae_detached` had no caller anywhere, so the admin UI's
listeners for both events were dead. The live app (`create_app`) with the SAE service stubbed and the
emitter's two methods recorded: the payload and the call count are asserted, and a refused request
emits nothing.

MUTATION CONTROLS (recorded in 0xcc/reviews/unload_detaches_saes_2026-11-21.md §11):
  E1 the attach route's emit removed  -> test_attach_emits_once_with_the_payload
  E2 the detach route's emit removed  -> test_detach_emits_once_with_the_payload
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from miserver.core.errors import SAENotAttachedError


@pytest.fixture
def app_and_events(monkeypatch):
    from miserver.api.dependencies import get_sae_service
    from miserver.main import create_app
    from miserver.sockets.progress import progress_emitter

    app = create_app()
    service = MagicMock()
    service.attach_sae = AsyncMock(return_value={
        "status": "attached", "sae_id": "sae_alpha", "layer": 12, "memory_usage_mb": 96,
        "warnings": [], "layer_module_path": "model.layers.12"})
    service.detach_sae = AsyncMock(return_value={
        "status": "detached", "sae_id": "sae_alpha", "memory_freed_mb": 96,
        "deactivated_circuits": [], "deactivated_profiles": []})
    app.dependency_overrides[get_sae_service] = lambda: service
    attached, detached = AsyncMock(), AsyncMock()
    monkeypatch.setattr(progress_emitter, "emit_sae_attached", attached)
    monkeypatch.setattr(progress_emitter, "emit_sae_detached", detached)
    return TestClient(app), service, attached, detached


def test_attach_emits_once_with_the_payload(app_and_events):
    client, _service, attached, detached = app_and_events
    r = client.post("/api/saes/sae_alpha/attach", json={"layer": 12})
    assert r.status_code == 200, r.text
    attached.assert_awaited_once_with(sae_id="sae_alpha", layer=12, memory_mb=96)
    detached.assert_not_awaited()


def test_detach_emits_once_with_the_payload(app_and_events):
    client, _service, attached, detached = app_and_events
    r = client.post("/api/saes/sae_alpha/detach")
    assert r.status_code == 200, r.text
    detached.assert_awaited_once_with(sae_id="sae_alpha")
    attached.assert_not_awaited()


def test_a_refused_detach_emits_nothing(app_and_events):
    client, service, attached, detached = app_and_events
    service.detach_sae.side_effect = SAENotAttachedError("SAE 'sae_alpha' is not attached")
    r = client.post("/api/saes/sae_alpha/detach")
    assert r.status_code >= 400
    detached.assert_not_awaited()
    attached.assert_not_awaited()
