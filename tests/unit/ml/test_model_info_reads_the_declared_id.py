"""Model preview and registration read the repo id from `ModelInfo.id`, never `modelId` (044 debt, 2026-11-25).

huggingface_hub 2.x declares `id`; `modelId` is not a field. It is the Hub's legacy JSON key, which
`ModelInfo.__init__` copies onto the object only because it keeps every key it does not know
(`self.__dict__.update(**kwargs)`). The Hub's full `/api/models/{repo}` answer still sends it; an
`expand[]=` answer does not (checked 2026-11-25), and nothing promises the full one keeps it. When it
is absent, `info.modelId` raises AttributeError, and:

* `POST /api/models/preview` answered 502 DOWNLOAD_FAILED ("Failed to fetch model info: 'ModelInfo'
  object has no attribute 'modelId'") for a repo that exists;
* `POST /api/models` swallowed the same error as a warning and registered the row with no `params`
  and no `estimated_memory_mb` (and the architecture defaulted), silently.

The existing tests used `MagicMock()` infos, on which every attribute exists, so they could not see it.
These build a REAL `ModelInfo` from the keys the Hub sends, without `modelId`.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from huggingface_hub.hf_api import ModelInfo

from miserver.ml.model_downloader import ModelDownloader

REPO = "acme/tiny-chat-7b"


def hub_info(repo_id: str = REPO, **over: Any) -> ModelInfo:
    """`ModelInfo` as huggingface_hub builds it from an `/api/models/{repo}` answer that has no
    `modelId` key (the `expand[]=` shape)."""
    body: dict[str, Any] = dict(
        _id="0" * 24, id=repo_id, sha="1" * 40, gated=False, pipeline_tag="text-generation",
        library_name="transformers", downloads=10, likes=2, tags=["text-generation"],
        config={"model_type": "llama", "architectures": ["LlamaForCausalLM"]},
        cardData={"license": "apache-2.0", "language": ["en"]},
        safetensors={"parameters": {"BF16": 1_200_000_000}, "total": 1_200_000_000},
        siblings=[{"rfilename": "model.safetensors", "size": 2_400_000_000},
                  {"rfilename": "config.json", "size": 700}],
    )
    body.update(over)
    info = ModelInfo(**body)
    assert not hasattr(info, "modelId"), "fixture: the Hub answer must not carry the legacy key"
    return info


@pytest.fixture
def downloader(tmp_path):
    return ModelDownloader(cache_dir=str(tmp_path / "cache"))


def test_the_preview_reads_the_declared_id(downloader):
    with patch.object(downloader.hf_api, "model_info", return_value=hub_info()):
        result = downloader.get_model_info(REPO)

    assert result["repo_id"] == REPO
    assert result["name"] == "tiny-chat-7b"
    assert result["params"] == "1.2B"
    assert result["revision"] == "1" * 40


def test_the_name_fallback_for_params_reads_the_declared_id(downloader):
    """With no safetensors and no gguf block, the count is guessed from the repo id."""
    with patch.object(downloader.hf_api, "model_info",
                      return_value=hub_info(safetensors=None)):
        result = downloader.get_model_info(REPO)

    assert result["params"] == "7B"


def test_the_preview_route_answers_for_an_answer_without_model_id(downloader):
    from fastapi.testclient import TestClient

    from miserver.api.dependencies import get_model_service
    from miserver.main import create_app
    from miserver.services.model_service import ModelService

    service = ModelService(repository=MagicMock(), downloader=downloader, emitter=MagicMock())
    app = create_app()
    app.dependency_overrides[get_model_service] = lambda: service
    with patch.object(downloader.hf_api, "model_info", return_value=hub_info()):
        response = TestClient(app).post("/api/models/preview", json={"repo_id": REPO})

    assert response.status_code == 200, response.text
    data = response.json()["data"]
    assert data["name"] == "tiny-chat-7b"
    assert data["params"] == "1.2B" and data["revision"] == "1" * 40
    assert data["estimated_sizes"]["FP16"]["disk_mb"] == 2400


@pytest.mark.asyncio
async def test_a_registration_keeps_the_hub_metadata(downloader):
    """The download path swallowed the AttributeError and registered the row without its metadata."""
    from miserver.api.schemas.model import ModelDownloadRequest
    from miserver.db.models.model import ModelSource, QuantizationType
    from miserver.services.model_service import ModelService
    from tests.support.factories import make_model

    repository = MagicMock()
    repository.find_by_repo_quantization = AsyncMock(return_value=None)
    repository.find_by_exact_name = AsyncMock(return_value=None)
    repository.create = AsyncMock(return_value=make_model(id=1))
    service = ModelService(repository=repository, downloader=downloader, emitter=MagicMock())
    request = ModelDownloadRequest(source=ModelSource.HUGGINGFACE, repo_id=REPO,
                                   quantization=QuantizationType.FP16)
    with patch.object(downloader.hf_api, "model_info", return_value=hub_info()), \
         patch.object(ModelService, "_download_worker", lambda self, *a, **k: None):
        await service.download_model(request)

    kwargs = repository.create.call_args.kwargs
    assert kwargs["params"] == "1.2B"
    assert kwargs["estimated_memory_mb"] is not None
    assert kwargs["architecture"] == "text-generation"
