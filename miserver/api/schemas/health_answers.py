"""044 answer models: two health routes that answer a BARE dict (no management envelope).

Documented-only typing (044, TD9, T1 bare dict): each route keeps its body; the decorator gains
`response_model=None` and `responses={200: {"model": <Answer>}}`.

* `GET /api/health/inference` -> `InferenceBackendInfo` (`InferenceService.get_backend_info`,
  `miserver/services/inference_service.py:1727`). Three branches build it: `llamacpp` (a resident
  GGUF model), `cbm` (the continuous batching manager is running) and `serial` (otherwise,
  including an idle server with no model). Keys only one branch sends are `may_be_absent()`.
* `GET /api/health/version` -> `VersionInfo` (`get_version`,
  `miserver/api/routes/system/health.py:1266`).
"""

from typing import Any

from pydantic import BaseModel, Field

from miserver.api.schemas.common import may_be_absent


class InferenceCapabilities(BaseModel):
    """What a request can ask of the active backend."""

    streaming: bool
    per_request_sampling_params: bool = Field(description="Per-request temperature and top_p")
    per_request_profile_override: bool = Field(description="A request may name a steering profile")
    speculative_decoding: bool = Field(description="serial: true when SPECULATIVE_MODEL is set")


class CbmSettings(BaseModel):
    """The continuous batching manager's fixed sampling and queue settings."""

    default_temperature: float = Field(description="Requests asking another value go serial")
    default_top_p: float = Field(description="Requests asking another value go serial")
    max_queue_size: int


class SerialQueueState(BaseModel):
    """The serial request queue, read now."""

    max_concurrent: int
    max_pending: int = Field(description="MAX_PENDING_REQUESTS, read at call time")
    current_pending: int


class InferenceBackendInfo(BaseModel):
    """The active inference backend, its per-request capabilities and its known limitations."""

    backend: str = Field(description="llamacpp, cbm or serial")
    description: str
    capabilities: InferenceCapabilities
    limitations: list[str]
    # llama.cpp always sends it (inference_service.py:1749; 0 = the model's full declared context).
    # serial and cbm send it only when a transformers model is resident AND its config states a
    # window (inference_service.py:1852-1853); an idle server, or a config without one, omits it.
    context_length: int = may_be_absent(
        description="The context window requests are held to (prompt + generation)"
    )
    # llama.cpp branch only (inference_service.py:1755); cbm and serial never send it.
    kv_cache_type: str | None = may_be_absent(
        description="llamacpp: the KV cache type the context was created with; null if unrecorded"
    )
    # llama.cpp branch only (inference_service.py:1756). Free-form: the load's decision record
    # (`model_loader.load_gguf_model`, its `attempts`/`trials` lists and nested dicts), declared
    # as a map exactly as `ModelResponse.gguf_load` is (`miserver/api/schemas/model.py:499`).
    gguf_load: dict[str, Any] | None = may_be_absent(
        description="llamacpp: why the load got its cache, embeddings and window; null if none"
    )
    # cbm branch only (inference_service.py:1804).
    cbm_config: CbmSettings = may_be_absent(description="cbm: the manager's fixed settings")
    # serial branch only (inference_service.py:1833).
    queue: SerialQueueState = may_be_absent(description="serial: the request queue now")


class VersionInfo(BaseModel):
    """The application version: the VERSION file, else the package version."""

    version: str
    app: str = Field(description="The application name, `miServer`")
