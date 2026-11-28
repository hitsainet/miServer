"""044 answer models: the OpenAI-shaped Batch API objects (Feature 26), served BARE.

No management envelope. Documented-only typing (044, TD9, T1 bare dict): each route keeps its body
and its `-> dict[str, Any]` annotation; the decorator gains `response_model=None` and
`responses={200: {"model": <Answer>}}`.

`miserver/api/routes/openai/batches.py` (router mounted under `/v1`):
* `POST /v1/batches`, `GET /v1/batches/{batch_id}`, `POST /v1/batches/{batch_id}/cancel`,
  `POST /v1/batches/{batch_id}/lease` -> `BatchObject` (`batch_object`, services/batch/state.py:136)
* `GET /v1/batches` -> `BatchList` (`openai_list`, state.py:176)

`miserver/api/routes/openai/files.py`:
* `POST /v1/files`, `GET /v1/files/{file_id}` -> `FileObject` (`file_object`, state.py:121)
* `GET /v1/files` -> `FileList`
* `DELETE /v1/files/{file_id}` -> `FileDeleted` (files.py:189)
(`GET /v1/files/{file_id}/content` streams `application/jsonl` and is not modelled here.)

Timestamps are Unix seconds (`epoch`). The batch's server extension object is sent under its
FROZEN wire key (legacy register DR-34); the Python attribute is `extension`, the key its alias.
"""

from typing import Literal

from pydantic import BaseModel, Field

# ── the batch object ──────────────────────────────────────────────────────────────────────────


class BatchErrorEntry(BaseModel):
    """One batch-level error (validator.py:121, runner.py:263-350 and :679, reconcile.py:66)."""

    code: str
    line: int | None = Field(description="1-based input line; null for a batch-wide error")
    param: str | None
    message: str


class BatchErrorList(BaseModel):
    """OpenAI's `errors`: the first BATCH_ERRORS_SHOWN entries."""

    object: Literal["list"]
    data: list[BatchErrorEntry]


class BatchRequestCounts(BaseModel):
    total: int
    completed: int
    failed: int


#: DR-34 (frozen): the batch extension key, spelled as `services/batch/state.py` writes it.
_FROZEN_EXTENSION_KEY = {
    "millm": "the server's batch extension fields",
}


class BatchExtensionFields(BaseModel):
    """The server's own batch fields (FTDD §5), outside OpenAI's object."""

    pack: bool = Field(description="Rows packed into one forward (T-63)")
    waiting_reason: str | None = Field(description="Why an in_progress batch is not running a row")
    lease_mode: str | None = Field(description="own or caller; null until a lease is taken")
    completion_window_extension: bool = Field(
        description="True when completion_window is not OpenAI's only value, 24h (T-65)"
    )
    output_expires_after: int = Field(description="Seconds the output and error files are kept")


class BatchObject(BaseModel):
    """OpenAI's batch object (FR-26.1.2) plus the server's extension object."""

    id: str
    object: Literal["batch"]
    endpoint: str
    model: str | None = Field(description="Set at the end of validation from the lines' one model")
    errors: BatchErrorList | None
    input_file_id: str | None = Field(description="null once the input file row is gone")
    completion_window: str = Field(description="`<N>h`")
    status: str = Field(description="One of OpenAI's eight batch statuses (BatchStatus)")
    output_file_id: str | None
    error_file_id: str | None
    created_at: int
    in_progress_at: int | None
    expires_at: int
    finalizing_at: int | None
    completed_at: int | None
    failed_at: int | None
    expired_at: int | None
    cancelling_at: int | None
    cancelled_at: int | None
    request_counts: BatchRequestCounts
    metadata: dict[str, str] | None = Field(description="As given at create (at most 16 pairs)")
    extension: BatchExtensionFields = Field(
        alias=next(iter(_FROZEN_EXTENSION_KEY)),
        title="Extension",
        description="The server's batch extension fields, under the frozen key",
    )


class BatchList(BaseModel):
    """OpenAI's list shape over batches, newest first."""

    object: Literal["list"]
    data: list[BatchObject]
    first_id: str | None = Field(description="null on an empty page")
    last_id: str | None = Field(description="null on an empty page")
    has_more: bool


# ── the file object ───────────────────────────────────────────────────────────────────────────


class FileObject(BaseModel):
    """OpenAI's file object (FR-26.1.1). The storage path is never returned."""

    id: str
    object: Literal["file"]
    bytes: int
    created_at: int
    expires_at: int
    filename: str
    purpose: str = Field(description="batch or batch_output (FilePurpose)")
    status: str = Field(description="processed, error, deleted or expired (FileStatus)")
    status_details: None = Field(description="Always null")


class FileList(BaseModel):
    """OpenAI's list shape over files."""

    object: Literal["list"]
    data: list[FileObject]
    first_id: str | None = Field(description="null on an empty page")
    last_id: str | None = Field(description="null on an empty page")
    has_more: bool


class FileDeleted(BaseModel):
    """OpenAI's deletion answer (FR-26.6.10): the bytes are gone, the record is marked."""

    id: str
    object: Literal["file"]
    deleted: bool
