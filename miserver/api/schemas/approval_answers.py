"""Typed answers of `/api/approvals` and of the approval gate's pending 202 (044, documented-only
typing; ENH-008 D1, D6; ENH-006 D79, D81).

The routes (`miserver/api/routes/management/approvals.py`):

* `GET  /api/approvals`                       -> `ApiResponse[ApprovalList]`
* `GET  /api/approvals/actions`               -> `ApiResponse[ApprovalActions]`
* `POST /api/approvals/approve-batch`  (202)  -> `ApiResponse[ApprovalBatchDecision]`
* `GET  /api/approvals/{approval_id}`         -> `ApiResponse[ApprovalRead]`
* `POST /api/approvals/{approval_id}/approve` (202) -> `ApiResponse[ApprovalRead]`
* `POST /api/approvals/{approval_id}/reject`  -> `ApiResponse[ApprovalRead]`

`ApprovalPending` is the `data` of the 202 every gated route answers an agent with
(`approval_gate.requires_approval_when_agent` -> `approval_service.pending_body`).

The row is `approval_service.as_payload`. ⚠ It never carries `secret_payload` (the lease id an agent
sent), and so neither does this model.
"""

from typing import Any, Literal

from pydantic import BaseModel, Field

from miserver.api.schemas.common import ErrorDetails, history_form

#: `db.models.approval.APPROVAL_STATUSES` (the table's CHECK constraint).
ApprovalStatus = Literal["pending", "executing", "executed", "failed", "rejected", "expired"]
#: `approval_gate.ApprovalAction.batch_kind`: which "Approve all" batch may approve an action.
ApprovalBatchKind = Literal["settings", "code_rules"]


class ApprovalPending(BaseModel):
    """The `data` of a gated route's 202 to an agent: the request is stored, not run
    (`approval_service.pending_body`)."""

    approval_id: str = Field(description="`apr_` + 32 hex")
    status: Literal["pending"]
    action: str = Field(description="The gated action, e.g. `model_load`")
    summary: dict[str, Any] = Field(
        description="The card facts, `{label: value}` in display order; a value is usually text, "
        "sometimes a number, a map or a list (`approval_facts`)"
    )
    request_digest: str = Field(description="`sha256:` + 64 hex of the canonical stored payload")
    expires_at: str = Field(description="ISO 8601")
    hint: str = Field(description="Where the operator decides it")


class ApprovalRead(BaseModel):
    """One request on the wire (`approval_service.as_payload`)."""

    approval_id: str = Field(description="`apr_` + 32 hex")
    action: str
    action_label: str = Field(description="The action's label; the action itself if not served now")
    settings_change: bool = Field(description="True when the `settings` batch may approve it")
    batch_kind: ApprovalBatchKind | None = Field(
        description='Which "Approve all" batch may approve it; null: one at a time only'
    )
    status: ApprovalStatus
    summary: dict[str, Any] = Field(
        description="The card facts, `{label: value}` in display order (values not all text)"
    )
    payload: dict[str, Any] = Field(
        description="Every value the route was called with (path, query, body), shown in full"
    )
    request_digest: str = Field(description="`sha256:` + 64 hex of the canonical stored payload")
    requested_by: str = Field(description="`agent:<name>` from the agent header")
    decided_by: str | None = Field(description="`operator`, or `system` for an expiry or restart")
    reason: str | None = Field(description="The operator's reason for a rejection")
    result: Any = history_form(
        description="The approved route's `data` once executed (its own answer, as stored then); "
        "null until then"
    )
    error: ErrorDetails | None = Field(
        description="Why it failed (a tamper, a restart, the route's refusal); null otherwise"
    )
    created_at: str = Field(description="ISO 8601")
    decided_at: str | None = Field(description="ISO 8601")
    expires_at: str = Field(description="ISO 8601")
    batch_id: str | None = Field(description='`bat_` + 32 hex when approved by "Approve all"')


class ApprovalList(BaseModel):
    """`GET /api/approvals`: newest first (`status=decided`: newest decision first)."""

    approvals: list[ApprovalRead]
    count: int = Field(description="How many rows this answer holds")
    pending: int = Field(description="How many requests are waiting, whatever the filter")


class ApprovalActionEntry(BaseModel):
    """One action this miServer gates, with the live routes that carry it."""

    action: str
    label: str
    description: str
    settings_change: bool
    batch_kind: ApprovalBatchKind | None
    routes: list[str] = Field(description="`METHOD /path` of each live route gated with it")


class ApprovalActions(BaseModel):
    """`GET /api/approvals/actions`: read from the LIVE route table; an action no route carries is
    not listed."""

    actions: list[ApprovalActionEntry]


class ApprovalBatchOutcome(BaseModel):
    """One id of an "Approve all", in the order given (`approval_service.approve_batch`)."""

    approval_id: str
    status: ApprovalStatus | Literal["not_run"] = Field(
        description="The row's status after its approve (usually `executing`, `executed` or "
        "`failed`); `not_run` when it was decided elsewhere, expired or tampered"
    )
    error: ErrorDetails | None


class ApprovalBatchDecision(BaseModel):
    """`POST /api/approvals/approve-batch` (202): every named request of one kind, each approved
    through the single approve with the shared `batch_id`."""

    batch_id: str = Field(description="`bat_` + 32 hex")
    kind: ApprovalBatchKind
    results: list[ApprovalBatchOutcome]
