"""Typed answers of `/api/rule-sets` (044, documented-only typing; 039 FR-39.17, FR-39.19,
FR-39.25).

The routes (`miserver/api/routes/management/rule_sets.py`):

* `GET    /api/rule-sets/validators`             -> `ApiResponse[ValidatorListing]`
* `POST   /api/rule-sets/validate`               -> `ApiResponse[RuleSetValidation]`
* `POST   /api/rule-sets/import` (gated)         -> `ApiResponse[RuleSetImported]` (the route's own
  answer; an agent's pending 202 is the gate's, declared by `gated_responses`)
* `GET    /api/rule-sets`                        -> `PageOf[RuleSetSummary]`
* `GET    /api/rule-sets/{rule_set_id}`          -> `ApiResponse[RuleSetDetail]`
* `DELETE /api/rule-sets/{rule_set_id}`          -> `ApiResponse[RuleSetDeleted]`
* `POST   /api/rule-sets/{rule_set_id}/evaluate` -> `ApiResponse[RuleSetEvaluation]`

The summary and the detail are built by `miserver/services/rule_set_service.py` (`summary`,
`detail`). The stored rule-set document is the vendored producer schema, referenced by its `$id`,
not re-modelled.
"""

from typing import Any, Literal

from pydantic import BaseModel, Field

from miserver.api.schemas.common import may_be_absent

#: The vendored rule-set document schema (`docs/schemas/midataworks-rule-set-v1.json`).
RULE_SET_DOCUMENT_ID = (
    "https://raw.githubusercontent.com/hitsainet/miDataworks/main/docs/schemas/"
    "midataworks-rule-set-v1.json"
)

#: `rules.document.origin_of`: `derived` when the document carries provenance, else `authored`.
RuleSetOrigin = Literal["authored", "derived"]
#: `RuleSet.vector_status`: `passed` at import, `vectors_failed` from the startup self-check.
RuleSetVectorStatus = Literal["passed", "vectors_failed"]
#: The four text windows (`rules.windows`); the evaluate body's `windows` is this Literal too.
RuleWindow = Literal["prompt", "last_user", "response", "all"]


# ── GET /api/rule-sets/validators ─────────────────────────────────────────────────────────────


class RuleValidatorOutput(BaseModel):
    """One named output of a validator."""

    name: str
    type: str = Field(description="The output's value type, e.g. `bool` or `string`")


class RuleValidatorMaskShape(BaseModel):
    """How a masked value of a format may look (`validators.base.MaskShape`)."""

    lengths: list[int]
    layouts: list[list[int]] = Field(description="Each layout: the digit-group lengths, in order")
    delimiters: str = Field(description="Every delimiter character a layout may use")
    keep_leading: int
    keep_trailing: int


class RuleValidatorEntry(BaseModel):
    """One registered validator (`rules.validators.declaration`)."""

    id: str
    version: int
    ref: str = Field(description="`<id>@<version>`, the exact form a rule names")
    description: str
    params_schema: dict[str, Any] = Field(
        description="The JSON Schema a rule's free-form `params` map must satisfy"
    )
    reasons: list[str] = Field(description="Every reason code the validator can report")
    outputs: list[RuleValidatorOutput]
    algorithms: list[str]
    value_bearing: bool
    mask_shape: RuleValidatorMaskShape | None = Field(
        description="Null for a validator with no masked form"
    )
    superseded_by: int | None = Field(
        description="The next registered version of the same id; null when this is the newest"
    )


class RuleFormEntry(BaseModel):
    """One form the vendored schema declares, and where this miServer evaluates it."""

    form: str
    evaluated: Literal["inline", "after_response", "unavailable"] = Field(
        description="`inline` on the live path; `after_response` by the rule runner (operator "
        "code); `unavailable` when this miServer neither implements nor defers it"
    )


class ValidatorListing(BaseModel):
    """The live validator registry and the forms this miServer evaluates (FR-39.19)."""

    validators: list[RuleValidatorEntry]
    forms: list[RuleFormEntry]


# ── POST /api/rule-sets/validate ──────────────────────────────────────────────────────────────


class RuleSetValidation(BaseModel):
    """A dry run that passed every import check and wrote nothing (a failure is a refusal)."""

    valid: bool = Field(description="Always true: a set that fails a check is refused instead")
    digest: str = Field(description="The document's identity digest (RFC 8785 + SHA-256), 64 hex")
    origin: RuleSetOrigin
    behaviour_key: str
    cases_passed: int = Field(description="The set's own cases that agreed")
    cases_unverified: dict[str, list[str]] = Field(
        description="`{case_id: [column, ...]}` not checked because they need operator code (D41)"
    )
    pending_signals: list[str] = Field(
        description="Signals accepted and scored after the response (operator code), sorted"
    )
    code_modules: list[str] = Field(
        description="SHA-256 of each code document the import carried and verified, sorted"
    )


# ── the stored rule set: summary (list, import), detail (get) ─────────────────────────────────


class RuleSetSummary(BaseModel):
    """One stored rule set (`rule_set_service.summary`, called without attachments). Also the item
    of `GET /api/rule-sets` and of a behaviour's attachment view (`attachment_view.rule_sets`)."""

    id: str = Field(description="`rs_` + 12 hex")
    name: str
    version: int
    behaviour_key: str
    origin: RuleSetOrigin
    digest: str
    vector_status: RuleSetVectorStatus
    signals: int = Field(description="How many signals the document declares")
    expressions: list[str] = Field(description="The document's expression names, sorted")
    imported_at: str | None = Field(description="ISO 8601")
    checked_at: str | None = Field(description="ISO 8601; null until the startup self-check ran")
    code_modules: list[str] = Field(
        description="SHA-256 of each code document held (hashes only, never the source), sorted"
    )
    code_modules_missing: list[str] = Field(
        description="Hashes a code_rule signal names whose document miServer does not hold"
    )


class RuleSetImported(RuleSetSummary):
    """`POST /api/rule-sets/import`: the stored set (an identical document returns the existing
    one)."""

    created: bool = Field(description="False when an identical document was already imported")


class RuleSetVectorReport(BaseModel):
    """Where a stored set first disagreed with its own cases under this build
    (`rules.selfcheck.check_document`)."""

    case_index: int | None = Field(description="Null when the set no longer compiles")
    case_id: str | None
    where: str = Field(description="The column that disagreed, or the refusal code")
    signal: str | None
    build: str = Field(description="The evaluator build that ran the check")


class RuleSetMatch(BaseModel):
    """The set's positive (D53, `rules.document.match_of`)."""

    source: Literal["explicit", "inferred"]
    version: int | None = Field(description="The document's `match.version`; null when inferred")
    any_of: list[str] = Field(description="The expressions whose truth is the positive")


class RuleSetDetail(RuleSetSummary):
    """`GET /api/rule-sets/{rule_set_id}` (`rule_set_service.detail`)."""

    attached_to: list[str] = Field(description="Behaviour keys this set is attached to, sorted")
    description: str | None
    forms: dict[str, str] = Field(description="`{expression name: form}`")
    provenance: dict[str, Any] | None = Field(
        description="The document's `provenance` (the vendored schema's `Provenance`); null when "
        "authored"
    )
    vector_report: RuleSetVectorReport | None = Field(
        description="Null while `vector_status` is `passed`"
    )
    not_used_at_serving: dict[str, Any] = Field(
        description="The document's fields the rule results never read, each as stored, present "
        "only when the document has it"
    )
    match: RuleSetMatch
    document: dict[str, Any] = Field(
        description="The stored rule-set document", json_schema_extra={"$ref": RULE_SET_DOCUMENT_ID}
    )


# ── DELETE /api/rule-sets/{rule_set_id} ───────────────────────────────────────────────────────


class RuleSetDeleted(BaseModel):
    """`rule_set_service.delete_rule_set`: detached from every behaviour, then deleted."""

    id: str
    deleted: bool = Field(description="Always true: a refused delete is a refusal")
    detached_from: list[str] = Field(description="Behaviour keys it was detached from, sorted")


# ── POST /api/rule-sets/{rule_set_id}/evaluate ────────────────────────────────────────────────


class RuleSetEvaluatedBehaviour(BaseModel):
    key: str
    version: int | None = Field(description="Always null: a stored set names no behaviour version")


class RuleSetEvaluationResult(BaseModel):
    """One expression on one window (`rules.serving.RuleItem`)."""

    kind: Literal["rule"]
    signal: str = Field(description="The expression name")
    form: str
    window: RuleWindow
    fires: bool | None = Field(description="Null unless the expression completed")
    count: int
    outcome: Literal["completed", "skipped", "failed", "interrupted"]
    reason: str | None = Field(description="A published reason code; null when completed")
    status: Literal["final", "partial", "unavailable"] = Field(
        description="`unavailable` while operator code has no runner result"
    )
    # rule_sets.py:226-228 — sent only with `return_rule_positions: true`, for a `completed`
    # result whose spans were computed (never for one a code rule decided: spans None).
    spans: list[list[int]] = may_be_absent(
        description="Matched `[start, end)` code-point spans of the caller's own text"
    )


class RuleSetEvaluation(BaseModel):
    """One stored set run on the caller's text (FR-39.25). Nothing is written."""

    rule_set_id: str
    rule_set_digest: str
    rule_set_version: int
    behaviour: RuleSetEvaluatedBehaviour
    match: RuleSetMatch
    results: list[RuleSetEvaluationResult]
