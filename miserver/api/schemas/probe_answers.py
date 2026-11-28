"""Typed answers of the probe routes (044, documented-only typing; BRD-MILLM-SHAPES-001).

The routes of `miserver/api/routes/management/probes.py` declare these models in `responses=` with
`response_model=None` (SH6): FastAPI documents them, neither validates nor filters the answer, and the
answer contract (`answers_contract.py`) publishes them. Every field is `required` in the contract
because it is always sent, except a `may_be_absent()` field, whose comment cites the branch that
leaves it out. The test validator checks every answer against them in strict form (SH13).

Three parts: the probe objects (`ProbeSummary`, `ProbeIdentityDetail`), the behaviours (list, rule
signals, verdict, arm and disarm) and the runtime (status, events, scoring, parity, recalibration).
"""

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, WithJsonSchema

from miserver.api.schemas.common import may_be_absent

#: The vendored `$id` of `mistudio.probe-definition/v1` (docs/schemas/probe-definition-v1.json).
PROBE_DEFINITION_ID = (
    "https://raw.githubusercontent.com/hitsainet/miStudio/main/docs/schemas/"
    "probe-definition-v1.json"
)


class ProbeBehaviourLabel(BaseModel):
    """``behaviour_of(definition)`` (core/probe_labels.py): the definition's ``behaviour``
    block, cut to three keys. The route sends ``null`` instead when the block is absent, not an
    object or has no usable ``key``."""

    key: str = Field(description="The behaviour key, at most 64 characters")
    name: str | None = Field(description="The catalogue name; null when absent or blank")
    # 044 R2-2: `behaviour_of` sends a version only as a positive integer (`behaviour_version_of`'s
    # rule); a string or boolean in the raw document is uncatalogued, as everywhere else.
    version: int | None = Field(description="The catalogue version; null = uncatalogued")


class ProbeParityVectorRecord(BaseModel):
    """One test vector of a stored parity report (``ParityReport.as_details``,
    services/probe_parity.py:288-300; ``VectorResult`` at :52-62)."""

    index: int
    max_abs_diff: float | None = Field(description="Worst per-token difference (informational)")
    combined_diff: float | None = Field(description="Difference of the combined score")
    comparable: bool
    reason: str | None = Field(description="Why the vector could not be compared, if it could not")
    # Added by 84604c00 (2026-09-27). The report is STORED (`probes.parity`, JSONB) and served as
    # stored, so a row whose last parity ran before that commit leaves these out.
    robust_combined_diff: float | None = may_be_absent(
        description="Combined difference over reproducible positions; null = not measured"
    )
    at_risk_tokens: int = may_be_absent(description="Positions a step gate makes unreproducible")
    n_tokens: int = may_be_absent(description="Positions scored in this vector")


class ProbeParityRecord(BaseModel):
    """The probe's last parity report as stored on its row (``probes.parity``), written by
    ``ParityReport.as_details`` (services/probe_parity.py:267-307) from arming
    (probe_arming.py:700) and from ``POST /api/probes/{id}/parity`` (`check_parity`).

    ⚠ A STORED RECORD KEEPS THE SHAPE OF THE BUILD THAT WROTE IT, and it is rewritten only when
    parity runs again. The keys of the first runtime (61331608, 2026-09-27) are required; every key
    added since is ``may_be_absent``, citing the commit that added it.
    """

    passed: bool
    tolerance: float = Field(description="The document's tolerance, floored by the deployment's")
    max_abs_diff: float | None = Field(description="Worst per-token difference (informational)")
    vector_index: int | None = Field(description="The vector with the worst per-token difference")
    vectors: list[ProbeParityVectorRecord]
    tokenization_drift: dict[str, Any] = Field(
        description="Whether the document's messages re-render to its token ids (`_drift`); "
        "reported, never gated. Its keys are the parity report's own and have grown over time"
    )
    error: str | None
    # aaa946bc (2026-09-27): the gate moved to the combined score.
    score_tolerance: float = may_be_absent(description="The tolerance the gate used")
    max_combined_diff: float | None = may_be_absent()
    per_token_is_informational: bool = may_be_absent(description="Always true when present")
    # 84604c00 (2026-09-27): reproducible positions for a k-sparse JumpReLU probe.
    max_robust_combined_diff: float | None = may_be_absent()
    max_gated_diff: float | None = may_be_absent(description="The figure the gate compared")
    at_risk_tokens: int = may_be_absent()
    scored_tokens: int = may_be_absent()
    # 12ede0c5 (2026-10-03): the definition's precision beside the loaded one.
    dtype: dict[str, Any] = may_be_absent(
        description="`dtype_comparison` (probe_parity.py:330-347): recorded, loaded, matched and "
        "the two quantizations; its keys have grown over time"
    )
    # 56b15517 (2026-10-06): the slot the parity forward ran in.
    model: dict[str, Any] | None = may_be_absent(
        description="`model_summary` (probe_parity.py:310-325) of the load parity ran against"
    )
    checked_at: str | None = may_be_absent(description="ISO 8601, when parity ran")
    # f987a5fd (2026-11-16, 043 G1).
    weights_note: str | None = may_be_absent(
        description="Set when the served weights differ from the base the scores were taken on"
    )


class ProbeSummary(BaseModel):
    """One probe as the UI and the MCP tools read it: ``_probe_summary`` (routes/management/probes.py).

    Without its result totals: a route answering with one probe after a change (import, hub import,
    arm, disarm, a behaviour's arm) passes none, and ``_probe_summary`` adds them only when given
    (``if totals is not None``). The list and the detail carry them (`ProbeListItem`,
    `ProbeIdentityDetail`).
    """

    id: str
    name: str
    hf_id: str
    layer: int
    n_layers: int = Field(description="The model's layer count, as the definition recorded it")
    rule: str = Field(description="The aggregation rule")
    rule_params: dict[str, Any] = Field(
        description='`definition.aggregation.params` (free-form per rule, e.g. {"window": 32}); '
        "{} when absent"
    )
    window_thresholds: dict[str, float] = Field(
        description="Each window's own bar (`decision.windows`), only windows that placed one"
    )
    length_band_count: int = Field(description="How many length bands refine the probe's bar")
    scope: str
    basis: str
    streamable: bool
    threshold: float | None = Field(description="null = no threshold placed (ranks, never decides)")
    target_fpr: float | None
    threshold_revision: int = Field(description="miServer's own counter of the bar (the row's)")
    rung: int = Field(description="The evidence rung, 0-3")
    rung_language: str = Field(description="The rung in the suite's shared words")
    next_step: str = Field(description="What evidence would raise the rung")
    concept: str | None = Field(description="The definition's own `concept` sentence")
    behaviour: ProbeBehaviourLabel | None
    derived_from: dict[str, Any] | None = Field(
        description="`definition.provenance.derived_from` verbatim (043 FR-043.18): the lineage of "
        "a probe weight-space programming derived from another; null for a plain probe"
    )
    source_probe_id: str | None = Field(
        description="The producer's id for this probe (`definition.provenance.probe_id`)"
    )
    label_mapping: dict[str, str] = Field(
        description="`{raw_label: side}` from `definition.provenance.label_mapping`, raw; {} when "
        "absent"
    )
    load_dtype: str | None = Field(
        description="The precision the definition states it was fitted at; null = not stated"
    )
    armed: bool
    paused_reason: str | None = Field(description="Why an armed probe is not scoring, if it is not")
    parity: ProbeParityRecord | None = Field(description="The last parity report; null = never run")
    created_at: str = Field(description="ISO 8601")
    # 044 R2-5: no result totals here (see the class docstring); the strict form refuses them.


class ProbeListItem(ProbeSummary):
    """One row of `GET /api/probes`: the summary WITH its result totals. The list always passes
    `totals.get(p.id, NO_RESULTS)` to `_probe_summary` (044 R1-3: as `ProbeSummary` the three were
    may-be-absent here, so a list row without them would have passed)."""

    results_total: int = Field(description="Every stored result of this probe")
    results_fired: int = Field(description="Those whose verdict fired")
    last_result_at: str | None = Field(
        description="ISO 8601 of the newest result; null when there is none"
    )


class ProbeIdentityDetail(ProbeSummary):
    """``GET /api/probes/{id}`` (`get_probe`): the summary WITH its result totals, the stored
    definition, and the identity under the suite's names (038 FR-38.7).

    ⚠ A CROSS-REPO CONTRACT: miWorkbench 039's `test_miserver_probe_identity_names.py` reads these
    names from this app's served schema. The suite spells them the same everywhere
    (cross-document review 2026-10-23). The class name is kept for the same reason.
    """

    # Always sent here: the route passes `totals.get(probe.id, NO_RESULTS)` (`get_probe`).
    results_total: int = Field(description="Every stored result of this probe")
    results_fired: int = Field(description="Those whose verdict fired")
    last_result_at: str | None = Field(
        description="ISO 8601 of the newest result; null when there is none"
    )
    # NOT a `$ref` to the vendored schema (finding F-044-1, 2026-11-25): the import checks the
    # document with miServer's own model (`api/schemas/probe.py`), which is looser than the vendored
    # `probe-definition-v1.json` (evaluations without `dataset`, extra provenance keys and one test
    # vector all import), so a stored definition can fail the vendored schema. A map, described.
    definition: dict[str, Any] = Field(
        description="The stored `mistudio.probe-definition/v1` document, as imported (raw; its "
        "schema is " + PROBE_DEFINITION_ID + ", which the import does not enforce)",
    )
    detector_digest: str | None = Field(None, description="64 hex; null = not digestible")
    detector_digest_source: Literal["producer", "computed", "unknown"] = "unknown"
    identity_algorithm: str | None = Field(None, description="mistudio.probe-identity/1")
    decision_digest: str | None = Field(None, description="64 hex, over the current decision")
    decision_revision: int | None = Field(None, description="miWorkbench's revision, when stated")
    threshold_revision: int = Field(1, description="miServer's own counter of the bar")


class ProbeDeleted(BaseModel):
    """``DELETE /api/probes/{probe_id}`` (`delete_probe`): ``{"deleted": probe_id}``."""

    deleted: str = Field(description="The deleted probe's id")


# ─── the behaviours ──────────────────────────────────────────────────────────────────────────────

#: The vendored `mistudio.behaviour-verdict/v1` schema (`docs/schemas/behaviour-verdict-v1.json`).
BEHAVIOUR_VERDICT_ID = "https://hitsai.net/schemas/mistudio/behaviour-verdict-v1.json"

#: A stored verdict definition, declared as a reference to the vendored schema. ⚠ The reference sits
#: on the dict MEMBER, so `VerdictDefinition | None` generates `anyOf: [{$ref}, {type: null}]`.
#: `Field(json_schema_extra={"$ref": ...})` on a nullable field would put `$ref` BESIDE the `anyOf`,
#: and under Draft 2020-12 a sibling `$ref` is conjunctive: a null value fails the referenced
#: schema's `type: object` (checked with `Draft202012Validator`; see NOTES.md).
VerdictDefinition = Annotated[dict[str, Any], WithJsonSchema({"$ref": BEHAVIOUR_VERDICT_ID})]


# ── the behaviour summary (`probe_behaviours.summarise`) ─────────────────────────────────────


class BehaviourVersionMembers(BaseModel):
    """One behaviour version's members (`probe_behaviours._versions`)."""

    version: int | None = Field(..., description="The behaviour version; null = uncatalogued")
    probe_ids: list[str] = Field(..., description="Every imported probe of this version, sorted")
    imported: int
    armed: int = Field(..., description="How many of them are armed IN THE REGISTRY")
    missing_probe_ids: list[str] = Field(
        ..., description="The members not armed; empty when none of the version is armed"
    )


class BehaviourArmedSetState(BaseModel):
    """The armed version's partial state ("partial: n of m armed", FR-39.4)."""

    version: int | None
    armed_members: int
    total_members: int
    complete: bool
    missing_probe_ids: list[str]
    text: str | None = Field(..., description="`partial: n of m armed`; null when complete")


class BehaviourStoppedMembers(BaseModel):
    """The members that were armed when they stopped without the operator asking (FR-39.5)."""

    probe_ids: list[str]
    reasons: list[str] = Field(..., description="The involuntary `paused_reason`s, sorted")


class BehaviourAttachedRuleSetRef(BaseModel):
    """One rule set attached to the behaviour, from the compiled rule registry (`attachments_of`)."""

    rule_set_id: str
    rule_set_digest: str
    rule_set_version: int
    vector_status: Literal["passed", "vectors_failed"] = Field(
        ..., description="Derived from the registry's `vectors_failed` flag (two values only)"
    )


class BehaviourOverview(BaseModel):
    """One behaviour key: its probes per version, how many are armed (from the registry), the
    partial state, "armed before unload / restart" and the attached rule sets (FR-39.4, FR-39.5)."""

    key: str
    imported: int
    armed: int
    armed_probe_ids: list[str]
    versions: list[BehaviourVersionMembers]
    set: BehaviourArmedSetState | None = Field(
        ..., description="The first armed version's state; null when nothing is armed"
    )
    armed_before_unload: BehaviourStoppedMembers | None = Field(
        ..., description="Null when no unarmed member stopped involuntarily"
    )
    attached_rule_sets: list[BehaviourAttachedRuleSetRef] = Field(
        ..., description="Filled by the list route; always empty in the disarm answer's summary"
    )


# ── rule-signal attachment (`rule_set_service.attachment_view`) ──────────────────────────────


class RuleSignalAttachmentRuleSet(BaseModel):
    """One attached rule set as `rule_set_service.summary(row)` gives it, called WITHOUT
    attachments, so the `attached_to` key the rule-set routes may add is never sent here."""

    id: str
    name: str
    version: int
    behaviour_key: str
    origin: str = Field(..., description="`authored` or `derived`")
    digest: str
    vector_status: str = Field(..., description="`passed` or `vectors_failed` (stored column)")
    signals: int = Field(..., description="How many signals the document declares")
    expressions: list[str] = Field(..., description="The document's expression names, sorted")
    imported_at: str | None
    checked_at: str | None
    code_modules: list[str] = Field(..., description="Hashes of the code documents held, sorted")
    code_modules_missing: list[str] = Field(
        ..., description="Hashes a code rule names whose document is not held"
    )


class RuleSignalAttachmentChange(BaseModel):
    """One row of the attachment's history (newest first, at most 20)."""

    changed_at: str | None
    action: str = Field(..., description="`attach`, `clear`, `rule_set_deleted`, ...")
    rule_set_ids: list[str]
    signals: dict[str, list[str]] | None = Field(
        ..., description="`{rule_set_id: [expression name, ...]}`; null = every signal"
    )
    by: str | None
    reason: str | None


class RuleSignalAttachmentView(BaseModel):
    """The rule sets attached to one behaviour, and the attachment's history (FR-39.22)."""

    behaviour: str
    rule_set_ids: list[str] = Field(..., description="The attached ids, in attach order")
    signals: dict[str, list[str]] | None = Field(
        ..., description="The narrowing per rule set; null = every signal of every set"
    )
    rule_sets: list[RuleSignalAttachmentRuleSet] = Field(
        ..., description="The attached sets that are still imported, in attach order"
    )
    history: list[RuleSignalAttachmentChange]


# ── behaviour verdict (`behaviour_verdict_store.view`) ───────────────────────────────────────


class BehaviourVerdictParityReport(BaseModel):
    """The parity layers the attach checked (counts and codes)."""

    members: str = Field(..., description="Always `armed_digests_match` when an attach succeeds")
    combiner_vectors: int = Field(..., description="How many combiner vectors were recomputed")
    end_to_end: str = Field(..., description="The definition's `end_to_end_status`")


class BehaviourVerdictDecisionDrift(BaseModel):
    """A member armed under another decision than the definition records (`decision_changed`)."""

    detector_digest: str
    probe_id: str
    definition: str = Field(..., description="The decision digest the definition records")
    armed: str | None = Field(..., description="The armed probe's decision digest, if any")


class BehaviourVerdictMembersReport(BaseModel):
    """`behaviour_verdict_store.check_members`: the armed members and any decision drift."""

    armed_probe_ids: list[str]
    decision_changed: list[BehaviourVerdictDecisionDrift]


class BehaviourVerdictRulesReport(BaseModel):
    """`behaviour_verdict_store.check_rules` for a `rules_or_*` definition."""

    rule_set_id: str
    match_source: str = Field(..., description="The definition's `rules.match.source`")
    match_any_of: list[str]
    pending_code: list[str] = Field(
        ..., description="Match expressions whose code-rule document is not held yet"
    )


class BehaviourVerdictAttachReport(BaseModel):
    """What the attach found, as stored on the row (`attach_report`)."""

    parity: BehaviourVerdictParityReport
    members: BehaviourVerdictMembersReport
    rules: BehaviourVerdictRulesReport | None = Field(
        ..., description="Null for a definition without a rule side"
    )


class BehaviourVerdictChange(BaseModel):
    """One row of the verdict attachment's history (newest first, at most 20)."""

    changed_at: str | None
    action: str = Field(..., description="`attach` or `clear`")
    definition_sha256: str | None
    report_id: str | None
    by: str | None
    reason: str | None


class BehaviourVerdictView(BaseModel):
    """The behaviour's attached verdict definition, whether it is serving, what the attach found,
    and the attachment's history (ENH-006 G19). Every key is sent; the attachment's fields are null
    when nothing is attached (after a clear, or before any attach)."""

    behaviour: str
    attached: bool
    definition: VerdictDefinition | None = Field(
        ..., description="The stored definition as received; null when none is attached"
    )
    definition_sha256: str | None
    serving: bool = Field(
        ..., description="True when the serving registry holds this exact definition"
    )
    attach_report: BehaviourVerdictAttachReport | None
    attached_at: str | None
    attached_by: str | None
    history: list[BehaviourVerdictChange]


# ── arm / disarm ─────────────────────────────────────────────────────────────────────────────


class BehaviourSetRef(BaseModel):
    """The behaviour and the one version the selection shares (`probe_selection.Selection`)."""

    key: str
    version: int | None = Field(..., description="Null = the uncatalogued version")


class BehaviourArmGateTimings(BaseModel):
    """How long one probe's parity and cost gates took (`ArmPlan.gate_ms`), in ms."""

    parity: float
    cost: float


class BehaviourArmProbeCost(BaseModel):
    """One passed probe's measured cost (`probe_arming._plan_cost`). The cost fields are read with
    `getattr(cost, ..., None)`, so the code allows null although a measured plan always has them."""

    decode: float | None
    prefill: float | None
    layer_fixed_decode: float | None
    layer_fixed_prefill: float | None
    first_on_layer: bool
    layer: int
    device: str | None


class BehaviourArmProbeOutcome(BaseModel):
    """One probe's gate outcome (`probe_arming._outcome`)."""

    probe_id: str
    gate: str = Field(..., description="`all` when passed; the refusing gate, or `error`")
    outcome: Literal["passed", "refused"]
    # probe_arming.py:497 `_outcome` adds `code` only when given: refused outcomes only
    # (`check_all`, probe_arming.py:951-952); a passed outcome leaves it out.
    code: str = may_be_absent(description="The refusal code (refused outcomes only)")
    # probe_arming.py:499-502: `cost` and `gate_ms` only on a passed outcome
    # (`check_all`, probe_arming.py:956-957); a refused outcome leaves both out.
    cost: BehaviourArmProbeCost = may_be_absent(
        description="The measured cost (passed outcomes only)"
    )
    gate_ms: BehaviourArmGateTimings = may_be_absent(
        description="Gate timings in ms (passed outcomes only)"
    )


class BehaviourArmProjection(BaseModel):
    """The armed set's projected cost per pass (`probe_cost.project`), in ms."""

    decode: float
    prefill: float = Field(..., description="Per 1k prefill tokens")


class BehaviourArmBudget(BaseModel):
    """The budget the set is admitted against (`EffectiveLimits.budget()`)."""

    decode_ms_per_pass: float
    prefill_ms_per_1k: float
    headroom: float


class BehaviourArmRefusal(BaseModel):
    """What a dry run would have been refused with (FR-39.6)."""

    code: str
    details: dict[str, Any] = Field(
        ...,
        description="The refusal's details; free-form because each refusal class names its "
        "own figures (limits, budgets, the per-probe outcomes under `probes`)",
    )


class BehaviourArmBudgetSources(BaseModel):
    """Where each budget half came from (`EffectiveLimits.sources`)."""

    decode: str | None
    prefill: str | None


class BehaviourArmLimits(BaseModel):
    """The limits the arm ran under (built in the route)."""

    max_behaviours: None = Field(
        ..., description="Always null since D83 (no behaviour count); kept for older readers"
    )
    source: str | None = Field(..., description="`sources.max_behaviours`: why it is not enforced")
    budget: BehaviourArmBudget
    budget_sources: BehaviourArmBudgetSources
    stale: bool = Field(
        ..., description="The approval rests on a measurement under another fingerprint"
    )
    max_armed: int = Field(..., description="The armed-probe backstop")


class BehaviourArmOutcome(BaseModel):
    """`POST /api/probes/behaviours/{key}/arm`: the set armed all or nothing, or dry-run (FR-39.2,
    FR-39.6). Built from `ProbeArmingService.arm_set` plus the route's `armed`, `set`, `limits` and
    `skipped_identity`."""

    behaviour: BehaviourSetRef
    already_armed: list[str] = Field(..., description="Members armed before this call (untouched)")
    dry_run: bool
    probes: list[BehaviourArmProbeOutcome] = Field(
        ..., description="Every remaining member's gate outcome"
    )
    armed: list[ProbeSummary] = Field(
        ..., description="The probes this call armed; empty on a dry run"
    )
    projected: BehaviourArmProjection
    budget: BehaviourArmBudget
    # probe_arming.py:995-997 and 1014-1016: only a dry run sends `would_refuse`, and the second
    # dry-run branch only when a gate would refuse; a real arm (probe_arming.py:1037) leaves it out.
    would_refuse: BehaviourArmRefusal = may_be_absent(
        description="Dry run only: the refusal the arm would have met"
    )
    # probe_arming.py:1037-1039: only the real arm sends `gate_ms`; both dry-run returns
    # (probe_arming.py:995, 1014) leave it out.
    gate_ms: dict[str, BehaviourArmGateTimings] = may_be_absent(
        description="Real arm only: gate timings per newly armed probe id"
    )
    set: BehaviourArmedSetState | None = Field(
        ..., description="The partial state after the call; null when nothing of the key is armed"
    )
    limits: BehaviourArmLimits
    skipped_identity: list[str] = Field(
        ..., description="Probes of the key fitted on another model (not members here)"
    )


class BehaviourDisarmOutcome(BaseModel):
    """`POST /api/probes/behaviours/{key}/disarm` (FR-39.3)."""

    behaviour: BehaviourSetRef
    disarmed: list[str] = Field(..., description="The probe ids disarmed, sorted")
    summary: BehaviourOverview = Field(
        ..., description="The behaviour after the disarm (its `attached_rule_sets` is empty)"
    )


# ─── the runtime ─────────────────────────────────────────────────────────────────────────────────

# ── the `mi.probe.result/v1` keys (`miserver/api/schemas/probe_result.py`, through `wire`) ────


class ResultProducer(BaseModel):
    """`producer`: who produced the result."""

    name: str
    version: str


class ResultBehaviourRef(BaseModel):
    """`behaviour`: the behaviour the probe's DEFINITION names (FR-38.1)."""

    key: str
    version: int | None


class ResultAttachedRuleSet(BaseModel):
    """One rule set attached to the behaviour, inside the result's `set` block."""

    rule_set_id: str
    rule_set_digest: str


class ResultSetBlock(BaseModel):
    """`set`: the behaviour-set slot (OQ-38.3 (a), D31)."""

    armed_members: int
    total_members: int
    complete: bool
    missing_probe_ids: list[str]
    members_digest: str
    attached_rule_sets: list[ResultAttachedRuleSet]


class ResultSteeringRef(BaseModel):
    """`steering_ref`: what steered the residual this result read (`unknown` when unreported)."""

    kind: str
    id: str | None


class ResultDecision(BaseModel):
    """`decision`: the bar's calibration and whether the verdict is provisional."""

    calibration_id: str | None
    target_fpr: float | None
    provisional: bool


class ResultMemberRef(BaseModel):
    """A member of a behaviour verdict: its detector digest, and miServer's probe id when armed."""

    detector_digest: str
    probe_id: str | None


class ResultVerdictRules(BaseModel):
    """The rule side of a behaviour verdict (after the vetoes)."""

    match: bool | None
    veto: bool | None
    unknown: bool
    reason: str | None
    rule_set_digest: str | None


class ResultVerdictCandidate(BaseModel):
    """The behaviour verdict's combination rule and the value it computed."""

    name: str
    k: int | None
    value: float | None
    threshold: float | None


class _ResultCommonKeys(BaseModel):
    """The keys every result kind carries (`_Common`), WITHOUT the two recorded-only keys
    (`persistence`, `source_event_id`), which `wire` drops while they are unset."""

    model_config = ConfigDict(protected_namespaces=())

    schema_: str = Field(alias="schema", description="`mi.probe.result/v1`")
    result_id: str
    event_time: str = Field(description="RFC 3339, UTC")
    event_time_source: str = Field(description="`finalized` or `inserted` (a migrated row)")
    origin: str = Field(description="`live`, `batch` or `stateless`")
    producer: ResultProducer
    migrated: bool
    request_id: str | None
    trace_id: str | None
    span_id: str | None
    batch_id: str | None
    batch_line: int | None
    session_id: str | None
    turn_index: int | None
    behaviour: ResultBehaviourRef | None
    set_: ResultSetBlock | None = Field(alias="set")
    outcome: str = Field(description="`completed`, `skipped`, `failed` or `interrupted`")
    reason: str | None = Field(description="A published reason code; null when completed")
    window: str
    status: str = Field(description="`final`, `partial` or `unavailable`")


class _ProbeResultKeys(_ResultCommonKeys):
    """`kind: "probe"` — one armed probe over one window of one request (`ProbeResult`)."""

    kind: Literal["probe"]
    choice_index: int | None
    input_index: int | None
    probe_id: str
    probe_name: str | None
    detector_digest: str = Field(description="64 hex, or `unknown`")
    detector_digest_source: str
    identity_algorithm: str | None
    definition_schema: str
    target: str
    score_semantics: str
    layer: int | None
    hook: str
    basis: str
    rule: str
    served_model: str = Field(description="The model whose residual was read, or `unknown`")
    model_revision: str
    template_sha256: str
    precision: str
    engine: str
    execution_mode: str
    steering_ref: ResultSteeringRef
    definition_scope: str
    n_scored_tokens: int
    scored: bool
    score: float | None
    probability: None = Field(description="Always null in v1")
    calibration_ref: None = Field(description="Always null in v1")
    threshold: float | None
    verdict: bool | None = Field(description="null = the probe said nothing (no bar, or unscored)")
    fires: bool | None = Field(description="v1 legacy duplicate of `verdict`")
    comparator: str
    decision_digest: str = Field(description="64 hex, or `unknown`")
    decision_revision: int | None
    threshold_revision: int
    decision: ResultDecision
    rung: int
    rung_language: str
    parity_status: str
    model_load_id: str
    evaluation_ms: float | None
    completed_at: str


class _RuleResultKeys(_ResultCommonKeys):
    """`kind: "rule"` — one rule signal over one window (`RuleResult`, 039 S2). No text."""

    kind: Literal["rule"]
    rule_set_id: str
    rule_set_digest: str
    rule_set_version: int
    signal: str
    form: str
    fires: bool | None
    count: int


class _BehaviourResultKeys(_ResultCommonKeys):
    """`kind: "behaviour"` — one verdict for a behaviour's probe stack (`BehaviourResult`, G19)."""

    kind: Literal["behaviour"]
    input_index: int | None
    fires: bool | None = Field(description='null = not proven either way, never "clean"')
    decided_by: str
    members_fired: list[ResultMemberRef]
    members_vetoed: list[ResultMemberRef]
    members_missing: list[ResultMemberRef]
    rules: ResultVerdictRules | None
    candidate: ResultVerdictCandidate
    provisional: bool
    provisional_reasons: list[str]
    definition_schema: str
    definition_sha256: str
    members_digest: str
    report_id: str | None


class _RecordedKeys(BaseModel):
    """The two keys of a RECORDED copy, both set on a stored row (`stored_result` writes
    `persistence: persisted`; `recorded_from_row` injects the row's id)."""

    persistence: str = Field(description="`persisted` for a stored row")
    source_event_id: int = Field(description="The row's own id")


# ── GET /api/probes/events, GET /api/probes/events/{event_id} ───────────────────────────────


class RecordedProbeEvent(_RecordedKeys, _ProbeResultKeys):
    """One stored probe verdict, WITHOUT its context (`probe_event_service.event_summary`): the
    legacy keys, then the stored `mi.probe.result/v1` object merged over them (its keys win)."""

    id: int
    not_scored_reason: str | None
    top_positions: list[int] | None = Field(description="Token positions of the top scores")
    summary: str | None
    provisional: bool
    created_at: str


class RecordedProbeEventDetail(RecordedProbeEvent):
    """One probe event WITH its decoded context window — the only route that serves it."""

    context_text: str | None = Field(description="The decoded window around the top position")
    context_token_ids: list[int] | None


class RecordedRuleEvent(_RecordedKeys, _RuleResultKeys):
    """One stored rule result (`probe_event_service.rule_summary`): the legacy keys, then the
    stored object merged over them. No text: a count, never a value."""

    id: int
    created_at: str


class ProbeEventsCleared(BaseModel):
    """`DELETE /api/probes/events`: how many probe events were removed."""

    removed: int


# ── GET /api/probes/status ──────────────────────────────────────────────────────────────────


class ProbeCostMeasurement(BaseModel):
    """One armed probe's measured cost per pass on its layer's card (`ProbeCost.as_dict`)."""

    decode_ms: float
    prefill_ms_per_1k: float
    layer_fixed_ms: float = Field(description="The layer's shared part, charged once per layer")
    layer_fixed_prefill_ms_per_1k: float
    layer: int
    device: str
    k: int = Field(description="Timed calls behind the measurement")
    measured_at: str


class ArmedProbeStatus(BaseModel):
    """One row claiming `armed`, with what the LIVE registry says about it."""

    id: str
    name: str
    layer: int
    rule: str
    rung: int
    rung_language: str
    next_step: str
    streamable: bool
    basis: str
    paused_reason: str | None = Field(description="Why it is not scoring; null when it is")
    hook_installed: bool = Field(description="Whether THIS process has its hook installed")
    windows: list[str] | None = Field(description="The windows scored; null = not armed here")
    threshold: float | None = Field(description="The bar in force; null when not armed here")
    threshold_revision: int | None = Field(description="Null when not armed here")
    row_threshold: float | None
    row_threshold_revision: int
    armed_threshold_revision: int | None
    threshold_disagreement: str | None = Field(
        description="A sentence when the live bar is not the row's; null when they agree"
    )
    cost: ProbeCostMeasurement | None = Field(description="Null when not armed or not measured")


class ArmedBeforeUnloadBehaviour(BehaviourStoppedMembers):
    """A behaviour none of whose probes is armed, with members that stopped involuntarily."""

    key: str


class ProbeCatalogueCounts(BaseModel):
    """The catalogue as counts (FR-39.15)."""

    imported: int
    behaviours: int = Field(description="Distinct behaviour keys")


class ProbeLimitsApproval(BaseModel):
    """Who approved the limits, when, on which measurement (`probe_limits.approval_of`)."""

    source: str
    superseded: bool = Field(description="A pre-D83 approval in ms, which no longer binds")
    set_by: str | None
    approved_at: str | None
    measurement_id: int | None
    newer_recommendation_available: bool


class ProbeStatusLimits(BaseModel):
    """The limits in force for the resident model, else the globals."""

    max_behaviours: None = Field(description="Always null since D83 (no behaviour count)")
    max_behaviours_source: str = Field(description="Why the behaviour count is not enforced")
    slowdown_budget_pct: float
    max_armed: int
    stale: bool = Field(description="The approval rests on another fingerprint (still in force)")
    approval: ProbeLimitsApproval | None


class ProbeBudgetSources(BaseModel):
    """Where each limit came from (`EffectiveLimits.sources`: four keys on both builders)."""

    max_armed: str
    decode: str
    prefill: str
    max_behaviours: str


class ProjectedProbeCost(BehaviourArmProjection):
    """The armed set's projected cost per pass, and whether it still fits the budget."""

    over_budget: bool


class SocketDropCounts(BaseModel):
    """What the socket throttle dropped: whole requests, and the results they carried."""

    requests: int
    results: int


class ServedWeightsFact(BaseModel):
    """Whether the resident weights are the base the probes were fitted on (043 FR-043.5)."""

    name: str | None
    origin: str = Field(description="`model`, `checkpoint`, `instantiated`, or `none`")
    commits_since_load: int | None = Field(description="Null when the counter cannot answer")
    differs_from_base: bool


class ProbeStatus(BaseModel):
    """What every armed probe is doing, and why any of them is not scoring (FR-24.9)."""

    armed: list[ArmedProbeStatus] = Field(description="Every row claiming `armed`")
    armed_count: int = Field(description="Probes actually hooked and able to score")
    armed_rows: int = Field(description="Rows claiming `armed`")
    stale_armed: list[str] = Field(description="Rows claiming `armed` with no hook installed")
    max_armed: int
    imported_count: int
    behaviours: list[BehaviourOverview] = Field(description="Each armed behaviour")
    armed_before_unload: list[ArmedBeforeUnloadBehaviour]
    catalogue: ProbeCatalogueCounts
    limits: ProbeStatusLimits
    paused_reasons: list[str]
    last_request_overhead_ms: float | None
    overhead_warn_threshold_ms: float
    last_request_n_passes: int
    last_overhead_ms_per_pass: float | None
    overhead_warn_threshold_ms_per_pass: float
    events_recorded: int
    socket_events_dropped: SocketDropCounts
    force_serial: bool
    session_resolve_failures: int
    session_write_failures: int
    retention_failures: int
    last_retention_pass_at: str | None
    transcripts: str = Field(description="`off` or `on` (PROBE_TRANSCRIPTS)")
    budget: BehaviourArmBudget
    budget_sources: ProbeBudgetSources | None = Field(
        description="Null only when the service has no session to read the limits with"
    )
    projected: ProjectedProbeCost
    served_weights: ServedWeightsFact


# ── POST /api/probes/score ──────────────────────────────────────────────────────────────────


class LoadedModelSummary(BaseModel):
    """The load a parity report or a scoring run read (`probe_parity.model_summary`)."""

    hf_id: str | None
    revision: str | None
    dtype: str | None
    quantization: str | None
    served_weights: ServedWeightsFact


class StoredParityStatus(BaseModel):
    """A probe's STORED parity, reported and never required (`probe_scoring.parity_status`)."""

    status: str = Field(description="`passed`, `failed` or `never_run`")
    checked_against: dict[str, Any] | str | None = Field(
        description="The stored report's model block; `unknown` for a report that predates it; "
        "null when never run. Free-form: a stored block keeps the shape of its date"
    )
    checked_at: str | None


class ScoredProbe(BaseModel):
    """One probe the request scored with."""

    probe_id: str
    name: str
    layer: int
    parity: StoredParityStatus


class ScoreSkippedProbe(BaseModel):
    """A probe skipped because it cannot score on the loaded model (only when ids were omitted)."""

    probe_id: str
    code: str
    reason: str


class ScoreInputError(BaseModel):
    """Why one input was not scored (`TOKENIZATION_FAILED`, `MODEL_CHANGED`)."""

    code: str
    message: str


class StatelessProbeVerdict(_ProbeResultKeys):
    """One probe's verdict on one input: `verdict_payload`, then `surface_entry` (the legacy
    projection and the stateless `mi.probe.result/v1` object) merged over it. Nothing is stored,
    so there is no `source_event_id`; `persistence` is `not_persisted`."""

    persistence: str = Field(description="`not_persisted`")
    name: str | None = Field(description="The probe's name as the runtime knows it")
    provisional: bool
    not_scored_reason: str | None


class StatelessRuleResult(_RuleResultKeys):
    """One rule signal on one input (039 S2): the stateless object alone, never stored."""

    # probe_scoring.py:760-761: added only with `return_rule_positions` on a completed result
    # that reports spans (a code rule reports none).
    spans: list[list[int]] = may_be_absent(description="Match spans, as [start, end] pairs")


class StatelessBehaviourVerdict(_BehaviourResultKeys):
    """One behaviour verdict on one input (G19): the stateless object, never stored."""

    persistence: str = Field(description="`not_persisted`")


class ScoredInput(BaseModel):
    """One input's tokens, verdicts and (when it was not scored) its error."""

    index: int
    input_kind: str = Field(description="`text`, `messages` or `token_ids`")
    n_tokens: int
    prompt_tokens: int | None
    token_ids: list[int] | None = Field(description="Only with `return_token_ids`; else null")
    verdicts: list[StatelessProbeVerdict] = Field(description="Empty when the input failed")
    error: ScoreInputError | None
    # probe_scoring.py:553-554: only on a scored input with a behaviour definition attached.
    behaviours: list[StatelessBehaviourVerdict] = may_be_absent(
        description="Behaviour verdicts; absent when no definition is attached"
    )
    # probe_scoring.py:555-560: only on a scored input of a request naming `behaviours`.
    rules: list[StatelessRuleResult] = may_be_absent(
        description="Rule signals of the requested behaviours"
    )
    # probe_scoring.py:561-562: only when the rule text was decoded from `token_ids`.
    rules_decoded_from_token_ids: bool = may_be_absent(
        description="True when the rule text was decoded from token ids"
    )


class ScoreExportSummary(BaseModel):
    """The stateless export gate's outcome (FR-38.17)."""

    requested: bool
    permitted: bool
    records_queued: int


class ProbeScoreResults(BaseModel):
    """`POST /api/probes/score`: stored inputs scored with imported probes; nothing persisted."""

    model: LoadedModelSummary
    probes: list[ScoredProbe]
    skipped: list[ScoreSkippedProbe]
    results: list[ScoredInput]
    # probe_scoring.py:452-454: only when the request body names `export_results`.
    export: ScoreExportSummary = may_be_absent(
        description="Absent unless the request sent `export_results`"
    )


# ── POST /api/probes/{probe_id}/parity ──────────────────────────────────────────────────────


class ParityVectorDetail(BaseModel):
    """One test vector's comparison."""

    index: int
    max_abs_diff: float | None
    combined_diff: float | None
    robust_combined_diff: float | None = Field(
        description="Over the reproducible positions; null when not recombinable or not measured"
    )
    at_risk_tokens: int
    n_tokens: int
    comparable: bool
    reason: str | None


class TokenizationDrift(BaseModel):
    """Whether the document's `messages` re-render to its recorded ids. Informational only."""

    checked: int
    messages_reproduce_token_ids: int
    truncated_vectors: list[int]
    mismatched: int
    mismatched_vectors: list[int] = Field(description="The first ten")
    render: dict[str, Any] | None = Field(description="The definition's `render` block")
    rendered_with: str = Field(description="`served_form` or `no_generation_prompt`")
    note: str
    # probe_parity.py:673-678: only when some vector was truncated keeping the tail.
    truncation_note: str = may_be_absent()
    # probe_parity.py:679-689: absent when the render block records the served form.
    render_note: str = may_be_absent()


class TokenizationDriftNotChecked(BaseModel):
    """`{}`: the check ended before the vectors were compared (a quantization mismatch, no test
    vectors) or no tokenizer was at hand (`probe_parity.py:407-422`, `:523`)."""


class ParityDtypeComparison(BaseModel):
    """The definition's stated precision beside this server's (`dtype_comparison`)."""

    recorded: str | None
    loaded: str | None
    matched: bool | None = Field(description="Null when either side is unknown")
    recorded_quantization: str | None
    loaded_quantization: str | None


class ParityDetails(BaseModel):
    """`ParityReport.as_details()`: whether this build reproduces miWorkbench's recorded scores."""

    passed: bool
    tolerance: float = Field(
        description="The document's tolerance, floored by the deployment's (PROBE_PARITY_TOLERANCE)"
    )
    score_tolerance: float = Field(description="What the gate used")
    max_combined_diff: float | None
    max_robust_combined_diff: float | None
    max_gated_diff: float | None = Field(description="The figure the gate compared")
    at_risk_tokens: int
    scored_tokens: int
    max_abs_diff: float | None = Field(description="Per-token; informational, never gated")
    per_token_is_informational: bool
    vector_index: int | None = Field(description="The vector with the largest `max_abs_diff`")
    vectors: list[ParityVectorDetail]
    tokenization_drift: TokenizationDrift | TokenizationDriftNotChecked
    dtype: ParityDtypeComparison
    error: str | None = Field(description="Why the vectors could not be compared at all")
    model: LoadedModelSummary
    checked_at: str
    weights_note: str | None = Field(
        description="Set when the check failed on weights that differ from the base"
    )


# ── POST /api/probes/{probe_id}/recalibrate ─────────────────────────────────────────────────


class RecalibrationOutcome(BaseModel):
    """The moved bar, the live registry's refresh, and which windows gained or lost a bar."""

    id: str
    threshold: float | None
    target_fpr: float | None
    threshold_revision: int
    previous_threshold: float | None
    previous_revision: int
    armed: bool
    registry_updated: bool = Field(description="Whether the live armed probe was refreshed")
    stale_armed: bool = Field(description="The row says armed but nothing is live")
    length_bands: int = Field(description="How many length bands the new decision carries")
    windows_newly_calibrated: list[str]
    windows_no_longer_calibrated: list[str]
