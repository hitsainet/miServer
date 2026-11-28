"""Typed answers of the per-model probe capacity and limits routes (044, documented-only typing;
operator 2026-10-21; 039 FR-39.8, FR-39.9, TD39-15, TD39-31; ENH-006 D83).

The routes (`miserver/api/routes/management/model_probe_capacity.py`):

* `GET  /api/models/{model_id}/probe-capacity`         -> `ApiResponse[ProbeCapacity]`
* `GET  /api/models/{model_id}/probe-limits/recommendation`
                                                      -> `ApiResponse[ProbeLimitRecommendation]`
* `PUT  /api/models/{model_id}/probe-limits` (gated)   -> `ApiResponse[ProbeCapacity]`
* `POST /api/models/{model_id}/probe-capacity/measure` (gated) -> `ApiResponse[ProbeCapacity]`

`ProbeCapacity` is `probe_limits.capacity` (plus `preview` on the GET when a set is named);
`ProbeLimitRecommendation` is `probe_limit_recommendation.recommend`, or the route's own answer for
a model with no usable measurement. The two gated routes answer an agent with the gate's pending
202, declared separately (`gated_responses`).
"""

from typing import Any, Literal

from pydantic import BaseModel, Field

from miserver.api.schemas.common import may_be_absent

#: `probe_limits.UNITS`: a budget half in milliseconds or as a percentage of the measured step time.
BudgetUnit = Literal["ms", "pct"]
#: `probe_limits.SOURCES`: who set a model's stored limits.
LimitsSource = Literal["recommended-approved", "operator"]
#: What refuses one more probe or set (`probe_limit_recommendation._fits`).
CapacityBinding = Literal["max_armed", "decode", "prefill"]


# ── shared parts ──────────────────────────────────────────────────────────────────────────────


class ProbeCostLine(BaseModel):
    """One cost line: a probe's own part and its layer's shared part, per half (ms)."""

    decode_own_ms: float
    decode_fixed_ms: float
    prefill_own_ms_per_1k: float
    prefill_fixed_ms_per_1k: float
    # probe_capacity_measure.py:52 `_line` writes it; the record seeds of migration
    # 031_probe_capacity.py:39 (and a measurement stored before 039) have no `layer`.
    layer: int = may_be_absent(description="The layer the line was measured at")
    # Only the 031 seed's k-sparse line carries `k` (031_probe_capacity.py:45); a dense line never.
    k: int = may_be_absent(description="Active features of a k-sparse line")


class CapacityCostTotals(BaseModel):
    """A projected cost per pass (`probe_cost.project`): ms per decode pass, ms per 1,000 prompt
    tokens."""

    decode: float
    prefill: float


class CapacityBudget(BaseModel):
    """The budget admission checks against (`EffectiveLimits.budget`)."""

    decode_ms_per_pass: float
    prefill_ms_per_1k: float
    headroom: float = Field(description="The share of each budget arming may fill")


# ── GET …/probe-capacity: measured, configured, effective, implied, armed ─────────────────────


class ProbeCapacityMeasured(BaseModel):
    """The newest measurement for the model's Hub repo and precision (`measurement_dict`)."""

    id: int
    repo_id: str
    quantization: str
    device: str = Field(description="The card label(s), `+`-joined, by UUID")
    decode_ms_per_token: float | None
    prefill_ms_per_1k: float | None
    dense: ProbeCostLine | None
    sparse: ProbeCostLine | None = Field(description="Null unless a record seeded a k-sparse line")
    method: str = Field(description="How each figure was obtained, in words")
    source: str | None
    measured_at: str = Field(description="ISO 8601")
    lines: dict[str, ProbeCostLine] | None = Field(
        description="`{rule: line}` (039 TD39-3); null for a measurement taken before 039"
    )


class ProbeBudgetHalf(BaseModel):
    value: float
    unit: BudgetUnit


class ProbeLimitFingerprint(BaseModel):
    """What an approved recommendation's measurement was taken under (TD39-15)."""

    repo_id: str | None
    quantization: str | None
    dtype: str | None = Field(description="Null when the model was not resident at approval")
    devices: list[str]


class ProbeLimitsConfigured(BaseModel):
    """This model's stored limits (`probe_limits.limits_dict`); every value null = the default."""

    max_armed: int | None
    decode_budget: ProbeBudgetHalf | None
    prefill_budget: ProbeBudgetHalf | None
    reason: str | None
    # probe_limits.py:223-226: the no-row branch (nothing configured yet) leaves `updated_at` out.
    updated_at: str = may_be_absent(description="ISO 8601")
    max_behaviours: int | None = Field(description="Stored for older readers; binds nothing (D83)")
    source: LimitsSource | None
    set_by: str | None
    approved_at: str | None = Field(description="ISO 8601")
    approved_measurement_id: int | None
    fingerprint: ProbeLimitFingerprint | None


class ProbeLimitSources(BaseModel):
    """Where each effective value came from, in words."""

    max_armed: str
    decode: str
    prefill: str
    max_behaviours: str


class ProbeLimitApproval(BaseModel):
    """Who approved the stored limits, when, on which measurement (`probe_limits.approval_of`)."""

    source: LimitsSource
    superseded: bool = Field(description="An `ms` approval from before D83: it no longer binds")
    set_by: str | None
    approved_at: str | None = Field(description="ISO 8601")
    measurement_id: int | None
    newer_recommendation_available: bool


class EffectiveProbeLimits(BaseModel):
    """What admission applies for this model (`EffectiveLimits.as_dict`)."""

    max_armed: int = Field(description="The armed-probe backstop")
    decode_ms_per_pass: float
    prefill_ms_per_1k: float
    headroom: float
    sources: ProbeLimitSources
    max_behaviours: int | None = Field(description="Always null since D83 (not enforced)")
    stale: bool = Field(description="The approval rests on another fingerprint; it stays in force")
    approval: ProbeLimitApproval | None


class ImpliedProbeCount(BaseModel):
    """The largest N of one kind on one layer within one budget half."""

    fits_budget: int | None
    fits_with_headroom: int | None


class ImpliedProbeKind(BaseModel):
    """`probe_limits.implied` for one cost line."""

    decode: ImpliedProbeCount
    prefill: ImpliedProbeCount
    # probe_limits.py:194-198: set only when both halves' `fits_with_headroom` are numbers.
    binding: Literal["decode", "prefill"] = may_be_absent(description="The half that limits N")
    n: int = may_be_absent(description="N within both halves and the armed-probe backstop")


class ProbeCapacityImplied(BaseModel):
    """The limits the cost lines imply under the effective budget."""

    # probe_limits.py:184-186: a kind with no line is left out (no measurement at all; `sparse`
    # is null on every row the measure route writes, probe_capacity_measure.py:229).
    dense: ImpliedProbeKind = may_be_absent()
    sparse: ImpliedProbeKind = may_be_absent()


class CapacityArmedShare(BaseModel):
    """How close the armed set is to each limit, as a fraction."""

    count: float | None = Field(description="Armed probes over the backstop; null if it is 0")
    decode: float
    prefill: float


class ProbeCapacityArmed(BaseModel):
    """The armed set of the RESIDENT model against each budget."""

    count: int
    projected: CapacityCostTotals
    share: CapacityArmedShare


class CapacityProbeCost(BaseModel):
    """One probe's cost on one device (`probe_cost.ProbeCost.as_dict`)."""

    decode_ms: float
    prefill_ms_per_1k: float
    layer_fixed_ms: float
    layer_fixed_prefill_ms_per_1k: float
    layer: int
    device: str
    k: int
    measured_at: str = Field(description="ISO 8601; empty for an estimate from a measurement")


class PreviewedProbe(BaseModel):
    """One member of the previewed set."""

    probe_id: str
    layer: int
    rule: str
    source: Literal["measured", "estimated", "unmeasured"] = Field(
        description="`measured` under the resident fingerprint, `estimated` from the model's line"
    )
    armed: bool
    cost: CapacityProbeCost | None = Field(description="Null when unmeasured (it costs nothing)")


class ProbeArmingPreview(BaseModel):
    """The projected cost of arming a set NOW, without arming (039 FR-39.9,
    `probe_limits.preview`)."""

    behaviour: str | None
    probes: list[PreviewedProbe]
    unmeasured: list[str] = Field(description="Ids of the members with no cost")
    projected: CapacityCostTotals
    budget: CapacityBudget
    fits: bool
    stale: bool


class ProbeCapacity(BaseModel):
    """`probe_limits.capacity`: measured, configured, effective, implied, and the armed set."""

    measured: ProbeCapacityMeasured | None = Field(description="Null until the model is measured")
    configured: ProbeLimitsConfigured
    effective: EffectiveProbeLimits
    implied: ProbeCapacityImplied
    armed: ProbeCapacityArmed | None = Field(description="Null unless this is the resident model")
    # model_probe_capacity.py:71-74: only the GET, and only with `behaviour` or `probe_ids`;
    # PUT and measure return `capacity` alone.
    preview: ProbeArmingPreview = may_be_absent(
        description="The preview of arming the named behaviour's set or probes now"
    )


# ── GET …/probe-limits/recommendation ─────────────────────────────────────────────────────────


class RecommendedProbeLimits(BaseModel):
    """The values approving the recommendation writes (D83: percentages)."""

    max_behaviours: int | None = Field(description="Always null since D83")
    decode_budget_pct: float
    prefill_budget_pct: float
    decode_budget_ms: float | None = Field(
        description="The percentage in ms on this measurement; null when the half is unmeasured"
    )
    prefill_budget_ms: float | None = Field(
        description="The percentage in ms on this measurement; null when the half is unmeasured"
    )


class RecommendedShares(BaseModel):
    decode_pct: float | None
    prefill_pct: float | None


class RecommendedCapacity(BaseModel):
    """What fits under the recommended budget."""

    reference_sets: int
    reference_set_probes: int
    dense_probes: int = Field(description="Dense probes on one layer")
    binding: CapacityBinding | None = Field(description="What stops the next reference set")
    dense_binding: CapacityBinding | None = Field(description="What stops the next dense probe")
    upper_bound: bool = Field(description="True when a rule's line was estimated")


class RecommendationMeasurement(BaseModel):
    id: int
    measured_at: str | None = Field(description="ISO 8601")
    devices: list[str]


class RecommendedSetCost(BaseModel):
    """One reference set's projected cost."""

    decode_ms: float
    prefill_ms_per_1k: float


class ProbeLimitRecommendation(BaseModel):
    """This model's explained recommendation (039 FR-39.8, D83). With no usable measurement the
    route answers only `model_id`, `values` (null), `recommendation_digest` (null) and `reasons`."""

    model_id: int
    values: RecommendedProbeLimits | None
    recommendation_digest: str | None = Field(description="64 hex; an approval must name it")
    reasons: list[str] = Field(description="The explanation, in words and numbers")
    # Every field below: model_probe_capacity.py:93-98 — with no measurement (or no usable
    # line: `recommend` -> None) the route answers the four fields above only.
    budget: CapacityBudget = may_be_absent()
    shares: RecommendedShares = may_be_absent()
    capacity: RecommendedCapacity = may_be_absent()
    achieved_sets: int = may_be_absent(description="v1.16 key: the reference sets that fit")
    target_sets: int | None = may_be_absent(description="v1.16 key: always null since D83")
    reference_set: dict[str, Any] = may_be_absent(
        description="The reference set shape (`PROBE_REFERENCE_SET_SHAPE`, a setting)"
    )
    estimated: bool = may_be_absent()
    measurement: RecommendationMeasurement = may_be_absent()
    would_change: list[str] = may_be_absent()
    per_request_wall_ms_estimate: float = may_be_absent()
    set_cost: RecommendedSetCost = may_be_absent()
    cost_model: str = may_be_absent()
