"""044 answer models: the probe session read API.

Routes in `miserver/api/routes/management/probe_sessions.py`. Documented-only typing (044, TD9):
each route keeps its body and its `-> ApiResponse` annotation; only the decorator names the model.

* `GET /api/probes/sessions`                        -> `PageOf[ProbeSessionSummary]`
* `GET /api/probes/sessions/{session_id}`           -> `ApiResponse[ProbeSessionDetail]`
* `GET /api/probes/sessions/{session_id}/transcript` -> `ApiResponse[SessionTranscriptAnswer]`
  (a LIST of turns, or ONE turn when `turn_index` is given: a union)
* `DELETE /api/probes/sessions/{session_id}`        -> `ApiResponse[ProbeSessionDeleted]`
* `POST /api/probes/sessions/sample`                -> `ApiResponse[SessionSample]`

A timeline cell is the route's legacy keys with the stored `mi.probe.result/v1` object spread over
them (`**recorded_from_row(...)`, probe_sessions.py:163 and :191). The spread object is flattened
here field by field: a strict answer check closes each object, and an `allOf` of two closed objects
would refuse every key. Its field set is `miserver/api/schemas/probe_result.py` (`_Common` plus
`ProbeResult` / `RuleResult`), plus `persistence` and `source_event_id`: every stored row carries
the first (`stored_result`, `rule_columns`, migration 032) and `recorded_from_row` injects the
second. Open-list values (`engine`, `outcome`, `window`, ...) are `str`, not `Literal`: the
probe-result contract keeps them open within v1 (D38.10). `kind` is a `Literal` because the table
read fixes it (`probe_events` holds only `kind: "probe"`, `rule_signal_results` only `"rule"`).
"""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, RootModel

from miserver.api.schemas.common import may_be_absent

# ── the list and the detail (`_session_row`, probe_sessions.py:51) ─────────────────────────────


class SessionBehaviourTally(BaseModel):
    """One column's counts across a session (`_summaries`, probe_sessions.py:69)."""

    n_fired: int = Field(description="Events in this column whose verdict is true")
    n_scored: int = Field(description="Events in this column that were scored")
    max_score: float | None = Field(description="The highest scored event's score; null if none")


class ProbeSessionSummary(BaseModel):
    """A session as the list serves it: identity, liveness, counts, per-column tallies. No text."""

    id: str
    client_session_id: str | None = Field(description="The caller's session header value, if any")
    source: str = Field(description="How the session was resolved: header, fingerprint or single")
    predecessor_id: str | None = Field(description="The session this one continues, if linked")
    link_ambiguous: bool
    served_model: str | None
    opened_at: str = Field(description="ISO 8601")
    last_seen_at: str = Field(description="ISO 8601")
    open: bool = Field(description="Seen within PROBE_SESSION_IDLE_MINUTES (strict <)")
    n_turns: int
    n_transcribed: int = Field(description="Turns whose text is recorded")
    behaviours: dict[str, SessionBehaviourTally] = Field(
        description="Keyed by column: a behaviour key, else the probe's own name"
    )


# ── the timeline's columns (probe_sessions.py:140-175) ─────────────────────────────────────────


class TimelineColumnBehaviour(BaseModel):
    """The behaviour a column groups (`behaviour_of`, or a rule result's `behaviour_key`)."""

    key: str
    # Absent when the column was first made by a RULE result (probe_sessions.py:171-172 writes
    # `{"key", "version": None}`); a probe-made column carries `behaviour_of`'s three keys.
    name: str | None = may_be_absent(description="The definition's behaviour name; null if unnamed")
    # 044 R3-5: `behaviour_of` sends a version only as a positive integer (R2-2).
    version: int | None = Field(description="The catalogue version; null if uncatalogued or a rule")


class TimelineColumn(BaseModel):
    column: str = Field(description="A behaviour key, a probe name, or `rule:<rule_set_id>`")
    behaviour: TimelineColumnBehaviour | None
    probe_ids: list[str]
    # Added only when a rule result joins the column (probe_sessions.py:174, `setdefault`); a
    # column made of probe events alone has no such key.
    rule_set_ids: list[str] = may_be_absent(description="The rule sets with results in this column")


# ── the stored result's nested blocks (`miserver/api/schemas/probe_result.py`) ────────────────


class RecordedResultProducer(BaseModel):
    name: str
    version: str


class RecordedResultBehaviour(BaseModel):
    key: str
    version: int | None


class RecordedResultSetRuleSet(BaseModel):
    rule_set_id: str
    rule_set_digest: str


class RecordedResultSet(BaseModel):
    """`set`: the behaviour-set slot (`probe_result.SetBlock`)."""

    armed_members: int
    total_members: int
    complete: bool
    missing_probe_ids: list[str]
    members_digest: str
    attached_rule_sets: list[RecordedResultSetRuleSet]


class RecordedResultSteering(BaseModel):
    kind: str = Field(description="none, profile, circuit, inline or unknown")
    id: str | None


class RecordedResultDecision(BaseModel):
    calibration_id: str | None
    target_fpr: float | None
    provisional: bool


class _RecordedResultCommon(BaseModel):
    """The fields every recorded result carries (`probe_result._Common`), as stored and served."""

    model_config = ConfigDict(protected_namespaces=())

    schema_: str = Field(alias="schema", description="mi.probe.result/v1")
    result_id: str
    event_time: str = Field(description="RFC 3339, UTC")
    event_time_source: str
    origin: str
    producer: RecordedResultProducer
    migrated: bool
    request_id: str | None
    trace_id: str | None
    span_id: str | None
    source_event_id: int = Field(description="This row's own id (`recorded_from_row`)")
    batch_id: str | None
    batch_line: int | None
    session_id: str | None
    turn_index: int | None
    behaviour: RecordedResultBehaviour | None
    set_: RecordedResultSet | None = Field(alias="set")
    outcome: str
    reason: str | None = Field(description="A published reason code; null when completed")
    persistence: str = Field(description="`persisted` on every stored row")
    window: str
    status: str


# ── the timeline's cells ──────────────────────────────────────────────────────────────────────


class TimelineProbeCell(_RecordedResultCommon):
    """A probe event in a turn: the legacy keys (probe_sessions.py:149-163), then the stored
    `kind: "probe"` result over them. Keys both carry (`probe_id`, `probe_name`, `window`, `scored`,
    `score`, `threshold`, `verdict`) hold the result's value."""

    # legacy keys only the timeline sends
    column: str
    event_id: int
    not_scored_reason: str | None
    provisional: bool
    # the stored result (`probe_result.ProbeResult`)
    kind: Literal["probe"]
    choice_index: int | None
    input_index: int | None
    probe_id: str
    probe_name: str | None = Field(description="The result's pattern-gated name, else null")
    detector_digest: str
    detector_digest_source: str
    identity_algorithm: str | None
    definition_schema: str
    target: str
    score_semantics: str
    layer: int | None
    hook: str
    basis: str
    rule: str
    served_model: str
    model_revision: str
    template_sha256: str
    precision: str
    engine: str
    execution_mode: str
    steering_ref: RecordedResultSteering
    definition_scope: str
    n_scored_tokens: int
    scored: bool
    score: float | None
    probability: None = Field(description="Always null in v1")
    calibration_ref: None = Field(description="Always null in v1")
    threshold: float | None
    verdict: bool | None
    fires: bool | None = Field(description="v1 duplicate of verdict")
    comparator: str
    decision_digest: str
    decision_revision: int | None
    threshold_revision: int
    decision: RecordedResultDecision
    rung: int
    rung_language: str
    parity_status: str
    model_load_id: str
    evaluation_ms: float | None
    completed_at: str = Field(description="RFC 3339, UTC")


class TimelineRuleCell(_RecordedResultCommon):
    """A rule result in a turn (039 S2): the legacy keys (probe_sessions.py:179-191), then the
    stored `kind: "rule"` result over them. A count, never a score."""

    # legacy key only the timeline sends
    column: str
    rule_result_id: int
    # the stored result (`probe_result.RuleResult`)
    kind: Literal["rule"]
    rule_set_id: str
    rule_set_digest: str
    rule_set_version: int
    signal: str
    form: str
    fires: bool | None = Field(description="null when the outcome is not completed")
    count: int = Field(description="How many matches; never a value")


TimelineCell = Annotated[TimelineProbeCell | TimelineRuleCell, Field(discriminator="kind")]


class TimelineTurn(BaseModel):
    """One turn of the timeline (probe_sessions.py:193-205). A turn with verdicts but no turn row
    (pruned) is still listed, with nulls."""

    turn_index: int
    request_id: str | None = Field(description="null when the turn row is gone")
    created_at: str | None = Field(description="ISO 8601; null when the turn row is gone")
    has_transcript: bool
    transcript_incomplete: bool
    verdicts: list[TimelineCell] = Field(description="Probe entries, then rule entries")


class ProbeSessionDetail(ProbeSessionSummary):
    """One session and its timeline (FR-36.11). No text."""

    columns: list[TimelineColumn] = Field(description="Sorted by column")
    turns: list[TimelineTurn] = Field(description="In turn order")


# ── the transcript (the ONE route that serves session text) ───────────────────────────────────


class SessionTranscriptTurn(BaseModel):
    """One recorded turn's text (`_transcript_row`, probe_sessions.py:244)."""

    turn_index: int
    request_id: str
    user_text: str | None = Field(description="The newest user message")
    reply_text: str | None = Field(description="The visible reply")
    n_choices: int
    incomplete: bool
    created_at: str = Field(description="ISO 8601")


class SessionTranscriptAnswer(RootModel[list[SessionTranscriptTurn] | SessionTranscriptTurn]):
    """Every recorded turn in order (a list), or the one turn `turn_index` names (an object,
    probe_sessions.py:240; miWorkbench 038's spot-check reads it straight off `data`)."""


# ── delete ────────────────────────────────────────────────────────────────────────────────────


class ProbeSessionDeleted(BaseModel):
    """What went (`ProbeSessionRepository.delete_session`); counts taken before the delete."""

    turns: int
    transcripts: int
    events_deleted: int = Field(description="0 unless events=delete")
    events_kept: int = Field(description="Events detached from the session; 0 when events=delete")


# ── the seeded, stratified sample (probe_sessions.py:338) ─────────────────────────────────────


class SampleStratumCount(BaseModel):
    requested: int
    returned: int
    short: int = Field(description="requested - returned, never padded from another stratum")
    available: int


class SampleStrataCounts(BaseModel):
    fired: SampleStratumCount
    near_threshold: SampleStratumCount
    not_fired: SampleStratumCount


class SampledTurn(BaseModel):
    stratum: Literal["fired", "near_threshold", "not_fired"]
    session_id: str
    request_id: str = Field(description="Joined on equality with the turn's request id: never null")
    turn_index: int | None
    event_id: int
    probe_id: str
    source_probe_id: str | None = Field(description="The producer's (miWorkbench's) probe id")
    window: str
    score: float | None
    threshold: float | None
    verdict: bool | None
    threshold_revision: int
    has_transcript: bool


class SessionSample(BaseModel):
    """The same seed returns the same rows (FR-36.12, D15)."""

    seed: int
    near_band: float
    strata: SampleStrataCounts
    turns: list[SampledTurn]
