"""Typed answers for circuit management routes (044, documented-only typing).

Routes (``miserver/api/routes/management/circuits.py``):
  * ``POST /api/circuits/claims/release``  -> ``ClaimsReleased``
  * ``GET /api/circuits/claims``           -> ``list[CircuitClaim]``
  * ``GET /api/circuits/active``           -> ``ActiveCircuits`` (a list, one row, or null)
  * ``DELETE /api/circuits/{circuit_id}``  -> ``CircuitDeleted``

``ActiveCircuit`` is the row ``CircuitSummary`` sends on this route. It is a subclass, not a copy,
so a field added to the summary reaches this answer too; it narrows the two timestamps, which the
summary declares ``Any`` but which are always ``datetime`` (``Circuit.created_at`` /
``updated_at``, non-null, ``server_default=now()``) and so always serialize as ISO strings.
"""

from typing import Literal

from pydantic import BaseModel, Field, RootModel

from miserver.api.schemas.circuit import CircuitSummary


class ClaimsReleased(BaseModel):
    """The release report built in ``release_claims`` (circuits.py:195)."""

    circuit_id: str = Field(description="The circuit id the release was scoped to, as given")
    released_layers: list[int] = Field(
        description="Layers whose live claims this call released (empty when none were held)"
    )
    warnings: list[str] = Field(
        description="Unknown circuit id, no live claims, or the circuit still reads active"
    )


class CircuitClaim(BaseModel):
    """One live layer claim, from ``CircuitClaimRegistry.live_claims`` (circuits.py:242)."""

    layer: int
    circuit_id: str
    circuit_name: str | None = Field(
        description="The claiming circuit's name; null when its row is gone"
    )
    composed: bool = Field(description="True when this layer is shared with another circuit")
    steering_keys: list[int] = Field(
        description="Feature indices this circuit wrote on this layer (empty when not recorded)"
    )


class ActiveCircuit(CircuitSummary):
    """One serving circuit, as ``GET /api/circuits/active`` sends it: every ``CircuitSummary`` key,
    with ``steering`` evaluated."""

    created_at: str = Field(description="ISO-8601 creation time")
    updated_at: str = Field(description="ISO-8601 time of the last change")
    steering: bool | None = Field(
        description="Whether this circuit is actually steering generation now. Always a boolean "
        "in the list form (its owner keys in the steering state); null in the ?single=true form "
        "when the check itself failed (circuits.py:352)."
    )


class ActiveCircuits(RootModel[list[ActiveCircuit] | ActiveCircuit | None]):
    """The ``data`` of ``GET /api/circuits/active``: a list of every serving circuit (the default,
    possibly empty); with ``?single=true``, the most recently activated row, or null when none
    serves."""


class CircuitDeleted(BaseModel):
    """``CircuitService.delete`` (circuit_service.py:1561)."""

    circuit_id: str
    deleted: Literal[True] = Field(
        description="Always true: a missing circuit is refused before anything is deleted"
    )


#: SH8 / FR-044.23: `GET /api/circuits/{id}/export` answers the stored `mistudio.circuit-definition/v1`
#: document losslessly; the contract references the vendored schema by this `$id`.
CIRCUIT_DEFINITION_ID = (
    "https://raw.githubusercontent.com/hitsainet/miStudio/main/docs/schemas/"
    "circuit-definition-v1.json"
)
