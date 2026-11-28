"""Typed answers for the cluster management routes (044, documented-only typing).

Routes (``miserver/api/routes/management/clusters.py``):
  * ``POST /api/clusters/{cluster_id}/activate``     -> ``ClusterActivated``
  * ``POST /api/clusters/{cluster_id}/deactivate``   -> ``ClusterDeactivated``
  * ``DELETE /api/clusters/{cluster_id}``            -> ``ClusterDeleted``
  * ``PUT /api/clusters/active/intensity``           -> ``ClusterIntensitySet``
  * ``PUT /api/clusters/{cluster_id}/intensity``     -> ``ClusterIntensitySet``

Each dict is built by ``ClusterService`` (``miserver/services/cluster_service.py``), which delegates
activate/deactivate/delete to ``ProfileService`` (``miserver/services/profile_service.py``). Every
key below is written unconditionally by its builder, so every field is required.
"""

from pydantic import BaseModel, Field


class ClusterActivated(BaseModel):
    """``ProfileService.activate_profile`` (profile_service.py:518), returned unchanged by
    ``ClusterService.activate``."""

    profile_id: str = Field(description="The activated cluster's profile id")
    applied_steering: bool = Field(
        description="True when steering was written to the profile's SAE (values, or a clear of an "
        "empty profile's entry); false when no SAE target was attached"
    )
    feature_count: int = Field(description="Number of steered features applied (0 when none)")
    sensing_armed: bool = Field(
        description="Whether co-activation sensing armed for this cluster (011 R2)"
    )
    warnings: list[str] = Field(
        description="One line per active circuit this activation released (013 R3); empty when "
        "no circuit was active"
    )


class ClusterDeactivated(BaseModel):
    """``ProfileService.deactivate_profile`` (profile_service.py:587), returned unchanged by
    ``ClusterService.deactivate``."""

    profile_id: str
    cleared_steering: bool = Field(
        description="True only when this row was the active one and its own SAE entry was cleared"
    )


class ClusterDeleted(BaseModel):
    """``ProfileService.delete_profile`` (profile_service.py:642), returned unchanged by
    ``ClusterService.delete``."""

    profile_id: str
    was_active: bool = Field(
        description="True when the deleted cluster was active (it was deactivated, its live "
        "steering cleared and sensing disarmed first)"
    )


class ClusterIntensitySet(BaseModel):
    """``ClusterService.set_intensity`` (cluster_service.py:272). Shared by the by-id route and the
    global ``active`` route, which resolves the active cluster and calls the same method."""

    profile_id: str
    intensity: float = Field(description="The persisted lambda dial")
    reapplied: bool = Field(
        description="True when the cluster was active, re-apply was requested and steering was "
        "re-applied at the new lambda; false when not active or reapply=false"
    )


#: SH8 / FR-044.23: `GET /api/clusters/{id}/export` answers the stored `mistudio.cluster-definition/v1`
#: document losslessly; the contract references the vendored schema by this `$id`.
CLUSTER_DEFINITION_ID = (
    "https://raw.githubusercontent.com/hitsainet/miStudio/main/docs/schemas/"
    "cluster-definition-v1.json"
)
