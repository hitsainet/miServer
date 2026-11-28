"""What an unload takes down with the model, before its weights go (RR-1, 2026-11-21).

⚠ ONE PLACE, CALLED FROM `ModelService.unload_model`, WHICH EVERY UNLOAD GOES THROUGH: an explicit
unload (`POST /api/models/{id}/unload`), a load that swaps the resident out (`load_model`'s
auto-unload), a reload of the resident (the same auto-unload, `resident_reload`), a `/v1` auto-load
swap (`load_model_and_wait` → `load_model`), a `/v1` reload of a carrier row
(`load_model_and_wait(reload=True)`), and the weights API's full reload
(`WeightsCheckpointService`, `unload_model(under_barrier=True)`). Process shutdown is the one
`loader.unload()` that skips it: the process ends, and startup reconciles the rows
(`main.STALE_STATE_RESETS`).

Measured 2026-11-21 before this existed: the real `unload_model` left every `AttachedSAEState`
entry in place. The SAE stayed listed as attached and kept its GPU memory, its hook sat on the
freed model's modules and steered nothing, and circuits, clusters and profiles that steered
through it went on reading active. A silent failure: status said steering was live while nothing
steered.

What it does, in this order, each step through the owner's OWN path:

1. every active circuit: `CircuitService.deactivate` (its steering owner released, its edge
   sensing disarmed, its layer claims released, the row inactive);
2. every active profile or cluster: `ClusterService.deactivate` / `ProfileService.deactivate_profile`
   (its steering cleared, co-activation sensing disarmed, the row inactive). Read AFTER step 1,
   because a slice-fallback circuit deactivates its own slice profile;
3. every attached SAE: `SAEService.detach_for_unload`, the body of `detach_sae` (hooks removed,
   Dynamo reset, steering and monitoring cleared, sensing and edge sensing disarmed, moved to the
   CPU, the registry entry dropped, the row `cached`, the attachment inactive, the steering lock
   released) without taking the admission slot, which the unload has already closed
   (`begin_unload`) and drained. Then `sae:detached` is emitted for it.

Every step is guarded on its own and a failure is logged at ERROR and returned: a bookkeeping
failure must not stop the unload (the weights go either way), and it must not be silent. An SAE
whose detach raised is still cleared from the registry by the closing sweep, so its hook cannot
outlive the model.

A MANUAL DETACH (`SAEService.detach_sae`, RR-8) uses the same deactivation,
`deactivate_steering_through`, filtered to what steers through that one SAE.

**Nothing is attached or activated again by the load that follows, a reload of the same model
included** (decided 2026-11-21): an attach must find room on the card after the new load, and a
circuit below rung 2 must be acknowledged again at activation (the reason startup deactivates
circuits too). The approval cards name what goes (`read_dependents`, the same read).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import structlog

logger = structlog.get_logger()


@dataclass(frozen=True)
class SaeDependent:
    sae_id: str
    name: str
    layers: tuple[int, ...]

    def words(self) -> str:
        where = ", ".join(str(layer) for layer in self.layers)
        return f"{self.name} (id {self.sae_id}) at layer {where}"


@dataclass(frozen=True)
class RowDependent:
    id: str
    name: str
    kind: str  # "circuit", "profile" or "cluster"

    def words(self) -> str:
        suffix = ", a cluster" if self.kind == "cluster" else ""
        return f"{self.name} (id {self.id}{suffix})"

    def as_dict(self) -> dict[str, str]:
        return {"id": self.id, "name": self.name, "kind": self.kind}


@dataclass(frozen=True)
class Dependents:
    """Everything an unload detaches or deactivates, as it stands now."""

    saes: tuple[SaeDependent, ...] = ()
    circuits: tuple[RowDependent, ...] = ()
    profiles: tuple[RowDependent, ...] = ()
    #: Reads that failed (the database is down): the card says it could not read them.
    unreadable: tuple[str, ...] = ()


@dataclass
class TeardownReport:
    model_id: int
    detached_saes: list[str] = field(default_factory=list)
    deactivated_circuits: list[str] = field(default_factory=list)
    deactivated_profiles: list[str] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)


def attached_sae_layers() -> dict[str, tuple[int, ...]]:
    """`{sae_id: (layer, ...)}` from the live registry, in attach order (no database)."""
    from miserver.services.sae_service import AttachedSAEState

    out: dict[str, list[int]] = {}
    for entry in AttachedSAEState().entries():
        out.setdefault(entry.sae_id, []).append(int(entry.layer))
    return {sae_id: tuple(layers) for sae_id, layers in out.items()}


async def _active_profiles(session: Any) -> list[Any]:
    from sqlalchemy import select

    from miserver.db.models.profile import Profile

    # Every active row, not `get_active()`'s `scalar_one_or_none()`: a second stale active row
    # would make that raise, and an unload must not fail on it.
    result = await session.execute(
        select(Profile).where(Profile.is_active == True).order_by(Profile.id)  # noqa: E712
    )
    return list(result.scalars().all())


def _steers_through(sae_ids: set[str], *, owner_keys: Any = None, layers: Any = None,
                    sae_layers: set[int]) -> bool:
    """Whether steering recorded by `owner_keys` (`{(sae_id, layer): ...}`) or served on `layers`
    goes through one of `sae_ids`. The owner map names the SAE; `layers` is the fallback for a
    circuit whose steering predates it (its `deactivate` then clears circuit steering wholesale)."""
    if owner_keys:
        return any(key[0] in sae_ids for key in owner_keys)
    return bool(set(int(x) for x in (layers or [])) & sae_layers)


def _profile_steers_through(profile: Any, sae_ids: set[str]) -> bool:
    """Whether the active `profile` steers one of `sae_ids`: its OWN attached entry (028 FU-1), or
    the SAE it is bound to when that entry cannot be resolved."""
    from miserver.services.profile_targeting import profile_sae_or_none
    from miserver.services.sae_service import AttachedSAEState

    entries = AttachedSAEState().entries()
    sae = profile_sae_or_none(profile, entries)
    if sae is not None:
        return any(e.sae is sae and e.sae_id in sae_ids for e in entries)
    return getattr(profile, "sae_id", None) in sae_ids


async def read_dependents(session: Any, sae_ids: Optional[set[str]] = None) -> Dependents:
    """What would be detached and deactivated now. Read-only; never raises.

    `sae_ids` None: an UNLOAD, which detaches every SAE and so deactivates every active circuit and
    profile. A set: a manual DETACH of those SAEs (RR-8), which deactivates only the circuits and
    the profile or cluster that steer through them; a co-tenant on another SAE keeps serving.

    The approval cards, `release_dependents` and `SAEService.detach_sae` all read this, so what is
    named cannot differ from what is taken."""
    from miserver.db.repositories.circuit_repository import CircuitRepository
    from miserver.db.repositories.profile_repository import ProfileRepository
    from miserver.db.repositories.sae_repository import SAERepository
    from miserver.services.sae_service import AttachedSAEState

    unreadable: list[str] = []
    saes: list[SaeDependent] = []
    for sae_id, layers in attached_sae_layers().items():
        if sae_ids is not None and sae_id not in sae_ids:
            continue
        name = sae_id
        try:
            row = await SAERepository(session).get(sae_id)
            if row is not None and getattr(row, "name", None):
                name = row.name
        except Exception:  # noqa: BLE001 - the id stands in for the name
            unreadable.append("sae names")
        saes.append(SaeDependent(sae_id=sae_id, name=name, layers=layers))
    sae_layers = {layer for dep in saes for layer in dep.layers}
    owners = AttachedSAEState()._owners
    circuits: list[RowDependent] = []
    try:
        for c in await CircuitRepository(session).list_active():
            if sae_ids is not None:
                slice_id = (c.provenance or {}).get("slice_profile_id") \
                    if c.serving_mode == "slice_fallback" else None
                slice_row = (await ProfileRepository(session).get(slice_id)
                             if slice_id else None)
                if not (_steers_through(sae_ids, owner_keys=owners.get(f"circuit:{c.id}"),
                                        layers=c.layers, sae_layers=sae_layers)
                        or (slice_row is not None
                            and _profile_steers_through(slice_row, sae_ids))):
                    continue
            circuits.append(RowDependent(id=c.id, name=c.name, kind="circuit"))
    except Exception:  # noqa: BLE001
        unreadable.append("circuits")
    profiles: list[RowDependent] = []
    try:
        profiles = [RowDependent(id=p.id, name=p.name,
                                 kind="cluster" if p.source_kind == "cluster" else "profile")
                    for p in await _active_profiles(session)
                    if sae_ids is None or _profile_steers_through(p, sae_ids)]
    except Exception:  # noqa: BLE001
        unreadable.append("profiles")
    if unreadable:
        try:
            await session.rollback()
        except Exception:  # noqa: BLE001
            pass
    return Dependents(saes=tuple(saes), circuits=tuple(circuits), profiles=tuple(profiles),
                      unreadable=tuple(dict.fromkeys(unreadable)))


def _services(session: Any, sae: Any) -> tuple[Any, Any, Any]:
    """The profile, cluster and circuit services, built as the request dependencies build them, on
    one session, over the SAE service `sae`."""
    from miserver.db.repositories.circuit_repository import CircuitRepository
    from miserver.db.repositories.profile_repository import ProfileRepository
    from miserver.services.circuit_service import CircuitService
    from miserver.services.cluster_service import ClusterService
    from miserver.services.profile_service import ProfileService

    profiles = ProfileService(repository=ProfileRepository(session), sae_service=sae)
    clusters = ClusterService(profile_service=profiles, repository=ProfileRepository(session),
                              sae_service=sae)
    circuits = CircuitService(repository=CircuitRepository(session), sae_service=sae,
                              cluster_service=clusters)
    return profiles, clusters, circuits


@dataclass
class Deactivated:
    circuits: list[RowDependent] = field(default_factory=list)
    profiles: list[RowDependent] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)


async def deactivate_steering_through(session: Any, sae: Any,
                                      sae_ids: Optional[set[str]] = None) -> Deactivated:
    """Deactivate every active circuit, then every active profile or cluster, that steers through
    `sae_ids` (None: all of them, for an unload), each through its OWN deactivate path. Never
    raises; a failure is logged at ERROR and returned. The SAEs themselves are not touched.

    Circuits first: a slice-fallback circuit deactivates its own slice profile, so the profiles are
    read after them. A profile's deactivation disarms its co-activation sensing, and a later attach
    re-arms sensing only for an ACTIVE profile, so nothing is re-armed by one (RR-8)."""
    profiles, clusters, circuits = _services(session, sae)
    out = Deactivated()
    for circuit in (await read_dependents(session, sae_ids)).circuits:
        try:
            await circuits.deactivate(circuit.id)
            out.circuits.append(circuit)
        except Exception as exc:  # noqa: BLE001
            out.failures.append(_failure(f"circuit {circuit.id}", exc))
            await _rollback(session)
    for profile in (await read_dependents(session, sae_ids)).profiles:
        try:
            if profile.kind == "cluster":
                await clusters.deactivate(profile.id)
            else:
                await profiles.deactivate_profile(profile.id)
            out.profiles.append(profile)
        except Exception as exc:  # noqa: BLE001
            out.failures.append(_failure(f"profile {profile.id}", exc))
            await _rollback(session)
    return out


def _failure(what: str, exc: BaseException) -> str:
    logger.error("steering_owner_deactivate_failed", step=what, error=str(exc)[:200],
                 error_type=type(exc).__name__)
    return f"{what}: {type(exc).__name__}: {str(exc)[:200]}"


def _failed(report: TeardownReport, what: str, exc: BaseException) -> None:
    report.failures.append(f"{what}: {type(exc).__name__}: {str(exc)[:200]}")
    logger.error("unload_teardown_step_failed", model_id=report.model_id, step=what,
                 error=str(exc)[:200], error_type=type(exc).__name__)


async def _rollback(session: Any) -> None:
    try:
        await session.rollback()
    except Exception:  # noqa: BLE001
        pass


def _force_clear(sae_id: str) -> None:
    """An SAE whose detach raised: its hooks and registry entries still go with the model."""
    from miserver.services.sae_service import AttachedSAEState

    state = AttachedSAEState()
    for entry in [e for e in state.entries() if e.sae_id == sae_id]:
        try:
            if entry.sae is not None:
                entry.sae.to_cpu()
        except Exception:  # noqa: BLE001
            pass
    state.clear(sae_id=sae_id)


async def release_dependents(
    model_id: int, *, emitter: Any = None, inference_service: Any = None,
    session_factory: Optional[Any] = None,
) -> TeardownReport:
    """Deactivate every circuit and profile and detach every SAE, before `model_id`'s weights go.

    Never raises (see the module docstring); the report names what went and what failed."""
    from miserver.db import base as db_base

    factory = session_factory or db_base.async_session_factory
    report = TeardownReport(model_id=model_id)
    try:
        await _release(factory, report, emitter, inference_service)
    except Exception as exc:  # noqa: BLE001 - never stops the unload; never silent
        _failed(report, "teardown", exc)
    # THE SWEEP: whatever the steps above could not detach (a detach that raised, a step that
    # never ran) still loses its hooks and its registry entry with the model.
    for sae_id in attached_sae_layers():
        _failed(report, f"sae {sae_id}", RuntimeError("still attached after the teardown; "
                                                      "its hooks are removed here"))
        _force_clear(sae_id)
    if report.detached_saes or report.deactivated_circuits or report.deactivated_profiles:
        logger.info("unload_released_dependents", model_id=model_id,
                    saes=report.detached_saes, circuits=report.deactivated_circuits,
                    profiles=report.deactivated_profiles)
    if report.failures:
        logger.error("unload_teardown_incomplete", model_id=model_id, failures=report.failures)
    return report


async def _release(factory: Any, report: TeardownReport, emitter: Any,
                   inference_service: Any) -> None:
    async with factory() as session:
        before = await read_dependents(session)
        for read in before.unreadable:
            report.failures.append(f"could not read {read}")
        if not (before.saes or before.circuits or before.profiles):
            return
        from miserver.db.repositories.sae_repository import SAERepository
        from miserver.services.sae_service import SAEService

        sae = SAEService.for_unload(SAERepository(session), emitter=emitter,
                                    inference_service=inference_service)
        done = await deactivate_steering_through(session, sae, None)
        report.deactivated_circuits.extend(c.words() for c in done.circuits)
        report.deactivated_profiles.extend(p.words() for p in done.profiles)
        report.failures.extend(done.failures)
        for dependent in before.saes:
            try:
                await sae.detach_for_unload(dependent.sae_id)
                report.detached_saes.append(dependent.words())
            except Exception as exc:  # noqa: BLE001 - its hook goes in the sweep after this
                _failed(report, f"sae {dependent.sae_id}", exc)
                await _rollback(session)
            if emitter is not None:
                try:
                    await emitter.emit_sae_detached(dependent.sae_id)
                except Exception:  # noqa: BLE001 - a socket must not stop an unload
                    logger.warning("unload_sae_detached_emit_failed", sae_id=dependent.sae_id)
        if before.saes and emitter is not None:
            try:
                await emitter.emit_steering_changed(enabled=False, values={}, active_count=0)
            except Exception:  # noqa: BLE001
                logger.warning("unload_steering_emit_failed", model_id=report.model_id)
