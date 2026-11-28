"""The facts on each approval card (ENH-008; BUILD_SPEC §3.1 "Card facts").

One describer per gated action: ``async (values, session) -> {label: value}``. The labels are
the words the operator reads, in the order they are shown; the admin UI renders keys and values as
text, in this order. Each describer runs the route's own first refusal (an unknown model is the
route's 404 ``MODEL_NOT_FOUND``) BEFORE anything is stored, so an agent never files a request that
could only fail.

⚠ Never a lease id: the load and unload describers say only whether one was sent.
"""

from __future__ import annotations

import json
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from miserver.core.errors import ModelBusyError, ModelNotFoundError, ModelNotLoadedError

NOTHING_LOADED = "nothing (no model is loaded)"


async def _model(session: AsyncSession, model_id: int) -> Any:
    from miserver.db.repositories.model_repository import ModelRepository

    row = await ModelRepository(session).get_by_id(int(model_id))
    if row is None:
        raise ModelNotFoundError(f"Model with ID '{model_id}' not found",
                                 details={"model_id": model_id})
    return row


def _name(row: Any) -> str:
    return f"{row.name} (id {row.id})"


def _resident() -> Any | None:
    """The model being served now, from the loader's process state."""
    from miserver.ml.model_loader import LoadedModelState

    return LoadedModelState().current


def _gpu_words(request: Any) -> str:
    gpu = None
    if isinstance(request, dict):
        gpu = request.get("gpu")
    elif request is not None:
        gpu = getattr(request, "gpu", None)
    if gpu is None or gpu == "auto":
        return "automatic: the card with the most free memory that fits"
    if gpu == "all":
        return "split across every visible card"
    return str(gpu)


def _field(request: Any, name: str) -> Any:
    if isinstance(request, dict):
        return request.get(name)
    return getattr(request, name, None) if request is not None else None


async def _ceiling_words(row: Any, requested: Any) -> str:
    """The context ceiling the load will use, refused here as the load itself would refuse it."""
    import asyncio

    from miserver.core.config import settings
    from miserver.services import context_ceiling

    if not (getattr(row, "gguf_files", None) or []):
        if requested is not None:
            context_ceiling.validate_for_model(row, requested)  # raises: GGUF only
        return "not applicable (not a GGUF model)"
    await asyncio.to_thread(context_ceiling.validate_for_model, row, requested)
    ceiling, source = context_ceiling.effective(row, requested)
    return context_ceiling.words(ceiling, source, int(settings.GGUF_CONTEXT_LENGTH))


def _locked_now(row: Any) -> bool:
    return bool(getattr(row, "locked", False))


def _armed_probe_names() -> list[str]:
    from miserver.services.probe_runtime import ProbeRuntimeState

    return [getattr(p, "name", None) or p.probe_id for p in ProbeRuntimeState().armed()]


#: What a card says the load that follows does with what the unload detached (RR-1: nothing is
#: attached or activated again, a reload of the same model included).
AFTER_UNLOAD = "a later load does not attach or activate them again"
AFTER_RELOAD = "the reload does not attach or activate them again"


def _after_swap(new_row: Any) -> str:
    return f"they are not attached to or activated on {_name(new_row)}"


async def _unload_effects(session: AsyncSession, resident_row: Any | None, *,
                          after: str) -> dict[str, Any]:
    """What unloading the resident model takes with it, named, as it REALLY happens: the same
    read `unload_teardown.release_dependents` acts on (`read_dependents`), so the card cannot name
    something the unload leaves or leave out something it takes.

    RR-1 (2026-11-21): the unload detaches every attached SAE and deactivates every active circuit
    and profile (cluster) before the weights go. Before that fix it detached nothing, and these
    lines said so."""
    from miserver.services import unload_teardown

    found = await unload_teardown.read_dependents(session)
    probes = _armed_probe_names()

    def _listed(items: Any, what: str, tail: str) -> str:
        if what in found.unreadable:
            return ("could not be read now; the unload deactivates every active one "
                    "when it runs")
        if not items:
            return "none"
        return ", ".join(item.words() for item in items) + f"; {tail}"

    return {
        "SAEs it detaches": _listed(
            found.saes, "sae", "detached before the weights are released (hooks removed, GPU "
            f"memory freed); {after} (POST /api/saes/{{sae_id}}/attach attaches one)"),
        "Circuits it deactivates": _listed(found.circuits, "circuits", after),
        "Profiles it deactivates": _listed(found.profiles, "profiles", after),
        "Armed probes it disarms": ", ".join(probes) if probes else "none",
        "Steering lock it releases": ("yes (the model is locked for steering now)"
                                      if resident_row is not None and _locked_now(resident_row)
                                      else "none"),
    }


def _refuse_weights_origin(row: Any) -> None:
    """A weights-origin row is refused by every management load (it loads only through
    `POST /api/weights/load`): a card for it could only fail."""
    from miserver.core.errors import LoadThroughWeightsApiError

    origin = getattr(row, "weights_origin", None)
    if isinstance(origin, str) and origin:
        raise LoadThroughWeightsApiError(
            f"Model '{row.name}' holds weights from a checkpoint and is loaded only through "
            "POST /api/weights/load, where its files are verified.",
            details={"model_id": row.id, "weights_origin": origin, "operation": "load"})


async def model_load(values: dict[str, Any], session: AsyncSession) -> dict[str, Any]:
    import asyncio

    from miserver.services import resident_reload

    row = await _model(session, values["model_id"])
    _refuse_weights_origin(row)
    request = values.get("request")
    requested = _field(request, "context_ceiling")
    ceiling = await _ceiling_words(row, requested)
    resident = _resident()
    # The load's own rule (`resident_reload`, the one `ModelService.load_model` applies): the
    # resident named again is a reload when it changes the ceiling or the card, and refused HERE,
    # before anything is filed, when it changes nothing (2026-11-21: a card once promised a load
    # at max that the action could only refuse as "already loaded").
    plan = await asyncio.to_thread(
        resident_reload.compare, row, resident, requested, _field(request, "gpu"))
    if plan is not None and not plan.reload:
        raise resident_reload.already_loaded(row, plan)
    facts: dict[str, Any] = {
        "Model": _name(row),
        "Repository": row.repo_id or row.local_path or "—",
        "Precision": getattr(row.quantization, "value", row.quantization),
    }
    if resident is None:
        facts["Unloads first"] = NOTHING_LOADED
    elif plan is not None:
        facts["Unloads first"] = f"this model, {_name(row)}: it is reloaded"
        facts["Reloads"] = _reload_words(plan, ceiling, values.get("request"))
    else:
        facts["Unloads first"] = f"{resident.model_name} (id {resident.model_id})"
    if resident is not None:
        resident_row = row if plan is not None else await _resident_row(session, resident)
        facts.update(await _unload_effects(
            session, resident_row, after=AFTER_RELOAD if plan is not None else _after_swap(row)))
    facts["Graphics card"] = _gpu_words(values.get("request"))
    facts["Context ceiling"] = ceiling
    facts["Lease"] = ("sent by the agent; kept hidden and checked again when the load runs"
                      if values.get("x_miserver_lease") else "none sent")
    return facts


async def _resident_row(session: AsyncSession, resident: Any) -> Any | None:
    from miserver.db.repositories.model_repository import ModelRepository

    try:
        return await ModelRepository(session).get_by_id(int(resident.model_id))
    except Exception:  # noqa: BLE001 - the lock line says "none"; never blocks filing
        return None


def _reload_words(plan: Any, ceiling: str, request: Any) -> str:
    from miserver.services import resident_reload

    window = (f"a {plan.window_now:,}-token window" if plan.window_now
              else "its current window")
    now = (f"this model is loaded with {window} (ceiling "
           f"{resident_reload.ceiling_words(plan.ceiling_now)}) on "
           f"{resident_reload.cards_words(plan.gpu_indices)}")
    after: list[str] = []
    if resident_reload.CHANGE_CEILING in plan.changes:
        after.append(f"with {ceiling}")
    if resident_reload.CHANGE_PLACEMENT in plan.changes:
        after.append(f"on {_gpu_words(request)}")
    return (f"{now}; after approval it is unloaded (requests already running are given time "
            f"to finish) and loaded again {' and '.join(after)}")


async def model_context_write(values: dict[str, Any], session: AsyncSession) -> dict[str, Any]:
    """The card for PUT /api/models/{id}/context-ceiling: the ceiling now and after."""
    import asyncio

    from miserver.core.config import settings
    from miserver.services import context_ceiling

    row = await _model(session, values["model_id"])
    requested = _field(values.get("request"), "context_ceiling")
    await asyncio.to_thread(context_ceiling.validate_for_model, row, requested)
    server_default = int(settings.GGUF_CONTEXT_LENGTH)
    now = getattr(row, "context_ceiling", None)
    return {
        "Model": _name(row),
        "Context ceiling now": context_ceiling.words(
            now, context_ceiling.SOURCE_MODEL if now is not None else context_ceiling.SOURCE_SERVER,
            server_default),
        "After approval": context_ceiling.words(
            requested,
            context_ceiling.SOURCE_MODEL if requested is not None
            else context_ceiling.SOURCE_SERVER,
            server_default),
        "Takes effect": ("at the model's next load, the OpenAI path's automatic loads included; "
                         "a loaded model keeps its window until then"),
        **_context_write_resident_words(row),
    }


def _context_write_resident_words(row: Any) -> dict[str, Any]:
    """For the loaded model: the window it keeps, and that ONE load applies the new ceiling (a
    load of the resident that changes its ceiling reloads it, `resident_reload`)."""
    from miserver.services import resident_reload

    resident = _resident()
    if resident is None or resident.model_id != row.id:
        return {}
    window = int(getattr(resident, "context_length", 0) or 0)
    return {"Loaded now": (
        f"yes, with {f'a {window:,}-token window' if window else 'its current window'} (ceiling "
        f"{resident_reload.ceiling_words(resident_reload.loaded_ceiling(resident))}); to apply the "
        f"new ceiling now, load it again (POST /api/models/{row.id}/load): that one approval "
        "reloads it")}


async def model_unload(values: dict[str, Any], session: AsyncSession) -> dict[str, Any]:
    row = await _model(session, values["model_id"])
    resident = _resident()
    if resident is None or resident.model_id != row.id:
        # The unload's own refusal, before anything is filed: the card said "Loaded now: no
        # (unloading it changes nothing)" and the approved unload could only fail (2026-11-21).
        raise ModelNotLoadedError(
            f"Model {row.id} is not currently loaded",
            details={"model_id": row.id,
                     "loaded_model_id": resident.model_id if resident is not None else None,
                     "fix": "Nothing to unload. miserver_list_models shows which model is "
                            "loaded; unload that one, or load this one."})
    return {
        "Model": _name(row),
        "Loaded now": "yes",
        **(await _unload_effects(session, row, after=AFTER_UNLOAD)),
        "Lease": ("sent by the agent; kept hidden and checked again when the unload runs"
                  if values.get("x_miserver_lease") else "none sent"),
    }


async def model_delete(values: dict[str, Any], session: AsyncSession) -> dict[str, Any]:
    row = await _model(session, values["model_id"])
    resident = _resident()
    if getattr(row.status, "value", row.status) == "loaded" or (
            resident is not None and resident.model_id == row.id):
        # The delete's own refusal (`ModelService.delete_model`), before anything is filed: a card
        # for a loaded model could only fail (2026-11-21).
        raise ModelBusyError(
            f"Cannot delete model {row.id} while it is loaded. Unload it first.",
            details={"model_id": row.id,
                     "fix": "Unload it first (miserver_unload_model), then delete it."})
    size = row.disk_size_mb
    return {
        "Model": _name(row),
        "Repository": row.repo_id or row.local_path or "—",
        "Size on disk": f"{size:,} MB" if size else "unknown",
        "Warning": "A deleted model cannot be recovered; it has to be downloaded again.",
    }


def _limits_words(limits: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in ("max_armed", "max_behaviours", "decode_budget", "prefill_budget"):
        value = limits.get(key)
        if isinstance(value, dict) and "value" in value:
            value = f"{value['value']} {value.get('unit', '')}".strip()
        out[key] = "global default" if value is None else value
    return out


async def probe_limits_write(values: dict[str, Any], session: AsyncSession) -> dict[str, Any]:
    from miserver.services import probe_limits

    row = await _model(session, values["model_id"])
    body: Any = values.get("request")
    requested: dict[str, Any] = (body.model_dump() if hasattr(body, "model_dump")
                                 else dict(body or {}))
    current: dict[str, Any] = {}
    try:
        capacity = await probe_limits.capacity(session, row)
        current = dict(capacity.get("configured") or {})
    except Exception:  # noqa: BLE001 - the card says it could not read them; never blocks filing
        current = {}
    facts: dict[str, Any] = {
        "Model": _name(row),
        "Current limits": _limits_words(current) if current else "could not be read",
        "Requested limits": _limits_words(requested),
        "Source": requested.get("source") or "operator",
    }
    if requested.get("source") == "recommended-approved":
        facts["Recommendation digest"] = requested.get("recommendation_digest")
        facts["Note"] = ("The values written are the recommendation miServer recomputes when this "
                         "runs; a changed recommendation is refused then.")
    facts["Reason"] = requested.get("reason") or "none given"
    return facts


async def capacity_measure(values: dict[str, Any], session: AsyncSession) -> dict[str, Any]:
    from miserver.services.probe_capacity_measure import refuse_unmeasurable

    row = await _model(session, values["model_id"])
    # The measurement's own refusals that waiting cannot clear (not resident, an engine with no
    # module tree), before anything is filed: the card said "no (the measurement needs it loaded
    # and will refuse)" and filed it anyway (2026-11-21). Armed probes can be disarmed while it
    # waits, so that refusal stays at run time.
    refuse_unmeasurable(row)
    resident = _resident()
    return {
        "Model": _name(row),
        "Loaded now": "yes",
        "Graphics card": str(getattr(resident, "device", "")) or "unknown",
        "What it does": "Runs the resident model on its graphics card to time each step and the "
                        "cost of each probe. Chat requests wait while it runs.",
    }


async def code_rule_import(values: dict[str, Any], _session: AsyncSession) -> dict[str, Any]:
    """The D4 code-rule card: every code document IN FULL, its hash, functions and runner
    contract, and the set's own case results. Runs every check import runs (it writes nothing),
    so an import that would be refused is refused now, with import's own error."""
    from miserver.rules import runner_contract as rc
    from miserver.services import rule_set_service

    raw = str(values.get("request_body") or "").encode("utf-8")
    prepared = rule_set_service.check(raw, declared_digest=values.get("digest"),
                                      origin=values.get("origin"))
    document = prepared.document
    modules = []
    for code_sha256, text in sorted((prepared.code_modules or {}).items()):
        try:
            parsed = rc.parse_code_document(code_sha256, text)
            functions = sorted(parsed.functions)
            source = parsed.source
        except Exception:  # noqa: BLE001 - already verified by check(); never hide the text
            functions, source = [], text
        modules.append({"code_sha256": code_sha256, "functions": functions,
                        "runner_contract": rc.contract_format(), "source": source})
    named = sorted({s.get("code_sha256") for s in document.get("signals") or []
                    if s.get("kind") == "code_rule" and s.get("code_sha256")})
    missing = [h for h in named if h not in (prepared.code_modules or {})]
    return {
        "Rule set": f"{document.get('name')} (behaviour {document.get('behaviour_key')})",
        "Digest": prepared.digest,
        "Version": values.get("version"),
        "Cases": f"{prepared.cases.passed} passed, {prepared.cases.unverified} not verifiable here",
        "Code rules named": ", ".join(named) if named else "none",
        "Code documents missing": (", ".join(missing) + " (those rules stay pending)"
                                   if missing else "none"),
        "Revocation": "miServer asks for each import; there is no revoked state here.",
        "code_modules": modules,
    }


def names_operator_code(values: dict[str, Any]) -> bool:
    """True when a rule-set import names operator code anywhere. Port of miWorkbench's
    ``names_operator_code``, FAILING CLOSED: any ``code_modules`` key, or any object anywhere
    whose ``kind`` is ``code_rule``. A body that is not JSON raises, and the gate then gates."""
    from miserver.rules.code_modules import names_operator_code as scan

    return scan(json.loads(str(values.get("request_body") or "")))


# ENH-006 D76: an agent's change to telemetry settings. Each describer runs the save's own
# validation (`runtime_settings.plan_changes`): a refused value — an endpoint with a user name or a
# query string, an environment-only key — answers the route's own 422 and is never stored.


def _setting_values(values: dict[str, Any]) -> tuple[dict[str, Any], list[str], Any]:
    """(values to set, keys to reset, reason) from any of the three environment write routes."""
    body: Any = values.get("body")
    data: dict[str, Any] = (body.model_dump() if hasattr(body, "model_dump")
                            else dict(body or {}))
    key = values.get("key")
    if key is not None and "value" in data:  # PUT /api/environment/{key}
        return {str(key): data.get("value")}, [], data.get("reason")
    if key is not None:  # DELETE /api/environment/{key}
        return {}, [str(key)], values.get("reason")
    return dict(data.get("values") or {}), list(data.get("reset") or []), data.get("reason")


async def telemetry_write(values: dict[str, Any], session: AsyncSession) -> dict[str, Any]:
    from miserver.services import runtime_settings

    to_set, to_reset, reason = _setting_values(values)
    return await runtime_settings.describe_save(session, to_set, to_reset, reason)


def names_telemetry(values: dict[str, Any]) -> bool:
    """True when an environment write names a telemetry setting (the gate's predicate). Read
    generously (`runtime_settings.is_telemetry_key`); a body it cannot read gates (it raises)."""
    from miserver.services.runtime_settings import is_telemetry_key

    to_set, to_reset, _ = _setting_values(values)
    return any(is_telemetry_key(k) for k in [*to_set, *to_reset])
