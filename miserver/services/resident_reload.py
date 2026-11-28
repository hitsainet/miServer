"""A load that names the RESIDENT model: a reload, or a request that would change nothing (2026-11-21).

`POST /api/models/{id}/load` on the model being served used to be refused `MODEL_ALREADY_LOADED`
whatever it asked for. So a new context ceiling (`PUT /api/models/{id}/context-ceiling`) or another
card could be applied only by two approvals, an unload and then a load, and nothing said so: on
2026-11-21 an agent filed a load of model 17 after its ceiling was set to max, the card promised
"Unloads first: nothing (this model is already loaded)" and "Context ceiling: max", and the approved
load could only fail.

The rule, ONE place for the load itself (`ModelService.load_model`) and its approval card
(`approval_facts.model_load`):

* the load would change the window or the card(s) -> a RELOAD: the approved load unloads the model
  (draining in-flight inference, as an unload does) and loads it again with what it asked for;
* it would change nothing -> refused `409 MODEL_ALREADY_LOADED` with a `fix`, before an approval is
  filed and before anything moves.

"Changes the window" compares the ceiling the resident was loaded WITH (`load_decisions.context.
ceiling`, recorded by the GGUF loader: tokens or "max", the server default already resolved) with
the one this load would use (`context_ceiling.effective`, the server default resolved the same way),
not the window that resulted: a 65,536 ceiling that fitted 40,960 asked again is no change.
"Changes the card" holds only when the request NAMES a placement (an index, a UUID or "all") that is
not where the model is; no `gpu` (or "auto") states no preference, so it never moves a resident.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional, Union

from miserver.core.errors import ModelAlreadyLoadedError
from miserver.services import context_ceiling as ceilings

#: The loaded ceiling as recorded and compared: a number of tokens, or "max".
Ceiling = Union[int, str]

CHANGE_CEILING = "context_ceiling"
CHANGE_PLACEMENT = "gpu"


@dataclass(frozen=True)
class ResidentLoad:
    """What a load of the resident model would change."""

    model_id: int
    #: The ceiling it was loaded with (tokens or "max"); None when not recorded (transformers).
    ceiling_now: Optional[Ceiling]
    #: The ceiling this load would use, resolved the same way; None for a transformers row.
    ceiling_after: Optional[Ceiling]
    #: The window the context was created with (tokens), 0 when unknown.
    window_now: int
    #: The cards it holds memory on now.
    gpu_indices: tuple[int, ...]
    #: The placement the request names, parsed (None: no preference).
    gpu_requested: Any
    #: What differs, in a fixed order: `context_ceiling`, `gpu`. Empty: nothing would change.
    changes: tuple[str, ...]

    @property
    def reload(self) -> bool:
        return bool(self.changes)


def _as_ceiling(value: Optional[int], server_default: int) -> Ceiling:
    resolved = server_default if value is None else int(value)
    return ceilings.CEILING_MAX_WORD if resolved <= 0 else resolved


def loaded_ceiling(resident: Any) -> Optional[Ceiling]:
    """The ceiling the resident was loaded with, as the GGUF loader recorded it."""
    decisions = getattr(resident, "load_decisions", None) or {}
    context = decisions.get("context") if isinstance(decisions, dict) else None
    value = context.get("ceiling") if isinstance(context, dict) else None
    if isinstance(value, str) and value.lower() == ceilings.CEILING_MAX_WORD:
        return ceilings.CEILING_MAX_WORD
    if isinstance(value, int) and not isinstance(value, bool):
        return ceilings.CEILING_MAX_WORD if value <= 0 else value
    return None


def _uuid_index(uuid: str, gpus: Callable[[], list[Any]]) -> Optional[int]:
    try:
        for gpu in gpus():
            if getattr(gpu, "uuid", None) == uuid:
                return int(gpu.index)
    except Exception:  # noqa: BLE001 - an unreadable card list cannot prove "same card"
        return None
    return None


def _placement_changes(resident: Any, wanted: Any, gpus: Callable[[], list[Any]]) -> bool:
    from miserver.ml.gpu_placement import ALL

    if wanted is None:
        return False  # no preference: the resident stays where it is
    held = list(getattr(resident, "gpu_indices", None) or [])
    if wanted == ALL:
        placement = getattr(resident, "placement", None) or {}
        return placement.get("requested") != ALL
    if isinstance(wanted, int):
        return held != [wanted]
    index = _uuid_index(str(wanted), gpus)
    return index is None or held != [index]


def compare(
    row: Any,
    resident: Any,
    requested_ceiling: Optional[int],
    gpu: Any,
    *,
    gpus: Optional[Callable[[], list[Any]]] = None,
) -> Optional[ResidentLoad]:
    """What loading `row` would change, or None when `row` is not the resident model.

    `requested_ceiling` is the load's own (None, tokens, or 0 for max) and `gpu` its raw or parsed
    `gpu`. May read the cards (a UUID is resolved through nvidia-smi): call it off the event loop.
    Never validates the ceiling: the load's own pre-check refuses a bad one before anything moves.
    """
    if resident is None or getattr(resident, "model_id", None) != getattr(row, "id", None):
        return None
    from miserver.core.config import settings
    from miserver.ml.gpu_placement import list_gpus, parse_gpu_request

    wanted = parse_gpu_request(gpu)
    gguf = bool(getattr(row, "gguf_files", None) or [])
    now = loaded_ceiling(resident)
    after: Optional[Ceiling] = None
    changes: list[str] = []
    if gguf:
        value, _source = ceilings.effective(row, requested_ceiling)
        after = _as_ceiling(value, int(settings.GGUF_CONTEXT_LENGTH))
        # Not recorded (a load before the record existed): only a ceiling the request NAMES
        # counts as a change; nothing proves the row's or the server's would differ.
        if (now != after) if now is not None else (requested_ceiling is not None):
            changes.append(CHANGE_CEILING)
    if _placement_changes(resident, wanted, gpus or list_gpus):
        changes.append(CHANGE_PLACEMENT)
    return ResidentLoad(
        model_id=int(row.id),
        ceiling_now=now,
        ceiling_after=after,
        window_now=int(getattr(resident, "context_length", 0) or 0),
        gpu_indices=tuple(getattr(resident, "gpu_indices", None) or ()),
        gpu_requested=wanted,
        changes=tuple(changes),
    )


def ceiling_words(value: Optional[Ceiling]) -> str:
    if value is None:
        return "not recorded"
    if value == ceilings.CEILING_MAX_WORD:
        return "max"
    return f"{int(value):,} tokens"


def cards_words(indices: tuple[int, ...]) -> str:
    if not indices:
        return "the CPU"
    if len(indices) == 1:
        return f"GPU {indices[0]}"
    return "GPUs " + ", ".join(str(i) for i in indices)


def already_loaded(row: Any, plan: ResidentLoad) -> ModelAlreadyLoadedError:
    """The refusal for a load that would change nothing, with what to send instead."""
    name = getattr(row, "name", None) or f"model {plan.model_id}"
    loaded = (f" with a {plan.window_now:,}-token window (ceiling {ceiling_words(plan.ceiling_now)})"
              if plan.ceiling_now is not None else "")
    fix = (
        f"Nothing to do: it is served now. To change its window, send a different "
        f'"context_ceiling" (tokens or "max"), or set the model\'s own with '
        f"PUT /api/models/{plan.model_id}/context-ceiling and then load it: the load reloads "
        f'it. To move it, name another card in "gpu".'
    )
    return ModelAlreadyLoadedError(
        f"Model {plan.model_id} ({name}) is already loaded{loaded} on "
        f"{cards_words(plan.gpu_indices)}; this load would change nothing.",
        details={
            "model_id": plan.model_id,
            "loaded_ceiling": plan.ceiling_now,
            "requested_ceiling": plan.ceiling_after,
            "loaded_on": list(plan.gpu_indices),
            "fix": fix,
        },
    )
