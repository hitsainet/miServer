"""Artefacts recorded before ENH-011 X-1 whose layer indices now name a different module.

Before X-1, `layer_resolution.get_layer` took the first layer stack in the module tree. On a composite model
whose vision tower is registered before its language model (Gemma 3 and 4, LLaVA, Qwen2.5-VL) that was the vision
encoder, so a probe, circuit or cluster profile armed or activated then was hooked to a layer a text forward never
runs, and its indices were recorded by miWorkbench's resolver, which made the same choice. After the fix the same
index means a text decoder layer.

Operator decision Q1 (2026-11-15): each such artefact is refused for serving and arming until it is re-validated on
the corrected resolver; nothing is deleted. OQ-3 (2026-11-26): the flag is built even though the inventory found
nothing to flag.

HOW miServer KNOWS, without guessing. miServer's rows do not record their model's architecture (a circuit or cluster
document names only `hf_id`), so a migration cannot tell a composite model's rows from a single-tower model's.
Instead:
  - migration 043 stamps every row that exists when it runs, in `probes`, `circuits` and `profiles`, with
    `recorded_before_text_tower` (rows made later are NULL: recorded on the corrected resolver);
  - at arm, score and activation time the LIVE model answers the question exactly: for each layer the artefact
    uses, `layer_resolution.moved_by_the_text_tower_fix` compares what the old first-match resolver hooked with
    what `get_layer` hooks now. A stamped artefact is refused only where they differ; on a single-tower model they
    never do, so nothing that served before is refused.

RE-VALIDATION is re-recording: record the artefact again in miWorkbench on the corrected resolver and import it,
which makes a new row. The stamped row stays, readable, and stays refused on a model where its indices moved.

The doors (each pinned by a test that fails when its call is removed, `tests/unit/services/test_layer_meaning.py`):
  - a probe: `ProbeArmingService.check` (arm, dry run, behaviour set) and `ProbeScoringService._resolve` (scoring);
  - a circuit: `CircuitService.activate`;
  - a cluster or a manual profile: `ProfileService._validate_activation`, the gate every activation path shares.
"""

from __future__ import annotations

from typing import Any, Iterable, Optional

from miserver.core.errors import LayerMeaningChangedError

#: The column on `probes`, `circuits` and `profiles` (migration 043).
COLUMN = "recorded_before_text_tower"
CODE = LayerMeaningChangedError.code


def stamp_of(row: Any) -> Optional[dict[str, Any]]:
    """The row's stamp, or None. Anything but a non-empty dict reads as recorded after the fix."""
    stamp = getattr(row, COLUMN, None) if row is not None else None
    return stamp if isinstance(stamp, dict) and stamp else None


def resident_model() -> Any:
    """The resident transformers model, or None (nothing loaded, or an engine with no module tree)."""
    from miserver.ml.model_loader import LoadedModelState

    current = LoadedModelState().current
    if current is None or not getattr(current, "supports_hooks", False):
        return None
    return getattr(current, "model", None)


def refuse_if_changed(kind: str, row: Any, model: Any, layers: Iterable[int]) -> None:
    """Raise `LayerMeaningChangedError` when `row` was recorded before the fix and any of `layers` now names
    a different module of `model`; return None otherwise."""
    stamp = stamp_of(row)
    if stamp is None or model is None:
        return
    from miserver.ml.layer_resolution import moved_by_the_text_tower_fix

    moved = []
    for layer in dict.fromkeys(int(x) for x in layers):
        change = moved_by_the_text_tower_fix(model, layer)
        if change is not None:
            moved.append({"layer": layer, "old_module_path": change[0], "new_module_path": change[1]})
    if not moved:
        return
    artefact_id = str(getattr(row, "id", "?"))
    first = moved[0]
    raise LayerMeaningChangedError(
        f"{kind} {artefact_id} was recorded before ENH-011 X-1, when layer {first['layer']} meant "
        f"{first['old_module_path']!r}; on this model it now means {first['new_module_path']!r} (the text "
        f"decoder's), so its layer numbers no longer mean what they meant when it was recorded. Refused until it "
        f"is re-validated on the corrected resolver: record it again in miWorkbench and import it, which makes a "
        f"new {kind}. This row is kept, never deleted.",
        details={"kind": kind, "id": artefact_id, "layers": moved,
                 "recorded_before": stamp.get("stamped_by")},
    )
