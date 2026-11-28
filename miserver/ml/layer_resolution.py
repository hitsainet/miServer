"""Finding a decoder layer in an arbitrary transformer, and which device it sits on.

Lifted verbatim out of `SAEHooker` (Feature 024 task 1.3). The behaviour is unchanged — the same
accessor patterns in the same order, the same fallbacks, the same error messages — so the existing
hooker tests pass against it without modification. `SAEHooker` now delegates here.

⚠ IT LIVES IN ITS OWN MODULE RATHER THAN IN `sae_hooker`, and that is an architectural statement
rather than tidiness. BR-002 says probes do not depend on SAE attachment; a probe hooker that had
to `from miserver.ml.sae_hooker import ...` would make that claim false at the import level even while
it held at runtime, and the next person to read the imports would reasonably conclude a probe needs
an SAE. Resolving a layer index to a module has nothing to do with SAEs.

`reset_dynamo_for_hook_change` is here for the same reason. The FTID said to import it from
`SAEService` rather than duplicate it — but `SAEService` pulls in `AttachedSAEState`, the singleton
that task 4A.4 spies on to prove a probe never touches it, so importing it into probe code would
couple exactly what the design is trying to keep apart. Lifting it is the same move as the layer
helpers, and `SAEService._reset_dynamo_for_hook_change` now delegates here.
"""

from __future__ import annotations

import itertools
import logging
from typing import Any, Optional

import torch
from torch import nn

logger = logging.getLogger(__name__)


# ── The text decoder first (ENH-011 X-1, #2) ──────────────────────────────────────────────────
#
# A composite checkpoint (Gemma 3 and 4 `...ForConditionalGeneration`, LLaVA, Qwen2.5-VL) registers
# its vision tower BEFORE its language model. The fixed accessors below miss its text decoder
# (`model.language_model.layers`), so the structural search took the first list of layers it met:
# the vision encoder. Every probe, SAE, steering edit and circuit then hooked a layer a text forward
# never runs; nothing raised, and nothing was read or moved (a steering edit measured 0.0 against
# 0.329, R2). miWorkbench resolves the same way (its `discover_transformer_structure`), and both
# repos test their resolver on miWorkbench's `docs/schemas/layer-resolution-cases.json`, vendored
# here byte-identically.

#: Child names that mark a sequence mixer (attention, or LFM2's convolution) or a feed-forward
#: block. Tuples, not sets: matched first-to-last. The same lists miWorkbench's discovery uses.
_ATTENTION_PATTERNS = ("self_attn", "attention", "attn", "self_attention", "mha", "conv")
_MLP_PATTERNS = ("mlp", "feed_forward", "ffn", "dense", "ff", "intermediate")
_NORM_SUFFIXES = ("norm", "layernorm", "layer_norm", "_norm")


class LayerResolutionError(ValueError):
    """The text decoder's layers cannot be named with certainty, so resolution refuses.

    A `ValueError`, so every caller that already refused on "could not find layer" refuses here too
    (`SAEHooker.validate_layer` answers False; the compatibility check reports it). Raised when a
    composite model's stacks cannot be told apart, and when the stack found disagrees with the depth
    the text config states: a hook on the wrong stack never fires on a text forward and reports
    nothing, so a guess is worse than a refusal.
    """


def text_config_of(config: Any) -> Any:
    """The TEXT DECODER's config: the config itself on a flat model, its text section on a composite.

    Asks transformers (`get_text_config(decoder=True)`), which knows every composite layout it
    supports, rather than naming `text_config`.
    """
    if config is None:
        return None
    getter = getattr(config, "get_text_config", None)
    if callable(getter):
        try:
            text = getter(decoder=True)
        except Exception:  # noqa: BLE001 - an exotic config that cannot answer is read as flat
            text = None
        if text is not None:
            return text
    return config


def _positive_int(value: Any) -> Optional[int]:
    return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else None


def _first_int(config: Any, names: tuple[str, ...]) -> Optional[int]:
    """The first of `names` the TEXT config states as a positive int, else the top-level config's.

    The top level is read only when the text config states none of them: a composite config's top
    level is fusion metadata, and an older or hand-written one can carry a top-level width that is
    the vision tower's or the projector's.
    """
    text = text_config_of(config)
    for source in ((text, config) if text is not config else (config,)):
        for name in names:
            value = _positive_int(getattr(source, name, None))
            if value is not None:
                return value
    return None


def text_hidden_size(config: Any) -> Optional[int]:
    """The text decoder's residual width (`hidden_size`, else `d_model`, else `n_embd`), or None.

    None when nothing states one: a width is never assumed.
    """
    return _first_int(config, ("hidden_size", "d_model", "n_embd"))


def text_depth(config: Any) -> Optional[int]:
    """The text decoder's stated depth (`num_hidden_layers`, `n_layer`, `num_layers`, `n_layers`)."""
    return _first_int(config, ("num_hidden_layers", "n_layer", "num_layers", "n_layers"))


def _looks_like_a_layer(module: nn.Module) -> bool:
    """A transformer layer has a sequence mixer or a feed-forward child (miWorkbench's rule)."""
    children = [name.lower() for name in module._modules]

    def matches(child: str, pattern: str) -> bool:
        if child == pattern:
            return True
        return pattern in child and not child.endswith(_NORM_SUFFIXES)

    return any(matches(c, p) for p in _ATTENTION_PATTERNS + _MLP_PATTERNS for c in children)


def transformer_stacks(model: nn.Module) -> list[tuple[str, nn.ModuleList]]:
    """Every layer stack in the model: each OUTERMOST `nn.ModuleList` whose first entry looks like a
    transformer layer, in registration order.

    A list nested inside another stack's layer is part of that layer, not a second tower. A
    single-tower model has one stack; a vision-language model has its vision encoder's and its
    language model's.
    """
    stacks: list[tuple[str, nn.ModuleList]] = []
    for name, module in model.named_modules():
        if not (isinstance(module, nn.ModuleList) and len(module) > 0):
            continue
        if any(name.startswith(outer + ".") for outer, _ in stacks):
            continue
        if _looks_like_a_layer(module[0]):
            stacks.append((name, module))
    return stacks


def _contains(container: nn.Module, target: nn.Module) -> bool:
    return any(module is target for module in container.modules())


def _parent_of(model: nn.Module, path: str) -> nn.Module:
    parent_path = path.rsplit(".", 1)[0] if "." in path else ""
    return model.get_submodule(parent_path) if parent_path else model


def _text_stack(
    model: nn.Module, stacks: list[tuple[str, nn.ModuleList]]
) -> tuple[str, nn.ModuleList, str]:
    """Which of several stacks is the text decoder's, and by what evidence.

    1. The stack whose owning module also owns the model's input TOKEN embedding. A vision encoder
       embeds patches; only the language model owns the table a token id indexes.
    2. Else the stack inside `get_decoder()`, when the model answers it with a strict sub-module.

    Both agree on every family in `docs/schemas/layer-resolution-cases.json` on transformers
    5.20.2. Neither answering refuses: never a guess.
    """
    try:
        embedding = model.get_input_embeddings()
    except Exception:  # noqa: BLE001 - a model without the method has no table to ask
        embedding = None
    if isinstance(embedding, nn.Module):
        owners = [(p, s) for p, s in stacks if _contains(_parent_of(model, p), embedding)]
        if len(owners) == 1:
            return owners[0][0], owners[0][1], "owns the input embedding"

    getter = getattr(model, "get_decoder", None)
    if callable(getter):
        try:
            decoder = getter()
        except Exception:  # noqa: BLE001 - an exotic layout that cannot answer
            decoder = None
        if isinstance(decoder, nn.Module) and decoder is not model:
            inside = [(p, s) for p, s in stacks if _contains(decoder, s)]
            if len(inside) == 1:
                return inside[0][0], inside[0][1], "inside get_decoder()"

    raise LayerResolutionError(
        f"{type(model).__name__} has {len(stacks)} layer stacks "
        f"({', '.join(f'{p!r} ({len(s)} layers)' for p, s in stacks)}) and none can be named the "
        f"text decoder: no single stack's owner holds the input embedding, and get_decoder() does "
        f"not single one out. Refusing rather than hooking a stack a text forward may never run."
    )


def _check_text_depth(model: nn.Module, path: str, layers: nn.Module) -> None:
    """Refuse when the stack found disagrees with the depth the text config states."""
    config = getattr(model, "config", None)
    stated = _positive_int(getattr(text_config_of(config), "num_hidden_layers", None))
    if stated is None or stated == len(layers):
        return
    raise LayerResolutionError(
        f"the text decoder's layers at {path!r} hold {len(layers)} layers but "
        f"{type(text_config_of(config)).__name__} states num_hidden_layers={stated}. Refusing: a "
        f"layer index into the wrong stack hooks a layer that does not mean what the index says "
        f"(ENH-011 #2)."
    )


def text_layers(model: nn.Module) -> Optional[tuple[str, nn.ModuleList]]:
    """`(path, layers)` of the text decoder when the model has MORE THAN ONE layer stack, else None.

    None leaves a single-stack model to the fixed accessors and the structural search, exactly as
    before X-1. Either way the depth is cross-checked against the text config by the caller (one
    stack) or here (several), and a disagreement raises `LayerResolutionError`.
    """
    stacks = transformer_stacks(model)
    if len(stacks) == 1:
        _check_text_depth(model, *stacks[0])
    if len(stacks) <= 1:
        return None
    path, layers, evidence = _text_stack(model, stacks)
    _check_text_depth(model, path, layers)
    logger.debug("text decoder layers at %s (%d layers; it %s); other stacks: %s",
                 path, len(layers), evidence, [p for p, _ in stacks if p != path])
    return path, layers


def get_layer(model: nn.Module, layer_idx: int) -> nn.Module:
    """The decoder layer module at `layer_idx`.

    Supports several transformer layouts:
      - Gemma / Llama / Mistral:  model.model.layers[i]
      - GPT-2 / GPT-Neo:          model.transformer.h[i]
      - Some HF decoders:         model.model.decoder.layers[i]
      - Generic:                  model.layers[i], model.encoder.layer[i], model.decoder.layer[i]

    Raises ValueError when no pattern matches, naming the patterns tried — a hook installed on the
    wrong module is silent, so failing loudly here is the point.

    THE TEXT DECODER FIRST (ENH-011 #2): on a model with more than one layer stack the index is into
    the text decoder's (`text_layers`), and the accessors below run only on a single-stack model. A
    depth that disagrees with the text config raises `LayerResolutionError` (a ValueError).
    """
    found = text_layers(model)
    if found is not None:
        path, layers = found
        try:
            return layers[layer_idx]
        except IndexError:
            raise ValueError(
                f"Could not find layer {layer_idx}: the text decoder's layers at {path!r} hold "
                f"{len(layers)} layers."
            ) from None
    return first_match_layer(model, layer_idx)


def first_match_layer(model: nn.Module, layer_idx: int) -> nn.Module:
    """The layer the resolver took BEFORE ENH-011 X-1: fixed accessors, then the first long-enough
    `ModuleList` named like a layer list. Unchanged, and still the answer on a single-stack model.

    On a composite model it is the VISION encoder's layer. `get_layer` calls it only for a model with
    at most one layer stack; `layer_meaning.refuse_if_changed` calls it to learn what an artefact
    recorded before the fix was hooked to, and refuses when that is not what `get_layer` hooks now.
    """
    layer_access_patterns = [
        lambda m: m.model.layers[layer_idx],
        lambda m: m.transformer.h[layer_idx],
        lambda m: m.model.decoder.layers[layer_idx],
        lambda m: m.layers[layer_idx],
        lambda m: m.encoder.layer[layer_idx],
        lambda m: m.decoder.layer[layer_idx],
    ]

    for accessor in layer_access_patterns:
        try:
            layer = accessor(model)
            logger.debug(f"Found layer {layer_idx} using accessor pattern")
            return layer
        except (AttributeError, IndexError, TypeError, KeyError):
            continue

    for name, module in model.named_modules():
        if isinstance(module, nn.ModuleList) and len(module) > layer_idx:
            if "layer" in name.lower() or "block" in name.lower() or name == "h":
                logger.debug(f"Found layer via ModuleList search: {name}[{layer_idx}]")
                return module[layer_idx]

    raise ValueError(
        f"Could not find layer {layer_idx}. "
        f"Model architecture may not be supported. "
        f"Supported patterns: Llama/Gemma (model.model.layers), "
        f"GPT-2 (transformer.h), generic (layers). "
        f"Check model.named_modules() for layer structure."
    )


def layer_device(model: nn.Module, layer: int) -> torch.device:
    """The device holding `layer`'s weights.

    A model spread across GPUs (device_map="auto") keeps later layers on another card. Anything
    hooked there must live on that card, or every forward pass mixes devices.
    """
    target_layer = get_layer(model, layer)
    for tensor in itertools.chain(target_layer.parameters(), target_layer.buffers()):
        return tensor.device
    for tensor in model.parameters():
        return tensor.device
    return torch.device("cpu")


def get_layer_count(model: nn.Module) -> int:
    """Total number of decoder layers.

    The config is tried first because it is the only source that is right by construction; the
    structural fallbacks exist for models whose config omits the attribute.

    THE TEXT DECODER FIRST (ENH-011 #2): on a model with several layer stacks this is the text
    decoder's length (checked against the text config); otherwise the TEXT config's depth, read
    before the top level. A composite's top-level config states no depth, and the structural search
    below then counted the vision tower.
    """
    found = text_layers(model)
    if found is not None:
        return len(found[1])
    stated = text_depth(getattr(model, "config", None))
    if stated is not None:
        return stated

    layer_access_patterns = [
        lambda m: len(m.model.layers),
        lambda m: len(m.transformer.h),
        lambda m: len(m.layers),
        lambda m: len(m.encoder.layer),
    ]

    for accessor in layer_access_patterns:
        try:
            count = accessor(model)
            if isinstance(count, int) and count > 0:
                return count
        except (AttributeError, TypeError):
            continue

    for name, module in model.named_modules():
        if isinstance(module, nn.ModuleList) and len(module) > 0:
            first_child = list(module.children())[0] if len(list(module.children())) > 0 else None
            if first_child is not None and hasattr(first_child, "self_attn"):
                return len(module)

    raise ValueError(
        "Could not determine layer count. "
        "Model config should have num_hidden_layers or similar attribute."
    )


def reset_dynamo_for_hook_change() -> None:
    """Reset TorchDynamo so compiled graphs re-trace after a hook change.

    A hook added or removed after `torch.compile` may not be reflected in a cached graph, so the
    next forward pass must re-trace. Best-effort: never raises, because failing to clear a cache is
    not a reason to refuse to install a hook.
    """
    try:
        import torch._dynamo as _dynamo

        _dynamo.reset()
        logger.debug("dynamo_reset_after_hook_change")
    except Exception as e:
        logger.debug("dynamo_reset_skipped: %s", e)


def module_path(model: nn.Module, target: nn.Module) -> Optional[str]:
    """The dotted name of `target` inside `model`, or None."""
    return next((name for name, module in model.named_modules() if module is target), None)


def moved_by_the_text_tower_fix(model: Any, layer: int) -> Optional[tuple[Optional[str], Optional[str]]]:
    """`(old_path, new_path)` when index `layer` named a different module before ENH-011 X-1, else None.

    The old answer is `first_match_layer`'s; the new one is `get_layer`'s. A side that cannot resolve is
    None, and two Nones are no move (a model without a module tree, such as a llama.cpp handle, is refused
    elsewhere). On a single-tower model the two always agree.
    """
    if not isinstance(model, nn.Module):
        return None

    def path(resolve: Any) -> Optional[str]:
        try:
            return module_path(model, resolve(model, layer))
        except (ValueError, IndexError, AttributeError, TypeError):
            return None

    old, new = path(first_match_layer), path(get_layer)
    return None if old == new else (old, new)
