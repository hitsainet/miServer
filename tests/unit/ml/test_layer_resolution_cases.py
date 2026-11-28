"""ENH-011 X-1 (#2, N6): on a multi-tower model, miServer must hook the TEXT decoder's layers.

`get_layer` tried six fixed accessors and then took the first `nn.ModuleList` whose name holds "layer" and is
long enough. A composite checkpoint (Gemma 3 and 4 `...ForConditionalGeneration`, LLaVA, Qwen2.5-VL) registers its
vision tower before its language model, so on Gemma 4 indices 0-15 resolved to `vision_tower.encoder.layers.N`:
every probe, SAE, steering edit and circuit hooked a layer a text forward never runs, and reported nothing. A
steering edit measured 0.0 against 0.329 on the right layer (R2). `get_layer_count` read the top-level config, which
on a composite has no `num_hidden_layers`, and then counted the first list with a `self_attn` child: the vision
tower again. Two width reads took the top-level `hidden_size`, which a composite does not have: `sae_service`
skipped its dimension check and `probe_capacity_measure` raised (N6).

Every case comes from `docs/schemas/layer-resolution-cases.json`, miWorkbench's file, vendored byte-identically
(`tests/unit/test_layer_resolution_cases_sync.py`), and is built as a tiny model with random weights from its own
config class. miWorkbench tests its resolver on the same cases, so the two cannot drift apart silently.

The helpers the fix adds are read off the module when a test runs, so this file collects on the base and its red
there is the defect's own (a wrong module, a hook that never fires), not an ImportError.
"""

from __future__ import annotations

import copy
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import torch
from torch import nn

from miserver.ml import layer_resolution
from miserver.ml.layer_resolution import get_layer, get_layer_count
from miserver.ml.probe_hooker import ProbeHooker
from miserver.ml.sae_hooker import SAEHooker
from tests.support.layer_cases import (
    case,
    case_ids,
    load_cases,
    module_path,
    text_input_ids,
    tiny_config,
    tiny_model,
)

CASES = load_cases()
COMPOSITE = [c for c in CASES if c["config_class"] != "LlamaConfig"]
#: The composite cases whose vision stack a first-match search took (gemma4_unified has one stack).
TWO_STACKS = [c for c in CASES if "vision_layers_path" in c]


def LayerResolutionError():  # noqa: N802 - stands in for the class, read when the test runs
    return layer_resolution.LayerResolutionError


def _text_layers(c):
    return tiny_model(c).get_submodule(c["text_layers_path"])


def _logits(model, hook=None):
    handle = None
    if hook is not None:
        module, fn = hook
        handle = module.register_forward_hook(fn)
    try:
        with torch.no_grad():
            return model(input_ids=text_input_ids()).logits.clone()
    finally:
        if handle is not None:
            handle.remove()


def _add(delta):
    def edit(_module, _inputs, output):
        if isinstance(output, tuple):
            return (output[0] + delta,) + tuple(output[1:])
        return output + delta
    return edit


def _text_width(c) -> int:
    return tiny_config(c).get_text_config(decoder=True).hidden_size


class TestTheVendoredCasesAreWhatTheModelsHold:
    """The cases file cannot drift from the module trees it describes (control D1: rename a path)."""

    def test_the_file_has_every_family_the_brief_names(self):
        families = {c["model_class"] for c in CASES}
        assert {"Gemma4ForConditionalGeneration", "Gemma3ForConditionalGeneration",
                "Qwen2_5_VLForConditionalGeneration", "LlavaForConditionalGeneration",
                "LlamaForCausalLM"} <= families
        assert all({"family", "config_class", "model_class", "text_layers_path", "text_depth", "tiny_config"}
                   <= set(c) for c in CASES)

    @pytest.mark.parametrize("c", CASES, ids=case_ids)
    def test_the_file_names_every_layer_stack_the_model_has(self, c):
        stacks = [path for path, _ in layer_resolution.transformer_stacks(tiny_model(c))]
        expected = [c["text_layers_path"]] + ([c["vision_layers_path"]] if "vision_layers_path" in c else [])
        assert sorted(stacks) == sorted(expected)

    @pytest.mark.parametrize("c", CASES, ids=case_ids)
    def test_text_depth_is_the_tiny_text_configs_depth(self, c):
        assert tiny_config(c).get_text_config(decoder=True).num_hidden_layers == c["text_depth"]
        assert len(_text_layers(c)) == c["text_depth"]


class TestGetLayerIsTheTextDecoders:
    """AC-1 (control C1: restore first-match-wins)."""

    @pytest.mark.parametrize("c", CASES, ids=case_ids)
    def test_every_index_is_the_text_towers_module(self, c):
        model, layers = tiny_model(c), _text_layers(c)
        resolved = [get_layer(model, i) for i in range(c["text_depth"])]
        assert [module_path(model, m) for m in resolved] == [
            f"{c['text_layers_path']}.{i}" for i in range(c["text_depth"])]
        assert all(m is layers[i] for i, m in enumerate(resolved))

    @pytest.mark.parametrize("c", CASES, ids=case_ids)
    def test_the_count_is_the_text_depth(self, c):
        assert get_layer_count(tiny_model(c)) == c["text_depth"]

    @pytest.mark.parametrize("c", CASES, ids=case_ids)
    def test_an_index_past_the_text_depth_is_refused(self, c):
        with pytest.raises(ValueError):
            get_layer(tiny_model(c), c["text_depth"])

    @pytest.mark.parametrize("c", CASES, ids=case_ids)
    def test_validate_layer_accepts_the_last_text_layer_and_no_further(self, c):
        hooker, model = SAEHooker(), tiny_model(c)
        assert hooker.validate_layer(model, c["text_depth"] - 1) is True
        assert hooker.validate_layer(model, c["text_depth"]) is False

    @pytest.mark.parametrize("c", CASES, ids=case_ids)
    def test_the_layer_device_is_the_text_layers(self, c):
        model = tiny_model(c)
        assert layer_resolution.layer_device(model, 1) == next(_text_layers(c)[1].parameters()).device

    @pytest.mark.parametrize("c", CASES, ids=case_ids)
    def test_a_probe_hook_at_layer_1_fires_once_on_a_text_forward(self, c):
        """The payload and the count: one call, the text layer's own output, at the text width."""
        model, hooker = tiny_model(c), ProbeHooker()
        seen: list[torch.Tensor] = []
        handle = hooker.install(model, 1, seen.append)
        try:
            _logits(model)
        finally:
            hooker.remove(handle)
        assert len(seen) == 1, f"layer 1 fired {len(seen)} times on a text-only forward"
        assert seen[0].shape == (1, text_input_ids().shape[1], _text_width(c))
        assert hooker.last_resolved_module_path == f"{c['text_layers_path']}.1"

    @pytest.mark.parametrize("c", CASES, ids=case_ids)
    def test_an_edit_through_get_layer_1_moves_the_logits(self, c):
        """R2 in miniature: an edit on a layer the text forward never runs changes nothing (0.0)."""
        model = tiny_model(c)
        layer = get_layer(model, 1)
        base = _logits(model)
        edited = _logits(model, hook=(layer, _add(torch.full((_text_width(c),), 0.5))))
        assert torch.equal(base, _logits(model)), "the edit hook must not outlive its forward"
        assert (edited - base).abs().max().item() > 1e-4

    def test_the_single_tower_control_resolves_exactly_as_before(self):
        llama = case("llama")
        model = tiny_model(llama)
        assert [get_layer(model, i) for i in range(3)] == list(model.model.layers)
        assert get_layer_count(model) == 3
        assert get_layer(model, -1) is model.model.layers[-1], "negative indices index as before"


class TestADepthMismatchIsRefused:
    """AC-2 (control C2: drop the depth cross-check)."""

    @pytest.mark.parametrize("family", ["gemma3", "llama"])
    def test_a_text_config_that_disagrees_with_the_stack_refuses(self, family):
        c = case(family)
        model = copy.deepcopy(tiny_model(c))
        model.config.get_text_config(decoder=True).num_hidden_layers = c["text_depth"] + 1
        for call in (lambda: get_layer(model, 0), lambda: get_layer_count(model)):
            with pytest.raises(LayerResolutionError()) as refused:
                call()
            message = str(refused.value)
            assert c["text_layers_path"] in message
            assert f"{c['text_depth']} layers" in message
            assert f"num_hidden_layers={c['text_depth'] + 1}" in message
        assert SAEHooker().validate_layer(model, 0) is False, "a refused model validates no layer"

    def test_the_refusal_is_a_value_error_so_existing_callers_still_refuse(self):
        assert issubclass(LayerResolutionError(), ValueError)

    def test_a_config_that_states_no_depth_is_not_refused(self):
        """Only a stated, different number refuses; a config that states none cannot disagree."""
        model = copy.deepcopy(tiny_model(case("llama")))
        model.config = SimpleNamespace(model_type="llama")
        assert get_layer(model, 2) is model.model.layers[2]
        assert get_layer_count(model) == 3


class _Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = nn.Linear(4, 4)
        self.mlp = nn.Linear(4, 4)


class TestTwoStacksThatCannotBeToldApartAreRefused:

    def test_no_embedding_owner_and_no_decoder_refuses(self):
        class TwoTowers(nn.Module):
            def __init__(self):
                super().__init__()
                self.vision_layers = nn.ModuleList([_Block(), _Block()])
                self.text_layers = nn.ModuleList([_Block(), _Block(), _Block()])

        for call in (lambda: get_layer(TwoTowers(), 0), lambda: get_layer_count(TwoTowers())):
            with pytest.raises(LayerResolutionError(), match="vision_layers.*text_layers"):
                call()

    def test_a_list_nested_inside_a_layer_is_not_a_second_tower(self):
        class Inner(_Block):
            def __init__(self):
                super().__init__()
                self.sub = nn.ModuleList([_Block()])

        class Flat(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = nn.ModuleList([Inner(), Inner()])

        model = Flat()
        assert [p for p, _ in layer_resolution.transformer_stacks(model)] == ["layers"]
        assert get_layer(model, 1) is model.layers[1]
        assert get_layer_count(model) == 2


def _planted(family: str = "gemma3"):
    """A composite whose TOP-LEVEL config states a width (48) that is not the text tower's (32).

    A stock composite config on 5.20.2 carries no top-level `hidden_size`, so a test on it agrees with a top-level
    read by construction. This is what an older or a hand-written composite config.json looks like.
    """
    model = copy.deepcopy(tiny_model(case(family)))
    model.config.hidden_size = 48
    return model


class TestWidthsComeFromTheTextConfig:
    """AC-3 (controls C3: top-level first in `probe_capacity_measure`; C4: the same in `sae_service`)."""

    def test_text_hidden_size_reads_the_text_tower(self):
        assert layer_resolution.text_hidden_size(_planted().config) == 32
        assert layer_resolution.text_hidden_size(tiny_config(case("llama"))) == 32
        assert layer_resolution.text_hidden_size(SimpleNamespace(d_model=16)) == 16
        assert layer_resolution.text_hidden_size(SimpleNamespace(hidden_size=True)) is None
        assert layer_resolution.text_hidden_size(None) is None

    @staticmethod
    def _measure(monkeypatch, model):
        from miserver.services import probe_capacity_measure as measure_mod
        from miserver.services import probe_cost

        seen = []

        def spy(probe, *, device, d_model, k, dtype, **_kw):
            seen.append((probe.layer, d_model, int(probe.head.weight.shape[-1])))
            return SimpleNamespace(decode_ms=0.001, prefill_ms_per_1k=0.001, layer_fixed_ms=0.001,
                                   layer_fixed_prefill_ms_per_1k=0.001)

        monkeypatch.setattr(probe_cost, "measure", spy)
        monkeypatch.setattr(measure_mod, "_clock", lambda fn, devices: (fn(), 1.0)[1])
        measure_mod.measure_resident(model, prefill_tokens=8, context_tokens=4, steps=1, repeats=1, k=1)
        return seen

    @pytest.mark.parametrize("c", COMPOSITE, ids=case_ids)
    def test_measure_resident_runs_on_a_composite_at_the_text_width(self, c, monkeypatch):
        """N6: `getattr(model.config, "d_model")` raised on every composite config."""
        seen = self._measure(monkeypatch, tiny_model(c))
        middle = c["text_depth"] // 2
        assert seen == [(middle, 32, 32)] * 3, "one synthetic probe per rule line, at the text width"

    def test_measure_resident_ignores_a_planted_top_level_width(self, monkeypatch):
        assert self._measure(monkeypatch, _planted()) == [(1, 32, 32)] * 3

    @staticmethod
    async def _compatibility(model, *, d_in: int, layer: int):
        from miserver.db.models.sae import SAE, SAEStatus
        from miserver.services.sae_service import SAEService

        sae = MagicMock(spec=SAE)
        sae.id, sae.d_in, sae.status = "sae_x", d_in, SAEStatus.CACHED
        sae.trained_layer, sae.trained_on = None, None
        repo = MagicMock()
        repo.get = AsyncMock(return_value=sae)
        with patch("miserver.services.sae_service.SAEDownloader"), \
             patch("miserver.services.sae_service.SAELoader"):
            service = SAEService(repository=repo, cache_dir="/data/saes", emitter=MagicMock())
        state = MagicMock()
        state.is_loaded = True
        state.current = SimpleNamespace(model=model, supports_hooks=True, engine="transformers")
        with patch("miserver.services.sae_service.LoadedModelState", return_value=state):
            return await service.check_compatibility("sae_x", layer)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("c", TWO_STACKS, ids=case_ids)
    async def test_sae_compatibility_checks_the_text_width_and_depth(self, c):
        model, last = tiny_model(c), c["text_depth"] - 1
        wrong = await self._compatibility(model, d_in=16, layer=last)
        assert wrong.compatible is False
        assert wrong.errors == ["Dimension mismatch: SAE d_in=16, model hidden_size=32"]
        right = await self._compatibility(model, d_in=32, layer=last)
        assert (right.compatible, right.errors) == (True, [])

    @pytest.mark.asyncio
    async def test_sae_compatibility_ignores_a_planted_top_level_width(self):
        answer = await self._compatibility(_planted(), d_in=32, layer=2)
        assert (answer.compatible, answer.errors) == (True, [])
        mismatch = await self._compatibility(_planted(), d_in=48, layer=2)
        assert mismatch.errors == ["Dimension mismatch: SAE d_in=48, model hidden_size=32"]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("c", COMPOSITE + [case("llama")], ids=case_ids)
    async def test_a_probes_loaded_identity_reads_the_text_width_and_depth(self, c, monkeypatch):
        """Arming compares a probe's `d_model`/`n_layers` with these; on a composite they were None and every
        probe was refused as "cannot read the loaded model's width and depth"."""
        from miserver.ml import model_loader
        from miserver.services import probe_arm_bridge

        current = SimpleNamespace(model=tiny_model(c), supports_hooks=True, engine="transformers", model_id=1,
                                  model_name="tiny", tokenizer=None, dtype="float32")
        monkeypatch.setattr(model_loader, "LoadedModelState", lambda: SimpleNamespace(current=current))
        monkeypatch.setattr(probe_arm_bridge, "_model_row", AsyncMock(return_value=None))
        identity, model, _ = await probe_arm_bridge.loaded_identity(None)
        assert (identity.d_model, identity.n_layers) == (32, c["text_depth"])
        assert model is current.model

    @pytest.mark.asyncio
    async def test_a_probes_loaded_identity_ignores_a_planted_top_level_width(self, monkeypatch):
        from miserver.ml import model_loader
        from miserver.services import probe_arm_bridge

        current = SimpleNamespace(model=_planted(), supports_hooks=True, engine="transformers", model_id=1,
                                  model_name="tiny", tokenizer=None, dtype="float32")
        monkeypatch.setattr(model_loader, "LoadedModelState", lambda: SimpleNamespace(current=current))
        monkeypatch.setattr(probe_arm_bridge, "_model_row", AsyncMock(return_value=None))
        identity, _, _ = await probe_arm_bridge.loaded_identity(None)
        assert (identity.d_model, identity.n_layers) == (32, 3)
