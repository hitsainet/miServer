"""ENH-011 X-1 (Q1 under OQ-3): an artefact recorded before the text-tower fix is refused where its layer moved.

Migration 043 stamps every probe, circuit and profile row that exists when it runs (`recorded_before_text_tower`).
On the live model, `layer_resolution.moved_by_the_text_tower_fix` compares what the old first-match resolver hooked at
an index with what `get_layer` hooks now; a stamped row is refused (`LAYER_MEANING_CHANGED`, 409) only where they
differ. Every door is driven here through its real function, and each has a control that removes its call.
"""

from __future__ import annotations

import ast
import inspect
import textwrap
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from miserver.core.errors import LayerMeaningChangedError
from miserver.services import layer_meaning
from tests.support.layer_cases import case, tiny_model

STAMP = {"stamped_by": "migration 043", "reason": "recorded before ENH-011 X-1"}


def _row(stamp=STAMP, layer: int = 1, **extra):
    return SimpleNamespace(id="pr_x", layer=layer, recorded_before_text_tower=stamp, **extra)


@pytest.fixture
def gemma3():
    return tiny_model(case("gemma3"))


@pytest.fixture
def llama():
    return tiny_model(case("llama"))


class TestTheGate:
    def test_a_stamped_row_on_a_composite_is_refused_with_both_paths(self, gemma3):
        with pytest.raises(LayerMeaningChangedError) as refused:
            layer_meaning.refuse_if_changed("probe", _row(), gemma3, [1])
        error = refused.value
        assert (error.code, error.status_code) == ("LAYER_MEANING_CHANGED", 409)
        assert error.details == {
            "kind": "probe", "id": "pr_x", "recorded_before": "migration 043",
            "layers": [{"layer": 1, "old_module_path": "model.vision_tower.encoder.layers.1",
                        "new_module_path": "model.language_model.layers.1"}],
        }
        assert "model.vision_tower.encoder.layers.1" in str(error)

    @pytest.mark.parametrize("family,layer,old", [("qwen2_5_vl", 0, "model.visual.blocks.0"),
                                                  ("llava", 1, "model.vision_tower.encoder.layers.1"),
                                                  ("gemma4", 0, "model.vision_tower.encoder.layers.0")])
    def test_every_two_stack_case_moves(self, family, layer, old):
        with pytest.raises(LayerMeaningChangedError) as refused:
            layer_meaning.refuse_if_changed("circuit", _row(), tiny_model(case(family)), [layer])
        assert refused.value.details["layers"] == [
            {"layer": layer, "old_module_path": old, "new_module_path": f"model.language_model.layers.{layer}"}]

    def test_only_the_layers_that_moved_are_named(self, gemma3):
        """The tiny Gemma 3 vision tower has 2 layers: index 2 resolved to the text decoder before the fix too."""
        with pytest.raises(LayerMeaningChangedError) as refused:
            layer_meaning.refuse_if_changed("circuit", _row(), gemma3, [2, 0, 2])
        assert [m["layer"] for m in refused.value.details["layers"]] == [0]
        layer_meaning.refuse_if_changed("circuit", _row(), gemma3, [2])

    def test_an_unstamped_row_is_never_refused(self, gemma3):
        for stamp in (None, {}, "x"):
            layer_meaning.refuse_if_changed("probe", _row(stamp=stamp), gemma3, [0, 1])

    def test_a_single_tower_model_never_moves(self, llama):
        layer_meaning.refuse_if_changed("probe", _row(), llama, [0, 1, 2])

    def test_a_one_stack_composite_never_moves(self):
        layer_meaning.refuse_if_changed("probe", _row(), tiny_model(case("gemma4_unified")), [0, 1, 2])

    def test_no_model_or_no_module_tree_decides_nothing(self):
        layer_meaning.refuse_if_changed("probe", _row(), None, [1])
        layer_meaning.refuse_if_changed("probe", _row(), object(), [1])

    def test_the_resident_model_is_the_loaded_transformers_model(self, monkeypatch, gemma3):
        from miserver.ml import model_loader

        state = SimpleNamespace(current=SimpleNamespace(model=gemma3, supports_hooks=True))
        monkeypatch.setattr(model_loader, "LoadedModelState", lambda: state)
        assert layer_meaning.resident_model() is gemma3
        state.current.supports_hooks = False
        assert layer_meaning.resident_model() is None
        state.current = None
        assert layer_meaning.resident_model() is None


class TestTheProbeArmingDoor:
    """`ProbeArmingService.check` is arm, dry run and the behaviour set's per-probe gate (control G1)."""

    @staticmethod
    async def _check(monkeypatch, probe, model):
        from miserver.services import probe_arming

        monkeypatch.setattr(probe_arming, "identity_refusal", lambda probe, loaded: None)
        service = probe_arming.ProbeArmingService.__new__(probe_arming.ProbeArmingService)
        return await service.check(probe, model=model, loaded=None, forward=None, executor=None)

    @pytest.mark.asyncio
    async def test_a_stamped_probe_is_refused_before_any_other_gate(self, monkeypatch, gemma3):
        probe = _row(rung=0, scope="all")  # rung 0 would refuse next; the layer gate must come first
        with pytest.raises(LayerMeaningChangedError):
            await self._check(monkeypatch, probe, gemma3)

    @pytest.mark.asyncio
    async def test_an_unstamped_probe_passes_to_the_next_gate(self, monkeypatch, gemma3):
        from miserver.core.errors import UnvalidatedProbeError

        with pytest.raises(UnvalidatedProbeError):
            await self._check(monkeypatch, _row(stamp=None, rung=0, scope="all"), gemma3)

    def test_the_gate_is_called_once_with_the_probes_own_layer_and_the_checked_model(self):
        from miserver.services import probe_arming

        calls = _calls(probe_arming.ProbeArmingService.check, "refuse_if_changed")
        assert [[ast.unparse(a) for a in c.args] for c in calls] == [["'probe'", "probe", "model", "[probe.layer]"]]


class TestTheProbeScoringDoor:
    """`ProbeScoringService._resolve`: given ids refuse, omitted ids skip with the code (control G2)."""

    @staticmethod
    def _service(rows):
        from miserver.services.probe_scoring import ProbeScoringService

        repo = MagicMock()
        repo.get = AsyncMock(side_effect=lambda pid: next(r for r in rows if r.id == pid))
        repo.list = AsyncMock(return_value=rows)
        return ProbeScoringService(repo, inference=None)

    @pytest.fixture(autouse=True)
    def _gates_pass(self, monkeypatch):
        from miserver.services import probe_arm_bridge, probe_arming

        monkeypatch.setattr(probe_arming, "identity_refusal", lambda row, loaded: None)
        monkeypatch.setattr(probe_arming, "scope_refusal", lambda row: None)
        monkeypatch.setattr(probe_arm_bridge, "build_probe_encoder", AsyncMock(return_value=None))

    @pytest.mark.asyncio
    async def test_a_named_stamped_probe_is_refused(self, gemma3):
        request = SimpleNamespace(probe_ids=["pr_x"], behaviours=None)
        with pytest.raises(LayerMeaningChangedError):
            await self._service([_row()])._resolve(request, None, None, model=gemma3)

    @pytest.mark.asyncio
    async def test_an_unnamed_stamped_probe_is_skipped_with_the_code(self, gemma3):
        fresh = SimpleNamespace(id="pr_new", layer=1, recorded_before_text_tower=None)
        request = SimpleNamespace(probe_ids=None, behaviours=None)
        kept, _encoders, skipped = await self._service([_row(), fresh])._resolve(request, None, None, model=gemma3)
        assert [r.id for r in kept] == ["pr_new"]
        assert [(s["probe_id"], s["code"]) for s in skipped] == [("pr_x", "LAYER_MEANING_CHANGED")]

    def test_score_hands_the_resident_model_to_the_gate(self):
        """The CALLER (memory `mutation-survivors-cluster-on-callers`): `_score` passes the model it scores on."""
        from miserver.services.probe_scoring import ProbeScoringService

        calls = _calls(ProbeScoringService._score, "_resolve")
        assert len(calls) == 1
        assert {k.arg: ast.unparse(k.value) for k in calls[0].keywords} == {"model": "model"}
        inner = _calls(ProbeScoringService._resolve, "refuse_if_changed")
        assert [[ast.unparse(a) for a in c.args] for c in inner] == [["'probe'", "row", "model", "[row.layer]"]]


class TestTheCircuitDoor:
    """`CircuitService.activate` (control G3)."""

    @staticmethod
    def _service(circuit):
        from miserver.services.circuit_service import CircuitService

        service = CircuitService(repository=MagicMock())
        service.get = AsyncMock(return_value=circuit)
        service._parse_stored = MagicMock(side_effect=RuntimeError("past the layer gate"))
        return service

    @pytest.mark.asyncio
    async def test_a_stamped_circuit_is_refused_on_a_composite(self, monkeypatch, gemma3):
        monkeypatch.setattr(layer_meaning, "resident_model", lambda: gemma3)
        circuit = SimpleNamespace(id="crc_x", name="c", rung=2, layers=[1, 2],
                                  recorded_before_text_tower=STAMP)
        with pytest.raises(LayerMeaningChangedError) as refused:
            await self._service(circuit).activate("crc_x")
        assert refused.value.details["kind"] == "circuit"
        assert [m["layer"] for m in refused.value.details["layers"]] == [1]

    @pytest.mark.asyncio
    async def test_an_unstamped_circuit_goes_on_to_serve(self, monkeypatch, gemma3):
        monkeypatch.setattr(layer_meaning, "resident_model", lambda: gemma3)
        circuit = SimpleNamespace(id="crc_x", name="c", rung=2, layers=[1], recorded_before_text_tower=None)
        with pytest.raises(RuntimeError, match="past the layer gate"):
            await self._service(circuit).activate("crc_x")


class TestTheProfileDoor:
    """`ProfileService._validate_activation`, the gate every cluster and profile activation shares (control G4)."""

    @staticmethod
    def _validate(profile, target):
        from miserver.services.profile_service import ProfileService

        return ProfileService(repository=MagicMock(), sae_service=MagicMock())._validate_activation(profile, target)

    @staticmethod
    def _profile(stamp=STAMP, layer=1):
        return SimpleNamespace(id="prof_x", name="p", layer=layer, source_kind="cluster", cluster_meta={},
                               steering={}, recorded_before_text_tower=stamp)

    def test_a_stamped_cluster_is_refused_on_a_composite(self, monkeypatch, gemma3):
        monkeypatch.setattr(layer_meaning, "resident_model", lambda: gemma3)
        target = SimpleNamespace(sae=SimpleNamespace(d_sae=16), layer=1)
        with pytest.raises(LayerMeaningChangedError) as refused:
            self._validate(self._profile(), target)
        assert refused.value.details["kind"] == "profile"

    def test_an_unbound_cluster_is_judged_at_the_attachments_layer(self, monkeypatch, gemma3):
        monkeypatch.setattr(layer_meaning, "resident_model", lambda: gemma3)
        target = SimpleNamespace(sae=SimpleNamespace(d_sae=16), layer=0)
        with pytest.raises(LayerMeaningChangedError) as refused:
            self._validate(self._profile(layer=None), target)
        assert refused.value.details["layers"][0]["layer"] == 0

    def test_an_unstamped_cluster_and_a_single_tower_model_pass(self, monkeypatch, gemma3, llama):
        target = SimpleNamespace(sae=SimpleNamespace(d_sae=16), layer=1)
        monkeypatch.setattr(layer_meaning, "resident_model", lambda: gemma3)
        assert self._validate(self._profile(stamp=None), target) is None
        monkeypatch.setattr(layer_meaning, "resident_model", lambda: llama)
        assert self._validate(self._profile(), target) is None


def _calls(fn, name: str) -> list[ast.Call]:
    tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            called = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
            if called == name:
                found.append(node)
    return found
