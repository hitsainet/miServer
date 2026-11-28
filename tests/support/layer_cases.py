"""The shared layer-resolution cases (`docs/schemas/layer-resolution-cases.json`) as tiny models.

miWorkbench publishes the file (ENH-011 X-1) and miServer keeps a byte-identical copy
(`tests/unit/test_layer_resolution_cases_sync.py` pins it). Each case names a transformers config
class, a model class and the keyword arguments of a CPU-sized copy with random weights
(`tiny_config`). Nothing is downloaded and no miWorkbench code is imported: the model is built from
its family's own config class, so both repos test their resolvers on the same module trees.
"""

from __future__ import annotations

import json
from functools import lru_cache
from typing import Any

from tests.support.sibling_repos import REPO_ROOT

CASES_PATH = REPO_ROOT / "docs" / "schemas" / "layer-resolution-cases.json"


def load_cases() -> list[dict[str, Any]]:
    return json.loads(CASES_PATH.read_text())["cases"]


def case_ids(case: dict[str, Any]) -> str:
    return case["family"]


def case(family: str) -> dict[str, Any]:
    return next(c for c in load_cases() if c["family"] == family)


def tiny_config(case: dict[str, Any]):
    import transformers

    return getattr(transformers, case["config_class"])(**case["tiny_config"])


@lru_cache(maxsize=None)
def _built(family: str):
    import torch
    import transformers

    found = case(family)
    torch.manual_seed(0)
    model = getattr(transformers, found["model_class"])(tiny_config(found))
    model.eval()
    return model


def tiny_model(case: dict[str, Any]):
    """The case's tiny model, built once per process (eval mode, seed 0). Callers must not mutate
    it: deep-copy it first."""
    return _built(case["family"])


def text_input_ids(length: int = 6):
    """A text-only prompt: ids well inside the tiny vocabulary (128) and clear of the image tokens."""
    import torch

    return torch.arange(3, 3 + length).unsqueeze(0)


def module_path(model, target) -> str | None:
    return next((name for name, module in model.named_modules() if module is target), None)


__all__ = ["CASES_PATH", "case", "case_ids", "load_cases", "module_path", "text_input_ids",
           "tiny_config", "tiny_model"]
