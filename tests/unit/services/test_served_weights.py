"""`resident_identity.served_weights` (043 FR-043.1): is the resident parameter set the base?

The fact is false only for a plain row load (no identity, origin `model`) whose barrier counter
reads exactly 0. Any other origin, any commit, and any counter that cannot be read is "differs".
"""

from __future__ import annotations

import typing
from datetime import datetime
from types import SimpleNamespace

import pytest

import miserver.services.weight_barrier as barrier_module
from miserver.services import resident_identity
from miserver.services.resident_identity import (
    DIFFERENT_ORIGINS,
    NOTHING_LOADED,
    Origin,
    ResidentIdentity,
    ServedWeights,
    served_weights,
)

#: A value no real counter takes, read back to prove the monkeypatch reached the lookup.
SENTINEL_STEPS = 7_919


def _loaded(identity=None):
    return SimpleNamespace(model_id=1, model_name="tiny", weights_identity=identity,
                           loaded_at=datetime(2026, 11, 16))


def _identity(origin: str) -> ResidentIdentity:
    return ResidentIdentity(name="org/tiny:run:step-1", base_model="org/tiny", origin=origin,
                            checkpoint_dir="run/step-000001", manifest_sha256="a" * 64,
                            expected_parent=None, commits_at_identity=0)


@pytest.fixture
def steps(monkeypatch):
    """Point the barrier lookup the function makes AT CALL TIME at a stand-in snapshot.

    `served_weights` imports `get_weight_barrier` inside its body, so the module attribute is
    the name it reads (memory `monkeypatches-that-silently-do-nothing`)."""

    def set_steps(value):
        if isinstance(value, BaseException):
            def get():
                raise value
        else:
            def get():
                return SimpleNamespace(snapshot=lambda: SimpleNamespace(steps_since_load=value))
        monkeypatch.setattr(barrier_module, "get_weight_barrier", get)

    return set_steps


def test_the_monkeypatch_took_effect(steps):
    steps(SENTINEL_STEPS)
    fact = served_weights(_loaded())
    assert fact.commits_since_load == SENTINEL_STEPS and fact.differs_from_base is True


def test_a_plain_row_load_with_no_commits_is_the_base(steps):
    steps(0)
    fact = served_weights(_loaded())
    assert fact == ServedWeights("tiny", "model", 0, False)
    assert fact.as_dict() == {"name": "tiny", "origin": "model", "commits_since_load": 0,
                              "differs_from_base": False}


def test_one_commit_differs(steps):
    steps(1)
    fact = served_weights(_loaded())
    assert fact == ServedWeights("tiny", "model", 1, True)


def test_a_checkpoint_origin_differs_with_the_counter_at_zero(steps):
    steps(0)
    fact = served_weights(_loaded(_identity("checkpoint")))
    assert fact.differs_from_base is True
    assert fact.origin == "checkpoint" and fact.name == "org/tiny:run:step-1"


def test_a_raising_barrier_is_unknown_and_differs(steps):
    steps(RuntimeError("barrier unavailable"))
    fact = served_weights(_loaded())
    assert fact.commits_since_load is None
    assert fact.differs_from_base is True, "an unreadable counter must never read as 0"


@pytest.mark.parametrize("value", [None, "0", 0.0, False])
def test_a_counter_that_is_not_an_int_is_unknown(steps, value):
    steps(value)
    fact = served_weights(_loaded())
    assert fact.commits_since_load is None and fact.differs_from_base is True


def test_nothing_loaded(monkeypatch):
    monkeypatch.setattr(resident_identity, "_current_loaded", lambda: None)
    assert served_weights() is NOTHING_LOADED
    assert NOTHING_LOADED.as_dict() == {"name": None, "origin": "none",
                                        "commits_since_load": None, "differs_from_base": False}


def test_every_origin_is_the_base_or_differs():
    """No hand-kept list: a new `Origin` value must be placed in `DIFFERENT_ORIGINS` or be `model`."""
    origins = set(typing.get_args(Origin))
    assert origins, "Origin is a Literal with values"
    assert origins == {"model"} | set(DIFFERENT_ORIGINS)
    assert "model" not in DIFFERENT_ORIGINS


@pytest.mark.parametrize("origin", sorted(DIFFERENT_ORIGINS))
def test_each_different_origin_differs_at_zero_commits(steps, origin):
    steps(0)
    assert served_weights(_loaded(_identity(origin))).differs_from_base is True
