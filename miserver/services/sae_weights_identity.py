"""Is the attached SAE the one a circuit or cluster definition was built on? (ENH-011 X-2, #12)

A definition binds by `mistudio_sae_id`, `layer`, `hook_type`, `n_features` and `d_model`, which another SAE of the
same shape (another training run, another release of one repository) matches just as well. miWorkbench now writes
`DefinitionSAERef.weights_sha256`, the sha256 of the weights file the definition was built on; the attached SAE's is
`LoadedSAE.weights_sha256`, recorded by `SAELoader.load` from the file it read.

When both are known and differ, the bind says so. It is a WARNING, never a refusal, in this first release (planner
decision E11-6): the circuit keeps its `warn` verdict (still serveable) and the cluster still binds. When either is
unknown (an older export, an SAE with no local copy, a loader that recorded nothing) nothing is said: absent means
"not recorded", never a mismatch.
"""

from __future__ import annotations

import re
from typing import Any

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _known(value: Any) -> str | None:
    """`value` when it is a sha256 in lowercase hex, else None (a test double's attribute is not a hash)."""
    return value if isinstance(value, str) and _SHA256.fullmatch(value) else None


def weights_mismatch(declared: Any, attached_sae: Any) -> str | None:
    """The warning text when the definition's hash and the attached SAE's are both known and differ, else None."""
    want = _known(declared)
    have = _known(getattr(attached_sae, "weights_sha256", None))
    if want is None or have is None or want == have:
        return None
    return (
        f"the definition was built on SAE weights sha256 {want[:12]}…, but the attached SAE's weights are "
        f"{have[:12]}…: a different SAE of the same shape (bound anyway; a warning, not a refusal)"
    )
