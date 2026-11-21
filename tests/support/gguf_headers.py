"""Stand-in GGUF metadata from real files, and a writer for small GGUF headers.

The two metadata sets are the architecture keys of real headers, read 2026-11-21 by range
request (the first 24 MiB of each file), tokenizer and provenance keys left out:

* ``QWEN36_27B``: ``unsloth/Qwen3.6-27B-GGUF``, ``Qwen3.6-27B-Q4_K_M.gguf``: miServer model 17,
  16,817,244,384 bytes = 16,038 MiB, the weights figure of its live load on 2026-11-21.
* ``GEMMA4_31B``: ``mradermacher/gemma-4-31b-it-3MPER0RR-abliterated-i1-GGUF``,
  ``…i1-IQ4_XS.gguf``: the model behind every gemma measurement in ``model_loader``.
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Any

QWEN36_27B: dict[str, Any] = {
    "general.architecture": "qwen35",
    "qwen35.block_count": 64,
    "qwen35.context_length": 262144,
    "qwen35.embedding_length": 5120,
    "qwen35.feed_forward_length": 17408,
    "qwen35.attention.head_count": 24,
    "qwen35.attention.head_count_kv": 4,
    "qwen35.attention.key_length": 256,
    "qwen35.attention.value_length": 256,
    "qwen35.ssm.conv_kernel": 4,
    "qwen35.ssm.state_size": 128,
    "qwen35.ssm.group_count": 16,
    "qwen35.ssm.time_step_rank": 48,
    "qwen35.ssm.inner_size": 6144,
    "qwen35.full_attention_interval": 4,
}

_GEMMA_SWA = ([True] * 5 + [False]) * 10

GEMMA4_31B: dict[str, Any] = {
    "general.architecture": "gemma4",
    "gemma4.block_count": 60,
    "gemma4.context_length": 262144,
    "gemma4.embedding_length": 5376,
    "gemma4.feed_forward_length": 21504,
    "gemma4.attention.head_count": 32,
    "gemma4.attention.head_count_kv": [4 if not swa else 16 for swa in _GEMMA_SWA],
    "gemma4.attention.key_length": 512,
    "gemma4.attention.value_length": 512,
    "gemma4.attention.sliding_window": 1024,
    "gemma4.attention.sliding_window_pattern": list(_GEMMA_SWA),
    "gemma4.attention.key_length_swa": 256,
    "gemma4.attention.value_length_swa": 256,
}

#: Measured live loads on the RTX 3090 (24,576 MiB).
QWEN36_27B_WEIGHTS_MB = 16038
QWEN36_27B_F16_32K_IN_USE_MB = 19206


def _value(value: Any) -> tuple[int, bytes]:
    if isinstance(value, bool):
        return 7, struct.pack("<?", value)
    if isinstance(value, int):
        return 4, struct.pack("<I", value) if value >= 0 else struct.pack("<i", value)
    if isinstance(value, float):
        return 6, struct.pack("<f", value)
    if isinstance(value, str):
        raw = value.encode()
        return 8, struct.pack("<Q", len(raw)) + raw
    if isinstance(value, list):
        kinds = {_value(v)[0] for v in value} or {4}
        (kind,) = kinds
        body = b"".join(_value(v)[1] for v in value)
        return 9, struct.pack("<IQ", kind, len(value)) + body
    raise TypeError(type(value))


def gguf_header_bytes(meta: dict[str, Any], version: int = 3) -> bytes:
    """A GGUF header carrying `meta` and no tensors."""
    out = bytearray(b"GGUF" + struct.pack("<IQQ", version, 0, len(meta)))
    for key, value in meta.items():
        raw = key.encode()
        kind, body = _value(value)
        out += struct.pack("<Q", len(raw)) + raw + struct.pack("<I", kind) + body
    return bytes(out)


def write_gguf(path: Path, meta: dict[str, Any], *, weights_mb: int = 0) -> Path:
    """Write a GGUF header for `meta`, made `weights_mb` long (sparse) so it sizes like a model."""
    path.write_bytes(gguf_header_bytes(meta))
    if weights_mb:
        with open(path, "r+b") as handle:
            handle.truncate(weights_mb * 1024 * 1024)
    return path
