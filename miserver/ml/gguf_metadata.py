"""A GGUF file's metadata, read from its header, and the KV-cache geometry it implies.

Why this reads the file itself rather than asking llama.cpp: `llama_model_meta_val_str` skips
every ARRAY-valued key when the model loads its hparams, and the keys that say which layers hold
a KV cache are arrays on the models that need them most — `gemma4.attention.head_count_kv` (16 on
the sliding-window layers, 4 on the global ones), `gemma4.attention.sliding_window_pattern`, and a
per-layer `head_count_kv` with zeros for the recurrent layers of Jamba-style hybrids. Through the
binding they are invisible, and `llama_model_n_head_kv` answers for layer 0 alone.

The header is read through `mmap`, so a 16 GiB file costs the pages of its metadata and nothing
else; no weights are read, no VRAM is touched, and llama.cpp is not needed at all.

THE GEOMETRY (`kv_geometry`) follows what llama.cpp allocates, per layer:

* a layer holds a KV cache that grows with the context when it is an ATTENTION layer: for
  `qwen3next`/`qwen35`-style hybrids, every `full_attention_interval`-th layer (llama.cpp:
  `recurrent_layer_arr[il] = (il + 1) % interval != 0`); otherwise every layer whose
  `head_count_kv` is not 0;
* its K and V widths are `head_count_kv(il) x key_length` and `x value_length` — the
  `_swa` variants on a sliding-window layer — defaulting to `embedding_length / head_count`,
  llama.cpp's own default when the file gives no length;
* a sliding-window layer is counted at the FULL context: llama-cpp-python passes llama.cpp's
  default `swa_full = true`, and the gemma-4-31b measurements on the RTX 3090 (2026-09) match a
  full-size cache on those layers;
* a recurrent layer holds a FIXED state per sequence, `(d_conv - 1) x (d_inner + 2 x n_group x
  d_state)` conv cells plus `d_state x d_inner` SSM cells, f32 (llama.cpp `n_embd_r`/`n_embd_s`),
  known only when the file has the `ssm.*` keys.

MEASURED, model 17 (`unsloth/Qwen3.6-27B-GGUF`, Q4_K_M, arch `qwen35`), header read 2026-11-21:
64 layers, `full_attention_interval` 4, `head_count_kv` 4, `key_length` = `value_length` = 256,
`embedding_length` 5120, `head_count` 24. The previous formula, `2 x n_layer x n_head_kv x
(n_embd // n_head)`, gave 2 x 64 x 4 x 213 elements a token: four times the layers that hold a
cache, and a head width of 213 that is no layer's. 16 x 4 x 512 is the truth, 3.33x less.
"""

from __future__ import annotations

import functools
import mmap
import os
import struct
from dataclasses import dataclass
from typing import Any, Optional

GGUF_MAGIC = b"GGUF"

#: GGUF value types -> (struct format, size). 8 is a string, 9 an array.
_SCALARS: dict[int, tuple[str, int]] = {
    0: ("<B", 1), 1: ("<b", 1), 2: ("<H", 2), 3: ("<h", 2), 4: ("<I", 4), 5: ("<i", 4),
    6: ("<f", 4), 7: ("<?", 1), 10: ("<Q", 8), 11: ("<q", 8), 12: ("<d", 8),
}
_STRING = 8
_ARRAY = 9

#: Arrays longer than this are stepped over, not kept. The tokenizer's vocabulary is a quarter of
#: a million strings and nothing here reads it; every per-layer array is one value a layer.
_ARRAY_KEEP = 4096

#: Bytes per element of a KV-cache entry, by cache type. q8_0 and q4_0 carry an f16 scale per
#: 32-element block, which the fractions include.
KV_BYTES_PER_ELEMENT: dict[str, float] = {"f16": 2.0, "q8_0": 1.0 + 2 / 32, "q4_0": 0.5 + 2 / 32}

#: llama.cpp keeps recurrent state in f32 (`type_r`/`type_s` default).
_RECURRENT_STATE_BYTES = 4


class GgufMetadataError(ValueError):
    """The file is not a GGUF file this reader understands (v2 or v3), or its header is cut short."""


class _Reader:
    def __init__(self, buf: Any) -> None:
        self.buf = buf
        self.pos = 0

    def unpack(self, fmt: str, size: int) -> Any:
        if self.pos + size > len(self.buf):
            raise GgufMetadataError("the GGUF header ends early")
        (value,) = struct.unpack_from(fmt, self.buf, self.pos)
        self.pos += size
        return value

    def string(self) -> str:
        n = self.unpack("<Q", 8)
        end = self.pos + n
        if end > len(self.buf):
            raise GgufMetadataError("the GGUF header ends early")
        text = bytes(self.buf[self.pos:end]).decode("utf-8", "replace")
        self.pos = end
        return text

    def skip_string(self) -> None:
        n = self.unpack("<Q", 8)
        self.pos += n

    def value(self, kind: int) -> Any:
        if kind in _SCALARS:
            fmt, size = _SCALARS[kind]
            return self.unpack(fmt, size)
        if kind == _STRING:
            return self.string()
        if kind == _ARRAY:
            inner = self.unpack("<I", 4)
            count = self.unpack("<Q", 8)
            if count > _ARRAY_KEEP:
                self._skip_array(inner, count)
                return None
            return [self.value(inner) for _ in range(count)]
        raise GgufMetadataError(f"unknown GGUF value type {kind}")

    def _skip_array(self, inner: int, count: int) -> None:
        if inner in _SCALARS:
            self.pos += _SCALARS[inner][1] * count
        elif inner == _STRING:
            for _ in range(count):
                self.skip_string()
        else:
            for _ in range(count):
                self.value(inner)
        if self.pos > len(self.buf):
            raise GgufMetadataError("the GGUF header ends early")


def parse_gguf_metadata(buf: Any) -> dict[str, Any]:
    """Every key-value pair of a GGUF header in `buf` (bytes or an mmap). Long arrays are None.

    Raises:
        GgufMetadataError: not GGUF v2/v3, or the header is cut short.
    """
    reader = _Reader(buf)
    if len(buf) < 24 or bytes(buf[0:4]) != GGUF_MAGIC:
        raise GgufMetadataError("not a GGUF file")
    reader.pos = 4
    version = reader.unpack("<I", 4)
    if version not in (2, 3):
        # v1 used 32-bit lengths; nothing current writes it.
        raise GgufMetadataError(f"GGUF version {version} is not read here")
    reader.unpack("<Q", 8)  # tensor count
    n_kv = reader.unpack("<Q", 8)
    meta: dict[str, Any] = {}
    for _ in range(n_kv):
        key = reader.string()
        kind = reader.unpack("<I", 4)
        meta[key] = reader.value(kind)
    return meta


def read_gguf_metadata(path: str) -> dict[str, Any]:
    """The header metadata of the GGUF file at `path`, read through mmap.

    Raises:
        GgufMetadataError: as `parse_gguf_metadata`; OSError: the file cannot be opened.
    """
    with open(path, "rb") as handle:
        if os.fstat(handle.fileno()).st_size < 24:
            raise GgufMetadataError("not a GGUF file")
        with mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as mapped:
            return parse_gguf_metadata(mapped)


@dataclass(frozen=True)
class KvGeometry:
    """What a GGUF model's KV cache and recurrent state cost, from its metadata."""

    architecture: str
    n_layer: int
    #: Layers whose KV cache grows with the context.
    kv_layers: int
    #: K plus V elements one token costs, summed over those layers.
    kv_elements_per_token: int
    #: Layers that hold a recurrent (SSM / linear-attention) state instead.
    recurrent_layers: int
    #: That state for ONE sequence, in bytes; fixed in the context length. 0 when the file does
    #: not say (no `ssm.*` keys) or the model has no recurrent layer.
    recurrent_state_bytes_per_seq: int
    #: K plus V elements one token costs in the WIDEST single attention layer. Flash attention
    #: over a quantized cache converts one layer's whole K and V to f16 in its compute buffer
    #: (`compute_bytes_per_token`), so this, not the sum, is what that buffer grows by.
    widest_layer_kv_elements_per_token: int = 0

    def bytes_per_token(self, kv_cache_type: str) -> float:
        per_element = KV_BYTES_PER_ELEMENT.get((kv_cache_type or "f16").strip().lower(), 2.0)
        return float(self.kv_elements_per_token * per_element)

    def recurrent_state_mb(self, n_seq: int = 1) -> float:
        return self.recurrent_state_bytes_per_seq * max(int(n_seq), 1) / (1024 * 1024)

    def compute_bytes_per_token(self, kv_cache_type: str, n_ubatch: int = 512) -> float:
        """Compute-buffer bytes one token of WINDOW costs (not of prompt), on each card.

        llama.cpp reserves its compute buffers when the context is created, sized for `n_ubatch`
        tokens against the full cache (`llama_context::sched_reserve`, `memory->init_full()`), so
        two of them grow with the window (llama-cpp-python 0.3.36's vendored llama.cpp):
        * the f16 KQ mask, `n_ubatch` x 2 bytes for every cache cell;
        * with a QUANTIZED cache, flash attention's f16 copy of the layer's whole K and V
          (`ggml_cuda_flash_attn_ext_get_alloc_size`: the tile and MMA kernels need f16, and
          prompt processing uses them), 2 bytes an element of the widest layer.
        An f16 cache needs no copy. Model 17 at q8_0: 2 x (4 KV heads x (256 + 256)) = 4,096 for the
        copy + 1,024 for the mask = 5,120 B a token of window. MEASURED 2026-11-21 on the RTX 3090:
        between 32,768 tokens (17,112 MiB) and 222,208 (24,040 MiB) the card used 38,347 B a token,
        3,531 more than the 34,816 B cache: the term is real, and 5,120 bounds it.
        """
        mask = 2.0 * max(int(n_ubatch), 1)
        name = (kv_cache_type or "f16").strip().lower()
        if name == "f16":
            return mask
        return mask + 2.0 * self.widest_layer_kv_elements_per_token

    def to_dict(self) -> dict[str, Any]:
        return {
            "architecture": self.architecture,
            "n_layer": self.n_layer,
            "kv_layers": self.kv_layers,
            "kv_elements_per_token": self.kv_elements_per_token,
            "recurrent_layers": self.recurrent_layers,
            "recurrent_state_mb_per_sequence": round(self.recurrent_state_mb(1), 1),
        }


def _int(value: Any) -> Optional[int]:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _per_layer(value: Any, n_layer: int) -> Optional[list[int]]:
    """A scalar for every layer, or a per-layer array; None when it cannot be one."""
    if isinstance(value, list):
        if len(value) != n_layer:
            return None
        out = [_int(v) for v in value]
        return None if any(v is None for v in out) else out  # type: ignore[misc]
    scalar = _int(value)
    return None if scalar is None else [scalar] * n_layer


def kv_geometry(meta: dict[str, Any]) -> Optional[KvGeometry]:
    """The KV-cache geometry `meta` (a GGUF header's key-value pairs) implies, or None.

    None means the metadata does not give the terms (an architecture key missing, a per-layer
    array of the wrong length): the caller falls back. See the module docstring for the rules.
    """
    arch = meta.get("general.architecture")
    if not isinstance(arch, str) or not arch:
        return None

    def key(name: str) -> Any:
        return meta.get(f"{arch}.{name}")

    n_layer = _int(key("block_count"))
    if not n_layer or n_layer <= 0:
        return None
    n_head = _per_layer(key("attention.head_count"), n_layer)
    if n_head is None:
        return None
    raw_kv = key("attention.head_count_kv")
    n_head_kv = n_head if raw_kv is None else _per_layer(raw_kv, n_layer)
    if n_head_kv is None:
        return None
    n_embd = _int(key("embedding_length"))
    widest = max(n_head) if n_head else 0
    default_width = (n_embd // widest) if n_embd and widest else None
    k_len = _int(key("attention.key_length")) or default_width
    v_len = _int(key("attention.value_length")) or default_width
    if not k_len or not v_len:
        return None
    k_swa = _int(key("attention.key_length_swa")) or k_len
    v_swa = _int(key("attention.value_length_swa")) or v_len
    pattern = key("attention.sliding_window_pattern")
    swa = (
        [bool(v) for v in pattern]
        if isinstance(pattern, list) and len(pattern) == n_layer
        else [False] * n_layer
    )
    interval = _int(key("full_attention_interval"))

    elements = 0
    widest_layer = 0
    kv_layers = 0
    recurrent = 0
    for il in range(n_layer):
        if interval and interval > 0:
            attention = (il + 1) % interval == 0
        else:
            attention = n_head_kv[il] > 0
        if not attention:
            recurrent += 1
            continue
        kv_layers += 1
        width_k, width_v = (k_swa, v_swa) if swa[il] else (k_len, v_len)
        elements += n_head_kv[il] * (width_k + width_v)
        widest_layer = max(widest_layer, n_head_kv[il] * (width_k + width_v))

    state = 0
    d_conv = _int(key("ssm.conv_kernel"))
    d_inner = _int(key("ssm.inner_size"))
    d_state = _int(key("ssm.state_size"))
    if recurrent and d_conv and d_inner and d_state:
        n_group = _int(key("ssm.group_count")) or 0
        cells = (max(d_conv - 1, 0) * (d_inner + 2 * n_group * d_state)) + d_state * d_inner
        state = cells * _RECURRENT_STATE_BYTES * recurrent

    return KvGeometry(
        architecture=arch,
        n_layer=n_layer,
        kv_layers=kv_layers,
        kv_elements_per_token=elements,
        recurrent_layers=recurrent,
        recurrent_state_bytes_per_seq=state,
        widest_layer_kv_elements_per_token=widest_layer,
    )


@functools.lru_cache(maxsize=32)
def _geometry_of(path: str, size: int, mtime_ns: int) -> Optional[KvGeometry]:
    del size, mtime_ns  # the cache key: a replaced file is read again
    return kv_geometry(read_gguf_metadata(path))


def gguf_kv_geometry(path: str) -> Optional[KvGeometry]:
    """`kv_geometry` of the file at `path`, cached per (path, size, mtime). None on any failure."""
    try:
        stat = os.stat(path)
        return _geometry_of(str(path), stat.st_size, stat.st_mtime_ns)
    except (OSError, GgufMetadataError, ValueError, struct.error):
        return None
