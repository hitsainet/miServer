"""The context ceiling a GGUF load uses: where it comes from, and what is refused (2026-11-21).

Three sources, the first set wins:

1. the load request's `context_ceiling` (`POST /api/models/{id}/load`), for that load only — an
   agent's request carries it in the approval payload, so it is in the `request_digest` and on the
   card the operator approves;
2. the model row's `context_ceiling` (`PUT /api/models/{id}/context-ceiling`), for every load of
   that model, the OpenAI path's auto-loads included;
3. the server's GGUF_CONTEXT_LENGTH.

0 (or "max" on the wire) is as large a window as fits, up to what the model declares: the loader
starts at the predicted fit for the card(s) it places the model on, walks down if that fails and
probes up toward the declared length if it loads (`model_loader.load_gguf_model`).

Refused, with the reason (422 VALIDATION_ERROR), before anything is stored or unloaded: a ceiling on
a model that is not GGUF (the transformers path has no ceiling to apply, and accepting it would be
a setting that silently does nothing), one below GGUF_MIN_CONTEXT, and one above the length the
file declares.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from miserver.core.errors import ValidationError
from miserver.ml.gguf_metadata import GgufMetadataError, read_gguf_metadata

#: The stored and loader value of "max".
CEILING_MAX = 0
CEILING_MAX_WORD = "max"

#: Mirrors `model_loader.GGUF_MIN_CONTEXT`, the smallest window the ladder loads (pinned equal by
#: a test; not imported, so this module loads without torch).
MIN_CONTEXT = 2048

SOURCE_REQUEST = "request"
SOURCE_MODEL = "model"
SOURCE_SERVER = "server"


def parse_ceiling(value: Any) -> Optional[int]:
    """None, "max" (-> 0), or a whole number of tokens >= 0. Raises ValueError for anything else.

    Checked before coercion, so `true` is not read as 1 and "32k" is not guessed at.
    """
    if value is None:
        return None
    if isinstance(value, str):
        if value.strip().lower() == CEILING_MAX_WORD:
            return CEILING_MAX
        raise ValueError(f"context_ceiling must be a number of tokens or 'max', not {value!r}")
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"context_ceiling must be a number of tokens or 'max', not {value!r}")
    if value < 0:
        raise ValueError("context_ceiling cannot be negative; 0 or 'max' is as large as fits")
    return value


def _gguf_path(model: Any) -> Optional[Path]:
    from miserver.services.model_service import resolve_cache_path

    files = getattr(model, "gguf_files", None) or []
    if not files or not getattr(model, "cache_path", None):
        return None
    return Path(resolve_cache_path(model.cache_path)) / files[0]


def declared_length(model: Any) -> Optional[int]:
    """The context the model's GGUF file declares (`<arch>.context_length`), or None if unreadable."""
    path = _gguf_path(model)
    if path is None or not path.is_file():
        return None
    try:
        meta = read_gguf_metadata(str(path))
    except (OSError, GgufMetadataError, ValueError):
        return None
    arch = meta.get("general.architecture")
    value = meta.get(f"{arch}.context_length") if isinstance(arch, str) else None
    return int(value) if isinstance(value, int) and not isinstance(value, bool) and value > 0 else None


def validate_for_model(model: Any, ceiling: Optional[int]) -> None:
    """Refuse a ceiling this model cannot take. None (no ceiling named) is always accepted.

    Reads the file's header (an mmap of its metadata, ~0.1 s on a 16 GiB file): call it off the
    event loop.

    Raises:
        ValidationError: not a GGUF model; below the smallest window; above the declared length.
    """
    if ceiling is None:
        return
    if not (getattr(model, "gguf_files", None) or []):
        raise ValidationError(
            "context_ceiling applies to GGUF models only: a transformers model's window is its "
            "config's max_position_embeddings, and nothing would apply the ceiling.",
            details={"model_id": getattr(model, "id", None), "context_ceiling": ceiling},
        )
    if CEILING_MAX < ceiling < MIN_CONTEXT:
        raise ValidationError(
            f"context_ceiling {ceiling} is below the smallest window miServer loads "
            f"({MIN_CONTEXT}); give at least {MIN_CONTEXT}, or 'max'.",
            details={"model_id": getattr(model, "id", None), "context_ceiling": ceiling,
                     "minimum": MIN_CONTEXT},
        )
    declared = declared_length(model)
    if declared is not None and ceiling > declared:
        raise ValidationError(
            f"context_ceiling {ceiling} is above the {declared} tokens this model declares; a "
            "longer window is not a bigger one, it is past what the model was trained on. Give "
            f"at most {declared}, or 'max' for as large as fits up to it.",
            details={"model_id": getattr(model, "id", None), "context_ceiling": ceiling,
                     "declared": declared},
        )


def effective(model: Any, requested: Optional[int]) -> tuple[Optional[int], str]:
    """(the ceiling to load with, its source). None with SOURCE_SERVER: GGUF_CONTEXT_LENGTH."""
    if requested is not None:
        return requested, SOURCE_REQUEST
    stored = getattr(model, "context_ceiling", None)
    if stored is not None:
        return int(stored), SOURCE_MODEL
    return None, SOURCE_SERVER


def words(ceiling: Optional[int], source: str, server_default: int) -> str:
    """The ceiling as the operator reads it on an approval card."""
    where = {
        SOURCE_REQUEST: "named in this request",
        SOURCE_MODEL: "this model's setting",
        SOURCE_SERVER: "the server default, GGUF_CONTEXT_LENGTH",
    }[source]
    value = server_default if ceiling is None else ceiling
    if value == CEILING_MAX:
        return (f"max: as large a window as fits on the card, up to the model's declared "
                f"length ({where})")
    return f"{value:,} tokens ({where})"
