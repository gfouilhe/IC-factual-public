"""Load models and the ParaConflict dataset from staged assets or the Hub."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset, load_dataset, load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer

from ic_factual.config import (
    DEFAULT_DATASET_ID,
    apply_offline_env,
    is_offline_mode,
    resolve_model_id_or_path,
    resolve_model_size,
    resolve_paraconflict_source,
    staged_model_available,
)


_DTYPE_ALIASES: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "fp32": torch.float32,
    "float16": torch.float16,
    "fp16": torch.float16,
    "half": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
}


def _local_files_only(size: str | None = None) -> bool:
    apply_offline_env()
    return is_offline_mode() or staged_model_available(size)


def resolve_dtype(dtype: str | torch.dtype | None) -> torch.dtype | None:
    """Translate a string alias to a ``torch.dtype``; pass through dtypes/None.

    Accepted aliases (case-insensitive): ``float32``/``fp32``, ``float16``/
    ``fp16``/``half``, ``bfloat16``/``bf16``.  ``None``, ``"auto"`` and
    ``""`` resolve to ``None`` (let HF pick).
    """
    if dtype is None or isinstance(dtype, torch.dtype):
        return dtype
    key = str(dtype).strip().lower()
    if key in {"auto", ""}:
        return None
    if key not in _DTYPE_ALIASES:
        raise ValueError(
            f"Unsupported dtype alias: {dtype!r}. "
            f"Expected one of: {sorted(_DTYPE_ALIASES)} or a torch.dtype."
        )
    return _DTYPE_ALIASES[key]


# Backwards-compatible private alias (used by older callers).
_resolve_dtype = resolve_dtype


def _snapshot_config(path: str | Path) -> dict[str, Any] | None:
    cfg_path = Path(path).expanduser() / "config.json"
    if not cfg_path.is_file():
        return None
    try:
        return json.loads(cfg_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def is_ministral3_snapshot(path: str | Path) -> bool:
    """Return True when ``path`` points at a Ministral-3 / Mistral3 checkpoint."""
    cfg = _snapshot_config(path)
    if cfg and cfg.get("model_type") == "mistral3":
        return True
    name = Path(path).name.lower()
    if name.startswith("ministral3-"):
        return True
    lowered = str(path).lower()
    return "ministral-3-" in lowered and "mistralai/" in lowered


def _ministral3_is_fp8(config: Any) -> bool:
    qcfg = getattr(config, "quantization_config", None)
    if qcfg is None:
        return False
    quant_method = getattr(qcfg, "quant_method", None)
    if quant_method is None and isinstance(qcfg, dict):
        quant_method = qcfg.get("quant_method")
    return quant_method == "fp8"


def _load_ministral3_tokenizer(model_id_or_path: str, *, local_files_only: bool):
    from transformers import MistralCommonBackend

    return MistralCommonBackend.from_pretrained(
        model_id_or_path,
        local_files_only=local_files_only,
    )


def _load_ministral3_causal_lm(
    model_id_or_path: str,
    *,
    local_files_only: bool,
    dtype: torch.dtype | None,
    from_pretrained_kwargs: dict[str, Any],
):
    from transformers import AutoConfig, Mistral3ForConditionalGeneration

    config = AutoConfig.from_pretrained(
        model_id_or_path,
        local_files_only=local_files_only,
    )
    kwargs: dict[str, Any] = dict(from_pretrained_kwargs)
    kwargs.setdefault("local_files_only", local_files_only)

    if _ministral3_is_fp8(config):
        # Instruct checkpoints ship FP8 weights; dequantize when the caller
        # asks for a floating dtype, otherwise keep native FP8 kernels.
        if dtype is not None:
            from transformers import FineGrainedFP8Config

            kwargs.setdefault(
                "quantization_config",
                FineGrainedFP8Config(dequantize=True),
            )
            kwargs.setdefault("torch_dtype", dtype)
    elif dtype is not None:
        kwargs.setdefault("torch_dtype", dtype)

    return Mistral3ForConditionalGeneration.from_pretrained(
        model_id_or_path,
        config=config,
        **kwargs,
    )


def load_tokenizer(
    model_id_or_path: str | None = None,
    *,
    model_size: str | None = None,
):
    """Load a tokenizer for the active (or specified) model."""
    if model_id_or_path is None:
        model_id_or_path = resolve_model_id_or_path(model_size)
    local_only = _local_files_only(model_size)
    if is_ministral3_snapshot(model_id_or_path):
        return _load_ministral3_tokenizer(model_id_or_path, local_files_only=local_only)
    return AutoTokenizer.from_pretrained(
        model_id_or_path,
        local_files_only=local_only,
    )


def load_causal_lm(
    model_id_or_path: str | None = None,
    device: str | None = None,
    *,
    model_size: str | None = None,
    dtype: str | torch.dtype | None = None,
    **from_pretrained_kwargs: Any,
):
    """Load a causal LM, moved to the appropriate device.

    Parameters
    ----------
    model_id_or_path:
        Explicit Hub id or local snapshot path; takes precedence over ``model_size``.
    device:
        Optional device override (``"cuda"``, ``"cuda:0"``, ``"cpu"``, ...).
    model_size:
        Pythia size shortcut (``"160m"``, ``"410m"``, ...).  Falls back to
        ``IC_FACTUAL_MODEL_SIZE`` and then ``DEFAULT_MODEL_SIZE``.
    dtype:
        Optional torch dtype (alias string or ``torch.dtype``).  Recommended
        for the larger Pythia variants (``"float16"`` for 6.9B/12B on a
        single 80 GB A100).  For Ministral-3 Instruct (FP8), ``bf16``/
        ``fp16`` triggers on-the-fly dequantization.
    """
    from ic_factual.device import resolve_device

    if model_id_or_path is None:
        model_id_or_path = resolve_model_id_or_path(model_size)
    target = resolve_device(device)
    resolved_dtype = resolve_dtype(dtype)

    local_only = _local_files_only(model_size)
    kwargs: dict[str, Any] = dict(from_pretrained_kwargs)
    kwargs.setdefault("local_files_only", local_only)

    if is_ministral3_snapshot(model_id_or_path):
        model = _load_ministral3_causal_lm(
            model_id_or_path,
            local_files_only=local_only,
            dtype=resolved_dtype,
            from_pretrained_kwargs=kwargs,
        )
    else:
        if resolved_dtype is not None:
            kwargs.setdefault("torch_dtype", resolved_dtype)
        model = AutoModelForCausalLM.from_pretrained(model_id_or_path, **kwargs)

    model.to(target)
    model.eval()
    return model, target


def load_paraconflict(split: str = "test") -> Dataset:
    source = resolve_paraconflict_source()
    apply_offline_env()

    if isinstance(source, Path):
        stored = load_from_disk(str(source))
        if hasattr(stored, "keys") and split in stored:
            return stored[split]
        return stored

    return load_dataset(DEFAULT_DATASET_ID, split=split)


__all__ = [
    "is_ministral3_snapshot",
    "load_tokenizer",
    "load_causal_lm",
    "load_paraconflict",
    "resolve_dtype",
    "resolve_model_size",
]
