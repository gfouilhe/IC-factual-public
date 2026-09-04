"""Path and offline-mode configuration.

This module centralizes:

* The list of supported Pythia sizes (``160m`` ... ``12b``).
* The on-disk layout for staged assets
  (``<assets_root>/models/pythia-<size>``, ``<assets_root>/datasets/...``).
* Environment-variable overrides.

Staged assets are produced by ``scripts/stage_assets.py`` on a node with
internet access and consumed offline by compute jobs.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

# --- Pythia model catalogue -------------------------------------------------

#: Pythia sizes supported by the staging helpers and loaders.
#: Order goes from smallest to largest so we can iterate predictably.
PYTHIA_SIZES: tuple[str, ...] = (
    "160m",
    "410m",
    "1b",
    "1.4b",
    "2.8b",
    "6.9b",
    "12b",
)

#: Qwen3 dense sizes supported by the staging helpers.
#: Same convention as Pythia: lowercase, no separator (``"0.6b"``, ``"1.7b"``, ...).
QWEN3_SIZES: tuple[str, ...] = (
    "0.6b",
    "1.7b",
    "4b",
    "8b",
    "14b",
    "32b",
)

#: Qwen3-Base (pretraining checkpoint) sizes.  Same size tokens as :data:`QWEN3_SIZES`
#: except ``32b``: Qwen never published ``Qwen/Qwen3-32B-Base`` (only the post-trained
#: ``Qwen/Qwen3-32B`` exists on the Hub).
QWEN3_BASE_SIZES: tuple[str, ...] = (
    "0.6b",
    "1.7b",
    "4b",
    "8b",
    "14b",
)

#: Ministral-3 (Dec 2025) variant families on the Hub.
MINISTRAL3_VARIANTS: tuple[str, ...] = (
    "instruct",
    "reasoning",
    "base",
)

#: Ministral-3 dense sizes (Hub names: 3B / 8B / 14B).
MINISTRAL3_SIZES: tuple[str, ...] = (
    "3b",
    "8b",
    "14b",
)

#: OpenAI GPT-2 sizes (``small`` = base 124M through ``xl`` = 1.5B).
GPT2_SIZES: tuple[str, ...] = (
    "small",
    "medium",
    "large",
    "xl",
)


def _size_token_to_billions(size: str) -> float:
    """Parse a size token like ``'0.6b'`` or ``'14b'`` into billions of parameters."""
    token = str(size).strip().lower()
    if token.endswith("b"):
        token = token[:-1]
    return float(token)


#: Approximate parameter counts (billions) for log-scale cross-family plots.
#: Keys match ``evaluate_capitals.py`` ``model_size`` / ``label`` metadata.
MODEL_PARAM_BILLIONS: dict[str, float] = {
    "160m": 0.16,
    "410m": 0.41,
    "1b": 1.0,
    "1.4b": 1.4,
    "2.8b": 2.8,
    "6.9b": 6.9,
    "12b": 12.0,
    **{f"qwen3-{s}": _size_token_to_billions(s) for s in QWEN3_SIZES},
    **{f"qwen3-base-{s}": _size_token_to_billions(s) for s in QWEN3_BASE_SIZES},
    **{
        f"ministral3-{v}-{s}": _size_token_to_billions(s)
        for v in MINISTRAL3_VARIANTS
        for s in MINISTRAL3_SIZES
    },
    "gpt2": 0.124,
    "gpt2-small": 0.124,
    "gpt2-medium": 0.355,
    "gpt2-large": 0.774,
    "gpt2-xl": 1.5,
}

# Hub suffix for each :data:`MINISTRAL3_VARIANTS` entry.
_MINISTRAL3_VARIANT_HUB: dict[str, str] = {
    "instruct": "Instruct",
    "reasoning": "Reasoning",
    "base": "Base",
}

DEFAULT_MODEL_SIZE = "160m"
DEFAULT_DATASET_ID = "gaotang/ParaConflict"

# Legacy alias: scripts that import ``DEFAULT_MODEL_ID`` still get a sensible
# default (the smallest supported Pythia).
DEFAULT_MODEL_ID = f"EleutherAI/pythia-{DEFAULT_MODEL_SIZE}"

# --- Environment variable names ---------------------------------------------

ENV_ASSETS_ROOT = "IC_FACTUAL_ASSETS_ROOT"
ENV_MODELS_ROOT = "IC_FACTUAL_MODELS_ROOT"
ENV_MODEL_SIZE = "IC_FACTUAL_MODEL_SIZE"
ENV_MODEL_PATH = "IC_FACTUAL_MODEL_PATH"
ENV_PARACONFLICT_PATH = "IC_FACTUAL_PARACONFLICT_PATH"
ENV_OFFLINE = "IC_FACTUAL_OFFLINE"
ENV_REQUIRE_OFFLINE = "IC_FACTUAL_REQUIRE_OFFLINE_ASSETS"


# --- Helpers ----------------------------------------------------------------


def _normalize_size(size: str) -> str:
    """Return the canonical lowercase form of a Pythia size string."""
    if size is None:
        raise ValueError("Pythia model size must not be None")
    normalized = str(size).strip().lower()
    if normalized.startswith("pythia-"):
        normalized = normalized[len("pythia-"):]
    if normalized not in PYTHIA_SIZES:
        raise ValueError(
            f"Unsupported Pythia size: {size!r}. "
            f"Expected one of: {', '.join(PYTHIA_SIZES)}"
        )
    return normalized


def pythia_model_id(size: str) -> str:
    """Return the Hugging Face Hub id for a Pythia size, e.g. ``EleutherAI/pythia-1.4b``."""
    return f"EleutherAI/pythia-{_normalize_size(size)}"


def pythia_model_dirname(size: str) -> str:
    """Return the on-disk directory name for a Pythia size, e.g. ``pythia-1.4b``."""
    return f"pythia-{_normalize_size(size)}"


def _normalize_qwen3_size(size: str) -> str:
    """Return the canonical lowercase form of a Qwen3 size string."""
    if size is None:
        raise ValueError("Qwen3 model size must not be None")
    normalized = str(size).strip().lower()
    if normalized.startswith("qwen3-"):
        normalized = normalized[len("qwen3-"):]
    if normalized not in QWEN3_SIZES:
        raise ValueError(
            f"Unsupported Qwen3 size: {size!r}. "
            f"Expected one of: {', '.join(QWEN3_SIZES)}"
        )
    return normalized


def qwen3_model_id(size: str) -> str:
    """Return the Hub id for a post-trained Qwen3 size, e.g. ``Qwen/Qwen3-1.7B``.

    These are the instruction-tuned / thinking variants (no ``-Base`` suffix).
    Qwen3 repo ids use uppercase ``B`` and preserve the dot in fractional sizes
    (``Qwen/Qwen3-0.6B``, ``Qwen/Qwen3-1.7B``, ``Qwen/Qwen3-14B``).
    """
    return f"Qwen/Qwen3-{_normalize_qwen3_size(size).upper()}"


def qwen3_model_dirname(size: str) -> str:
    """Return the on-disk directory name for a post-trained Qwen3 size, e.g. ``qwen3-1.7b``."""
    return f"qwen3-{_normalize_qwen3_size(size)}"


def _normalize_qwen3_base_size(size: str) -> str:
    """Return the canonical lowercase form of a Qwen3-Base size string."""
    if size is None:
        raise ValueError("Qwen3-Base model size must not be None")
    normalized = str(size).strip().lower()
    if normalized.startswith("qwen3-base-"):
        normalized = normalized[len("qwen3-base-"):]
    if normalized not in QWEN3_BASE_SIZES:
        raise ValueError(
            f"Unsupported Qwen3-Base size: {size!r}. "
            f"Expected one of: {', '.join(QWEN3_BASE_SIZES)}"
        )
    return normalized


def qwen3_base_model_id(size: str) -> str:
    """Return the Hub id for a Qwen3-Base size, e.g. ``Qwen/Qwen3-1.7B-Base``."""
    return f"Qwen/Qwen3-{_normalize_qwen3_base_size(size).upper()}-Base"


def qwen3_base_model_dirname(size: str) -> str:
    """Return the on-disk directory name for a Qwen3-Base size, e.g. ``qwen3-base-1.7b``."""
    return f"qwen3-base-{_normalize_qwen3_base_size(size)}"


def _normalize_ministral3_size(size: str) -> str:
    """Return the canonical lowercase form of a Ministral-3 size string."""
    if size is None:
        raise ValueError("Ministral-3 model size must not be None")
    normalized = str(size).strip().lower()
    if normalized not in MINISTRAL3_SIZES:
        raise ValueError(
            f"Unsupported Ministral-3 size: {size!r}. "
            f"Expected one of: {', '.join(MINISTRAL3_SIZES)}"
        )
    return normalized


def _normalize_ministral3_variant(variant: str) -> str:
    """Return the canonical lowercase form of a Ministral-3 variant string."""
    if variant is None:
        raise ValueError("Ministral-3 variant must not be None")
    normalized = str(variant).strip().lower()
    if normalized.startswith("ministral3-"):
        normalized = normalized[len("ministral3-"):]
    if normalized not in MINISTRAL3_VARIANTS:
        raise ValueError(
            f"Unsupported Ministral-3 variant: {variant!r}. "
            f"Expected one of: {', '.join(MINISTRAL3_VARIANTS)}"
        )
    return normalized


def ministral3_model_id(variant: str, size: str) -> str:
    """Return the Hub id, e.g. ``mistralai/Ministral-3-3B-Instruct-2512``."""
    v = _MINISTRAL3_VARIANT_HUB[_normalize_ministral3_variant(variant)]
    s = _normalize_ministral3_size(size).upper()
    return f"mistralai/Ministral-3-{s}-{v}-2512"


def ministral3_model_dirname(variant: str, size: str) -> str:
    """Return the on-disk directory name, e.g. ``ministral3-instruct-3b``."""
    return (
        f"ministral3-{_normalize_ministral3_variant(variant)}"
        f"-{_normalize_ministral3_size(size)}"
    )


_GPT2_SIZE_HUB_SUFFIX: dict[str, str] = {
    "small": "",
    "medium": "-medium",
    "large": "-large",
    "xl": "-xl",
}


def _normalize_gpt2_size(size: str) -> str:
    """Return the canonical lowercase form of a GPT-2 size string."""
    if size is None:
        raise ValueError("GPT-2 model size must not be None")
    normalized = str(size).strip().lower()
    if normalized.startswith("gpt2-"):
        normalized = normalized[len("gpt2-"):]
    if normalized in {"base", ""}:
        normalized = "small"
    if normalized not in GPT2_SIZES:
        raise ValueError(
            f"Unsupported GPT-2 size: {size!r}. "
            f"Expected one of: {', '.join(GPT2_SIZES)}"
        )
    return normalized


def gpt2_model_id(size: str) -> str:
    """Return the Hub id for a GPT-2 size, e.g. ``openai-community/gpt2-xl``."""
    suffix = _GPT2_SIZE_HUB_SUFFIX[_normalize_gpt2_size(size)]
    return f"openai-community/gpt2{suffix}"


def gpt2_model_dirname(size: str) -> str:
    """Return the on-disk directory name, e.g. ``gpt2-xl``."""
    normalized = _normalize_gpt2_size(size)
    if normalized == "small":
        return "gpt2"
    return f"gpt2-{normalized}"


def gpt2_model_label(size: str) -> str:
    """Return the short label used in eval JSONL metadata and plot panels."""
    return gpt2_model_dirname(size)


def ministral3_model_label(variant: str, size: str) -> str:
    """Return the short label used in eval JSONL metadata and plot panels."""
    return ministral3_model_dirname(variant, size)


def ministral3_model_path(variant: str, size: str) -> Path:
    """Directory containing a staged Ministral-3 snapshot."""
    return default_models_root() / ministral3_model_dirname(variant, size)


def model_param_billions(label: str) -> float | None:
    """Return approximate parameter count (billions) for a model label, if known."""
    key = str(label).strip().lower()
    if key in MODEL_PARAM_BILLIONS:
        return MODEL_PARAM_BILLIONS[key]
    if key.startswith("pythia-"):
        bare = key[len("pythia-"):]
        if bare in MODEL_PARAM_BILLIONS:
            return MODEL_PARAM_BILLIONS[bare]
    if key.startswith("gpt2-"):
        bare = key[len("gpt2-"):]
        if bare in MODEL_PARAM_BILLIONS:
            return MODEL_PARAM_BILLIONS[bare]
        if f"gpt2-{bare}" in MODEL_PARAM_BILLIONS:
            return MODEL_PARAM_BILLIONS[f"gpt2-{bare}"]
    return None


def model_family(label: str) -> str:
    """Coarse family string for cross-family plots (``pythia``, ``qwen3``, ...)."""
    key = str(label).strip().lower()
    if key in PYTHIA_SIZES or key.startswith("pythia-"):
        return "pythia"
    if key.startswith("qwen3-base"):
        return "qwen3-base"
    if key.startswith("qwen3"):
        return "qwen3"
    for variant in MINISTRAL3_VARIANTS:
        if key.startswith(f"ministral3-{variant}"):
            return f"ministral3-{variant}"
    if key == "gpt2" or key.startswith("gpt2-"):
        return "gpt2"
    return "other"


def resolve_model_size(size: str | None = None) -> str:
    """Resolve the active Pythia size from (in order): argument, env, default."""
    if size:
        return _normalize_size(size)
    env = os.environ.get(ENV_MODEL_SIZE)
    if env:
        return _normalize_size(env)
    return DEFAULT_MODEL_SIZE


def default_assets_root() -> Path:
    """Root directory for staged models and datasets (shared filesystem on clusters)."""
    raw = os.environ.get(ENV_ASSETS_ROOT)
    if raw:
        return Path(raw).expanduser().resolve()
    return (Path.cwd() / "assets").resolve()


def default_models_root() -> Path:
    """Directory containing one subdirectory per staged Pythia size."""
    raw = os.environ.get(ENV_MODELS_ROOT)
    if raw:
        return Path(raw).expanduser().resolve()
    return default_assets_root() / "models"


def model_path(size: str | None = None) -> Path:
    """Directory containing a Hugging Face Pythia snapshot for ``size``.

    Lookup order:
        1. If ``size`` is omitted and ``IC_FACTUAL_MODEL_PATH`` is set,
           honor that explicit override (back-compat).
        2. Otherwise, ``<models_root>/pythia-<size>`` where ``size`` comes
           from the argument, ``IC_FACTUAL_MODEL_SIZE``, or the default.
    """
    if size is None:
        override = os.environ.get(ENV_MODEL_PATH)
        if override:
            return Path(override).expanduser().resolve()
    resolved = resolve_model_size(size)
    return default_models_root() / pythia_model_dirname(resolved)


def paraconflict_path() -> Path:
    """Directory produced by ``datasets.Dataset.save_to_disk`` for ParaConflict."""
    override = os.environ.get(ENV_PARACONFLICT_PATH)
    if override:
        return Path(override).expanduser().resolve()
    return default_assets_root() / "datasets" / "paraconflict"


def is_offline_mode() -> bool:
    """True when Hub access must not be used (compute nodes without internet)."""
    if os.environ.get(ENV_OFFLINE, "").lower() in ("1", "true", "yes"):
        return True
    if os.environ.get("HF_HUB_OFFLINE", "").lower() in ("1", "true", "yes"):
        return True
    if os.environ.get("TRANSFORMERS_OFFLINE", "").lower() in ("1", "true", "yes"):
        return True
    return False


def require_offline_assets() -> bool:
    """When set, missing staged assets raise instead of falling back to the Hub."""
    return os.environ.get(ENV_REQUIRE_OFFLINE, "").lower() in ("1", "true", "yes")


def apply_offline_env() -> None:
    """Set standard Hugging Face offline flags (idempotent)."""
    if is_offline_mode():
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        os.environ.setdefault("HF_DATASETS_OFFLINE", "1")


def staged_model_available(size: str | None = None) -> bool:
    path = model_path(size)
    return path.is_dir() and (path / "config.json").is_file()


def staged_paraconflict_available() -> bool:
    path = paraconflict_path()
    return path.is_dir() and any(path.iterdir())


def resolve_model_id_or_path(size: str | None = None) -> str:
    """Hub model id when online and unstaged; otherwise local snapshot path.

    ``size`` selects which Pythia variant to resolve.  When omitted the
    active size comes from ``IC_FACTUAL_MODEL_SIZE`` or :data:`DEFAULT_MODEL_SIZE`.
    """
    if staged_model_available(size):
        return str(model_path(size))
    if is_offline_mode() or require_offline_assets():
        raise FileNotFoundError(
            f"Staged Pythia model not found at {model_path(size)}. "
            "On a login node with internet, run: "
            "uv run python scripts/stage_assets.py "
            f"--sizes {resolve_model_size(size)} "
            "(or stage all sizes without --sizes)."
        )
    return pythia_model_id(resolve_model_size(size))


def resolve_paraconflict_source() -> str | Path:
    if staged_paraconflict_available():
        return paraconflict_path()
    if is_offline_mode() or require_offline_assets():
        raise FileNotFoundError(
            f"Staged ParaConflict dataset not found at {paraconflict_path()}. "
            "Run scripts/stage_assets.py on a node with internet access."
        )
    return DEFAULT_DATASET_ID


def list_staged_models() -> list[str]:
    """Return Pythia sizes for which a snapshot is present on disk."""
    return [size for size in PYTHIA_SIZES if staged_model_available(size)]


def resolve_hub_id(label: str) -> str:
    """Map an eval/plot label to a Hugging Face Hub model id."""
    key = str(label).strip().lower()
    if key in PYTHIA_SIZES:
        return pythia_model_id(key)
    if key.startswith("pythia-"):
        return pythia_model_id(key[len("pythia-") :])
    if key == "gpt2":
        return gpt2_model_id("small")
    if key.startswith("gpt2-"):
        return gpt2_model_id(key[len("gpt2-") :])
    if key.startswith("qwen3-base-"):
        return qwen3_base_model_id(key[len("qwen3-base-") :])
    if key.startswith("qwen3-"):
        return qwen3_model_id(key[len("qwen3-") :])
    m = re.fullmatch(r"ministral3-(instruct|reasoning|base)-(\d+b)", key)
    if m:
        return ministral3_model_id(m.group(1), m.group(2))
    raise ValueError(f"Cannot resolve Hub id for label: {label!r}")


# Paper 31 model catalogue: generative eval on capitals & ParaConflict.
FIGURE1_MODEL_LABELS: tuple[str, ...] = (
    *(s for s in PYTHIA_SIZES),
    *(gpt2_model_label(s) for s in GPT2_SIZES),
    *(f"qwen3-{s}" for s in QWEN3_SIZES),
    *(f"qwen3-base-{s}" for s in QWEN3_BASE_SIZES),
    *(
        ministral3_model_label(v, s)
        for v in MINISTRAL3_VARIANTS
        for s in MINISTRAL3_SIZES
    ),
)

# Paper Fig. 3: cross-family logit lens checkpoints.
FIGURE2_MODEL_LABELS: tuple[str, ...] = (
    "2.8b",
    "12b",
    "gpt2",
    "gpt2-xl",
    "qwen3-base-4b",
    "qwen3-4b",
    "qwen3-base-14b",
    "qwen3-14b",
    *(
        ministral3_model_label(v, s)
        for v in MINISTRAL3_VARIANTS
        for s in ("3b", "14b")
    ),
)

# Representative mini-sweeps for smoke testing.
SMOKE_FIGURE1_MODEL_LABELS: tuple[str, ...] = (
    "160m",
)

SMOKE_FIGURE2_MODEL_LABELS: tuple[str, ...] = (
    "160m",
)

# All 12 figure paths under paper/figures/
PAPER_FIGURE_SYNCS: tuple[tuple[str, str], ...] = (
    ("figures/context_memory_score_gen_all.png", "paper/figures/context_memory_score_gen_all.png"),
    ("figures/exp22_gen_length.png", "paper/figures/exp22_gen_length.png"),
    ("figures/logit_lens_family_gap_summary.png", "paper/figures/logit_lens_family_gap_summary.png"),
    ("figures/clean_factual_recall.png", "paper/figures/clean_factual_recall.png"),
    ("figures/clean_factual_recall_lp.png", "paper/figures/clean_factual_recall_lp.png"),
    ("figures/exp22_gen_length_full.png", "paper/figures/exp22_gen_length_full.png"),
    ("figures/cross_family_prose_l128_delta.png", "paper/figures/cross_family_prose_l128_delta.png"),
    ("figures/exp02_position_sweep.png", "paper/figures/exp02_position_sweep.png"),
    ("figures/country_freq_mem_combined.png", "paper/figures/country_freq_mem_combined.png"),
    ("figures/exp12_template_comparison.png", "paper/figures/exp12_template_comparison.png"),
    ("figures/cross_family_phrasing_swings.png", "paper/figures/cross_family_phrasing_swings.png"),
    ("figures/exp07_copies.png", "paper/figures/exp07_copies.png"),
)

# All table paths under paper/tables/
PAPER_TABLE_SYNCS: tuple[tuple[str, str], ...] = (
    ("tables/relations_posttraining_by_size.tex", "paper/tables/relations_posttraining_by_size.tex"),
    ("tables/relations_size_mem_by_family.tex", "paper/tables/relations_size_mem_by_family.tex"),
    ("tables/paraconflict_native_sub_vs_coh.tex", "paper/tables/paraconflict_native_sub_vs_coh.tex"),
    ("tables/country_freq_mem_hypothesis.tex", "paper/tables/country_freq_mem_hypothesis.tex"),
    ("tables/paraconflict_entity_frequency.tex", "paper/tables/paraconflict_entity_frequency.tex"),
    ("tables/model_inference_specs.tex", "paper/tables/model_inference_specs.tex"),
    ("tables/clean_factual_recall.tex", "paper/tables/clean_factual_recall.tex"),
    ("tables/clean_factual_recall_lp.tex", "paper/tables/clean_factual_recall_lp.tex"),
    ("tables/cross_family_phrasing.tex", "paper/tables/cross_family_phrasing.tex"),
    ("tables/cross_family_prose_l128.tex", "paper/tables/cross_family_prose_l128.tex"),
)

