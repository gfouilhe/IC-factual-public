#!/usr/bin/env python3
"""Download Pythia + Qwen3 + Ministral-3 models and the ParaConflict dataset on a node with internet.

Stores artifacts under ``IC_FACTUAL_ASSETS_ROOT`` (default ``./assets``) so they
can be consumed offline by compute jobs.

By default this stages **all** supported Pythia sizes (160m -> 12b), **all**
post-trained Qwen3 sizes (0.6B -> 32B), **all** Qwen3-Base sizes (0.6B -> 32B),
**all** Ministral-3 variants (instruct / reasoning / base at 3B / 8B / 14B),
plus the ParaConflict dataset.  Use ``--sizes`` to restrict Pythia,
``--qwen3-sizes`` / ``--qwen3-base-sizes`` to restrict Qwen3 families,
``--ministral3-variants`` / ``--ministral3-sizes`` for Ministral-3, or
``--skip-pythia`` / ``--skip-qwen3`` / ``--skip-qwen3-base`` / ``--skip-ministral3``
to drop a family.

``--models-root`` places per-snapshot directories under a different root than
the shared assets root (useful when keeping large weights on ``$MYTMP`` and the
dataset on ``/work``).

Examples
--------
Stage everything to the default assets root::

    uv run python scripts/stage_assets.py --assets-root /shared/$USER/ic-factual/assets

Stage only the smallest Pythia models and skip Qwen3::

    uv run python scripts/stage_assets.py --sizes 160m 410m --skip-qwen3

Stage only the Qwen3-Base family (skip post-trained Qwen3 and Pythia)::

    uv run python scripts/stage_assets.py --skip-pythia --skip-qwen3 --skip-dataset

Stage all weights into ``$MYTMP`` while keeping the dataset on ``/work``::

    uv run python scripts/stage_assets.py \\
        --models-root "$MYTMP/ic-factual/models" \\
        --assets-root /work/$USER/ic-factual/assets
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# Allow running as `python scripts/stage_assets.py` without installing the package.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from ic_factual.config import (  # noqa: E402
    DEFAULT_DATASET_ID,
    ENV_ASSETS_ROOT,
    ENV_MODELS_ROOT,
    GPT2_SIZES,
    MINISTRAL3_SIZES,
    MINISTRAL3_VARIANTS,
    PYTHIA_SIZES,
    QWEN3_BASE_SIZES,
    QWEN3_SIZES,
    gpt2_model_dirname,
    gpt2_model_id,
    ministral3_model_dirname,
    ministral3_model_id,
    pythia_model_dirname,
    pythia_model_id,
    qwen3_base_model_dirname,
    qwen3_base_model_id,
    qwen3_model_dirname,
    qwen3_model_id,
)


def _snapshot_is_complete(dest: Path) -> bool:
    """Return True iff ``dest`` looks like a fully-downloaded snapshot.

    Requires ``config.json`` plus, when the repo is sharded, every shard
    listed in ``model.safetensors.index.json``.  A snapshot that was
    interrupted mid-download (e.g. only the second shard present) will be
    reported as incomplete so ``--skip-existing`` does not silently keep it.
    """
    if not (dest / "config.json").is_file():
        return False

    index = dest / "model.safetensors.index.json"
    if index.is_file():
        try:
            data = json.loads(index.read_text())
        except (OSError, ValueError):
            return False
        shards = set((data.get("weight_map") or {}).values())
        if not shards:
            return False
        return all((dest / shard).is_file() for shard in shards)

    # Non-sharded: a single weights file is enough.
    return any(dest.glob("*.safetensors")) or any(dest.glob("*.bin"))


# Files we want from each Pythia repository.  We deliberately skip the legacy
# ``pytorch_model.bin`` shards when safetensors are present (they would double
# the on-disk footprint of the largest models).
_MODEL_ALLOW_PATTERNS: tuple[str, ...] = (
    "*.json",
    "*.txt",
    "*.model",
    "tokenizer*",
    "special_tokens_map.json",
    "*.safetensors",
    "*.safetensors.index.json",
)


def _snapshot_download(repo_id: str, dest: Path) -> Path:
    """Download a single HF repo into ``dest`` using ``snapshot_download``.

    Tries safetensors-only first; falls back to ``.bin`` weights when no
    safetensors shipped (older / less common repos).
    """
    from huggingface_hub import snapshot_download

    dest.mkdir(parents=True, exist_ok=True)

    print(f"[stage] Downloading {repo_id} -> {dest}", flush=True)
    # Modern huggingface_hub copies files directly into ``local_dir`` (no
    # symlinks to the HF cache), so the staged snapshot is self-contained.
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(dest),
        allow_patterns=list(_MODEL_ALLOW_PATTERNS),
    )

    # If the repo only ships pytorch_model.bin (no safetensors), retry once
    # with that pattern so we end up with a usable snapshot.
    has_weights = any(dest.glob("*.safetensors")) or any(dest.glob("*.bin"))
    if not has_weights:
        print(
            f"[stage] No safetensors found for {repo_id}, fetching .bin weights...",
            flush=True,
        )
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(dest),
            allow_patterns=["*.bin", "*.bin.index.json"],
        )
    return dest


def stage_model(models_root: Path, size: str) -> Path:
    """Download a single Pythia variant via ``huggingface_hub.snapshot_download``."""
    dest = models_root / pythia_model_dirname(size)
    return _snapshot_download(pythia_model_id(size), dest)


def stage_qwen3(models_root: Path, size: str) -> Path:
    """Download a single post-trained Qwen3 variant via ``snapshot_download``."""
    dest = models_root / qwen3_model_dirname(size)
    return _snapshot_download(qwen3_model_id(size), dest)


def stage_qwen3_base(models_root: Path, size: str) -> Path:
    """Download a single Qwen3-Base variant via ``snapshot_download``."""
    dest = models_root / qwen3_base_model_dirname(size)
    return _snapshot_download(qwen3_base_model_id(size), dest)


def stage_ministral3(models_root: Path, variant: str, size: str) -> Path:
    """Download a single Ministral-3 variant via ``snapshot_download``."""
    dest = models_root / ministral3_model_dirname(variant, size)
    return _snapshot_download(ministral3_model_id(variant, size), dest)


def stage_gpt2(models_root: Path, size: str) -> Path:
    """Download a single GPT-2 variant via ``snapshot_download``."""
    dest = models_root / gpt2_model_dirname(size)
    return _snapshot_download(gpt2_model_id(size), dest)


def stage_paraconflict(assets_root: Path, dataset_id: str) -> Path:
    from datasets import load_dataset

    dest = assets_root / "datasets" / "paraconflict"
    dest.mkdir(parents=True, exist_ok=True)

    print(f"[stage] Downloading dataset {dataset_id} -> {dest}", flush=True)
    dataset = load_dataset(dataset_id)
    dataset.save_to_disk(str(dest))
    return dest


def write_manifest(
    assets_root: Path,
    models_root: Path,
    pythia_dirs: dict[str, Path],
    qwen3_dirs: dict[str, Path],
    qwen3_base_dirs: dict[str, Path],
    ministral3_dirs: dict[str, Path],
    gpt2_dirs: dict[str, Path],
    dataset_dir: Path,
) -> Path:
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_id": DEFAULT_DATASET_ID,
        "dataset_path": str(dataset_dir),
        "models_root": str(models_root),
        "models": {
            size: {"hub_id": pythia_model_id(size), "path": str(path)}
            for size, path in pythia_dirs.items()
        },
        "qwen3": {
            size: {"hub_id": qwen3_model_id(size), "path": str(path)}
            for size, path in qwen3_dirs.items()
        },
        "qwen3_base": {
            size: {"hub_id": qwen3_base_model_id(size), "path": str(path)}
            for size, path in qwen3_base_dirs.items()
        },
        "ministral3": {
            key: {
                "hub_id": ministral3_model_id(
                    key.rsplit("-", 1)[0], key.rsplit("-", 1)[1]
                ),
                "path": str(path),
            }
            for key, path in ministral3_dirs.items()
        },
        "gpt2": {
            size: {"hub_id": gpt2_model_id(size), "path": str(path)}
            for size, path in gpt2_dirs.items()
        },
        "env": {
            "IC_FACTUAL_ASSETS_ROOT": str(assets_root),
            "IC_FACTUAL_MODELS_ROOT": str(models_root),
            "IC_FACTUAL_PARACONFLICT_PATH": str(dataset_dir),
        },
    }
    path = assets_root / "manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return path


def _parse_family_sizes(
    values: list[str] | None,
    catalogue: tuple[str, ...],
    *,
    family: str,
    prefix: str,
) -> list[str]:
    """Parse ``--sizes``-style values against a catalogue (Pythia or Qwen3)."""
    if not values:
        return list(catalogue)
    requested: list[str] = []
    for raw in values:
        for token in raw.replace(",", " ").split():
            token = token.strip().lower()
            if token in {"all", "*"}:
                requested.extend(catalogue)
                continue
            if token.startswith(prefix):
                token = token[len(prefix):]
            if token not in catalogue:
                raise SystemExit(
                    f"Unsupported {family} size: {raw!r}. "
                    f"Expected one of: {', '.join(catalogue)}"
                )
            requested.append(token)
    seen: set[str] = set()
    out: list[str] = []
    for s in requested:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def _parse_sizes(values: list[str] | None) -> list[str]:
    return _parse_family_sizes(values, PYTHIA_SIZES, family="Pythia", prefix="pythia-")


def _parse_qwen3_sizes(values: list[str] | None) -> list[str]:
    return _parse_family_sizes(values, QWEN3_SIZES, family="Qwen3", prefix="qwen3-")


def _parse_qwen3_base_sizes(values: list[str] | None) -> list[str]:
    return _parse_family_sizes(
        values, QWEN3_BASE_SIZES, family="Qwen3-Base", prefix="qwen3-base-"
    )


def _parse_ministral3_variants(values: list[str] | None) -> list[str]:
    return _parse_family_sizes(
        values, MINISTRAL3_VARIANTS, family="Ministral-3 variant", prefix="ministral3-"
    )


def _parse_ministral3_sizes(values: list[str] | None) -> list[str]:
    return _parse_family_sizes(
        values, MINISTRAL3_SIZES, family="Ministral-3", prefix=""
    )


def _parse_gpt2_sizes(values: list[str] | None) -> list[str]:
    return _parse_family_sizes(values, GPT2_SIZES, family="GPT-2", prefix="gpt2-")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage Pythia models + ParaConflict for offline cluster runs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--assets-root",
        type=Path,
        default=None,
        help=(
            "Directory for datasets/ and manifest.json "
            "(default: $IC_FACTUAL_ASSETS_ROOT or ./assets)."
        ),
    )
    parser.add_argument(
        "--models-root",
        type=Path,
        default=None,
        help=(
            "Directory for per-size model snapshots "
            "(default: $IC_FACTUAL_MODELS_ROOT or <assets_root>/models)."
        ),
    )
    parser.add_argument(
        "--sizes",
        nargs="+",
        default=None,
        metavar="SIZE",
        help=(
            "Pythia sizes to stage. Accepts space- or comma-separated values, "
            "or 'all'. "
            f"Default: all of {', '.join(PYTHIA_SIZES)}."
        ),
    )
    parser.add_argument(
        "--qwen3-sizes",
        nargs="+",
        default=None,
        metavar="SIZE",
        help=(
            "Post-trained Qwen3 sizes to stage. Accepts space- or comma-separated "
            "values, or 'all'. "
            f"Default: all of {', '.join(QWEN3_SIZES)}."
        ),
    )
    parser.add_argument(
        "--qwen3-base-sizes",
        nargs="+",
        default=None,
        metavar="SIZE",
        help=(
            "Qwen3-Base (pretraining checkpoint) sizes to stage. Accepts space- or "
            "comma-separated values, or 'all'. "
            f"Default: all of {', '.join(QWEN3_BASE_SIZES)}."
        ),
    )
    parser.add_argument("--dataset-id", default=DEFAULT_DATASET_ID)
    parser.add_argument(
        "--skip-models",
        action="store_true",
        help="Do not download any model weights (Pythia, Qwen3, Qwen3-Base, Ministral-3).",
    )
    parser.add_argument(
        "--skip-pythia",
        action="store_true",
        help="Do not download any Pythia weights.",
    )
    parser.add_argument(
        "--skip-qwen3",
        action="store_true",
        help="Do not download post-trained Qwen3 weights.",
    )
    parser.add_argument(
        "--skip-qwen3-base",
        action="store_true",
        help="Do not download Qwen3-Base weights.",
    )
    parser.add_argument(
        "--ministral3-variants",
        nargs="+",
        default=None,
        metavar="VARIANT",
        help=(
            "Ministral-3 variants to stage (instruct, reasoning, base). "
            "Accepts space- or comma-separated values, or 'all'. "
            f"Default: all of {', '.join(MINISTRAL3_VARIANTS)}."
        ),
    )
    parser.add_argument(
        "--ministral3-sizes",
        nargs="+",
        default=None,
        metavar="SIZE",
        help=(
            "Ministral-3 sizes to stage. Accepts space- or comma-separated "
            "values, or 'all'. "
            f"Default: all of {', '.join(MINISTRAL3_SIZES)}."
        ),
    )
    parser.add_argument(
        "--skip-ministral3",
        action="store_true",
        help="Do not download Ministral-3 weights.",
    )
    parser.add_argument(
        "--gpt2-sizes",
        nargs="+",
        default=None,
        metavar="SIZE",
        help=(
            "GPT-2 sizes to stage (small, medium, large, xl). Accepts space- or "
            "comma-separated values, or 'all'. "
            f"Default: all of {', '.join(GPT2_SIZES)}."
        ),
    )
    parser.add_argument(
        "--skip-gpt2",
        action="store_true",
        help="Do not download GPT-2 weights.",
    )
    parser.add_argument(
        "--skip-dataset",
        action="store_true",
        help="Do not download the ParaConflict dataset.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip snapshots that already have a complete download on disk.",
    )
    args = parser.parse_args()

    assets_root = args.assets_root
    if assets_root is None:
        env = os.environ.get(ENV_ASSETS_ROOT)
        assets_root = Path(env) if env else Path("assets")
    assets_root = assets_root.expanduser().resolve()
    assets_root.mkdir(parents=True, exist_ok=True)

    models_root = args.models_root
    if models_root is None:
        env = os.environ.get(ENV_MODELS_ROOT)
        models_root = Path(env) if env else assets_root / "models"
    models_root = models_root.expanduser().resolve()
    models_root.mkdir(parents=True, exist_ok=True)

    pythia_sizes = _parse_sizes(args.sizes)
    qwen3_sizes = _parse_qwen3_sizes(args.qwen3_sizes)
    qwen3_base_sizes = _parse_qwen3_base_sizes(args.qwen3_base_sizes)
    ministral3_variants = _parse_ministral3_variants(args.ministral3_variants)
    ministral3_sizes = _parse_ministral3_sizes(args.ministral3_sizes)
    gpt2_sizes = _parse_gpt2_sizes(args.gpt2_sizes)
    do_pythia = not (args.skip_models or args.skip_pythia)
    do_qwen3 = not (args.skip_models or args.skip_qwen3)
    do_qwen3_base = not (args.skip_models or args.skip_qwen3_base)
    do_ministral3 = not (args.skip_models or args.skip_ministral3)
    do_gpt2 = not (args.skip_models or args.skip_gpt2)

    print(f"[stage] Assets root: {assets_root}")
    print(f"[stage] Models root: {models_root}")
    print(f"[stage] Pythia sizes: {', '.join(pythia_sizes) if do_pythia else '(skipped)'}")
    print(f"[stage] Qwen3  sizes: {', '.join(qwen3_sizes) if do_qwen3 else '(skipped)'}")
    print(
        f"[stage] Qwen3-Base sizes: "
        f"{', '.join(qwen3_base_sizes) if do_qwen3_base else '(skipped)'}"
    )
    if do_ministral3:
        ministral3_desc = ", ".join(
            f"{v}-{s}" for v in ministral3_variants for s in ministral3_sizes
        )
    else:
        ministral3_desc = "(skipped)"
    print(f"[stage] Ministral-3: {ministral3_desc}")
    print(f"[stage] GPT-2 sizes: {', '.join(gpt2_sizes) if do_gpt2 else '(skipped)'}")

    def _stage_family(
        sizes: list[str],
        enabled: bool,
        dirname_fn,
        stage_fn,
        family: str,
    ) -> dict[str, Path]:
        out: dict[str, Path] = {}
        if enabled:
            for size in sizes:
                dest = models_root / dirname_fn(size)
                if args.skip_existing and _snapshot_is_complete(dest):
                    print(f"[stage] Skipping {family}-{size}: complete snapshot at {dest}")
                    out[size] = dest
                    continue
                if args.skip_existing and dest.is_dir():
                    print(
                        f"[stage] {family}-{size} present but incomplete; "
                        f"re-downloading missing files into {dest}"
                    )
                out[size] = stage_fn(models_root, size)
        else:
            for size in sizes:
                dest = models_root / dirname_fn(size)
                if dest.is_dir():
                    out[size] = dest
        return out

    pythia_dirs = _stage_family(
        pythia_sizes, do_pythia, pythia_model_dirname, stage_model, "pythia"
    )
    qwen3_dirs = _stage_family(
        qwen3_sizes, do_qwen3, qwen3_model_dirname, stage_qwen3, "qwen3"
    )
    qwen3_base_dirs = _stage_family(
        qwen3_base_sizes,
        do_qwen3_base,
        qwen3_base_model_dirname,
        stage_qwen3_base,
        "qwen3-base",
    )

    ministral3_dirs: dict[str, Path] = {}
    if do_ministral3:
        for variant in ministral3_variants:
            for size in ministral3_sizes:
                key = f"{variant}-{size}"
                dest = models_root / ministral3_model_dirname(variant, size)
                if args.skip_existing and _snapshot_is_complete(dest):
                    print(
                        f"[stage] Skipping {dest.name}: complete snapshot at {dest}"
                    )
                    ministral3_dirs[key] = dest
                    continue
                if args.skip_existing and dest.is_dir():
                    print(
                        f"[stage] ministral3-{key} present but incomplete; "
                        f"re-downloading missing files into {dest}"
                    )
                ministral3_dirs[key] = stage_ministral3(models_root, variant, size)
    else:
        for variant in ministral3_variants:
            for size in ministral3_sizes:
                key = f"{variant}-{size}"
                dest = models_root / ministral3_model_dirname(variant, size)
                if dest.is_dir():
                    ministral3_dirs[key] = dest

    gpt2_dirs = _stage_family(
        gpt2_sizes, do_gpt2, gpt2_model_dirname, stage_gpt2, "gpt2"
    )

    dataset_dir = assets_root / "datasets" / "paraconflict"
    if not args.skip_dataset:
        dataset_dir = stage_paraconflict(assets_root, args.dataset_id)
    elif not dataset_dir.is_dir():
        print(
            f"[stage] Skipping dataset download; expected existing path: {dataset_dir}",
            file=sys.stderr,
        )

    manifest = write_manifest(
        assets_root,
        models_root,
        pythia_dirs,
        qwen3_dirs,
        qwen3_base_dirs,
        ministral3_dirs,
        gpt2_dirs,
        dataset_dir,
    )
    print(f"\n[stage] Staged assets under: {assets_root}")
    print(f"[stage] Models under:       {models_root}")
    print(f"[stage] Manifest:           {manifest}")
    print("\n[stage] On compute nodes (no internet), export:")
    print(f"  export IC_FACTUAL_ASSETS_ROOT={assets_root}")
    if models_root != assets_root / "models":
        print(f"  export IC_FACTUAL_MODELS_ROOT={models_root}")
    print("  export IC_FACTUAL_OFFLINE=1")
    print("  export IC_FACTUAL_REQUIRE_OFFLINE_ASSETS=1")
    print("  export IC_FACTUAL_MODEL_SIZE=160m   # pick any staged Pythia size")


if __name__ == "__main__":
    main()
