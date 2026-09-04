"""Cross-family logit-lens summary: per-layer Δ logprob (gap) only.

Four columns (Pythia, GPT-2, Qwen3, Ministral-3).  All checkpoints belonging
to the same family share one column — e.g. Qwen3-Base 4B/14B and post-trained
4B/14B are overlaid in the Qwen3 column; Ministral instruct / reasoning /
base at 3B and 14B share the Ministral column.

Rows are the four shared exp01/exp11 conditions.  Visual style follows exp21.

Usage::

    python scripts/plot_logit_lens_family_summary.py \\
        --output experiments/exp19_qwen3_logit_lens/plots/logit_lens_family_gap_summary.png
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

CONDITIONS: tuple[str, ...] = (
    "L0_std",
    "L0_neg",
    "L128_prose_std",
    "L128_prose_neg",
)

_COLUMN_ORDER: tuple[str, ...] = ("pythia", "gpt2", "qwen3", "ministral")

# Base vs post-trained (Qwen3) and Ministral variant hues.
_QWEN3_BASE_COLOR = "#98df8a"
_QWEN3_POST_COLOR = "#2ca02c"
# Teal family: reasoning (darkest) > instruct > base (lightest).
_MINISTRAL_COLORS = {
    "reasoning": "#004d40",
    "instruct": "#00897b",
    "base": "#6ec4bc",
}

# Marker shape encodes checkpoint size (shared across all families).
_SMALL_SIZE_KEYS = frozenset({"small", "2.8b", "3b", "4b"})
_SIZE_MARKER_SMALL = "o"
_SIZE_MARKER_LARGE = "^"


def _marker_for_size(size_key: str) -> str:
    return _SIZE_MARKER_SMALL if size_key in _SMALL_SIZE_KEYS else _SIZE_MARKER_LARGE


@dataclass(frozen=True)
class TraceSpec:
    label: str
    path: Path
    color: str
    size_key: str
    marker: str
    linestyle: str = "solid"
    linewidth: float = 1.5


_COLUMN_TITLE: dict[str, str] = {
    "pythia": "Pythia",
    "gpt2": "GPT-2",
    "qwen3": "Qwen3",
    "ministral": "Ministral-3",
}

_COLUMN_ACCENT: dict[str, str] = {
    "pythia": "#2171b5",
    "gpt2": "#e377c2",
    "qwen3": "#2ca02c",
    "ministral": "#004d40",
}

# All traces use solid lines; marker shape encodes size.
def _trace(
    label: str,
    path: Path,
    color: str,
    size_key: str,
    *,
    linewidth: float = 1.5,
) -> "TraceSpec":
    return TraceSpec(
        label=label,
        path=path,
        color=color,
        size_key=size_key,
        marker=_marker_for_size(size_key),
        linewidth=linewidth,
    )


def _load(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _condition(payload: dict, condition: str) -> dict | None:
    for c in payload.get("conditions", []):
        if c.get("label") == condition and c.get("n_used", 0) > 0:
            return c
    return None


def _frac_depth(num_layers: int) -> np.ndarray:
    n = max(num_layers - 1, 1)
    return np.arange(num_layers) / n


def _add_direction_guide(ax: plt.Axes) -> None:
    """At y=0: up = in-context (positive), down = memorized (negative)."""
    from matplotlib.transforms import blended_transform_factory

    trans = blended_transform_factory(ax.transAxes, ax.transData)
    y0 = 0.0
    ymin, ymax = ax.get_ylim()
    dy = max((ymax - ymin) * 0.14, 1.0)
    arrow_kw = dict(arrowstyle="-|>", color="0.45", lw=1.6, mutation_scale=11, alpha=0.65)
    label_kw = dict(fontsize=7.5, color="0.45", alpha=0.75, ha="left", va="center")
    x_arrow = 0.03
    x_label = 0.075

    ax.annotate(
        "",
        xy=(x_arrow, y0 + dy),
        xytext=(x_arrow, y0),
        xycoords=trans,
        textcoords=trans,
        arrowprops=arrow_kw,
    )
    ax.text(x_label, y0 + dy * 0.55, "favors\nin-context", transform=trans, **label_kw)

    ax.annotate(
        "",
        xy=(x_arrow, y0 - dy),
        xytext=(x_arrow, y0),
        xycoords=trans,
        textcoords=trans,
        arrowprops=arrow_kw,
    )
    ax.text(x_label, y0 - dy * 0.55, "favors\nmemorized", transform=trans, **label_kw)


def _default_columns(repo_root: Path) -> dict[str, list[TraceSpec]]:
    logit_lens_dir = repo_root / "summaries/logit_lens"
    pythia_c = _COLUMN_ACCENT["pythia"]
    gpt2_c = _COLUMN_ACCENT["gpt2"]

    return {
        "pythia": [
            _trace("2.8b", logit_lens_dir / "per_layer_2.8b.json", pythia_c, "2.8b"),
            _trace("12b", logit_lens_dir / "per_layer_12b.json", pythia_c, "12b"),
        ],
        "gpt2": [
            _trace("small", logit_lens_dir / "per_layer_gpt2.json", gpt2_c, "small"),
            _trace("xl", logit_lens_dir / "per_layer_gpt2-xl.json", gpt2_c, "xl"),
        ],
        "qwen3": [
            _trace(
                "4b-base",
                logit_lens_dir / "per_layer_qwen3-base-4b.json",
                _QWEN3_BASE_COLOR,
                "4b",
            ),
            _trace(
                "4b",
                logit_lens_dir / "per_layer_qwen3-4b.json",
                _QWEN3_POST_COLOR,
                "4b",
            ),
            _trace(
                "14b-base",
                logit_lens_dir / "per_layer_qwen3-base-14b.json",
                _QWEN3_BASE_COLOR,
                "14b",
            ),
            _trace(
                "14b",
                logit_lens_dir / "per_layer_qwen3-14b.json",
                _QWEN3_POST_COLOR,
                "14b",
            ),
        ],
        "ministral": [
            _trace(
                "3b reasoning",
                logit_lens_dir / "per_layer_ministral3-reasoning-3b.json",
                _MINISTRAL_COLORS["reasoning"],
                "3b",
            ),
            _trace(
                "3b instruct",
                logit_lens_dir / "per_layer_ministral3-instruct-3b.json",
                _MINISTRAL_COLORS["instruct"],
                "3b",
            ),
            _trace(
                "3b base",
                logit_lens_dir / "per_layer_ministral3-base-3b.json",
                _MINISTRAL_COLORS["base"],
                "3b",
            ),
            _trace(
                "14b reasoning",
                logit_lens_dir / "per_layer_ministral3-reasoning-14b.json",
                _MINISTRAL_COLORS["reasoning"],
                "14b",
            ),
            _trace(
                "14b instruct",
                logit_lens_dir / "per_layer_ministral3-instruct-14b.json",
                _MINISTRAL_COLORS["instruct"],
                "14b",
            ),
            _trace(
                "14b base",
                logit_lens_dir / "per_layer_ministral3-base-14b.json",
                _MINISTRAL_COLORS["base"],
                "14b",
            ),
        ],
    }


def plot_family_gap_summary(
    columns: dict[str, list[TraceSpec]],
    output_path: Path,
    *,
    title: str = "Per-layer Δ logprob (distractor − memorized)",
) -> None:
    column_keys = [k for k in _COLUMN_ORDER if k in columns]
    n_rows, n_cols = len(CONDITIONS), len(column_keys)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(3.8 * n_cols, 2.6 * n_rows),
        squeeze=False,
        sharex=True,
        sharey="row",
    )

    for c_idx, col_key in enumerate(column_keys):
        traces = columns[col_key]
        accent = _COLUMN_ACCENT[col_key]
        col_handles: list[plt.Line2D] = []

        for r_idx, cond in enumerate(CONDITIONS):
            ax = axes[r_idx][c_idx]
            plotted_any = False
            for trace in traces:
                if not trace.path.is_file():
                    continue
                payload = _load(trace.path)
                cond_data = _condition(payload, cond)
                if cond_data is None:
                    continue
                gap = -np.array(cond_data["mean_gap_logprob"], dtype=float)
                xs = _frac_depth(int(cond_data["num_layers"]))
                ax.plot(
                    xs,
                    gap,
                    linestyle=trace.linestyle,
                    marker=trace.marker,
                    ms=3.0,
                    lw=trace.linewidth,
                    color=trace.color,
                    label=trace.label,
                )
                plotted_any = True

            if not plotted_any:
                ax.text(
                    0.5,
                    0.5,
                    "missing data",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    fontsize=8,
                    color="gray",
                )

            ax.axhline(0.0, color="black", linewidth=0.8, linestyle="--", alpha=0.6)
            ax.grid(True, which="both", alpha=0.25)
            ax.set_xlim(-0.02, 1.02)
            if r_idx == 0:
                ax.set_title(
                    _COLUMN_TITLE[col_key],
                    fontsize=12,
                    fontweight="bold",
                    color=accent,
                )
            if c_idx == 0:
                ax.set_ylabel(f"{cond}\nΔ logprob", fontsize=10)

        for trace in traces:
            col_handles.append(
                plt.Line2D(
                    [0],
                    [0],
                    color=trace.color,
                    linestyle=trace.linestyle,
                    marker=trace.marker,
                    ms=4,
                    lw=trace.linewidth,
                    label=trace.label,
                )
            )
        ncol = 2 if len(col_handles) > 3 else 1
        axes[0][c_idx].legend(
            handles=col_handles,
            loc="upper right",
            fontsize=6.5 if len(col_handles) > 4 else 7.5,
            framealpha=0.92,
            handlelength=2.0,
            ncol=ncol,
        )

    _add_direction_guide(axes[0][0])

    fig.suptitle(title, y=1.01, fontsize=13)
    fig.supxlabel(
        "Fractional depth in the stack (0 = embedding read-off, 1 = last transformer block)",
        y=0.02,
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.99))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Wrote {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cross-family logit-lens gap summary (one column per family)."
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output PNG path.",
    )
    parser.add_argument(
        "--title",
        default="Per-layer Δ logprob (distractor − memorized)",
        help="Figure title.",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository root for default JSON paths.",
    )
    args = parser.parse_args()

    columns = _default_columns(args.repo_root)
    plot_family_gap_summary(columns, args.output, title=args.title)


if __name__ == "__main__":
    main()
