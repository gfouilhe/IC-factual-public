"""Plot the exp19 Qwen3 logit-lens grid (base vs post-trained x {4B,8B,14B}).

Consumes the six JSONs produced by ``scripts/logit_lens_capitals.py``
(one per (family, size)) and produces:

* ``exp19_logit_lens_gap_grid.png`` — rows = condition (L0_std,
  L0_neg, L128_prose_std, L128_prose_neg), cols = size (4B/8B/14B).
  Each panel overlays the *base* vs *post-trained* gap-vs-fractional-
  depth curves so the post-training delta is read off directly.
* ``exp19_logit_lens_delta_grid.png`` — same row/col layout but plots
  ``gap(post) - gap(base)`` per fractional-depth bin for each condition,
  isolating the post-training mechanism.

Usage::

    python scripts/plot_logit_lens_qwen3.py \\
        --inputs experiments/exp19_qwen3_logit_lens/summaries/per_layer_qwen3-base-4b.json \\
                 experiments/exp19_qwen3_logit_lens/summaries/per_layer_qwen3-base-8b.json \\
                 experiments/exp19_qwen3_logit_lens/summaries/per_layer_qwen3-base-14b.json \\
                 experiments/exp19_qwen3_logit_lens/summaries/per_layer_qwen3-4b.json \\
                 experiments/exp19_qwen3_logit_lens/summaries/per_layer_qwen3-8b.json \\
                 experiments/exp19_qwen3_logit_lens/summaries/per_layer_qwen3-14b.json \\
        --output-dir experiments/exp19_qwen3_logit_lens/plots
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


CONDITIONS: tuple[str, ...] = (
    "L0_std",
    "L0_neg",
    "L128_prose_std",
    "L128_prose_neg",
)

COND_COLOR = {
    "L0_std": "#2c7fb8",
    "L0_neg": "#d95f0e",
    "L128_prose_std": "#41ab5d",
    "L128_prose_neg": "#cb181d",
}

# (family-key, line-style, marker, alpha)
FAMILY_STYLE = {
    "base": ("solid", "o", 1.0),
    "post": ("dashed", "s", 0.85),
}


def _load(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _family_size(label: str) -> tuple[str, str]:
    """Map a JSON label like ``qwen3-base-4b`` to ``("base", "4b")``.

    ``qwen3-4b`` -> ``("post", "4b")``.  Unknown labels fall back to
    ``("?", label)``.
    """
    key = label.lower().strip()
    if key.startswith("qwen3-base-"):
        return "base", key[len("qwen3-base-"):]
    if key.startswith("qwen3-"):
        return "post", key[len("qwen3-"):]
    return "?", key


def _by_label(payloads: list[dict]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for p in payloads:
        label = str(p.get("label") or p.get("model_size"))
        out[label] = p
    return out


def _condition(p: dict, condition: str) -> dict | None:
    for c in p.get("conditions", []):
        if c.get("label") == condition and c.get("n_used", 0) > 0:
            return c
    return None


def _frac_depth(num_layers: int) -> np.ndarray:
    # Layer index 0 = embedding read-off; final index = num_layers - 1.
    # Fractional depth normalizes against the number of blocks
    # (num_layers - 1) so we can compare across model depths.
    n = max(num_layers - 1, 1)
    return np.arange(num_layers) / n


def _interp_to_grid(xs: np.ndarray, ys: np.ndarray, grid: np.ndarray) -> np.ndarray:
    return np.interp(grid, xs, ys)


#: Median fractional-depth onset where ``|gap| ≥ 0.5`` across all
#: 24 (family × size × condition) cells (exp19 README §"Divergence
#: onset").  Drawn as a vertical guide on every panel.
_ONSET_DEPTH: float = 0.67


def _annotate_endpoints(
    ax: plt.Axes,
    xs: np.ndarray,
    gap: np.ndarray,
    color: str,
    family_key: str,
) -> None:
    """Mark the peak |gap| and the final-layer gap with small text labels."""
    if len(gap) == 0:
        return
    final = float(gap[-1])
    peak_i = int(np.argmax(np.abs(gap)))
    peak = float(gap[peak_i])
    peak_x = float(xs[peak_i])
    # Peak: star marker
    ax.scatter(
        [peak_x], [peak],
        marker="*", s=70,
        edgecolor=color, facecolor="white",
        linewidths=1.4, zorder=5,
    )
    # Final-layer: text label to the right of the curve. Stack
    # base/post labels vertically so they don't overlap.
    y_offset = 0.0 if family_key == "base" else 0.18 * abs(peak - final + 1e-9)
    va = "bottom" if family_key == "base" else "top"
    ax.text(
        1.005, final - y_offset,
        f"{final:+.1f}",
        color=color,
        fontsize=8,
        va=va, ha="left",
        fontweight="bold" if family_key == "post" else "normal",
        clip_on=False,
    )


def plot_gap_grid(
    payloads_by_label: dict[str, dict],
    sizes: tuple[str, ...],
    output_path: Path,
) -> None:
    n_rows, n_cols = len(CONDITIONS), len(sizes)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(4.8 * n_cols, 3.0 * n_rows),
        squeeze=False,
        sharex=True,
        sharey="row",
    )

    for r_idx, cond in enumerate(CONDITIONS):
        for c_idx, size in enumerate(sizes):
            ax = axes[r_idx][c_idx]
            base_label = f"qwen3-base-{size}"
            post_label = f"qwen3-{size}"
            for family_key, plot_label in (("base", base_label), ("post", post_label)):
                payload = payloads_by_label.get(plot_label)
                if payload is None:
                    continue
                cond_data = _condition(payload, cond)
                if cond_data is None:
                    continue
                gap = np.array(cond_data["mean_gap_logprob"], dtype=float)
                num_layers = int(cond_data["num_layers"])
                xs = _frac_depth(num_layers)
                ls, mk, alpha = FAMILY_STYLE[family_key]
                ax.plot(
                    xs, gap,
                    linestyle=ls, marker=mk, ms=3.0, lw=1.5,
                    alpha=alpha,
                    color=COND_COLOR.get(cond, "black"),
                    label=plot_label,
                )
                _annotate_endpoints(
                    ax, xs, gap,
                    color=COND_COLOR.get(cond, "black"),
                    family_key=family_key,
                )

            ax.axhline(0, color="gray", lw=0.7)
            ax.axvline(_ONSET_DEPTH, color="0.6", lw=0.7, ls=":",
                       alpha=0.8, zorder=0)
            ax.grid(True, alpha=0.25)
            ax.set_xlim(-0.02, 1.05)
            if r_idx == 0:
                ax.set_title(f"qwen3 {size}", fontsize=12, fontweight="bold")
            if r_idx == n_rows - 1:
                ax.set_xlabel("Fractional depth (0 = embedding, 1 = final layer)")
            if c_idx == 0:
                ax.set_ylabel(f"{cond}\ngap = ans − dist", fontsize=10)

    legend_handles = [
        plt.Line2D([0], [0], color="0.3",
                   linestyle=FAMILY_STYLE["base"][0],
                   marker=FAMILY_STYLE["base"][1], ms=4, lw=1.5,
                   label="qwen3-base (pretrained)"),
        plt.Line2D([0], [0], color="0.3",
                   linestyle=FAMILY_STYLE["post"][0],
                   marker=FAMILY_STYLE["post"][1], ms=4, lw=1.5,
                   label="qwen3 (post-trained)"),
        plt.Line2D([0], [0], marker="*", color="0.4",
                   markerfacecolor="white", markeredgecolor="0.4",
                   linestyle="None", ms=10, label="peak |gap|"),
        plt.Line2D([0], [0], color="0.6", lw=0.7, ls=":",
                   label=f"median divergence onset (depth ≈ {_ONSET_DEPTH:.2f})"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=4,
        bbox_to_anchor=(0.5, -0.015),
        frameon=False,
        fontsize=9,
    )

    fig.suptitle(
        "Per-layer logit-lens gap on Qwen3 vs Qwen3-Base\n"
        "rows = condition, cols = size; numbers near right edge = final-layer gap "
        "(bold = post-trained); shared y-axis per row",
        y=1.005,
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0.025, 1, 0.99))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Wrote {output_path}")


def plot_delta_grid(
    payloads_by_label: dict[str, dict],
    sizes: tuple[str, ...],
    output_path: Path,
    n_grid: int = 41,
) -> None:
    """``gap(post) - gap(base)`` resampled onto a common fractional-depth grid."""
    grid = np.linspace(0.0, 1.0, n_grid)
    n_rows, n_cols = len(CONDITIONS), len(sizes)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(4.8 * n_cols, 3.0 * n_rows),
        squeeze=False,
        sharex=True,
        sharey="row",
    )

    for r_idx, cond in enumerate(CONDITIONS):
        for c_idx, size in enumerate(sizes):
            ax = axes[r_idx][c_idx]
            base = _condition(payloads_by_label.get(f"qwen3-base-{size}", {}), cond)
            post = _condition(payloads_by_label.get(f"qwen3-{size}", {}), cond)
            if base is None or post is None:
                ax.text(0.5, 0.5, "missing", ha="center", va="center",
                        transform=ax.transAxes, color="gray")
                ax.axhline(0, color="gray", lw=0.7)
                ax.grid(True, alpha=0.3)
                continue

            base_xs = _frac_depth(int(base["num_layers"]))
            post_xs = _frac_depth(int(post["num_layers"]))
            base_g = _interp_to_grid(base_xs, np.array(base["mean_gap_logprob"]), grid)
            post_g = _interp_to_grid(post_xs, np.array(post["mean_gap_logprob"]), grid)
            delta = post_g - base_g

            color = COND_COLOR.get(cond, "black")
            ax.plot(grid, delta, "-o", color=color, ms=3.0, lw=1.5)

            peak_i = int(np.argmax(np.abs(delta)))
            ax.scatter(
                [grid[peak_i]], [delta[peak_i]],
                marker="*", s=70,
                edgecolor=color, facecolor="white",
                linewidths=1.4, zorder=5,
            )
            final_delta = float(delta[-1])
            peak_delta = float(delta[peak_i])
            ax.text(
                1.005, final_delta,
                f"final {final_delta:+.1f}\npeak {peak_delta:+.1f}",
                color=color, fontsize=8,
                va="center", ha="left",
                clip_on=False,
            )

            ax.axhline(0, color="gray", lw=0.7)
            ax.axvline(_ONSET_DEPTH, color="0.6", lw=0.7, ls=":",
                       alpha=0.8, zorder=0)
            ax.grid(True, alpha=0.25)
            ax.set_xlim(-0.02, 1.05)
            if r_idx == 0:
                ax.set_title(f"qwen3 {size}", fontsize=12, fontweight="bold")
            if r_idx == n_rows - 1:
                ax.set_xlabel("Fractional depth")
            if c_idx == 0:
                ax.set_ylabel(f"{cond}\nΔgap = post − base", fontsize=10)

    legend_handles = [
        plt.Line2D([0], [0], marker="*", color="0.4",
                   markerfacecolor="white", markeredgecolor="0.4",
                   linestyle="None", ms=10, label="peak |Δgap|"),
        plt.Line2D([0], [0], color="0.6", lw=0.7, ls=":",
                   label=f"divergence onset (depth ≈ {_ONSET_DEPTH:.2f})"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=2,
        bbox_to_anchor=(0.5, -0.015),
        frameon=False,
        fontsize=9,
    )

    fig.suptitle(
        "Post-training delta on the per-layer gap: Δgap = gap(post) − gap(base)\n"
        "rows = condition, cols = size; positive = post-trained more memorized; "
        "numbers = final-layer Δ and peak |Δ|; shared y-axis per row",
        y=1.005,
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0.025, 1, 0.99))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Wrote {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Plot exp19 logit-lens grids comparing Qwen3 base vs "
            "post-trained at 4B/8B/14B."
        )
    )
    parser.add_argument(
        "--inputs",
        type=str,
        nargs="+",
        required=True,
        help=(
            "Per-layer JSONs produced by logit_lens_capitals.py. Expected "
            "labels: qwen3-base-{4b,8b,14b} and qwen3-{4b,8b,14b}."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory in which to write the PNG figures.",
    )
    parser.add_argument(
        "--sizes",
        type=str,
        nargs="+",
        default=["4b", "8b", "14b"],
    )
    args = parser.parse_args()

    payloads = [_load(Path(p)) for p in args.inputs]
    by_label = _by_label(payloads)

    out_dir = Path(args.output_dir)
    plot_gap_grid(
        by_label, tuple(args.sizes), out_dir / "exp19_logit_lens_gap_grid.png"
    )
    plot_delta_grid(
        by_label, tuple(args.sizes), out_dir / "exp19_logit_lens_delta_grid.png"
    )


if __name__ == "__main__":
    main()
