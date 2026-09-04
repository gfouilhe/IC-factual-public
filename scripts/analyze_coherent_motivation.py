"""Aggregate exp25 ParaConflict coherent-motivation sweeps and emit paper figures.

Produces:
* ``<prefix>_coherent_native.png`` -- native Sub vs Coh (highlight sizes).
* ``<prefix>_decomp.png`` -- length sweep (main categories, highlight sizes).
* ``<prefix>_decomp_all.png`` -- length sweep (all six categories; appendix).
* ``<prefix>_controls_L128.png`` -- L=128 filler heatmaps (2.8B / 12B).
* ``<prefix>_summary.json`` -- long-form aggregates.
"""

from __future__ import annotations

import argparse
import glob
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

_SIZE_ORDER = ("160m", "410m", "1b", "1.4b", "2.8b", "6.9b", "12b")
_HIGHLIGHT_SIZES = ("2.8b", "12b")
_CATEGORIES = (
    "World Capital",
    "Athlete Sport",
    "Book Author",
    "Company Headquarter",
    "Company Founder",
    "Official Language",
)
_DECOMP_CATEGORIES = ("World Capital", "Official Language")
_LENGTH_SWEEP_FILLERS = ("prose", "coherent_passage", "relation_noise")
_LENGTHS = (0, 32, 128, 512, 1024)
_L128_FILLERS = (
    "prose",
    "varied_prose",
    "coherent_passage",
    "coherent_full",
    "relation_noise",
    "relation_noise_false",
    "passage_shuffled",
    "passage_scrambled",
)
_FILLER_SHORT = {
    "prose": "prose",
    "varied_prose": "varied",
    "coherent_passage": "coh. passage",
    "coherent_full": "coh. full",
    "relation_noise": "rel. noise",
    "relation_noise_false": "false noise",
    "passage_shuffled": "shuffled",
    "passage_scrambled": "scrambled",
}
_CAT_SHORT = {
    "World Capital": "Capitals",
    "Athlete Sport": "Athletes",
    "Book Author": "Authors",
    "Company Headquarter": "HQ",
    "Company Founder": "Founders",
    "Official Language": "Language",
}
_SIZE_COLORS = {
    "160m": "#c6dbef",
    "410m": "#9ecae1",
    "1b": "#6baed6",
    "1.4b": "#4292c6",
    "2.8b": "#2171b5",
    "6.9b": "#238b45",
    "12b": "#cb181d",
}


def _apply_paper_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linestyle": "-",
            "grid.linewidth": 0.6,
        }
    )


def _size_sort_key(size: str) -> tuple:
    try:
        return (0, _SIZE_ORDER.index(size))
    except ValueError:
        return (1, size)


def _load(path: str) -> tuple[dict, list[dict]]:
    with open(path, "r", encoding="utf-8") as f:
        first = f.readline()
        if not first.strip():
            return {}, []
        header = json.loads(first)
        rows = [json.loads(line) for line in f if line.strip()]
    return header, rows


def _metric_flags(metric: str) -> tuple[str, str]:
    if metric == "lp":
        return "memorized_lp", "in_context_lp"
    return "memorized_gen", "in_context_gen"


def _parse_sweep_filename(path: str) -> dict | None:
    name = Path(path).name
    m = re.match(
        r"per_sample_(?:L(\d+)_)?([a-z_]+)_(160m|410m|1b|1\.4b|2\.8b|6\.9b|12b)\.jsonl",
        name,
    )
    if not m:
        return None
    length = int(m.group(1)) if m.group(1) else None
    return {"length": length, "filler_kind": m.group(2), "size": m.group(3)}


def _aggregate_native(paths: list[str], metric: str) -> dict:
    mem_k, ctx_k = _metric_flags(metric)
    counts: dict[tuple[str, str, str], dict[str, int]] = defaultdict(
        lambda: {"n": 0, "memorized": 0, "in_context": 0}
    )
    for path in paths:
        header, rows = _load(path)
        if not rows:
            continue
        size = header.get("model_size") or Path(path).stem.replace("per_sample_", "")
        for r in rows:
            cat = r.get("category") or ""
            if not cat:
                continue
            for ptype in ("substitution_conflict", "coherent_conflict"):
                sub = r.get(ptype, {})
                if not sub:
                    continue
                key = (size, cat, ptype)
                counts[key]["n"] += 1
                if sub.get(mem_k):
                    counts[key]["memorized"] += 1
                if sub.get(ctx_k):
                    counts[key]["in_context"] += 1
    return counts


def _aggregate_sweep(paths: list[str], metric: str) -> dict:
    mem_k, ctx_k = _metric_flags(metric)
    counts: dict[tuple[str, str, str, int | None], dict[str, int]] = defaultdict(
        lambda: {"n": 0, "memorized": 0, "in_context": 0}
    )
    for path in paths:
        header, rows = _load(path)
        if not rows:
            continue
        parsed = _parse_sweep_filename(path)
        in_meta = header.get("input_meta") or {}
        size = header.get("model_size") or (parsed or {}).get("size", "")
        filler = in_meta.get("filler_kind") or (parsed or {}).get("filler_kind", "")
        length = in_meta.get("length_tokens_target")
        if length is None and parsed:
            length = parsed.get("length")
        if filler == "coherent_full":
            length = length if length is not None else -1
        for r in rows:
            cat = r.get("category") or ""
            block = r.get("substitution_conflict") or {}
            if not cat or not block:
                continue
            key = (size, cat, filler, length)
            counts[key]["n"] += 1
            if block.get(mem_k):
                counts[key]["memorized"] += 1
            if block.get(ctx_k):
                counts[key]["in_context"] += 1
    return counts


def _rate(d: dict, field: str) -> float:
    n = d.get("n", 0)
    if not n:
        return float("nan")
    return d[field] / n


def _plot_native(counts, highlight_sizes, out_path: Path) -> None:
    """Signed Δ = P(ctx) − P(mem) for Sub vs Coh; two panels, highlight sizes only."""
    categories = [c for c in _CATEGORIES if any((s, c, _) in counts for s in highlight_sizes for _ in ("substitution_conflict", "coherent_conflict"))]
    if not categories:
        return

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4), squeeze=True)
    width = 0.34
    x = np.arange(len(categories))
    labels = [_CAT_SHORT.get(c, c) for c in categories]

    panel_titles = (
        "Substitution conflict",
        "Coherent conflict",
    )
    for ax_i, ptype in enumerate(("substitution_conflict", "coherent_conflict")):
        ax = axes[ax_i]
        for s_idx, size in enumerate(highlight_sizes):
            ys = []
            for cat in categories:
                d = counts.get((size, cat, ptype), {"n": 0, "memorized": 0, "in_context": 0})
                n = d["n"]
                if not n:
                    ys.append(float("nan"))
                else:
                    ys.append((d["in_context"] - d["memorized"]) / n)
            offset = (s_idx - 0.5) * width
            ax.bar(
                x + offset,
                ys,
                width,
                color=_SIZE_COLORS.get(size, "#888888"),
                edgecolor="white",
                linewidth=0.6,
                label=f"Pythia-{size}",
                zorder=3,
            )
        ax.axhline(0.0, color="0.15", linewidth=0.9, zorder=2)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylim(-1.05, 1.05)
        ax.set_ylabel(r"$\Delta = P(\mathrm{ctx}) - P(\mathrm{mem})$")
        ax.set_title(panel_titles[ax_i], pad=8)
        if ax_i == 0:
            ax.text(
                0.02,
                0.97,
                "context wins",
                transform=ax.transAxes,
                fontsize=8,
                va="top",
                color="0.35",
            )
            ax.text(
                0.02,
                0.03,
                "memory wins",
                transform=ax.transAxes,
                fontsize=8,
                va="bottom",
                color="0.35",
            )

    handles = [
        Patch(facecolor=_SIZE_COLORS[s], edgecolor="white", label=f"Pythia-{s}")
        for s in highlight_sizes
    ]
    fig.legend(handles=handles, loc="upper center", ncol=len(highlight_sizes), frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _plot_native_shift(counts, highlight_sizes, out_path: Path) -> None:
    """Dumbbell: P(in-context) under Sub → Coh per category (shows coherent shift)."""
    categories = [c for c in _CATEGORIES if any((s, c, _) in counts for s in highlight_sizes for _ in ("substitution_conflict", "coherent_conflict"))]
    if not categories:
        return

    n_sizes = len(highlight_sizes)
    fig, axes = plt.subplots(1, n_sizes, figsize=(3.6 * n_sizes, 3.6), squeeze=False)
    labels = [_CAT_SHORT.get(c, c) for c in categories]

    for ax_i, size in enumerate(highlight_sizes):
        ax = axes[0, ax_i]
        y = np.arange(len(categories))
        for yi, cat in enumerate(categories):
            sub = counts.get((size, cat, "substitution_conflict"), {"n": 0, "in_context": 0})
            coh = counts.get((size, cat, "coherent_conflict"), {"n": 0, "in_context": 0})
            p_sub = _rate(sub, "in_context")
            p_coh = _rate(coh, "in_context")
            if np.isnan(p_sub) or np.isnan(p_coh):
                continue
            ax.plot(
                [p_sub, p_coh],
                [yi, yi],
                color="0.55",
                linewidth=1.6,
                zorder=1,
                solid_capstyle="round",
            )
            ax.scatter(
                [p_sub],
                [yi],
                s=42,
                color="#a6bddb",
                edgecolors="#2171b5",
                linewidths=1.0,
                zorder=3,
                label="Sub" if yi == 0 else None,
            )
            ax.scatter(
                [p_coh],
                [yi],
                s=52,
                color="#2171b5",
                edgecolors="white",
                linewidths=0.6,
                zorder=4,
                label="Coh" if yi == 0 else None,
            )
        ax.set_yticks(y)
        ax.set_yticklabels(labels)
        ax.set_xlim(-0.02, 1.05)
        ax.set_xlabel(r"$P(\mathrm{in\text{-}context})$")
        ax.set_title(f"Pythia-{size}", pad=8)
        ax.invert_yaxis()
        if ax_i == 0:
            ax.legend(loc="lower right", frameon=True, framealpha=0.9, fontsize=8)

    fig.suptitle("Effect of native coherent passage on in-context rate", y=1.02, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _plot_decomp(
    counts,
    sizes,
    highlight_sizes,
    categories,
    out_path: Path,
    *,
    show_faint: bool = True,
) -> None:
    """Length sweep: rows = categories, cols = filler kinds."""
    categories = [c for c in categories if c in _CATEGORIES]
    if not categories:
        return

    n_rows = len(categories)
    n_cols = len(_LENGTH_SWEEP_FILLERS)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(3.5 * n_cols, 2.6 * n_rows + 0.3),
        sharex=True,
        sharey=True,
        squeeze=False,
    )

    for r_idx, cat in enumerate(categories):
        for c_idx, filler in enumerate(_LENGTH_SWEEP_FILLERS):
            ax = axes[r_idx, c_idx]
            if show_faint:
                for size in sizes:
                    if size in highlight_sizes:
                        continue
                    xs, ys = [], []
                    for L in _LENGTHS:
                        d = counts.get((size, cat, filler, L), {"n": 0, "in_context": 0})
                        if d["n"]:
                            xs.append(L)
                            ys.append(_rate(d, "in_context"))
                    if xs:
                        ax.plot(xs, ys, "-", color="0.82", linewidth=0.9, alpha=0.55, zorder=1)

            for size in highlight_sizes:
                xs, ys = [], []
                for L in _LENGTHS:
                    d = counts.get((size, cat, filler, L), {"n": 0, "in_context": 0})
                    if d["n"]:
                        xs.append(L)
                        ys.append(_rate(d, "in_context"))
                if xs:
                    ax.plot(
                        xs,
                        ys,
                        "-o",
                        color=_SIZE_COLORS.get(size, "#333"),
                        linewidth=2.0,
                        markersize=5,
                        markerfacecolor="white",
                        markeredgewidth=1.4,
                        label=f"Pythia-{size}",
                        zorder=3,
                    )

            # Native coherent_full anchor (highlight sizes median).
            anchor_vals = []
            for size in highlight_sizes:
                d = counts.get((size, cat, "coherent_full", -1), {"n": 0, "in_context": 0})
                if not d["n"]:
                    d = counts.get((size, cat, "coherent_full", 0), {"n": 0, "in_context": 0})
                if d["n"]:
                    anchor_vals.append(_rate(d, "in_context"))
            if anchor_vals:
                ax.axhline(
                    float(np.median(anchor_vals)),
                    color="0.45",
                    linestyle=(0, (4, 3)),
                    linewidth=1.2,
                    zorder=2,
                )

            ax.set_xscale("symlog", linthresh=32, linscale=0.4)
            ax.set_xticks(list(_LENGTHS))
            ax.set_xticklabels([str(L) for L in _LENGTHS])
            ax.set_ylim(-0.02, 1.02)
            if r_idx == 0:
                ax.set_title(filler.replace("_", " "), fontsize=9, style="italic")
            if c_idx == 0:
                ax.set_ylabel(_CAT_SHORT.get(cat, cat))
            if r_idx == n_rows - 1:
                ax.set_xlabel("Filler length $L$ (tokens)")

    legend_handles = [
        Line2D(
            [0],
            [0],
            color=_SIZE_COLORS[s],
            marker="o",
            linewidth=2,
            markersize=5,
            markerfacecolor="white",
            label=f"Pythia-{s}",
        )
        for s in highlight_sizes
    ]
    legend_handles.append(
        Line2D([0], [0], color="0.45", linestyle=(0, (4, 3)), linewidth=1.2, label="native coherent full")
    )
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=len(legend_handles),
        frameon=False,
        bbox_to_anchor=(0.5, 1.01),
        fontsize=8,
    )
    fig.supylabel(r"$P(\mathrm{in\text{-}context})$", fontsize=10, x=0.02)
    fig.tight_layout(rect=(0.03, 0, 1, 0.97))
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _plot_controls_heatmap(counts, highlight_sizes, out_path: Path) -> None:
    """Heatmap of P(in-context) at L=128 for all fillers × categories."""
    categories = [c for c in _CATEGORIES if any((s, c, f, 128) in counts or (s, c, f, -1) in counts for s in highlight_sizes for f in _L128_FILLERS)]
    if not categories:
        return

    cmap = LinearSegmentedColormap.from_list("ctx_blues", ["#f7fbff", "#6baed6", "#08306b"])
    n_sizes = len(highlight_sizes)
    fig, axes = plt.subplots(1, n_sizes, figsize=(4.8 * n_sizes, 3.8), squeeze=False)

    cat_labels = [_CAT_SHORT.get(c, c) for c in categories]
    filler_labels = [_FILLER_SHORT.get(f, f) for f in _L128_FILLERS]

    for ax_i, size in enumerate(highlight_sizes):
        ax = axes[0, ax_i]
        mat = np.full((len(categories), len(_L128_FILLERS)), np.nan)
        for r_i, cat in enumerate(categories):
            for c_i, filler in enumerate(_L128_FILLERS):
                if filler == "coherent_full":
                    for L in (-1, 0):
                        d = counts.get((size, cat, filler, L), {"n": 0, "in_context": 0})
                        if d["n"]:
                            mat[r_i, c_i] = _rate(d, "in_context")
                            break
                else:
                    d = counts.get((size, cat, filler, 128), {"n": 0, "in_context": 0})
                    if d["n"]:
                        mat[r_i, c_i] = _rate(d, "in_context")

        im = ax.imshow(mat, aspect="auto", cmap=cmap, vmin=0.0, vmax=1.0, origin="upper")
        ax.set_xticks(np.arange(len(_L128_FILLERS)))
        ax.set_xticklabels(filler_labels, rotation=40, ha="right", fontsize=7)
        ax.set_yticks(np.arange(len(categories)))
        ax.set_yticklabels(cat_labels)
        ax.set_title(f"Pythia-{size}", pad=10)

        for r_i in range(len(categories)):
            for c_i in range(len(_L128_FILLERS)):
                val = mat[r_i, c_i]
                if np.isnan(val):
                    continue
                color = "white" if val > 0.55 else "0.15"
                ax.text(c_i, r_i, f"{val:.2f}", ha="center", va="center", fontsize=7, color=color)

    fig.subplots_adjust(right=0.88, wspace=0.28)
    cbar_ax = fig.add_axes([0.90, 0.18, 0.018, 0.68])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label(r"$P(\mathrm{in\text{-}context})$", fontsize=9)
    fig.suptitle("Filler controls at L=128 (coherent_full = native passage)", y=0.98, fontsize=10)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--native-glob", nargs="+", default=[])
    parser.add_argument("--sweep-glob", nargs="+", default=[])
    parser.add_argument("--output-prefix", type=str, required=True)
    parser.add_argument("--metric", choices=("gen", "lp"), default="gen")
    parser.add_argument("--sizes", nargs="*", default=None)
    parser.add_argument(
        "--highlight-sizes",
        nargs="*",
        default=list(_HIGHLIGHT_SIZES),
        help="Sizes emphasised in main figures (default: 2.8b 12b).",
    )
    parser.add_argument(
        "--decomp-categories",
        nargs="*",
        default=list(_DECOMP_CATEGORIES),
        help="Categories in the main decomposition figure.",
    )
    args = parser.parse_args()
    _apply_paper_style()

    out_prefix = Path(args.output_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    summaries_dir = out_prefix.parent.parent / "summaries"
    if out_prefix.parent.name == "plots":
        summaries_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict] = []
    highlight = list(args.highlight_sizes)

    native_paths = sorted(
        p for pat in args.native_glob for p in glob.glob(pat) if Path(p).stat().st_size > 0
    )
    sweep_paths = sorted(
        p for pat in args.sweep_glob for p in glob.glob(pat) if Path(p).stat().st_size > 0
    )

    sizes_set: set[str] = set()
    if native_paths:
        native_counts = _aggregate_native(native_paths, args.metric)
        sizes_set.update(s for s, _, _ in native_counts.keys())
        sizes = sorted(sizes_set, key=_size_sort_key)
        if args.sizes:
            sizes = [s for s in args.sizes if s in sizes_set]

        native_out = out_prefix.parent / f"{out_prefix.name}_coherent_native.png"
        _plot_native(native_counts, highlight, native_out)
        print(f"Wrote {native_out}")

        shift_out = out_prefix.parent / f"{out_prefix.name}_native_shift.png"
        _plot_native_shift(native_counts, highlight, shift_out)
        print(f"Wrote {shift_out}")

        for (size, cat, ptype), d in sorted(native_counts.items()):
            n = d["n"] or 1
            summary_rows.append(
                {
                    "kind": "native",
                    "size": size,
                    "category": cat,
                    "prompt_type": ptype,
                    "n": d["n"],
                    "p_in_context": d["in_context"] / n,
                    "p_memorized": d["memorized"] / n,
                }
            )

    if sweep_paths:
        sweep_counts = _aggregate_sweep(sweep_paths, args.metric)
        sizes_set.update(s for s, _, _, _ in sweep_counts.keys())
        sizes = sorted(sizes_set, key=_size_sort_key)
        if args.sizes:
            sizes = [s for s in args.sizes if s in sizes_set]

        decomp = out_prefix.parent / f"{out_prefix.name}_decomp.png"
        _plot_decomp(
            sweep_counts,
            sizes,
            highlight,
            args.decomp_categories,
            decomp,
            show_faint=False,
        )
        print(f"Wrote {decomp}")

        decomp_all = out_prefix.parent / f"{out_prefix.name}_decomp_all.png"
        _plot_decomp(sweep_counts, sizes, highlight, _CATEGORIES, decomp_all)
        print(f"Wrote {decomp_all}")

        controls = out_prefix.parent / f"{out_prefix.name}_controls_L128.png"
        _plot_controls_heatmap(sweep_counts, highlight, controls)
        print(f"Wrote {controls}")

        for (size, cat, filler, length), d in sorted(sweep_counts.items()):
            n = d["n"] or 1
            summary_rows.append(
                {
                    "kind": "sweep",
                    "size": size,
                    "category": cat,
                    "filler_kind": filler,
                    "length": length,
                    "n": d["n"],
                    "p_in_context": d["in_context"] / n,
                    "p_memorized": d["memorized"] / n,
                }
            )

    summary_path = summaries_dir / f"{out_prefix.name}_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump({"metric": args.metric, "rows": summary_rows}, f, indent=2)
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
