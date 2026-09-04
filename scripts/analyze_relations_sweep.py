"""Aggregate per-sample JSONL outputs from evaluate_model.py over multiple
ParaConflict categories and Pythia sizes; emit a clean per-(size, category,
prompt-type) summary plus comparison plots.

Reads ``per_sample_<size>.jsonl`` files (one record per sample, with
``substitution_conflict`` and/or ``coherent_conflict`` subdicts including
``memorized_lp`` / ``in_context_lp`` booleans). Records without one of the
prompt-type subdicts are silently skipped for that cell, so exp00 capitals
records (which have only ``substitution_conflict``) can be mixed in
alongside exp15 relations records.

Two plots are written:

* ``<prefix>.png`` -- stacked-bar P(in-ctx) vs P(memorized) per category x
  size, per prompt-type (legacy view).
* ``<prefix>_diff.png`` -- signed bar of ``P(in-context) - P(memorized)``;
  positive bars mean context wins, negative bars mean memorization wins.

Usage::

    python scripts/analyze_relations_sweep.py \
        --per-sample-glob 'experiments/exp15_other_relations/per_sample/per_sample_*.jsonl' \
                          'experiments/exp00_baseline_capitals_sweep/per_sample/per_sample_capitals_*.jsonl' \
        --output-prefix experiments/exp15_other_relations/plots/exp15_relations \
        --sizes 2.8b 12b
"""

from __future__ import annotations

import argparse
import glob
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch


_SIZE_ORDER = ("160m", "410m", "1b", "1.4b", "2.8b", "6.9b", "12b")


def _size_sort_key(size: str) -> tuple:
    try:
        return (0, _SIZE_ORDER.index(size))
    except ValueError:
        return (1, size)


def _size_from_filename(path: str) -> str:
    m = re.search(r"per_sample_(?:capitals_)?([^/]+)\.jsonl$", path)
    if not m:
        return Path(path).stem
    return m.group(1)


def _load(path: str):
    with open(path, "r", encoding="utf-8") as f:
        header = json.loads(f.readline())
        rows = []
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return header, rows


def _category_from_header(header: dict) -> str | None:
    # exp00 capitals files don't tag rows with `category`; the input_meta
    # source is "ParaConflict World Capital". Use that as a fallback.
    src = (header.get("input_meta") or {}).get("source", "")
    if "World Capital" in src:
        return "World Capital"
    return None


def _plot_bars(counts, categories, sizes, colors, out_path: Path, metric: str):
    """Original twin-bar (P(in-ctx) solid, P(mem) hatched) view."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), squeeze=True)
    width = 0.8 / max(len(sizes), 1)
    for ax_i, ptype in enumerate(("substitution_conflict", "coherent_conflict")):
        ax = axes[ax_i]
        x = np.arange(len(categories))
        for s_idx, size in enumerate(sizes):
            ys_ic, ys_mem = [], []
            for cat in categories:
                d = counts.get((size, cat, ptype),
                               {"n": 0, "memorized": 0, "in_context": 0})
                n = d["n"] or 1
                ys_ic.append(d["in_context"] / n if d["n"] else np.nan)
                ys_mem.append(d["memorized"] / n if d["n"] else np.nan)
            offset = (s_idx - (len(sizes) - 1) / 2) * width
            col = colors.get(size)
            ax.bar(x + offset, ys_ic, width, color=col, alpha=0.95,
                   label=f"{size} P(in-ctx)")
            ax.bar(x + offset, ys_mem, width, color=col, alpha=0.35,
                   hatch="//", edgecolor="white", label=f"{size} P(mem)")
        ax.set_xticks(x)
        ax.set_xticklabels(categories, rotation=20, ha="right", fontsize=9)
        ax.set_ylim(0, 1)
        ax.set_ylabel("rate")
        ax.set_title(
            "L=0 baseline (substitution conflict)"
            if ptype == "substitution_conflict"
            else "ParaConflict coherent passage (~L=120-160 toks)"
        )
        ax.grid(True, axis="y", alpha=0.3)
    axes[0].legend(loc="upper left", fontsize=8, ncol=1)
    fig.suptitle(
        f"ParaConflict relations: in-context vs memorized "
        f"(metric={metric})"
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_diff(counts, categories, sizes, colors, out_path: Path, metric: str):
    """Signed-bar Δ = P(in-context) − P(memorized) per category × size.

    Positive bars mean context wins; negative bars mean the memorized answer
    wins.  Bars get full saturation when positive and a hatched, desaturated
    fill when negative so the direction is also visible at a glance.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), squeeze=True)
    width = 0.8 / max(len(sizes), 1)
    for ax_i, ptype in enumerate(("substitution_conflict", "coherent_conflict")):
        ax = axes[ax_i]
        x = np.arange(len(categories))
        for s_idx, size in enumerate(sizes):
            ys = []
            present = []
            for cat in categories:
                d = counts.get((size, cat, ptype))
                if not d or not d["n"]:
                    ys.append(0.0)
                    present.append(False)
                    continue
                n = d["n"]
                ys.append((d["in_context"] - d["memorized"]) / n)
                present.append(True)
            offset = (s_idx - (len(sizes) - 1) / 2) * width
            col = colors.get(size, "#888888")
            for xi, (yi, ok) in enumerate(zip(ys, present)):
                if not ok:
                    # Mark "no data" cells with a small grey tick at 0.
                    ax.plot(xi + offset, 0.0, marker="x", color="#888888",
                            markersize=6, linestyle="none")
                    continue
                ax.bar(xi + offset, yi, width, color=col, alpha=0.95)
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(categories, rotation=20, ha="right", fontsize=9)
        ax.set_ylim(-1.05, 1.05)
        ax.set_ylabel(r"$\Delta = P(\mathrm{in\text{-}context}) - P(\mathrm{memorized})$")
        ax.set_title(
            "L=0 baseline (substitution conflict)"
            if ptype == "substitution_conflict"
            else "ParaConflict coherent passage (~L=120-160 toks)"
        )
        ax.grid(True, axis="y", alpha=0.3)
        # Annotate the two halves once per subplot.
        ax.text(0.99, 0.97, "context wins", ha="right", va="top",
                transform=ax.transAxes, fontsize=9, color="#444444")
        ax.text(0.99, 0.03, "memorization wins", ha="right", va="bottom",
                transform=ax.transAxes, fontsize=9, color="#444444")
    legend_handles = [
        Patch(facecolor=colors.get(s, "#888888"), edgecolor="none", label=s)
        for s in sizes
    ]
    axes[0].legend(handles=legend_handles, loc="lower left", fontsize=9,
                   ncol=1, title="Pythia size")
    fig.suptitle(
        "ParaConflict relations: context vs memorization "
        f"(Δ = P(ctx) − P(mem), metric={metric})"
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--per-sample-glob",
        type=str,
        nargs="+",
        required=True,
        help=(
            "One or more globs for per_sample_<size>.jsonl files. "
            "Pass the exp15 relations glob plus the exp00 capitals glob to "
            "fold World Capital data into the diff plot."
        ),
    )
    parser.add_argument("--output-prefix", type=str, required=True)
    parser.add_argument(
        "--metric",
        type=str,
        default="lp",
        choices=("lp", "gen"),
        help="Use logprob-argmax (memorized_lp / in_context_lp) or generative.",
    )
    parser.add_argument(
        "--sizes",
        type=str,
        nargs="+",
        default=None,
        help=(
            "Restrict the plots to this list of Pythia sizes "
            "(default: every size found in the data)."
        ),
    )
    args = parser.parse_args()

    paths: list[str] = []
    for pat in args.per_sample_glob:
        paths.extend(sorted(glob.glob(pat)))
    paths = sorted(set(paths))
    if not paths:
        raise SystemExit(f"No files match {args.per_sample_glob!r}")

    counts: dict[tuple[str, str, str], dict[str, int]] = defaultdict(
        lambda: {"n": 0, "memorized": 0, "in_context": 0}
    )
    for path in paths:
        header, rows = _load(path)
        size = header.get("model_size") or _size_from_filename(path)
        header_category = _category_from_header(header)
        for r in rows:
            category = r.get("category") or header_category or ""
            if not category:
                continue
            for ptype in ("substitution_conflict", "coherent_conflict"):
                sub = r.get(ptype, {})
                if not sub:
                    continue
                key = (size, category, ptype)
                counts[key]["n"] += 1
                if args.metric == "lp":
                    flag_mem, flag_ic = "memorized_lp", "in_context_lp"
                else:
                    flag_mem, flag_ic = "memorized_gen", "in_context_gen"
                if sub.get(flag_mem):
                    counts[key]["memorized"] += 1
                if sub.get(flag_ic):
                    counts[key]["in_context"] += 1

    summary = []
    for (size, cat, ptype), d in sorted(counts.items()):
        n = d["n"] or 1
        summary.append({
            "size": size,
            "category": cat,
            "prompt_type": ptype,
            "n": d["n"],
            "memorized": d["memorized"],
            "in_context": d["in_context"],
            "p_memorized": d["memorized"] / n,
            "p_in_context": d["in_context"] / n,
            "p_diff_ctx_minus_mem": (d["in_context"] - d["memorized"]) / n,
        })

    out_prefix = Path(args.output_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    if out_prefix.parent.name == "plots":
        summaries_dir = out_prefix.parent.parent / "summaries"
        summaries_dir.mkdir(parents=True, exist_ok=True)
        summary_path = summaries_dir / f"{out_prefix.name}_summary.json"
    else:
        summary_path = out_prefix.parent / f"{out_prefix.name}_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump({"metric": args.metric, "rows": summary}, f, indent=2)

    sizes_in_data = sorted({s for (s, _, _) in counts.keys()},
                           key=_size_sort_key)
    if args.sizes:
        sizes = [s for s in args.sizes if s in sizes_in_data]
        missing = [s for s in args.sizes if s not in sizes_in_data]
        if missing:
            print(f"Warning: requested sizes not present in data: {missing}")
        if not sizes:
            sizes = sizes_in_data
    else:
        sizes = sizes_in_data

    categories = sorted({c for (_, c, _) in counts.keys()})
    # Put World Capital first (it's the canonical relation in this repo).
    categories.sort(key=lambda c: (c != "World Capital", c))

    colors = {
        "160m": "#9ecae1", "410m": "#6baed6", "1b": "#4292c6",
        "1.4b": "#2171b5", "2.8b": "#1f77b4", "6.9b": "#2ca02c",
        "12b": "#d62728",
    }

    out_png = out_prefix.parent / f"{out_prefix.name}.png"
    _plot_bars(counts, categories, sizes, colors, out_png, args.metric)
    print(f"Wrote {out_png}")

    out_diff_png = out_prefix.parent / f"{out_prefix.name}_diff.png"
    _plot_diff(counts, categories, sizes, colors, out_diff_png, args.metric)
    print(f"Wrote {out_diff_png}")

    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
