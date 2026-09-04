"""Plot P(in-context) − P(memorized) vs model size (log scale) across families.

Designed for exp00 capitals (``evaluate_capitals.py``) and exp23 ParaConflict
relations (``evaluate_model.py``).  Positive y ⇒ the model picks the in-context
distractor more often than the memorized gold answer on average; negative y ⇒
memorization wins.

Usage::

    # exp21 — world capitals
    python scripts/plot_context_memory_score.py \\
        --per-sample-glob 'experiments/exp21_context_memory_score/per_sample/per_sample_capitals_*.jsonl' \\
        --output-prefix experiments/exp21_context_memory_score/plots/context_memory_score

    # exp23 — five non-capital ParaConflict categories (one panel each)
    python scripts/plot_context_memory_score.py \\
        --per-sample-glob 'experiments/exp23_context_memory_relations/per_sample/per_sample_relations_*.jsonl' \\
        --output-prefix experiments/exp23_context_memory_relations/plots/context_memory_score \\
        --by-category --metric gen --exclude-categories 'World Capital'

    # exp23 — all six categories (fold in exp21 World Capital per-sample JSONLs)
    python scripts/plot_context_memory_score.py \\
        --per-sample-glob \\
            'experiments/exp23_context_memory_relations/per_sample/per_sample_relations_*.jsonl' \\
            'experiments/exp21_context_memory_score/per_sample/per_sample_capitals_*.jsonl' \\
        --output-prefix experiments/exp23_context_memory_relations/plots/context_memory_score_gen_all \\
        --by-category --metric gen --legend-panel bottom-middle

    # exp24 — bottom / top 20% countries by wordfreq Zipf
    python scripts/plot_context_memory_score.py \\
        --per-sample-glob 'experiments/exp24_context_memory_country_freq/per_sample/per_sample_capitals_*.jsonl' \\
        --output-prefix experiments/exp24_context_memory_country_freq/plots/context_memory_score_gen_bottom20 \\
        --metric gen --freq-json experiments/exp00_baseline_capitals_sweep/summaries/capitals_subject_frequency.json \\
        --country-fraction 0.2 --country-tail bottom

    # exp24 — hypothesis test (heatmap + Δ panel with shared y-axis)
    python scripts/test_country_freq_memorization.py \\
        --per-sample-glob 'experiments/exp21_context_memory_score/per_sample/per_sample_capitals_*.jsonl' \\
        --frequency-json experiments/exp00_baseline_capitals_sweep/summaries/capitals_subject_frequency.json \\
        --output-prefix experiments/exp24_context_memory_country_freq/plots/country_freq_mem_hypothesis \\
        --metric gen --n-bins 5
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt

# Plotting-only: load config without ic_factual.__init__ (which imports torch).
import importlib.util

_config_path = Path(__file__).resolve().parent.parent / "ic_factual" / "config.py"
_spec = importlib.util.spec_from_file_location("_ic_factual_config", _config_path)
assert _spec and _spec.loader
_config_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_config_mod)
model_family = _config_mod.model_family
model_param_billions = _config_mod.model_param_billions


# Small log-scale x offsets so families that share a size (e.g. Ministral 3b/8b/14b)
# do not sit on top of each other.
_FAMILY_X_JITTER: dict[str, float] = {
    "gpt2": 0.90,
    "pythia": 0.96,
    "qwen3-base": 1.0,
    "qwen3": 1.04,
    "ministral3-base": 0.94,
    "ministral3-instruct": 1.0,
    "ministral3-reasoning": 1.06,
    "other": 1.0,
}

# Multi-panel (--by-category) typography tuned for paper figures.
_BY_CATEGORY_FONTS = {
    "axis_label": 14,
    "legend": 10,
    "panel_title": 11,
    "tick": 11,
    "direction_guide": 10,
    "suptitle": 14,
}

_ANNOT_OFFSET: dict[str, tuple[int, int]] = {
    "gpt2": (5, 7),
    "pythia": (5, -9),
    "qwen3-base": (5, 5),
    "qwen3": (5, -7),
    "ministral3-base": (-14, 5),
    "ministral3-instruct": (5, 5),
    "ministral3-reasoning": (5, -9),
    "other": (4, 4),
}


def _collect_paths(patterns: list[str]) -> list[Path]:
    """Expand globs; keep readable files without requiring absolute resolve()."""
    paths: list[Path] = []
    for pat in patterns:
        paths.extend(Path(p) for p in glob.glob(pat))
    seen: set[str] = set()
    out: list[Path] = []
    for path in paths:
        key = str(path)
        if key in seen:
            continue
        try:
            readable = path.is_file() or (path.is_symlink() and path.resolve().is_file())
        except OSError:
            readable = False
        if not readable:
            continue
        seen.add(key)
        out.append(path)
    return sorted(out)


_FAMILY_STYLE: dict[str, dict] = {
    "pythia": {"color": "#2171b5", "marker": "o", "label": "Pythia"},
    "qwen3": {"color": "#2ca02c", "marker": "s", "label": "Qwen3 (post-trained)"},
    "qwen3-base": {"color": "#98df8a", "marker": "D", "label": "Qwen3-Base"},
    "ministral3-reasoning": {"color": "#004d40", "marker": "v", "label": "Ministral-3 Reasoning"},
    "ministral3-instruct": {"color": "#00897b", "marker": "^", "label": "Ministral-3 Instruct"},
    "ministral3-base": {"color": "#6ec4bc", "marker": "P", "label": "Ministral-3 Base"},
    "gpt2": {"color": "#e377c2", "marker": "h", "label": "GPT-2"},
    "other": {"color": "#7f7f7f", "marker": "x", "label": "other"},
}


def _load_jsonl(path: Path) -> tuple[dict, list[dict]]:
    meta: dict = {}
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if obj.get("_meta"):
                meta = obj
            else:
                records.append(obj)
    return meta, records


_PARACONFLICT_RELATION_CATEGORIES: tuple[str, ...] = (
    "World Capital",
    "Athlete Sport",
    "Book Author",
    "Company Headquarter",
    "Company Founder",
    "Official Language",
)


def _canonical_category(category: str) -> str:
    if category == "Athelete Sport":
        return "Athlete Sport"
    return category


def _load_country_freq(path: Path, *, metric: str = "zipf") -> dict[str, float]:
    data = json.loads(path.read_text(encoding="utf-8"))
    subjects = data.get("subjects", data)
    out: dict[str, float] = {}
    for country, info in subjects.items():
        if country.startswith("_"):
            continue
        if isinstance(info, dict):
            val = info.get(metric) or info.get("zipf") or info.get("zipf_min_word")
            if val is None:
                continue
            out[country] = float(val)
        else:
            out[country] = float(info)
    return out


def _countries_by_freq_tail(
    freq: dict[str, float], *, fraction: float, tail: str
) -> tuple[set[str], dict]:
    """Return the bottom or top ``fraction`` of countries by frequency (count-based)."""
    if not (0.0 < fraction <= 1.0):
        raise ValueError(f"country-fraction must be in (0, 1], got {fraction}")
    items = sorted(freq.items(), key=lambda kv: kv[1])
    k = max(1, round(len(items) * fraction))
    if tail == "bottom":
        selected_items = items[:k]
        bin_idx = 0
    elif tail == "top":
        selected_items = items[-k:]
        bin_idx = 1
    else:
        raise ValueError(f"country-tail must be 'bottom' or 'top', got {tail!r}")
    selected = {c for c, _ in selected_items}
    lo = selected_items[0][1]
    hi = selected_items[-1][1]
    meta = {
        "n_countries_total": len(freq),
        "n_countries_selected": len(selected),
        "country_fraction": fraction,
        "country_tail": tail,
        "freq_bin_index": bin_idx,
        "freq_bin_range": [lo, hi],
    }
    return selected, meta


def _label_from_path(path: Path, meta: dict) -> str:
    label = meta.get("label") or meta.get("model_size")
    if label:
        return str(label)
    for pat in (
        r"per_sample_capitals_(.+)\.jsonl$",
        r"per_sample_relations_(.+)\.jsonl$",
        r"per_sample_(.+)\.jsonl$",
    ):
        m = re.search(pat, path.name)
        if m:
            return m.group(1)
    return path.stem



def _aggregate(
    path: Path,
    metric: str,
    *,
    category: str | None = None,
    exclude_categories: set[str] | None = None,
    countries: set[str] | None = None,
) -> dict | None:
    meta, records = _load_jsonl(path)
    if not records:
        return None
    label = _label_from_path(path, meta)
    mem_key = f"memorized_{metric}"
    ctx_key = f"in_context_{metric}"
    n = mem = ctx = 0
    for rec in records:
        rec_cat = _canonical_category((rec.get("category") or "").strip())
        if exclude_categories and rec_cat.lower() in exclude_categories:
            continue
        if category is not None and rec_cat != category:
            continue
        if countries is not None:
            country = (rec.get("country") or rec.get("subject") or "").strip()
            if country not in countries:
                continue
        block = rec.get("substitution_conflict") or {}
        if not block:
            continue
        n += 1
        if block.get(mem_key):
            mem += 1
        if block.get(ctx_key):
            ctx += 1
    if n == 0:
        return None
    p_mem = mem / n
    p_ctx = ctx / n
    params_b = model_param_billions(label)
    row: dict = {
        "label": label,
        "family": model_family(label),
        "n": n,
        "memorized": mem,
        "in_context": ctx,
        "p_memorized": p_mem,
        "p_in_context": p_ctx,
        "score_ctx_minus_mem": p_ctx - p_mem,
        "params_b": params_b,
        "path": str(path),
    }
    if category is not None:
        row["category"] = category
    return row


def _discover_categories(
    paths: list[Path],
    metric: str,
    *,
    exclude_categories: set[str] | None = None,
) -> list[str]:
    found: set[str] = set()
    for path in paths:
        _, records = _load_jsonl(path)
        for rec in records:
            cat = _canonical_category((rec.get("category") or "").strip())
            if not cat:
                continue
            if exclude_categories and cat.lower() in exclude_categories:
                continue
            found.add(cat)
    if not found:
        return []
    ordered = [c for c in _PARACONFLICT_RELATION_CATEGORIES if c in found]
    ordered.extend(sorted(found - set(ordered)))
    return ordered


def _format_params_label(params_b: float) -> str:
    """Short point label: parameter count in billions, at most one decimal."""
    truncated = math.floor(params_b * 10) / 10
    if truncated >= 10:
        return f"{truncated:.0f}"
    return f"{truncated:.1f}".rstrip("0").rstrip(".")


def _add_direction_guide(ax: plt.Axes, *, fontsize: float = 8.5) -> None:
    """Left-side guide centered on y=0: up favors in-context, down favors memorized."""
    from matplotlib.transforms import blended_transform_factory

    trans = blended_transform_factory(ax.transAxes, ax.transData)
    y0 = 0.0
    ymin, ymax = ax.get_ylim()
    dy = (ymax - ymin) * 0.11
    x_arrow = 0.03
    x_label = 0.065

    arrow_kw = dict(arrowstyle="-|>", color="0.45", lw=1.8, mutation_scale=12, alpha=0.65)
    label_kw = dict(fontsize=fontsize, color="0.45", alpha=0.65)

    ax.annotate(
        "",
        xy=(x_arrow, y0 + dy),
        xytext=(x_arrow, y0),
        xycoords=trans,
        textcoords=trans,
        arrowprops=arrow_kw,
    )
    ax.annotate(
        "favors in-context",
        xy=(x_label, y0 + dy * 0.5),
        xycoords=trans,
        textcoords=trans,
        va="center",
        ha="left",
        **label_kw,
    )
    ax.annotate(
        "",
        xy=(x_arrow, y0 - dy),
        xytext=(x_arrow, y0),
        xycoords=trans,
        textcoords=trans,
        arrowprops=arrow_kw,
    )
    ax.annotate(
        "favors memorized",
        xy=(x_label, y0 - dy * 0.5),
        xycoords=trans,
        textcoords=trans,
        va="center",
        ha="left",
        **label_kw,
    )


def _plot_families_on_axes(
    ax: plt.Axes,
    points: list[dict],
    *,
    title: str,
    show_legend: bool = True,
    annotate_points: bool = True,
    direction_guide: bool = True,
    panel_title_size: float = 10,
    legend_size: float = 7,
    direction_guide_size: float = 8.5,
) -> None:
    plotted = [p for p in points if p["params_b"] is not None]
    if not plotted:
        ax.set_title(title)
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
        return

    families = sorted({p["family"] for p in plotted}, key=lambda f: (f == "other", f))
    for family in families:
        style = _FAMILY_STYLE.get(family, _FAMILY_STYLE["other"])
        pts = sorted(
            (p for p in plotted if p["family"] == family),
            key=lambda p: p["params_b"],
        )
        xs = [p["params_b"] * _FAMILY_X_JITTER.get(family, 1.0) for p in pts]
        ys = [p["score_ctx_minus_mem"] for p in pts]
        ax.plot(
            xs,
            ys,
            color=style["color"],
            marker=style["marker"],
            linestyle="-",
            linewidth=1.2,
            markersize=6 if not annotate_points else 7,
            label=style["label"],
        )
        if annotate_points:
            ox, oy = _ANNOT_OFFSET.get(family, (4, 4))
            for p, x in zip(pts, xs):
                ax.annotate(
                    _format_params_label(p["params_b"]),
                    (x, p["score_ctx_minus_mem"]),
                    textcoords="offset points",
                    xytext=(ox, oy),
                    fontsize=6,
                    color=style["color"],
                )

    ax.axhline(0.0, color="black", linewidth=0.8, linestyle="--", alpha=0.6)
    ax.set_xscale("log")
    ymax = max(p["score_ctx_minus_mem"] for p in plotted)
    ymin = min(p["score_ctx_minus_mem"] for p in plotted)
    ax.set_ylim(min(-1.0, ymin - 0.08), max(1.05, ymax + 0.08))
    ax.set_title(title, fontsize=panel_title_size)
    ax.grid(True, which="both", alpha=0.25)
    if direction_guide:
        _add_direction_guide(ax, fontsize=direction_guide_size)
    if show_legend:
        ax.legend(loc="lower left", fontsize=legend_size, framealpha=0.92)


def _plot(points: list[dict], out_prefix: Path, *, title: str) -> None:
    skipped = [p for p in points if p["params_b"] is None]
    if skipped:
        print("  skip (unknown params):", ", ".join(p["label"] for p in skipped))

    fig, ax = plt.subplots(figsize=(10, 5.5))
    _plot_families_on_axes(ax, points, title=title)
    ax.set_xlabel("Parameters (billions, log scale)")
    ax.set_ylabel("P(in-context) − P(memorized)")
    fig.subplots_adjust(left=0.11, right=0.97)
    fig.tight_layout()
    png = out_prefix.parent / f"{out_prefix.name}.png"
    png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {png}")


def _plot_by_category(
    paths: list[Path] | None,
    metric: str,
    out_prefix: Path,
    *,
    categories: list[str],
    exclude_categories: set[str] | None,
    suptitle: str,
    legend_panel: str = "bottom-right",
    precomputed_points: list[dict] | None = None,
) -> list[dict]:
    ncat = len(categories)
    ncols = 3
    fonts = _BY_CATEGORY_FONTS
    legend_inside = legend_panel == "bottom-middle" and ncat == math.ceil(ncat / ncols) * ncols
    if legend_inside:
        nrows = math.ceil(ncat / ncols)
    else:
        # Reserve an empty subplot for the family legend (exp23 five-category view).
        nrows = math.ceil((ncat + 1) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4.2 * nrows), squeeze=False)
    all_points: list[dict] = []

    for idx, category in enumerate(categories):
        row_i, col_i = divmod(idx, ncols)
        ax = axes[row_i][col_i]
        cat_points: list[dict] = []
        if precomputed_points is not None:
            cat_points = [p for p in precomputed_points if p.get("category") == category]
        elif paths:
            for path in paths:
                row = _aggregate(
                    path,
                    metric,
                    category=category,
                    exclude_categories=exclude_categories,
                )
                if row is None:
                    continue
                cat_points.append(row)
                print(
                    f"  {category:24s}  {row['label']:28s}  n={row['n']:5d}  "
                    f"score={row['score_ctx_minus_mem']:+.3f}"
                )
        all_points.extend(cat_points)
        _plot_families_on_axes(
            ax,
            cat_points,
            title=category,
            show_legend=False,
            annotate_points=False,
            direction_guide=(col_i == 0),
            panel_title_size=fonts["panel_title"],
            direction_guide_size=fonts["direction_guide"],
        )
        ax.tick_params(axis="both", labelsize=fonts["tick"])
        if col_i == 0:
            ax.tick_params(axis="y", pad=7)
        if row_i < nrows - 1 or col_i > 0:
            ax.tick_params(labelbottom=False)
        if col_i > 0:
            ax.tick_params(labelleft=False)

    legend_ax = None
    if legend_inside:
        legend_ax = axes[nrows - 1][ncols // 2]
    elif ncat < nrows * ncols:
        legend_ax = axes[nrows - 1][ncols - 1]

    for idx in range(ncat, nrows * ncols):
        row_i, col_i = divmod(idx, ncols)
        ax = axes[row_i][col_i]
        if ax is legend_ax:
            continue
        ax.axis("off")

    handles, labels = axes[0][0].get_legend_handles_labels()
    if not handles:
        for ax_row in axes:
            for ax in ax_row:
                if ax is legend_ax:
                    continue
                h, l = ax.get_legend_handles_labels()
                if h:
                    handles, labels = h, l
                    break
            if handles:
                break

    if handles and legend_ax is not None:
        if not legend_inside:
            legend_ax.axis("off")
        legend_ax.legend(
            handles,
            labels,
            loc="lower center" if legend_inside else "center",
            fontsize=fonts["legend"],
            framealpha=0.92,
        )

    fig.suptitle(suptitle, fontsize=fonts["suptitle"], y=1.01)
    fig.supxlabel("Parameters (billions, log scale)", fontsize=fonts["axis_label"])
    fig.supylabel(
        "P(in-context) − P(memorized)",
        fontsize=fonts["axis_label"],
        x=0.008,
    )
    fig.tight_layout()
    fig.subplots_adjust(left=0.09, bottom=0.08)
    png = out_prefix.parent / f"{out_prefix.name}.png"
    png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {png}")
    return all_points


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Plot signed context-vs-memorization score against model size "
            "(log scale) for capitals or ParaConflict relation eval JSONLs."
        )
    )
    parser.add_argument(
        "--per-sample-glob",
        nargs="*",
        default=None,
        help="One or more globs for per_sample_*.jsonl files.",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=None,
        help="Plot directly from an existing <prefix>_summary.json without re-aggregating per-sample files.",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        required=True,
        help="Output path prefix (writes <prefix>.png and <prefix>_summary.json).",
    )
    parser.add_argument(
        "--metric",
        choices=("lp", "gen"),
        default="lp",
        help="Classifier column prefix (default: lp = logprob argmax).",
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Plot title (default depends on mode).",
    )
    parser.add_argument(
        "--exclude-pattern",
        default="_smoke",
        help="Skip paths whose filename contains this substring (default: _smoke).",
    )
    parser.add_argument(
        "--by-category",
        action="store_true",
        help=(
            "Emit one subplot per ParaConflict category (exp23 relations). "
            "Records are filtered by the per-row ``category`` field."
        ),
    )
    parser.add_argument(
        "--exclude-categories",
        nargs="*",
        default=None,
        help="Drop these categories (case-insensitive), e.g. 'World Capital'.",
    )
    parser.add_argument(
        "--legend-panel",
        choices=("bottom-right", "bottom-middle"),
        default="bottom-right",
        help=(
            "With --by-category: place the family legend in an empty subplot "
            "(bottom-right, default) or overlay it on the bottom-center data "
            "panel (bottom-middle; keeps a 2x3 grid for six categories)."
        ),
    )
    parser.add_argument(
        "--freq-json",
        type=Path,
        default=None,
        help=(
            "Country subject-frequency JSON (e.g. capitals_subject_frequency.json). "
            "Required with --country-fraction."
        ),
    )
    parser.add_argument(
        "--freq-metric",
        default="zipf",
        help="Field inside the frequency JSON subjects table (default: zipf).",
    )
    parser.add_argument(
        "--country-fraction",
        type=float,
        default=None,
        help=(
            "Keep prompts whose country lies in one frequency percentile bin "
            "(e.g. 0.2 = 20%%). Use with --country-tail."
        ),
    )
    parser.add_argument(
        "--country-tail",
        choices=("bottom", "top"),
        default=None,
        help="With --country-fraction: bottom = least frequent, top = most frequent.",
    )
    args = parser.parse_args()
    exclude_cats = (
        {c.strip().lower() for c in args.exclude_categories}
        if args.exclude_categories
        else None
    )
    if (args.country_fraction is None) != (args.country_tail is None):
        raise SystemExit("--country-fraction and --country-tail must be set together.")
    if args.country_fraction is not None and args.freq_json is None:
        raise SystemExit("--freq-json is required when filtering by country frequency.")

    countries: set[str] | None = None
    country_filter_meta: dict | None = None
    if args.country_fraction is not None:
        freq = _load_country_freq(args.freq_json, metric=args.freq_metric)
        countries, country_filter_meta = _countries_by_freq_tail(
            freq, fraction=args.country_fraction, tail=args.country_tail
        )
        print(
            f"Country filter: {args.country_tail} {args.country_fraction:.0%} "
            f"({country_filter_meta['n_countries_selected']} / "
            f"{country_filter_meta['n_countries_total']} countries, "
            f"Zipf {country_filter_meta['freq_bin_range'][0]:.2f}–"
            f"{country_filter_meta['freq_bin_range'][1]:.2f})"
        )

    out_prefix = args.output_prefix
    if out_prefix.parent.name == "plots":
        summaries_dir = out_prefix.parent.parent / "summaries"
    else:
        summaries_dir = out_prefix.parent
    summaries_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.summary_json or (summaries_dir / f"{out_prefix.name}_summary.json")

    if args.summary_json or (not args.per_sample_glob and summary_path.is_file()):
        target_json = args.summary_json or summary_path
        print(f"Loading precomputed summary from {target_json}")
        summary = json.loads(target_json.read_text(encoding="utf-8"))
        if args.by_category:
            categories = summary.get("categories", list(_PARACONFLICT_RELATION_CATEGORIES))
            if exclude_cats:
                categories = [c for c in categories if c.lower() not in exclude_cats]
            title = args.title or "In-context vs memorized (ParaConflict relations)"
            _plot_by_category(
                paths=None,
                metric=args.metric,
                out_prefix=out_prefix,
                categories=categories,
                exclude_categories=exclude_cats,
                suptitle=title,
                legend_panel=args.legend_panel,
                precomputed_points=summary.get("models", []),
            )
            return
        else:
            title = args.title or "In-context vs memorized (world capitals)"
            points = summary.get("models", summary if isinstance(summary, list) else [])
            _plot(points, out_prefix, title=title)
            return

    paths = _collect_paths(args.per_sample_glob or [])
    if args.exclude_pattern:
        paths = [p for p in paths if args.exclude_pattern not in p.name]
    if not paths:
        raise SystemExit(f"No files match {args.per_sample_glob!r}")

    if args.by_category:
        categories = _discover_categories(
            paths, args.metric, exclude_categories=exclude_cats
        )
        if not categories:
            raise SystemExit("No categories found in per-sample files.")
        title = args.title or "In-context vs memorized (ParaConflict relations)"
        points = _plot_by_category(
            paths,
            args.metric,
            out_prefix,
            categories=categories,
            exclude_categories=exclude_cats,
            suptitle=title,
            legend_panel=args.legend_panel,
        )
        if not points:
            raise SystemExit("No usable per-category aggregates.")
        summary = {
            "metric": args.metric,
            "score_definition": "p_in_context - p_memorized",
            "categories": categories,
            "models": sorted(
                points,
                key=lambda p: (p.get("category", ""), p["family"], p["params_b"] or 0),
            ),
        }
    else:
        if args.title:
            title = args.title
        elif countries is not None and country_filter_meta is not None:
            tail_label = (
                "least frequent" if args.country_tail == "bottom" else "most frequent"
            )
            pct = int(round(args.country_fraction * 100))
            title = (
                f"In-context vs memorized (world capitals, "
                f"{pct}% {tail_label} countries)"
            )
        else:
            title = "In-context vs memorized (world capitals)"
        points = []
        for path in paths:
            row = _aggregate(
                path, args.metric, exclude_categories=exclude_cats, countries=countries
            )
            if row is None:
                print(f"  skip empty: {path.name}")
                continue
            points.append(row)
            print(
                f"  {row['label']:28s}  n={row['n']:6d}  "
                f"ctx={row['p_in_context']:.3f}  mem={row['p_memorized']:.3f}  "
                f"score={row['score_ctx_minus_mem']:+.3f}"
            )
        if not points:
            raise SystemExit("No usable per-sample files.")
        summary = {
            "metric": args.metric,
            "score_definition": "p_in_context - p_memorized",
            "models": sorted(points, key=lambda p: (p["family"], p["params_b"] or 0)),
        }
        if country_filter_meta is not None:
            summary["country_filter"] = {
                **country_filter_meta,
                "freq_json": str(args.freq_json),
                "freq_metric": args.freq_metric,
                "countries": sorted(countries or []),
            }
        _plot(points, out_prefix, title=title)

    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
