"""Replicate the 'in-context vs memorized answer by subject frequency' figure.

Inputs
------
* ``output/subject_frequency.json`` produced by ``compute_subject_frequency.py``
  (either ``--proxy wordfreq`` or ``--proxy model``).
* ``output/per_sample_<size>.jsonl`` files produced by either:
    - ``evaluate_model.py --per-sample-output`` (ParaConflict, has both
      ``substitution_conflict`` and ``coherent_conflict`` blocks); or
    - ``scripts/evaluate_capitals.py`` (paper cross-product, has only
      ``substitution_conflict`` plus a ``distractor_country`` field).

Output
------
* ``<out_prefix>.png``           -- multi-panel matplotlib figure.
* ``<out_prefix>.html``          -- interactive plotly version (optional).
* ``<out_prefix>_summary.json``  -- raw per-bin numbers.

The script is agnostic to which generator wrote the per-sample JSONLs: it
auto-detects which conflict rows are present and which bin-by fields are
available.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ic_factual.config import PYTHIA_SIZES, QWEN3_BASE_SIZES, QWEN3_SIZES


_CONFLICT_KEYS = ("substitution_conflict", "coherent_conflict")


def _panel_label(size: str) -> str:
    """Panel title for a per-sample file's model identifier.

    Bare Pythia sizes (``"160m"``, ``"1.4b"``, ...) get the ``"Pythia-"``
    prefix to match the paper.  Anything already family-prefixed
    (``"qwen3-0.6b"``, ``"llama-7b"``, ...) is shown verbatim so we don't
    end up with labels like ``"Pythia-qwen3-0.6b"``.
    """
    if "-" in size:
        return size
    return f"Pythia-{size}"


def _load_jsonl(path: Path) -> tuple[dict, list[dict]]:
    meta: dict = {}
    records: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
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


def assign_percentile_bins(
    values: dict[str, float], n_bins: int
) -> tuple[dict[str, int], list[tuple[float, float]]]:
    """Return ``{key: bin_idx in [0, n_bins)}`` and bin edges as (lo, hi) pairs."""
    if not values:
        return {}, []
    items = sorted(values.items(), key=lambda kv: kv[1])
    keys = [k for k, _ in items]
    vals = np.array([v for _, v in items], dtype=float)
    quantiles = np.linspace(0, 100, n_bins + 1)
    edges = np.percentile(vals, quantiles)
    edges[0] -= 1e-9
    edges[-1] += 1e-9
    bin_idx = np.clip(np.searchsorted(edges[1:-1], vals, side="right"), 0, n_bins - 1)
    bins: dict[str, int] = {k: int(b) for k, b in zip(keys, bin_idx)}
    bin_ranges = [(float(edges[i]), float(edges[i + 1])) for i in range(n_bins)]
    return bins, bin_ranges


def aggregate(
    per_sample: list[dict],
    key_bin: dict[str, int],
    n_bins: int,
    conflict_key: str,
    metric: str = "gen",  # "gen" or "lp"
    bin_field: str = "subject",
) -> dict:
    """Return per-bin counts/proportions for one conflict type and metric."""
    mem_key = f"memorized_{metric}"
    ctx_key = f"in_context_{metric}"

    counts = np.zeros(n_bins, dtype=int)
    mem = np.zeros(n_bins, dtype=int)
    ctx = np.zeros(n_bins, dtype=int)
    for rec in per_sample:
        key = rec.get(bin_field)
        if key is None or key not in key_bin:
            continue
        b = key_bin[key]
        block = rec.get(conflict_key, {})
        if not block:
            continue
        counts[b] += 1
        if block.get(mem_key):
            mem[b] += 1
        if block.get(ctx_key):
            ctx[b] += 1

    with np.errstate(divide="ignore", invalid="ignore"):
        p_mem = np.where(counts > 0, mem / np.maximum(counts, 1), np.nan)
        p_ctx = np.where(counts > 0, ctx / np.maximum(counts, 1), np.nan)

    return {
        "counts": counts.tolist(),
        "memorized": mem.tolist(),
        "in_context": ctx.tolist(),
        "p_memorized": p_mem.tolist(),
        "p_in_context": p_ctx.tolist(),
    }


def _save_matplotlib(
    summary: dict,
    sizes: list[str],
    n_bins: int,
    out_path: Path,
    title_suffix: str,
    rows: list[tuple[str, str]],
    metric_kind: str = "gen",
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_rows = len(rows)
    fig, axes = plt.subplots(
        n_rows,
        len(sizes),
        figsize=(max(2.6 * len(sizes), 7.0), 3.2 * n_rows + 0.6),
        sharex=True,
        sharey=True,
        squeeze=False,
    )

    x = np.arange(1, n_bins + 1)
    xticks_labels = [f"{int(round(100 * (i + 1) / n_bins))}th" for i in range(n_bins)]

    for col, size in enumerate(sizes):
        for row, (conflict_key, row_title) in enumerate(rows):
            ax = axes[row][col]
            data = summary["models"][size][metric_kind].get(conflict_key)
            if data is None:
                ax.set_visible(False)
                continue
            p_mem = np.array(data["p_memorized"], dtype=float)
            p_ctx = np.array(data["p_in_context"], dtype=float)

            ax.plot(x, p_ctx, color="#2E86DE", marker="o", label="in-context answer", lw=1.6)
            ax.fill_between(x, 0, p_ctx, color="#2E86DE", alpha=0.18)
            ax.plot(x, p_mem, color="#E74C3C", marker="x", label="memorized answer", lw=1.6)
            ax.fill_between(x, 0, p_mem, color="#E74C3C", alpha=0.18)

            ax.set_ylim(0.0, 1.0)
            ax.set_xticks(x)
            ax.set_xticklabels(xticks_labels, rotation=45, fontsize=7)

            if row == 0:
                ax.set_title(_panel_label(size), fontsize=10)
            if col == 0:
                ax.set_ylabel(f"{row_title}\nProportion of Answers", fontsize=9)
            if row == n_rows - 1:
                ax.set_xlabel("Percentile of Frequency", fontsize=8)
            if col == 0 and row == 0:
                ax.legend(loc="upper left", fontsize=7, framealpha=0.85)

    suptitle = (
        "Memorized vs In-Context Answers across Model Sizes "
        f"({title_suffix})"
    )
    fig.suptitle(suptitle, fontsize=11, y=0.995)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def _save_plotly(
    summary: dict,
    sizes: list[str],
    n_bins: int,
    out_path: Path,
    title_suffix: str,
    rows: list[tuple[str, str]],
    metric_kind: str = "gen",
) -> None:
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        print("plotly not available; skipping HTML output")
        return

    n_rows = len(rows)
    subplot_titles = [_panel_label(s) for s in sizes] + [""] * (len(sizes) * (n_rows - 1))
    fig = make_subplots(
        rows=n_rows,
        cols=len(sizes),
        shared_xaxes=True,
        shared_yaxes=True,
        subplot_titles=subplot_titles,
        horizontal_spacing=0.02,
        vertical_spacing=0.08,
    )

    x = list(range(1, n_bins + 1))
    xticks_labels = [f"{int(round(100 * (i + 1) / n_bins))}th" for i in range(n_bins)]

    show_legend = True
    for col, size in enumerate(sizes, start=1):
        for row, (conflict_key, row_title) in enumerate(rows, start=1):
            data = summary["models"][size][metric_kind].get(conflict_key)
            if data is None:
                continue
            fig.add_trace(
                go.Scatter(
                    x=x, y=data["p_in_context"], mode="lines+markers", name="in-context",
                    line=dict(color="#2E86DE"), fill="tozeroy",
                    fillcolor="rgba(46,134,222,0.18)",
                    showlegend=show_legend,
                ),
                row=row, col=col,
            )
            fig.add_trace(
                go.Scatter(
                    x=x, y=data["p_memorized"], mode="lines+markers", name="memorized",
                    line=dict(color="#E74C3C"), fill="tozeroy",
                    fillcolor="rgba(231,76,60,0.18)",
                    showlegend=show_legend,
                ),
                row=row, col=col,
            )
            show_legend = False
            if col == 1:
                fig.update_yaxes(title_text=row_title, row=row, col=col)
            fig.update_xaxes(
                tickmode="array", tickvals=x, ticktext=xticks_labels, row=row, col=col
            )

    fig.update_layout(
        title=(
            "Count of Memorized and In-Context Answers across Model Sizes "
            f"— {title_suffix}"
        ),
        height=350 * n_rows,
        width=240 * max(len(sizes), 4),
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(out_path, include_plotlyjs="cdn")
    print(f"Wrote {out_path}")


def _detect_conflict_rows(records: list[dict]) -> list[str]:
    found = []
    for k in _CONFLICT_KEYS:
        if any(k in r for r in records):
            found.append(k)
    return found or ["substitution_conflict"]


def _detect_metric(
    records: list[dict], conflict_keys: list[str]
) -> str:
    """Pick 'gen' or 'lp' depending on which one was populated.

    For each candidate metric we count how many records have any boolean
    classification set; whichever has more wins. Ties prefer 'gen' (paper
    matches generation). If neither, fall back to 'gen' (the script writes
    NaN columns -- nothing to plot but won't crash).
    """
    counts = {"gen": 0, "lp": 0}
    for rec in records:
        for ck in conflict_keys:
            block = rec.get(ck) or {}
            for m in counts:
                if block.get(f"memorized_{m}") or block.get(f"in_context_{m}"):
                    counts[m] += 1
                    break
    if counts["lp"] > counts["gen"]:
        return "lp"
    return "gen"


def main():
    parser = argparse.ArgumentParser(
        description="Plot in-context vs memorized answers by subject/distractor frequency."
    )
    parser.add_argument(
        "--per-sample-glob",
        type=str,
        default="output/per_sample_*.jsonl",
        help="Glob for per-sample JSONLs.",
    )
    parser.add_argument(
        "--frequency-json",
        type=str,
        default="output/subject_frequency.json",
        help="Path to the subject-frequency JSON.",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="output/frequency_breakdown",
        help="Prefix for output files (.png, .html, _summary.json).",
    )
    parser.add_argument(
        "--n-bins", type=int, default=10,
        help="Number of frequency percentile bins.",
    )
    parser.add_argument(
        "--categories",
        type=str,
        nargs="*",
        default=None,
        help="Optional categories to restrict the per-sample records to.",
    )
    parser.add_argument(
        "--frequency-metric",
        type=str,
        default=None,
        help=(
            "Which field of the frequency JSON to use. "
            "Defaults to 'zipf' for wordfreq, else 'first_token_logprob'."
        ),
    )
    parser.add_argument(
        "--bin-by",
        type=str,
        default="subject",
        help=(
            "Per-sample field used to look up the frequency proxy. "
            "Use 'subject' (= queried country) for the paper's country axis, "
            "or 'distractor_country' for the in-context-city axis."
        ),
    )
    parser.add_argument(
        "--rows",
        type=str,
        nargs="*",
        default=None,
        help=(
            "Conflict rows to plot, in order. Default: auto-detect from data."
        ),
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="auto",
        choices=("auto", "gen", "lp"),
        help=(
            "Which classifier output to plot: 'gen' (generation + substring), "
            "'lp' (logprob argmax). 'auto' picks whichever is populated."
        ),
    )
    args = parser.parse_args()

    freq_path = Path(args.frequency_json)
    if not freq_path.exists():
        raise SystemExit(
            f"Subject frequency file not found: {freq_path}. "
            "Run scripts/compute_subject_frequency.py first."
        )
    freq_payload = json.loads(freq_path.read_text())
    subjects = freq_payload["subjects"]

    metric = args.frequency_metric
    if metric is None:
        if freq_payload.get("_meta", {}).get("proxy") == "wordfreq":
            metric = "zipf"
        else:
            metric = "first_token_logprob"
    print(f"Frequency metric: {metric}")

    wanted_cats: set[str] | None = None
    if args.categories:
        wanted_cats = {c.strip().lower() for c in args.categories}

    subject_freq: dict[str, float] = {}
    missing_metric = 0
    for subj, info in subjects.items():
        if wanted_cats and info.get("category", "").lower() not in wanted_cats:
            continue
        if metric not in info:
            missing_metric += 1
            continue
        try:
            subject_freq[subj] = float(info[metric])
        except (TypeError, ValueError):
            missing_metric += 1
    if missing_metric:
        print(f"  warning: {missing_metric} subjects missing metric {metric!r}")
    print(f"Subjects in frequency table (after filter): {len(subject_freq)}")

    key_bin, bin_ranges = assign_percentile_bins(subject_freq, args.n_bins)

    per_sample_paths = sorted(Path().glob(args.per_sample_glob))
    if not per_sample_paths:
        raise SystemExit(
            f"No per-sample JSONLs match {args.per_sample_glob!r}."
        )

    per_model_records: dict[str, list[dict]] = {}
    per_model_meta: dict[str, dict] = {}
    for path in per_sample_paths:
        meta, records = _load_jsonl(path)
        size = meta.get("model_size") or path.stem.replace("per_sample_", "")
        if wanted_cats:
            records = [r for r in records if r.get("category", "").lower() in wanted_cats]
        per_model_records[size] = records
        per_model_meta[size] = meta
        print(f"  {path.name}: size={size}, records={len(records)}")

    if not any(per_model_records.values()):
        raise SystemExit("No per-sample records survived filtering.")

    # Auto-detect available conflict rows from the first non-empty model.
    sample_records = next(r for r in per_model_records.values() if r)
    detected = _detect_conflict_rows(sample_records)
    if args.rows:
        conflict_keys = list(args.rows)
    else:
        conflict_keys = detected
    print(f"Conflict rows: {conflict_keys}")

    if args.metric == "auto":
        metric_kind = _detect_metric(sample_records, conflict_keys)
    else:
        metric_kind = args.metric
    print(f"Classifier metric: {metric_kind}")

    row_titles = {
        "substitution_conflict": "Substitution Conflict",
        "coherent_conflict": "Coherent Conflict",
    }
    rows = [(k, row_titles.get(k, k)) for k in conflict_keys]

    # Interleave post-trained Qwen3 with the matching Qwen3-Base so that the
    # combined "all families" figure puts qwen3-0.6b next to qwen3-base-0.6b,
    # etc. -- the natural side-by-side comparison.
    qwen3_pairs: list[str] = []
    base_set = set(QWEN3_BASE_SIZES)
    for s in QWEN3_SIZES:
        qwen3_pairs.append(f"qwen3-{s}")
        if s in base_set:
            qwen3_pairs.append(f"qwen3-base-{s}")
    known_order: list[str] = list(PYTHIA_SIZES) + qwen3_pairs
    size_order = [s for s in known_order if s in per_model_records]
    for s in per_model_records:
        if s not in size_order:
            size_order.append(s)
    print(f"Model sizes in plot: {size_order}")

    summary = {
        "_meta": {
            "n_bins": args.n_bins,
            "frequency_metric": metric,
            "classifier_metric": metric_kind,
            "bin_by": args.bin_by,
            "categories": sorted(wanted_cats) if wanted_cats else None,
            "bin_ranges": bin_ranges,
            "rows": conflict_keys,
            "freq_proxy": freq_payload.get("_meta"),
        },
        "models": {},
    }
    for size in size_order:
        per = per_model_records[size]
        summary["models"][size] = {
            "n_samples_used": sum(1 for r in per if r.get(args.bin_by) in key_bin),
            "gen": {
                ck: aggregate(
                    per, key_bin, args.n_bins, ck, "gen", bin_field=args.bin_by
                )
                for ck in conflict_keys
            },
            "lp": {
                ck: aggregate(
                    per, key_bin, args.n_bins, ck, "lp", bin_field=args.bin_by
                )
                for ck in conflict_keys
            },
        }

    out_prefix = Path(args.output_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    summary_path = Path(out_prefix.as_posix() + "_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"Wrote {summary_path}")

    if wanted_cats:
        title_suffix = ", ".join(sorted(wanted_cats))
    else:
        title_suffix = "all categories"
    title_suffix += f" | bin by {args.bin_by} ({metric}, {metric_kind})"

    _save_matplotlib(
        summary, size_order, args.n_bins,
        Path(out_prefix.as_posix() + ".png"), title_suffix, rows,
        metric_kind=metric_kind,
    )
    _save_plotly(
        summary, size_order, args.n_bins,
        Path(out_prefix.as_posix() + ".html"), title_suffix, rows,
        metric_kind=metric_kind,
    )


if __name__ == "__main__":
    main()
