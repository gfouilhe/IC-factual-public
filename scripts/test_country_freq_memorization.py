"""Test whether P(memorized) rises with queried-country Zipf frequency.

Hypothesis (Yu et al. / pretraining-frequency intuition): models should be
more inclined to output the memorized gold capital as the *country* appears
more often in pretraining corpora.

For each model we report:

* Country-level Spearman ρ(Zipf, P(memorized)) over 218 queried countries
* Binned P(memorized) curves (paper Fig.~3 resolution) and ΔP(mem) across deciles
* Verdict from country ρ and endpoint ΔP(mem)

Usage::

    python scripts/test_country_freq_memorization.py \\
        --per-sample-glob 'experiments/exp21_context_memory_score/per_sample/per_sample_capitals_*.jsonl' \\
        --frequency-json experiments/exp00_baseline_capitals_sweep/summaries/capitals_subject_frequency.json \\
        --output-prefix experiments/exp24_context_memory_country_freq/plots/country_freq_mem_hypothesis \\
        --metric gen --n-bins 10
"""

from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_config_path = Path(__file__).resolve().parent.parent / "ic_factual" / "config.py"
_config_spec = importlib.util.spec_from_file_location("_ic_config", _config_path)
assert _config_spec and _config_spec.loader
_config = importlib.util.module_from_spec(_config_spec)
_config_spec.loader.exec_module(_config)
model_family = _config.model_family
model_param_billions = _config.model_param_billions


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


def assign_percentile_bins(
    values: dict[str, float], n_bins: int
) -> tuple[dict[str, int], list[tuple[float, float]]]:
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
    metric: str = "gen",
    bin_field: str = "subject",
) -> dict:
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


def aggregate_by_country(
    per_sample: list[dict],
    subject_freq: dict[str, float],
    conflict_key: str,
    metric: str = "gen",
    *,
    country_field: str = "subject",
) -> dict:
    """Per-country P(memorized) pooled over all distractors for that country."""
    mem_key = f"memorized_{metric}"
    counts: dict[str, int] = {}
    mem: dict[str, int] = {}
    for rec in per_sample:
        country = (rec.get(country_field) or rec.get("country") or "").strip()
        if not country or country not in subject_freq:
            continue
        block = rec.get(conflict_key, {})
        if not block:
            continue
        counts[country] = counts.get(country, 0) + 1
        if block.get(mem_key):
            mem[country] = mem.get(country, 0) + 1

    countries = sorted(counts)
    zipfs = [subject_freq[c] for c in countries]
    p_mem = [mem.get(c, 0) / counts[c] for c in countries]
    return {
        "n_countries": len(countries),
        "zipf": zipfs,
        "p_memorized": p_mem,
        "counts": [counts[c] for c in countries],
    }


def _collect_paths(patterns: list[str], exclude: str | None) -> list[Path]:
    paths: list[Path] = []
    for pat in patterns:
        paths.extend(Path(p) for p in glob.glob(pat))
    seen: set[str] = set()
    out: list[Path] = []
    for path in sorted(paths):
        key = str(path)
        if key in seen:
            continue
        if exclude and exclude in path.name:
            continue
        try:
            ok = path.is_file() or (path.is_symlink() and path.resolve().is_file())
        except OSError:
            ok = False
        if ok:
            seen.add(key)
            out.append(path)
    return out


def _label_from_path(path: Path, meta: dict) -> str:
    label = meta.get("label") or meta.get("model_size")
    if label:
        return str(label)
    m = re.search(r"per_sample_capitals_(.+)\.jsonl$", path.name)
    if m:
        return m.group(1)
    return path.stem


def spearman_rho(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 3:
        return float("nan")
    rx = np.argsort(np.argsort(x, kind="mergesort"), kind="mergesort").astype(float)
    ry = np.argsort(np.argsort(y, kind="mergesort"), kind="mergesort").astype(float)
    if np.std(rx) == 0 or np.std(ry) == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def classify_hypothesis(rho_country: float, delta_p_mem: float) -> str:
    """Verdict from country-level Spearman ρ and binned endpoint ΔP(mem)."""
    if not (math.isfinite(rho_country) and math.isfinite(delta_p_mem)):
        return "insufficient_data"
    if rho_country >= 0.3 and delta_p_mem >= 0.05:
        return "supports"
    if rho_country > 0.0 and delta_p_mem > 0.0:
        return "weak_support"
    if rho_country <= -0.3 and delta_p_mem <= -0.05:
        return "inverts"
    if rho_country < 0.0 and delta_p_mem < 0.0:
        return "weak_inverts"
    return "flat"


_VERDICT_COLORS = {
    "supports": "#2166ac",
    "weak_support": "#92c5de",
    "flat": "#bdbdbd",
    "weak_inverts": "#f4a582",
    "inverts": "#b2182b",
    "insufficient_data": "#969696",
}


def _display_label(label: str) -> str:
    if label == "gpt2":
        return "GPT-2 small"
    if label.startswith("gpt2-"):
        return f"GPT-2 {label.removeprefix('gpt2-')}"
    if label.startswith("pythia-"):
        return f"Pythia-{label.removeprefix('pythia-')}"
    if label.startswith("qwen3-base-"):
        return f"Qwen3-Base-{label.removeprefix('qwen3-base-')}"
    if label.startswith("qwen3-"):
        return f"Qwen3-{label.removeprefix('qwen3-')}"
    if label.startswith("ministral3-"):
        variant, size = label.removeprefix("ministral3-").split("-", 1)
        return f"Ministral-3 {variant.capitalize()}-{size}"
    return label


def _sort_models(models: list[dict]) -> list[dict]:
    """Shared row order: ascending ΔP(mem) so panels align on one y-axis."""
    return sorted(models, key=lambda m: m["delta_p_memorized"])


def _plot_combined(
    models: list[dict],
    n_bins: int,
    out_path: Path,
    *,
    title: str,
) -> None:
    ordered = _sort_models(models)
    labels = [_display_label(m["label"]) for m in ordered]
    y = np.arange(len(labels))
    mat = np.array([m["p_memorized_by_bin"] for m in ordered], dtype=float)
    deltas = [m["delta_p_memorized"] for m in ordered]
    colors = [_VERDICT_COLORS.get(m["verdict"], "#969696") for m in ordered]

    fig_h = max(7.0, 0.24 * len(labels) + 1.8)
    fig, (ax_h, ax_d) = plt.subplots(
        1,
        2,
        figsize=(12.0, fig_h),
        sharey=True,
        gridspec_kw={"width_ratios": [1.35, 0.95], "wspace": 0.32},
    )

    vmax = max(0.8, float(np.nanmax(mat)))
    im = ax_h.imshow(
        mat,
        aspect="auto",
        origin="lower",
        cmap="YlOrRd",
        vmin=0.0,
        vmax=vmax,
        extent=(-0.5, n_bins - 0.5, -0.5, len(labels) - 0.5),
    )
    ax_h.set_xticks(np.arange(n_bins))
    ax_h.set_xticklabels(
        [f"Q{int(round(100 * (i + 1) / n_bins))}" for i in range(n_bins)],
        fontsize=9,
    )
    ax_h.set_xlabel("Country Zipf bin (low → high)")
    ax_h.set_yticks(y)
    ax_h.set_yticklabels(labels, fontsize=7)
    ax_h.set_title("$P(\\mathrm{memorized})$ by bin", fontsize=10)

    cbar = fig.colorbar(im, ax=ax_h, fraction=0.042, pad=0.03)
    cbar.set_label("$P(\\mathrm{memorized})$", fontsize=9)

    ax_d.barh(y, deltas, color=colors, edgecolor="white", linewidth=0.4, height=0.82)
    ax_d.axvline(0.0, color="black", linewidth=0.8, linestyle="--", alpha=0.6)
    q_lo = int(round(100 / n_bins))
    ax_d.set_xlabel(f"$\\Delta P(\\mathrm{{mem}})$ (Q100 $-$ Q{q_lo})")
    ax_d.set_title("High $-$ low bin", fontsize=10)
    ax_d.tick_params(axis="y", labelleft=False)
    ax_d.grid(True, axis="x", alpha=0.25)

    fig.suptitle(title, fontsize=11, y=1.01)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def _plot_delta_bar(models: list[dict], out_path: Path, *, title: str) -> None:
    ordered = _sort_models(models)
    labels = [_display_label(m["label"]) for m in ordered]
    deltas = [m["delta_p_memorized"] for m in ordered]
    colors = [_VERDICT_COLORS.get(m["verdict"], "#969696") for m in ordered]

    fig_h = max(6.0, 0.22 * len(labels) + 1.5)
    fig, ax = plt.subplots(figsize=(9, fig_h))
    y = np.arange(len(labels))
    ax.barh(y, deltas, color=colors, edgecolor="white", linewidth=0.4)
    ax.axvline(0.0, color="black", linewidth=0.8, linestyle="--", alpha=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("ΔP(memorized): highest − lowest country-frequency bin")
    ax.set_title(title, fontsize=11)
    ax.grid(True, axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def _plot_heatmap(models: list[dict], n_bins: int, out_path: Path, *, title: str) -> None:
    ordered = _sort_models(models)
    mat = np.array([m["p_memorized_by_bin"] for m in ordered], dtype=float)
    labels = [_display_label(m["label"]) for m in ordered]

    fig_h = max(6.0, 0.22 * len(labels) + 1.8)
    fig, ax = plt.subplots(figsize=(8.5, fig_h))
    y = np.arange(len(labels))
    im = ax.imshow(
        mat,
        aspect="auto",
        origin="lower",
        cmap="YlOrRd",
        vmin=0.0,
        vmax=max(0.8, np.nanmax(mat)),
        extent=(-0.5, n_bins - 0.5, -0.5, len(labels) - 0.5),
    )
    ax.set_xticks(np.arange(n_bins))
    ax.set_xticklabels(
        [f"Q{int(round(100 * (i + 1) / n_bins))}" for i in range(n_bins)],
        fontsize=9,
    )
    ax.set_xlabel("Country Zipf percentile bin (low → high)")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_title(title, fontsize=11)
    cbar = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    cbar.set_label("P(memorized)", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test memorization vs queried-country Zipf frequency per model."
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=None,
        help="Precomputed summary JSON to replot directly.",
    )
    parser.add_argument(
        "--per-sample-glob",
        nargs="+",
        default=None,
        help="Glob(s) for per_sample_capitals_*.jsonl files.",
    )
    parser.add_argument(
        "--frequency-json",
        type=Path,
        default=None,
        help="capitals_subject_frequency.json (wordfreq Zipf per country).",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        required=True,
        help="Prefix for .png outputs and _summary.json.",
    )
    parser.add_argument(
        "--metric",
        choices=("gen", "lp"),
        default="gen",
        help="Classifier column prefix (default: gen).",
    )
    parser.add_argument(
        "--n-bins",
        type=int,
        default=10,
        help="Equal-count percentile bins over countries (default: 10, paper Fig. 3).",
    )
    parser.add_argument(
        "--exclude-pattern",
        default="_smoke",
        help="Skip paths whose filename contains this substring.",
    )
    args = parser.parse_args()

    if args.summary_json:
        summary = json.loads(args.summary_json.read_text(encoding="utf-8"))
        models_out = summary["models"]
        n_bins = summary.get("n_bins", args.n_bins)
        metric = summary.get("metric", args.metric)
        args.n_bins = n_bins
        args.metric = metric
        args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    else:
        if not args.per_sample_glob or not args.frequency_json:
            raise SystemExit(
                "Must specify --summary-json or both --per-sample-glob and --frequency-json"
            )
        freq_payload = json.loads(args.frequency_json.read_text(encoding="utf-8"))
        metric_name = "zipf"
        if freq_payload.get("_meta", {}).get("proxy") != "wordfreq":
            metric_name = "first_token_logprob"

        subject_freq: dict[str, float] = {}
        for subj, info in freq_payload["subjects"].items():
            if metric_name not in info:
                continue
            subject_freq[subj] = float(info[metric_name])

        key_bin, bin_ranges = assign_percentile_bins(subject_freq, args.n_bins)
        paths = _collect_paths(args.per_sample_glob, args.exclude_pattern)
        if not paths:
            raise SystemExit(f"No files match {args.per_sample_glob!r}")

        models_out: list[dict] = []
        print(
            f"Country bins: n={args.n_bins}, metric={metric_name}, "
            f"classifier={args.metric}"
        )
        print(
            f"{'model':<28} {'ρ_ctry':>7} {'ρ_bin':>7} {'ΔP(mem)':>8}  verdict"
        )

        for path in paths:
            meta, records = _load_jsonl(path)
            label = _label_from_path(path, meta)
            country_data = aggregate_by_country(
                records,
                subject_freq,
                "substitution_conflict",
                metric=args.metric,
            )
            zipfs = np.array(country_data["zipf"], dtype=float)
            p_mem_country = np.array(country_data["p_memorized"], dtype=float)
            rho_country = spearman_rho(zipfs, p_mem_country)

            data = aggregate(
                records,
                key_bin,
                args.n_bins,
                "substitution_conflict",
                metric=args.metric,
                bin_field="subject",
            )
            p_mem = np.array(data["p_memorized"], dtype=float)
            bins = np.arange(args.n_bins, dtype=float)
            rho_bins = spearman_rho(bins, p_mem)
            delta = float(p_mem[-1] - p_mem[0]) if len(p_mem) >= 2 else float("nan")
            mono_steps = int(np.sum(np.diff(p_mem) >= 0)) if len(p_mem) >= 2 else 0
            strict_mono = mono_steps == args.n_bins - 1
            verdict = classify_hypothesis(rho_country, delta)
            row = {
                "label": label,
                "family": model_family(label),
                "params_b": model_param_billions(label),
                "n_bins": args.n_bins,
                "n_countries": country_data["n_countries"],
                "p_memorized_by_bin": p_mem.tolist(),
                "p_in_context_by_bin": data["p_in_context"],
                "counts_by_bin": data["counts"],
                "spearman_rho": rho_country,
                "spearman_rho_bins": rho_bins,
                "delta_p_memorized": delta,
                "strict_monotone_increasing": strict_mono,
                "monotone_steps": mono_steps,
                "verdict": verdict,
                "path": str(path),
            }
            models_out.append(row)
            print(
                f"{label:<28} {rho_country:+7.3f} {rho_bins:+7.3f} {delta:+8.3f}  {verdict}"
            )

        if args.output_prefix.parent.name == "plots":
            summaries_dir = args.output_prefix.parent.parent / "summaries"
        else:
            summaries_dir = args.output_prefix.parent
        summaries_dir.mkdir(parents=True, exist_ok=True)
        args.output_prefix.parent.mkdir(parents=True, exist_ok=True)

        summary = {
            "hypothesis": (
                "P(memorized) increases with queried-country pretraining frequency (Zipf)"
            ),
            "metric": args.metric,
            "frequency_metric": metric_name,
            "n_bins": args.n_bins,
            "bin_ranges": bin_ranges,
            "verdict_legend": {
                "supports": "country Spearman rho >= 0.3 and ΔP(mem) >= 0.05",
                "weak_support": "country rho > 0 and ΔP(mem) > 0 (below strong thresholds)",
                "flat": "otherwise near zero trend",
                "weak_inverts": "country rho < 0 and ΔP(mem) < 0 (below strong thresholds)",
                "inverts": "country rho <= -0.3 and ΔP(mem) <= -0.05",
            },
            "correlation_method": "country_level_spearman",
            "models": sorted(
                models_out,
                key=lambda m: (m["family"], m["params_b"] or 0),
            ),
            "counts_by_verdict": {},
        }
        for v in ("supports", "weak_support", "flat", "weak_inverts", "inverts", "insufficient_data"):
            summary["counts_by_verdict"][v] = sum(1 for m in models_out if m["verdict"] == v)

        summary_path = summaries_dir / f"{args.output_prefix.name}_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(f"\nWrote {summary_path}")
        print("Verdict counts:", summary["counts_by_verdict"])

    title = (
        f"P(memorized) vs country Zipf ({args.n_bins} bins, {args.metric} classifier)"
    )
    _plot_combined(
        models_out,
        args.n_bins,
        args.output_prefix.parent / f"{args.output_prefix.name}_combined.png",
        title=title,
    )
    _plot_delta_bar(
        models_out,
        args.output_prefix.parent / f"{args.output_prefix.name}_delta.png",
        title=f"ΔP(memorized) high−low country Zipf — {title}",
    )
    _plot_heatmap(
        models_out,
        args.n_bins,
        args.output_prefix.parent / f"{args.output_prefix.name}_heatmap.png",
        title=title,
    )


if __name__ == "__main__":
    main()
