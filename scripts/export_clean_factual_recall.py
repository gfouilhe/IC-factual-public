"""Export baseline factual recall without conflict across all models and relations.

Aggregates the clean prompt evaluations (zero-shot factual recall without any
counterfactual context or conflict) across all 31 models and all six
ParaConflict relation categories.

Usage::

    python3.11 scripts/export_clean_factual_recall.py
    python3.11 scripts/export_clean_factual_recall.py --metric both
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

# Load config directly without triggering ic_factual.__init__ (which requires torch)
_config_path = Path(__file__).resolve().parent.parent / "ic_factual" / "config.py"
_spec = importlib.util.spec_from_file_location("_ic_factual_config", _config_path)
assert _spec and _spec.loader
_config_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_config_mod)
model_family = _config_mod.model_family
model_param_billions = _config_mod.model_param_billions

_PYTHIA_SIZES: tuple[str, ...] = ("160m", "410m", "1b", "1.4b", "2.8b", "6.9b", "12b")
_EXPECTED_N = 2146

_CATEGORIES: tuple[str, ...] = (
    "World Capital",
    "Athlete Sport",
    "Book Author",
    "Company Headquarter",
    "Company Founder",
    "Official Language",
)

_CATEGORY_COL_NAMES: dict[str, str] = {
    "World Capital": "World Capital",
    "Athlete Sport": "Athlete Sport",
    "Book Author": "Book Author",
    "Company Headquarter": "Company HQ",
    "Company Founder": "Company Founder",
    "Official Language": "Official Lang.",
}

_FAMILY_ORDER: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Pythia", _PYTHIA_SIZES),
    ("GPT-2", ("gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl")),
    (
        "Qwen3",
        ("qwen3-0.6b", "qwen3-1.7b", "qwen3-4b", "qwen3-8b", "qwen3-14b", "qwen3-32b"),
    ),
    (
        "Qwen3-Base",
        (
            "qwen3-base-0.6b",
            "qwen3-base-1.7b",
            "qwen3-base-4b",
            "qwen3-base-8b",
            "qwen3-base-14b",
        ),
    ),
    (
        "Ministral-3 Instruct",
        ("ministral3-instruct-3b", "ministral3-instruct-8b", "ministral3-instruct-14b"),
    ),
    (
        "Ministral-3 Reasoning",
        ("ministral3-reasoning-3b", "ministral3-reasoning-8b", "ministral3-reasoning-14b"),
    ),
    (
        "Ministral-3 Base",
        ("ministral3-base-3b", "ministral3-base-8b", "ministral3-base-14b"),
    ),
)


def _canonical_category(cat: str) -> str:
    cat = (cat or "").strip()
    if cat.lower() in ("athlete sport", "athelete sport"):
        return "Athlete Sport"
    return cat


def _display_label(label: str) -> str:
    if label in _PYTHIA_SIZES:
        return f"Pythia-{label}"
    if label == "gpt2":
        return "GPT-2 small"
    if label.startswith("gpt2-"):
        return "GPT-2 " + label.removeprefix("gpt2-")
    if label.startswith("qwen3-base-"):
        return "Qwen3-Base-" + label.removeprefix("qwen3-base-").upper()
    if label.startswith("qwen3-"):
        return "Qwen3-" + label.removeprefix("qwen3-").upper()
    if label.startswith("ministral3-"):
        rest = label.removeprefix("ministral3-")
        variant, size = rest.rsplit("-", 1)
        return f"Ministral-3 {variant.title()} {size.upper()}"
    return label


def _load_model_records(
    label: str,
    exp25_native: Path,
    exp23_relations: Path,
    exp25_wc: Path,
    exp25_native_other: Path,
) -> list[dict]:
    """Load all records for a model across all 6 categories."""
    if label in _PYTHIA_SIZES:
        path = exp25_native / f"per_sample_{label}.jsonl"
        if path.is_file() and path.stat().st_size > 0:
            return _read_jsonl_records(path)

    full_other = exp25_native_other / f"per_sample_{label}.jsonl"
    if full_other.is_file() and full_other.stat().st_size > 0:
        return _read_jsonl_records(full_other)

    rel_path = exp23_relations / f"per_sample_relations_{label}.jsonl"
    wc_path = exp25_wc / f"per_sample_{label}.jsonl"
    records: list[dict] = []
    if rel_path.is_file() and rel_path.stat().st_size > 0:
        records.extend(_read_jsonl_records(rel_path))
    if wc_path.is_file() and wc_path.stat().st_size > 0:
        records.extend(_read_jsonl_records(wc_path))
    return records


def _read_jsonl_records(path: Path) -> list[dict]:
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("_meta"):
                continue
            records.append(rec)
    return records


def collect_clean_stats(
    exp25_native: Path,
    exp23_relations: Path,
    exp25_wc: Path,
    exp25_native_other: Path,
) -> dict[str, dict]:
    """Aggregate clean prompt accuracy across all models and categories."""
    model_stats: dict[str, dict] = {}

    for family_name, labels in _FAMILY_ORDER:
        for label in labels:
            records = _load_model_records(
                label, exp25_native, exp23_relations, exp25_wc, exp25_native_other
            )
            cat_groups: dict[str, list[dict]] = defaultdict(list)
            for r in records:
                cat = _canonical_category(r.get("category", ""))
                cat_groups[cat].append(r)

            cat_stats: dict[str, dict] = {}
            tot_n = tot_gen = tot_lp = 0

            for cat in _CATEGORIES:
                recs = cat_groups.get(cat, [])
                n = len(recs)
                gen_c = sum(bool(r.get("clean_prompt", {}).get("correct_gen")) for r in recs)
                lp_c = sum(bool(r.get("clean_prompt", {}).get("answer_beats_distractor")) for r in recs)

                cat_stats[cat] = {
                    "n": n,
                    "gen_correct": gen_c,
                    "gen_acc": (gen_c / n) if n > 0 else float("nan"),
                    "lp_correct": lp_c,
                    "lp_acc": (lp_c / n) if n > 0 else float("nan"),
                }
                tot_n += n
                tot_gen += gen_c
                tot_lp += lp_c

            overall = {
                "n": tot_n,
                "gen_correct": tot_gen,
                "gen_acc": (tot_gen / tot_n) if tot_n > 0 else float("nan"),
                "lp_correct": tot_lp,
                "lp_acc": (tot_lp / tot_n) if tot_n > 0 else float("nan"),
            }

            model_stats[label] = {
                "label": label,
                "display_label": _display_label(label),
                "family": model_family(label),
                "params_b": model_param_billions(label),
                "categories": cat_stats,
                "overall": overall,
            }

    return model_stats


def build_latex_table(stats: dict[str, dict], metric: str = "gen") -> str:
    """Generate LaTeX table string for paper/tables/clean_factual_recall.tex."""
    metric_key = "gen_acc" if metric == "gen" else "lp_acc"
    metric_name = "generative substring match" if metric == "gen" else "log-probability argmax"

    lines: list[str] = [
        "% Auto-generated by scripts/export_clean_factual_recall.py",
        r"\begin{table*}[t]",
        r"  \centering",
        r"  \small",
        r"  \setlength{\tabcolsep}{4.5pt}",
        f"  \\caption{{Baseline factual recall accuracy without conflict ($P(\\text{{correct}} \\mid \\text{{clean prompt}})$; {metric_name}; six ParaConflict categories; $n{{=}}2146$ per model). Overall column reports accuracy pooled across all categories.}}",
        r"  \label{tab:clean-factual-recall}",
        r"  \begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}l rrrrrr r@{}}",
        r"    \toprule",
        r"    Model & World Capital & Athlete Sport & Book Author & Co.\ HQ & Co.\ Founder & Official Lang. & Overall \\",
        r"    \midrule",
    ]

    for family_name, labels in _FAMILY_ORDER:
        lines.append(f"    \\multicolumn{{8}}{{l}}{{\\textbf{{{family_name}}}}} \\\\")
        for label in labels:
            m = stats.get(label)
            if not m:
                continue
            disp = m["display_label"]
            cols = []
            for cat in _CATEGORIES:
                val = m["categories"][cat][metric_key]
                if math.isnan(val):
                    cols.append("---")
                else:
                    cols.append(f"{val:.2f}")
            ov = m["overall"][metric_key]
            cols.append(f"\\textbf{{{ov:.2f}}}")
            lines.append(f"    {disp} & " + " & ".join(cols) + r" \\")
        lines.append(r"    \addlinespace")

    # Remove trailing \addlinespace if present
    if lines[-1] == r"    \addlinespace":
        lines.pop()

    lines.extend([
        r"    \bottomrule",
        r"  \end{tabular*}",
        r"\end{table*}",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export baseline clean prompt factual recall to JSON and LaTeX."
    )
    parser.add_argument(
        "--exp25-native-dir",
        type=Path,
        default=Path("experiments/exp25_coherent_motivates_sweep/per_sample_native"),
    )
    parser.add_argument(
        "--exp23-relations-dir",
        type=Path,
        default=Path("experiments/exp23_context_memory_relations/per_sample"),
    )
    parser.add_argument(
        "--exp25-wc-dir",
        type=Path,
        default=Path("experiments/exp25_coherent_motivates_sweep/per_sample_native_wc"),
    )
    parser.add_argument(
        "--exp25-native-other-dir",
        type=Path,
        default=Path("experiments/exp25_coherent_motivates_sweep/per_sample_native_other"),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path(
            "experiments/exp23_context_memory_relations/summaries/clean_factual_recall_summary.json"
        ),
    )
    parser.add_argument(
        "--output-tex",
        type=Path,
        default=Path("paper/tables/clean_factual_recall.tex"),
    )
    parser.add_argument(
        "--metric",
        choices=("gen", "lp", "both"),
        default="gen",
        help="Metric for LaTeX table ('gen' = generative substring match, 'lp' = logprob argmax).",
    )
    parser.add_argument(
        "--require-full",
        action="store_true",
        help="Exit non-zero if any model has fewer than 2146 samples.",
    )
    args = parser.parse_args()

    stats = collect_clean_stats(
        args.exp25_native_dir,
        args.exp23_relations_dir,
        args.exp25_wc_dir,
        args.exp25_native_other_dir,
    )

    incomplete: list[str] = []
    for label, info in stats.items():
        if info["overall"]["n"] != _EXPECTED_N:
            incomplete.append(f"{label} (n={info['overall']['n']})")

    if incomplete:
        msg = f"Warning: {len(incomplete)} models have incomplete data: {', '.join(incomplete[:5])}"
        if args.require_full:
            print(f"ERROR: {msg}", file=sys.stderr)
            sys.exit(1)
        else:
            print(msg, file=sys.stderr)
    else:
        print(f"All {len(stats)} models verified with complete n={_EXPECTED_N} samples.")

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    summary_data = {
        "_meta": {
            "description": "Baseline factual recall without conflict across all 31 models and 6 relations.",
            "categories": list(_CATEGORIES),
            "n_expected_per_model": _EXPECTED_N,
        },
        "models": stats,
    }
    args.output_json.write_text(json.dumps(summary_data, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote summary JSON to {args.output_json}")

    tex_metric = "gen" if args.metric in ("gen", "both") else "lp"
    tex_content = build_latex_table(stats, metric=tex_metric)
    args.output_tex.parent.mkdir(parents=True, exist_ok=True)
    args.output_tex.write_text(tex_content, encoding="utf-8")
    print(f"Wrote LaTeX table to {args.output_tex}")

    if args.metric == "both":
        lp_tex_path = args.output_tex.parent / f"{args.output_tex.stem}_lp.tex"
        lp_tex_content = build_latex_table(stats, metric="lp")
        lp_tex_path.write_text(lp_tex_content, encoding="utf-8")
        print(f"Wrote logprob LaTeX table to {lp_tex_path}")


if __name__ == "__main__":
    main()
