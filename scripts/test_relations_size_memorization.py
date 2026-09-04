"""Test whether P(memorized) rises with model size across ParaConflict relations.

For each relation category, pools 31 checkpoints and reports Spearman rho between
parameter count and generative P(memorized), with two-sided p-values from
``scipy.stats.spearmanr``.  Also tests within-family series and meta-analytic
combinations across categories.

Usage::

    python scripts/test_relations_size_memorization.py \\
        --summary-json experiments/exp23_context_memory_relations/summaries/context_memory_score_gen_all_summary.json \\
        --output-prefix experiments/exp23_context_memory_relations/summaries/relations_size_mem_test
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

_CATEGORIES: tuple[str, ...] = (
    "World Capital",
    "Athlete Sport",
    "Book Author",
    "Company Headquarter",
    "Company Founder",
    "Official Language",
)


def _format_p(p: float) -> str:
    if not math.isfinite(p):
        return "---"
    if p < 0.001:
        return f"{p:.1e}"
    return f"{p:.3f}"


def _format_p_tex(p: float) -> str:
    if not math.isfinite(p):
        return "---"
    if p < 0.001:
        return r"$<0.001$"
    return f"{p:.3f}"


def _load_summary(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _rows_by_category(summary: dict) -> dict[str, list[dict]]:
    by_cat: dict[str, list[dict]] = defaultdict(list)
    for row in summary["models"]:
        if row.get("params_b") is None:
            continue
        by_cat[row["category"]].append(row)
    return by_cat


def _category_stats(rows: list[dict]) -> dict:
    ordered = sorted(rows, key=lambda r: r["params_b"])
    xs = [r["params_b"] for r in ordered]
    mem = [r["p_memorized"] for r in ordered]
    rho, p = stats.spearmanr(xs, mem)
    sm, lg = ordered[0], ordered[-1]
    return {
        "n_models": len(ordered),
        "n_prompts": ordered[0]["n"],
        "mean_p_memorized": float(np.mean(mem)),
        "spearman_rho": float(rho),
        "p_value": float(p),
        "delta_pp": 100.0 * (lg["p_memorized"] - sm["p_memorized"]),
        "smallest_label": sm["label"],
        "largest_label": lg["label"],
    }


def _within_family_rhos(summary: dict) -> list[dict]:
    by_fc: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in summary["models"]:
        if row.get("params_b") is None:
            continue
        by_fc[(row["family"], row["category"])].append(row)

    out: list[dict] = []
    for (family, category), pts in sorted(by_fc.items()):
        ordered = sorted(pts, key=lambda r: r["params_b"])
        if len(ordered) < 3:
            continue
        xs = [r["params_b"] for r in ordered]
        mem = [r["p_memorized"] for r in ordered]
        rho, p = stats.spearmanr(xs, mem)
        if not math.isfinite(rho):
            continue
        out.append(
            {
                "family": family,
                "category": category,
                "n_checkpoints": len(ordered),
                "spearman_rho": float(rho),
                "p_value": float(p),
            }
        )
    return out


def _export_tex(path: Path, category_rows: list[dict], pooled: dict, meta: dict) -> None:
    lines = [
        r"\begin{table}[t]",
        r"  \centering",
        r"  \small",
        r"  \setlength{\tabcolsep}{4pt}",
        r"  \caption{Generative $P(\text{memorized})$ vs.\ model size across ParaConflict",
        r"    \texttt{Substitution Conflict} prompts and six relation categories",
        r"    (Figure~\ref{fig:exp23-relations-context-mem-all}).",
        r"    For each category we pool 31 checkpoints (Pythia, GPT-2, Qwen3 Base/post-trained,",
        r"    Ministral-3) and report the mean memorization rate, Spearman $\rho$ between",
        r"    parameter count and $P(\text{memorized})$, two-sided $p$-values from",
        r"    \texttt{scipy.stats.spearmanr}, and the end-point gap $\Delta$ between the",
        r"    smallest- and largest-parameter models in the pool (percentage points).",
        r"    World Capital uses the full capitals cross-product ($n{\approx}47{,}300$);",
        r"    other rows use the ParaConflict test split.}",
        r"  \label{tab:relations-size-mem}",
        r"  \begin{tabular}{@{}lrrrrr@{}}",
        r"    \toprule",
        r"    Category & $n$ & $\bar{P}(\text{mem})$ & $\rho$ & $p$ & $\Delta$\,pp \\",
        r"    \midrule",
    ]
    for row in category_rows:
        n_prompts = f"{row['n_prompts']:,}"
        lines.append(
            f"    {row['category']:<20} & {n_prompts:>7} & "
            f"{row['mean_p_memorized']:.2f} & {row['spearman_rho']:+.2f} & "
            f"{_format_p_tex(row['p_value'])} & {row['delta_pp']:+.1f} \\\\"
        )
    lines.extend(
        [
            r"    \midrule",
            f"    All (pooled)         &         & {pooled['mean_p_memorized']:.2f} & "
            f"{pooled['spearman_rho']:+.2f} & {_format_p_tex(pooled['p_value'])} &      \\\\",
            r"    \bottomrule",
            r"  \end{tabular}",
            r"\end{table}",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Spearman tests: model size vs P(memorized) on ParaConflict relations."
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        required=True,
        help="context_memory_score_gen_all_summary.json from plot_context_memory_score.py",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        required=True,
        help="Writes <prefix>_summary.json and exports LaTeX if --export-tex is set.",
    )
    parser.add_argument(
        "--export-tex",
        type=Path,
        default=None,
        help="Optional path for ACL table snippet (e.g. paper/tables/relations_size_mem_correlation.tex).",
    )
    args = parser.parse_args()

    summary = _load_summary(args.summary_json)
    categories = summary.get("categories", list(_CATEGORIES))
    by_cat = _rows_by_category(summary)

    category_rows: list[dict] = []
    category_p_values: list[float] = []
    print("Per-category Spearman rho(params_b, P(memorized)), n=31 models")
    print(f"{'Category':<22} {'rho':>7} {'p':>12}")
    for category in categories:
        row = _category_stats(by_cat[category])
        row["category"] = category
        category_rows.append(row)
        category_p_values.append(row["p_value"])
        print(
            f"{category:<22} {row['spearman_rho']:>+7.3f} {_format_p(row['p_value']):>12}"
        )

    all_rows = [r for rows in by_cat.values() for r in rows]
    pooled_xs = [r["params_b"] for r in all_rows]
    pooled_mem = [r["p_memorized"] for r in all_rows]
    pooled_rho, pooled_p = stats.spearmanr(pooled_xs, pooled_mem)
    pooled = {
        "n_cells": len(all_rows),
        "mean_p_memorized": float(np.mean(pooled_mem)),
        "spearman_rho": float(pooled_rho),
        "p_value": float(pooled_p),
        "note": "Pooled cells are not independent (same models repeat across categories).",
    }
    print(
        f"\nPooled n={pooled['n_cells']}: rho={pooled['spearman_rho']:+.3f} "
        f"p={_format_p(pooled['p_value'])}"
    )

    fisher_stat, fisher_p = stats.combine_pvalues(category_p_values, method="fisher")
    stouffer_stat, stouffer_p = stats.combine_pvalues(
        category_p_values, method="stouffer"
    )
    print(f"Fisher combine (6 categories): p={_format_p(fisher_p)}")
    print(f"Stouffer combine:            p={_format_p(stouffer_p)}")

    within = _within_family_rhos(summary)
    rhos = [r["spearman_rho"] for r in within]
    t_stat, t_p = stats.ttest_1samp(rhos, 0.0, alternative="greater")
    n_pos = sum(1 for r in rhos if r > 0)
    sign_p = stats.binomtest(n_pos, len(rhos), 0.5, alternative="greater").pvalue
    print(
        f"\nWithin-family series: n={len(within)} mean rho={np.mean(rhos):+.3f}"
    )
    print(f"  one-sided t-test (mean rho > 0): t={t_stat:.3f} p={_format_p(t_p)}")
    print(f"  sign test ({n_pos}/{len(rhos)} positive): p={_format_p(sign_p)}")

    out = {
        "summary_json": str(args.summary_json),
        "categories": category_rows,
        "pooled": pooled,
        "meta_analysis": {
            "fisher_p": float(fisher_p),
            "stouffer_p": float(stouffer_p),
        },
        "within_family": {
            "n_series": len(within),
            "mean_spearman_rho": float(np.mean(rhos)),
            "fraction_positive": float(n_pos / len(rhos)),
            "one_sided_ttest_p": float(t_p),
            "sign_test_p": float(sign_p),
            "series": within,
        },
    }

    out_path = Path(f"{args.output_prefix}_summary.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote {out_path}")

    if args.export_tex is not None:
        meta = {"fisher_p": fisher_p, "stouffer_p": stouffer_p}
        _export_tex(args.export_tex, category_rows, pooled, meta)
        print(f"Wrote {args.export_tex}")


if __name__ == "__main__":
    main()
