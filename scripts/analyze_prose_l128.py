#!/usr/bin/env python3
"""Aggregate and visualize cross-family L=128 unrelated prose effect for exp27."""

import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

FAMILY_ORDER = [
    "Pythia",
    "GPT-2",
    "Qwen3-Base",
    "Qwen3 (post-trained)",
    "Ministral-3 Base",
    "Ministral-3 Instruct",
    "Ministral-3 Reasoning",
]

# Exact palette matching Figure 1 of the paper (scripts/plot_context_memory_score.py)
FAMILY_COLORS = {
    "Pythia": "#2171b5",
    "GPT-2": "#e377c2",
    "Qwen3-Base": "#98df8a",
    "Qwen3 (post-trained)": "#2ca02c",
    "Ministral-3 Base": "#6ec4bc",
    "Ministral-3 Instruct": "#00897b",
    "Ministral-3 Reasoning": "#004d40",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze cross-family L=128 prose results.")
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=Path("summaries/exp27_prose_l128_summary.json"),
        help="Path to precomputed summary JSON.",
    )
    parser.add_argument(
        "--exp27-dir",
        type=Path,
        default=Path("results/exp27"),
        help="Path to exp27 directory for recomputing.",
    )
    parser.add_argument(
        "--exp26-dir",
        type=Path,
        default=Path("results/exp26"),
        help="Path to exp26 directory for L=0 baseline.",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("summaries"),
        help="Prefix for output summaries.",
    )
    return parser.parse_args()


def classify_model(model_name: str) -> tuple[str, str, float]:
    """Return (family, display_name, param_count_billions)."""
    m = model_name.lower()
    if m.startswith("pythia-"):
        tag = m.replace("pythia-", "")
        scale_map = {
            "160m": 0.16, "410m": 0.41, "1b": 1.0, "1.4b": 1.4,
            "2.8b": 2.8, "6.9b": 6.9, "12b": 12.0
        }
        return "Pythia", f"Pythia-{tag}", scale_map.get(tag, 1.0)
    elif m.startswith("gpt2"):
        scale_map = {
            "gpt2": 0.124, "gpt2-medium": 0.355, "gpt2-large": 0.774, "gpt2-xl": 1.5
        }
        name_map = {
            "gpt2": "GPT-2 small", "gpt2-medium": "GPT-2 medium",
            "gpt2-large": "GPT-2 large", "gpt2-xl": "GPT-2 XL"
        }
        return "GPT-2", name_map.get(m, m), scale_map.get(m, 0.5)
    elif "qwen3-base" in m:
        tag = m.replace("qwen3-base-", "")
        val = float(tag.replace("b", ""))
        return "Qwen3-Base", f"Qwen3-Base-{tag.upper()}", val
    elif "qwen3" in m:
        tag = m.replace("qwen3-", "")
        val = float(tag.replace("b", ""))
        return "Qwen3 (post-trained)", f"Qwen3-{tag.upper()}", val
    elif "ministral3" in m:
        parts = m.split("-")
        variant = parts[1].capitalize()  # Base, Instruct, Reasoning
        size = parts[2].upper()          # 3B, 8B, 14B
        val = float(size.replace("B", ""))
        return f"Ministral-3 {variant}", f"Ministral-3 {variant} {size}", val
    else:
        return "Other", m, 1.0


def read_jsonl_counts(file_path: Path) -> dict:
    n = 0
    n_ctx_lp = 0
    n_mem_lp = 0
    n_ctx_gen = 0
    n_mem_gen = 0
    has_gen = False

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("_meta"):
                continue
            n += 1
            block = rec.get("substitution_conflict", {})
            if block.get("in_context_lp"):
                n_ctx_lp += 1
            if block.get("memorized_lp"):
                n_mem_lp += 1
            if "in_context_gen" in block:
                has_gen = True
                if block.get("in_context_gen"):
                    n_ctx_gen += 1
                if block.get("memorized_gen"):
                    n_mem_gen += 1

    return {
        "n": n,
        "p_ctx_lp": n_ctx_lp / n if n > 0 else 0.0,
        "p_mem_lp": n_mem_lp / n if n > 0 else 0.0,
        "p_ctx_gen": n_ctx_gen / n if n > 0 and has_gen else None,
        "p_mem_gen": n_mem_gen / n if n > 0 and has_gen else None,
    }


def main():
    args = parse_args()
    exp27_dir = Path(args.exp27_dir)
    exp26_dir = Path(args.exp26_dir)
    out_prefix = Path(args.output_prefix)

    per_sample_l128_dir = exp27_dir / "per_sample"
    per_sample_l0_dir = exp26_dir / "per_sample"

    model_records = []

    summary_path = args.summary_json
    files = list(per_sample_l128_dir.glob("per_sample_*_L128_prose.jsonl")) if per_sample_l128_dir.is_dir() else []

    if (summary_path and summary_path.is_file()) and not files:
        print(f"Loading precomputed summary from {summary_path}")
        with open(summary_path, "r", encoding="utf-8") as f:
            model_records = json.load(f)
    elif summary_path and summary_path.is_file() and not per_sample_l0_dir.is_dir():
        print(f"Loading precomputed summary from {summary_path}")
        with open(summary_path, "r", encoding="utf-8") as f:
            model_records = json.load(f)
    else:
        for p_l128 in sorted(files):
            stem = p_l128.stem.replace("per_sample_", "").replace("_L128_prose", "")
            model_name = stem
            
            # Load L=128 counts
            res_l128 = read_jsonl_counts(p_l128)
            if res_l128["n"] < 100:
                continue

            # Load L=0 counts from exp26 (input_qa)
            p_l0 = per_sample_l0_dir / f"per_sample_{model_name}_qa.jsonl"
            if not p_l0.is_file():
                print(f"Warning: L=0 file not found for {model_name}: {p_l0}")
                continue
            res_l0 = read_jsonl_counts(p_l0)

            fam, disp, scale = classify_model(model_name)

            # Delta P(in-context) = P(ctx | L=128) - P(ctx | L=0)
            delta_ctx_lp = (res_l128["p_ctx_lp"] - res_l0["p_ctx_lp"]) * 100
            delta_mem_lp = (res_l128["p_mem_lp"] - res_l0["p_mem_lp"]) * 100

            delta_ctx_gen = None
            if res_l128["p_ctx_gen"] is not None and res_l0["p_ctx_gen"] is not None:
                delta_ctx_gen = (res_l128["p_ctx_gen"] - res_l0["p_ctx_gen"]) * 100

            model_records.append({
                "model_name": model_name,
                "family": fam,
                "display_name": disp,
                "scale": scale,
                "n": res_l128["n"],
                "p_ctx_l0_lp": round(res_l0["p_ctx_lp"] * 100, 1),
                "p_ctx_l128_lp": round(res_l128["p_ctx_lp"] * 100, 1),
                "delta_ctx_lp_pp": round(delta_ctx_lp, 1),
                "p_mem_l0_lp": round(res_l0["p_mem_lp"] * 100, 1),
                "p_mem_l128_lp": round(res_l128["p_mem_lp"] * 100, 1),
                "delta_mem_lp_pp": round(delta_mem_lp, 1),
                "p_ctx_l128_gen": round(res_l128["p_ctx_gen"] * 100, 1) if res_l128["p_ctx_gen"] is not None else None,
                "p_mem_l128_gen": round(res_l128["p_mem_gen"] * 100, 1) if res_l128["p_mem_gen"] is not None else None,
            })

        # Sort models by family order, then scale
        def sort_key(rec):
            try:
                fam_idx = FAMILY_ORDER.index(rec["family"])
            except ValueError:
                fam_idx = 99
            return (fam_idx, rec["scale"], rec["display_name"])

        model_records.sort(key=sort_key)
        print(f"Processed {len(model_records)} models.")

        # Save summary JSON
        if summary_path:
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(model_records, f, indent=2)
            print(f"Saved summary JSON to {summary_path}")

    # Generate LaTeX Table
    table_lines = [
        "% Auto-generated by analyze_prose_l128.py\n",
        "\\begin{table*}[t]\n",
        "  \\centering\n",
        "  \\small\n",
        "  \\setlength{\\tabcolsep}{6pt}\n",
        "  \\caption{Cross-family effect of inserting $L{=}128$ unrelated prose filler between substitution conflict and query (World Capital, $n{=}2{,}000$, log-probability classifier). $\\Delta = P(\\text{ctx} \\mid L{=}128) - P(\\text{ctx} \\mid L{=}0)$.}\n",
        "  \\label{tab:cross-family-prose-l128}\n",
        "  \\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}ll rrr rrr@{}}\n",
        "    \\toprule\n",
        "    Family & Model & $P(\\text{ctx}\\mid L{=}0)$ & $P(\\text{ctx}\\mid L{=}128)$ & $\\Delta P(\\text{ctx})$ & $P(\\text{mem}\\mid L{=}0)$ & $P(\\text{mem}\\mid L{=}128)$ & $\\Delta P(\\text{mem})$ \\\\\n",
        "    \\midrule\n",
    ]
    curr_fam = None
    for r in model_records:
        if r["family"] != curr_fam:
            if curr_fam is not None:
                table_lines.append("    \\addlinespace\n")
            curr_fam = r["family"]
            table_lines.append(f"    \\multicolumn{{8}}{{l}}{{\\textbf{{{curr_fam}}}}} \\\\\n")

        d_ctx_sign = "+" if r["delta_ctx_lp_pp"] > 0 else ""
        d_mem_sign = "+" if r["delta_mem_lp_pp"] > 0 else ""
        table_lines.append(
            f"    & {r['display_name']} & {r['p_ctx_l0_lp']:4.1f}\\% & {r['p_ctx_l128_lp']:4.1f}\\% & "
            f"\\textbf{{{d_ctx_sign}{r['delta_ctx_lp_pp']:.1f}}} & "
            f"{r['p_mem_l0_lp']:4.1f}\\% & {r['p_mem_l128_lp']:4.1f}\\% & "
            f"{d_mem_sign}{r['delta_mem_lp_pp']:.1f} \\\\\n"
        )
    table_lines.append("    \\bottomrule\n")
    table_lines.append("  \\end{tabular*}\n")
    table_lines.append("\\end{table*}\n")

    for tex_path in [Path("tables/cross_family_prose_l128.tex"), Path("paper/tables/cross_family_prose_l128.tex")]:
        tex_path.parent.mkdir(parents=True, exist_ok=True)
        with open(tex_path, "w", encoding="utf-8") as f:
            f.writelines(table_lines)
        print(f"Saved LaTeX table to {tex_path}")

    # Generate Plot: Delta P(in-context) across all models
    plot_path = Path("figures/cross_family_prose_l128_delta.png")
    plot_path.parent.mkdir(parents=True, exist_ok=True)

    if model_records:
        fig, ax = plt.subplots(figsize=(15, 6.5))
        names = [r["display_name"] for r in model_records]
        deltas = [r["delta_ctx_lp_pp"] for r in model_records]
        families = [r["family"] for r in model_records]
        colors = [FAMILY_COLORS.get(f, "#7f7f7f") for f in families]

        bars = ax.bar(
            range(len(names)),
            deltas,
            color=colors,
            edgecolor="black",
            linewidth=0.7,
            width=0.75,
        )

        # Baseline reference line at 0
        ax.axhline(0, color="black", linestyle="-", linewidth=0.9)

        # Vertical dividers between families
        for i in range(1, len(families)):
            if families[i] != families[i - 1]:
                ax.axvline(i - 0.5, color="#888888", linestyle="--", linewidth=0.9, alpha=0.6)

        # Value labels on bars
        for bar in bars:
            h = bar.get_height()
            va = "bottom" if h >= 0 else "top"
            offset = 1.0 if h >= 0 else -1.5
            sign = "+" if h > 0 else ""
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                h + offset,
                f"{sign}{h:.1f}",
                ha="center",
                va=va,
                fontsize=7,
                rotation=90,
            )

        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8.5)
        ax.set_ylabel(r"$\Delta P(\mathrm{in\text{-}context}) = P(L{=}128) - P(L{=}0)$ (percentage points)", fontsize=11)
        ax.set_title(
            "Effect of 128 Tokens of Unrelated Prose Filler on In-Context Following Across Model Families",
            fontsize=13,
            fontweight="bold",
            pad=32,
        )

        min_y = min(deltas) - 15 if deltas else -20
        max_y = max(deltas) + 20 if deltas else 30
        ax.set_ylim(min_y, max_y)
        ax.grid(axis="y", linestyle="--", alpha=0.4)

        # Legend
        present_families = [f for f in FAMILY_ORDER if f in families]
        handles = [
            plt.Rectangle((0, 0), 1, 1, facecolor=FAMILY_COLORS[f], edgecolor="black", linewidth=0.7)
            for f in present_families
        ]
        ax.legend(
            handles,
            present_families,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.10),
            ncol=4,
            frameon=True,
            framealpha=0.95,
            fontsize=9,
        )

        plt.tight_layout()
        plt.savefig(plot_path, dpi=300)
        plt.close()
        print(f"Saved delta plot to {plot_path}")

        # Also copy to paper/figures
        paper_fig_path = Path("paper/figures/cross_family_prose_l128_delta.png")
        paper_fig_path.parent.mkdir(parents=True, exist_ok=True)
        import shutil
        shutil.copy(plot_path, paper_fig_path)
        print(f"Copied figure to {paper_fig_path}")


if __name__ == "__main__":
    main()
