#!/usr/bin/env python3
"""Aggregate and visualize cross-family phrasing sensitivity for exp26."""

import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

TEMPLATES = ["qa", "bare", "possessive", "learned", "of_course"]

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
    parser = argparse.ArgumentParser(description="Analyze cross-family phrasing results.")
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=Path("summaries/exp26_phrasing_summary.json"),
        help="Path to precomputed phrasing summary JSON.",
    )
    parser.add_argument(
        "--per-sample-dir",
        type=Path,
        default=Path("results/per_sample_phrasing"),
        help="Optional path to raw per_sample directory for recomputing.",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("summaries"),
        help="Directory prefix for output summaries.",
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


def read_jsonl_results(file_path: Path) -> dict:
    n = 0
    n_mem = 0
    n_ctx = 0
    n_tie = 0
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("_meta"):
                continue
            n += 1
            block = rec.get("substitution_conflict", {})
            if block.get("memorized_lp"):
                n_mem += 1
            elif block.get("in_context_lp"):
                n_ctx += 1
            else:
                n_tie += 1
    return {
        "n": n,
        "p_mem": n_mem / n if n > 0 else 0.0,
        "p_ctx": n_ctx / n if n > 0 else 0.0,
        "p_tie": n_tie / n if n > 0 else 0.0,
    }


def main():
    args = parse_args()
    per_sample_dir = Path(args.per_sample_dir)
    out_prefix = Path(args.output_prefix)

    data_by_model: dict[str, dict[str, dict]] = {}

    for p in per_sample_dir.glob("per_sample_*_*.jsonl"):
        stem = p.stem.replace("per_sample_", "")
        matched_tmpl = None
        for tmpl in sorted(TEMPLATES, key=len, reverse=True):
            if stem.endswith(f"_{tmpl}"):
                matched_tmpl = tmpl
                model_name = stem[: -len(tmpl) - 1]
                break
        if not matched_tmpl:
            continue

        res = read_jsonl_results(p)
        if res["n"] < 100:
            continue
        data_by_model.setdefault(model_name, {})[matched_tmpl] = res

    summary_path = args.summary_json

    # Build structured records
    model_records = []
    if not data_by_model and summary_path.is_file():
        print(f"Loading precomputed phrasing summary from {summary_path}")
        with open(summary_path, "r", encoding="utf-8") as f:
            model_records = json.load(f)
    else:
        for model_name, tmpl_dict in data_by_model.items():
            fam, disp, scale = classify_model(model_name)
            mem_rates = {t: tmpl_dict[t]["p_mem"] for t in TEMPLATES if t in tmpl_dict}
            ctx_rates = {t: tmpl_dict[t]["p_ctx"] for t in TEMPLATES if t in tmpl_dict}
            
            if mem_rates:
                max_swing = max(mem_rates.values()) - min(mem_rates.values())
            else:
                max_swing = 0.0

            of_course_minus_learned = None
            if "of_course" in mem_rates and "learned" in mem_rates:
                of_course_minus_learned = mem_rates["of_course"] - mem_rates["learned"]

            model_records.append({
                "model_name": model_name,
                "family": fam,
                "display_name": disp,
                "scale": scale,
                "p_mem": mem_rates,
                "p_ctx": ctx_rates,
                "max_swing_pp": round(max_swing * 100, 1),
                "of_course_swing_pp": round(of_course_minus_learned * 100, 1) if of_course_minus_learned is not None else None,
            })

        # Sort models by family order, then scale
        def sort_key(rec):
            try:
                fam_idx = FAMILY_ORDER.index(rec["family"])
            except ValueError:
                fam_idx = 99
            return (fam_idx, rec["scale"], rec["display_name"])

        model_records.sort(key=sort_key)

        # Save summary JSON
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(model_records, f, indent=2)

    # Generate LaTeX Table
    for tex_path in (Path("paper/tables/cross_family_phrasing.tex"), Path("tables/cross_family_phrasing.tex")):
        tex_path.parent.mkdir(parents=True, exist_ok=True)
        with open(tex_path, "w", encoding="utf-8") as f:
            f.write("% Auto-generated by analyze_phrasing.py\n")
            f.write("\\begin{table*}[t]\n")
            f.write("  \\centering\n")
            f.write("  \\small\n")
            f.write("  \\setlength{\\tabcolsep}{5pt}\n")
            f.write("  \\caption{Cross-family question template sensitivity ($P(\\text{memorized})$ under substitution conflict at $L{=}0$, log-probability classifier, $n{=}2{,}000$). $\\Delta_{\\max} = \\max(P_{\\text{mem}}) - \\min(P_{\\text{mem}})$ across templates.}\n")
            f.write("  \\label{tab:cross-family-phrasing}\n")
            f.write("  \\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}ll rrrrr r@{}}\n")
            f.write("    \\toprule\n")
            f.write("    Family & Model & \\texttt{qa} & \\texttt{bare} & \\texttt{possessive} & \\texttt{learned} & \\texttt{of\\_course} & $\\Delta_{\\max}$ (pp) \\\\\n")
            f.write("    \\midrule\n")

            curr_fam = None
            for r in model_records:
                if r["family"] != curr_fam:
                    if curr_fam is not None:
                        f.write("    \\addlinespace\n")
                    curr_fam = r["family"]
                    f.write(f"    \\multicolumn{{8}}{{l}}{{\\textbf{{{curr_fam}}}}} \\\\\n")
                
                p_m = r["p_mem"]
                qa_str = f"{p_m.get('qa', 0)*100:4.1f}\\%" if 'qa' in p_m else "---"
                bare_str = f"{p_m.get('bare', 0)*100:4.1f}\\%" if 'bare' in p_m else "---"
                poss_str = f"{p_m.get('possessive', 0)*100:4.1f}\\%" if 'possessive' in p_m else "---"
                lrn_str = f"{p_m.get('learned', 0)*100:4.1f}\\%" if 'learned' in p_m else "---"
                ofc_str = f"{p_m.get('of_course', 0)*100:4.1f}\\%" if 'of_course' in p_m else "---"
                swing_str = f"\\textbf{{{r['max_swing_pp']:.1f}}}"

                f.write(f"    & {r['display_name']} & {qa_str} & {bare_str} & {poss_str} & {lrn_str} & {ofc_str} & {swing_str} \\\\\n")

            f.write("    \\bottomrule\n")
            f.write("  \\end{tabular*}\n")
            f.write("\\end{table*}\n")
        print(f"Saved LaTeX table to {tex_path}")

    # Generate Plot
    plot_paths = [Path("figures/cross_family_phrasing_swings.png"), Path("paper/figures/cross_family_phrasing_swings.png")]
    for p_path in plot_paths:
        p_path.parent.mkdir(parents=True, exist_ok=True)
    
    if model_records:
        fig, ax = plt.subplots(figsize=(15, 6.5))
        names = [r["display_name"] for r in model_records]
        swings = [r["max_swing_pp"] for r in model_records]
        families = [r["family"] for r in model_records]
        
        colors = [FAMILY_COLORS.get(f, "#7f7f7f") for f in families]

        bars = ax.bar(
            range(len(names)),
            swings,
            color=colors,
            edgecolor="black",
            linewidth=0.7,
            width=0.75,
        )

        # Add vertical dividers between distinct families
        for i in range(1, len(families)):
            if families[i] != families[i - 1]:
                ax.axvline(i - 0.5, color="#888888", linestyle="--", linewidth=0.9, alpha=0.6)

        # Value labels above bars
        for bar in bars:
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                h + 1.0,
                f"{h:.1f}",
                ha="center",
                va="bottom",
                fontsize=7,
                rotation=90,
            )

        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8.5)
        ax.set_ylabel(r"Max Phrasing Swing $\Delta P(\mathrm{mem})$ (percentage points)", fontsize=11)
        ax.set_title("Question Template Sensitivity Across Model Families (L=0)", fontsize=13, fontweight="bold", pad=32)
        ax.set_ylim(0, 115)
        ax.grid(axis="y", linestyle="--", alpha=0.4)

        # Legend matching Figure 1 styling and exact colors
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
        for p_path in plot_paths:
            plt.savefig(p_path, dpi=300)
            print(f"Saved plot to {p_path}")
        plt.close()


if __name__ == "__main__":
    main()
