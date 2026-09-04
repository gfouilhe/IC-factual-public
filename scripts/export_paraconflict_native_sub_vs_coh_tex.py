"""Export native ParaConflict Sub vs Coherent P(in-context) table to LaTeX."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_PYTHIA_SIZES = ("160m", "410m", "1b", "1.4b", "2.8b", "6.9b", "12b")
_EXPECTED_N = 2146

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
        "Ministral-3",
        tuple(
            f"ministral3-{v}-{s}"
            for v in ("instruct", "reasoning", "base")
            for s in ("3b", "8b", "14b")
        ),
    ),
)


def _load_counts(path: Path, metric: str) -> tuple[int, int, int]:
    """Return (sub_in_context, coh_in_context, n)."""
    if not path.is_file() or path.stat().st_size == 0:
        return 0, 0, 0
    ctx_k = "in_context_lp" if metric == "lp" else "in_context_gen"
    sub_ctx = coh_ctx = n = 0
    with path.open(encoding="utf-8") as f:
        f.readline()
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            sub = row.get("substitution_conflict") or {}
            coh = row.get("coherent_conflict") or {}
            if not sub or not coh:
                continue
            n += 1
            sub_ctx += int(bool(sub.get(ctx_k)))
            coh_ctx += int(bool(coh.get(ctx_k)))
    return sub_ctx, coh_ctx, n


def _merge_counts(paths: list[Path], metric: str) -> tuple[float, float, int]:
    sub_total = coh_total = n_total = 0
    for path in paths:
        sub, coh, n = _load_counts(path, metric)
        sub_total += sub
        coh_total += coh
        n_total += n
    if not n_total:
        return float("nan"), float("nan"), 0
    return sub_total / n_total, coh_total / n_total, n_total


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


def _collect(
    exp25_native: Path,
    exp23_relations: Path,
    exp25_wc: Path,
    exp25_native_other: Path,
    metric: str,
) -> dict[str, tuple[float, float, int]]:
    out: dict[str, tuple[float, float, int]] = {}
    missing_wc: list[str] = []

    for size in _PYTHIA_SIZES:
        path = exp25_native / f"per_sample_{size}.jsonl"
        p_sub, p_coh, n = _merge_counts([path], metric)
        if n:
            out[size] = (p_sub, p_coh, n)

    for family, labels in _FAMILY_ORDER:
        if family == "Pythia":
            continue
        for label in labels:
            full = exp25_native_other / f"per_sample_{label}.jsonl"
            if full.is_file() and full.stat().st_size > 0:
                p_sub, p_coh, n = _merge_counts([full], metric)
            else:
                rel = exp23_relations / f"per_sample_relations_{label}.jsonl"
                wc = exp25_wc / f"per_sample_{label}.jsonl"
                paths = [p for p in (rel, wc) if p.is_file() and p.stat().st_size > 0]
                p_sub, p_coh, n = _merge_counts(paths, metric)
                if wc.is_file() is False or wc.stat().st_size == 0:
                    missing_wc.append(label)
            if n:
                out[label] = (p_sub, p_coh, n)

    if missing_wc:
        print(
            f"Warning: missing World Capital supplement for {len(missing_wc)} models "
            f"(merged five-category exp23 only). First: {missing_wc[:3]}",
            file=sys.stderr,
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
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
        "--output-tex",
        type=Path,
        default=Path("paper/tables/paraconflict_native_sub_vs_coh.tex"),
    )
    parser.add_argument("--metric", choices=("gen", "lp"), default="gen")
    parser.add_argument(
        "--require-full",
        action="store_true",
        help="Exit non-zero unless every model has n=2146.",
    )
    args = parser.parse_args()

    rates = _collect(
        args.exp25_native_dir,
        args.exp23_relations_dir,
        args.exp25_wc_dir,
        args.exp25_native_other_dir,
        args.metric,
    )
    clf = "generative" if args.metric == "gen" else "logprob"

    if args.require_full:
        incomplete = [
            lb for lb, (_, _, n) in rates.items() if n != _EXPECTED_N
        ]
        if incomplete:
            print(f"Incomplete models ({len(incomplete)}): {incomplete[:5]}...", file=sys.stderr)
            sys.exit(1)

    lines = [
        "% Auto-generated by scripts/export_paraconflict_native_sub_vs_coh_tex.py",
        "\\begin{table*}[t]",
        "  \\centering",
        "  \\footnotesize",
        "  \\setlength{\\tabcolsep}{4pt}",
        "  \\caption{Native ParaConflict prompts: $P(\\text{in-context})$ under",
        " \\texttt{Substitution Conflict} vs.\\ \\texttt{Coherent Conflict}",
        f" ({clf} classifier; all six relation categories, $n" + "{=}" + "2146$).",
        " $\\Delta = P(\\text{ctx}\\mid\\text{Coh}) -",
        " P(\\text{ctx}\\mid\\text{Sub})$.}",
        "  \\label{tab:paraconflict-native-sub-vs-coh}",
        "  \\begin{tabular}{@{}lrrr@{}}",
        "    \\toprule",
        "    Model & $P(\\text{ctx}\\mid\\text{Sub})$ & $P(\\text{ctx}\\mid\\text{Coh})$ & $\\Delta$ \\\\",
        "    \\midrule",
    ]

    for family, labels in _FAMILY_ORDER:
        present = [lb for lb in labels if lb in rates]
        if not present:
            continue
        lines.append(f"    \\multicolumn{{4}}{{l}}{{\\textbf{{{family}}}}} \\\\")
        for label in present:
            p_sub, p_coh, n = rates[label]
            delta = p_coh - p_sub
            name = _display_label(label).replace(" ", "~")
            lines.append(
                f"    {name} & {p_sub:.2f} & {p_coh:.2f} & {delta:+.2f} \\\\"
            )
        lines.append("    \\addlinespace")

    if lines[-1] == "    \\addlinespace":
        lines.pop()

    lines.extend(
        [
            "    \\bottomrule",
            "  \\end{tabular}",
            "\\end{table*}",
            "",
        ]
    )

    args.output_tex.parent.mkdir(parents=True, exist_ok=True)
    args.output_tex.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {args.output_tex}")


if __name__ == "__main__":
    main()
