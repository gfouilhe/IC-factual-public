"""Cross-template comparison plot for exp12.

The default analyzer emits separate P(in-context)/P(memorized) length
sweeps per scaffold.  With only L ∈ {0, 128} and the headline being
*which closing template* is used, this script overlays all five
templates (paper ``qa`` and ``bare`` baselines plus the three exp12
scaffolds) on shared axes: model size on the x-axis, one panel per
filler length.

Run from the repo root::

    python experiments/exp12_alt_scaffolds/plot_template_comparison.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ic_factual.config import PYTHIA_SIZES

REPO_ROOT = Path(__file__).resolve().parent.parent
PLOTS_DIR = REPO_ROOT / "figures"
OUT_PATH = PLOTS_DIR / "exp12_template_comparison.png"
OUT_PAPER = REPO_ROOT / "paper/figures/exp12_template_comparison.png"

TEMPLATE_SOURCES: dict[str, Path] = {
    "qa": REPO_ROOT / "summaries/capitals_ctx_sweep_v1_summary.json",
    "bare": REPO_ROOT / "summaries/capitals_ctx_sweep_bare_summary.json",
    "possessive": REPO_ROOT / "summaries/exp12_possessive_summary.json",
    "learned": REPO_ROOT / "summaries/exp12_learned_summary.json",
    "of_course": REPO_ROOT / "summaries/exp12_of_course_summary.json",
}

TEMPLATE_ORDER: tuple[str, ...] = (
    "qa",
    "bare",
    "possessive",
    "learned",
    "of_course",
)

# Distinct, colorblind-friendly palette (one color per template).
TEMPLATE_COLORS: dict[str, str] = {
    "qa": "#4c72b0",
    "bare": "#dd8452",
    "possessive": "#55a868",
    "learned": "#c44e52",
    "of_course": "#8172b3",
}

TEMPLATE_MARKERS: dict[str, str] = {
    "qa": "o",
    "bare": "s",
    "possessive": "^",
    "learned": "D",
    "of_course": "v",
}


def _load_prose_tail_rows(path: Path) -> list[dict]:
    rows = json.loads(path.read_text())["rows"]
    return [
        r
        for r in rows
        if r["filler_kind"] == "prose"
        and r["conflict_position"] == "tail"
        and r["n_copies"] == 1
    ]


def _build_index(rows: list[dict]) -> dict[tuple[str, int], float]:
    return {(r["size"], r["length_tokens_target"]): r["p_in_context"] for r in rows}


def main() -> None:
    lengths = [0, 128]
    indices: dict[str, dict[tuple[str, int], float]] = {}
    for name, path in TEMPLATE_SOURCES.items():
        indices[name] = _build_index(_load_prose_tail_rows(path))

    sizes = [
        s
        for s in PYTHIA_SIZES
        if any((s, L) in indices[t] for t in TEMPLATE_ORDER for L in lengths)
    ]
    x_idx = list(range(len(sizes)))

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(
        1,
        len(lengths),
        figsize=(6.5 * len(lengths), 4.8),
        sharey=True,
        squeeze=False,
    )

    for col, L in enumerate(lengths):
        ax = axes[0][col]
        for template in TEMPLATE_ORDER:
            ys: list[float | None] = []
            for size in sizes:
                val = indices[template].get((size, L))
                ys.append(val)
            valid_x = [i for i, y in enumerate(ys) if y is not None]
            valid_y = [ys[i] for i in valid_x]
            if not valid_x:
                continue
            ax.plot(
                valid_x,
                valid_y,
                f"-{TEMPLATE_MARKERS[template]}",
                color=TEMPLATE_COLORS[template],
                lw=1.8,
                ms=5,
                label=template,
            )
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.25)
        ax.set_title(f"L={L} prose filler (tail, K=1)")
        ax.set_xlabel("Pythia size")
        ax.set_xticks(x_idx)
        ax.set_xticklabels(sizes)
        if col == 0:
            ax.set_ylabel("P(in-context answer)")

    axes[0][-1].legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        fontsize=8,
        framealpha=0.9,
    )

    fig.suptitle(
        "Alternative question scaffolds: P(in-context) by template "
        "across Pythia sizes (lp-argmax classifier)"
    )
    fig.tight_layout(rect=(0, 0, 0.88, 0.93))
    for out in (OUT_PATH, OUT_PAPER):
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=140, bbox_inches="tight")
        print(f"Wrote {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
