"""Paper figure for exp02 conflict-position sweep (L=128 and L=512 only).

Run from the repo root::

    python experiments/exp02_position_sweep/plot_paper_position.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ic_factual.config import PYTHIA_SIZES

REPO_ROOT = Path(__file__).resolve().parent.parent
SUMMARY = REPO_ROOT / "summaries/capitals_ctx_sweep_pos_summary.json"
OUT_PAPER = REPO_ROOT / "paper/figures/exp02_position_sweep.png"
OUT_FIG = REPO_ROOT / "figures/exp02_position_sweep.png"

_LENGTHS = (128, 512)
_POSITIONS = ("tail", "middle", "front")
_FILLERS = ("capital_noise", "prose")
# exp02 position cells were run for 160m..6.9b only (12b has tail-only reuse).
_SIZES = tuple(s for s in PYTHIA_SIZES if s != "12b")


def _load_rows() -> list[dict]:
    rows = json.loads(SUMMARY.read_text())["rows"]
    return [
        r
        for r in rows
        if r["n_copies"] == 1
        and r["length_tokens_target"] in _LENGTHS
        and r["conflict_position"] in _POSITIONS
        and r["filler_kind"] in _FILLERS
        and r["size"] in _SIZES
    ]


def _color_map(sizes: tuple[str, ...]) -> dict[str, tuple]:
    cmap = plt.cm.viridis
    n = max(len(sizes) - 1, 1)
    return {s: cmap(i / n) for i, s in enumerate(sizes)}


def main() -> None:
    rows = _load_rows()
    cmap = _color_map(_SIZES)

    fig, axes = plt.subplots(
        len(_FILLERS),
        len(_LENGTHS),
        figsize=(6.8, 4.6),
        sharex=True,
        sharey=True,
        squeeze=False,
    )

    x_idx = list(range(len(_POSITIONS)))
    for r_idx, filler in enumerate(_FILLERS):
        for c_idx, length in enumerate(_LENGTHS):
            ax = axes[r_idx][c_idx]
            for size in _SIZES:
                ys: list[float] = []
                for pos in _POSITIONS:
                    matches = [
                        r
                        for r in rows
                        if r["size"] == size
                        and r["filler_kind"] == filler
                        and r["length_tokens_target"] == length
                        and r["conflict_position"] == pos
                    ]
                    ys.append(matches[0]["p_in_context"] if matches else float("nan"))
                ax.plot(
                    x_idx,
                    ys,
                    "-o",
                    color=cmap[size],
                    lw=1.8,
                    ms=4.5,
                    label=f"Pythia-{size}",
                )
            ax.set_xticks(x_idx)
            ax.set_xticklabels(_POSITIONS)
            ax.set_ylim(0, 1)
            ax.grid(True, alpha=0.25)
            if r_idx == 0:
                ax.set_title(f"$L={length}$")
            if c_idx == 0:
                ax.set_ylabel(filler)

    fig.supylabel("$P(\\mathrm{in\\text{-}context})$")
    fig.supxlabel("Conflict position")
    axes[0][-1].legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        fontsize=7.5,
        framealpha=0.9,
        borderaxespad=0.0,
    )
    fig.tight_layout(rect=(0.04, 0.04, 0.88, 1.0))

    for out in (OUT_PAPER, OUT_FIG):
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=160, bbox_inches="tight")
        print(f"Wrote {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
