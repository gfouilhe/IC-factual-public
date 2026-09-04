"""Plot per-layer logit-lens results from logit_lens_capitals.py outputs.

Reads one or more JSON files produced by ``scripts/logit_lens_capitals.py``
(one per model size) and produces a small grid:

* row = model size (input JSON file)
* col = quantity (mean answer/distractor logprob, gap, prob mass)

Conditions are overlaid in each cell as colored lines, one per
condition label.

Usage::

    python scripts/plot_logit_lens_capitals.py \\
        --inputs experiments/exp13_logit_lens/summaries/per_layer_2.8b.json \\
                 experiments/exp13_logit_lens/summaries/per_layer_12b.json \\
        --output experiments/exp13_logit_lens/plots/exp13_logit_lens.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _load(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _palette():
    return {
        "L0_std": "#2c7fb8",
        "L0_neg": "#d95f0e",
        "L128_prose_std": "#41ab5d",
        "L128_prose_neg": "#cb181d",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inputs",
        type=str,
        nargs="+",
        required=True,
        help="One or more per_layer_<size>.json files.",
    )
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    payloads = [_load(Path(p)) for p in args.inputs]
    # Per-row display label: prefer explicit `label`, fall back to the
    # legacy `pythia-<size>` for old exp13 JSONs.
    sizes: list[str] = []
    for i, p in enumerate(payloads):
        if p.get("label"):
            sizes.append(str(p["label"]))
        elif p.get("model_size"):
            sizes.append(f"pythia-{p['model_size']}")
        else:
            sizes.append(Path(args.inputs[i]).stem)

    palette = _palette()
    n_rows = len(payloads)
    n_cols = 3  # answer logprob | distractor logprob | gap

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(5.5 * n_cols, 3.6 * n_rows),
        squeeze=False,
    )

    handles_labels: list[tuple] = []  # for shared legend

    for r_idx, payload in enumerate(payloads):
        size = sizes[r_idx]
        conds = payload["conditions"]
        for cond in conds:
            if cond.get("n_used", 0) == 0:
                continue
            label = cond["label"]
            color = palette.get(label, None)
            num_layers = cond["num_layers"]
            xs = np.arange(num_layers)
            ans_lp = np.array(cond["mean_answer_logprob"])
            dist_lp = np.array(cond["mean_distractor_logprob"])
            gap = ans_lp - dist_lp

            ax_a = axes[r_idx][0]
            ax_d = axes[r_idx][1]
            ax_g = axes[r_idx][2]
            l1, = ax_a.plot(xs, ans_lp, "-o", color=color, ms=3, lw=1.4, label=label)
            ax_d.plot(xs, dist_lp, "-o", color=color, ms=3, lw=1.4, label=label)
            ax_g.plot(xs, gap, "-o", color=color, ms=3, lw=1.4, label=label)
            if r_idx == 0:
                handles_labels.append((l1, label))

        # cosmetics per row.  `size` is the display label (e.g. ``2.8b``
        # for old Pythia JSONs, or ``qwen3-base-4b`` for the exp19 JSONs);
        # we no longer hard-code the ``Pythia-`` prefix.
        for c_idx, (ax, title) in enumerate(zip(
            axes[r_idx],
            [
                f"{size}: logprob(memorized first-tok)",
                f"{size}: logprob(distractor first-tok)",
                f"{size}: gap = ans - dist (>0 → memorized)",
            ],
        )):
            ax.set_title(title, fontsize=11)
            ax.set_xlabel("Layer (0 = embedding read-off, 1..N = transformer block out)")
            ax.set_ylabel("Mean logprob" if c_idx < 2 else "Mean Δ logprob")
            ax.grid(True, alpha=0.3)
            ax.axhline(0, color="gray", lw=0.7) if c_idx == 2 else None

    # Shared legend at the bottom
    if handles_labels:
        fig.legend(
            [h for h, _ in handles_labels],
            [l for _, l in handles_labels],
            loc="lower center",
            ncol=len(handles_labels),
            bbox_to_anchor=(0.5, -0.02),
            frameon=False,
        )

    fig.suptitle("Per-layer logit lens: where memorized vs in-context wins", y=1.00)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
