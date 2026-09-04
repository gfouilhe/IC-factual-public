"""Cross-tab the frequency-bin axis (paper Fig. 3 / Fig. 2) with the
context-length axis (exp01).

Given per-sample JSONLs from the capitals context sweep (tail position,
K=1) and the per-country Zipf table from
``scripts/compute_subject_frequency.py``, produce two figures:

* ``<prefix>_by_country.png``     -- P(in-context) vs queried-country Zipf
                                    quantile, faceted by Pythia size
                                    (rows) and filler kind (cols), with
                                    one line per filler length.  Tells
                                    us whether the prose-tail flip
                                    (exp01 Finding 2) flattens or
                                    steepens the P(memorized) vs country
                                    Zipf slope.
* ``<prefix>_by_distractor.png``  -- same but binned by the distractor
                                    capital's true country's Zipf
                                    (the in-context answer side).

Inputs
------
* ``--per-sample-glob``           glob into per_sample_*.jsonl
* ``--freq-json``                 capitals_subject_frequency.json with
                                  ``subjects[country].zipf``.
* ``--output-prefix``             output prefix (one per binning axis).

The analyzer only considers position=tail, n_copies=1 cells (the
canonical "vary filler length" grid).  Bins are quantile-based on the
union of Zipf values actually appearing in the data.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def _load_jsonl(path: Path):
    meta = None
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if obj.get("_meta") is True:
                meta = obj
            else:
                records.append(obj)
    return meta, records


def _load_zipf(path: Path) -> dict[str, float]:
    data = json.loads(Path(path).read_text())
    subjects = data.get("subjects", {})
    return {k: v.get("zipf") for k, v in subjects.items() if v.get("zipf") is not None}


def _quantile_edges(values: list[float], n_bins: int) -> list[float]:
    """Closed-open quantile bin edges with a tiny pad at both ends."""
    if not values:
        return []
    import numpy as np

    qs = np.linspace(0.0, 1.0, n_bins + 1)
    edges = list(map(float, np.quantile(values, qs)))
    edges[0] -= 1e-9
    edges[-1] += 1e-9
    return edges


def _assign_bin(z: float, edges: list[float]) -> int:
    for i in range(len(edges) - 1):
        if edges[i] < z <= edges[i + 1]:
            return i
    return len(edges) - 2


def _plot(rows, edges, n_bins, bin_by_label, out_path: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    PYTHIA_ORDER = ["160m", "410m", "1b", "1.4b", "2.8b", "6.9b", "12b"]
    sizes = [s for s in PYTHIA_ORDER if any(r["size"] == s for r in rows)]
    fillers = sorted({r["filler"] for r in rows})
    Ls = sorted({r["L"] for r in rows})

    by_key = defaultdict(lambda: {"n": 0, "mem": 0, "ctx": 0})
    for r in rows:
        b = _assign_bin(r["zipf"], edges)
        d = by_key[(r["size"], r["L"], r["filler"], b)]
        d["n"] += 1
        if r["mem"]:
            d["mem"] += 1
        if r["ctx"]:
            d["ctx"] += 1

    centers = [0.5 * (edges[i] + edges[i + 1]) for i in range(n_bins)]
    cmap = plt.cm.viridis
    n = max(len(Ls) - 1, 1)
    L_colors = {L: cmap(i / n) for i, L in enumerate(Ls)}

    fig, axes = plt.subplots(
        len(sizes),
        len(fillers),
        figsize=(4.6 * len(fillers), 1.9 * len(sizes) + 0.6),
        sharex=True,
        sharey=True,
        squeeze=False,
    )

    for r_idx, sz in enumerate(sizes):
        for c_idx, fk in enumerate(fillers):
            ax = axes[r_idx][c_idx]
            for L in Ls:
                xs, ys = [], []
                for b in range(n_bins):
                    d = by_key[(sz, L, fk, b)]
                    if d["n"] >= 30:
                        xs.append(centers[b])
                        ys.append(d["ctx"] / d["n"])
                if xs:
                    ax.plot(
                        xs,
                        ys,
                        "-o",
                        color=L_colors[L],
                        label=f"L={L}" if (r_idx == 0 and c_idx == 0) else None,
                        lw=1.5,
                        ms=4,
                    )
            ax.set_ylim(0, 1)
            ax.grid(True, alpha=0.25)
            if c_idx == 0:
                ax.set_ylabel(f"Pythia-{sz}\nP(ctx)", fontsize=9)
            if r_idx == 0:
                ax.set_title(f"filler={fk}", fontsize=10)
            if r_idx == len(sizes) - 1:
                ax.set_xlabel(f"{bin_by_label} Zipf (quantile bins)", fontsize=9)
    if axes[0][0].get_legend_handles_labels()[0]:
        axes[0][0].legend(loc="best", fontsize=8, framealpha=0.85, ncol=2)

    fig.suptitle(
        f"P(in-context) vs {bin_by_label} country Zipf\n"
        f"(tail position, K=1, paper Q/A template; bin requires n>=30)",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")

    return by_key, centers


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-sample-glob", required=True)
    parser.add_argument(
        "--freq-json", default="output/capitals_subject_frequency.json"
    )
    parser.add_argument(
        "--output-prefix",
        default="experiments/exp09_freq_x_context/plots/exp09",
    )
    parser.add_argument("--n-bins", type=int, default=5)
    parser.add_argument(
        "--bin-by",
        choices=("country", "distractor", "both"),
        default="both",
        help=(
            "Bin prompts by the Zipf of the queried country, the "
            "distractor-capital's true country, or both (default: both)."
        ),
    )
    parser.add_argument(
        "--metric",
        choices=("lp", "gen"),
        default="lp",
        help="Classifier metric inside substitution_conflict records.",
    )
    args = parser.parse_args()

    mem_key = f"memorized_{args.metric}"
    ctx_key = f"in_context_{args.metric}"

    zipf = _load_zipf(Path(args.freq_json))
    if not zipf:
        raise SystemExit("Empty Zipf table.")

    rows_country = []
    rows_distractor = []
    for p in sorted(Path().glob(args.per_sample_glob)):
        meta, recs = _load_jsonl(p)
        if not meta:
            print(f"  skip {p.name}: no _meta")
            continue
        in_meta = meta.get("input_meta") or {}
        L = in_meta.get("length_tokens_target", in_meta.get("length_tokens", 0))
        filler = in_meta.get("filler_kind", "prose")
        pos = in_meta.get("conflict_position", "tail")
        K = int(in_meta.get("n_copies", 1))
        size = meta.get("model_size")
        if pos != "tail" or K != 1:
            continue
        for r in recs:
            country = r.get("country")
            d_country = r.get("distractor_country")
            sc = r.get("substitution_conflict") or {}
            mem = bool(sc.get(mem_key))
            ctx = bool(sc.get(ctx_key))
            z_c = zipf.get(country)
            z_d = zipf.get(d_country)
            base = {
                "size": size,
                "L": L,
                "filler": filler,
                "mem": mem,
                "ctx": ctx,
            }
            if z_c is not None:
                rows_country.append({**base, "zipf": z_c})
            if z_d is not None:
                rows_distractor.append({**base, "zipf": z_d})

    print(f"loaded {len(rows_country)} rows for country, {len(rows_distractor)} for distractor")

    summaries = {}
    if args.bin_by in ("country", "both") and rows_country:
        edges = _quantile_edges([r["zipf"] for r in rows_country], args.n_bins)
        by_key, centers = _plot(
            rows_country,
            edges,
            args.n_bins,
            "queried-country",
            Path(args.output_prefix + "_by_country.png"),
        )
        summaries["by_country"] = {
            "edges": edges,
            "centers": centers,
            "rows": [
                {
                    "size": sz, "L": L, "filler": fk, "bin": b,
                    "n": d["n"], "mem": d["mem"], "ctx": d["ctx"],
                    "p_mem": d["mem"] / d["n"] if d["n"] else None,
                    "p_ctx": d["ctx"] / d["n"] if d["n"] else None,
                }
                for (sz, L, fk, b), d in sorted(by_key.items())
            ],
        }

    if args.bin_by in ("distractor", "both") and rows_distractor:
        edges = _quantile_edges([r["zipf"] for r in rows_distractor], args.n_bins)
        by_key, centers = _plot(
            rows_distractor,
            edges,
            args.n_bins,
            "distractor-country",
            Path(args.output_prefix + "_by_distractor.png"),
        )
        summaries["by_distractor"] = {
            "edges": edges,
            "centers": centers,
            "rows": [
                {
                    "size": sz, "L": L, "filler": fk, "bin": b,
                    "n": d["n"], "mem": d["mem"], "ctx": d["ctx"],
                    "p_mem": d["mem"] / d["n"] if d["n"] else None,
                    "p_ctx": d["ctx"] / d["n"] if d["n"] else None,
                }
                for (sz, L, fk, b), d in sorted(by_key.items())
            ],
        }

    out_json = Path(args.output_prefix + "_summary.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summaries, indent=2))
    print(f"wrote {out_json}")


if __name__ == "__main__":
    main()
