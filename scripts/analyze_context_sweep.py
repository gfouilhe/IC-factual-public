"""Plot in-context vs memorized rate across Pythia sizes and filler lengths.

Inputs
------
Per-sample JSONLs produced by ``scripts/evaluate_capitals.py`` on inputs
built by ``scripts/build_capitals_context_sweep.py``.  Each input file's
``_meta`` line carries the per-file condition
(``length_tokens_target``, ``filler_kind``, ``conflict_position``,
``n_copies``) via the evaluator's ``input_meta`` field.

Outputs
-------
* ``<prefix>_length.png``                -- main figure: P(in-context) and
                                            P(memorized) vs filler length,
                                            one line per Pythia size,
                                            faceted by filler_kind.
* ``<prefix>_position.png`` (optional)   -- only emitted when multiple
                                            conflict positions are present.
* ``<prefix>_copies.png`` (optional)     -- only emitted when n_copies > 1
                                            is present.
* ``<prefix>_summary.json``              -- per-condition aggregate counts.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
from collections import defaultdict
from pathlib import Path

_config_path = Path(__file__).resolve().parent.parent / "ic_factual" / "config.py"
_config_spec = importlib.util.spec_from_file_location("_ic_config", _config_path)
assert _config_spec and _config_spec.loader
_config = importlib.util.module_from_spec(_config_spec)
_config_spec.loader.exec_module(_config)
MINISTRAL3_SIZES = _config.MINISTRAL3_SIZES
MINISTRAL3_VARIANTS = _config.MINISTRAL3_VARIANTS
PYTHIA_SIZES = _config.PYTHIA_SIZES
QWEN3_BASE_SIZES = _config.QWEN3_BASE_SIZES
QWEN3_SIZES = _config.QWEN3_SIZES


#: Global display / colour order across model families.  Pythia first
#: (smallest -> largest), then Qwen3-Base, then post-trained Qwen3, then
#: Ministral-3 (base, reasoning, instruct).  The evaluator writes
#: ``model_size`` as the short Pythia size ("160m") or the full family
#: label ("qwen3-base-4b" / "ministral3-instruct-3b"), so we match on both.
_KNOWN_SIZE_ORDER: tuple[str, ...] = (
    tuple(PYTHIA_SIZES)
    + tuple(f"qwen3-base-{s}" for s in QWEN3_BASE_SIZES)
    + tuple(f"qwen3-{s}" for s in QWEN3_SIZES)
    + tuple(f"ministral3-{v}-{s}" for v in MINISTRAL3_VARIANTS for s in MINISTRAL3_SIZES)
)


def _size_rank(size: str) -> int:
    """Sort key for a model-size label; unknown labels sort last (alphabetical)."""
    if size in _KNOWN_SIZE_ORDER:
        return _KNOWN_SIZE_ORDER.index(size)
    return len(_KNOWN_SIZE_ORDER)


def _order_sizes(values: set[str]) -> list[str]:
    known = [s for s in _KNOWN_SIZE_ORDER if s in values]
    extras = sorted(values - set(known))
    return known + extras


def _size_label(size: str) -> str:
    """Legend label for a size: Pythia sizes get a 'Pythia-' prefix; other
    families already carry their family name in the label string."""
    if size in PYTHIA_SIZES:
        return f"Pythia-{size}"
    return size


#: Display order for the conflict-position panels.  Sorted recency-first
#: (``tail`` is closest to the confict/furthest from the question, ``front`` is furthest from the confict/closest to the question) so the
#: left-to-right reading mirrors the recency hypothesis (§2 Finding 5).
#: Positions not in this tuple are appended alphabetically.
_POSITION_DISPLAY_ORDER: tuple[str, ...] = ("tail", "middle", "front")


def _order_positions(values: set[str]) -> list[str]:
    ordered = [p for p in _POSITION_DISPLAY_ORDER if p in values]
    extras = sorted(values - set(ordered))
    return ordered + extras


#: Paper-facing filler names that map to on-disk ``filler_kind`` values.
_FILLER_DATA_ALIASES: dict[str, str] = {
    "prose": "varied_prose",
}


def _parse_filler_labels(spec: str | None) -> dict[str, str]:
    """Parse ``key=label`` pairs (comma-separated) for panel titles."""
    if not spec:
        return {}
    out: dict[str, str] = {}
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise SystemExit(
                f"--filler-labels entries must be key=label, got {part!r}"
            )
        key, label = part.split("=", 1)
        out[key.strip()] = label.strip()
    return out


def _resolve_fillers(
    requested: list[str] | None,
    available: set[str],
    filler_labels: dict[str, str],
) -> tuple[list[str], dict[str, str]]:
    """Return ordered data filler kinds and display titles."""
    if requested:
        data_kinds: list[str] = []
        display: dict[str, str] = {}
        for name in requested:
            data_kind = _FILLER_DATA_ALIASES.get(name, name)
            if data_kind not in available:
                raise SystemExit(
                    f"Filler {name!r} (data key {data_kind!r}) not in loaded "
                    f"conditions; available: {sorted(available)}"
                )
            data_kinds.append(data_kind)
            display[data_kind] = filler_labels.get(name, filler_labels.get(data_kind, name))
        return data_kinds, display

    data_kinds = sorted(available)
    display = {
        fk: filler_labels.get(fk, fk)
        for fk in data_kinds
    }
    for alias, data_kind in _FILLER_DATA_ALIASES.items():
        if data_kind in display and alias in filler_labels:
            display[data_kind] = filler_labels[alias]
    return data_kinds, display


def _load_jsonl(path: Path) -> tuple[dict, list[dict]]:
    meta: dict = {}
    records: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
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


def _aggregate(records: list[dict], metric: str) -> dict:
    n = len(records)
    if n == 0:
        return {"n": 0, "p_memorized": float("nan"),
                "p_in_context": float("nan"), "p_other": float("nan")}
    mem_key = f"memorized_{metric}"
    ctx_key = f"in_context_{metric}"
    mem = 0
    ctx = 0
    for r in records:
        block = r.get("substitution_conflict") or {}
        if block.get(mem_key):
            mem += 1
        if block.get(ctx_key):
            ctx += 1
    other = n - mem - ctx
    return {
        "n": n,
        "p_memorized": mem / n,
        "p_in_context": ctx / n,
        "p_other": other / n,
    }


def _detect_metric(records: list[dict]) -> str:
    counts = {"gen": 0, "lp": 0}
    for r in records:
        block = r.get("substitution_conflict") or {}
        for m in counts:
            if block.get(f"memorized_{m}") or block.get(f"in_context_{m}"):
                counts[m] += 1
                break
    if counts["lp"] >= counts["gen"]:
        return "lp"
    return "gen"


def _color_map(sizes: list[str]):
    import matplotlib.pyplot as plt

    if not sizes:
        return {}
    cmap = plt.cm.viridis
    n = max(len(sizes) - 1, 1)
    return {s: cmap(i / n) for i, s in enumerate(sizes)}


#: Approximate token length of native ParaConflict ``Coherent Conflict``.
_NATIVE_COHERENT_LENGTH = 128


def _load_native_sub_coh_rates(
    paths: list[Path],
    metric: str,
    *,
    category: str | None = None,
) -> dict[str, tuple[float, float]]:
    """Map model size -> (P(ctx|Sub), P(ctx|Coh)) on native ParaConflict."""
    ctx_k = f"in_context_{metric}"
    out: dict[str, tuple[float, float]] = {}
    for path in paths:
        meta, records = _load_jsonl(path)
        size = meta.get("model_size") or meta.get("label")
        if not size or not records:
            continue
        if category:
            records = [r for r in records if r.get("category") == category]
        if not records:
            continue
        n = len(records)
        sub = sum(
            1 for r in records
            if (r.get("substitution_conflict") or {}).get(ctx_k)
        )
        coh = sum(
            1 for r in records
            if (r.get("coherent_conflict") or {}).get(ctx_k)
        )
        out[size] = (sub / n, coh / n)
    return out


def _native_coh_mean_pctx(
    native_rates: dict[str, tuple[float, float]],
    sizes: list[str],
) -> float | None:
    values = [
        coh for size in sizes
        if (rates := native_rates.get(size)) is not None
        for _, coh in [rates]
    ]
    if not values:
        return None
    return statistics.mean(values)


def _draw_native_coh_mean_refs(
    ax,
    mean_p_coh: float,
    *,
    x_ref: float = _NATIVE_COHERENT_LENGTH,
) -> None:
    ax.axhline(
        mean_p_coh, color="0.45", ls="--", lw=1.2, alpha=0.85, zorder=2,
    )
    ax.axvline(
        x_ref, color="0.45", ls="--", lw=1.2, alpha=0.85, zorder=2,
    )


def _bottom_row_axes(axes) -> list:
    """Return visible axes on the lowest subplot row."""
    import numpy as np

    arr = np.atleast_2d(axes)
    for row in range(arr.shape[0] - 1, -1, -1):
        row_axes = [ax for ax in arr[row] if ax.get_visible()]
        if row_axes:
            return row_axes
    return [ax for ax in axes.flat if ax.get_visible()]


def _tick_label_bottom(fig, axes) -> float:
    """Lowest figure-y extent of x tick labels on the bottom subplot row."""
    renderer = fig.canvas.get_renderer()
    visible = [ax for ax in axes.flat if ax.get_visible()]
    tick_bottom = min(ax.get_position().y0 for ax in visible)
    for ax in _bottom_row_axes(axes):
        bb = ax.xaxis.get_tightbbox(renderer)
        if bb is not None:
            tick_bottom = min(
                tick_bottom,
                bb.transformed(fig.transFigure.inverted()).y0,
            )
    return tick_bottom


def _place_bottom_labels(
    fig,
    axes,
    xlabel: str,
    *,
    footnote: str | None = None,
) -> None:
    """Place shared xlabel (and optional footnote) below x tick labels."""
    import matplotlib.pyplot as plt

    visible = [ax for ax in axes.flat if ax.get_visible()]
    if not visible:
        fig.supxlabel(xlabel)
        return

    reserve = 0.13 if footnote else 0.08
    fig.tight_layout(rect=(0.03, reserve, 1, 0.95))
    fig.canvas.draw()

    tick_bottom = _tick_label_bottom(fig, axes)
    xlabel_artist = fig.text(
        0.5, tick_bottom - 0.020, xlabel,
        ha="center", va="top",
        transform=fig.transFigure,
        fontsize=plt.rcParams["axes.labelsize"],
    )
    footnote_artist = None
    if footnote:
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        xbb = xlabel_artist.get_window_extent(renderer).transformed(
            fig.transFigure.inverted(),
        )
        footnote_artist = fig.text(
            0.5, xbb.y0 - 0.014, footnote,
            ha="center", va="top", fontsize=7, color="0.35",
            transform=fig.transFigure,
        )

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    label_bottom = reserve
    for artist in (xlabel_artist, footnote_artist):
        if artist is None:
            continue
        bb = artist.get_window_extent(renderer).transformed(
            fig.transFigure.inverted(),
        )
        label_bottom = min(label_bottom, bb.y0)
    if label_bottom < 0.008:
        fig.subplots_adjust(bottom=reserve + (0.008 - label_bottom))


def _row_x(row: dict, x_axis: str) -> float | None:
    """X-coordinate for a row given the chosen axis.

    ``length``: filler tokens between conflict and question (default).
    ``lrel``:   relative input length = realised median prompt tokens /
                context window (Veseli et al. 2025).  Falls back to the
                target-length L_rel, then to None when no context window
                was recorded for the condition.
    """
    if x_axis == "lrel":
        v = row.get("l_rel_median_prompt")
        if v is None:
            v = row.get("l_rel_target")
        return v
    return row["length_tokens_target"]


def _plot_length_sweep(
    rows: list[dict],
    out_path: Path,
    metric: str,
    x_axis: str = "length",
    *,
    compact_rate: str | None = None,
    filler_kinds: list[str] | None = None,
    filler_display: dict[str, str] | None = None,
    native_sub_coh_rates: dict[str, tuple[float, float]] | None = None,
    native_ref_length: float = _NATIVE_COHERENT_LENGTH,
    native_sub_coh_category: str | None = None,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    base = [
        r for r in rows
        if r["conflict_position"] == "tail" and r["n_copies"] == 1
    ]
    available_fillers = {r["filler_kind"] for r in base}
    filler_kinds, display_names = _resolve_fillers(
        filler_kinds,
        available_fillers,
        filler_display or {},
    )
    sizes_present = _order_sizes({r["size"] for r in base})
    if not filler_kinds or not sizes_present:
        print("No length-sweep rows to plot (need position=tail, n_copies=1).")
        return

    cmap = _color_map(sizes_present)

    xlabel = (
        "Relative input length  L_rel = prompt tokens / context window"
        if x_axis == "lrel"
        else "Filler tokens between conflict and question"
    )

    import math

    rate_fields = {
        "mem": ("p_memorized", "P(memorized answer)", "P(memorized)"),
        "ctx": ("p_in_context", "P(in-context answer)", "P(in-context)"),
    }

    if compact_rate:
        field, ylabel, title_rate = rate_fields[compact_rate]
        n_cols = min(3, len(filler_kinds))
        n_rows = math.ceil(len(filler_kinds) / n_cols)
        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(4.2 * n_cols, 3.8 * n_rows),
            sharex=True, sharey=True, squeeze=False,
        )
        show_native_refs = (
            bool(native_sub_coh_rates)
            and compact_rate == "ctx"
            and x_axis == "length"
        )
        mean_native_p_coh = (
            _native_coh_mean_pctx(native_sub_coh_rates, sizes_present)
            if show_native_refs and native_sub_coh_rates
            else None
        )
        for idx, fk in enumerate(filler_kinds):
            row, col = divmod(idx, n_cols)
            ax = axes[row][col]
            for size in sizes_present:
                pts = [r for r in base if r["filler_kind"] == fk and r["size"] == size]
                pts = [r for r in pts if _row_x(r, x_axis) is not None]
                pts.sort(key=lambda r: _row_x(r, x_axis))
                if not pts:
                    continue
                xs = [_row_x(r, x_axis) for r in pts]
                ax.plot(
                    xs, [r[field] for r in pts], "-o",
                    color=cmap[size], label=_size_label(size), lw=1.7,
                )
            ax.set_title(f"filler = {display_names[fk]}")
            ax.set_ylim(0, 1)
            if x_axis == "lrel":
                ax.axvline(0.5, color="0.4", ls="--", lw=1.0, alpha=0.7)
            ax.grid(True, alpha=0.25)
            if mean_native_p_coh is not None:
                _draw_native_coh_mean_refs(
                    ax, mean_native_p_coh, x_ref=native_ref_length,
                )
        if filler_kinds:
            if n_rows == 1 and n_cols >= 3:
                mid = n_cols // 2
                handles, labels = axes[0][0].get_legend_handles_labels()
                axes[0][mid].legend(
                    handles, labels,
                    loc="upper center",
                    ncol=2,
                    fontsize=8,
                    framealpha=0.85,
                )
            else:
                axes[0][0].legend(loc="best", fontsize=8, framealpha=0.85)
        for idx in range(len(filler_kinds), n_rows * n_cols):
            row, col = divmod(idx, n_cols)
            axes[row][col].set_visible(False)
        for ax in axes.flat:
            if ax.get_visible():
                ax.label_outer()
        fig.supylabel(ylabel)
        fig.suptitle(f"{title_rate} vs. filler length")
        footnote = None
        if mean_native_p_coh is not None:
            cat_note = (
                native_sub_coh_category
                if native_sub_coh_category
                else "all six categories"
            )
            footnote = (
                f"Dashed lines: mean native ParaConflict Coherent Conflict "
                f"$P(\\mathrm{{ctx}})={mean_native_p_coh:.2f}$ at "
                f"$L={int(native_ref_length)}$ ({cat_note})"
            )
        _place_bottom_labels(fig, axes, xlabel, footnote=footnote)
    else:
        fig, axes = plt.subplots(
            2, len(filler_kinds),
            figsize=(5.5 * len(filler_kinds), 8),
            sharex=True, sharey=True, squeeze=False,
        )
        for col, fk in enumerate(filler_kinds):
            ax_ctx = axes[0][col]
            ax_mem = axes[1][col]
            for size in sizes_present:
                pts = [r for r in base if r["filler_kind"] == fk and r["size"] == size]
                pts = [r for r in pts if _row_x(r, x_axis) is not None]
                pts.sort(key=lambda r: _row_x(r, x_axis))
                if not pts:
                    continue
                xs = [_row_x(r, x_axis) for r in pts]
                ax_ctx.plot(
                    xs, [r["p_in_context"] for r in pts], "-o",
                    color=cmap[size], label=_size_label(size), lw=1.7,
                )
                ax_mem.plot(
                    xs, [r["p_memorized"] for r in pts], "-o",
                    color=cmap[size], label=_size_label(size), lw=1.7,
                )
            ax_ctx.set_title(f"filler = {display_names[fk]}")
            ax_ctx.set_ylim(0, 1)
            ax_mem.set_ylim(0, 1)
            if x_axis == "lrel":
                ax_ctx.axvline(0.5, color="0.4", ls="--", lw=1.0, alpha=0.7)
                ax_mem.axvline(0.5, color="0.4", ls="--", lw=1.0, alpha=0.7)
            ax_ctx.grid(True, alpha=0.25)
            ax_mem.grid(True, alpha=0.25)
            if col == 0:
                ax_ctx.legend(loc="best", fontsize=8, framealpha=0.85)

        for ax in axes.flat:
            ax.label_outer()
        axes[0][0].set_ylabel("P(in-context answer)")
        axes[1][0].set_ylabel("P(memorized answer)")
        fig.suptitle(
            f"Capitals cross-product: memorized vs in-context "
            f"across model sizes ({metric}-argmax classifier)"
        )
        _place_bottom_labels(fig, axes, xlabel)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"Wrote {out_path}")


def _plot_position_sweep(rows: list[dict], out_path: Path, metric: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pos_rows = [r for r in rows if r["n_copies"] == 1]
    positions = _order_positions({r["conflict_position"] for r in pos_rows})
    if len(positions) < 2:
        return

    lengths = sorted(
        L
        for L in {r["length_tokens_target"] for r in pos_rows}
        if sum(
            1
            for p in positions
            if any(
                r["length_tokens_target"] == L and r["conflict_position"] == p
                for r in pos_rows
            )
        )
        >= 2
    )
    fillers = sorted({r["filler_kind"] for r in pos_rows})
    sizes_present = _order_sizes({r["size"] for r in pos_rows})
    cmap = _color_map(sizes_present)

    fig, axes = plt.subplots(
        len(fillers), len(lengths),
        figsize=(3.6 * max(len(lengths), 1), 3.2 * max(len(fillers), 1) + 0.5),
        sharex=True, sharey=True, squeeze=False,
    )
    for r_idx, fk in enumerate(fillers):
        for c_idx, L in enumerate(lengths):
            ax = axes[r_idx][c_idx]
            x_idx = list(range(len(positions)))
            for size in sizes_present:
                ys = []
                for pos in positions:
                    matches = [
                        r for r in pos_rows
                        if r["size"] == size and r["filler_kind"] == fk
                        and r["length_tokens_target"] == L
                        and r["conflict_position"] == pos
                    ]
                    if matches:
                        ys.append(matches[0]["p_in_context"])
                    else:
                        ys.append(float("nan"))
                ax.plot(x_idx, ys, "-o", color=cmap[size],
                        label=_size_label(size), lw=1.5)
            ax.set_xticks(x_idx)
            ax.set_xticklabels(positions)
            ax.set_ylim(0, 1)
            ax.grid(True, alpha=0.25)
            if r_idx == 0:
                ax.set_title(f"L={L} tok")
            if c_idx == 0:
                ax.set_ylabel(f"{fk}\nP(in-context)")
            if r_idx == len(fillers) - 1:
                ax.set_xlabel("Conflict position")
            if r_idx == 0 and c_idx == 0:
                ax.legend(loc="best", fontsize=7, framealpha=0.85)

    fig.suptitle("Position sweep: P(in-context) by conflict location")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def _plot_copies_sweep(rows: list[dict], out_path: Path, metric: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rep_rows = [r for r in rows if r["conflict_position"] == "tail"]
    copies_values = sorted({r["n_copies"] for r in rep_rows})
    if len(copies_values) < 2:
        return

    lengths = sorted({r["length_tokens_target"] for r in rep_rows})
    fillers = sorted({r["filler_kind"] for r in rep_rows})
    sizes_present = _order_sizes({r["size"] for r in rep_rows})
    cmap = _color_map(sizes_present)

    fig, axes = plt.subplots(
        len(fillers), len(lengths),
        figsize=(3.6 * max(len(lengths), 1), 3.2 * max(len(fillers), 1) + 0.5),
        sharex=True, sharey=True, squeeze=False,
    )
    for r_idx, fk in enumerate(fillers):
        for c_idx, L in enumerate(lengths):
            ax = axes[r_idx][c_idx]
            for size in sizes_present:
                pts = [
                    r for r in rep_rows
                    if r["size"] == size and r["filler_kind"] == fk
                    and r["length_tokens_target"] == L
                ]
                pts.sort(key=lambda r: r["n_copies"])
                if not pts:
                    continue
                xs = [r["n_copies"] for r in pts]
                ys = [r["p_in_context"] for r in pts]
                ax.plot(xs, ys, "-o", color=cmap[size],
                        label=_size_label(size), lw=1.5)
            ax.set_ylim(0, 1)
            ax.grid(True, alpha=0.25)
            if r_idx == 0:
                ax.set_title(f"L={L} tok")
            if c_idx == 0:
                ax.set_ylabel(f"{fk}\nP(in-context)")
            if r_idx == len(fillers) - 1:
                ax.set_xlabel("# conflict copies")
            if r_idx == 0 and c_idx == 0:
                ax.legend(loc="best", fontsize=7, framealpha=0.85)

    fig.suptitle("Repetition sweep: P(in-context) by # conflict copies")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate and plot the capitals context-length sweep "
            "(P(in-context) and P(memorized) vs filler length, faceted by "
            "filler kind, with optional position and repetition panels)."
        )
    )
    parser.add_argument(
        "--per-sample-glob",
        type=str,
        default=None,
        help="Glob for per-sample JSONLs (relative to cwd).",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=None,
        help="Path to an existing _summary.json to replot directly.",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="output/capitals_ctx_sweep",
        help="Prefix for output files (.png, _summary.json).",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="auto",
        choices=("auto", "lp", "gen"),
        help="Classifier metric to plot: 'lp' (logprob argmax) or 'gen'.",
    )
    parser.add_argument(
        "--x-axis",
        type=str,
        default="length",
        choices=("length", "lrel"),
        help=(
            "X-axis for the main length figure.  'length' (default): filler "
            "tokens between conflict and question.  'lrel': relative input "
            "length = prompt tokens / context window (Veseli et al. 2025), "
            "read from each condition's recorded context_window.  Use 'lrel' "
            "to compare families with different context windows on one axis."
        ),
    )
    parser.add_argument(
        "--compact-rate",
        type=str,
        default=None,
        choices=("mem", "ctx"),
        help=(
            "Main length figure: plot only P(mem) or P(ctx) in a 2-row grid "
            "(fillers left-to-right, top row then bottom row)."
        ),
    )
    parser.add_argument(
        "--mem-only",
        action="store_true",
        help="Alias for --compact-rate mem.",
    )
    parser.add_argument(
        "--fillers",
        type=str,
        default=None,
        help=(
            "Comma-separated filler kinds to plot (default: all present).  "
            "The alias 'prose' maps to on-disk 'varied_prose'."
        ),
    )
    parser.add_argument(
        "--filler-labels",
        type=str,
        default=None,
        help="Comma-separated key=label pairs for panel titles.",
    )
    parser.add_argument(
        "--native-sub-coh-glob",
        type=str,
        default=None,
        help=(
            "Optional glob for native ParaConflict per-sample JSONLs with "
            "both substitution_conflict and coherent_conflict blocks.  "
            "Overlays size-matched reference markers at --native-ref-length."
        ),
    )
    parser.add_argument(
        "--native-ref-length",
        type=float,
        default=_NATIVE_COHERENT_LENGTH,
        help="X position for native Sub/Coh reference markers (default: 128).",
    )
    parser.add_argument(
        "--native-sub-coh-category",
        type=str,
        default=None,
        help=(
            "If set, restrict native Sub/Coh reference markers to this "
            "ParaConflict category (e.g. 'World Capital')."
        ),
    )
    args = parser.parse_args()
    compact_rate = args.compact_rate
    if args.mem_only:
        if compact_rate and compact_rate != "mem":
            raise SystemExit("Use only one of --compact-rate and --mem-only.")
        compact_rate = "mem"
    filler_kinds = None
    if args.fillers:
        filler_kinds = [f.strip() for f in args.fillers.split(",") if f.strip()]
    filler_labels = _parse_filler_labels(args.filler_labels)

    if args.summary_json:
        print(f"Loading precomputed summary from {args.summary_json}")
        data = json.loads(args.summary_json.read_text(encoding="utf-8"))
        metric = args.metric if args.metric != "auto" else data.get("metric", "gen")
        rows = data["rows"]
    else:
        if not args.per_sample_glob:
            raise SystemExit("Must specify either --per-sample-glob or --summary-json")
        paths = sorted(Path().glob(args.per_sample_glob))
        if not paths:
            raise SystemExit(f"No files match {args.per_sample_glob!r}")

        by_cond: dict[tuple, list[dict]] = defaultdict(list)
        meta_by_cond: dict[tuple, dict] = {}
        print(f"Loading {len(paths)} files matching {args.per_sample_glob!r}")

        detected_metric = None
        for p in paths:
            meta, records = _load_jsonl(p)
            size = meta.get("model_size")
            if not size:
                print(f"  skip {p.name}: no model_size in _meta")
                continue
            in_meta = (meta.get("input_meta") or {})
            # Backwards compat: fall back to old fields if present.
            length = in_meta.get("length_tokens_target", in_meta.get("length_tokens", 0))
            filler = in_meta.get("filler_kind", "prose")
            position = in_meta.get("conflict_position", "tail")
            n_copies = int(in_meta.get("n_copies", 1))
            median_tokens = in_meta.get("median_prompt_tokens")
            median_filler = in_meta.get("median_filler_tokens")
            context_window = in_meta.get("context_window")
            l_rel_target = in_meta.get("l_rel_target")
            l_rel_median_prompt = in_meta.get("l_rel_median_prompt")
            # Back-compat: if a context window was recorded but per-length L_rel
            # was not, derive it here from the filler/median-prompt token counts.
            if context_window and not l_rel_target:
                l_rel_target = length / context_window
            if context_window and l_rel_median_prompt is None and median_tokens:
                l_rel_median_prompt = median_tokens / context_window
            cond = (size, length, filler, position, n_copies)
            by_cond[cond].extend(records)
            meta_by_cond.setdefault(cond, {
                "median_prompt_tokens": median_tokens,
                "median_filler_tokens": median_filler,
                "context_window": context_window,
                "l_rel_target": l_rel_target,
                "l_rel_median_prompt": l_rel_median_prompt,
            })
            print(
                f"  {p.name}: size={size}, L={length} ({median_filler} med filler "
                f"tok / {median_tokens} med prompt tok), filler={filler}, "
                f"pos={position}, K={n_copies}, records={len(records)}"
            )

            if detected_metric is None and records:
                detected_metric = _detect_metric(records)

        if not by_cond:
            raise SystemExit("No valid per-sample files found.")

        metric = args.metric
        if metric == "auto":
            metric = detected_metric or "lp"
    print(f"Classifier metric: {metric}")

    native_sub_coh_rates: dict[str, tuple[float, float]] | None = None
    if args.native_sub_coh_glob:
        native_paths = sorted(Path().glob(args.native_sub_coh_glob))
        if not native_paths:
            raise SystemExit(
                f"No files match --native-sub-coh-glob {args.native_sub_coh_glob!r}"
            )
        native_sub_coh_rates = _load_native_sub_coh_rates(
            native_paths,
            metric,
            category=args.native_sub_coh_category,
        )
        cat_msg = (
            f", category={args.native_sub_coh_category!r}"
            if args.native_sub_coh_category
            else ""
        )
        print(
            f"Loaded native Sub/Coh references for {len(native_sub_coh_rates)} sizes"
            f"{cat_msg}"
        )

    if not args.summary_json:
        rows = []
        for (size, length, filler, position, n_copies), recs in by_cond.items():
            agg = _aggregate(recs, metric)
            row = {
                "size": size,
                "length_tokens_target": length,
                "filler_kind": filler,
                "conflict_position": position,
                "n_copies": n_copies,
                **agg,
                **meta_by_cond.get((size, length, filler, position, n_copies), {}),
            }
            rows.append(row)

        rows.sort(key=lambda r: (
            _size_rank(r["size"]),
            r["size"],
            r["filler_kind"],
            r["conflict_position"],
            r["n_copies"],
            r["length_tokens_target"],
        ))

        out_prefix = Path(args.output_prefix)
        out_prefix.parent.mkdir(parents=True, exist_ok=True)
        # If the user gave a prefix under a `plots/` directory (the
        # experiments/exp<NN>_*/ convention), automatically redirect the
        # summary JSON to the parallel `summaries/` directory.
        if out_prefix.parent.name == "plots":
            summaries_dir = out_prefix.parent.parent / "summaries"
            summaries_dir.mkdir(parents=True, exist_ok=True)
            summary_path = summaries_dir / (out_prefix.name + "_summary.json")
        else:
            summary_path = Path(out_prefix.as_posix() + "_summary.json")
        summary_path.write_text(json.dumps(
            {"metric": metric, "rows": rows}, indent=2
        ))
        print(f"Wrote {summary_path}")
    else:
        out_prefix = Path(args.output_prefix)
        out_prefix.parent.mkdir(parents=True, exist_ok=True)

    _plot_length_sweep(
        rows, Path(out_prefix.as_posix() + "_length.png"), metric,
        x_axis=args.x_axis,
        compact_rate=compact_rate,
        filler_kinds=filler_kinds,
        filler_display=filler_labels,
        native_sub_coh_rates=native_sub_coh_rates,
        native_ref_length=args.native_ref_length,
        native_sub_coh_category=args.native_sub_coh_category,
    )
    _plot_position_sweep(rows, Path(out_prefix.as_posix() + "_position.png"), metric)
    _plot_copies_sweep(rows, Path(out_prefix.as_posix() + "_copies.png"), metric)


if __name__ == "__main__":
    main()
