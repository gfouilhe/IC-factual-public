"""Plot baseline factual recall accuracy without conflict vs model size across families.

Generates a 6-panel figure (one per ParaConflict relation category) showing
generative factual recall accuracy on clean prompts without any conflict.

Can render via matplotlib (if installed) or cairo fallback.

Usage::

    python scripts/plot_clean_factual_recall.py
    python scripts/plot_clean_factual_recall.py --metric lp
"""


import argparse
import json
import math
import sys
from pathlib import Path

# Add project root to sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

_CATEGORIES = [
    "World Capital",
    "Athlete Sport",
    "Book Author",
    "Company Headquarter",
    "Company Founder",
    "Official Language",
]

_FAMILY_STYLE = {
    "pythia": {"color": "#2171b5", "rgb": (0.13, 0.44, 0.71), "marker": "o", "label": "Pythia"},
    "qwen3": {"color": "#2ca02c", "rgb": (0.17, 0.63, 0.17), "marker": "s", "label": "Qwen3 (post-trained)"},
    "qwen3-base": {"color": "#98df8a", "rgb": (0.60, 0.87, 0.54), "marker": "D", "label": "Qwen3-Base"},
    "ministral3-instruct": {"color": "#00897b", "rgb": (0.0, 0.54, 0.48), "marker": "^", "label": "Ministral-3 Instruct"},
    "ministral3-reasoning": {"color": "#004d40", "rgb": (0.0, 0.30, 0.25), "marker": "v", "label": "Ministral-3 Reasoning"},
    "ministral3-base": {"color": "#6ec4bc", "rgb": (0.43, 0.77, 0.74), "marker": "P", "label": "Ministral-3 Base"},
    "gpt2": {"color": "#e377c2", "rgb": (0.89, 0.47, 0.76), "marker": "h", "label": "GPT-2"},
}

_FAMILY_ORDER = [
    "pythia",
    "gpt2",
    "qwen3-base",
    "qwen3",
    "ministral3-base",
    "ministral3-instruct",
    "ministral3-reasoning",
]

_FAMILY_X_JITTER = {
    "gpt2": 0.90,
    "pythia": 0.96,
    "qwen3-base": 1.0,
    "qwen3": 1.04,
    "ministral3-base": 0.94,
    "ministral3-instruct": 1.0,
    "ministral3-reasoning": 1.06,
}


def _load_summary(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _render_cairo(
    data: dict,
    out_paths,
    metric_key: str,
    metric_name: str,
    width: int = 2100,
    height: int = 1350,
) :
    import cairo

    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, width, height)
    ctx = cairo.Context(surface)

    # White background
    ctx.set_source_rgb(1.0, 1.0, 1.0)
    ctx.paint()

    # Layout parameters
    margin_left = 140
    margin_right = 60
    margin_top = 110
    margin_bottom = 140
    cols = 3
    rows = 2
    gap_x = 80
    gap_y = 90

    plot_w = (width - margin_left - margin_right - (cols - 1) * gap_x) / cols
    plot_h = (height - margin_top - margin_bottom - (rows - 1) * gap_y) / rows

    x_min, x_max = 0.1, 40.0
    log_x_min, log_x_max = math.log10(x_min), math.log10(x_max)
    y_min, y_max = 0.0, 1.05

    def to_screen(x_val: float, y_val: float, col_i: int, row_i: int) :
        px0 = margin_left + col_i * (plot_w + gap_x)
        py0 = margin_top + row_i * (plot_h + gap_y)
        lx = math.log10(max(x_val, x_min))
        sx = px0 + (lx - log_x_min) / (log_x_max - log_x_min) * plot_w
        sy = py0 + plot_h - (y_val - y_min) / (y_max - y_min) * plot_h
        return sx, sy

    # Title
    ctx.select_font_face("DejaVu Sans, Helvetica, Arial, Sans", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD)
    ctx.set_font_size(30)
    ctx.set_source_rgb(0.1, 0.1, 0.1)
    title_text = f"Baseline Factual Recall Without Conflict Across Models and Relations ({metric_name})"
    ext = ctx.text_extents(title_text)
    ctx.move_to((width - ext.width) / 2, 55)
    ctx.show_text(title_text)

    # Subplots
    for cat_idx, cat in enumerate(_CATEGORIES):
        row_i, col_i = divmod(cat_idx, cols)
        px0 = margin_left + col_i * (plot_w + gap_x)
        py0 = margin_top + row_i * (plot_h + gap_y)

        # Background panel & border
        ctx.set_source_rgb(0.98, 0.98, 0.98)
        ctx.rectangle(px0, py0, plot_w, plot_h)
        ctx.fill()

        # Grid lines (y)
        ctx.set_line_width(1.0)
        ctx.set_source_rgba(0.85, 0.85, 0.85, 0.8)
        for y_tick in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0):
            _, sy = to_screen(1.0, y_tick, col_i, row_i)
            ctx.move_to(px0, sy)
            ctx.line_to(px0 + plot_w, sy)
            ctx.stroke()

        # Grid lines (x: 0.1, 1, 10)
        for x_tick in (0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0):
            sx, _ = to_screen(x_tick, 0.5, col_i, row_i)
            is_major = x_tick in (0.1, 1.0, 10.0)
            ctx.set_line_width(1.2 if is_major else 0.6)
            ctx.set_source_rgba(0.80 if is_major else 0.90, 0.80 if is_major else 0.90, 0.80 if is_major else 0.90, 0.8)
            ctx.move_to(sx, py0)
            ctx.line_to(sx, py0 + plot_h)
            ctx.stroke()

        # Border
        ctx.set_line_width(1.5)
        ctx.set_source_rgb(0.2, 0.2, 0.2)
        ctx.rectangle(px0, py0, plot_w, plot_h)
        ctx.stroke()

        # Panel title
        ctx.select_font_face("DejaVu Sans, Helvetica, Arial, Sans", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD)
        ctx.set_font_size(20)
        ctx.set_source_rgb(0.15, 0.15, 0.15)
        ext = ctx.text_extents(cat)
        ctx.move_to(px0 + (plot_w - ext.width) / 2, py0 - 15)
        ctx.show_text(cat)

        # Ticks and tick labels (y-axis)
        ctx.select_font_face("DejaVu Sans, Helvetica, Arial, Sans", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL)
        ctx.set_font_size(14)
        for y_tick in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0):
            _, sy = to_screen(1.0, y_tick, col_i, row_i)
            if col_i == 0:
                lbl = f"{y_tick:.1f}"
                lext = ctx.text_extents(lbl)
                ctx.set_source_rgb(0.3, 0.3, 0.3)
                ctx.move_to(px0 - lext.width - 10, sy + 5)
                ctx.show_text(lbl)

        # Ticks and tick labels (x-axis)
        if row_i == rows - 1:
            for x_tick, lbl in [(0.1, "0.1"), (1.0, "1"), (10.0, "10")]:
                sx, _ = to_screen(x_tick, 0.0, col_i, row_i)
                lext = ctx.text_extents(lbl)
                ctx.set_source_rgb(0.3, 0.3, 0.3)
                ctx.move_to(sx - lext.width / 2, py0 + plot_h + 22)
                ctx.show_text(lbl)

        # Plot curves by family
        models = data["models"]
        for family in _FAMILY_ORDER:
            style = _FAMILY_STYLE[family]
            f_pts = []
            for m_key, m_val in models.items():
                if m_val["family"] != family:
                    continue
                pb = m_val.get("params_b")
                if pb is None:
                    continue
                acc = m_val["categories"][cat][metric_key]
                if math.isnan(acc):
                    continue
                jit = _FAMILY_X_JITTER.get(family, 1.0)
                f_pts.append((pb, pb * jit, acc, m_val["display_label"]))

            f_pts.sort(key=lambda t: t[0])
            if not f_pts:
                continue

            r, g, b = style["rgb"]
            ctx.set_source_rgb(r, g, b)
            ctx.set_line_width(2.2)

            # Draw lines
            first = True
            for _, jit_x, acc, _ in f_pts:
                sx, sy = to_screen(jit_x, acc, col_i, row_i)
                if first:
                    ctx.move_to(sx, sy)
                    first = False
                else:
                    ctx.line_to(sx, sy)
            ctx.stroke()

            # Draw markers
            marker = style["marker"]
            m_size = 5.5
            for _, jit_x, acc, _ in f_pts:
                sx, sy = to_screen(jit_x, acc, col_i, row_i)
                ctx.set_source_rgb(r, g, b)
                if marker == "o":
                    ctx.arc(sx, sy, m_size, 0, 2 * math.pi)
                    ctx.fill()
                elif marker == "s":
                    ctx.rectangle(sx - m_size, sy - m_size, m_size * 2, m_size * 2)
                    ctx.fill()
                elif marker == "D":
                    ctx.move_to(sx, sy - m_size * 1.2)
                    ctx.line_to(sx + m_size * 1.2, sy)
                    ctx.line_to(sx, sy + m_size * 1.2)
                    ctx.line_to(sx - m_size * 1.2, sy)
                    ctx.close_path()
                    ctx.fill()
                elif marker == "^":
                    ctx.move_to(sx, sy - m_size * 1.3)
                    ctx.line_to(sx + m_size * 1.2, sy + m_size * 0.9)
                    ctx.line_to(sx - m_size * 1.2, sy + m_size * 0.9)
                    ctx.close_path()
                    ctx.fill()
                elif marker == "v":
                    ctx.move_to(sx, sy + m_size * 1.3)
                    ctx.line_to(sx + m_size * 1.2, sy - m_size * 0.9)
                    ctx.line_to(sx - m_size * 1.2, sy - m_size * 0.9)
                    ctx.close_path()
                    ctx.fill()
                elif marker == "P":
                    arm = m_size * 1.2
                    th = m_size * 0.45
                    ctx.rectangle(sx - th, sy - arm, th * 2, arm * 2)
                    ctx.rectangle(sx - arm, sy - th, arm * 2, th * 2)
                    ctx.fill()
                elif marker == "h":
                    ctx.arc(sx, sy, m_size, 0, 2 * math.pi)
                    ctx.fill()

    # Shared X and Y axis labels
    ctx.select_font_face("DejaVu Sans, Helvetica, Arial, Sans", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD)
    ctx.set_font_size(22)
    ctx.set_source_rgb(0.2, 0.2, 0.2)

    xlabel = "Parameters (billions, log scale)"
    xext = ctx.text_extents(xlabel)
    ctx.move_to((width - xext.width) / 2, height - 70)
    ctx.show_text(xlabel)

    # Rotated Y-axis label
    ctx.save()
    ctx.translate(45, (height + margin_top - margin_bottom) / 2)
    ctx.rotate(-math.pi / 2)
    ylabel = "Factual Recall Accuracy: P(correct | clean prompt)"
    yext = ctx.text_extents(ylabel)
    ctx.move_to(-yext.width / 2, 0)
    ctx.show_text(ylabel)
    ctx.restore()

    # Legend at the bottom
    leg_y = height - 25
    ctx.select_font_face("DejaVu Sans, Helvetica, Arial, Sans", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL)
    ctx.set_font_size(15)

    # Compute total legend width
    item_widths = []
    for fam in _FAMILY_ORDER:
        st = _FAMILY_STYLE[fam]
        ext = ctx.text_extents(st["label"])
        item_widths.append(ext.width + 50)

    total_leg_w = sum(item_widths)
    cur_x = (width - total_leg_w) / 2

    for fam, iw in zip(_FAMILY_ORDER, item_widths):
        st = _FAMILY_STYLE[fam]
        r, g, b = st["rgb"]
        ctx.set_source_rgb(r, g, b)
        ctx.set_line_width(2.5)
        ctx.move_to(cur_x, leg_y - 5)
        ctx.line_to(cur_x + 24, leg_y - 5)
        ctx.stroke()
        ctx.arc(cur_x + 12, leg_y - 5, 5, 0, 2 * math.pi)
        ctx.fill()

        ctx.set_source_rgb(0.2, 0.2, 0.2)
        ctx.move_to(cur_x + 32, leg_y)
        ctx.show_text(st["label"])
        cur_x += iw

    for op in out_paths:
        op.parent.mkdir(parents=True, exist_ok=True)
        surface.write_to_png(str(op))
        print(f"Wrote plot to {op}")


def main() :
    parser = argparse.ArgumentParser(
        description="Plot baseline clean prompt factual recall vs model size."
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=Path("summaries/clean_factual_recall_summary.json"),
    )
    parser.add_argument(
        "--output-png",
        type=Path,
        default=Path("figures/clean_factual_recall.png"),
    )
    parser.add_argument(
        "--metric",
        choices=("gen", "lp"),
        default="gen",
        help="Metric to plot ('gen' = generative substring match, 'lp' = logprob argmax).",
    )
    args = parser.parse_args()

    data = _load_summary(args.summary_json)
    metric_key = "gen_acc" if args.metric == "gen" else "lp_acc"
    metric_name = "Generative Substring Match" if args.metric == "gen" else "Log-Probability Argmax"

    base_name = args.output_png.stem if args.metric == "gen" else f"{args.output_png.stem}_lp"
    out_paths = [
        Path(f"figures/{base_name}.png"),
        Path(f"paper/figures/{base_name}.png"),
    ]

    _render_cairo(data, out_paths, metric_key, metric_name)


if __name__ == "__main__":
    main()
