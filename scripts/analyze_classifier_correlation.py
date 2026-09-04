"""Correlate logprob-argmax vs greedy-generative classifiers on capitals sweeps.

Pairs records when both classifiers are available on the same prompt:

* **same_run** — per_sample JSONLs with ``method=both`` (or generative
  records that also carry ``lp_answer``), plus exp00-style dual labels.
* **exp22_merged** — exp22 generative outputs joined to the matching
  historical logprob file (exp01 / exp10 / exp16) by filename.

Outputs a JSON summary with per-prompt agreement, Cohen's kappa, Pearson /
Spearman correlation on cell-level P(mem) and P(ctx), and per-filler
breakdowns.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path

FNAME_RE = re.compile(
    r"per_sample(?:_qwen3(?:-base)?)?_L(\d+)_(.+)_tail_K1_(.+)\.jsonl$"
)

LP_DIR_BY_FILLER: dict[str, str] = {
    "capital_noise": "experiments/exp01_context_length_sweep/per_sample",
    "capital_noise_false": "experiments/exp10_capital_noise_false/data",
    "varied_prose": "experiments/exp16_varied_and_related_prose/data",
    "wiki_country": "experiments/exp16_varied_and_related_prose/data",
    "wiki_distractor_country": "experiments/exp16_varied_and_related_prose/data",
    "wiki_unrelated_country": "experiments/exp16_varied_and_related_prose/data",
    "prose": "experiments/exp01_context_length_sweep/per_sample",
}

EXTRA_BOTH_GLOBS: tuple[str, ...] = (
    "experiments/exp04_1b_bare_prose_anomaly/data/per_sample_*.jsonl",
    "experiments/exp06_2.8b_capital_noise_method/data/per_sample_*.jsonl",
    "experiments/exp16_varied_and_related_prose/data_method_both/per_sample_*.jsonl",
    "experiments/exp18_register_vs_facts/data/per_sample_*.jsonl",
    "/work/m24047/m24047flhg/ic-factual/experiments_data/exp16_data_method_both/per_sample_*.jsonl",
    "/work/m24047/m24047flhg/ic-factual/experiments_data/exp18_register_vs_facts/per_sample_*.jsonl",
)

EXP22_GLOB = "experiments/exp22_generative_filler_sweep/data/per_sample_*.jsonl"


def _load_jsonl(path: Path) -> tuple[dict | None, list[dict]]:
    meta = None
    records: list[dict] = []
    with open(path, encoding="utf-8") as f:
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


def _label(sc: dict, metric: str) -> str:
    mem = bool(sc.get(f"memorized_{metric}"))
    ctx = bool(sc.get(f"in_context_{metric}"))
    if mem:
        return "mem"
    if ctx:
        return "ctx"
    return "other"


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    n = len(xs)
    if n < 2:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    denx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    deny = math.sqrt(sum((y - my) ** 2 for y in ys))
    if denx == 0 or deny == 0:
        return None
    return num / (denx * deny)


def _rank(vals: list[float]) -> list[float]:
    order = sorted(range(len(vals)), key=lambda i: vals[i])
    ranks = [0.0] * len(vals)
    i = 0
    while i < len(vals):
        j = i
        while j + 1 < len(vals) and vals[order[j + 1]] == vals[order[i]]:
            j += 1
        avg = 0.5 * (i + j) + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def _spearman(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2:
        return None
    return _pearson(_rank(xs), _rank(ys))


def _cohens_kappa(y1: list[str], y2: list[str]) -> float | None:
    if not y1:
        return None
    labels = sorted(set(y1) | set(y2))
    n = len(y1)
    conf = {a: {b: 0 for b in labels} for a in labels}
    for a, b in zip(y1, y2):
        conf[a][b] += 1
    p_o = sum(conf[l][l] for l in labels) / n
    p_e = 0.0
    for l in labels:
        row = sum(conf[l][b] for b in labels) / n
        col = sum(conf[a][l] for a in labels) / n
        p_e += row * col
    if math.isclose(1.0 - p_e, 0.0):
        return None
    return (p_o - p_e) / (1.0 - p_e)


def _pair_from_same_file(path: Path, meta: dict | None, records: list[dict], source: str):
    method = (meta or {}).get("method", "")
    rows = []
    for rec in records:
        sc = rec.get("substitution_conflict") or {}
        if "lp_answer" not in sc:
            continue
        if method != "both" and "gen" not in sc:
            continue
        rows.append(
            {
                "country": rec.get("country"),
                "distractor": rec.get("distractor"),
                "lp": _label(sc, "lp"),
                "gen": _label(sc, "gen"),
                "source": source,
                "file": str(path),
            }
        )
    return rows


def _pair_exp22_with_lp(gen_path: Path, lp_path: Path):
    _, gen_records = _load_jsonl(gen_path)
    _, lp_records = _load_jsonl(lp_path)
    lp_by_key = {(r.get("country"), r.get("distractor")): r for r in lp_records}
    rows = []
    for rec in gen_records:
        key = (rec.get("country"), rec.get("distractor"))
        lp_rec = lp_by_key.get(key)
        if lp_rec is None:
            continue
        gsc = rec.get("substitution_conflict") or {}
        lsc = lp_rec.get("substitution_conflict") or {}
        if "gen" not in gsc or "lp_answer" not in lsc:
            continue
        rows.append(
            {
                "country": rec.get("country"),
                "distractor": rec.get("distractor"),
                "lp": _label(lsc, "lp"),
                "gen": _label(gsc, "gen"),
                "source": "exp22_merged",
                "file": str(gen_path),
            }
        )
    return rows


def _summarize(rows: list[dict], cell_key_fn) -> dict:
    if not rows:
        return {"n_prompts": 0}

    n = len(rows)
    lp_labels = [r["lp"] for r in rows]
    gen_labels = [r["gen"] for r in rows]

    mem_agree = sum(1 for a, b in zip(lp_labels, gen_labels) if (a == "mem") == (b == "mem"))
    ctx_agree = sum(1 for a, b in zip(lp_labels, gen_labels) if (a == "ctx") == (b == "ctx"))
    exact_agree = sum(1 for a, b in zip(lp_labels, gen_labels) if a == b)

    cells: dict[tuple, dict[str, int]] = defaultdict(lambda: {"n": 0, "mem_lp": 0, "mem_gen": 0, "ctx_lp": 0, "ctx_gen": 0})
    for r in rows:
        ck = cell_key_fn(r)
        cells[ck]["n"] += 1
        cells[ck]["mem_lp"] += r["lp"] == "mem"
        cells[ck]["mem_gen"] += r["gen"] == "mem"
        cells[ck]["ctx_lp"] += r["lp"] == "ctx"
        cells[ck]["ctx_gen"] += r["gen"] == "ctx"

    p_mem_lp = []
    p_mem_gen = []
    p_ctx_lp = []
    p_ctx_gen = []
    for c in cells.values():
        if c["n"] == 0:
            continue
        p_mem_lp.append(c["mem_lp"] / c["n"])
        p_mem_gen.append(c["mem_gen"] / c["n"])
        p_ctx_lp.append(c["ctx_lp"] / c["n"])
        p_ctx_gen.append(c["ctx_gen"] / c["n"])

    by_filler: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        m = FNAME_RE.match(Path(r["file"]).name)
        by_filler[m.group(2) if m else "unknown"].append(r)

    filler_rates = {}
    for f, rs in sorted(by_filler.items()):
        if not rs:
            continue
        lp_l = [x["lp"] for x in rs]
        gen_l = [x["gen"] for x in rs]
        filler_rates[f] = {
            "n_prompts": len(rs),
            "agreement_exact": sum(a == b for a, b in zip(lp_l, gen_l)) / len(rs),
            "agreement_mem": sum((a == "mem") == (b == "mem") for a, b in zip(lp_l, gen_l)) / len(rs),
            "agreement_ctx": sum((a == "ctx") == (b == "ctx") for a, b in zip(lp_l, gen_l)) / len(rs),
            "cohen_kappa_3class": _cohens_kappa(lp_l, gen_l),
        }

    return {
        "n_prompts": n,
        "n_cells": len(cells),
        "rates": {
            "agreement_exact": exact_agree / n,
            "agreement_mem": mem_agree / n,
            "agreement_ctx": ctx_agree / n,
        },
        "cohen_kappa_3class": _cohens_kappa(lp_labels, gen_labels),
        "cohen_kappa_mem": _cohens_kappa(
            ["mem" if x == "mem" else "not_mem" for x in lp_labels],
            ["mem" if x == "mem" else "not_mem" for x in gen_labels],
        ),
        "cohen_kappa_ctx": _cohens_kappa(
            ["ctx" if x == "ctx" else "not_ctx" for x in lp_labels],
            ["ctx" if x == "ctx" else "not_ctx" for x in gen_labels],
        ),
        "cell_pearson_p_mem": _pearson(p_mem_lp, p_mem_gen),
        "cell_pearson_p_ctx": _pearson(p_ctx_lp, p_ctx_gen),
        "cell_spearman_p_mem": _spearman(p_mem_lp, p_mem_gen),
        "cell_spearman_p_ctx": _spearman(p_ctx_lp, p_ctx_gen),
        "lp_margin_mem_minus_ctx": (lp_labels.count("mem") - lp_labels.count("ctx")) / n,
        "gen_margin_mem_minus_ctx": (gen_labels.count("mem") - gen_labels.count("ctx")) / n,
        "by_filler": filler_rates,
    }


def _cell_key_from_file(r: dict) -> tuple:
    p = Path(r["file"])
    m = FNAME_RE.match(p.name)
    if m:
        return (int(m.group(1)), m.group(2), m.group(3))
    return (r.get("source"), p.name)


def _collect_rows() -> tuple[list[dict], list[dict], list[dict]]:
    same_run: list[dict] = []
    exp22_rows: list[dict] = []
    seen_same_run: set[str] = set()

    for pattern in EXTRA_BOTH_GLOBS:
        paths = (
            sorted(Path(pattern).parent.glob(Path(pattern).name))
            if not pattern.startswith("/")
            else sorted(Path("/").glob(pattern.lstrip("/")))
        )
        for path in paths:
            if not path.is_file():
                continue
            key = str(path.resolve())
            if key in seen_same_run:
                continue
            meta, records = _load_jsonl(path)
            rows = _pair_from_same_file(path, meta, records, "same_run")
            if rows:
                seen_same_run.add(key)
                same_run.extend(rows)

    # exp00-style dual-label generative baselines
    for path in sorted(Path("experiments").rglob("per_sample_capitals*.jsonl")):
        meta, records = _load_jsonl(path)
        if (meta or {}).get("method") != "generative":
            continue
        key = str(path.resolve())
        if key in seen_same_run:
            continue
        rows = _pair_from_same_file(path, meta, records, "same_run_baseline")
        if rows:
            seen_same_run.add(key)
            same_run.extend(rows)

    for gen_path in sorted(Path().glob(EXP22_GLOB)):
        m = FNAME_RE.match(gen_path.name)
        if not m:
            continue
        filler = m.group(2)
        lp_dir = LP_DIR_BY_FILLER.get(filler)
        if not lp_dir:
            continue
        lp_path = Path(lp_dir) / gen_path.name
        if not lp_path.is_file():
            continue
        exp22_rows.extend(_pair_exp22_with_lp(gen_path, lp_path))

    # Deduplicate combined: prefer same_run over exp22_merged for identical prompts
    combined: dict[tuple, dict] = {}
    for r in exp22_rows:
        combined[(r["country"], r["distractor"], _cell_key_from_file(r))] = r
    for r in same_run:
        combined[(r["country"], r["distractor"], _cell_key_from_file(r))] = r

    return same_run, exp22_rows, list(combined.values())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default="experiments/exp22_generative_filler_sweep/summaries/classifier_correlation_summary.json",
    )
    args = parser.parse_args()

    same_run, exp22_rows, combined = _collect_rows()
    out = {
        "_meta": {
            "description": "Correlation / agreement between logprob-argmax (lp) and greedy generative (gen) classifiers.",
            "label_sets": ["mem", "ctx", "other"],
        },
        "same_run": _summarize(same_run, _cell_key_from_file),
        "exp22_merged_with_historical_lp": _summarize(exp22_rows, _cell_key_from_file),
        "combined_deduped": _summarize(combined, _cell_key_from_file),
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"wrote {out_path}")
    for section in ("same_run", "exp22_merged_with_historical_lp", "combined_deduped"):
        s = out[section]
        if not s.get("n_prompts"):
            print(f"\n{section}: (empty)")
            continue
        print(f"\n{section}: n={s['n_prompts']:,} cells={s['n_cells']}")
        print(f"  exact agreement: {s['rates']['agreement_exact']:.4f}")
        print(f"  mem agreement:   {s['rates']['agreement_mem']:.4f}")
        print(f"  ctx agreement:   {s['rates']['agreement_ctx']:.4f}")
        print(f"  kappa (3-class): {s['cohen_kappa_3class']:.4f}")
        print(f"  kappa (mem):     {s['cohen_kappa_mem']:.4f}")
        print(f"  cell Pearson r (P_mem): {s['cell_pearson_p_mem']:.4f}")
        print(f"  cell Pearson r (P_ctx): {s['cell_pearson_p_ctx']:.4f}")


if __name__ == "__main__":
    main()
