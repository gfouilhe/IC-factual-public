"""Compute wordfreq Zipf scores for ParaConflict subjects and objects.

For each relation category in the ParaConflict test split, harvests unique
*subjects* (queried entities), *memorized answers* (gold objects from
``Answer``), and *distractors* (in-context objects from ``Distracted Token``),
then scores each string with the ``wordfreq`` package (same proxy as
``scripts/compute_subject_frequency.py`` and the training-frequency section of Methods).

Example::

    python scripts/compute_paraconflict_entity_frequency.py \\
        --output output/paraconflict_entity_frequency.json

    python scripts/export_paraconflict_freq_appendix_tex.py \\
        --frequency-json output/paraconflict_entity_frequency.json \\
        --output-tex paper/tables/paraconflict_entity_frequency.tex
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm

_STRIPPABLE_PREFIXES = ("the ", "The ")
_DEFAULT_DATASET_ID = "gaotang/ParaConflict"
_CANONICAL_CATEGORIES = (
    "World Capital",
    "Athlete Sport",
    "Book Author",
    "Company Headquarter",
    "Company Founder",
    "Official Language",
)


def _canonical_category(category: str) -> str:
    if category == "Athelete Sport":
        return "Athlete Sport"
    return category


def _normalize_for_logprob(text: str) -> str:
    s = text.strip()
    for prefix in _STRIPPABLE_PREFIXES:
        if s.startswith(prefix):
            return s[len(prefix) :]
    return s


def _wordfreq_proxy(name: str, lang: str = "en") -> dict:
    from wordfreq import word_frequency, zipf_frequency

    normalized = _normalize_for_logprob(name)
    z = zipf_frequency(normalized, lang)
    f = word_frequency(normalized, lang)
    words = [w for w in normalized.split() if w]
    if len(words) > 1:
        per_word = [zipf_frequency(w, lang) for w in words]
        z_min = min(per_word) if per_word else 0.0
    else:
        z_min = z
    return {
        "zipf": float(z),
        "zipf_min_word": float(z_min),
        "freq": float(f),
        "normalized": normalized,
    }


def _resolve_paraconflict_source() -> str | Path:
    assets_root = os.environ.get("IC_FACTUAL_ASSETS_ROOT")
    if assets_root:
        staged = Path(assets_root) / "datasets" / "paraconflict"
        if staged.exists():
            return staged
    override = os.environ.get("IC_FACTUAL_PARACONFLICT_PATH")
    if override and Path(override).exists():
        return Path(override)
    return _DEFAULT_DATASET_ID


def _load_paraconflict_rows(split: str = "test") -> list[dict]:
    from datasets import load_dataset

    source = _resolve_paraconflict_source()
    dataset = load_dataset(str(source), split=split)
    return [dict(row) for row in dataset]


def _harvest_entities(rows: list[dict]) -> dict[str, dict[str, set[str]]]:
    """Return {category: {role: {entity, ...}}}."""
    out: dict[str, dict[str, set[str]]] = defaultdict(
        lambda: {
            "subjects": set(),
            "memorized_answers": set(),
            "distractors": set(),
        }
    )
    for row in rows:
        category = _canonical_category(row.get("Category", ""))
        out[category]["subjects"].add(row["Subject"])
        answers = row["Answer"]
        if isinstance(answers, str):
            answers = [answers]
        for answer in answers:
            out[category]["memorized_answers"].add(answer)
        out[category]["distractors"].add(row["Distracted Token"])
    return out


def _score_entities(
    entities: dict[str, dict[str, set[str]]],
    *,
    lang: str,
) -> dict[str, dict[str, dict[str, dict]]]:
    scored: dict[str, dict[str, dict[str, dict]]] = {}
    tasks: list[tuple[str, str, str]] = []
    for category, roles in entities.items():
        for role, names in roles.items():
            for name in names:
                tasks.append((category, role, name))

    bucket: dict[str, dict[str, dict[str, dict]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for category, role, name in tqdm(tasks, desc="wordfreq"):
        info = _wordfreq_proxy(name, lang=lang)
        bucket[category][role][name] = info

    scored = {
        category: {
            role: dict(sorted(items.items(), key=lambda kv: (-kv[1]["zipf"], kv[0])))
            for role, items in roles.items()
        }
        for category, roles in bucket.items()
    }
    return scored


def _summary_stats(scored: dict) -> dict[str, dict[str, dict[str, float | int]]]:
    summary: dict[str, dict[str, dict[str, float | int]]] = {}
    for category in _CANONICAL_CATEGORIES:
        if category not in scored:
            continue
        summary[category] = {}
        for role, items in scored[category].items():
            zipfs = [info["zipf"] for info in items.values()]
            if not zipfs:
                continue
            summary[category][role] = {
                "n": len(zipfs),
                "zipf_mean": sum(zipfs) / len(zipfs),
                "zipf_min": min(zipfs),
                "zipf_median": sorted(zipfs)[len(zipfs) // 2],
                "zipf_max": max(zipfs),
            }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compute wordfreq Zipf for ParaConflict subjects and objects "
            "(memorized answers and distractors) per relation category."
        )
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output/paraconflict_entity_frequency.json",
        help="JSON output path.",
    )
    parser.add_argument(
        "--lang",
        type=str,
        default="en",
        help="wordfreq language code (default: en).",
    )
    args = parser.parse_args()

    rows = _load_paraconflict_rows()
    print(f"Loaded {len(rows)} ParaConflict rows")
    entities = _harvest_entities(rows)
    scored = _score_entities(entities, lang=args.lang)

    payload = {
        "_meta": {
            "proxy": "wordfreq",
            "lang": args.lang,
            "description": (
                "Per-entity wordfreq Zipf on ParaConflict test split. "
                "Roles: subjects (queried entities), memorized_answers "
                "(gold Answer strings), distractors (Distracted Token)."
            ),
            "roles": {
                "subjects": "Queried entity (Subject field).",
                "memorized_answers": "Gold object(s) from Answer.",
                "distractors": "In-context object from Distracted Token.",
            },
            "n_rows": len(rows),
            "summary": _summary_stats(scored),
        },
        "categories": scored,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
