"""Build context-length sweep variants of ParaConflict relation prompts.

Extends the capitals context-sweep idea to all six ParaConflict categories.
Each output JSONL line is one (subject, distractor) prompt with a configurable
filler between the conflict clause and the clean continuation prefix.

Filler kinds mirror the capitals taxonomy (see experiments/exp25 README):

* ``prose`` -- semantically inert plant-cell paragraph (tiled to L tokens).
* ``varied_prose`` -- per-prompt topic-shuffled prose pool at L tokens.
* ``coherent_passage`` -- fluent passage extracted from ``Coherent Conflict``.
* ``coherent_full`` -- native ``Coherent Conflict`` field (anchor, no tiling).
* ``relation_noise`` -- true statements in the category template.
* ``relation_noise_false`` -- shuffled (false) pairings, same templates.
* ``passage_shuffled`` -- coherent passage with sentences permuted.
* ``passage_scrambled`` -- coherent passage with proper nouns/years scrambled.

Usage::

    python scripts/build_paraconflict_context_sweep.py \\
        --output experiments/exp25_coherent_motivates_sweep/data/input_L128_prose.jsonl \\
        --filler-kind prose --length-tokens 128 \\
        --categories "Official Language" "Athlete Sport"
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
from pathlib import Path

from ic_factual.config import PYTHIA_SIZES, resolve_model_size
from ic_factual.loaders import load_paraconflict, load_tokenizer
from ic_factual.paraconflict_decompose import (
    answers_list,
    assemble_prompt,
    canonical_category,
    decompose_row,
)

_BCCS_PATH = Path(__file__).resolve().parent / "build_capitals_context_sweep.py"
_spec = importlib.util.spec_from_file_location("build_capitals_context_sweep", _BCCS_PATH)
_bccs = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_bccs)

PROSE_SNIPPET = _bccs.PROSE_SNIPPET
_build_capital_noise_filler = _bccs._build_capital_noise_filler
_build_varied_prose_filler = _bccs._build_varied_prose_filler
_encode = _bccs._encode
_scramble_proper_nouns_and_dates = _bccs._scramble_proper_nouns_and_dates
_shuffle_sentences = _bccs._shuffle_sentences
_tile_to_n_tokens = _bccs._tile_to_n_tokens

ALL_CATEGORIES = (
    "World Capital",
    "Athlete Sport",
    "Book Author",
    "Company Headquarter",
    "Company Founder",
    "Official Language",
)

# ParaConflict typo alias
_CATEGORY_ALIASES = {
    "athlete sport": ("Athlete Sport", "Athelete Sport"),
    "athelete sport": ("Athlete Sport", "Athelete Sport"),
}

RELATION_NOISE_TEMPLATES: dict[str, str] = {
    "World Capital": "The capital of {subject} is {answer}. ",
    "Athlete Sport": "{subject} plays the sport of {answer}. ",
    "Book Author": "{book} was written by {author}. ",
    "Company Headquarter": "{company} is headquartered in {city}. ",
    "Company Founder": "{company} was founded by {person}. ",
    "Official Language": "The official language of {country} is {language}. ",
}

def _normalize_categories(categories: list[str] | None) -> list[str]:
    if not categories:
        return list(ALL_CATEGORIES)
    out: list[str] = []
    for cat in categories:
        low = cat.strip().lower()
        if low in _CATEGORY_ALIASES:
            out.append(_CATEGORY_ALIASES[low][0])
        else:
            out.append(cat.strip())
    return out


def _pool_pair(row: dict, category: str) -> tuple[str, str]:
    cat = canonical_category(category)
    subject = row["Subject"]
    answers = answers_list(row)
    gold = answers[0] if answers else ""
    if cat == "Book Author":
        return subject, gold
    if cat in ("Company Headquarter", "Company Founder"):
        return subject, gold
    return subject, gold


def _format_noise_stmt(
    category: str, left: str, right: str, template: str
) -> str:
    cat = canonical_category(category)
    if cat == "Book Author":
        return template.format(book=left, author=right)
    if cat == "Company Headquarter":
        return template.format(company=left, city=right)
    if cat == "Company Founder":
        return template.format(company=left, person=right)
    if cat == "Official Language":
        return template.format(country=left, language=right)
    if cat == "World Capital":
        return template.format(subject=left, answer=right)
    return template.format(subject=left, answer=right)


def _build_relation_noise_filler(
    n_tokens: int,
    tokenizer,
    pool: list[tuple[str, str]],
    exclude_left: set[str],
    exclude_right: set[str],
    category: str,
    seed: int,
    falsify: bool = False,
) -> str:
    if n_tokens <= 0:
        return ""
    template = RELATION_NOISE_TEMPLATES[canonical_category(category)]
    rng = random.Random(seed)
    candidates = [
        (left, right)
        for left, right in pool
        if left not in exclude_left and right not in exclude_right
    ]
    if not candidates:
        return ""
    if falsify and len(candidates) > 1:
        lefts = [a for a, _ in candidates]
        rights = [b for _, b in candidates]
        rights_shuffled = list(rights)
        attempts = 0
        while attempts < 32:
            rng.shuffle(rights_shuffled)
            if all(a != b for a, b in zip(rights, rights_shuffled)):
                break
            attempts += 1
        candidates = list(zip(lefts, rights_shuffled))
    rng.shuffle(candidates)

    out_parts: list[str] = []
    out_token_count = 0
    i = 0
    safety_budget = max(8 * n_tokens, 256)
    while out_token_count < n_tokens and safety_budget > 0:
        left, right = candidates[i % len(candidates)]
        stmt = _format_noise_stmt(category, left, right, template)
        out_parts.append(stmt)
        out_token_count += len(_encode(stmt, tokenizer))
        i += 1
        safety_budget -= 1
        if i % len(candidates) == 0:
            rng.shuffle(candidates)

    text = "".join(out_parts)
    ids = _encode(text, tokenizer)[:n_tokens]
    return tokenizer.decode(ids, skip_special_tokens=True)


def _build_passage_filler(
    passage: str,
    n_tokens: int,
    tokenizer,
    *,
    shuffled: bool = False,
    scrambled: bool = False,
    seed: int = 0,
) -> str:
    if n_tokens <= 0 or not passage:
        return ""
    text = passage
    if scrambled:
        text = _scramble_proper_nouns_and_dates(text, seed)
    if shuffled:
        text = _shuffle_sentences(text, seed)
    return _tile_to_n_tokens(text, n_tokens, tokenizer)


def build_records(
    rows: list[dict],
    *,
    filler_kind: str,
    length_tokens: int,
    tokenizer,
    pools_by_category: dict[str, list[tuple[str, str]]],
    shuffle_seed: int,
) -> list[dict]:
    out: list[dict] = []
    shared_prose = ""
    if filler_kind == "prose" and length_tokens > 0:
        shared_prose = _tile_to_n_tokens(PROSE_SNIPPET, length_tokens, tokenizer)

    for idx, row in enumerate(rows):
        category = canonical_category(row.get("Category", ""))
        subject = row["Subject"]
        answers = answers_list(row)
        distractor = row["Distracted Token"]
        conflict_clause, passage, continuation = decompose_row(row)
        seed = shuffle_seed + idx * 7919

        if filler_kind == "coherent_full":
            prompt = row["Coherent Conflict"].strip()
            filler_text = ""
            filler_tokens = 0
        else:
            if filler_kind == "prose":
                filler_text = shared_prose if length_tokens > 0 else ""
            elif filler_kind == "varied_prose":
                filler_text = (
                    _build_varied_prose_filler(length_tokens, tokenizer, seed)
                    if length_tokens > 0
                    else ""
                )
            elif filler_kind == "coherent_passage":
                filler_text = _build_passage_filler(
                    passage, length_tokens, tokenizer, seed=seed
                )
            elif filler_kind == "passage_shuffled":
                filler_text = _build_passage_filler(
                    passage, length_tokens, tokenizer, shuffled=True, seed=seed
                )
            elif filler_kind == "passage_scrambled":
                filler_text = _build_passage_filler(
                    passage, length_tokens, tokenizer, scrambled=True, seed=seed
                )
            elif filler_kind == "relation_noise":
                if category == "World Capital":
                    pool = pools_by_category[category]
                    filler_text = _build_capital_noise_filler(
                        length_tokens,
                        tokenizer,
                        pool,
                        exclude_countries={subject},
                        exclude_capitals={distractor, answers[0] if answers else ""},
                        seed=seed,
                        falsify=False,
                    )
                else:
                    pool = pools_by_category.get(category, [])
                    ex_left = {subject}
                    ex_right = {distractor}
                    if answers:
                        ex_right.add(answers[0])
                    filler_text = _build_relation_noise_filler(
                        length_tokens,
                        tokenizer,
                        pool,
                        exclude_left=ex_left,
                        exclude_right=ex_right,
                        category=category,
                        seed=seed,
                        falsify=False,
                    )
            elif filler_kind == "relation_noise_false":
                if category == "World Capital":
                    pool = pools_by_category[category]
                    filler_text = _build_capital_noise_filler(
                        length_tokens,
                        tokenizer,
                        pool,
                        exclude_countries={subject},
                        exclude_capitals={distractor, answers[0] if answers else ""},
                        seed=seed,
                        falsify=True,
                    )
                else:
                    pool = pools_by_category.get(category, [])
                    ex_left = {subject}
                    ex_right = {distractor}
                    if answers:
                        ex_right.add(answers[0])
                    filler_text = _build_relation_noise_filler(
                        length_tokens,
                        tokenizer,
                        pool,
                        exclude_left=ex_left,
                        exclude_right=ex_right,
                        category=category,
                        seed=seed,
                        falsify=True,
                    )
            else:
                raise ValueError(f"Unknown filler kind: {filler_kind}")

            prompt = assemble_prompt(conflict_clause, filler_text, continuation)
            filler_tokens = len(_encode(filler_text, tokenizer)) if filler_text else 0

        prompt_tokens = len(_encode(prompt, tokenizer))
        out.append(
            {
                "subject": subject,
                "answers": answers,
                "distractor": distractor,
                "category": category,
                "prompt": prompt,
                "conflict_clause": conflict_clause,
                "continuation": continuation,
                "filler_tokens": filler_tokens,
                "prompt_tokens": prompt_tokens,
                "coherent_passage_chars": len(passage),
            }
        )
    return out


def _build_pools(all_rows: list[dict]) -> dict[str, list[tuple[str, str]]]:
    pools: dict[str, list[tuple[str, str]]] = {}
    seen: dict[str, set[tuple[str, str]]] = {}
    for row in all_rows:
        cat = canonical_category(row.get("Category", ""))
        pair = _pool_pair(row, cat)
        if not pair[0] or not pair[1]:
            continue
        seen.setdefault(cat, set())
        if pair not in seen[cat]:
            seen[cat].add(pair)
            pools.setdefault(cat, []).append(pair)

    wc_rows = [
        r for r in all_rows
        if canonical_category(r.get("Category", "")) == "World Capital"
    ]
    if wc_rows:
        cap_map: dict[str, str] = {}
        for r in wc_rows:
            sub = r["Subject"]
            ans = answers_list(r)
            if ans:
                cap_map[sub] = ans[0]
            dist = r["Distracted Token"]
            if dist:
                cap_map.setdefault(f"__dist__{dist}", dist)
        pools["World Capital"] = sorted((c, k) for c, k in cap_map.items())
    return pools


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build ParaConflict context-length / filler sweep JSONLs."
    )
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument(
        "--filler-kind",
        type=str,
        required=True,
        choices=(
            "prose",
            "varied_prose",
            "coherent_passage",
            "coherent_full",
            "relation_noise",
            "relation_noise_false",
            "passage_shuffled",
            "passage_scrambled",
        ),
    )
    parser.add_argument(
        "--length-tokens",
        type=int,
        default=0,
        help="Target filler token count (0 = no filler between clause and continuation).",
    )
    parser.add_argument(
        "--categories",
        nargs="*",
        default=None,
        help="Subset of ParaConflict categories (default: all six).",
    )
    parser.add_argument("--model-size", type=str, default="160m", choices=list(PYTHIA_SIZES))
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--shuffle-seed", type=int, default=0)
    args = parser.parse_args()

    if args.filler_kind == "coherent_full" and args.length_tokens not in (0, None):
        print(
            "Note: coherent_full ignores --length-tokens; using native Coherent Conflict."
        )

    categories = _normalize_categories(args.categories)
    size = resolve_model_size(args.model_size)
    tokenizer = load_tokenizer(model_size=size)

    ds = load_paraconflict(split="test")
    wanted = set(categories) | {"Athelete Sport"}
    rows = [
        dict(r)
        for r in ds
        if canonical_category(r.get("Category", "")) in categories
        or r.get("Category") in wanted
    ]
    # filter strictly to requested canonical names
    rows = [r for r in rows if canonical_category(r.get("Category", "")) in categories]

    if args.shuffle_seed is not None:
        random.Random(args.shuffle_seed).shuffle(rows)
    if args.max_samples is not None:
        rows = rows[: args.max_samples]

    pools = _build_pools(list(ds))
    records = build_records(
        rows,
        filler_kind=args.filler_kind,
        length_tokens=args.length_tokens,
        tokenizer=tokenizer,
        pools_by_category=pools,
        shuffle_seed=args.shuffle_seed,
    )

    median_filler = 0
    if records:
        medians = sorted(r["filler_tokens"] for r in records)
        median_filler = medians[len(medians) // 2]
    median_prompt = 0
    if records:
        medians = sorted(r["prompt_tokens"] for r in records)
        median_prompt = medians[len(medians) // 2]

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "_meta": True,
        "source": "ParaConflict context sweep",
        "filler_kind": args.filler_kind,
        "length_tokens_target": args.length_tokens,
        "median_filler_tokens": median_filler,
        "median_prompt_tokens": median_prompt,
        "categories": categories,
        "num_records": len(records),
        "shuffle_seed": args.shuffle_seed,
        "tokenizer_size": size,
        "conflict_position": "tail",
        "n_copies": 1,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(meta) + "\n")
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    print(
        f"Wrote {len(records)} prompts to {out_path} "
        f"(filler={args.filler_kind}, L={args.length_tokens}, "
        f"median_filler={median_filler}, median_prompt={median_prompt})"
    )


if __name__ == "__main__":
    main()
