"""Build the Yu, Merullo & Pavlick (EMNLP 2023) style world-capital cross-product.

For every (country, true_capital) pair in ParaConflict's 'World Capital'
category, this script emits one prompt for every OTHER capital used as a
counterfactual in-context distractor. The prompt template defaults to the
exact form from Section 3 of the paper:

    The capital of {country} is {distractor}. Q: What is the capital of {country}? A:

Output: a JSONL with one record per prompt, plus a header `_meta` line:

    {"country": "Poland", "answers": ["Warsaw"], "distractor": "London",
     "distractor_country": "the United Kingdom",
     "prompt": "The capital of Poland is London. Q: ..."}

This file is the input to ``scripts/evaluate_capitals.py``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ic_factual.loaders import load_paraconflict


PAPER_TEMPLATE = (
    "The capital of {country} is {distractor}. "
    "Q: What is the capital of {country}? A:"
)


def build_crossproduct(
    template: str = PAPER_TEMPLATE,
    max_countries: int | None = None,
) -> tuple[list[dict], dict]:
    ds = load_paraconflict(split="test")
    ds = ds.filter(lambda r: r["Category"] == "World Capital")

    country_to_aliases: dict[str, list[str]] = {}
    for row in ds:
        country = row["Subject"]
        aliases = row["Answer"]
        if isinstance(aliases, str):
            aliases = [aliases]
        country_to_aliases.setdefault(country, list(aliases))

    countries = sorted(country_to_aliases)
    if max_countries is not None:
        countries = countries[:max_countries]

    capital_of: dict[str, str] = {c: country_to_aliases[c][0] for c in countries}
    capital_set: set[str] = set(capital_of.values())

    records: list[dict] = []
    for country in countries:
        true_aliases = country_to_aliases[country]
        true_aliases_lower = {a.strip().lower() for a in true_aliases}
        for other in countries:
            if other == country:
                continue
            distractor = capital_of[other]
            if distractor.strip().lower() in true_aliases_lower:
                continue
            prompt = template.format(country=country, distractor=distractor)
            records.append(
                {
                    "country": country,
                    "answers": true_aliases,
                    "distractor": distractor,
                    "distractor_country": other,
                    "prompt": prompt,
                }
            )

    meta = {
        "_meta": True,
        "source": "ParaConflict World Capital",
        "prompt_template": template,
        "n_countries": len(countries),
        "n_capitals": len(capital_set),
        "n_prompts": len(records),
    }
    return records, meta


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Build a (country x other-capital) cross-product dataset "
            "matching the paper's world-capital task."
        )
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output/capitals_crossproduct.jsonl",
        help="Where to write the JSONL.",
    )
    parser.add_argument(
        "--prompt-template",
        type=str,
        default=None,
        help=(
            "Override the prompt template. Use {country} and {distractor}. "
            "Default matches Yu et al. 2023 Section 3 exactly."
        ),
    )
    parser.add_argument(
        "--max-countries",
        type=int,
        default=None,
        help="Limit to the first N countries (alphabetical) -- for smoke tests.",
    )
    args = parser.parse_args()

    template = args.prompt_template or PAPER_TEMPLATE
    records, meta = build_crossproduct(template=template, max_countries=args.max_countries)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(meta) + "\n")
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    print(
        f"Wrote {len(records)} prompts from "
        f"{meta['n_countries']} countries x {meta['n_capitals']} capitals "
        f"to {out_path}"
    )


if __name__ == "__main__":
    main()
