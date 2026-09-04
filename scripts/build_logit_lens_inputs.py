"""Build the four logit-lens input conditions used in the paper figure.

Produces JSONL files compatible with ``scripts/logit_lens_capitals.py``:

* ``input_L0_std.jsonl`` — paper template, no filler
* ``input_L0_neg.jsonl`` — negated conflict, no filler
* ``input_L128_prose_std.jsonl`` — 128-token prose filler, standard conflict
* ``input_L128_prose_neg.jsonl`` — 128-token prose filler, negated conflict

Filler length is measured with the Pythia GPT-NeoX tokenizer (shared across
all Pythia sizes).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ic_factual.loaders import load_tokenizer

PROSE_SNIPPET = (
    "Plant cells contain a rigid outer cell wall composed primarily of "
    "cellulose, which provides structural support and resists turgor "
    "pressure. Within the cell, organelles such as chloroplasts capture "
    "light energy and convert it into chemical energy through "
    "photosynthesis. The mitochondria perform aerobic respiration, "
    "producing adenosine triphosphate that powers cellular processes. "
    "Vacuoles store water, ions, and metabolic byproducts, while "
    "ribosomes synthesize proteins from messenger RNA transcripts. The "
    "nucleus houses the genetic material and coordinates gene expression "
    "in response to environmental cues. Endoplasmic reticulum and Golgi "
    "bodies process and traffic newly synthesized proteins to their "
    "destinations. Cytoskeletal filaments give shape to the cell and "
    "enable intracellular transport. During mitosis, chromosomes "
    "condense and segregate into two daughter nuclei. Stomatal pores in "
    "the leaf epidermis regulate gas exchange and transpiration. "
)

CONFLICT_STANDARD = "The capital of {country} is {distractor}."
CONFLICT_NEGATED = "The capital of {country} is not {distractor}."
QUESTION_TEMPLATE = "Q: What is the capital of {country}? A:"

CONDITIONS: tuple[tuple[str, int, str], ...] = (
    ("input_L0_std.jsonl", 0, "standard"),
    ("input_L0_neg.jsonl", 0, "negated"),
    ("input_L128_prose_std.jsonl", 128, "standard"),
    ("input_L128_prose_neg.jsonl", 128, "negated"),
)


def _load_jsonl(path: Path) -> tuple[dict, list[dict]]:
    meta: dict = {}
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
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


def _encode(text: str, tokenizer) -> list[int]:
    return tokenizer.encode(text, add_special_tokens=False)


def _tile_to_n_tokens(snippet: str, n_tokens: int, tokenizer) -> str:
    if n_tokens <= 0:
        return ""
    ids = _encode(snippet, tokenizer)
    if not ids:
        return ""
    tiled = (ids * ((n_tokens // len(ids)) + 1))[:n_tokens]
    return tokenizer.decode(tiled, clean_up_tokenization_spaces=False)


def _assemble_prompt(
    country: str,
    distractor: str,
    *,
    filler: str,
    conflict_kind: str,
) -> str:
    if conflict_kind == "standard":
        conflict = CONFLICT_STANDARD.format(country=country, distractor=distractor)
    else:
        conflict = CONFLICT_NEGATED.format(country=country, distractor=distractor)
    question = QUESTION_TEMPLATE.format(country=country)
    if filler:
        return f"{conflict} {filler} {question}"
    return f"{conflict} {question}"


def build_condition(
    records: list[dict],
    *,
    length_tokens: int,
    conflict_kind: str,
    tokenizer,
    max_samples: int | None,
) -> tuple[list[dict], dict]:
    filler = _tile_to_n_tokens(PROSE_SNIPPET, length_tokens, tokenizer)
    filler_tokens = len(_encode(filler, tokenizer)) if filler else 0
    subset = records[:max_samples] if max_samples is not None else records

    out_records: list[dict] = []
    for rec in subset:
        prompt = _assemble_prompt(
            rec["country"],
            rec["distractor"],
            filler=filler,
            conflict_kind=conflict_kind,
        )
        out_records.append({**rec, "prompt": prompt})

    meta = {
        "_meta": True,
        "filler_kind": "prose",
        "length_tokens": length_tokens,
        "actual_filler_tokens": filler_tokens,
        "conflict_template": conflict_kind,
        "conflict_position": "tail",
        "n_copies": 1,
        "n_prompts": len(out_records),
    }
    return out_records, meta


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the four logit-lens capitals input conditions."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/capitals_crossproduct.jsonl"),
        help="Base cross-product JSONL.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/logit_lens"),
        help="Directory for the four output JSONLs.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Cap prompts per condition (smoke tests).",
    )
    args = parser.parse_args()

    in_meta, records = _load_jsonl(args.input)
    print(f"Loaded {len(records)} prompts from {args.input}")

    tokenizer = load_tokenizer(model_size="160m")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for filename, length_tokens, conflict_kind in CONDITIONS:
        out_records, meta = build_condition(
            records,
            length_tokens=length_tokens,
            conflict_kind=conflict_kind,
            tokenizer=tokenizer,
            max_samples=args.max_samples,
        )
        meta["source_crossproduct_meta"] = in_meta
        out_path = args.output_dir / filename
        with out_path.open("w", encoding="utf-8") as f:
            f.write(json.dumps(meta) + "\n")
            for rec in out_records:
                f.write(json.dumps(rec) + "\n")
        print(
            f"Wrote {len(out_records)} prompts to {out_path} "
            f"(L={length_tokens}, conflict={conflict_kind})"
        )


if __name__ == "__main__":
    main()
