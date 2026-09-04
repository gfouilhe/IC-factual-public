"""Estimate per-Subject corpus frequency for the memorized-vs-in-context plot.

Two proxies are supported via ``--proxy``:

* ``wordfreq`` (default, recommended)  --  Zipf frequency from the
  `wordfreq` package's mixed web/Wikipedia/news corpus.  This is the
  closest readily-available stand-in for Pile term frequency and is
  what Yu, Merullo & Pavlick (2023) approximate with actual Pile
  counts.  Requires ``pip install wordfreq``.

* ``model`` (legacy)  --  log-probability of the subject string under
  Pythia-160m (or another size).  Self-contained but biased by tokenizer
  artifacts (multi-token names, leading articles, etc.).  Kept for
  reference / ablation.

Both proxies emit the same JSON schema so downstream analysis doesn't
care which was used.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

from ic_factual.config import PYTHIA_SIZES, resolve_model_size
from ic_factual.device import resolve_device
from ic_factual.loaders import load_causal_lm, load_paraconflict, load_tokenizer


_STRIPPABLE_PREFIXES = ("the ", "The ")


def _normalize_for_logprob(text: str) -> str:
    """Drop a leading 'the ' / 'The ' article so two entities are comparable.

    Country names in ParaConflict mix "France" with "the United Kingdom" /
    "the Marshall Islands". Without this normalization every entity that
    starts with "the" ends up with an identical first-token log-prob, which
    destroys the frequency signal.
    """
    s = text.strip()
    for p in _STRIPPABLE_PREFIXES:
        if s.startswith(p):
            return s[len(p):]
    return s


def unconditional_logprob(
    text: str,
    model,
    tokenizer,
    device,
    prefix: str = "",
) -> tuple[float, float, int]:
    """Return ``(total_logprob, first_token_logprob, n_target_tokens)``.

    ``first_token_logprob`` is the model's log-prob of the first token of
    ``text`` given ``prefix`` (or BOS).  This is the cleanest single-number
    "is this entity familiar" signal -- it isn't inflated by predictable
    continuations like " and Herzegovina" after "Bosnia".

    ``total_logprob`` is the sum across all target tokens (useful for
    sanity / debugging; biased by length).
    """
    bos_id = tokenizer.bos_token_id or tokenizer.eos_token_id
    if prefix:
        prefix_ids = tokenizer.encode(prefix, return_tensors="pt").to(device)
    else:
        prefix_ids = torch.tensor([[bos_id]], device=device, dtype=torch.long)

    normalized = _normalize_for_logprob(text)
    target_ids = tokenizer.encode(
        " " + normalized, add_special_tokens=False, return_tensors="pt"
    ).to(device)

    input_ids = torch.cat([prefix_ids, target_ids], dim=1)
    with torch.no_grad():
        logits = model(input_ids).logits

    total = 0.0
    first_lp = float("nan")
    for i in range(target_ids.size(1)):
        token_logits = logits[0, prefix_ids.size(1) - 1 + i, :]
        token_logprobs = F.log_softmax(token_logits, dim=-1)
        lp_i = token_logprobs[target_ids[0, i]].item()
        if i == 0:
            first_lp = lp_i
        total += lp_i
    return total, first_lp, int(target_ids.size(1))


def _wordfreq_proxy(name: str, lang: str = "en") -> dict:
    """Return Zipf and raw frequencies from the wordfreq package."""
    from wordfreq import word_frequency, zipf_frequency

    normalized = _normalize_for_logprob(name)
    z = zipf_frequency(normalized, lang)
    f = word_frequency(normalized, lang)
    # For multi-word phrases also compute a per-word min Zipf (rarest word).
    # This guards against phrase entries that don't exist in the wordlist
    # collapsing to Zipf=0.
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


def _iter_subjects_from_args(args) -> dict[str, str]:
    """Return {subject: category} from either ParaConflict or a JSONL file."""
    if args.from_jsonl:
        subjects: dict[str, str] = {}
        with open(args.from_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if obj.get("_meta"):
                    continue
                for field in args.subject_fields:
                    val = obj.get(field)
                    if val and val not in subjects:
                        subjects[val] = obj.get("category", "")
        return subjects

    dataset = load_paraconflict(split="test")
    if args.categories:
        wanted = {c.strip().lower() for c in args.categories}
        dataset = dataset.filter(lambda r: r["Category"].lower() in wanted)
    print(f"Loaded {len(dataset)} ParaConflict samples")
    subjects: dict[str, str] = {}
    for row in dataset:
        subjects.setdefault(row["Subject"], row.get("Category", ""))
    return subjects


def main():
    parser = argparse.ArgumentParser(
        description="Compute a per-Subject frequency proxy and dump it as JSON."
    )
    parser.add_argument(
        "--proxy",
        type=str,
        default="wordfreq",
        choices=("wordfreq", "model"),
        help=(
            "Frequency proxy. 'wordfreq' uses the wordfreq Zipf score "
            "(recommended). 'model' uses the unconditional log-prob of the "
            "subject under a small Pythia model (legacy)."
        ),
    )
    parser.add_argument(
        "--model-size",
        type=str,
        default="160m",
        choices=list(PYTHIA_SIZES),
        help="Pythia size used by --proxy model (default: 160m).",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Override device (default: cuda if available, else cpu).",
    )
    parser.add_argument(
        "--dtype", type=str, default=None,
        help="Torch dtype (float32, float16/fp16, bfloat16/bf16). Defaults to float32.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output/subject_frequency.json",
        help="Where to write the per-subject frequency JSON.",
    )
    parser.add_argument(
        "--categories",
        type=str,
        nargs="*",
        default=None,
        help="Optional ParaConflict categories to restrict to.",
    )
    parser.add_argument(
        "--from-jsonl",
        type=str,
        default=None,
        help=(
            "Instead of reading ParaConflict, harvest subjects from a JSONL "
            "(e.g. capitals_crossproduct.jsonl). The script will read "
            "--subject-fields from each line."
        ),
    )
    parser.add_argument(
        "--subject-fields",
        type=str,
        nargs="*",
        default=("country", "distractor_country", "subject"),
        help=(
            "Which fields to harvest as 'subjects' when --from-jsonl is used. "
            "Default covers both the queried country and the in-context "
            "country whose capital is the distractor."
        ),
    )
    parser.add_argument(
        "--lang", type=str, default="en",
        help="wordfreq language code (default: en).",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="",
        help="(--proxy model only) prefix to condition on (default: BOS).",
    )
    args = parser.parse_args()

    subjects = _iter_subjects_from_args(args)
    print(f"Found {len(subjects)} unique subjects")

    out: dict[str, dict] = {}

    if args.proxy == "wordfreq":
        print(f"Frequency proxy:  wordfreq Zipf  (lang={args.lang})")
        for subject, category in tqdm(subjects.items(), desc="zipf"):
            info = _wordfreq_proxy(subject, lang=args.lang)
            info["category"] = category
            out[subject] = info
        meta_desc = (
            "Per-subject wordfreq Zipf frequency on the chosen language's "
            "mixed corpus (news + wiki + subtitles + web). Higher = more "
            "frequent."
        )
        meta_extra: dict = {"lang": args.lang}
    else:
        device = resolve_device(args.device)
        size = resolve_model_size(args.model_size)
        print(
            f"Frequency proxy:  Pythia-{size}  "
            f"(device={device}, dtype={args.dtype or 'fp32'})"
        )
        tokenizer = load_tokenizer(model_size=size)
        model, device = load_causal_lm(device=device, model_size=size, dtype=args.dtype)
        for subject, category in tqdm(subjects.items(), desc="logprob"):
            lp, first_lp, n = unconditional_logprob(
                subject, model, tokenizer, device, prefix=args.prefix
            )
            out[subject] = {
                "logprob": lp,
                "logprob_per_token": lp / max(n, 1),
                "first_token_logprob": first_lp,
                "n_tokens": n,
                "normalized": _normalize_for_logprob(subject),
                "category": category,
            }
        meta_desc = (
            f"Per-subject unconditional log-probability under Pythia-{size}. "
            "Higher (less negative) => more frequent / familiar."
        )
        meta_extra = {"proxy_model_size": size, "prefix": args.prefix}

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "_meta": {
            "proxy": args.proxy,
            "n_subjects": len(out),
            "description": meta_desc,
            **meta_extra,
        },
        "subjects": out,
    }
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {out_path}  ({len(out)} subjects)")


if __name__ == "__main__":
    main()
