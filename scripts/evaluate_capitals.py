"""Evaluate a Pythia size on the world-capital cross-product dataset.

This is the paper-faithful counterpart of ``evaluate_model.py``.  It:

* reads a JSONL produced by ``scripts/build_capitals_crossproduct.py``,
  where every line is one (country, distractor) prompt in the exact
  Yu et al. 2023 template;
* runs greedy generation of ``--max-new-tokens`` tokens per prompt, in
  batches (left-padded), so the 47k-prompt sweep stays tractable up to
  Pythia-12b;
* classifies each generation as "memorized" (contains a gold answer
  alias), "in-context" (contains the distractor capital), both, or
  "other";
* dumps one per-sample JSONL line in a schema compatible with
  ``scripts/analyze_frequency_breakdown.py``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm

from ic_factual.config import PYTHIA_SIZES, resolve_model_size
from ic_factual.device import resolve_device
from ic_factual.loaders import load_causal_lm, load_tokenizer


def _candidate_logprobs(
    model,
    tokenizer,
    device,
    prompts: list[str],
    candidates_per_prompt: list[list[str]],
) -> list[list[float]]:
    """Per-prompt log-prob of every candidate continuation.

    Builds one sequence per (prompt, candidate) and does a single forward
    pass per batch, then sums the log-probs of the candidate's tokens via
    teacher forcing. Returns log-probs in the same shape as
    ``candidates_per_prompt``.

    All sequences within a forward pass are left-padded to the same length;
    the per-prompt prompt-length is recorded so we can pull out only the
    candidate-token logprobs.

    Uses per-token ``logsumexp`` instead of a full ``(B, T, V)`` ``log_softmax``
    so long-context Qwen evals do not allocate tens of GiB for the vocab axis.
    """
    flat_prompts: list[str] = []
    flat_continuations: list[str] = []
    spans: list[int] = []
    for p, cands in zip(prompts, candidates_per_prompt):
        for c in cands:
            flat_prompts.append(p)
            flat_continuations.append(" " + c.strip())
        spans.append(len(cands))

    full_texts = [p + c for p, c in zip(flat_prompts, flat_continuations)]
    enc_prompt = tokenizer(flat_prompts, padding=False, add_special_tokens=False)
    prompt_lens = [len(ids) for ids in enc_prompt["input_ids"]]

    max_seq_len = max(
        len(tokenizer(t, add_special_tokens=False)["input_ids"]) for t in full_texts
    )
    # Long prompts: one (prompt, candidate) sequence per forward to cap peak memory.
    flat_batch = 1 if max_seq_len >= 12288 else min(8, len(full_texts))

    out: list[float] = []
    for start in range(0, len(full_texts), flat_batch):
        chunk_texts = full_texts[start : start + flat_batch]
        chunk_prompt_lens = prompt_lens[start : start + flat_batch]
        enc_full = tokenizer(
            chunk_texts, return_tensors="pt", padding=True, add_special_tokens=False
        ).to(device)

        with torch.no_grad():
            logits = model(**enc_full).logits  # (B, T, V)

        input_ids = enc_full["input_ids"]
        attn = enc_full["attention_mask"]
        seq_len = input_ids.size(1)

        for i in range(input_ids.size(0)):
            valid = int(attn[i].sum().item())
            offset = seq_len - valid
            plen = chunk_prompt_lens[i]
            total_lp = 0.0
            for t in range(plen, valid):
                pos_in_padded = offset + t
                logit_vec = logits[i, pos_in_padded - 1].float()
                log_z = torch.logsumexp(logit_vec, dim=-1)
                target_token = input_ids[i, pos_in_padded].item()
                total_lp += (logit_vec[target_token] - log_z).item()
            out.append(total_lp)

        del logits, enc_full
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()

    grouped: list[list[float]] = []
    j = 0
    for n in spans:
        grouped.append(out[j : j + n])
        j += n
    return grouped


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


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Greedy-generate world-capital answers for every (country, "
            "distractor) prompt and classify each output."
        )
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="JSONL from scripts/build_capitals_crossproduct.py.",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Per-sample JSONL output path.",
    )
    parser.add_argument(
        "--model-size",
        type=str,
        default=None,
        choices=list(PYTHIA_SIZES),
        help="Pythia size to evaluate.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help=(
            "Explicit local snapshot path or Hub id (e.g. a Qwen3 directory). "
            "Bypasses --model-size and the Pythia-only size lookup; useful for "
            "evaluating other model families with the same paper-faithful "
            "capitals pipeline."
        ),
    )
    parser.add_argument(
        "--label",
        type=str,
        default=None,
        help=(
            "Human-readable label written to the JSONL _meta and printed "
            "in progress. Defaults to 'pythia-<size>' or the basename of "
            "--model-path."
        ),
    )
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--dtype",
        type=str,
        default=None,
        help="Torch dtype (float32, float16/fp16, bfloat16/bf16). Recommended fp16 for 6.9b/12b.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=16,
        help="Batch size for generation. Reduce for the largest sizes.",
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=10,
        help=(
            "Number of tokens to generate per prompt. 10 leaves room for "
            "outputs like 'The capital city is Warsaw' that exceed the bare "
            "city name."
        ),
    )
    parser.add_argument(
        "--method",
        type=str,
        default="logprob",
        choices=("generative", "logprob", "both"),
        help=(
            "Classification method. 'logprob' (default) computes "
            "log P(' {city}' | prompt) for both candidates and picks the "
            "argmax -- a single forward pass instead of N generation steps "
            "(3-5x faster). 'generative' replicates the paper's "
            "decode-then-substring-match exactly. 'both' runs each."
        ),
    )
    parser.add_argument(
        "--max-samples", type=int, default=None,
        help="Cap the number of input lines (for smoke tests).",
    )
    parser.add_argument(
        "--shuffle-seed", type=int, default=None,
        help=(
            "Optional seed; when set the input is shuffled before sampling. "
            "Use this with --max-samples so a smoke test covers many "
            "countries / distractors instead of one country alphabetically."
        ),
    )
    args = parser.parse_args()

    device = resolve_device(args.device)
    if args.model_path is not None:
        if args.model_size is not None:
            raise SystemExit("--model-path and --model-size are mutually exclusive")
        size = None
        label = args.label or Path(args.model_path).name or args.model_path
    else:
        size = resolve_model_size(args.model_size)
        label = args.label or f"pythia-{size}"

    print(
        f"{label}  device={device}  dtype={args.dtype or 'auto'}  "
        f"batch_size={args.batch_size}  max_new_tokens={args.max_new_tokens}"
    )

    if args.model_path is not None:
        tokenizer = load_tokenizer(args.model_path)
    else:
        tokenizer = load_tokenizer(model_size=size)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    if args.model_path is not None:
        model, device = load_causal_lm(
            args.model_path, device=device, dtype=args.dtype
        )
    else:
        model, device = load_causal_lm(
            device=device, model_size=size, dtype=args.dtype
        )
    model.eval()

    in_meta, records = _load_jsonl(Path(args.input))
    if args.shuffle_seed is not None:
        import random
        random.Random(args.shuffle_seed).shuffle(records)
    if args.max_samples is not None:
        records = records[: args.max_samples]
    print(f"Loaded {len(records)} prompts from {args.input}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_fp = open(out_path, "w", encoding="utf-8")
    out_fp.write(json.dumps({
        "_meta": True,
        # ``model_size`` is what the analyzer keys on; for Pythia it stays the
        # short size string (``"160m"``), for other families we use the full
        # label (``"qwen3-0.6b"``) so plot panels self-identify.
        "model_size": size if size is not None else label,
        "model_path": args.model_path,
        "label": label,
        "dtype": args.dtype,
        "device": str(device),
        "num_samples": len(records),
        "batch_size": args.batch_size,
        "max_new_tokens": args.max_new_tokens,
        "method": args.method,
        "input_meta": in_meta,
    }) + "\n")

    do_gen = args.method in ("generative", "both")
    do_lp = args.method in ("logprob", "both")

    counts_gen = {"memorized": 0, "in_context": 0, "both": 0, "other": 0}
    counts_lp = {"memorized": 0, "in_context": 0, "tie": 0}
    pbar = tqdm(range(0, len(records), args.batch_size), desc=label)
    for start in pbar:
        batch = records[start : start + args.batch_size]
        prompts = [r["prompt"] for r in batch]

        gen_texts: list[str | None] = [None] * len(batch)
        lp_mem_list: list[float | None] = [None] * len(batch)
        lp_ctx_list: list[float | None] = [None] * len(batch)

        if do_gen:
            enc = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
            prompt_len = enc["input_ids"].size(1)
            with torch.no_grad():
                out_ids = model.generate(
                    **enc,
                    max_new_tokens=args.max_new_tokens,
                    pad_token_id=tokenizer.pad_token_id,
                    do_sample=False,
                )
            gen_only = out_ids[:, prompt_len:]
            decoded = tokenizer.batch_decode(gen_only, skip_special_tokens=True)
            gen_texts = [t.strip().lower() for t in decoded]

        if do_lp:
            cands_per_prompt: list[list[str]] = []
            for rec in batch:
                # First gold alias = memorized; ParaConflict's first alias is
                # typically the canonical English spelling.
                mem = rec["answers"][0]
                ctx = rec["distractor"]
                cands_per_prompt.append([mem, ctx])
            lps_grouped = _candidate_logprobs(
                model, tokenizer, device, prompts, cands_per_prompt
            )
            for i, lps in enumerate(lps_grouped):
                lp_mem_list[i] = float(lps[0])
                lp_ctx_list[i] = float(lps[1])

        for i, rec in enumerate(batch):
            answers_lower = [a.strip().lower() for a in rec["answers"]]
            distractor_lower = rec["distractor"].strip().lower()

            is_mem_gen = is_ctx_gen = False
            gen_text = gen_texts[i]
            if gen_text is not None:
                is_mem_gen = any(a in gen_text for a in answers_lower)
                is_ctx_gen = distractor_lower in gen_text
                if is_mem_gen and is_ctx_gen:
                    counts_gen["both"] += 1
                elif is_mem_gen:
                    counts_gen["memorized"] += 1
                elif is_ctx_gen:
                    counts_gen["in_context"] += 1
                else:
                    counts_gen["other"] += 1

            is_mem_lp = is_ctx_lp = False
            lp_mem = lp_mem_list[i]
            lp_ctx = lp_ctx_list[i]
            if lp_mem is not None and lp_ctx is not None:
                if lp_mem > lp_ctx:
                    is_mem_lp = True
                    counts_lp["memorized"] += 1
                elif lp_ctx > lp_mem:
                    is_ctx_lp = True
                    counts_lp["in_context"] += 1
                else:
                    counts_lp["tie"] += 1

            block = {
                "memorized_gen": bool(is_mem_gen),
                "in_context_gen": bool(is_ctx_gen),
                "memorized_lp": bool(is_mem_lp),
                "in_context_lp": bool(is_ctx_lp),
            }
            if gen_text is not None:
                block["gen"] = gen_text
            if lp_mem is not None:
                block["lp_answer"] = lp_mem
                block["lp_distractor"] = lp_ctx

            out_fp.write(json.dumps({
                "subject": rec["country"],
                "country": rec["country"],
                "answers": rec["answers"],
                "distractor": rec["distractor"],
                "distractor_country": rec.get("distractor_country"),
                "category": "World Capital",
                "substitution_conflict": block,
            }) + "\n")

        if (start // args.batch_size) % 25 == 0:
            postfix: dict[str, str] = {}
            if do_gen:
                n = sum(counts_gen.values()) or 1
                postfix["g_mem"] = f"{counts_gen['memorized']/n:.0%}"
                postfix["g_ctx"] = f"{counts_gen['in_context']/n:.0%}"
            if do_lp:
                n = sum(counts_lp.values()) or 1
                postfix["lp_mem"] = f"{counts_lp['memorized']/n:.0%}"
                postfix["lp_ctx"] = f"{counts_lp['in_context']/n:.0%}"
            pbar.set_postfix(postfix)

    out_fp.close()
    print(f"\n--- {label} on {len(records)} prompts (method={args.method}) ---")
    if do_gen:
        total = sum(counts_gen.values()) or 1
        print(" GENERATIVE:")
        for k, v in counts_gen.items():
            print(f"   {k:10s}: {v:6d}  ({v/total*100:5.1f}%)")
    if do_lp:
        total = sum(counts_lp.values()) or 1
        print(" LOGPROB (argmax memorized vs in-context):")
        for k, v in counts_lp.items():
            print(f"   {k:10s}: {v:6d}  ({v/total*100:5.1f}%)")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
