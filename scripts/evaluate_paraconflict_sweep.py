"""Evaluate ParaConflict context-sweep JSONLs (generative / logprob classifiers).

Reads JSONL from ``scripts/build_paraconflict_context_sweep.py`` and writes
per-sample classifications compatible with ``analyze_coherent_motivation.py``.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import torch
from tqdm import tqdm

from ic_factual.config import PYTHIA_SIZES, resolve_model_size
from ic_factual.device import resolve_device
from ic_factual.loaders import load_causal_lm, load_tokenizer

_EVAL_CAP_PATH = Path(__file__).resolve().parent / "evaluate_capitals.py"
_spec = importlib.util.spec_from_file_location("evaluate_capitals", _EVAL_CAP_PATH)
_eval_cap = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_eval_cap)
_candidate_logprobs = _eval_cap._candidate_logprobs


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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Classify ParaConflict sweep prompts (memorized vs in-context)."
    )
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--model-size", type=str, default=None, choices=list(PYTHIA_SIZES))
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--label", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--dtype", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=10)
    parser.add_argument(
        "--method",
        type=str,
        default="generative",
        choices=("generative", "logprob", "both"),
    )
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--shuffle-seed", type=int, default=None)
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

    if args.model_path is not None:
        tokenizer = load_tokenizer(args.model_path)
        model, device = load_causal_lm(args.model_path, device=device, dtype=args.dtype)
    else:
        tokenizer = load_tokenizer(model_size=size)
        model, device = load_causal_lm(device=device, model_size=size, dtype=args.dtype)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
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
    out_fp.write(
        json.dumps(
            {
                "_meta": True,
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
            }
        )
        + "\n"
    )

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
            cands_per_prompt = [
                [rec["answers"][0], rec["distractor"]] for rec in batch
            ]
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

            out_fp.write(
                json.dumps(
                    {
                        "subject": rec["subject"],
                        "answers": rec["answers"],
                        "distractor": rec["distractor"],
                        "category": rec.get("category"),
                        "prompt": rec["prompt"],
                        "substitution_conflict": block,
                    }
                )
                + "\n"
            )

    out_fp.close()
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
