import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

from ic_factual.config import PYTHIA_SIZES, resolve_model_size
from ic_factual.device import resolve_device
from ic_factual.loaders import load_causal_lm, load_paraconflict, load_tokenizer


# ParaConflict ships "Athelete Sport" (typo); paper/exp15 use "Athlete Sport".
_CATEGORY_FILTER_ALIASES: dict[str, frozenset[str]] = {
    "athlete sport": frozenset({"athlete sport", "athelete sport"}),
    "athelete sport": frozenset({"athlete sport", "athelete sport"}),
}


def _category_filter_keys(categories: list[str]) -> set[str]:
    keys: set[str] = set()
    for cat in categories:
        low = cat.strip().lower()
        keys.update(_CATEGORY_FILTER_ALIASES.get(low, {low}))
    return keys


def _canonical_category(category: str) -> str:
    if category == "Athelete Sport":
        return "Athlete Sport"
    return category


def calculate_logprob(prompt, target, model, tokenizer, device):
    """Calculate log probability of target given prompt."""
    prompt_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    target_ids = tokenizer.encode(
        target, add_special_tokens=False, return_tensors="pt"
    ).to(device)

    input_ids = torch.cat([prompt_ids, target_ids], dim=1)
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits

    logprobs = 0.0
    for i in range(target_ids.size(1)):
        # Predict the i-th token of target
        token_logits = logits[0, prompt_ids.size(1) - 1 + i, :]
        token_logprobs = F.log_softmax(token_logits, dim=-1)
        logprobs += token_logprobs[target_ids[0, i]].item()
    return logprobs


def evaluate_model(
    max_samples: int | None = None,
    device: str | None = None,
    model_size: str | None = None,
    model_path: str | None = None,
    label: str | None = None,
    dtype: str | None = None,
    per_sample_output: str | None = None,
    categories: list[str] | None = None,
):
    device = resolve_device(device)
    if model_path is not None:
        if model_size is not None:
            raise ValueError("--model-path and --model-size are mutually exclusive")
        size = None
        run_label = label or Path(model_path).name or model_path
    else:
        size = resolve_model_size(model_size)
        run_label = label or size
    print(f"Using device: {device}")
    print(f"Model label:  {run_label}")
    if size:
        print(f"Pythia size:  {size}")
    if model_path:
        print(f"Model path:   {model_path}")
    if dtype:
        print(f"Dtype:        {dtype}")

    print("Loading model...")
    if model_path is not None:
        tokenizer = load_tokenizer(model_path)
        model, device = load_causal_lm(
            model_path, device=device, dtype=dtype
        )
    else:
        tokenizer = load_tokenizer(model_size=size)
        model, device = load_causal_lm(device=device, model_size=size, dtype=dtype)

    print("Loading ParaConflict dataset...")
    dataset = load_paraconflict(split="test")
    if categories:
        wanted = _category_filter_keys(categories)
        dataset = dataset.filter(lambda r: r["Category"].lower() in wanted)
        print(f"Filtered to categories {sorted(wanted)}: {len(dataset)} samples")
    if max_samples is not None:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    print(f"Loaded {len(dataset)} samples")

    per_sample_fp = None
    if per_sample_output:
        Path(per_sample_output).parent.mkdir(parents=True, exist_ok=True)
        per_sample_fp = open(per_sample_output, "w", encoding="utf-8")
        header = {
            "_meta": True,
            "label": run_label,
            "model_size": size,
            "dtype": dtype,
            "device": str(device),
            "num_samples": len(dataset),
            "categories": categories,
        }
        per_sample_fp.write(json.dumps(header) + "\n")
        print(f"Writing per-sample results to {per_sample_output}")

    results = {
        "clean_prompt": {
            "correct": 0,
            "logprob_correct": 0,
            "total": 0,
        },
        "substitution_conflict": {
            "correct": 0,
            "distractor": 0,
            "other": 0,
            "logprob_correct": 0,
            "logprob_distractor": 0,
            "total": 0,
        },
        "coherent_conflict": {
            "correct": 0,
            "distractor": 0,
            "other": 0,
            "logprob_correct": 0,
            "logprob_distractor": 0,
            "total": 0,
        },
    }

    def _generate(prompt: str) -> str:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=5,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=False,
            )
        return tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1] :], skip_special_tokens=True
        ).strip().lower()

    print("Evaluating...")
    for idx, item in enumerate(tqdm(dataset)):
        subject = item["Subject"]
        answers = item["Answer"]  # List of gold facts
        distractor = item["Distracted Token"]
        clean_prompt = item["Clean Prompt"]
        substitution_conflict = item["Substitution Conflict"]
        coherent_conflict = item["Coherent Conflict"]
        category = _canonical_category(item.get("Category", ""))

        if isinstance(answers, str):
            answers = [answers]

        answers_lower = [ans.lower().strip() for ans in answers]
        distractor_lower = distractor.lower().strip()

        gen_clean = _generate(clean_prompt)
        clean_is_correct = any(a in gen_clean for a in answers_lower)
        results["clean_prompt"]["total"] += 1
        if clean_is_correct:
            results["clean_prompt"]["correct"] += 1

        lp_answer_clean = calculate_logprob(clean_prompt, answers[0], model, tokenizer, device)
        lp_distractor_clean = calculate_logprob(clean_prompt, distractor, model, tokenizer, device)
        clean_lp_correct = lp_answer_clean > lp_distractor_clean
        if clean_lp_correct:
            results["clean_prompt"]["logprob_correct"] += 1

        gen_sub = _generate(substitution_conflict)
        sub_is_correct = any(a in gen_sub for a in answers_lower)
        sub_is_distractor = distractor_lower in gen_sub
        results["substitution_conflict"]["total"] += 1
        if sub_is_correct:
            results["substitution_conflict"]["correct"] += 1
        elif sub_is_distractor:
            results["substitution_conflict"]["distractor"] += 1
        else:
            results["substitution_conflict"]["other"] += 1

        lp_answer_sub = calculate_logprob(
            substitution_conflict, answers[0], model, tokenizer, device
        )
        lp_distractor_sub = calculate_logprob(
            substitution_conflict, distractor, model, tokenizer, device
        )
        if lp_answer_sub > lp_distractor_sub:
            results["substitution_conflict"]["logprob_correct"] += 1
        elif lp_distractor_sub > lp_answer_sub:
            results["substitution_conflict"]["logprob_distractor"] += 1

        gen_coh = _generate(coherent_conflict)
        coh_is_correct = any(a in gen_coh for a in answers_lower)
        coh_is_distractor = distractor_lower in gen_coh
        results["coherent_conflict"]["total"] += 1
        if coh_is_correct:
            results["coherent_conflict"]["correct"] += 1
        elif coh_is_distractor:
            results["coherent_conflict"]["distractor"] += 1
        else:
            results["coherent_conflict"]["other"] += 1

        lp_answer_coh = calculate_logprob(coherent_conflict, answers[0], model, tokenizer, device)
        lp_distractor_coh = calculate_logprob(
            coherent_conflict, distractor, model, tokenizer, device
        )
        if lp_answer_coh > lp_distractor_coh:
            results["coherent_conflict"]["logprob_correct"] += 1
        elif lp_distractor_coh > lp_answer_coh:
            results["coherent_conflict"]["logprob_distractor"] += 1

        if per_sample_fp is not None:
            per_sample_fp.write(json.dumps({
                "idx": idx,
                "category": category,
                "subject": subject,
                "answers": answers,
                "distractor": distractor,
                "clean_prompt": {
                    "gen": gen_clean,
                    "correct_gen": bool(clean_is_correct),
                    "lp_answer": lp_answer_clean,
                    "lp_distractor": lp_distractor_clean,
                    "answer_beats_distractor": bool(clean_lp_correct),
                },
                "substitution_conflict": {
                    "gen": gen_sub,
                    "memorized_gen": bool(sub_is_correct),
                    "in_context_gen": bool(sub_is_distractor),
                    "lp_answer": lp_answer_sub,
                    "lp_distractor": lp_distractor_sub,
                    "memorized_lp": bool(lp_answer_sub > lp_distractor_sub),
                    "in_context_lp": bool(lp_distractor_sub > lp_answer_sub),
                },
                "coherent_conflict": {
                    "gen": gen_coh,
                    "memorized_gen": bool(coh_is_correct),
                    "in_context_gen": bool(coh_is_distractor),
                    "lp_answer": lp_answer_coh,
                    "lp_distractor": lp_distractor_coh,
                    "memorized_lp": bool(lp_answer_coh > lp_distractor_coh),
                    "in_context_lp": bool(lp_distractor_coh > lp_answer_coh),
                },
            }) + "\n")

    if per_sample_fp is not None:
        per_sample_fp.close()

    print("\n--- ParaConflict Evaluation Results ---")

    clean_total = results["clean_prompt"]["total"]
    clean_correct = results["clean_prompt"]["correct"]
    clean_lp_correct = results["clean_prompt"]["logprob_correct"]
    print("\nClean Prompt (Baseline):")
    print(f"  Total Samples: {clean_total}")
    if clean_total:
        print(
            f"  Correct Answer (Generative): {clean_correct}/{clean_total} ({clean_correct / clean_total * 100:.1f}%)"
        )
        print(
            f"  Correct Answer (Logprob): {clean_lp_correct}/{clean_total} ({clean_lp_correct / clean_total * 100:.1f}%)"
        )
    else:
        print("  (no samples — check --categories filter)")
        return

    sub_total = results["substitution_conflict"]["total"]
    sub_correct = results["substitution_conflict"]["correct"]
    sub_distractor = results["substitution_conflict"]["distractor"]
    sub_other = results["substitution_conflict"]["other"]
    sub_lp_correct = results["substitution_conflict"]["logprob_correct"]
    sub_lp_distractor = results["substitution_conflict"]["logprob_distractor"]
    print("\nSubstitution Conflict (Gold replaced by distractor):")
    print(f"  Total Samples: {sub_total}")
    print("  --- Generative ---")
    print(
        f"  Correct Answer: {sub_correct}/{sub_total} ({sub_correct / sub_total * 100:.1f}%)"
    )
    print(
        f"  Chose Distractor: {sub_distractor}/{sub_total} ({sub_distractor / sub_total * 100:.1f}%)"
    )
    print(
        f"  Other Output: {sub_other}/{sub_total} ({sub_other / sub_total * 100:.1f}%)"
    )
    print("  --- Logprob Comparison ---")
    print(
        f"  Answer > Distractor: {sub_lp_correct}/{sub_total} ({sub_lp_correct / sub_total * 100:.1f}%)"
    )
    print(
        f"  Distractor > Answer: {sub_lp_distractor}/{sub_total} ({sub_lp_distractor / sub_total * 100:.1f}%)"
    )
    print(
        f"  Knowledge Robustness (Generative): {(1 - sub_distractor / sub_total) * 100:.1f}%"
    )
    print(
        f"  Knowledge Robustness (Logprob): {(1 - sub_lp_distractor / sub_total) * 100:.1f}%"
    )

    coh_total = results["coherent_conflict"]["total"]
    coh_correct = results["coherent_conflict"]["correct"]
    coh_distractor = results["coherent_conflict"]["distractor"]
    coh_other = results["coherent_conflict"]["other"]
    coh_lp_correct = results["coherent_conflict"]["logprob_correct"]
    coh_lp_distractor = results["coherent_conflict"]["logprob_distractor"]
    print("\nCoherent Conflict (Fluent passage with false claim):")
    print(f"  Total Samples: {coh_total}")
    print("  --- Generative ---")
    print(
        f"  Correct Answer: {coh_correct}/{coh_total} ({coh_correct / coh_total * 100:.1f}%)"
    )
    print(
        f"  Chose Distractor: {coh_distractor}/{coh_total} ({coh_distractor / coh_total * 100:.1f}%)"
    )
    print(
        f"  Other Output: {coh_other}/{coh_total} ({coh_other / coh_total * 100:.1f}%)"
    )
    print("  --- Logprob Comparison ---")
    print(
        f"  Answer > Distractor: {coh_lp_correct}/{coh_total} ({coh_lp_correct / coh_total * 100:.1f}%)"
    )
    print(
        f"  Distractor > Answer: {coh_lp_distractor}/{coh_total} ({coh_lp_distractor / coh_total * 100:.1f}%)"
    )
    print(
        f"  Knowledge Robustness (Generative): {(1 - coh_distractor / coh_total) * 100:.1f}%"
    )
    print(
        f"  Knowledge Robustness (Logprob): {(1 - coh_lp_distractor / coh_total) * 100:.1f}%"
    )


def main():
    parser = argparse.ArgumentParser(description="Evaluate Pythia on ParaConflict.")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit evaluation samples (useful for smoke tests on cluster)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override device (default: cuda if available, else mps/cpu)",
    )
    parser.add_argument(
        "--model-size",
        type=str,
        default=None,
        choices=list(PYTHIA_SIZES),
        help=(
            "Pythia size to evaluate. Defaults to $IC_FACTUAL_MODEL_SIZE or "
            f"the smallest staged size ({PYTHIA_SIZES[0]})."
        ),
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help=(
            "Explicit local snapshot path or Hub id (Qwen3, GPT-2, Ministral-3, …). "
            "Bypasses --model-size."
        ),
    )
    parser.add_argument(
        "--label",
        type=str,
        default=None,
        help="Human-readable label for JSONL metadata and plots.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default=None,
        help=(
            "Torch dtype for model weights "
            "(e.g. float32, float16/fp16, bfloat16/bf16). "
            "Recommended: float16 for 6.9b/12b on a single GPU."
        ),
    )
    parser.add_argument(
        "--per-sample-output",
        type=str,
        default=None,
        help=(
            "Path to a JSONL file. When set, dump one record per sample with "
            "memorized/in-context classifications for substitution + coherent "
            "conflicts (used by scripts/analyze_frequency_breakdown.py)."
        ),
    )
    parser.add_argument(
        "--categories",
        type=str,
        nargs="*",
        default=None,
        help=(
            "Optional ParaConflict categories to restrict to (case-insensitive). "
            "E.g. --categories 'World Capital' 'Official Language'. "
            "Default: all categories."
        ),
    )
    args = parser.parse_args()
    if args.model_path is not None and args.model_size is not None:
        raise SystemExit("--model-path and --model-size are mutually exclusive")
    evaluate_model(
        max_samples=args.max_samples,
        device=args.device,
        model_size=args.model_size,
        model_path=args.model_path,
        label=args.label,
        dtype=args.dtype,
        per_sample_output=args.per_sample_output,
        categories=args.categories,
    )


if __name__ == "__main__":
    main()
