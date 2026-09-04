"""Per-layer logit-lens analysis on capitals context-sweep prompts.

For each condition (= JSONL input file produced by
``build_capitals_context_sweep.py``) we run the model with
``output_hidden_states=True`` and read off the first-token logprob of
the memorized capital and the in-context distractor at the **last
sequence position** at every layer (including the input embedding,
which we label as layer 0).

Output: a JSON record per condition with mean / median per-layer
logprobs for ``answer`` and ``distractor``, ``answer - distractor``
gap, plus the per-layer prob mass on each candidate.
We use standard ``transformers`` with ``output_hidden_states=True``
to extract per-layer hidden states with clean batching and no external dependencies.

Supports two architecture families out of the box:

* **GPT-NeoX / Pythia** (``model.gpt_neox.layers`` + ``final_layer_norm``
  + ``model.embed_out``);
* **Llama-like** (``model.model.layers`` + ``model.model.norm`` +
  ``model.lm_head``) — covers Qwen2/Qwen3, Llama, Mistral, etc.

Usage example::

    # Pythia (original exp13 invocation)
    python scripts/logit_lens_capitals.py \\
        --inputs experiments/exp01_*/data/input_L0_prose_tail_K1.jsonl \\
                 experiments/exp01_*/data/input_L128_prose_tail_K1.jsonl \\
                 experiments/exp11_*/data/input_L0_prose_tail_K1.jsonl \\
        --labels  L0_std L128_prose_std L0_negated \\
        --model-size 2.8b --dtype fp16 \\
        --max-samples 500 --batch-size 4 \\
        --output experiments/exp13_logit_lens/summaries/per_layer_28b.json

    # Qwen3-Base 4B (exp19 invocation)
    python scripts/logit_lens_capitals.py \\
        --inputs ... --labels ... \\
        --model-path "$IC_FACTUAL_MODELS_ROOT/qwen3-base-4b" \\
        --label qwen3-base-4b --dtype bf16 \\
        --output experiments/exp19_qwen3_logit_lens/summaries/per_layer_qwen3-base-4b.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from tqdm import tqdm

from ic_factual.config import PYTHIA_SIZES, resolve_model_size
from ic_factual.device import resolve_device
from ic_factual.loaders import load_causal_lm, load_tokenizer


def _resolve_arch(model) -> tuple[Any, Any, Any, str]:
    """Return ``(transformer_blocks, final_norm, lm_head, family)`` for ``model``.

    Supports GPT-NeoX (Pythia) and the standard ``model.model`` /
    ``model.lm_head`` Llama-like layout used by Qwen2, Qwen3, Llama,
    Mistral, etc.
    """
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        # GPT-2 (``GPT2LMHeadModel``): blocks in ``transformer.h``.
        return (
            model.transformer.h,
            model.transformer.ln_f,
            model.lm_head,
            "gpt2",
        )
    if hasattr(model, "gpt_neox"):
        return (
            model.gpt_neox.layers,
            model.gpt_neox.final_layer_norm,
            model.embed_out,
            "gpt_neox",
        )
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        # Standard HF causal-LM layout (Llama, Mistral, Qwen2, Qwen3, ...).
        inner = model.model
        final_norm = getattr(inner, "norm", None) or getattr(inner, "final_layer_norm")
        return inner.layers, final_norm, model.lm_head, "llama_like"
    if hasattr(model, "model") and hasattr(model.model, "language_model"):
        # Mistral3 / Ministral-3 multimodal wrapper (text-only eval path).
        inner = model.model.language_model
        if hasattr(inner, "layers"):
            final_norm = getattr(inner, "norm", None) or getattr(
                inner, "final_layer_norm", None
            )
            return inner.layers, final_norm, model.lm_head, "mistral3"
    raise RuntimeError(
        f"Unsupported model architecture for logit-lens: {type(model).__name__}. "
        "Expected GPT-NeoX (model.gpt_neox.*) or a Llama-like layout "
        "(model.model.layers + model.model.norm + model.lm_head)."
    )


def _read_records(path: Path, max_samples: int | None) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("_meta"):
                continue
            records.append(rec)
            if max_samples is not None and len(records) >= max_samples:
                break
    return records


def _first_token_id(tokenizer, text: str) -> int | None:
    ids = tokenizer.encode(" " + text, add_special_tokens=False)
    return ids[0] if ids else None


def _analyze_condition(
    path: Path,
    label: str,
    model,
    tokenizer,
    device: str,
    max_samples: int,
    batch_size: int,
) -> dict[str, Any]:
    records = _read_records(path, max_samples)
    if not records:
        return {"label": label, "input": str(path), "n_used": 0, "error": "no records"}

    layers, final_ln, lm_head, _family = _resolve_arch(model)
    num_blocks = len(layers)
    # transformers returns hidden_states tuple of length num_blocks + 1
    # (index 0 = embedding output, 1..num_blocks = block outputs).
    num_layers = num_blocks + 1

    # Sum and count per layer for memorized first-token and distractor first-token.
    ans_logprob_sum = [0.0] * num_layers
    dist_logprob_sum = [0.0] * num_layers
    ans_prob_sum = [0.0] * num_layers
    dist_prob_sum = [0.0] * num_layers
    n_used = 0
    n_dropped = 0

    # Pre-tokenize answer/distractor first tokens.  We process records
    # one at a time because prompts can have very different lengths and
    # we don't want to deal with padding masks for the last-token read.
    pbar = tqdm(records, desc=f"[{label}] {path.name}")
    for rec in pbar:
        prompt = rec["prompt"]
        answers = rec.get("answers") or []
        if isinstance(answers, str):
            answers = [answers]
        if not answers:
            n_dropped += 1
            continue
        answer = answers[0]
        distractor = rec["distractor"]

        a_id = _first_token_id(tokenizer, answer)
        d_id = _first_token_id(tokenizer, distractor)
        if a_id is None or d_id is None:
            n_dropped += 1
            continue

        enc = tokenizer(prompt, return_tensors="pt")
        input_ids = enc.input_ids.to(device)
        # Truncate to model context if necessary.
        max_len = getattr(model.config, "max_position_embeddings", 2048)
        if input_ids.shape[1] > max_len:
            input_ids = input_ids[:, -max_len:]

        with torch.no_grad():
            outputs = model(input_ids, output_hidden_states=True)
        hidden_states = outputs.hidden_states  # tuple length num_layers
        assert len(hidden_states) == num_layers, (
            f"Expected {num_layers} hidden states, got {len(hidden_states)} "
            f"for prompt of length {input_ids.shape[1]}"
        )

        for i, h in enumerate(hidden_states):
            last = h[0, -1, :]
            normed = final_ln(last)
            logits = lm_head(normed)
            log_probs = F.log_softmax(logits.float(), dim=-1)
            probs = log_probs.exp()
            ans_logprob_sum[i] += float(log_probs[a_id].item())
            dist_logprob_sum[i] += float(log_probs[d_id].item())
            ans_prob_sum[i] += float(probs[a_id].item())
            dist_prob_sum[i] += float(probs[d_id].item())

        n_used += 1

    mean_ans_lp = [s / max(n_used, 1) for s in ans_logprob_sum]
    mean_dist_lp = [s / max(n_used, 1) for s in dist_logprob_sum]
    mean_ans_p = [s / max(n_used, 1) for s in ans_prob_sum]
    mean_dist_p = [s / max(n_used, 1) for s in dist_prob_sum]
    gap_lp = [a - d for a, d in zip(mean_ans_lp, mean_dist_lp)]

    return {
        "label": label,
        "input": str(path),
        "n_used": n_used,
        "n_dropped": n_dropped,
        "num_layers": num_layers,
        "mean_answer_logprob": mean_ans_lp,
        "mean_distractor_logprob": mean_dist_lp,
        "mean_answer_prob": mean_ans_p,
        "mean_distractor_prob": mean_dist_p,
        "mean_gap_logprob": gap_lp,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Per-layer logit lens on capitals context-sweep inputs."
    )
    parser.add_argument(
        "--inputs",
        type=str,
        nargs="+",
        required=True,
        help="One or more JSONL input files (output of build_capitals_context_sweep.py).",
    )
    parser.add_argument(
        "--labels",
        type=str,
        nargs="+",
        required=True,
        help=(
            "Short label per input file (must match --inputs in length).  "
            "Used as dict keys in the output JSON."
        ),
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to write the JSON summary (one entry per --labels item).",
    )
    parser.add_argument(
        "--model-size",
        type=str,
        default=None,
        choices=list(PYTHIA_SIZES),
        help="Pythia size shortcut. Mutually exclusive with --model-path.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help=(
            "Explicit local snapshot path or Hub id. Bypasses --model-size "
            "and the Pythia-only size lookup; use this for Qwen3 / Llama-like "
            "families. Mutually exclusive with --model-size."
        ),
    )
    parser.add_argument(
        "--label",
        type=str,
        default=None,
        help=(
            "Human-readable label written to the JSON payload (used by plot "
            "scripts for panel titles). Defaults to 'pythia-<size>' or the "
            "basename of --model-path."
        ),
    )
    parser.add_argument("--dtype", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=500,
        help="Per condition.  500 is enough for stable per-layer means.",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args()

    if len(args.inputs) != len(args.labels):
        raise SystemExit(
            f"--inputs ({len(args.inputs)}) and --labels ({len(args.labels)}) "
            f"must have equal length"
        )
    if args.model_path is not None and args.model_size is not None:
        raise SystemExit("--model-path and --model-size are mutually exclusive")

    device = resolve_device(args.device)
    if args.model_path is not None:
        size = None
        model_label = args.label or Path(args.model_path).name or args.model_path
    else:
        size = resolve_model_size(args.model_size)
        model_label = args.label or f"pythia-{size}"

    print(f"Using device: {device}")
    print(f"Model label:  {model_label}")
    print(f"Model source: {'--model-path=' + args.model_path if args.model_path else 'pythia-' + size}")
    print(f"Dtype:        {args.dtype}")

    if args.model_path is not None:
        tokenizer = load_tokenizer(args.model_path)
        model, device = load_causal_lm(
            args.model_path, device=device, dtype=args.dtype
        )
    else:
        tokenizer = load_tokenizer(model_size=size)
        model, device = load_causal_lm(
            device=device, model_size=size, dtype=args.dtype
        )
    layers, _final_norm, _lm_head, family = _resolve_arch(model)
    print(
        f"Architecture: {family}  ({len(layers)} transformer blocks "
        f"+1 embedding read-off layer = {len(layers) + 1} logit-lens layers)"
    )

    results: list[dict[str, Any]] = []
    for inp, lab in zip(args.inputs, args.labels):
        path = Path(inp)
        if not path.exists():
            print(f"[warn] {path} does not exist, skipping {lab}")
            results.append({"label": lab, "input": str(path), "n_used": 0, "error": "missing"})
            continue
        r = _analyze_condition(
            path,
            lab,
            model,
            tokenizer,
            device,
            args.max_samples,
            args.batch_size,
        )
        results.append(r)
        print(
            f"  {lab}: n_used={r.get('n_used')}  "
            f"layers={r.get('num_layers')}  "
            f"final-layer gap(ans-dist)={r.get('mean_gap_logprob', [0])[-1]:.3f}"
        )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_size": size if size is not None else model_label,
        "label": model_label,
        "family": family,
        "model_path": args.model_path,
        "num_blocks": len(layers),
        "dtype": args.dtype,
        "device": str(device),
        "max_samples_per_condition": args.max_samples,
        "conditions": results,
    }
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\nWrote per-layer logit-lens summary to {out_path}")


if __name__ == "__main__":
    main()
