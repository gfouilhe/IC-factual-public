#!/usr/bin/env bash
# End-to-end reproduction of all paper figures and appendix table.
#
# Usage (from repository root):
#   ./scripts/reproduce.sh test-pythia # test locally on Pythia-160m (fast local CPU/MPS test)
#   ./scripts/reproduce.sh figures     # replot + sync all 12 paper figures + all tables
#   ./scripts/reproduce.sh smoke       # multi-family mini-sweep (representative plots)
#   ./scripts/reproduce.sh figure1     # Fig. 1: context_memory_score_gen_all
#   ./scripts/reproduce.sh figure2     # Fig. 2: exp22_gen_length
#   ./scripts/reproduce.sh figure3     # Fig. 3: logit_lens_family_gap_summary
#   ./scripts/reproduce.sh appendix    # Appendix figs + LaTeX tables
#   ./scripts/reproduce.sh all         # full paper reproduction
#
# Environment:
#   MAX_SAMPLES          cap prompts per capitals eval (optional)
#   MAX_SAMPLES_PER_COND logit-lens prompts per condition (default 500)
#   SKIP_DATA            1 => skip dataset build steps
#   SKIP_EVAL            1 => skip GPU eval loops
#   MODEL_FILTER         substring; only eval labels containing it
#   FORCE_REEVAL         1 => re-run even when per-sample JSONL exists

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if command -v uv >/dev/null 2>&1; then
  RUN=(uv run python)
else
  RUN=(python)
fi

MAX_SAMPLES="${MAX_SAMPLES:-}"
MAX_SAMPLES_PER_COND="${MAX_SAMPLES_PER_COND:-500}"
SKIP_DATA="${SKIP_DATA:-0}"
SKIP_EVAL="${SKIP_EVAL:-0}"
MODEL_FILTER="${MODEL_FILTER:-}"
BATCH_SIZE="${BATCH_SIZE:-}"
FORCE_REEVAL="${FORCE_REEVAL:-0}"

mkdir -p data data/logit_lens results/per_sample results/logit_lens figures paper/figures

eval_batch_size() {
  local label="$1"
  if [[ -n "$BATCH_SIZE" ]]; then
    echo "$BATCH_SIZE"
    return
  fi
  case "$label" in
    1.4b|2.8b|6.9b|12b|qwen3-*|ministral3-*|gpt2-medium|gpt2-large|gpt2-xl)
      echo "4"
      ;;
    *)
      echo "16"
      ;;
  esac
}

resolve_dtype() {
  local label="$1"
  case "$label" in
    2.8b|6.9b|12b|qwen3-32b) echo "fp16" ;;
    qwen3*|ministral3*) echo "bf16" ;;
    gpt2-xl) echo "fp16" ;;
    *) echo "" ;;
  esac
}

should_run_label() {
  local label="$1"
  if [[ -z "$MODEL_FILTER" ]]; then
    return 0
  fi
  [[ "$label" == *"$MODEL_FILTER"* ]]
}

labels_from_config() {
  local var_name="$1"
  "${RUN[@]}" -c "from ic_factual.config import ${var_name}; print(' '.join(${var_name}))"
}

build_data() {
  local smoke_mode="${1:-0}"
  echo "==> Building capitals cross-product"
  local build_args=(scripts/build_capitals_crossproduct.py --output data/capitals_crossproduct.jsonl)
  if [[ "$smoke_mode" == "1" ]]; then
    build_args+=(--max-countries 35 --max-prompts 400)
  elif [[ -n "${MAX_SAMPLES:-}" ]]; then
    build_args+=(--max-prompts "$MAX_SAMPLES")
  fi
  "${RUN[@]}" "${build_args[@]}"

  echo "==> Computing country Zipf frequencies"
  "${RUN[@]}" scripts/compute_subject_frequency.py \
    --proxy wordfreq \
    --from-jsonl data/capitals_crossproduct.jsonl \
    --subject-fields country \
    --output data/capitals_subject_frequency.json

  echo "==> Building logit-lens inputs"
  local lens_args=(scripts/build_logit_lens_inputs.py)
  if [[ "$smoke_mode" == "1" ]]; then
    lens_args+=(--max-samples "$MAX_SAMPLES_PER_COND")
  elif [[ -n "${MAX_SAMPLES:-}" ]]; then
    lens_args+=(--max-samples "$MAX_SAMPLES")
  fi
  "${RUN[@]}" "${lens_args[@]}"
}

eval_capitals_label() {
  local label="$1"
  local out="results/per_sample/per_sample_capitals_${label}.jsonl"
  if [[ -s "$out" && "$FORCE_REEVAL" != "1" ]]; then
    echo "  skip ${label} (exists)"
    return 0
  fi

  local dtype
  dtype="$(resolve_dtype "$label")"

  local eval_args=(
    scripts/evaluate_capitals.py
    --input data/capitals_crossproduct.jsonl
    --output "$out"
    --method generative
    --label "$label"
    --batch-size "$(eval_batch_size "$label")"
  )
  if [[ -n "$dtype" ]]; then
    eval_args+=(--dtype "$dtype")
  fi
  if [[ -n "$MAX_SAMPLES" ]]; then
    eval_args+=(--max-samples "$MAX_SAMPLES" --shuffle-seed 0)
  fi

  if [[ "$label" =~ ^(160m|410m|1b|1\.4b|2\.8b|6\.9b|12b)$ ]]; then
    eval_args+=(--model-size "$label")
  else
    local hub
    hub="$("${RUN[@]}" -c "from ic_factual.config import resolve_hub_id; print(resolve_hub_id('${label}'))")"
    eval_args+=(--model-path "$hub")
  fi

  echo "  eval ${label}"
  "${RUN[@]}" "${eval_args[@]}"
}

run_eval_labels() {
  local config_var="$1"
  local labels
  labels="$(labels_from_config "$config_var")"
  for label in $labels; do
    should_run_label "$label" || continue
    eval_capitals_label "$label"
  done
}

run_figure1_eval() {
  echo "==> Figure 1: generative capitals eval"
  run_eval_labels FIGURE1_MODEL_LABELS
}

logit_lens_label() {
  local label="$1"
  local out="results/logit_lens/per_layer_${label}.json"
  if [[ -s "$out" && "$FORCE_REEVAL" != "1" ]]; then
    echo "  skip logit-lens ${label} (exists)"
    return 0
  fi

  local dtype
  dtype="$(resolve_dtype "$label")"

  local ll_args=(
    scripts/logit_lens_capitals.py
    --inputs
    data/logit_lens/input_L0_std.jsonl
    data/logit_lens/input_L0_neg.jsonl
    data/logit_lens/input_L128_prose_std.jsonl
    data/logit_lens/input_L128_prose_neg.jsonl
    --labels L0_std L0_neg L128_prose_std L128_prose_neg
    --output "$out"
    --label "$label"
    --max-samples "$MAX_SAMPLES_PER_COND"
  )
  if [[ -n "$dtype" ]]; then
    ll_args+=(--dtype "$dtype")
  fi

  if [[ "$label" =~ ^(160m|410m|1b|1\.4b|2\.8b|6\.9b|12b)$ ]]; then
    ll_args+=(--model-size "$label")
  else
    local hub
    hub="$("${RUN[@]}" -c "from ic_factual.config import resolve_hub_id; print(resolve_hub_id('${label}'))")"
    ll_args+=(--model-path "$hub")
  fi

  echo "  logit-lens ${label}"
  "${RUN[@]}" "${ll_args[@]}"
}

run_logit_lens_labels() {
  local config_var="$1"
  local labels
  labels="$(labels_from_config "$config_var")"
  for label in $labels; do
    should_run_label "$label" || continue
    logit_lens_label "$label"
  done
}

run_figure2_eval() {
  echo "==> Figure 2: logit-lens eval"
  run_logit_lens_labels FIGURE2_MODEL_LABELS
}

plot_figure1() {
  echo "==> Figure 1: context_memory_score_gen_all"
  "${RUN[@]}" scripts/plot_context_memory_score.py \
    --summary-json summaries/context_memory_score_gen_summary.json \
    --output-prefix figures/context_memory_score_gen_all \
    --metric gen
}

plot_figure2() {
  echo "==> Figure 2: exp22_gen_length"
  "${RUN[@]}" scripts/analyze_context_sweep.py \
    --summary-json summaries/exp22_gen_main_summary.json \
    --output-prefix figures/exp22_gen \
    --compact-rate mem \
    --metric gen
}

plot_figure3() {
  echo "==> Figure 3: logit_lens_family_gap_summary"
  "${RUN[@]}" scripts/plot_logit_lens_family_summary.py \
    --output figures/logit_lens_family_gap_summary.png
}

plot_clean_recall() {
  echo "==> Figures 4 & 5: clean_factual_recall"
  "${RUN[@]}" scripts/plot_clean_factual_recall.py --metric gen --output-png figures/clean_factual_recall.png
  "${RUN[@]}" scripts/plot_clean_factual_recall.py --metric lp --output-png figures/clean_factual_recall.png
}

plot_exp22_full() {
  echo "==> Figure 6: exp22_gen_length_full"
  "${RUN[@]}" scripts/analyze_context_sweep.py \
    --summary-json summaries/exp22_gen_full_summary.json \
    --output-prefix figures/exp22_gen_full \
    --metric gen
  cp -f figures/exp22_gen_full_length.png figures/exp22_gen_length_full.png
}

plot_prose_l128() {
  echo "==> Figure 7: cross_family_prose_l128_delta & Table"
  "${RUN[@]}" scripts/analyze_prose_l128.py \
    --summary-json summaries/exp27_prose_l128_summary.json
}

plot_position() {
  echo "==> Figure 8: exp02_position_sweep"
  "${RUN[@]}" scripts/plot_paper_position.py
  cp -f paper/figures/exp02_position_sweep.png figures/exp02_position_sweep.png
}

run_appendix() {
  echo "==> Figure 9: country-frequency figures + table"
  "${RUN[@]}" scripts/test_country_freq_memorization.py \
    --summary-json summaries/country_freq_mem_hypothesis_summary.json \
    --output-prefix figures/country_freq_mem
  "${RUN[@]}" scripts/export_country_freq_appendix_tex.py \
    --summary-json summaries/country_freq_mem_hypothesis_summary.json
}

plot_template() {
  echo "==> Figure 10: exp12_template_comparison"
  "${RUN[@]}" scripts/plot_template_comparison.py
}

plot_phrasing() {
  echo "==> Figure 11: cross_family_phrasing_swings & Table"
  "${RUN[@]}" scripts/analyze_phrasing.py \
    --summary-json summaries/exp26_phrasing_summary.json
}

plot_copies() {
  echo "==> Figure 12: exp07_copies"
  "${RUN[@]}" scripts/analyze_context_sweep.py \
    --summary-json summaries/exp07_summary.json \
    --output-prefix figures/exp07 \
    --metric lp
}

sync_paper_figures() {
  echo "==> Syncing figures/ -> paper/figures/"
  mkdir -p paper/figures
  while IFS= read -r line; do
    [[ -z "$line" ]] && continue
    src="${line%% *}"
    dst="${line#* }"
    if [[ ! -f "$src" ]]; then
      echo "  WARNING: missing $src (skip)" >&2
      continue
    fi
    cp -f "$src" "$dst"
    echo "  -> $dst"
  done < <("${RUN[@]}" -c "
from ic_factual.config import PAPER_FIGURE_SYNCS
for src, dst in PAPER_FIGURE_SYNCS:
    print(f'{src} {dst}')
")
}

sync_paper_tables() {
  echo "==> Syncing tables/ -> paper/tables/"
  mkdir -p tables paper/tables
  for f in paper/tables/*.tex; do
    [[ -f "$f" ]] && cp -f "$f" "tables/$(basename "$f")"
  done
  while IFS= read -r line; do
    [[ -z "$line" ]] && continue
    src="${line%% *}"
    dst="${line#* }"
    if [[ ! -f "$src" ]]; then
      echo "  WARNING: missing $src (skip)" >&2
      continue
    fi
    cp -f "$src" "$dst"
    echo "  -> $dst"
  done < <("${RUN[@]}" -c "
from ic_factual.config import PAPER_TABLE_SYNCS
for src, dst in PAPER_TABLE_SYNCS:
    print(f'{src} {dst}')
")
}

plot_all_figures() {
  mkdir -p figures tables paper/figures paper/tables
  plot_figure1
  plot_figure2
  plot_figure3
  plot_clean_recall
  plot_exp22_full
  plot_prose_l128
  plot_position
  run_appendix
  plot_template
  plot_phrasing
  plot_copies
  sync_paper_figures
  sync_paper_tables
}

run_test_pythia() {
  echo "============================================================"
  echo "Running local test on Pythia-160m (single small model)"
  echo "============================================================"
  mkdir -p data results/per_sample results/logit_lens figures tables

  echo "1. Building small capitals dataset (5 countries)"
  "${RUN[@]}" scripts/build_capitals_crossproduct.py \
    --output data/capitals_test_160m.jsonl \
    --max-countries 5

  echo "2. Testing generative evaluation on Pythia-160m"
  "${RUN[@]}" scripts/evaluate_capitals.py \
    --model-size 160m \
    --input data/capitals_test_160m.jsonl \
    --output results/per_sample_capitals_160m.jsonl \
    --method generative \
    --batch-size 8

  echo "3. Testing logprob evaluation on Pythia-160m"
  "${RUN[@]}" scripts/evaluate_capitals.py \
    --model-size 160m \
    --input data/capitals_test_160m.jsonl \
    --output results/per_sample_capitals_160m_lp.jsonl \
    --method logprob \
    --batch-size 8

  echo "4. Testing ParaConflict evaluation on Pythia-160m (5 samples)"
  "${RUN[@]}" evaluate_model.py \
    --model-size 160m \
    --max-samples 5 \
    --per-sample-output results/per_sample_paraconflict_160m.jsonl

  echo "5. Testing logit lens on Pythia-160m"
  "${RUN[@]}" scripts/logit_lens_capitals.py \
    --inputs data/capitals_test_160m.jsonl \
    --labels test_cond \
    --model-size 160m \
    --max-samples 5 \
    --batch-size 4 \
    --output results/logit_lens/per_layer_160m_test.json

  echo "============================================================"
  echo "Pythia-160m local test PASSED successfully!"
  echo "============================================================"
}

cmd="${1:-all}"
case "$cmd" in
  test-pythia)
    run_test_pythia
    ;;
  smoke)
    MAX_SAMPLES="${MAX_SAMPLES:-100}"
    MAX_SAMPLES_PER_COND="${MAX_SAMPLES_PER_COND:-50}"
    if [[ "$SKIP_DATA" != "1" ]]; then
      build_data 1
    fi
    if [[ "$SKIP_EVAL" != "1" ]]; then
      echo "==> Smoke Figure 1 eval"
      run_eval_labels SMOKE_FIGURE1_MODEL_LABELS
      echo "==> Smoke Figure 2 eval"
      run_logit_lens_labels SMOKE_FIGURE2_MODEL_LABELS
    fi
    plot_all_figures
    ;;
  figures)
    plot_all_figures
    ;;
  data)
    build_data 0
    ;;
  figure1)
    if [[ "$SKIP_DATA" != "1" ]]; then build_data 0; fi
    if [[ "$SKIP_EVAL" != "1" ]]; then run_figure1_eval; fi
    plot_figure1
    sync_paper_figures
    ;;
  figure2)
    plot_figure2
    sync_paper_figures
    ;;
  figure3)
    if [[ "$SKIP_DATA" != "1" ]]; then build_data 0; fi
    if [[ "$SKIP_EVAL" != "1" ]]; then run_figure2_eval; fi
    plot_figure3
    sync_paper_figures
    ;;
  appendix)
    run_appendix
    sync_paper_figures
    sync_paper_tables
    ;;
  all)
    if [[ "$SKIP_DATA" != "1" ]]; then build_data 0; fi
    if [[ "$SKIP_EVAL" != "1" ]]; then
      run_figure1_eval
      run_figure2_eval
    fi
    plot_all_figures
    ;;
  -h|--help)
    sed -n '2,14p' "$0"
    ;;
  *)
    echo "Unknown command: $cmd" >&2
    exit 2
    ;;
esac

echo "Done ($cmd)."
