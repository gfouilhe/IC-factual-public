# ic-factual (public)

Reproduce all figures and tables from the paper on **Memory vs. Context? Influential Factors of Factual Recall in Language Models** using local Python/bash. Models and datasets are downloaded from the Hugging Face Hub on first use.

## Installation

```bash
uv sync
```

Dependencies: Python ≥3.10, PyTorch, Transformers, Accelerate, Datasets, Matplotlib, SciPy, PyCairo.
---

## 1. Fast Local Test (~15–30 s on Apple Silicon / CPU)

Verify the complete evaluation and logit lens pipeline locally using a single small model (`pythia-160m`):

```bash
./scripts/reproduce.sh test-pythia
```

This tests end-to-end:
1. Building capitals prompt cross-product (`scripts/build_capitals_crossproduct.py`).
2. Generative evaluation loop (`scripts/evaluate_capitals.py --method generative`).
3. Logprob evaluation loop (`scripts/evaluate_capitals.py --method logprob`).
4. ParaConflict evaluation (`evaluate_model.py`).
5. Logit-lens layer extraction without external dependencies (`scripts/logit_lens_capitals.py`).

---

## 2. Replot All 12 Paper Figures and 10 Tables

To regenerate all 12 figures and all 10 tables from precomputed experiment summaries and sync them to `paper/figures/` and `paper/tables/`:

```bash
./scripts/reproduce.sh figures
```

### Paper Artifacts Synced

| Figure | Description | Paper File |
|--------|-------------|------------|
| **Fig. 1** | Context vs. Memorization Score across 31 models | `paper/figures/context_memory_score_gen_all.png` |
| **Fig. 2** | Generative filler length sweep (P(mem) vs length) | `paper/figures/exp22_gen_length.png` |
| **Fig. 3** | Cross-family logit lens layer gap summary | `paper/figures/logit_lens_family_gap_summary.png` |
| **Fig. 4** | Clean factual recall accuracy across relations (generative) | `paper/figures/clean_factual_recall.png` |
| **Fig. 5** | Clean factual recall accuracy across relations (logprob) | `paper/figures/clean_factual_recall_lp.png` |
| **Fig. 6** | Generative filler sweep full breakdown (P(mem), P(ctx), P(other)) | `paper/figures/exp22_gen_length_full.png` |
| **Fig. 7** | Cross-family prose length effect ($\Delta P(\mathrm{mem})$ at $L=128$) | `paper/figures/cross_family_prose_l128_delta.png` |
| **Fig. 8** | Conflict position sweep (head vs. tail) | `paper/figures/exp02_position_sweep.png` |
| **Fig. 9** | Country Zipf frequency vs. memorization (deciles + delta) | `paper/figures/country_freq_mem_combined.png` |
| **Fig. 10** | Alternative scaffold template comparison | `paper/figures/exp12_template_comparison.png` |
| **Fig. 11** | Cross-family phrasing sensitivity swings | `paper/figures/cross_family_phrasing_swings.png` |
| **Fig. 12** | Redundancy / repetition sweep ($K \in \{1, 2, 4, 8\}$ copies at $L=0$) | `paper/figures/exp07_copies.png` |

All 10 LaTeX table fragments (`tables/*.tex`) are synced to `paper/tables/`:
- `clean_factual_recall.tex`, `clean_factual_recall_lp.tex`
- `country_freq_mem_hypothesis.tex`, `paraconflict_native_sub_vs_coh.tex`, `paraconflict_entity_frequency.tex`
- `cross_family_phrasing.tex`, `cross_family_prose_l128.tex`
- `relations_posttraining_by_size.tex`, `relations_size_mem_by_family.tex`
- `model_inference_specs.tex`

## 3. Generate Datasets

To generate the full evaluation prompt datasets, logit-lens inputs, and country Zipf frequencies from scratch:

```bash
./scripts/reproduce.sh data
```

---

## 4. Full Evaluation Sweeps (Requires GPU)

To re-run model inference across model families from scratch:

```bash
# Quick smoke test on representative models across families:
./scripts/reproduce.sh smoke

# Full paper evaluation across all 31 models:
./scripts/reproduce.sh all
```

Model evaluations automatically leverage `CUDA` (if available), `MPS` (Apple Silicon), or `CPU`.

---

## Repository Structure

```
.
├── ic_factual/               # Core library (catalogue, config, loaders, device handling)
├── scripts/                  # Evaluation, plotting, and export scripts
├── summaries/                # Precomputed evaluation summaries & per-layer logit-lens data
├── paper/                    # Paper figures (12 PNGs) and tables (10 LaTeX fragments)
├── figures/                  # Local figures output directory (gitignored)
├── tables/                   # Local tables output directory (gitignored)
├── data/                     # Generated evaluation prompts & datasets (gitignored)
└── results/                  # Evaluation outputs and per-sample logs (gitignored)
```
