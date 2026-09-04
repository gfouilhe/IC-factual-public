# ic-factual (public)

Reproduce all figures and tables from the paper on **Memory vs. Context? Influential Factors of Factual Recall in Language Models**. Models and datasets are downloaded from the Hugging Face Hub on first use.

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
