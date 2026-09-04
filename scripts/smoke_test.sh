#!/usr/bin/env bash
# Representative smoke test: 8 capitals models + 5 logit-lens checkpoints across
# Pythia, GPT-2, Qwen3, and Ministral-3 (no 2.8b/4b — avoids long Hub/MPS stalls).
# Defaults: 35 countries, 400 prompts, 100 eval samples, 50 logit-lens samples.
# Expect ~20–45 min depending on GPU and Hub cache.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
chmod +x scripts/reproduce.sh
./scripts/reproduce.sh smoke
echo "Smoke test passed (all paper figures synced to paper/figures/)."
