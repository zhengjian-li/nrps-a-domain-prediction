#!/usr/bin/env bash
# Demo: end-to-end run of the NRPS A-domain pipeline.
# Run from the repo root:  bash script/demo.sh
set -euo pipefail

cd "$(dirname "$0")/.."   # repo root

echo "==> [0/3] Installing dependencies (uv sync)"
uv sync

echo
echo "==> [1/3] Residue-position selection (quick demo: fungal, information gain, small k-sweep)"
uv run python script/featurisation/pipeline_k_selection.py \
    --train-kingdom fungal \
    --test-kingdom  fungal \
    --method        ig \
    --min-k 5 --max-k 15 --step 1 \
    --cv-folds 3 \
    --n-seeds 1
# Outputs → data/residue/pipeline_k_selection_peak_benchmark_substrates/

echo
echo "==> [2/3] Train the Random Forest classifiers (all residue sets x kingdoms)"
uv run python script/model_training/rf_substrate_classifier.py
# Models  → model/rf_substrate_<set>_<kingdom>.pkl
# Summary → data/results_rf_substrate.tsv

echo
echo "==> [3/3] Benchmark summary"
column -t -s$'\t' data/results_rf_substrate.tsv

echo
echo "Demo complete."
