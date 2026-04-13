#!/usr/bin/env bash
# Run all CIFAR-10 OpenOOD evaluations across one or more seeds.
# Usage:  bash scripts/run_ood_eval_multi_seed.sh [seed1 seed2 ...]
# Example: bash scripts/run_ood_eval_multi_seed.sh 1 2
#
# Frozen ID hyperparameters from the strengthening plan.
# All methods run sequentially (one at a time) to avoid GPU thrashing.
set -euo pipefail

PYTHON=".venv/bin/python"
LUIGI="$PYTHON -m luigi --module cifar_tasks --local-scheduler"
EPOCHS=300

SEEDS=("$@")
if [[ ${#SEEDS[@]} -eq 0 ]]; then
  SEEDS=(0 1 2)
fi

for SEED in "${SEEDS[@]}"; do
  echo "============================================================"
  echo "  SEED $SEED"
  echo "============================================================"

  # Single-model OOD baselines (cheap, run first)
  $LUIGI CIFAROpenOODPreActResNet18 --posthoc-calibrate --epochs $EPOCHS --seed $SEED
  $LUIGI CIFAROpenOODMSP            --posthoc-calibrate --epochs $EPOCHS --seed $SEED
  $LUIGI CIFAROpenOODEnergy         --posthoc-calibrate --epochs $EPOCHS --seed $SEED
  $LUIGI CIFAROpenOODMahalanobis    --posthoc-calibrate --epochs $EPOCHS --seed $SEED
  $LUIGI CIFAROpenOODReActEnergy    --posthoc-calibrate --epochs $EPOCHS --seed $SEED

  # Stochastic / ensemble UQ methods
  $LUIGI CIFAROpenOODMCDropout         --posthoc-calibrate --epochs $EPOCHS --n-perturbations 32 --dropout-rate 0.1 --seed $SEED
  $LUIGI CIFAROpenOODLLLA              --posthoc-calibrate --epochs $EPOCHS --n-perturbations 50 --prior-precision 10.0 --seed $SEED
  $LUIGI CIFAROpenOODEpinet            --posthoc-calibrate --epochs $EPOCHS --n-perturbations 50 --prior-scale 3.0 --seed $SEED
  $LUIGI CIFAROpenOODSWAG              --posthoc-calibrate --epochs $EPOCHS --n-perturbations 50 --swag-start-epoch 240 --seed $SEED

  # PnC variants
  # Single-block PnC: scale=25.0 chosen by P1.b sanity sweep (strictly beats 20 on
  # all OOD metrics, ID metrics flat).
  $LUIGI CIFAROpenOODPnC \
      --posthoc-calibrate --epochs $EPOCHS \
      --perturbation-sizes '[25.0]' --n-directions 20 --random-directions --seed $SEED

  $LUIGI CIFAROpenOODMultiBlockPnC \
      --posthoc-calibrate --epochs $EPOCHS \
      --perturbation-sizes '[7.0]' --n-directions 20 --random-directions --chunk-size 64 --seed $SEED

  # Deep Ensemble (last because it depends on multiple base checkpoints)
  $LUIGI CIFAROpenOODStandardEnsemble  --posthoc-calibrate --epochs $EPOCHS --n-models 5 --seed $SEED

  echo "Seed $SEED done."
done

echo "All seeds complete."
