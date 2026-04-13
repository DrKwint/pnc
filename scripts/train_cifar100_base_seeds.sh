#!/usr/bin/env bash
# Train CIFAR-100 PreActResNet18 base models for one or more seeds.
# Usage:  bash scripts/train_cifar100_base_seeds.sh [seed1 seed2 ...]
# Default: trains seeds 0, 1, 2.
#
# Each seed takes ~3 hours; runs strictly sequentially.
set -euo pipefail

PYTHON=".venv/bin/python"
LUIGI="$PYTHON -m luigi --module cifar_tasks --local-scheduler"

SEEDS=("$@")
if [[ ${#SEEDS[@]} -eq 0 ]]; then
  SEEDS=(0 1 2)
fi

for SEED in "${SEEDS[@]}"; do
  echo "============================================================"
  echo "  Train CIFAR-100 PreActResNet18 seed $SEED (~3 hours)"
  echo "============================================================"
  $LUIGI CIFARTrainPreActResNet18 \
      --dataset cifar100 --epochs 300 --seed $SEED
done
echo "All CIFAR-100 base seeds done."
