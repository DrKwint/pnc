#!/usr/bin/env bash
# Train Epinet with prior_scale=3.0 for seeds 1 and 2 (seed 0 already exists).
# Each seed trains on top of the existing PreActResNet18 base checkpoint for that seed.
set -euo pipefail

PYTHON=".venv/bin/python"
LUIGI="$PYTHON -m luigi --module cifar_tasks --local-scheduler"

for SEED in 1 2; do
  echo "============================================================"
  echo "  Train Epinet ps=3.0 seed $SEED"
  echo "============================================================"
  $LUIGI CIFARTrainEpinet \
      --dataset cifar10 --epochs 300 \
      --epinet-epochs 100 --epinet-lr 0.001 --epinet-wd 0.0001 \
      --index-dim 8 --epinet-hiddens '50,50' \
      --prior-scale 3.0 --seed $SEED
done
echo "All Epinet ps=3.0 seeds done."
