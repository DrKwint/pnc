#!/bin/bash
# Phase 0: Epinet baseline (Osband et al., 2023)
# Train epinet on frozen PreAct ResNet-18, evaluate with n=50 members.
# Prior scale sweep: 0.5, 1.0, 3.0 (select best by posthoc NLL).

set -e

VENV=".venv/bin/python"

echo "=== Phase 0e: Epinet Baseline ==="

for SEED in 0 1 2; do
  for PRIOR_SCALE in 0.5 1.0 3.0; do
    echo "--- Seed $SEED, prior_scale=$PRIOR_SCALE ---"
    $VENV -m luigi --module cifar_tasks CIFARPreActEpinet \
      --dataset cifar10 --epochs 300 --epinet-epochs 100 \
      --epinet-lr 1e-3 --epinet-wd 1e-4 \
      --n-perturbations 50 --index-dim 8 \
      --epinet-hiddens '50,50' --prior-scale $PRIOR_SCALE \
      --seed $SEED --posthoc-calibrate --local-scheduler
  done
done

echo "=== Epinet baseline complete ==="
