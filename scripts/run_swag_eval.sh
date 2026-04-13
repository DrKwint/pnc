#!/bin/bash
# Phase 0: SWAG evaluation with fixed BN refresh (cumulative averaging).
# Requires SWAG training checkpoints to already exist (sws=240).

set -e

VENV=".venv/bin/python"

echo "=== Phase 0: SWAG Eval (BN refresh fixed) ==="

for SEED in 0 1 2; do
  echo "--- Seed $SEED ---"
  $VENV -m luigi --module cifar_tasks CIFARPreActSWAG \
    --dataset cifar10 --epochs 300 --n-perturbations 50 \
    --swag-start-epoch 240 --swag-collect-freq 1 \
    --swag-use-bn-refresh --bn-refresh-subset-size 2048 \
    --seed $SEED --posthoc-calibrate --local-scheduler
done

echo "=== SWAG eval complete ==="
