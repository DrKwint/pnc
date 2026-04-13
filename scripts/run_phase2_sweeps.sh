#!/bin/bash
# Phase 2: Per-Block Scale Discovery (Single-Block PnC Sweeps)
# Post-patch-fix version with wider scale ranges
# Run from pnc/ directory: bash scripts/run_phase2_sweeps.sh
set -e

cd "$(dirname "$0")/.."

echo "=========================================="
echo "Phase 2a: Stage 4 Blocks (largest)"
echo "=========================================="
# Stage 4 Block 1 (last block)
python -m luigi --module cifar_tasks CIFARPnC \
  --dataset cifar10 --epochs 300 --n-directions 10 --n-perturbations 50 \
  --perturbation-sizes '[1.0,5.0,10.0,50.0,100.0,200.0,500.0]' \
  --target-stage-idx 3 --target-block-idx 1 \
  --lambda-reg 1e-3 --subset-size 1024 --chunk-size 64 \
  --seed 0 --posthoc-calibrate --local-scheduler

# Stage 4 Block 0
python -m luigi --module cifar_tasks CIFARPnC \
  --dataset cifar10 --epochs 300 --n-directions 10 --n-perturbations 50 \
  --perturbation-sizes '[1.0,5.0,10.0,50.0,100.0,200.0,500.0]' \
  --target-stage-idx 3 --target-block-idx 0 \
  --lambda-reg 1e-3 --subset-size 1024 --chunk-size 64 \
  --seed 0 --posthoc-calibrate --local-scheduler

echo "=========================================="
echo "Phase 2b: Stage 3 Blocks"
echo "=========================================="
for BLOCK in 0 1; do
  python -m luigi --module cifar_tasks CIFARPnC \
    --dataset cifar10 --epochs 300 --n-directions 15 --n-perturbations 50 \
    --perturbation-sizes '[5.0,10.0,50.0,100.0,200.0,500.0,1000.0]' \
    --target-stage-idx 2 --target-block-idx $BLOCK \
    --lambda-reg 1e-3 --subset-size 1024 --chunk-size 64 \
    --seed 0 --posthoc-calibrate --local-scheduler
done

echo "=========================================="
echo "Phase 2c: Stage 2 Blocks"
echo "=========================================="
for BLOCK in 0 1; do
  python -m luigi --module cifar_tasks CIFARPnC \
    --dataset cifar10 --epochs 300 --n-directions 20 --n-perturbations 50 \
    --perturbation-sizes '[10.0,50.0,100.0,200.0,500.0,1000.0,2000.0]' \
    --target-stage-idx 1 --target-block-idx $BLOCK \
    --lambda-reg 1e-3 --subset-size 1024 --chunk-size 64 \
    --seed 0 --posthoc-calibrate --local-scheduler
done

echo "=========================================="
echo "Phase 2d: Stage 1 Blocks"
echo "=========================================="
for BLOCK in 0 1; do
  python -m luigi --module cifar_tasks CIFARPnC \
    --dataset cifar10 --epochs 300 --n-directions 20 --n-perturbations 50 \
    --perturbation-sizes '[50.0,100.0,200.0,500.0,1000.0,2000.0,5000.0]' \
    --target-stage-idx 0 --target-block-idx $BLOCK \
    --lambda-reg 1e-3 --subset-size 1024 --chunk-size 64 \
    --seed 0 --posthoc-calibrate --local-scheduler
done

echo "=========================================="
echo "Phase 2 Sweeps Complete"
echo "=========================================="
python report.py --results_dir results --env cifar10
