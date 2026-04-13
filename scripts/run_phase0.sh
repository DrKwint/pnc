#!/bin/bash
# Phase 0: Corrected Baselines
# Run from pnc/ directory: bash scripts/run_phase0.sh
set -e

cd "$(dirname "$0")/.."

echo "=========================================="
echo "Phase 0a: Single Model Eval (seeds 1,2)"
echo "=========================================="
for SEED in 1 2; do
  python -m luigi --module cifar_tasks CIFAREvalPreActResNet18 \
    --dataset cifar10 --epochs 300 --seed $SEED \
    --posthoc-calibrate --local-scheduler
done

echo "=========================================="
echo "Phase 0b: LLLA Prior Sweep (seeds 0,1,2)"
echo "=========================================="
for SEED in 0 1 2; do
  for PRIOR in 0.01 0.1 1.0 10.0 100.0 1000.0; do
    python -m luigi --module cifar_tasks CIFARLLLA \
      --dataset cifar10 --epochs 300 --n-perturbations 50 \
      --prior-precision $PRIOR \
      --seed $SEED --posthoc-calibrate --local-scheduler
  done
done

echo "=========================================="
echo "Phase 0c: MC Dropout (seeds 0,1,2)"
echo "=========================================="
for SEED in 0 1 2; do
  python -m luigi --module cifar_tasks CIFARPreActMCDropout \
    --dataset cifar10 --epochs 300 --n-perturbations 50 --dropout-rate 0.1 \
    --seed $SEED --posthoc-calibrate --local-scheduler
done

echo "=========================================="
echo "Phase 0d: Deep Ensemble (seeds 0,1,2)"
echo "=========================================="
# seed 0: uses base models 0-4 (all exist)
# seed 1: uses base models 1-5 (need to train seed 5)
# seed 2: uses base models 2-6 (need to train seeds 5,6)
for SEED in 5 6; do
  python -m luigi --module cifar_tasks CIFARTrainPreActResNet18 \
    --dataset cifar10 --epochs 300 --seed $SEED --local-scheduler
done
for SEED in 0 1 2; do
  python -m luigi --module cifar_tasks CIFARStandardEnsemble \
    --dataset cifar10 --epochs 300 --n-models 5 --seed $SEED \
    --posthoc-calibrate --local-scheduler
done

echo "=========================================="
echo "Phase 0e: SWAG (seeds 0,1,2) - sws=240"
echo "=========================================="
for SEED in 0 1 2; do
  python -m luigi --module cifar_tasks CIFARPreActSWAG \
    --dataset cifar10 --epochs 300 --n-perturbations 50 \
    --swag-start-epoch 240 --swag-collect-freq 1 \
    --swag-use-bn-refresh --bn-refresh-subset-size 2048 \
    --seed $SEED --posthoc-calibrate --local-scheduler
done

echo "=========================================="
echo "Phase 0 Complete — Run report"
echo "=========================================="
python report.py --results_dir results --env cifar10
