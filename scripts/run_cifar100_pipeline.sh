#!/usr/bin/env bash
# Full CIFAR-100 pipeline: train all required models, then run multi-seed OOD eval.
# Sequential, single GPU lane. Total estimated wall time: ~25-30 hours.
#
# Phases:
#   1. Train SWAG seeds 0, 1, 2 (~9h - each does its own training from scratch)
#   2. Train Epinet seeds 0, 1, 2 (~1h - epinet only, base is shared from base training)
#   3. Train MCDropout seeds 0, 1, 2 (~9h)
#   4. Run all 12 OOD methods × 3 seeds (~7h)
#   5. Aggregate
#
# Prerequisite: CIFAR-100 base PreActResNet18 must already be trained for seeds 0, 1, 2.
#
# Frozen ID hyperparameters: same as CIFAR-10 (we explicitly do NOT retune for CIFAR-100;
# this is a transferability test).
set -euo pipefail

PYTHON=".venv/bin/python"
LUIGI="$PYTHON -m luigi --module cifar_tasks --local-scheduler"
EPOCHS=300
DATASET=cifar100

LOG_DIR="experiments/logs"
mkdir -p "$LOG_DIR"
TS=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/cifar100_pipeline_${TS}.log"

log() {
  echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG_FILE"
}

log "===== CIFAR-100 pipeline start ====="

# --- Phase 1: SWAG training ---
for SEED in 0 1 2; do
  log "Phase 1/5: Train CIFAR-100 SWAG seed $SEED (~3h)"
  $LUIGI CIFARTrainSWAGPreActResNet18 \
      --dataset $DATASET --epochs $EPOCHS \
      --swag-start-epoch 240 --swag-collect-freq 1 --swag-max-rank 20 \
      --seed $SEED 2>&1 | tee -a "$LOG_FILE"
done

# --- Phase 2: Epinet training (cheap; only the small MLP) ---
for SEED in 0 1 2; do
  log "Phase 2/5: Train CIFAR-100 Epinet ps=3.0 seed $SEED (~20min)"
  $LUIGI CIFARTrainEpinet \
      --dataset $DATASET --epochs $EPOCHS \
      --epinet-epochs 100 --epinet-lr 0.001 --epinet-wd 0.0001 \
      --index-dim 8 --epinet-hiddens '50,50' \
      --prior-scale 3.0 --seed $SEED 2>&1 | tee -a "$LOG_FILE"
done

# --- Phase 3: MCDropout training ---
for SEED in 0 1 2; do
  log "Phase 3/5: Train CIFAR-100 MCDropout dr=0.1 seed $SEED (~3h)"
  $LUIGI CIFARTrainMCDropoutPreActResNet18 \
      --dataset $DATASET --epochs $EPOCHS \
      --dropout-rate 0.1 --seed $SEED 2>&1 | tee -a "$LOG_FILE"
done

# --- Phase 4: OOD evaluation per seed ---
for SEED in 0 1 2; do
  log "Phase 4/5: OOD evaluation CIFAR-100 seed $SEED"

  # Single-model baselines
  $LUIGI CIFAROpenOODPreActResNet18 --dataset $DATASET --posthoc-calibrate --epochs $EPOCHS --seed $SEED 2>&1 | tee -a "$LOG_FILE"
  $LUIGI CIFAROpenOODMSP            --dataset $DATASET --posthoc-calibrate --epochs $EPOCHS --seed $SEED 2>&1 | tee -a "$LOG_FILE"
  $LUIGI CIFAROpenOODEnergy         --dataset $DATASET --posthoc-calibrate --epochs $EPOCHS --seed $SEED 2>&1 | tee -a "$LOG_FILE"
  $LUIGI CIFAROpenOODMahalanobis    --dataset $DATASET --posthoc-calibrate --epochs $EPOCHS --seed $SEED 2>&1 | tee -a "$LOG_FILE"
  $LUIGI CIFAROpenOODReActEnergy    --dataset $DATASET --posthoc-calibrate --epochs $EPOCHS --seed $SEED 2>&1 | tee -a "$LOG_FILE"

  # Stochastic / ensemble UQ methods
  $LUIGI CIFAROpenOODMCDropout     --dataset $DATASET --posthoc-calibrate --epochs $EPOCHS --n-perturbations 32 --dropout-rate 0.1 --seed $SEED 2>&1 | tee -a "$LOG_FILE"
  $LUIGI CIFAROpenOODLLLA          --dataset $DATASET --posthoc-calibrate --epochs $EPOCHS --n-perturbations 50 --prior-precision 10.0 --seed $SEED 2>&1 | tee -a "$LOG_FILE"
  $LUIGI CIFAROpenOODEpinet        --dataset $DATASET --posthoc-calibrate --epochs $EPOCHS --n-perturbations 50 --prior-scale 3.0 --seed $SEED 2>&1 | tee -a "$LOG_FILE"
  $LUIGI CIFAROpenOODSWAG          --dataset $DATASET --posthoc-calibrate --epochs $EPOCHS --n-perturbations 50 --swag-start-epoch 240 --seed $SEED 2>&1 | tee -a "$LOG_FILE"

  # PnC variants — using the CIFAR-10-tuned scales (transferability test)
  $LUIGI CIFAROpenOODPnC \
      --dataset $DATASET --posthoc-calibrate --epochs $EPOCHS \
      --perturbation-sizes '[50.0]' --n-directions 20 --random-directions --seed $SEED 2>&1 | tee -a "$LOG_FILE"

  $LUIGI CIFAROpenOODMultiBlockPnC \
      --dataset $DATASET --posthoc-calibrate --epochs $EPOCHS \
      --perturbation-sizes '[7.0]' --n-directions 20 --random-directions --chunk-size 64 --seed $SEED 2>&1 | tee -a "$LOG_FILE"

  # Deep Ensemble
  $LUIGI CIFAROpenOODStandardEnsemble --dataset $DATASET --posthoc-calibrate --epochs $EPOCHS --n-models 5 --seed $SEED 2>&1 | tee -a "$LOG_FILE"

  log "Seed $SEED done."
done

# --- Phase 5: Aggregate ---
log "Phase 5/5: Aggregate"
$PYTHON scripts/aggregate_ood_results.py results/cifar100 2>&1 | tee -a "$LOG_FILE"
$PYTHON scripts/aggregate_ood_results.py --markdown results/cifar100 > experiments/cifar_neurips_strengthening_aggregate_cifar100.md 2>&1 || true
log "Aggregator output saved to experiments/cifar_neurips_strengthening_aggregate_cifar100.md"

# Paper-ready table (grouped, dataset-aware)
$PYTHON scripts/make_ood_paper_table.py results/cifar100 > experiments/cifar100_ood_paper_table.md 2>&1 || true
log "Paper table saved to experiments/cifar100_ood_paper_table.md"

log "===== CIFAR-100 pipeline complete ====="
log "Full log: $LOG_FILE"
