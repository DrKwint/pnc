#!/usr/bin/env bash
# Complete the missing CIFAR-100 OOD evaluations from the NeurIPS strengthening plan.
# Based on audit 2026-04-13: 19 files missing (excluding seed 0 SWAG which is run separately).
# Resumes the pipeline that crashed due to merge conflict in ensembles.py.
#
# Phases (runs only what is missing):
#   Seed 0: LLLA, Multi-block PnC
#   Seed 1: SWAG, LLLA, Multi-block PnC, Standard Ensemble
#   Seed 2: ALL 12 methods
#
# Frozen CIFAR-10 hyperparameters (transferability test, no CIFAR-100 retuning).
set -uo pipefail   # NOTE: no -e so a single failure doesn't abort the rest.

PYTHON=".venv/bin/python"
LUIGI="$PYTHON -m luigi --module cifar_tasks --local-scheduler"
EPOCHS=300
DATASET=cifar100

LOG_DIR="experiments/logs"
mkdir -p "$LOG_DIR"
TS=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/cifar100_ood_resume_${TS}.log"

log() {
  echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG_FILE"
}

run_task() {
  local label="$1"
  shift
  log "=> START $label"
  "$@" 2>&1 | tee -a "$LOG_FILE"
  local rc=${PIPESTATUS[0]}
  if [ "$rc" -eq 0 ]; then
    log "<= OK    $label"
  else
    log "<= FAIL  $label (rc=$rc) -- continuing"
  fi
}

log "===== CIFAR-100 OOD resume start ====="

# ---------------------------------------------------------------------------
# Seed 0 — missing: LLLA, Multi-block PnC (SWAG seed 0 is run separately first)
# ---------------------------------------------------------------------------
SEED=0
log "--- Seed $SEED ---"

run_task "LLLA seed0" \
  $LUIGI CIFAROpenOODLLLA --dataset $DATASET --posthoc-calibrate --epochs $EPOCHS \
    --n-perturbations 50 --prior-precision 10.0 --seed $SEED

run_task "MultiBlockPnC seed0" \
  $LUIGI CIFAROpenOODMultiBlockPnC --dataset $DATASET --posthoc-calibrate --epochs $EPOCHS \
    --perturbation-sizes '[7.0]' --n-directions 20 --random-directions --chunk-size 64 --seed $SEED

# ---------------------------------------------------------------------------
# Seed 1 — missing: SWAG, LLLA, Multi-block PnC, Standard Ensemble
# ---------------------------------------------------------------------------
SEED=1
log "--- Seed $SEED ---"

run_task "SWAG seed1" \
  $LUIGI CIFAROpenOODSWAG --dataset $DATASET --posthoc-calibrate --epochs $EPOCHS \
    --n-perturbations 50 --swag-start-epoch 240 --seed $SEED

run_task "LLLA seed1" \
  $LUIGI CIFAROpenOODLLLA --dataset $DATASET --posthoc-calibrate --epochs $EPOCHS \
    --n-perturbations 50 --prior-precision 10.0 --seed $SEED

run_task "MultiBlockPnC seed1" \
  $LUIGI CIFAROpenOODMultiBlockPnC --dataset $DATASET --posthoc-calibrate --epochs $EPOCHS \
    --perturbation-sizes '[7.0]' --n-directions 20 --random-directions --chunk-size 64 --seed $SEED

run_task "StandardEnsemble seed1" \
  $LUIGI CIFAROpenOODStandardEnsemble --dataset $DATASET --posthoc-calibrate --epochs $EPOCHS \
    --n-models 5 --seed $SEED

# ---------------------------------------------------------------------------
# Seed 2 — all 12 methods
# ---------------------------------------------------------------------------
SEED=2
log "--- Seed $SEED ---"

# Single-model baselines
run_task "PreActResNet18 seed2" \
  $LUIGI CIFAROpenOODPreActResNet18 --dataset $DATASET --posthoc-calibrate --epochs $EPOCHS --seed $SEED
run_task "MSP seed2" \
  $LUIGI CIFAROpenOODMSP --dataset $DATASET --posthoc-calibrate --epochs $EPOCHS --seed $SEED
run_task "Energy seed2" \
  $LUIGI CIFAROpenOODEnergy --dataset $DATASET --posthoc-calibrate --epochs $EPOCHS --seed $SEED
run_task "Mahalanobis seed2" \
  $LUIGI CIFAROpenOODMahalanobis --dataset $DATASET --posthoc-calibrate --epochs $EPOCHS --seed $SEED
run_task "ReActEnergy seed2" \
  $LUIGI CIFAROpenOODReActEnergy --dataset $DATASET --posthoc-calibrate --epochs $EPOCHS --seed $SEED

# Stochastic / ensemble UQ methods
run_task "MCDropout seed2" \
  $LUIGI CIFAROpenOODMCDropout --dataset $DATASET --posthoc-calibrate --epochs $EPOCHS \
    --n-perturbations 32 --dropout-rate 0.1 --seed $SEED
run_task "LLLA seed2" \
  $LUIGI CIFAROpenOODLLLA --dataset $DATASET --posthoc-calibrate --epochs $EPOCHS \
    --n-perturbations 50 --prior-precision 10.0 --seed $SEED
run_task "Epinet seed2" \
  $LUIGI CIFAROpenOODEpinet --dataset $DATASET --posthoc-calibrate --epochs $EPOCHS \
    --n-perturbations 50 --prior-scale 3.0 --seed $SEED
run_task "SWAG seed2" \
  $LUIGI CIFAROpenOODSWAG --dataset $DATASET --posthoc-calibrate --epochs $EPOCHS \
    --n-perturbations 50 --swag-start-epoch 240 --seed $SEED

# PnC variants
run_task "PnC seed2" \
  $LUIGI CIFAROpenOODPnC --dataset $DATASET --posthoc-calibrate --epochs $EPOCHS \
    --perturbation-sizes '[50.0]' --n-directions 20 --random-directions --seed $SEED
run_task "MultiBlockPnC seed2" \
  $LUIGI CIFAROpenOODMultiBlockPnC --dataset $DATASET --posthoc-calibrate --epochs $EPOCHS \
    --perturbation-sizes '[7.0]' --n-directions 20 --random-directions --chunk-size 64 --seed $SEED

# Deep Ensemble
run_task "StandardEnsemble seed2" \
  $LUIGI CIFAROpenOODStandardEnsemble --dataset $DATASET --posthoc-calibrate --epochs $EPOCHS \
    --n-models 5 --seed $SEED

log "===== CIFAR-100 OOD resume complete ====="
log "Full log: $LOG_FILE"
