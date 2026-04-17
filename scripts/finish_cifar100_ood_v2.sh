#!/usr/bin/env bash
# Finalize CIFAR-100 OOD eval after the earlier resume run. Runs only what's still missing.
#
# Still to do (11 tasks):
#   Seed 1: StandardEnsemble
#   Seed 2: PreActResNet18, MSP, Energy, Mahalanobis, ReActEnergy,
#           MCDropout, Epinet, SWAG, PnC (single-block), StandardEnsemble
#
# Intentionally SKIPPED on CIFAR-100 (documented out-of-scope):
#   LLLA        -> full GGN (513*100)^2 tensor exceeds 8GB VRAM (no K-FAC impl)
#   MultiBlockPnC -> process hang during OOD score phase w/ 100-class ensemble
set -uo pipefail

PYTHON=".venv/bin/python"
LUIGI="$PYTHON -m luigi --module cifar_tasks --local-scheduler"
EPOCHS=300
DATASET=cifar100

LOG_DIR="experiments/logs"
mkdir -p "$LOG_DIR"
TS=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/cifar100_ood_finalize_${TS}.log"

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG_FILE"; }

run_task() {
  local label="$1"; shift
  log "=> START $label"
  "$@" 2>&1 | tee -a "$LOG_FILE"
  local rc=${PIPESTATUS[0]}
  if [ "$rc" -eq 0 ]; then
    log "<= OK    $label"
  else
    log "<= FAIL  $label (rc=$rc) -- continuing"
  fi
}

log "===== CIFAR-100 OOD finalize start ====="

# Seed 1 -------------------------------------------------------------------
SEED=1
log "--- Seed $SEED ---"
run_task "StandardEnsemble seed1" \
  $LUIGI CIFAROpenOODStandardEnsemble --dataset $DATASET --posthoc-calibrate --epochs $EPOCHS \
    --n-models 5 --seed $SEED

# Seed 2 -------------------------------------------------------------------
SEED=2
log "--- Seed $SEED ---"

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
run_task "MCDropout seed2" \
  $LUIGI CIFAROpenOODMCDropout --dataset $DATASET --posthoc-calibrate --epochs $EPOCHS \
    --n-perturbations 32 --dropout-rate 0.1 --seed $SEED
run_task "Epinet seed2" \
  $LUIGI CIFAROpenOODEpinet --dataset $DATASET --posthoc-calibrate --epochs $EPOCHS \
    --n-perturbations 50 --prior-scale 3.0 --seed $SEED
run_task "SWAG seed2" \
  $LUIGI CIFAROpenOODSWAG --dataset $DATASET --posthoc-calibrate --epochs $EPOCHS \
    --n-perturbations 50 --swag-start-epoch 240 --seed $SEED
run_task "PnC seed2" \
  $LUIGI CIFAROpenOODPnC --dataset $DATASET --posthoc-calibrate --epochs $EPOCHS \
    --perturbation-sizes '[50.0]' --n-directions 20 --random-directions --seed $SEED
run_task "StandardEnsemble seed2" \
  $LUIGI CIFAROpenOODStandardEnsemble --dataset $DATASET --posthoc-calibrate --epochs $EPOCHS \
    --n-models 5 --seed $SEED

log "===== CIFAR-100 OOD finalize complete ====="
log "Full log: $LOG_FILE"
