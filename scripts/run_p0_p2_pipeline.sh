#!/usr/bin/env bash
# End-to-end P0 + P2 pipeline (sequential, single GPU lane).
# Total: ~7-9 hours.
#
# 1. Train Epinet ps=3.0 seeds 1, 2 (~1 hour)
# 2. Run 12 OOD methods at seed 1 (~3-4 hours)
# 3. Run 12 OOD methods at seed 2 (~3-4 hours)
# 4. Run inference cost benchmark (~10 min)
# 5. Aggregate multi-seed results into a markdown table
#
# Each step writes its output to results/cifar10/. Failures abort the script.
set -euo pipefail

PYTHON=".venv/bin/python"
LOG_DIR="experiments/logs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
PIPELINE_LOG="$LOG_DIR/p0_p2_${TIMESTAMP}.log"

log() {
  echo "[$(date +%H:%M:%S)] $*" | tee -a "$PIPELINE_LOG"
}

log "===== P0+P2 pipeline start ====="

log "Step 1/5: Train Epinet ps=3.0 seeds 1, 2"
bash scripts/train_epinet_ps3_seeds.sh 2>&1 | tee -a "$PIPELINE_LOG"

log "Step 2/5: OOD evaluation seed 1"
bash scripts/run_ood_eval_multi_seed.sh 1 2>&1 | tee -a "$PIPELINE_LOG"

log "Step 3/5: OOD evaluation seed 2"
bash scripts/run_ood_eval_multi_seed.sh 2 2>&1 | tee -a "$PIPELINE_LOG"

log "Step 4/5: Inference cost benchmark"
$PYTHON scripts/benchmark_inference_cost.py 2>&1 | tee -a "$PIPELINE_LOG"

log "Step 5/5: Aggregate multi-seed results"
$PYTHON scripts/aggregate_ood_results.py --markdown 2>&1 | tee -a "$PIPELINE_LOG"

log "===== P0+P2 pipeline complete ====="
log "Full log: $PIPELINE_LOG"
