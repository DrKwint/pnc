#!/bin/bash
# Priority 3: risk-aware MPC downstream sweep.
# 3 methods × 2 gravity scales (amortized inside mpc_eval) × 1 seed × 3 episodes.
# Downsized from the plan's 200-episode target to fit in ~1 hour.
# Each gravity scale runs on the same trained model to avoid re-training.

set -u
LOG=/home/elean/pnc/experiments/logs/mpc_sweep.log
mkdir -p "$(dirname "$LOG")"
OUTDIR=/home/elean/pnc/experiments/mpc_results
mkdir -p "$OUTDIR"

echo "=== START $(date) ===" >> "$LOG"

run() {
    local name=$1
    shift
    echo "[$(date +%H:%M:%S)] RUN $name" >> "$LOG"
    .venv/bin/python experiments/scripts/mpc_eval.py "$@" >> "$LOG" 2>&1
    local rc=$?
    echo "[$(date +%H:%M:%S)] DONE $name rc=$rc" >> "$LOG"
}

SHARED="--env Ant-v5 --seed 0 --n-episodes 3 --gravity-scales 1.0,1.3 --horizon 4 --n-candidates 16 --n-iters 1 --max-steps 50"

# 1. Deterministic baseline (no variance) — control for the capability claim
run "det_beta0" --method deterministic --variance-penalty 0.0 --out "$OUTDIR/det_beta0.json" $SHARED

# 2. Deep Ensemble, no uncertainty penalty (β=0)
run "de_beta0" --method de --variance-penalty 0.0 --out "$OUTDIR/de_beta0.json" $SHARED

# 3. Deep Ensemble with uncertainty penalty (β=1)
run "de_beta1" --method de --variance-penalty 1.0 --out "$OUTDIR/de_beta1.json" $SHARED

# 4. Hybrid, β=0
run "hyb_beta0" --method hybrid --variance-penalty 0.0 --out "$OUTDIR/hyb_beta0.json" $SHARED

# 5. Hybrid, β=1
run "hyb_beta1" --method hybrid --variance-penalty 1.0 --out "$OUTDIR/hyb_beta1.json" $SHARED

echo "=== END $(date) ===" >> "$LOG"
