#!/bin/bash
# Hybrid PnC + DE sweep at matched total budget M*K = 50.
# Configs: M=2/K=25, M=5/K=10, M=10/K=5.
# 3 envs × 5 seeds × 2 VCal variants = 30 runs per config = 90 runs total.
# Runs sequentially (one GPU job at a time).

set -u
LOG=/home/elean/pnc/experiments/logs/hybrid_sweep.log
mkdir -p "$(dirname "$LOG")"

echo "=== START $(date) ===" >> "$LOG"

ENVS=(HalfCheetah-v5 Hopper-v5 Ant-v5)
SEEDS=(0 10 42 100 200)

run() {
    local name=$1
    shift
    echo "[$(date +%H:%M:%S)] RUN $name" >> "$LOG"
    .venv/bin/python -m luigi --module gym_tasks "$@" --local-scheduler >> "$LOG" 2>&1
    local rc=$?
    echo "[$(date +%H:%M:%S)] DONE $name rc=$rc" >> "$LOG"
}

for CFG in "2 25" "5 10" "10 5"; do
    read M K <<< "$CFG"
    for ENV in "${ENVS[@]}"; do
        for SEED in "${SEEDS[@]}"; do
            BASE="hyb_M${M}_K${K}_${ENV}_s${SEED}"
            CMD=(GymHybridPnCDE \
                --env "$ENV" --steps 10000 --hidden-dims "[200,200,200,200]" \
                --n-de "$M" --n-pjsvd-per-de "$K" --n-directions 20 \
                --perturbation-sizes '[5.0, 10.0, 20.0, 50.0]' \
                --pjsvd-family random --safe-subspace-backend projected_residual \
                --policy-preset neurips_minari --seed "$SEED")
            run "${BASE}_raw" "${CMD[@]}"
            run "${BASE}_vcal" "${CMD[@]}" --posthoc-calibrate
        done
    done
done

echo "=== END $(date) ===" >> "$LOG"
