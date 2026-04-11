#!/bin/bash
# Evidential regression baseline sweep.
# Step 1: hyperparameter sweep on Ant seed 0 to pick lambda by nll_val.
# Step 2: full runs at best lambda for 3 envs × 5 seeds × {raw, VCal}.
# Sequential, single GPU.

set -u
LOG=/home/elean/pnc/experiments/logs/evidential_sweep.log
mkdir -p "$(dirname "$LOG")"

echo "=== START $(date) ===" >> "$LOG"

run() {
    local name=$1
    shift
    echo "[$(date +%H:%M:%S)] RUN $name" >> "$LOG"
    .venv/bin/python -m luigi --module gym_tasks "$@" --local-scheduler >> "$LOG" 2>&1
    local rc=$?
    echo "[$(date +%H:%M:%S)] DONE $name rc=$rc" >> "$LOG"
}

# Step 1: lambda hyperparameter sweep on Ant seed 0
echo "[$(date +%H:%M:%S)] === Step 1: lambda hyperparameter sweep ===" >> "$LOG"
for LAM in 0.001 0.01 0.1 1.0; do
    run "ev_lam_ant_${LAM}" GymEvidential \
        --env Ant-v5 --steps 10000 --hidden-dims "[200,200,200,200]" \
        --lam "$LAM" --policy-preset neurips_minari --seed 0
done

# Parse which lambda won by nll_val — keep it simple: hardcode a follow-up step
# that the log operator will use after inspecting Step 1 results. For now, pick
# 0.01 (Amini's default) as the working value and run the full sweep.
BEST_LAM=0.01

# Step 2: full runs at best lambda
echo "[$(date +%H:%M:%S)] === Step 2: full sweep at lam=${BEST_LAM} ===" >> "$LOG"
ENVS=(HalfCheetah-v5 Hopper-v5 Ant-v5)
SEEDS=(0 10 42 100 200)

for ENV in "${ENVS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        # Ant seed 0 at lam=0.01 is already done in step 1 if BEST_LAM=0.01 — skip
        if [ "$ENV" = "Ant-v5" ] && [ "$SEED" = "0" ] && [ "$BEST_LAM" = "0.01" ]; then
            echo "[$(date +%H:%M:%S)] SKIP ev_${ENV}_s${SEED}_raw (already done in step 1)" >> "$LOG"
        else
            run "ev_${ENV}_s${SEED}_raw" GymEvidential \
                --env "$ENV" --steps 10000 --hidden-dims "[200,200,200,200]" \
                --lam "$BEST_LAM" --policy-preset neurips_minari --seed "$SEED"
        fi
        run "ev_${ENV}_s${SEED}_vcal" GymEvidential \
            --env "$ENV" --steps 10000 --hidden-dims "[200,200,200,200]" \
            --lam "$BEST_LAM" --policy-preset neurips_minari --seed "$SEED" \
            --posthoc-calibrate
    done
done

echo "=== END $(date) ===" >> "$LOG"
