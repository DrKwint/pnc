#!/bin/bash
# Re-run SWAG, Subspace, and Laplace experiments so that the newly added
# predict_intermediate methods populate Unc-L2-h / Unc-L2-z metrics.
#
# For each method+env+seed combination the existing JSON is backed up to
# .pre_l2.json and Luigi re-runs the task from scratch.
#
# Runs sequentially (one GPU job at a time).
#
# Usage: bash experiments/scripts/rerun_baselines_with_l2.sh
#
# Output appended to experiments/logs/rerun_baselines_l2.log

set -u
LOG=/home/elean/pnc/experiments/logs/rerun_baselines_l2.log
mkdir -p "$(dirname "$LOG")"

echo "=== START $(date) ===" >> "$LOG"

ENVS=(Ant-v5 HalfCheetah-v5 Hopper-v5)
SEEDS=(0 10 42 100 200)

backup() {
    local f=$1
    if [ -f "$f" ]; then
        mv "$f" "${f%.json}.pre_l2.json"
        echo "  backed up $f" >> "$LOG"
    fi
}

run() {
    local name=$1
    shift
    echo "[$(date +%H:%M:%S)] RUN $name" >> "$LOG"
    .venv/bin/python -m luigi --module gym_tasks "$@" --local-scheduler >> "$LOG" 2>&1
    local rc=$?
    echo "[$(date +%H:%M:%S)] DONE $name rc=$rc" >> "$LOG"
}

for SEED in "${SEEDS[@]}"; do
    for ENV in "${ENVS[@]}"; do
        COMMON="--env $ENV --steps 10000 --hidden-dims [200,200,200,200] --seed $SEED --policy-preset neurips_minari"

        # --- SWAG ---
        backup "results/$ENV/swag_n100_h200-200-200-200_act-relu_seed${SEED}.json"
        run "swag_${ENV}_${SEED}" GymSWAG $COMMON --n-perturbations 100

        backup "results/$ENV/swag_vcal_n100_h200-200-200-200_act-relu_seed${SEED}.json"
        run "swag_vcal_${ENV}_${SEED}" GymSWAG $COMMON --n-perturbations 100 --posthoc-calibrate

        # --- Subspace ---
        backup "results/$ENV/subspace_inference_n100_h200-200-200-200_act-relu_seed${SEED}_T0.0.json"
        run "sub_${ENV}_${SEED}" GymSubspaceInference $COMMON --n-perturbations 100

        backup "results/$ENV/subspace_inference_vcal_n100_h200-200-200-200_act-relu_seed${SEED}_T0.0.json"
        run "sub_vcal_${ENV}_${SEED}" GymSubspaceInference $COMMON --n-perturbations 100 --posthoc-calibrate

        # --- Laplace ---
        PRIORS='[1.0,10.0,100.0,1000.0,10000.0,100000.0]'
        backup "results/$ENV/laplace_priors1.0-10.0-100.0-1000.0-10000.0-100000.0_n100_h200-200-200-200_act-relu_seed${SEED}.json"
        run "lap_${ENV}_${SEED}" GymLaplace $COMMON --n-perturbations 100 --laplace-priors "$PRIORS"

        backup "results/$ENV/laplace_vcal_priors1.0-10.0-100.0-1000.0-10000.0-100000.0_n100_h200-200-200-200_act-relu_seed${SEED}.json"
        run "lap_vcal_${ENV}_${SEED}" GymLaplace $COMMON --n-perturbations 100 --laplace-priors "$PRIORS" --posthoc-calibrate
    done
done

echo "=== END $(date) ===" >> "$LOG"
