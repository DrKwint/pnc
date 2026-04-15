#!/bin/bash
# Evidential lever-1 (val_on_predictive_nll) sweep: replaces NIG-loss early
# stopping with predictive Gaussian-NLL early stopping. λ ∈ {0.01, 0.1, 1}
# across Ant/HC/Hopper, with and without VCal, seed 0 only.
# Total: 3 λ × 3 envs × 2 vcal = 18 runs, sequential.

set -u
LOG=/home/elean/pnc/experiments/logs/evidential_esn_sweep.log
mkdir -p "$(dirname "$LOG")"

echo "=== START $(date) ===" >> "$LOG"

ENVS=(Ant-v5 HalfCheetah-v5 Hopper-v5)
LAMBDAS=(0.01 0.1 1)
SEED=0

run() {
    local name=$1
    shift
    echo "[$(date +%H:%M:%S)] RUN $name" >> "$LOG"
    .venv/bin/python -m luigi --module gym_tasks "$@" --local-scheduler >> "$LOG" 2>&1
    local rc=$?
    echo "[$(date +%H:%M:%S)] DONE $name rc=$rc" >> "$LOG"
}

for LAM in "${LAMBDAS[@]}"; do
    for ENV in "${ENVS[@]}"; do
        COMMON="--env $ENV --steps 10000 --hidden-dims [200,200,200,200] --seed $SEED --policy-preset neurips_minari"
        run "evidential_esn_raw_lam${LAM}_${ENV}"  GymEvidential $COMMON --lam $LAM --val-on-predictive-nll
        run "evidential_esn_vcal_lam${LAM}_${ENV}" GymEvidential $COMMON --lam $LAM --val-on-predictive-nll --posthoc-calibrate
    done
done

echo "=== END $(date) ===" >> "$LOG"
