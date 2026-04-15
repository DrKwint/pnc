#!/bin/bash
# Scouting sweep: Evidential lambda in {2, 5, 10} and MC Dropout dropout_prob in
# {0.05, 0.2, 0.3, 0.5}, across Ant/HC/Hopper, with and without VCal, seed 0 only.
# Sequential. Intended to be launched *after* the Low-Proj sweep finishes.
#
# Outputs append to experiments/logs/evidential_mcd_scouting.log.

set -u
LOG=/home/elean/pnc/experiments/logs/evidential_mcd_scouting.log
mkdir -p "$(dirname "$LOG")"

echo "=== START $(date) ===" >> "$LOG"

ENVS=(Ant-v5 HalfCheetah-v5 Hopper-v5)
SEED=0

run() {
    local name=$1
    shift
    echo "[$(date +%H:%M:%S)] RUN $name" >> "$LOG"
    .venv/bin/python -m luigi --module gym_tasks "$@" --local-scheduler >> "$LOG" 2>&1
    local rc=$?
    echo "[$(date +%H:%M:%S)] DONE $name rc=$rc" >> "$LOG"
}

# Evidential lambda sweep (3 new values, extending existing {0.001, 0.01, 0.1, 1})
for LAM in 2 5 10; do
    for ENV in "${ENVS[@]}"; do
        COMMON="--env $ENV --steps 10000 --hidden-dims [200,200,200,200] --seed $SEED --policy-preset neurips_minari"
        run "evidential_raw_lam${LAM}_${ENV}_${SEED}"  GymEvidential $COMMON --lam $LAM
        run "evidential_vcal_lam${LAM}_${ENV}_${SEED}" GymEvidential $COMMON --lam $LAM --posthoc-calibrate
    done
done

# MC Dropout dropout_prob sweep (4 new values; existing results use dr=0.1)
for DR in 0.05 0.2 0.3 0.5; do
    for ENV in "${ENVS[@]}"; do
        COMMON="--env $ENV --steps 10000 --hidden-dims [200,200,200,200] --seed $SEED --policy-preset neurips_minari"
        run "mcd_raw_dr${DR}_${ENV}_${SEED}"  GymMCDropout $COMMON --n-perturbations 100 --dropout-prob $DR
        run "mcd_vcal_dr${DR}_${ENV}_${SEED}" GymMCDropout $COMMON --n-perturbations 100 --dropout-prob $DR --posthoc-calibrate
    done
done

echo "=== END $(date) ===" >> "$LOG"
