#!/bin/bash
# Sweep ridge regularization (lambda_reg) for PJSVD-Multi-LS across Low-Proj
# and Random-Proj, seed 0 only. Diagnoses whether Low-Proj's poor ID RMSE is
# driven by ill-conditioned pseudoinverse solves that ridge damps out.
#
# Grid: lambda_reg in {1e-4, 1e-2, 1, 100}
# Families: low, random
# Envs: Ant-v5, HalfCheetah-v5, Hopper-v5
# VCal: both on and off
# Total: 4 * 2 * 3 * 2 = 48 runs, sequential.
#
# Outputs append to experiments/logs/pjsvd_lreg_sweep.log.

set -u
LOG=/home/elean/pnc/experiments/logs/pjsvd_lreg_sweep.log
mkdir -p "$(dirname "$LOG")"

echo "=== START $(date) ===" >> "$LOG"

ENVS=(Ant-v5 HalfCheetah-v5 Hopper-v5)
FAMILIES=(low random)
LAMBDAS=(0.0001 0.01 1 100)
SEED=0

run() {
    local name=$1
    shift
    echo "[$(date +%H:%M:%S)] RUN $name" >> "$LOG"
    .venv/bin/python -m luigi --module gym_tasks "$@" --local-scheduler >> "$LOG" 2>&1
    local rc=$?
    echo "[$(date +%H:%M:%S)] DONE $name rc=$rc" >> "$LOG"
}

for LREG in "${LAMBDAS[@]}"; do
    for FAM in "${FAMILIES[@]}"; do
        for ENV in "${ENVS[@]}"; do
            COMMON="--env $ENV --steps 10000 --hidden-dims [200,200,200,200] --seed $SEED --policy-preset neurips_minari"
            PJSVD_COMMON="--n-directions 20 --n-perturbations 50 \
                --perturbation-sizes [5.0,10.0,20.0,50.0] \
                --layer-scope multi --correction-mode least_squares \
                --pjsvd-family $FAM --safe-subspace-backend projected_residual \
                --subset-size 4096 --probabilistic-base-model --lambda-reg $LREG"

            run "pjsvd_${FAM}_lreg${LREG}_raw_${ENV}"  GymPJSVD $COMMON $PJSVD_COMMON
            run "pjsvd_${FAM}_lreg${LREG}_vcal_${ENV}" GymPJSVD $COMMON $PJSVD_COMMON --posthoc-calibrate
        done
    done
done

echo "=== END $(date) ===" >> "$LOG"
