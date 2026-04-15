#!/bin/bash
# Run PJSVD-Multi-LS Low-Proj (with and without VCal) across Ant/HC/Hopper.
# Mirrors run_missing_vcal_seeds.sh conventions; sequential (one GPU job at a time).
#
# Outputs append to experiments/logs/low_proj_seeds.log.

set -u
LOG=/home/elean/pnc/experiments/logs/low_proj_seeds.log
mkdir -p "$(dirname "$LOG")"

echo "=== START $(date) ===" >> "$LOG"

ENVS=(Ant-v5 HalfCheetah-v5 Hopper-v5)
SEEDS=(0 10 42 100 200)

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

        # PJSVD-Multi-LS Low-Proj, no VCal
        run "pjsvd_low_raw_${ENV}_${SEED}" GymPJSVD $COMMON \
            --n-directions 20 --n-perturbations 50 \
            --perturbation-sizes '[5.0, 10.0, 20.0, 50.0]' \
            --layer-scope multi --correction-mode least_squares \
            --pjsvd-family low --safe-subspace-backend projected_residual \
            --subset-size 4096 --probabilistic-base-model

        # PJSVD-Multi-LS Low-Proj, with VCal
        run "pjsvd_low_vcal_${ENV}_${SEED}" GymPJSVD $COMMON \
            --n-directions 20 --n-perturbations 50 \
            --perturbation-sizes '[5.0, 10.0, 20.0, 50.0]' \
            --layer-scope multi --correction-mode least_squares \
            --pjsvd-family low --safe-subspace-backend projected_residual \
            --subset-size 4096 --probabilistic-base-model --posthoc-calibrate
    done
done

echo "=== END $(date) ===" >> "$LOG"
