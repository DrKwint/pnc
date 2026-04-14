#!/bin/bash
# Run uncorrected PnC (correction_mode=none) for 5 seeds across all envs.
#
# Current state:
#   Ant-v5:         seed0 exists  -> run 10, 42, 100, 200
#   HalfCheetah-v5: seed0 exists  -> run 10, 42, 100, 200
#   Hopper-v5:      seed0 exists  -> run 10, 42, 100, 200
#   Humanoid-v5:    none exist    -> run 0, 10, 42, 100, 200
#
# Parameters match the existing seed0 files:
#   pjsvd_multi_none_random_projected_residual_prob_k20_n50_ps5.0-10.0-20.0-50.0_h200-200-200-200_act-relu
#
# Runs sequentially (one GPU job at a time).
#
# Usage: bash experiments/scripts/run_uncorrected_pnc_seeds.sh
#
# Output appended to experiments/logs/uncorrected_pnc_seeds.log

set -u
LOG=/home/elean/pnc/experiments/logs/uncorrected_pnc_seeds.log
mkdir -p "$(dirname "$LOG")"

echo "=== START $(date) ===" >> "$LOG"

PNC_ARGS="--layer-scope multi --correction-mode none --pjsvd-family random \
--safe-subspace-backend projected_residual --probabilistic-base-model \
--n-directions 20 --n-perturbations 50 \
--perturbation-sizes [5.0,10.0,20.0,50.0]"

run() {
    local name=$1
    shift
    echo "[$(date +%H:%M:%S)] RUN $name" >> "$LOG"
    .venv/bin/python -m luigi --module gym_tasks GymPJSVD "$@" --local-scheduler >> "$LOG" 2>&1
    local rc=$?
    echo "[$(date +%H:%M:%S)] DONE $name rc=$rc" >> "$LOG"
}

# Ant, HalfCheetah, Hopper: seed0 already exists
for ENV in Ant-v5 HalfCheetah-v5 Hopper-v5; do
    for SEED in 10 42 100 200; do
        run "pnc_none_${ENV}_${SEED}" \
            --env $ENV --steps 10000 --hidden-dims '[200,200,200,200]' \
            --seed $SEED --policy-preset neurips_minari \
            $PNC_ARGS
    done
done

# Humanoid: no existing seeds
for SEED in 0 10 42 100 200; do
    run "pnc_none_Humanoid-v5_${SEED}" \
        --env Humanoid-v5 --steps 10000 --hidden-dims '[200,200,200,200]' \
        --seed $SEED --policy-preset neurips_minari \
        $PNC_ARGS
done

echo "=== END $(date) ===" >> "$LOG"
