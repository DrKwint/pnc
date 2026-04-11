#!/bin/bash
# Run missing VCal seeds (42, 100) for DE and PJSVD-Multi on Ant/HC/Hopper.
# PJSVD-Multi-VCal and standard_ensemble_vcal_n5 currently only have seeds
# {0, 10, 200}, while raw versions have {0, 10, 42, 100, 200}. This script
# closes the gap so the seed filter can produce a clean n=5 table.
#
# Uses neurips_minari preset. Sequential. Outputs append to
# experiments/logs/vcal_seeds.log.

set -u
LOG=/home/elean/pnc/experiments/logs/vcal_seeds.log
mkdir -p "$(dirname "$LOG")"

echo "=== START $(date) ===" >> "$LOG"

ENVS=(HalfCheetah-v5 Hopper-v5 Ant-v5)
SEEDS=(42 100)

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

        # Deep Ensemble + VCal at n=5 (matches current gym_tables row)
        run "de_vcal_n5_$ENV\_$SEED" GymStandardEnsemble $COMMON --n-baseline 5 --posthoc-calibrate

        # PJSVD-Multi + VCal at n=50, scale sweep [5, 10, 20, 50]
        run "pjsvd_vcal_$ENV\_$SEED" GymPJSVD $COMMON \
            --n-directions 20 --n-perturbations 50 \
            --perturbation-sizes '[5.0, 10.0, 20.0, 50.0]' \
            --layer-scope multi --correction-mode least_squares \
            --pjsvd-family random --safe-subspace-backend projected_residual \
            --subset-size 4096 --probabilistic-base-model --posthoc-calibrate
    done
done

echo "=== END $(date) ===" >> "$LOG"
