#!/bin/bash
# Run missing baseline seeds (42, 100) for MC Dropout, SWAG, Subspace,
# Laplace on Ant/HC/Hopper with the neurips_minari preset. Both raw and
# VCal variants. Runs sequentially (one GPU job at a time) so CUDA stays
# happy.
#
# Usage: bash experiments/scripts/run_missing_baseline_seeds.sh
#
# Output appended to experiments/logs/baseline_seeds.log

set -u
LOG=/home/elean/pnc/experiments/logs/baseline_seeds.log
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

        # MC Dropout (raw + VCal)
        run "mc_$ENV\_$SEED" GymMCDropout $COMMON --n-perturbations 100
        run "mc_vcal_$ENV\_$SEED" GymMCDropout $COMMON --n-perturbations 100 --posthoc-calibrate

        # SWAG
        run "swag_$ENV\_$SEED" GymSWAG $COMMON --n-perturbations 100
        run "swag_vcal_$ENV\_$SEED" GymSWAG $COMMON --n-perturbations 100 --posthoc-calibrate

        # Subspace (T=0.0 default)
        run "sub_$ENV\_$SEED" GymSubspaceInference $COMMON --n-perturbations 100
        run "sub_vcal_$ENV\_$SEED" GymSubspaceInference $COMMON --n-perturbations 100 --posthoc-calibrate

        # Laplace (sweep priors)
        PRIORS='[1.0,10.0,100.0,1000.0,10000.0,100000.0]'
        run "lap_$ENV\_$SEED" GymLaplace $COMMON --n-perturbations 100 --laplace-priors "$PRIORS"
        run "lap_vcal_$ENV\_$SEED" GymLaplace $COMMON --n-perturbations 100 --laplace-priors "$PRIORS" --posthoc-calibrate
    done
done

echo "=== END $(date) ===" >> "$LOG"
