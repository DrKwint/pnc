#!/bin/bash
# Revision closure, Part C: re-run the manuscript-facing MuJoCo P&C headline with
# ORIGINAL-CENTERED ridge.
#
# This is byte-for-byte the historical invocation in
# experiments/scripts/run_bootstrap_ls_lreg_paper_table.sh, with exactly one
# change: --ridge-toward-orig True. Everything else -- envs, seeds, base
# checkpoints (policy-preset neurips_minari), steps, subset size, K, M, the
# perturbation-size grid, layer scope, correction mode, backend, probabilistic
# base model, hidden dims, activation, lambda and bootstrap fraction -- is held
# fixed, so the paired comparison in Part 5 is apples to apples.
#
# The historical runs carried no _ridgeorig filename token (the old default was
# ridge_toward_orig=False, i.e. zero-centered). The new runs carry it, so the two
# sets coexist on disk and nothing historical is overwritten.
#
# Outputs e.g.
# results/{env}/pjsvd_multi_least_squares_random_projected_residual_prob_lreg0.0001_ridgeorig_bf0.1_k20_n50_ps5.0-10.0-20.0-50.0_h200-200-200-200_act-relu_seed{S}.json

set -u
LOG=/home/elean/pnc/experiments/logs/mujoco_original_centered_ridge.log
mkdir -p "$(dirname "$LOG")"

echo "=== START $(date) ===" >> "$LOG"

ENVS=(Ant-v5 HalfCheetah-v5 Hopper-v5 Humanoid-v5)
SEEDS=(0 10 42 100 200)
BFRAC=0.1
LREG=0.0001

run() {
    local name=$1
    shift
    echo "[$(date +%H:%M:%S)] RUN $name" >> "$LOG"
    .venv/bin/python -m luigi --module pnc_core.gym_tasks "$@" --local-scheduler >> "$LOG" 2>&1
    local rc=$?
    echo "[$(date +%H:%M:%S)] DONE $name rc=$rc" >> "$LOG"
}

for SEED in "${SEEDS[@]}"; do
    for ENV in "${ENVS[@]}"; do
        EXTRA_FLAGS=""
        # Humanoid's evaluation can OOM the uncorrected-l2 path; disable it.
        # (Identical to the historical script, so the pairing stays exact.)
        if [ "$ENV" = "Humanoid-v5" ]; then
            EXTRA_FLAGS="--compute-l2 False --compute-geometry False"
        fi
        run "pjsvd_ridgeorig_bf${BFRAC}_lreg${LREG}_${ENV}_s${SEED}" \
            GymPJSVD \
            --env "$ENV" \
            --steps 10000 \
            --subset-size 4096 \
            --n-directions 20 \
            --n-perturbations 50 \
            --perturbation-sizes '[5.0, 10.0, 20.0, 50.0]' \
            --layer-scope multi \
            --pjsvd-family random \
            --correction-mode least_squares \
            --safe-subspace-backend projected_residual \
            --probabilistic-base-model \
            --hidden-dims '[200,200,200,200]' \
            --activation relu \
            --lambda-reg "$LREG" \
            --bootstrap-frac "$BFRAC" \
            --ridge-toward-orig True \
            --policy-preset neurips_minari \
            --seed "$SEED" \
            $EXTRA_FLAGS
    done
done

echo "=== END $(date) ===" >> "$LOG"
