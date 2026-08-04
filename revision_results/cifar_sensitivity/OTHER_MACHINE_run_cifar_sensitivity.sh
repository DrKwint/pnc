#!/bin/bash
# Priority 1 CIFAR-10 hyperparameter sensitivity (run on the CIFAR machine).
# Mirrors scripts/neurips_2026_rebuttal/priority1_sensitivity.py (MuJoCo).
# Anchor/default (marked *): CIFAROpenOODPnC single-block stage4/block1,
#   n_directions=10, n_perturbations=50, subset_size=1024, lambda_reg=1e-3,
#   perturbation_sizes=[10,50,100,200], random_directions=False (Lanczos).
# Prereq: CIFARTrainPreActResNet18 checkpoints (seeds in SEEDS) + openood_data/.
# One factor varied at a time; all else at the anchor. Screen SEEDS="0" first, then "0 1 2".
set -u
cd "$(dirname "$0")/../../.."   # repo root
SEEDS="${SEEDS:-0}"
DS=cifar10
EP=300
COMMON="--dataset $DS --epochs $EP --n-perturbations 50 --random-directions False --local-scheduler"
run() { echo "[$(date +%H:%M:%S)] $*"; python -m luigi --module pnc_core.cifar_tasks "$@"; }

for S in $SEEDS; do
  # 1A perturbation scale (relative grid; default val-selected among {10,50,100,200})
  for PS in 5.0 10.0 20.0 40.0 80.0; do
    run CIFAROpenOODPnC $COMMON --seed $S --n-directions 10 --subset-size 1024 \
        --lambda-reg 1e-3 --target-stage-idx 3 --target-block-idx 1 --perturbation-sizes "[$PS]"
  done
  # 1B rank
  for K in 1 2 5 10 20; do
    run CIFAROpenOODPnC $COMMON --seed $S --n-directions $K --subset-size 1024 \
        --lambda-reg 1e-3 --target-stage-idx 3 --target-block-idx 1 --perturbation-sizes '[10.0,50.0,100.0,200.0]'
  done
  # 1C calibration subset size (CIFAR "fraction" = # calibration examples; no bootstrap_frac)
  for SS in 256 512 1024 2048; do
    run CIFAROpenOODPnC $COMMON --seed $S --n-directions 10 --subset-size $SS \
        --lambda-reg 1e-3 --target-stage-idx 3 --target-block-idx 1 --perturbation-sizes '[10.0,50.0,100.0,200.0]'
  done
  # 1D target block (early / mid / late*) + single-block; multi-block via CIFARMultiBlockPnC
  for SB in "0 0" "1 1" "3 1"; do set -- $SB
    run CIFAROpenOODPnC $COMMON --seed $S --n-directions 10 --subset-size 1024 \
        --lambda-reg 1e-3 --target-stage-idx $1 --target-block-idx $2 --perturbation-sizes '[10.0,50.0,100.0,200.0]'
  done
  # (multi-block: use CIFAROpenOODMultiBlockPnC with the same knobs)
  # 1E ridge strength
  for LAM in 0.0 1e-4 1e-3 1e-2 1e-1 1.0; do
    run CIFAROpenOODPnC $COMMON --seed $S --n-directions 10 --subset-size 1024 \
        --lambda-reg $LAM --target-stage-idx 3 --target-block-idx 1 --perturbation-sizes '[10.0,50.0,100.0,200.0]'
  done
done
echo "Done. Result JSONs under results/cifar10/openood_v1p5_pnc_*.json; aggregate to sensitivity_cifar.csv."
