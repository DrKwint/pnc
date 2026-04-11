#!/usr/bin/env bash
# Run all CIFAR-10 OpenOOD evaluations with frozen ID hyperparameters.
# No OOD data is used for tuning or model selection.
set -euo pipefail

PYTHON=".venv/bin/python"
LUIGI="$PYTHON -m luigi --module cifar_tasks --local-scheduler"
EPOCHS=300  # matches existing trained checkpoints

# --- Uncertainty-estimation methods (frozen ID hyperparameters) ---

$LUIGI CIFAROpenOODPreActResNet18 --posthoc-calibrate --epochs $EPOCHS --seed 0

$LUIGI CIFAROpenOODStandardEnsemble --posthoc-calibrate --epochs $EPOCHS --n-models 5 --seed 0

$LUIGI CIFAROpenOODMCDropout --posthoc-calibrate --epochs $EPOCHS --n-perturbations 32 --dropout-rate 0.1 --seed 0

$LUIGI CIFAROpenOODSWAG --posthoc-calibrate --epochs $EPOCHS --n-perturbations 50 --swag-start-epoch 240 --seed 0

$LUIGI CIFAROpenOODLLLA --posthoc-calibrate --epochs $EPOCHS --n-perturbations 50 --prior-precision 10.0 --seed 0

$LUIGI CIFAROpenOODEpinet --posthoc-calibrate --epochs $EPOCHS --n-perturbations 50 --prior-scale 3.0 --seed 0

$LUIGI CIFAROpenOODPnC --posthoc-calibrate --epochs $EPOCHS --perturbation-sizes '[20.0]' --n-directions 20 --random-directions --seed 0

$LUIGI CIFAROpenOODMultiBlockPnC --posthoc-calibrate --epochs $EPOCHS --perturbation-sizes '[6.0]' --n-directions 20 --random-directions --seed 0

# --- Dedicated OOD baselines (frozen single PreAct ResNet-18) ---

$LUIGI CIFAROpenOODMSP --posthoc-calibrate --epochs $EPOCHS --seed 0

$LUIGI CIFAROpenOODEnergy --posthoc-calibrate --epochs $EPOCHS --seed 0

$LUIGI CIFAROpenOODMahalanobis --posthoc-calibrate --epochs $EPOCHS --seed 0

$LUIGI CIFAROpenOODReActEnergy --posthoc-calibrate --epochs $EPOCHS --seed 0

echo "All OOD evaluations complete."
