# CIFAR-10 rebuttal experiments — run on the CIFAR machine

This machine (TITAN X Pascal, 12 GB) has **no CIFAR base checkpoints and no OpenOOD data**,
so the CIFAR arms of the rebuttal are prepared here for you to run where CIFAR is set up.
They mirror the MuJoCo Priority-1/2/3 designs so the two domains are directly comparable.

## Prerequisites (on the CIFAR machine)
1. Base ResNet-18 checkpoints for seeds 0,1,2:
   ```
   for S in 0 1 2; do
     python -m luigi --module pnc_core.cifar_tasks CIFARTrainPreActResNet18 \
       --dataset cifar10 --epochs 300 --seed $S --local-scheduler
   done
   ```
2. OpenOOD CIFAR-10 datasets available at `openood_data/` (Near: CIFAR-100, Tiny-ImageNet;
   Far: MNIST, SVHN, Textures, Places365) — see `data.py:_OPENOOD_CIFAR_SPEC`.

## Anchor / submitted default (mark clearly in all outputs)
`CIFARPnC` single-block at **stage4/block1**: `n_directions=10`, `n_perturbations=50`,
`subset_size=1024`, `lambda_reg=1e-3`, Lanczos directions, `perturbation_sizes=[10,50,100,200]`,
OOD score = predictive entropy of the ensemble-mean softmax. OOD AUROC/FPR95 via
`CIFAROpenOODPnC`. Keep every non-swept knob at this default; screen 1 seed then run seeds 0,1,2.

## Priority 1 — one-factor sensitivity (mirror of MuJoCo)
Run `bash run_cifar_sensitivity.sh` (edit SEEDS at top). It sweeps, one factor at a time:

| Axis | Knob | Grid |
|---|---|---|
| 1A scale | `--perturbation-sizes` | relative 0.25/0.5/1/2/4 × default (default = val-selected size) |
| 1B rank | `--n-directions` | 1, 2, 5, 10*, 20 |
| 1C subset | `--subset-size` | 256, 512, 1024*, 2048 (CIFAR "fraction" = # calibration examples; there is **no bootstrap_frac** in `CIFARPnC` — the calibration set is a fixed `subset_size` draw, so this axis is the CIFAR analogue of the MuJoCo bootstrap/subset study) |
| 1D block | `--target-stage-idx/--target-block-idx` + `CIFARMultiBlockPnC` | early (stage1/block0), mid (stage2/block1), late (stage4/block1*), and single-vs-multi block |
| 1E ridge | `--lambda-reg` | 0, 1e-4, 1e-3*, 1e-2, 1e-1, 1.0 |

`*` = submitted default. Report ID accuracy, Near AUROC/FPR95, Far AUROC/FPR95 per setting
(machine-readable CSV). For CIFAR, also record how BN is handled (frozen+absorbed, per
`cifar_tasks.py:1302`) — the conv correction absorbs BN2.

## Priority 3 — CIFAR distance–disagreement mechanism (the PREFERRED second domain)
Compute, on the corrected block's feature space (penultimate D=512 or the block output):
regularized Mahalanobis distance to the calibration distribution vs per-example P&C
disagreement (predictive entropy / mutual information), for ID test, Near-OOD, Far-OOD.
Report pooled + within-regime Spearman, regime-controlled log-distance OLS slope, binned
means. This is the direct CIFAR analogue of the submitted Ant-v5 diagnostic and of
`scripts/neurips_2026_rebuttal/priority3_mechanism.py` (MuJoCo). A helper skeleton is in
`cifar_mechanism_skeleton.py` (fill in the feature extractor for your checkpoint).

## Priority 2 — CIFAR linearization bridge (optional, if feasible)
Same design as `scripts/neurips_2026_rebuttal/priority2_linearization.py` but perturb the
block's conv1 kernel and refit conv2 (im2col patch ridge). Differentiate the corrected
residual through the ridge solve with `torch.func.jvp` (PyTorch base) or `jax.jvp`, and
cross-check with central differences at two epsilons.

## Outputs
Write everything under `results/neurips_2026_rebuttal/` on the CIFAR machine
(`sensitivity_cifar.csv`, `mechanism_second_domain.csv`, etc.) and copy back here to merge
into `REBUTTAL_RESULTS.md`. Record git commit, command, seed, hardware, runtime per result.
