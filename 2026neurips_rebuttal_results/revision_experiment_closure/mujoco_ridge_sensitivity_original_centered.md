# MuJoCo ridge sensitivity under ORIGINAL-centred correction (Part E.9)

**11 environments x 5 seeds (0, 10, 42, 100, 200) x 9 lambda values = 495 cells, all completed.** Every other factor is held at its per-environment anchor. `ridge_center` is now derived from the configuration instead of being a hardcoded literal.

Under this convention lambda is **repair conservatism toward the base affine map**: lambda -> infinity returns the uncorrected perturbed member rather than a collapsed one. That is what makes a flat curve the expected outcome.

Near and Mid AUROC exist only for Ant-v5, HalfCheetah-v5, Hopper-v5, Humanoid-v5; the remaining 7 environments define a Far tier only, so those two columns aggregate 20 cells and the rest aggregate 55 per lambda.

| lambda | n | ID RMSE | ID NLL | Near AUROC | Mid AUROC | Far AUROC | Far NLL |
|---|---|---|---|---|---|---|---|
| 0 | 55 | 12.3892 | -0.3294 | 0.8068 | 0.8681 | 0.9196 | 1.9180 |
| 1e-05 | 55 | 16.8236 | 0.0326 | 0.8085 | 0.8763 | 0.9235 | 2.6583 |
| 0.0001 | 55 | 13.5703 | -0.4120 | 0.8049 | 0.8752 | 0.9267 | 1.9014 |
| 0.0003 | 55 | 13.3568 | -0.4790 | 0.8007 | 0.8731 | 0.9264 | 1.7210 |
| 0.001 | 55 | 12.7940 | -0.4980 | 0.7934 | 0.8675 | 0.9246 | 1.8420 |
| 0.003 | 55 | 12.5934 | -0.5073 | 0.7841 | 0.8570 | 0.9224 | 2.0815 |
| 0.01 | 55 | 12.4623 | -0.5131 | 0.7729 | 0.8406 | 0.9167 | 2.4101 |
| 0.1 | 55 | 12.3485 | -0.5099 | 0.7550 | 0.8134 | 0.8967 | 2.9185 |
| 1 | 55 | 12.3076 | -0.4810 | 0.7417 | 0.7990 | 0.8832 | 3.0920 |

## Reading

- Far AUROC at the operating lambda = 1e-4 is **0.9267**; at lambda = 0 it is 0.9196 (+0.0071).
- The largest deviation anywhere on the grid is -0.0435 Far AUROC, at lambda = 1.
- Across lambda = 1e-5 ... 1e-2 (three orders of magnitude) Far AUROC stays within 0.010 (0.9167-0.9267), so the reported result does not rest on a delicately tuned lambda. Beyond that the curve degrades gently and monotonically (0.8967 at lambda=0.1, 0.8832 at lambda=1) — the graceful behaviour expected when lambda shrinks toward the original map rather than toward zero.

Source: `mujoco_ridge_sweep_original_centered.csv` (per-cell), this file's companion `.csv` (per-lambda aggregates).
