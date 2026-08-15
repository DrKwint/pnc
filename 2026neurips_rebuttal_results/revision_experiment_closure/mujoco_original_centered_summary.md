# MuJoCo: original-centred vs zero-centred ridge

Same base checkpoints, splits, seeds, K=20, M=50, bootstrap 0.1, target layers, perturbation-size grid and evaluation data. The only change is the ridge centre.

Pairs available: **20** of 20 (env x seed).

## Hyperparameter reselection (C.5)

Perturbation size is selected per (env, seed) by lowest ID-validation NLL — the manuscript's own ID-only rule, applied independently under each centre. **1 of 20** cells selected a different size under the original-centred solve.

| env | seed | size (zero) | size (original) | changed |
|---|---|---|---|---|
| Ant-v5 | 0 | 5 | 5 | no |
| Ant-v5 | 10 | 5 | 5 | no |
| Ant-v5 | 42 | 5 | 5 | no |
| Ant-v5 | 100 | 5 | 5 | no |
| Ant-v5 | 200 | 5 | 5 | no |
| HalfCheetah-v5 | 0 | 5 | 5 | no |
| HalfCheetah-v5 | 10 | 5 | 5 | no |
| HalfCheetah-v5 | 42 | 5 | 5 | no |
| HalfCheetah-v5 | 100 | 5 | 5 | no |
| HalfCheetah-v5 | 200 | 5 | 5 | no |
| Hopper-v5 | 0 | 5 | 5 | no |
| Hopper-v5 | 10 | 5 | 5 | no |
| Hopper-v5 | 42 | 5 | 5 | no |
| Hopper-v5 | 100 | 5 | 5 | no |
| Hopper-v5 | 200 | 5 | 5 | no |
| Humanoid-v5 | 0 | 50 | 50 | no |
| Humanoid-v5 | 10 | 50 | 50 | no |
| Humanoid-v5 | 42 | 50 | 50 | no |
| Humanoid-v5 | 100 | 50 | 50 | no |
| Humanoid-v5 | 200 | 10 | 50 | **yes** |

## Paired differences (original minus zero), headline envs only

Ant-v5, HalfCheetah-v5, Hopper-v5 — the three environments in the manuscript's gym table. Bootstrap: 10,000 replicates over the (env, seed) pairs, seed 20260815, percentile interval.

| metric | n | mean | median | max abs | 95% CI | excludes 0 |
|---|---|---|---|---|---|---|
| rmse_id | 15 | -0.0144 | -0.0002 | 0.2355 | [-0.0468, +0.0026] | no |
| nll_id | 15 | -0.0191 | +0.0000 | 0.4026 | [-0.0777, +0.0141] | no |
| nll_ood_near | 15 | -0.0313 | +0.0013 | 0.2591 | [-0.0819, +0.0081] | no |
| nll_ood_mid | 15 | -0.0355 | -0.0004 | 0.3434 | [-0.0854, -0.0037] | **yes** |
| nll_ood_far | 15 | -0.0073 | -0.0036 | 0.0614 | [-0.0195, +0.0048] | no |
| auroc_ood_far | 15 | -0.0018 | +0.0000 | 0.0351 | [-0.0077, +0.0025] | no |
| auroc_ood_near | 15 | +0.0017 | +0.0001 | 0.0447 | [-0.0054, +0.0098] | no |
| auroc_ood_mid | 15 | -0.0003 | +0.0000 | 0.0307 | [-0.0064, +0.0055] | no |

## Paired differences including Humanoid-v5

Humanoid is included for completeness only. P&C's Far AUROC there is 0.22-0.47 under **both** centres, i.e. at or below chance, so its swings are not informative about the ridge centre and they dominate any pooled mean.

| metric | n | mean | median | max abs | 95% CI | excludes 0 |
|---|---|---|---|---|---|---|
| rmse_id | 20 | +1.5693 | +0.0002 | 18.2882 | [-0.0098, +3.6794] | no |
| nll_id | 20 | +0.2868 | +0.0005 | 4.5015 | [-0.0143, +0.7867] | no |
| nll_ood_near | 20 | +0.2157 | +0.0021 | 3.6044 | [-0.0253, +0.6227] | no |
| nll_ood_mid | 20 | +0.1293 | +0.0002 | 2.5155 | [-0.0382, +0.4077] | no |
| nll_ood_far | 20 | +0.0770 | -0.0022 | 1.6770 | [-0.0325, +0.2579] | no |
| auroc_ood_far | 20 | +0.0061 | +0.0000 | 0.1857 | [-0.0098, +0.0292] | no |
| auroc_ood_near | 20 | +0.0052 | +0.0001 | 0.0647 | [-0.0024, +0.0141] | no |
| auroc_ood_mid | 20 | +0.0147 | +0.0000 | 0.2752 | [-0.0031, +0.0443] | no |

## Cross-environment ranks and wins

| metric | centre | mean rank | wins |
|---|---|---|---|
| rmse_id | zero | 1.45 | 11/20 |
| rmse_id | original | 1.55 | 9/20 |
| nll_id | zero | 1.25 | 15/20 |
| nll_id | original | 1.75 | 5/20 |
| nll_ood_near | zero | 1.35 | 13/20 |
| nll_ood_near | original | 1.65 | 7/20 |
| nll_ood_mid | zero | 1.40 | 12/20 |
| nll_ood_mid | original | 1.60 | 8/20 |
| nll_ood_far | zero | 1.60 | 8/20 |
| nll_ood_far | original | 1.40 | 12/20 |
| auroc_ood_far | zero | 1.60 | 8/20 |
| auroc_ood_far | original | 1.40 | 12/20 |
| auroc_ood_near | zero | 1.60 | 8/20 |
| auroc_ood_near | original | 1.40 | 12/20 |
| auroc_ood_mid | zero | 1.70 | 6/20 |
| auroc_ood_mid | original | 1.30 | 14/20 |

The purpose of this comparison is provenance and consistency, not to argue that either centre is empirically superior.
