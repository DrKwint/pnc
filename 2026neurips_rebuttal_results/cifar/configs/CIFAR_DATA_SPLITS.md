# CIFAR-10 P&C data splits & correction-set provenance (verified from code)

Source files (repo DrKwint/pnc, commit 854f4d8): `util.py`, `cifar_tasks.py`, `pnc.py`, `ensembles.py`, `data.py`.

## Dataset sizes
| split | size | source |
|---|---|---|
| CIFAR-10 train (full) | 50,000 | `data.load_cifar10()` |
| CIFAR-10 test | 10,000 | `data.load_cifar10()` |
| ID **train-fit** split | **45,000** | `util._split_data(..., val_split=0.1, seed=99)` → `RandomState(99).permutation(50000)[5000:]` |
| ID **validation** split | **5,000** | same → `permutation(50000)[:5000]` |
| P&C **correction/calibration subset** | **1,024** | `cifar_tasks._build_single_block_pnc_ensemble` |
| OOD (Near) | cifar100 10,000; tiny_imagenet 10,000 | `data.load_openood_cifar_benchmark`, `openood_data/` |
| OOD (Far) | mnist 10,000; svhn 26,032; textures 5,640; places365 ~36,500 | same |

## Correction subset — exact provenance (verified)
- **Size = 1024**, drawn from the **45,000-example train-fit split** (`x_tr`), NOT from validation or test.
  `idx = np.random.RandomState(seed).choice(len(x_tr)=45000, 1024, replace=False)` — **without replacement**; `seed` = the checkpoint seed. (`cifar_tasks.py` `_build_single_block_pnc_ensemble`, lines ~405–407.)
- **Same 1024 pool is reused across all hyperparameter configurations** for a given checkpoint (deterministic from the checkpoint seed) — so scale, bootstrap, rank, ridge, and block sweeps share identical calibration points per seed. (Confirmed: the seed argument = checkpoint seed for every config.)
- **Bootstrap happens INSIDE this 1024 pool.** When `0 < bootstrap_frac < 1`, the correction is fit on a resample of size `bootstrap_frac × 1024` drawn via `RandomState(bootstrap_seed)` (`ensembles.py` lines ~201, 353–356). bootstrap_seed = checkpoint seed. The bootstrap is a resample of the calibration pool; base pool selection is unchanged.

## Labels & leakage (verified)
- **The correction solve uses NO class labels.** It is a toward-original ridge regression on the Conv2 **output**: it accumulates `H = ΣMᵀM`, `b = ΣMᵀR` with `R_i = T_i − Y_v·w2_orig` (residual of the perturbed block output relative to the **original block output** `T_i`), and solves `(H+λI)Δw = b` for the correction to `w2_orig` plus a new bias. (`pnc.py` `_ridge_body_jit` lines 110–133, `_ridge_regression_solve` line 246.) No targets/labels appear.
- **Validation examples never enter the correction.** `x_va` is used only to fit the temperature (§ inference protocol); the correction uses `x_tr`'s 1024 subset only.
- **OOD examples never enter construction or selection.** OOD sets are loaded only at final OpenOOD evaluation, after the configuration and temperature are frozen. (Confirmed across SCOD/grid/sensitivity runs; the val-only grid builder never imports the OOD benchmark.)

## Summary
`correction subset size = 1024` ✓ — from the 45k train-fit split, without replacement, labels unused, reused across configs per seed, bootstrap resampling internal. No validation or OOD leakage into construction or hyperparameter selection.
