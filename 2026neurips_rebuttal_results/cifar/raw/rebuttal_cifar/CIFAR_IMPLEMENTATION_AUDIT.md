# CIFAR Implementation Audit (Section 0 — gate for the full program)

**Date:** 2026-07-24 · **Branch:** `neurips-2026-rebuttal` · **Purpose:** establish exactly what the CIFAR P&C
code does before any theory instrumentation or large grid. Complements `SUBMISSION_MANIFEST.md` (config
provenance) with the deeper residual-block / solver / data / provenance audit the program requires.

## 0.5 Machine provenance
- host `DESKTOP-B1JCO6F`; GPU **NVIDIA RTX 5060, 8151 MiB, driver 580.88**; 1 GPU; WSL2.
- Python 3.12.3 · JAX 0.9.1 · jaxlib 0.9.1 · Flax 0.12.3 · platform=gpu (CUDA via jaxlib wheel).
- git commit `854f4d8`, working tree **dirty** (pre-existing experiment infra + this program's outputs).
- Determinism env: `XLA_FLAGS` unset; runs set `XLA_PYTHON_CLIENT_PREALLOCATE=false`. No mixed precision;
  the correction solver runs in **float32** (see 0.3). Each program script re-stamps host/commit/GPU into its meta JSON.

## 0.1 Target-block reconciliation (RESOLVED)
Authoritative table (code indices are 0-based `(stage_idx, block_idx)`; `stages=[stage1,stage2,stage3,stage4]`):

| Name | Code (stage,block) | Human block | C_in→C_out (conv2) | conv2 spatial | shortcut | Submitted? | Saved results? |
|---|---|---|---|---|---|---|---|
| s0b0 | (0,0) | stage1 blk0 | 64→64 | 32×32 | identity | no | val-sweep only |
| s0b1 | (0,1) | stage1 blk1 | 64→64 | 32×32 | identity | no | — |
| s1b0 | (1,0) | stage2 blk0 | 128→128 | 16×16 | **projected** | no | sweep |
| s1b1 | (1,1) | stage2 blk1 | 128→128 | 16×16 | identity | no | sweep |
| s2b0 | (2,0) | stage3 blk0 | 256→256 | 8×8 | **projected** | no | sweep |
| s2b1 | (2,1) | stage3 blk1 | 256→256 | 8×8 | identity | no | sweep |
| **s3b0** | **(3,0)** | **stage4 blk0** | **256→512 (conv2 512→512)** | **4×4** | **projected** | **YES (submitted anchor)** | yes (3 seeds) |
| s3b1 | (3,1) | stage4 blk1 | 512→512 | 4×4 | identity | no (appendix lineage) | yes (3 seeds) |

- **Manuscript label "(3,0)" is CORRECT and equals code `s3b0`** (both 0-indexed). The confusion was: (a) the code
  *default* `target_block_idx=1` (→ s3b1), and (b) the appendix `cifar10_ood_bootstrap_table.tex` + stale narrative
  docs used s3b1. The submitted main-table row is reproduced to every digit by s3b0 (see `SUBMISSION_MANIFEST.md`
  §0; ECE 0.0050 fingerprint). Different blocks WERE used for exploration vs the final table — the switch
  s3b1→s3b0 happened Apr-28 via `scripts/sweep_pnc_block_position_seeds_1_2.sh` and was never written into the docs.
- Both `s3b0` and `s3b1` were re-run under submitted hyperparameters and each reproduces its own cached JSON at
  float level (`reproduction_cifar_seed0.md`). Author-confirmed anchor = s3b0.
- **Program convention:** always use code indices + tensor shapes; "s3b0 = (stage_idx=3, block_idx=0) = stage4
  block0, conv2 (3,3,512,512), 4×4 spatial, projected shortcut."

## 0.2 Exact residual-block computation (verified in `cifar_tasks.py`/`ensembles.py`/`models.py`)
Sequence for the target block (`_forward_member`, `ensembles.py:1718-1768`; `get_Y_fn`, `cifar_tasks.py:133-152`):
1. block input `h`; 2. `bn1(h, use_running_average=True)`; 3. `relu` → `out_relu1`; 4. **perturbed `conv1`**
(`w1_pert`, stride per block, SAME); 5. `bn2(use_running_average=True)`; 6. `relu` → `U_v(x)` (= `y_relu2`);
7. **corrected `conv2`** (`w2_new`) **+ new bias `b2_new`**; 8. shortcut = `downsample(out_relu1)` if projected
else `h`; 9. residual add `t + identity`; 10. downstream blocks; 11. global-average pool; 12. `fc`.
- **Perturbed representation** = `U_v(x)` = post-`bn2`-`relu` output of perturbed conv1 (the tensor `im2col`-ed for
  the conv2 correction). **Patches** are `extract_patches(U_v, k=3, stride=conv2.stride)`.
- conv2 stride/pad/dilation/groups: **uniform across all blocks** — stride (1,1), SAME, dilation 1, groups 1,
  `use_bias=False`. Verified for all 8 blocks.
- Shortcut is **identity** for block1 of every stage; **1×1 projected conv (stride 2)** for block0 of stages 1–3
  (s1b0,s2b0,s3b0). The shortcut is **never modified** by P&C.
- **Crucial for Section 4:** steps 1–3 (`h`, `bn1`, `relu`→`out_relu1`) precede conv1, so both the shortcut and
  `out_relu1` are **identical between the original block and any perturbed-corrected member**. Therefore the
  **block-output residual (after residual addition) equals the conv2-branch residual `R_v(x)` exactly** — the
  residual add and shortcut contribute zero deviation. (Empirically confirmed downstream in theorem validation.)
- Corrected bias `b2_new` is added to the conv2 output **before** the residual addition (step 7 before 9).
- **No activation follows the residual addition** inside the block (PreAct design; the next block's `bn1`/`relu`
  act on the sum). 
- BN uses **base running statistics (`use_running_average=True`) in every correction and evaluation pass** — no
  per-member BN update in the headline method.
- Data augmentation: the correction/calibration inputs come from the raw ID training tensors (no random crop/flip
  applied at correction time; augmentation is a train-time-only transform). Confirmed: `_load_cifar_openood_context`
  passes un-augmented `x_tr`.

## 0.3 Correction solver (verified in `pnc.py:110-289`)
- **dtype:** float32 throughout — activation accumulation, Gram `Xᵀv X_v` (`M_iᵀ M_i`), RHS, and solve are all
  float32 (`_ridge_body_jit`, `solve_chunked_conv2_correction`). No float64 anywhere in the shipped path.
- **Gram accumulation:** chunked — `H += M_iᵀM_i`, `b += M_iᵀR_i` over calibration chunks (`pnc.py:283-288`).
- **Solver:** `jnp.linalg.solve(H + λI, b)` — a linear solve, **no explicit inverse** is formed (`pnc.py:246-250`).
- **Ridge placement/scale:** **absolute** `λ·I_{D_M}` added to the full augmented Gram, `D_M = p = 9·C_in+1`
  (`pnc.py:249`). λ is the raw `lambda_reg` (submitted 1e-3), NOT trace-normalized.
- **Bias coordinate IS regularized:** the ones column is index 0 of `M=[1, Y_v]`, and `λI` spans all `D_M` rows
  including index 0. So the bias delta is ridge-penalized identically to weights.
- **Original conv2 bias treated as zero:** conv2 has `use_bias=False`; the target `R = T − Y_v·w2_orig` uses no
  original bias, and `b2_new = Θ_delta[0]` is the entire (delta-from-zero) bias.
- **Toward-original ridge (matches manuscript, C1):** regressing `M=[1,Y_v]` onto `R = XΘ − X_vΘ = −ΔX_v·Θ` and
  setting `w2_new = w2_orig + Δ` is algebraically `Θ_v = Θ − G_v⁻¹ X_vᵀ ΔX_v Θ` (derivation in
  `CIFAR_THEOREM_VALIDATION.md`). The penalty is on `‖Δθ‖²` = deviation from original weights.
- **Per output channel:** all `C_out` columns share one normal matrix `H` (same design `M`) and are solved jointly
  in one `solve` (`b` is `D_M×C_out`) — equivalent to independent per-channel ridge with a shared Gram factorization.
- **Bootstrap level:** IMAGE-level. `bootstrap_frac` resamples calibration *images* with replacement per member
  (`ensembles.py`), so a duplicated image contributes **all** of its spatial patch rows again (patch rows are not
  resampled independently). bootstrap_seed = seed.
- Calibration patches are **not centered or standardized**; the raw post-bn2-relu patches are used, with the ones
  column providing the intercept/bias.

## 0.4 Data protocol (verified; assertions to be enforced by program scripts)
Disjoint splits (`util._split_data(seed=99, val_split=0.1)`, `cifar_tasks._load_cifar_openood_context`):
- base-model training: full 50k CIFAR-10 train (the checkpoint is pre-trained).
- P&C calibration candidate pool: `x_tr` = the **45k training portion** (the 90% not in val); the correction
  subset is `RandomState(seed).choice(len(x_tr), subset_size, replace=False)`.
- BN-refresh split: N/A in headline (BN frozen).
- temperature-fit split: the **5000-image ID validation** split (first 10% of train, seed=99) — **disjoint** from
  the calibration pool.
- ID config-selection split: NOT yet separated from temp-fit in the shipped code → the program (Section 7) will
  introduce a stratified `temp_fit`/`config_select` partition or 2-fold cross-fitting to avoid selection leakage.
- final test: full 10k CIFAR-10 test.
- Near-OOD: CIFAR-100, Tiny-ImageNet-200; Far-OOD: MNIST, SVHN, Textures(DTD), Places365. Full official test sets
  (Places365 = 10k subset, DTD = native 5640). npz/imglist under `openood_data/`, normalized to CIFAR stats,
  32×32. Counts verified (cifar100 10000, tiny_imagenet 10000, mnist 10000, svhn 26032, textures 5640,
  places365 10000).
- **OOD-leak assertion (to add to every program script):** any function that fits a correction, selects ridge,
  scale, block, ensemble size, or temperature, or makes a stopping decision, must assert its inputs come only from
  {calibration pool, ID val, ID config-select} and never from any `near_ood`/`far_ood` loader. A shared
  `assert_no_ood(split_name)` guard will wrap those entry points.

## Status
Audit complete. **The program may proceed to Phase 1 (exact-identity validation).** Remaining Section-0 hardening
(the runtime `assert_no_ood` guard) will be added inside the Phase-2 selection scripts, where OOD leakage is an
actual risk (Phase 1 uses OOD data only for the residual identity, which is a legitimate final-evaluation use).
