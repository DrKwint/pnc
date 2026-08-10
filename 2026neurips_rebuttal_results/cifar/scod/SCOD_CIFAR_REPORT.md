# SCOD on CIFAR-10 (OpenOOD v1.5) — Report

**Variant:** SCOD-1024 (primary, resource-matched to P&C's 1024 ID calib pool). k=10, Meps=5000, T=124, tempered posterior_pred score. 3 checkpoint seeds [0, 1, 2]. Git 854f4d847c, JAX 0.9.1, Flax 0.12.3.

## 1. Protocol & checkpoint provenance
- Checkpoints (SHA-256): seed0=b4c45628adaa; seed1=2e942d691915; seed2=9b5d9de691ba
- Base temperatures (ID-val fit): seed0=1.3120, seed1=1.3595, seed2=1.3291
- Split: seed 99, 10% val (5000) / 45000 train; 1024 calib = RandomState(seed).choice(45000,1024,replace=False) on x_tr (no bootstrap).
- Normalization mean/std [0.4914, 0.4822, 0.4465], [0.247, 0.2435, 0.2616]; deterministic eval (no crop/flip).
- Near ['CIFAR-100', 'TinyImageNet'] / Far ['MNIST', 'SVHN', 'Textures', 'Places365']; macro-mean over datasets; ± = sample std over seeds. No OOD used for build/selection.

## 2. Implementation (JAX/Flax adaptation)
The SCOD spec is written for PyTorch; this repo is JAX/Flax nnx. We implement the faithful analogue: the categorical-logit Fisher square-root transform `ztilde_c = sqrt(p_c)(z_c - sum_k p_k z_k)` (p stop-gradiented), per-example Fisher-weighted parameter Jacobians via `jax.jacrev`, BN frozen with `use_running_average=True` (no state mutation), and a memory-safe randomized Fisher sketch (Nyström recovery, GPU column-blocks; the 5.5GB P×T state is streamed to host and touched O(1) times). All logit-influencing params are included (conv kernels, BN affine, fc weight+bias; P=11,172,170).

## 3. Categorical-Fisher validation (Section 6)
- 6.1 Fisher factor test: ||J_ztilde a||² == aᵀ(diag(p)−ppᵀ)a to 1e-15 (float64) / 5e-7 (float32). PASS.
- 6.2 Tiny-network reference: sketch+scorer vs exact eigendecomposition — eigenvalue match 1e-6, score correlation = 1.000000, scores nonnegative. PASS.
- 6.3 Model immutability: weights bit-unchanged after sketching on all seeds = True.

## 4. Memory & runtime
- Preflight: P=11172170, sketch state ≈ 5.54GB (host), peak GPU ≈ 4.6GB (8GB card).
- Build ≈ 342s/seed; serialized sketch 894MB.
- Online B=1: base 1.999ms, SCOD 21.447ms (10.7x), 46.6 img/s.
- Online B=32: base 4.036ms, SCOD 555.989ms (137.7x), 57.6 img/s.
- Online B=128: base 12.762ms, SCOD 2317.346ms (181.6x), 55.2 img/s.
SCOD needs a test-time per-example parameter Jacobian, so it is NOT a single ordinary forward pass.

## 5. Per-dataset results
See `tables/scod_per_dataset_auroc.md` and `_fpr95.md`.

## 6. Macro Near/Far results
| | Acc | Near AUROC | Near FPR95 | Far AUROC | Far FPR95 |
|---|---|---|---|---|---|
| SCOD-1024 (k=10) | 95.74±0.18 | 89.69±0.11 | 39.40±1.17 | 92.56±0.45 | 21.41±1.06 |

## 7-8. Sensitivity
Rank (k=5,10,20) and Meps (500/5000/50000) grid in `tables/scod_sensitivity.md` (ID-predeclared, not OOD-selected). Sketch-seed sensitivity: see `tables/` if multiple sketch seeds were built; otherwise the primary deterministic sketch seed (100000+checkpoint_seed) was used.

## 9. Comparison with P&C and baselines
Full table: `tables/cifar10_with_scod.md`. SCOD shares P&C's deployment setting (one frozen checkpoint, post-hoc ID data, no retraining) but returns a local Fisher-curvature score rather than explicit finite predictive members.

## 10. Limitations
- SCOD score needs per-example parameter Jacobians at test time (not a plain forward).
- Randomized sketch adds a sketch seed; primary uses one deterministic seed.
- Full-parameter sketch is memory-heavy (P×T≈5.5GB); recovered via Nyström on an 8GB GPU.
- JAX adaptation of a PyTorch method; validated by exact tiny-network agreement, not by matching the reference PyTorch code line-for-line.

## 11. Reviewer-facing paragraph
*(≈136 words)*

> We added SCOD, a directly relevant frozen-checkpoint post-hoc comparator, using the same three CIFAR-10 checkpoints, ID-only calibration data, temperature scaling, and OpenOOD v1.5 evaluation. SCOD constructs a low-rank Fisher sketch of the frozen classifier and returns a local curvature-based epistemic score; unlike P&C, it does not construct explicit predictive members. SCOD-1024 attains ID accuracy 95.74±0.18% (unchanged base classifier), Near-OOD AUROC 89.69±0.11 / FPR95 39.40±1.17 and Far-OOD AUROC 92.56±0.45 / FPR95 21.41±1.06. SCOD requires a one-off 342s Fisher-sketch build per checkpoint and a test-time per-example parameter Jacobian (21.447ms/image, 10.7x a base forward), compared with P&C's 1x training plus 50 corrected forward passes at inference. These results show SCOD is weaker than P&C on this benchmark while sharing its frozen-checkpoint, post-hoc setting, and clarify that P&C's gains are not solely due to comparison against training-time or last-layer methods.

## Addendum — sensitivity findings & completion status

**Meps-invariance (mechanistic):** across Meps ∈ {500, 5000, 50000} the scores are identical to
2 decimals. Reason: the recovered Fisher eigenvalues are O(1–9), so the shrinkage
`s_j² = λ_j/(λ_j + 1/(2·Meps))` ≈ 1 for every tested Meps (1/(2·500)=1e-3 ≪ λ). Meps would only
matter for eigenvalues near 1/(2·Meps). **Rank:** Near-AUROC saturates by k=10 (89.54→89.69→89.69
for k=5→10→20), consistent with a stable rank of only ~6–7. Temperature helps marginally
(tempered 89.69 vs untempered 89.32 Near-AUROC).

**Completion status.** DONE: SCOD-1024 on all 3 checkpoints; all 7 OpenOOD datasets; main +
per-dataset (AUROC/FPR95) + rank/Meps sensitivity tables; model-immutability, finite/nonneg-score,
ID-acc==base, and macro-reproducibility gates all PASSED; runtime/memory profile; per-example
Parquet retained; 9 figures; MANIFEST. NOT DONE (heavy, deferred): (1) **sketch-seed sensitivity**
(Section 12 — 3 sketch seeds per checkpoint for k=10; only the primary deterministic seed
100000+s was built), so the `SCOD_CIFAR_COMPLETE` marker is intentionally NOT written yet;
(2) **SCOD-full-train** secondary variant (Section 3.2); (3) `scod_vs_pnc_scatter.png` (needs P&C
per-example scores). The primary resource-matched SCOD-1024 result is complete and validated.
