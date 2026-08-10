# CIFAR target-block / layer-scope — recovered & verified

Two sources: the **3-block sweep** (early/middle/late: s1b0, s2b1, s3b0) in `../sensitivity/sensitivity_cifar_agg.csv` (block factor, anchor ps25/bf0.05/K20/λ1e-3), and the **6-block full grid** (`../joint_selection/tables/validation_nll_by_block.md`). Geometry/conditioning: `block_geometry_raw.json`, `../joint_selection/BLOCK_PATH_AUDIT.md`, `CIFAR_CONDITIONING_PHASE_DIAGRAM.md`.

## 3-block sweep (3-seed mean) — quoted ranges VERIFIED
| block | stage_idx.block | Flax layer | in→out ch | val NLL | Acc | Near AUROC | Near FPR95 | Far AUROC | Far FPR95 |
|---|---|---|---|---|---|---|---|---|---|
| s1b0 (early) | 1.0 | stage2.0 | 64→128 | 0.1473 | 95.27 | 89.39 | 45.61 | 92.01 | 28.42 |
| s2b1 (middle) | 2.1 | stage3.1 | 256→256 | 0.1515 | 94.76 | 89.34 | 39.31 | 92.81 | 25.19 |
| **s3b0 (late, selected)** | 3.0 | stage4.0 | 256→512 | **0.1304** | **95.59** | **91.55** | **33.08** | **95.09** | **18.15** |

Quoted ranges — **Acc 94.76–95.59 ✓, Near AUROC 89.3–91.6 ✓, Far AUROC 92.0–95.1 ✓** — reproduce exactly from these rows.

## Geometry (from `block_geometry_raw.json`, calib=1024)
Per-block correction dimensions (p = perturbed Conv1 param count; Cin_conv2; patch design rows). s3b0: Conv1 3×3×256×512; Conv2 corrected 3×3×512×512; bias-augmented correction design dim = 1 + 512·9. All 6 blocks + downsample flags in `BLOCK_PATH_AUDIT.md`. Conditioning per block (stable rank ~2–5; block0s ill-conditioned, ridge load-bearing) in `CIFAR_CONDITIONING_PHASE_DIAGRAM.md`.

## Why s3b0 was selected — ID-validation only (evidence)
- **Selection = lowest ID-validation NLL.** s3b0 val NLL 0.1304 < s1b0 0.1473 < s2b1 0.1515 (3-block sweep); in the **full 6-block × scale × bootstrap grid** s3b0/ps25/bf0.05 is the global ID-val-NLL optimum (rank 1/54) — see `../joint_selection/STATUS.md`. No OOD statistic entered selection (val-only builder; OOD loaded only post-freeze). `CIFAR_ID_ONLY_SELECTION.md` shows ID val-NLL recovers the OOD-optimal block with ≤0.16 oracle regret.
- **All blocks viable**, but the late block (s3b0) is both ID-optimal and OOD-optimal.

## Full 6-block coverage
The joint grid extends the 3-block sweep to all six positions (s1b0,s1b1,s2b0,s2b1,s3b0,s3b1) × 3 scales × 3 bootstrap; per-block 3×3 val-NLL tables in `../joint_selection/tables/validation_nll_by_block.md`. Solve time ≈ 92.5 s for the geometry probe (`block_geometry_raw.json` meta.runtime_sec); per-block build ~19–60 s (`../joint_selection/all_162_candidates.csv` build_secs). Peak memory per block in `../storage_memory/peak_memory/`.
