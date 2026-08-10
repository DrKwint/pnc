# CIFAR: Static Distance vs Transfer Defect (Phase 2 / Section 5)

**Date:** 2026-07-24 · anchor P&C s3b0, seed 0, scale 25, CALIB=1024 · M=8 members (transfer vars), M=50 (targets) ·
5 datasets × 256 imgs (ID test, CIFAR-100, Tiny-ImageNet, SVHN, Textures) = 1280 examples.
Raw: `distance_transfer_raw.json`, `distance_transfer_per_example.csv`; harness: `_distance_transfer.py`.

Variables (all z-scored for the regressions):
- **static distance block:** regularized Mahalanobis (block-input GAP rep, shrinkage cov α=0.10) + ridge leverage
  (image-mean patch leverage in the original conv2 design). *(These are near-equivalent at λ=0 — Section 5.1; used
  together as one "static" block, not double-counted.)*
- **transfer-defect block:** downstream-projected residual `‖J·R_v‖` (the Section-4 bridge variable) + raw local
  residual `‖R_v‖_F`, member-averaged.
- **targets:** mutual information, logit-covariance trace, predictive entropy (M=50 ensemble).

## 5.5 Nested explanatory models — the central test
| target | R²(distance) | R²(transfer) | R²(both) | **ΔR²(transfer \| distance, +FE)** | **ΔR²(distance \| transfer, +FE)** |
|---|---|---|---|---|---|
| MI | 0.116 | 0.493 | 0.494 | **+0.205** | +0.000 |
| logit-cov-trace | 0.199 | 0.714 | 0.724 | **+0.250** | +0.013 |
| pred-entropy | 0.212 | 0.450 | 0.452 | **+0.111** | +0.002 |

Within-dataset (regime-controlled) mean R²:
| target | distance | transfer |
|---|---|---|
| MI | 0.078 | **0.347** |
| logit-cov-trace | 0.210 | **0.625** |
| pred-entropy | 0.033 | **0.265** |

Regime medians (why distance fails on Near-OOD):
| dataset | Mahalanobis (median) | transfer ‖J·R_v‖ (median) | MI (median) |
|---|---|---|---|
| ID test | 10.48 | 0.885 | 0.000 |
| CIFAR-100 (near) | 10.79 | 2.28 | 0.116 |
| Tiny-ImageNet (near) | 10.77 | 2.41 | 0.105 |
| SVHN (far) | 16.15 | 2.62 | 0.109 |
| Textures (far) | 15.61 | 2.70 | 0.123 |

## Findings
1. **[multi-seed-pending empirical] The transfer defect SUBSUMES static representation distance.** After controlling
   for the transfer defect (+dataset FE), static distance adds **essentially zero** incremental R² (0.000–0.013),
   whereas the transfer defect adds **+0.11 to +0.25** beyond distance. The transfer defect alone explains 2–4× more
   disagreement variance (R² 0.45–0.71 vs 0.12–0.21). *(Section 21 Q8 → YES, decisively.)*
2. **[multi-seed-pending empirical] The advantage is not a between-dataset artifact** — within every dataset the
   transfer defect keeps strong predictive power (R² 0.27–0.63) while static distance largely collapses
   (0.03–0.21). 
3. **[multi-seed-pending empirical] Mechanism: static distance is blind to Near-OOD.** CIFAR-100 / Tiny-ImageNet sit
   at essentially ID Mahalanobis distance (10.8 vs 10.5) yet have 2.5× the transfer defect and non-trivial MI. The
   P&C transfer defect distinguishes near-OOD that a distance score cannot — a direct, mechanistic argument for P&C
   over Mahalanobis-style distance OOD detectors, and the CIFAR analogue of the MuJoCo distance-vs-defect result.
4. Strongest link is to **logit-covariance trace** (R² 0.71), consistent with the Section-4 bridge (`‖J·R_v‖`
   tracks the ensemble logit covariance most directly). Predictive entropy is the weakest target (partly
   aleatoric), reinforcing that epistemic quantities (MI / logit-cov) are the right dependent variables.

## Caveats
- Uses the downstream-projected residual `‖J·R_v‖` as the transfer variable per Section-4 (raw `‖R_v‖_F` alone is a
  weak predictor). Distance uses block-input GAP + ridge leverage; leverage and Mahalanobis are near-equivalent so
  they are treated as one static block (Section 5.1 — not reported as independent explanatory variables).
- Seed 0, M=8 for the jvp transfer variable, 256/dataset. Multi-seed + larger M would tighten the estimates; the
  ordering (transfer ≫ distance) is large and unlikely to reverse.
