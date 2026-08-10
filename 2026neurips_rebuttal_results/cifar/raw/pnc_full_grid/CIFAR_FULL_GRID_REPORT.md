# CIFAR-10 Full P&C Grid Search — Report

Exhaustive 6×3×3 = 54 configs/checkpoint × 3 checkpoints = **162 runs**, replacing the submitted coordinate-wise selection over {target block, perturbation scale, bootstrap fraction}. K=20, M=50, λ=1e-3, calib=1024 fixed. Selection by **mean ID-val NLL** (temperature-scaled, ID-only). This closes the *search-procedure* gap; the one-factor sweeps remain a separate *robustness* claim.

## Reviewer statement

> We clarify that CIFAR-10 originally used coordinate-wise ID-validation selection over target block, perturbation scale, and bootstrap fraction. We therefore evaluated the complete 6×3×3 = 54 cross-product on each of the same three checkpoints. The global optimum was **s3b0 ps25 bf0.05**, compared with the coordinate-descent choice **s3b0 ps25 bf0.05** (rank 1/54). Its mean ID-validation NLL changed by **+0.0000**, and Near/Far AUROC changed from **91.55±0.16/95.10±0.62** to **91.55±0.16/95.10±0.62**. Thus coordinate descent **did not miss** a material interaction among the tuned variables.

## Objective answers

1. **Global optimum vs coordinate-descent:** SAME config — global s3b0/ps25/bf0.05 (val NLL 0.1305); CD s3b0/ps25/bf0.05 rank 1/54 (val NLL 0.1305).
2. **Val-NLL lost to coordinate descent:** 0.0000 (median seed std 0.0067; exceeds seed variability: False).
3. **OOD/ID change:** Near AUROC 91.55±0.16 → 91.55±0.16, Far AUROC 95.10±0.62 → 95.10±0.62, ID acc 95.59±0.26 → 95.59±0.26.
4. **Interactions:** see interaction summary; CD-path forward→s3b0 vs reverse→s3b0 vs global s3b0.

## Final OOD comparison (3-seed mean ± std)

| Config | Acc | NLL | Near AUROC | Near FPR95 | Far AUROC | Far FPR95 |
|---|---|---|---|---|---|---|
| Coordinate-descent (s3b0/ps25/bf0.05) | 95.59±0.26 | 0.138±0.004 | 91.55±0.16 | 33.12±0.82 | 95.10±0.62 | 18.14±1.38 |
| Global grid (s3b0/ps25/bf0.05) | 95.59±0.26 | 0.138±0.004 | 91.55±0.16 | 33.12±0.82 | 95.10±0.62 | 18.14±1.38 |

# Interaction summary

## Best scale & bootstrap within each block (mean val NLL)

| block | best scale | best bf | best NLL | NLL range across other factors |
|---|---|---|---|---|
| s1b0 | ps25 | bf0.05 | 0.1473 | 0.0124 |
| s1b1 | ps25 | bf0.05 | 0.1546 | 0.0030 |
| s2b0 | ps25 | bf0.1 | 0.1468 | 0.0861 |
| s2b1 | ps25 | bf0.05 | 0.1515 | 0.0187 |
| s3b0 | ps25 | bf0.05 | 0.1305 | 5.5192 |
| s3b1 | ps25 | bf0.05 | 0.1363 | 3.1888 |

## Best bootstrap fraction at each scale (pooled over blocks)

| scale | best bf | mean NLL |
|---|---|---|
| ps25 | bf0.05 | 0.1497 |
| ps50 | bf0.05 | 0.3996 |
| ps100 | bf0.05 | 1.2826 |

## Coordinate-descent path reconstruction (from the completed grid)

- Forward (block→scale→bootstrap, init ps25/bf0.05): → **s3b0 ps25 bf0.05** (NLL 0.1305)
- Reverse (bootstrap→scale→block, init s3b0/ps25): → **s3b0 ps25 bf0.05** (NLL 0.1305)
- Global optimum: **s3b0 ps25 bf0.05** (NLL 0.1305)
- Paths converge to same config: True

