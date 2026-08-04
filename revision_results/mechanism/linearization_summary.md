# Priority 2 — linearization diagnostics (single-block P&C, correction interface)

- env=Ant-v5 seed=0 n_points=200 n_dirs=6 n_sub=4096
- jvp-vs-central-difference median rel-diff (should be ~0): eps1e-3=3.68e-02, eps1e-2=3.68e-02

## Relative linearization error (median over pts+dirs), by regime x alpha

| regime | α=0.05 | α=0.1 | α=0.25 | α=0.5 | α=1 | α=2 | α=5 | α=10 | α=20 | α=50 |
|---|---|---|---|---|---|---|---|---|---|---|
| id | 0.167 | 0.133 | 0.236 | 0.422 | 0.674 | 0.919 | 1.034 | 1.027 | 1.010 | 1.002 |
| near | 0.339 | 0.227 | 0.289 | 0.449 | 0.672 | 0.929 | 1.057 | 1.042 | 1.017 | 1.002 |
| mid | 0.697 | 0.399 | 0.371 | 0.500 | 0.727 | 0.959 | 1.061 | 1.033 | 1.011 | 0.999 |
| far | 6.031 | 2.993 | 1.216 | 0.843 | 0.889 | 1.058 | 1.075 | 1.025 | 1.006 | 1.001 |

## Cosine(r_actual, r_linear) (median), by regime x alpha

| regime | α=0.05 | α=0.1 | α=0.25 | α=0.5 | α=1 | α=2 | α=5 | α=10 | α=20 | α=50 |
|---|---|---|---|---|---|---|---|---|---|---|
| id | 0.987 | 0.992 | 0.974 | 0.913 | 0.760 | 0.493 | 0.177 | 0.055 | 0.015 | 0.008 |
| near | 0.947 | 0.975 | 0.959 | 0.902 | 0.770 | 0.517 | 0.201 | 0.084 | 0.039 | 0.036 |
| mid | 0.819 | 0.925 | 0.934 | 0.877 | 0.731 | 0.493 | 0.193 | 0.081 | 0.047 | 0.055 |
| far | 0.182 | 0.326 | 0.586 | 0.681 | 0.596 | 0.353 | 0.095 | 0.039 | 0.015 | 0.012 |

## Spearman(||r_linear||, ||r_actual||) across examples (median over dirs)

| regime | α=0.05 | α=0.1 | α=0.25 | α=0.5 | α=1 | α=2 | α=5 | α=10 | α=20 | α=50 |
|---|---|---|---|---|---|---|---|---|---|---|
| id | 0.990 | 0.993 | 0.979 | 0.961 | 0.920 | 0.863 | 0.797 | 0.739 | 0.700 | 0.662 |
| near | 0.940 | 0.964 | 0.976 | 0.968 | 0.943 | 0.896 | 0.818 | 0.773 | 0.679 | 0.664 |
| mid | 0.927 | 0.945 | 0.970 | 0.978 | 0.970 | 0.947 | 0.899 | 0.852 | 0.768 | 0.724 |
| far | 0.729 | 0.761 | 0.855 | 0.936 | 0.954 | 0.914 | 0.826 | 0.816 | 0.756 | 0.741 |

## Remainder ||r_actual - r_linear|| log-log slope vs α (small α ≤ 1)
A slope ≈ 2 supports a quadratic local remainder.

| regime | slope (α≤1) | slope (all α) |
|---|---:|---:|
| id | 1.51 | 1.31 |
| near | 1.26 | 1.24 |
| mid | 1.03 | 1.14 |
| far | 0.34 | 0.81 |

## Interpretation (two claims kept separate)

**Q1 — At what scales is the first-order approximation quantitatively accurate?**
Only at *small* perturbations. For ID/Near/Mid, cosine(r_actual, r_linear) is 0.9–0.99 for
α ≲ 0.5 and the relative error is ≤~0.5; by the operating scales used in the paper (α = 5–50,
selected by val-NLL) the pointwise approximation has fully deteriorated (cosine → 0, relative
error → ~1). So the linearization is NOT a quantitatively accurate description of the finite
corrected residual at the scales P&C actually runs at. (Far-OOD is never well-approximated,
even at α=0.05 — its residual is higher-order-dominated there.)

**Q2 — Does the linearized quantity still predict the RANKING/spatial pattern at operating scale?**
Yes. The Spearman correlation across examples between ‖r_linear‖ and ‖r_actual‖ stays high
across *all* scales, including the operating band: at α=10 it is 0.74 (ID) / 0.77 (Near) /
0.85 (Mid) / 0.82 (Far), and it is still 0.66–0.74 at α=50. So even where exact numerical
agreement is gone, the first-order corrected sensitivity remains a good predictor of *which*
inputs have large finite residuals — the ranking/geometry the theory relies on for OOD
detection survives to operating scale.

**On the quadratic remainder (reported separately, NOT conflated with the correlation).**
The log-log slope of the absolute remainder ‖r_actual − r_linear‖ vs α at small α is ≈1.0–1.5
(ID/Near/Mid), i.e. **sub-quadratic** — we do NOT observe a clean slope-2 quadratic remainder.
This is partly confounded: A_S(x)u is itself estimated by central difference (two-ε agreement
≈3.7%), whose O(ε²) error contributes a term linear in α, masking any true α² remainder at the
smallest scales. We therefore neither confirm nor claim a quadratic local remainder; we report
only that (a) small-scale pointwise accuracy is good and (b) rank-predictiveness persists to
operating scale.
