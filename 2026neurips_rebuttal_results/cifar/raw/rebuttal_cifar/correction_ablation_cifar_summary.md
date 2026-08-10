# Phase 5 — Correction vs No-Correction Ablation (CIFAR-10, anchor lineage, seed 0)

Single-block PnC at s3b0 (K=20, M=50, bf=0.05). Both variants share the IDENTICAL perturbed conv1 (same directions/coefficients/scale/members); they differ ONLY in whether the fitted affine conv2 correction is applied. `hidden_pert_mag` = mean per-member block-output shift with the uncorrected conv2 (the raw effect of the hidden conv1 perturbation). Metrics on balanced subsamples; OOD score = predictive_entropy; raw (untempered) logits. Anchor scale = 25.

## Full table

| scale | variant | hidden_pert_mag | corrected_shift | id_acc | id_nll | id_logit_change | id_pred_entropy | near_auroc | near_fpr95 | far_auroc | far_fpr95 | near_pred_entropy | far_pred_entropy |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 5.0 | corrected | 0.3208 | 0.2664 | 95.55 | 0.1539 | 0.3464 | 0.0622 | 88.769 | 58.05 | 93.576 | 26.875 | 0.5498 | 0.7193 |
| 5.0 | uncorrected | 0.3208 | 0.2664 | 95.6 | 0.1545 | 0.444 | 0.0649 | 88.419 | 58.7 | 93.133 | 30.475 | 0.5535 | 0.7164 |
| 12.5 | corrected | 1.3549 | 0.6195 | 95.65 | 0.1399 | 0.8975 | 0.0821 | 90.221 | 42.925 | 94.483 | 21.613 | 0.6703 | 0.8563 |
| 12.5 | uncorrected | 1.3549 | 0.6195 | 95.4 | 0.1571 | 2.2146 | 0.1488 | 88.539 | 50.65 | 91.843 | 33.013 | 0.8017 | 0.9185 |
| 25.0 | corrected | 4.3515 | 1.2592 | 95.7 | 0.1353 | 2.2062 | 0.1809 | 91.947 | 32.6 | 95.953 | 15.625 | 1.052 | 1.2979 |
| 25.0 | uncorrected | 4.3515 | 1.2592 | 87.3 | 0.5293 | 6.0225 | 0.9222 | 78.775 | 60.675 | 65.597 | 66.588 | 1.4831 | 1.2057 |
| 50.0 | corrected | 12.1476 | 5.0025 | 50.0 | 1.2793 | 8.0937 | 1.5104 | 66.258 | 81.1 | 79.435 | 58.05 | 1.7062 | 1.8461 |
| 50.0 | uncorrected | 12.1476 | 5.0025 | 54.4 | 1.1681 | 8.6175 | 1.1911 | 70.261 | 83.75 | 79.863 | 46.662 | 1.532 | 1.6783 |
| 100.0 | corrected | 29.2769 | 4823.8785 | 9.45 | 13.6453 | 2754.2898 | 0.8116 | 47.71 | 95.7 | 65.958 | 84.325 | 0.8001 | 0.9112 |
| 100.0 | uncorrected | 29.2769 | 4823.8785 | 20.65 | 2.6529 | 15.2369 | 1.0672 | 57.877 | 87.3 | 75.376 | 51.825 | 1.1946 | 1.4037 |

## Key contrasts

**ID output stability under growing hidden perturbation.** As the hidden (conv1) perturbation grows, the corrected variant holds ID accuracy/NLL and keeps ID logits close to the base, while the uncorrected variant degrades:

| scale | hidden_mag | corrected: id_acc / id_nll / id_logitΔ | uncorrected: id_acc / id_nll / id_logitΔ |
|---|---|---|---|
| 5.0 | 0.3208 | 95.55 / 0.154 / 0.35 | 95.60 / 0.154 / 0.44 |
| 12.5 | 1.3549 | 95.65 / 0.140 / 0.90 | 95.40 / 0.157 / 2.21 |
| 25.0 | 4.3515 | 95.70 / 0.135 / 2.21 | 87.30 / 0.529 / 6.02 |
| 50.0 | 12.1476 | 50.00 / 1.279 / 8.09 | 54.40 / 1.168 / 8.62 |
| 100.0 | 29.2769 | 9.45 / 13.645 / 2754.29 | 20.65 / 2.653 / 15.24 |

**OOD disagreement retained.** Near/Far AUROC (predictive_entropy) for corrected vs uncorrected:

| scale | corrected near/far AUROC | uncorrected near/far AUROC |
|---|---|---|
| 5.0 | 88.8 / 93.6 | 88.4 / 93.1 |
| 12.5 | 90.2 / 94.5 | 88.5 / 91.8 |
| 25.0 | 91.9 / 96.0 | 78.8 / 65.6 |
| 50.0 | 66.3 / 79.4 | 70.3 / 79.9 |
| 100.0 | 47.7 / 66.0 | 57.9 / 75.4 |

## Verdict (honest — the correction has an operating range)

**In the operating regime (scale ≤ anchor 25) the correction decisively separates the hidden perturbation from the ID output.** 
At the anchor scale (25.0) the hidden (conv1) perturbation moves the block output by ~4.3515, yet the CORRECTED model keeps ID accuracy 95.70% (vs 87.30% uncorrected), ID NLL 0.135 (vs 0.529), and a far smaller ID logit change (2.21 vs 6.02) — while OOD detection is BETTER (near/far AUROC 91.9/96.0 vs 78.8/65.6). The uncorrected perturbation of the same magnitude corrupts ID predictions, which also destroys the OOD signal (predictive-entropy separation collapses because ID entropy rises too).

**Beyond the operating regime (scale ≥ 50) the affine correction can no longer compensate and the advantage disappears.** 
At scale 50 both variants fall to ~50% ID acc; at scale 100 the corrected solve numerically blows up (ID logit change ≈ 2754, ID acc 9.45% vs uncorrected 20.65%). So the correction does NOT help unconditionally — it works by absorbing the perturbation in the affine conv2 layer, which succeeds only while the induced block-output shift is within the layer's compensating capacity. The submitted anchor (scale 25) sits near the top of that beneficial band, which is exactly where hidden diversity is maximal but ID output is still protected.

**Bottom line:** at the operating scale the affine correction permits a large hidden perturbation while suppressing ID output change and preserving/strengthening OOD disagreement — the core mechanism claim — with the honest caveat that this holds within a bounded scale range, not at arbitrary over-perturbation.

Artifacts: `correction_ablation_cifar.csv`, `correction_ablation_cifar_meta.json`.