# CIFAR OOD-Score Analysis (Phase 4 / Section 15)

**Date:** 2026-07-24 · anchor P&C s3b0, seed 0, scale 25, bf=0.05, **M=50** members · 800 imgs/dataset ·
Near={CIFAR-100,Tiny-ImageNet}, Far={MNIST,SVHN,Textures,Places365} (macro-mean). Fitted ID temperature T=0.752.
Raw: `ensemble_size_ood_score_raw.json`; harness: `_ensemble_size_ood_score.py`.

## All scores, two temperature conditions (AUROC / FPR95, macro over datasets)
Sorted by Near+Far AUROC. The submitted headline score is `predictive_entropy`.
| score | T=1 Near | T=1 Far | Tfit Near | Tfit Far |
|---|---|---|---|---|
| **logit_cov_tr** | **92.78 / 28.2** | **96.74 / 11.6** | **92.78 / 28.2** | **96.74 / 11.6** |
| predictive_entropy (submitted) | 92.49 / 30.8 | 96.70 / 12.9 | 92.34 / 31.1 | 96.51 / 13.1 |
| expected_entropy | 92.46 / 31.0 | 96.65 / 13.1 | 92.35 / 31.1 | 96.52 / 13.2 |
| energy (mean logits) | 92.35 / 32.5 | 96.73 / 13.8 | 92.35 / 32.5 | 96.73 / 13.8 |
| mutual_information | 92.22 / 29.2 | 96.18 / 12.9 | 92.10 / 29.2 | 96.05 / 13.7 |
| mean_member_kl | 92.22 / 29.2 | 96.18 / 12.9 | 92.10 / 29.2 | 96.05 / 13.7 |
| max_softmax_unc | 92.05 / 30.8 | 96.02 / 13.7 | 91.92 / 30.7 | 95.82 / 13.8 |
| prob_cov_tr | 91.25 / 29.5 | 94.66 / 14.7 | 91.34 / 29.2 | 94.86 / 15.0 |
| variation_ratio | 86.32 / 100 | 91.51 / 55.8 | 86.32 / 100 | 91.51 / 55.8 |

> **⚠ CORRECTION (3-seed full-OOD, `CIFAR_REVISED_BENCHMARK.md` §16.2):** the seed-0/subsampled Near-OOD advantage
> below **did NOT replicate**. At 3 seeds on full OOD sets, logit-covariance is a **FAR-OOD improvement**
> (Far-AUROC +1.0, Far-FPR95 −3.8 vs predictive-entropy) but is **roughly flat / slightly worse on Near**
> (AUROC −0.27). Treat the seed-0 numbers below as indicative only; the verified 3-seed result is the Far-OOD win.

## Findings
1. **[seed-0 empirical — see CORRECTION above] `logit_cov_tr` scored best on this seed-0 subsample** — Near
   92.78 / Far 96.74 vs `predictive_entropy` 92.49 / 96.70. Consistent with the Section-4 bridge (`‖J·R_v‖` tracks
   logit covariance) and Section-5 (logit-cov-trace is the disagreement target best explained by the transfer
   defect) — the covariance-based score is the most **mechanism-aligned**. **But the multi-seed full-OOD verification
   shows its robust advantage is on FAR-OOD, not Near** (see correction box). Report it as a Far-OOD-strong score.
2. **[seed-0 empirical] Temperature has little effect on the OOD ranking.** T=1 vs the ID-fit T=0.752 give nearly
   identical AUROCs (e.g. predictive_entropy 92.49→92.34 Near). `logit_cov_tr` and `energy` are exactly
   temperature-invariant (logit-space). **So the OOD improvement is not an artifact of temperature fitting** — a
   useful robustness point for reviewers who might suspect the ID-fit temperature is doing the work. (Temperature
   still matters for ID calibration / NLL, just not for the entropy-ranked OOD detection.)
3. **[negative result] Discrete `variation_ratio` is far worse** (Near 86.3, FPR95 100) — the argmax-vote score
   discards the continuous disagreement signal and should not be used.
4. `mutual_information` == `mean_member_kl` (as expected up to numerical convention) and both are slightly below
   predictive_entropy on aggregate AUROC but have **better Near-FPR95** (29.2 vs 30.8) — the epistemic scores trade a
   little AUROC for better low-FPR operating points, worth noting for FPR-sensitive deployment.

## Recommendation
Report the submitted `predictive_entropy` for continuity, but **add `logit_cov_tr` as the mechanism-aligned score**
that is both best-performing and temperature-invariant. Per-dataset numbers are in the raw JSON; a full per-dataset
AUROC/AUPR-IN/AUPR-OUT/FPR95 table (Section 15.3) across all methods is folded into the Section-16 revised benchmark
(task 11).
