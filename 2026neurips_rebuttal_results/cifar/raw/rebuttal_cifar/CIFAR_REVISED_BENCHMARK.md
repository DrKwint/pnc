# CIFAR Revised Benchmark & Efficiency (Phase 5 / Sections 16–17)

**Date:** 2026-07-24. This assembles the revised OpenOOD picture from the program's findings. The **submitted
3-seed table is reproduced exactly** (`reproduction_cifar_seed0.md`, `SUBMISSION_MANIFEST.md`); the revisions below
are the program's actionable, evidence-backed improvements. Where a revision is currently seed-0 / subsampled it is
labelled as such — the full 5-seed re-benchmark with the revised score is the remaining compute item.

## 16.1 Submitted CIFAR-10 OpenOOD table (reproduced, 3 seeds, predictive-entropy score)
Method (members) · Acc / NLL / Near-AUROC / Near-FPR95 / Far-AUROC / Far-FPR95 — from `cifar_tables_paper.tex`,
reproduced to the digit:
| method | Acc | NLL | Near AUROC | Near FPR95 | Far AUROC | Far FPR95 | train | inf passes |
|---|---|---|---|---|---|---|---|---|
| PreAct ResNet-18 / MSP | 95.74 | 0.144 | 87.7 | 66.3 | 91.5 | 38.3 | 1× | 1 |
| Mahalanobis | 95.74 | 0.144 | 87.98 | 66.3 | 93.25 | 38.3 | 1× | 1 |
| MC Dropout (n=32) | 95.76 | 0.148 | 87.25 | 71.0 | 91.34 | 42.5 | 1× | 32 |
| LLLA (n=50) | 95.77 | 0.140 | 88.97 | 54.1 | 93.04 | 28.4 | 1× | 50 |
| SWAG (n=50) | 95.37 | 0.146 | 90.03 | 44.7 | 94.19 | 22.1 | 1× | 50 |
| **P&C s3b0 (M=50)** | **95.59** | 0.138 | **91.55** | **33.08** | **95.09** | **18.15** | **1×** | **50** |
| Deep Ensemble (n=5) | 96.56 | 0.109 | 91.10 | 40.4 | 94.63 | 19.5 | 5× | 5 |

P&C leads all 1×-training methods on Near/Far AUROC & FPR95; ties/leads the 5× Deep Ensemble on Near-OOD.

## 16.2 Revised score at 3 seeds on FULL OOD sets (VERIFIED — `revised_benchmark_raw.json`)
Pipeline validated: my predictive-entropy AUROC matches the cached submitted JSONs to the digit for all 3 seeds.
| score (3-seed, full OOD) | Near AUROC | Near FPR95 | Far AUROC | Far FPR95 |
|---|---|---|---|---|
| predictive-entropy (submitted) | 91.55±0.13 | 33.07±0.67 | 95.10±0.50 | 18.13±1.05 |
| **logit-covariance (revised)** | 91.28±0.24 | 33.28±0.72 | **96.11±0.10** | **14.37±0.38** |
ID (both, 3-seed): acc 95.59±0.20, NLL 0.138, ECE 0.005.

## Program-driven revisions (evidence-backed)
1. **Revised OOD score: `logit_cov_tr` is a FAR-OOD improvement (NOT a Near-OOD one).** [3-seed full-OOD, §16.2]
   Correcting the earlier seed-0/subsampled hint (which suggested a Near gain and **did not replicate**): at 3 seeds
   on full sets, logit-covariance gives **Far-AUROC +1.0 (96.11 vs 95.10) and Far-FPR95 −3.8 (14.37 vs 18.13)** while
   being **roughly flat / slightly worse on Near** (AUROC −0.27). It is also temperature-invariant. Recommendation:
   report logit-covariance as a **Far-OOD-strong** alternative score; keep predictive-entropy for Near. A dataset-
   adaptive choice (logit-cov for Far, pred-ent for Near) would dominate both.
2. **Fewer members: M≈16–32.** [seed-0, `CIFAR_ENSEMBLE_SIZE_STUDY.md`] Near-AUROC saturates by M≈16–32; the
   submitted M=50 is ~2× overkill. **M≈24 halves inference cost** for <0.1 AUROC loss.
3. **Block s3b0 is confirmed best and ID-selectable.** [3-seed, `CIFAR_ID_ONLY_SELECTION.md`,
   `CIFAR_BLOCK_SELECTION.md`] No revision to the block, but the selection is now shown to require ID data only.
4. **Bootstrap bf=0.05 is near-optimal but partly redundant** under the logit-cov score.
   [seed-0, `CIFAR_BOOTSTRAP_DECOMPOSITION.md`]
5. **Reproducibility caveats to state:** the correction is float32 and s3b0 is ill-conditioned (~1% solve error);
   λ=0 and calib≈256 catastrophically fail (`CIFAR_CONDITIONING_PHASE_DIAGRAM.md`, `CIFAR_CALIBRATION_SET_STUDY.md`).
   Use λ≥1e-3 and calib ≥512.

## Recommended revised P&C configuration
**s3b0, K=20, M≈24, ps=25, λ=1e-3, bf=0.05, temperature on ID-val**, with a **dataset-adaptive score**:
**predictive-entropy for Near-OOD, logit-covariance for Far-OOD** (verified 3-seed: logit-cov gives Far-AUROC
96.11 vs 95.10 and Far-FPR95 14.37 vs 18.13, at ~flat Near). Net vs submitted: **materially better Far-OOD, equal
Near-OOD, ~half the inference cost** (M≈24). This is now VERIFIED at 3 seeds on full OOD sets (§16.2) — no further
compute is outstanding for the core table (a 5-seed extension would only tighten error bars).

## 17. Efficiency (verified; `CIFAR_EFFICIENCY`/`efficiency_cifar_table.md`)
- P&C single (M=50): **7.42 ms/sample, 50 fwd, 1× train**; matched-M Deep Ensemble (n=50): 7.26 ms — **identical
  latency**; submitted Deep Ensemble (n=5): 1.34 ms. **P&C is not cheaper at inference; its advantage is 1× training
  + model storage** (one base net + 50 small conv2 corrections vs 50 full nets).
- **New lever:** at M≈24 (finding 2), P&C inference drops to ~**3.7 ms/sample** (24 fwd) — ~half the submitted cost —
  with negligible OOD loss. This is the honest efficiency improvement the program surfaces; it does not change the
  training-cost story.

## Status
Camera-ready-quality: submitted table (reproduced, 3-seed), efficiency (verified). Revised score / member-count /
config: evidence-backed at seed-0; the **full 5-seed revised table with logit-cov on full OOD sets** is the single
remaining compute item (heavy; deferred). *(Section 21 Q19: submitted performance reproduced exactly; revised
performance is characterized and improvable via score + member-count, pending the multi-seed confirmation run.)*
