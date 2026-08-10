# CIFAR-10 Rebuttal Results (NeurIPS 2026, Perturb and Correct)

**Every claim below is backed by a saved result row; sources are cited inline.** Configs are the
author-confirmed submitted anchor unless labelled otherwise. Branch `neurips-2026-rebuttal`, commit `854f4d8`,
1× RTX 5060 8 GB / WSL2 / JAX 0.9.1. Generated 2026-07-24.

**Submitted anchor (all sensitivity/mechanism/ablation work varies from this):** single-block P&C, target
`s3b0` = (stage_idx=3, block_idx=0) = stage4-block0, K=20, M=50, perturbation scale=25, ridge λ=1e-3 centered on
original conv2 weights, bootstrap bf=0.05, random directions, subset=1024; OOD score = predictive_entropy
(macro-mean AUROC); temperature on the 5000-image ID-val split; BN frozen; bn2 not absorbed. Full audit:
`SUBMISSION_MANIFEST.md`; frozen config: `ANCHOR_CONFIG.json`.

---

## Reviewer concern #4 — Reproducible from an exact, auditable configuration? **YES.**

- The submitted CIFAR-10 Table 2 "PnC" row (`experiments/cifar_tables_paper.tex`) is reproduced to every
  printed digit by `s3b0/ps25/bf0.05`, 3 seeds — Acc 95.59, NLL 0.138, ECE 0.0050, Near AUROC 91.55, Near FPR95
  33.08, Far AUROC 95.09, Far FPR95 18.15. Source: `SUBMISSION_MANIFEST.md` §0; recomputed 3-seed aggregate.
- Seed-0 recomputed **from scratch** matches the cached submitted JSON at floating-point level (all metrics; the
  reproduction gate — scale selected, ID acc ≤0.1pp, AUROC ≤0.002, FPR95 ≤1pp — passes with margin).
  Source: `reproduction_cifar_seed0.md`, `reproduction_cifar_seed0_A.json`.
- A conflicting artifact set (the appendix `cifar10_ood_bootstrap_table.tex` and the Apr-10–14 narrative docs)
  describes a *different* config (`s3b1/ps50`, no bootstrap); that lineage also reproduces its own cache
  (`_B.json`) but is **not** the submitted row. The conflict is fully documented and author-resolved.

## Reviewer concern #1 — Is P&C sensitive to block / scale / rank / calibration size / ridge? **Mixed, and now precisely characterised (3 seeds).**

Source: `sensitivity_cifar_agg.csv`, `sensitivity_cifar_summary.md` (seeds 0,1,2; anchor pre-seeded from cache).

| knob | behaviour | evidence (near-AUROC / ID-acc) |
|---|---|---|
| **perturbation scale** | **narrow optimum; the dominant knob** | peak at anchor 25 (91.4); ps=50 → ID acc 35%, ps=100 → 10%. Below 25 degrades smoothly (6.25→88.6). |
| **rank K** | **flat / saturating (robust)** | K=1→90.9, K=5→91.2, K=20→91.4, K=40→91.4. Even rank-1 works; scale meaning is comparable across K (unit-norm coeffs × orthonormal dirs). |
| **calibration size** | **flat 256–1024, then a REPLICATING instability** | 256/512/1024 ≈91.4; **ss=2048 collapses in all 3 seeds** (ID acc 89/41/42%); ss=4096 healthy again. Non-monotonic → conditioning/chunk-boundary anomaly, not sample scarcity. |
| **ridge λ** | **threshold, not "no effect"** | **λ=0 fails (hard error @seed1; garbage acc≈10% @seeds0,2 — singular solve)**; λ=1e-4 degraded (~94%); λ≥1e-3 flat over 3 orders. Corrects the stale doc claim that "λ has no effect across 5 orders." |
| **target block** | **late block best** | s3b0 (anchor) 91.4 > s2b1 89.4 > s1b0 89.6. Caveat: fixed ps=25 has different meaning per block. |

**Solve failures/instabilities are recorded** (task 3D): λ=0 (singular) and ss=2048 (severe replicating collapse).
FPR95 tracks AUROC throughout — no masked regressions. Bottom line: P&C is **robust to rank and to ridge above a
threshold, flat to calibration size except one anomalous point, but genuinely sensitive to perturbation scale
(narrow optimum) and target block** — an honest, defensible sensitivity profile.

## Reviewer concern #2 — Does the distance→disagreement mechanism generalise from MuJoCo to CIFAR-10? **YES.**

Source: `mechanism_cifar_summary.md/json`, `mechanism_cifar_per_example.csv` (anchor P&C, seed 0; 17,500 examples,
2500/dataset; distance = shrinkage-covariance Mahalanobis in the block-input 256-d representation).

- **Survives regime/dataset control (Q1):** OLS slope of every disagreement metric on log₁₀(distance) with dataset
  fixed effects is positive and highly significant (t = 14–43). The link is not a between-dataset artifact.
- **Present within datasets (Q2):** member-to-base KL vs distance is positive in 7/7 datasets; strongest within
  the texture/SVHN far sets, weak-but-positive within ID; MNIST is near-flat (uniformly far, disagreement-saturated).
- **Stronger for epistemic disagreement (Q3):** within Far-OOD, member-to-base logit disagreement tracks distance
  far more tightly than predictive entropy (Spearman +0.33 vs +0.20); pooled, logit_l2 (+0.41) and KL (+0.37) lead.
  This isolates *epistemic* diversity — the reviewers' actual concern — from aleatoric class ambiguity.
- Corrected-model ID/OOD asymmetry is clear (ID mean distance 10.7 → Far 15.4; ID MI 0.031 → Far 0.249).
  Plots: `mechanism_cifar_dist_vs_{predictive_entropy,mutual_information,kl_to_base}.png`.

## Correction vs no-correction ablation (mechanism, supports #2) — **correction works within an operating range.**

Source: `correction_ablation_cifar.csv`, `correction_ablation_cifar_summary.md` (seed 0; identical perturbed conv1,
correction on/off; subsampled).

- **In the operating regime (scale ≤ anchor 25):** at scale 25 the hidden (conv1) perturbation shifts the block
  output by ~4.35, yet the CORRECTED model keeps ID acc **95.7%** (uncorrected 87.3%), ID NLL 0.135 (vs 0.529), a
  far smaller ID logit change (2.21 vs 6.02), AND better OOD (near/far AUROC 91.9/96.0 vs 78.8/65.6). The correction
  decouples large hidden perturbation from ID output change while preserving/strengthening OOD disagreement.
- **Honest caveat:** beyond the operating band (scale ≥50) both variants collapse, and at scale 100 the corrected
  solve numerically blows up. The correction absorbs the perturbation only while the induced block-output shift is
  within the affine layer's capacity; the submitted anchor (25) sits near the top of that beneficial band.

## Reviewer concern #3 — Does the first-order corrected-sensitivity object predict finite-perturbation behaviour?

**Not yet formally tested (Phase 6 linearization bridge — the task's lowest-priority, optional item).** Partial
evidence exists in the correction ablation: corrected finite-α behaviour is smooth and ID-output-suppressed through
the operating band (scale 5→25) and departs sharply only past it (≥50), consistent with a first-order regime that
holds locally and breaks at large α. A formal `jax.jvp`-through-the-ridge-solve bridge with finite-difference
cross-checks is the recommended follow-up if reviewer #3 remains a sticking point. Status: **open.**

## Efficiency (verified) — **P&C is NOT cheaper at inference.**

Source: `efficiency_cifar_table.md/csv` (GPU-synced, warmed-up; member counts verified: P&C M=50, MC Dropout n=32,
DE n=5, SWAG/LLLA/Epinet n=50).

- P&C single-block scale=25: **7.42 ms/sample, 50 fwd, 1× train**.
- **Matched-M** Deep Ensemble n=50: **7.26 ms/sample, 50 fwd, 50× train** — essentially identical latency to P&C.
- **Submitted-cost** Deep Ensemble n=5: 1.34 ms/sample, 5 fwd, 5× train — P&C is ~5.5× slower at inference.
- P&C's real advantages: **1× training cost** and **model storage** (one base net + 50 small conv2 corrections vs
  50 full network copies). Framing must not claim P&C is cheaper at inference.

---

## Deliverables (all in `results/neurips_2026_rebuttal/cifar/`)

`SUBMISSION_MANIFEST.md` · `ANCHOR_CONFIG.json` · `reproduction_cifar_seed0.md` (+JSONs) ·
`sensitivity_cifar_{raw,agg}.csv` + `sensitivity_cifar_summary.md` ·
`mechanism_cifar_per_example.csv` + `mechanism_cifar_summary.{md,json}` + 3 plots ·
`correction_ablation_cifar.csv` + `correction_ablation_cifar_summary.md` ·
`efficiency_cifar_table.{md,csv}` + `inference_cost_deep_ensemble_n50_matched.json` · `STATUS_UPDATE.md`.

## Status

Phases 0,1,2,3,4,5,7 COMPLETE (3 seeds for reproduction, sensitivity, and the submitted anchor; mechanism &
ablation at seed 0, subsampled — documented in each `_meta.json`). Phase 6 (linearization bridge) OPEN/optional.
Suggested extensions: (a) seed-1/2 for mechanism & ablation; (b) condition-number probe for the ss=2048 anomaly;
(c) the Phase-6 jvp bridge for reviewer #3.
