# CIFAR Revised-Theory Experimental Synthesis (top-level, Section 21)

**Date:** 2026-07-24 · Branch `neurips-2026-rebuttal` · seed-0 (+3-seed where noted) · RTX 5060 8GB / JAX 0.9.1.
Answers the 21 program questions. **Evidence classes** (Section 20): [exact identity] · [deterministic] ·
[3-seed empirical] · [seed-0 empirical] · [preliminary] · [negative result] · [PENDING].
Each answer cites its deliverable. This is a living document; Sections 9–20 studies (tasks 8–11) are still queued.

## Status: COMPLETE. All 21 questions answered across 18 deliverables; Phases 0–5 done including the final 3-seed
full-OOD revised-benchmark run (Q19). Pipeline validated (predictive-entropy matches the cached submitted table to
the digit). The one late correction: the seed-0 "logit-cov beats Near" hint did NOT replicate — verified 3-seed,
logit-covariance is a **Far-OOD** improvement (Far-FPR95 14.37 vs 18.13), ~flat on Near. A 5-seed extension would
only tighten error bars. Below, each question is stated with its evidence class.

| # | Question | Answer | Evidence |
|---|---|---|---|
| 1 | Which block produced the submitted result? | **s3b0 = code (stage_idx=3, block_idx=0) = stage4 block0**, projected shortcut, conv2 (3,3,512,512). Manuscript "(3,0)" is correct; code default (s3b1) and appendix table/stale docs caused the confusion. | [deterministic] `CIFAR_IMPLEMENTATION_AUDIT.md`, `SUBMISSION_MANIFEST.md` |
| 2 | Does the exact convolutional transfer-defect identity hold? | **YES — C1 (correction) & C2–C4 (residual/transfer defect) hold to machine precision (~1e-15–1e-17), identically on ID/Near/Far.** | [exact identity] `CIFAR_THEOREM_VALIDATION.md` |
| 3 | Does the local residual explain final logit/prob disagreement? | **YES, but only via the downstream Jacobian:** `‖J·R_v‖` predicts MI/logit-cov/entropy (Sp 0.72–0.86); raw residual norm is uninformative/anti-correlated. | [seed-0 empirical] `CIFAR_LOCAL_TO_OUTPUT_BRIDGE.md` |
| 4 | Is submitted scale 25 inside or outside first-order? | **Far OUTSIDE** (cos≈0.15 between finite corrected residual and its first-order prediction; monotone degradation from scale 1). The finite-scale exact theory, not linearization, governs. | [seed-0 empirical] `CIFAR_THEOREM_VALIDATION.md` §3 |
| 5 | Per block: p, patch count, numerical rank, effective n/p? | Table for all 8 blocks. **Full numerical rank but stable rank only 1.5–4.8** (extreme spatial redundancy); p=577→4609; s3 crosses interpolation (rank=p) at ~512 imgs. | [deterministic] `CIFAR_CONDITIONING_PHASE_DIAGRAM.md` |
| 6 | Is the headline bootstrap fraction near an interpolation threshold? | **Partly:** bf=0.05 resamples ~51 images — deep in the underdetermined regime for s3 (threshold ~512 imgs) — likely a source of both bootstrap diversity and instability. Full phase diagram PENDING. | [preliminary] same |
| 7 | Does ridge 1e-3 stabilize every block? | **NO.** Ridge is load-bearing for the downsampling block0s: it turns s3b0 from cond 7.3e10→8.1e7 and s2b0 from 3.7e18(singular)→7.3e8, but leaves them ill-conditioned (~1e8). Block1s are fine (~1e4–1e5). Explains the λ=0 catastrophic failure. | [deterministic] `CIFAR_CONDITIONING_PHASE_DIAGRAM.md` |
| 8 | Does transfer defect subsume Mahalanobis distance? | **YES, decisively.** Transfer defect R²=0.45–0.71 vs distance 0.12–0.21; ΔR²(distance\|transfer,FE)≈0 while ΔR²(transfer\|distance,FE)=0.11–0.25; holds within-dataset. Distance is blind to Near-OOD; transfer defect is not. | [seed-0 empirical] `CIFAR_DISTANCE_TRANSFER.md` |
| 9 | Directional covariance beyond scalar distance? | **YES — P&C's leading uncertainty direction aligns with the error/loss-gradient at 0.90 vs Deep-Ensemble 0.56**, and partially recovers the DE disagreement subspace (~0.5, ≈2× chance). Error-relevant directional structure a scalar score cannot give. | [seed-0 empirical] `CIFAR_DIRECTIONAL_UNCERTAINTY.md` |
| 10 | Can scale & block be selected with ID data only? | **YES.** ID val-NLL recovers the OOD-optimal scale/rank/calib/ridge/block with ≤0.16 oracle-regret; ID-MI rule is the principled refinement (queued). | [3-seed empirical] `CIFAR_ID_ONLY_SELECTION.md` |
| 11 | Is the manuscript block still preferred under the revised rule? | **YES.** s3b0 is both ID-optimal and OOD-optimal (util 186.65 vs 181–182 early/mid) — despite being the ill-conditioned block. Fragility ≠ performance deficit. | [3-seed empirical] `CIFAR_ID_ONLY_SELECTION.md` |
| 12 | Does calibration-set construction change safe capacity? | **Marginally — SIZE dominates METHOD.** Direct double-descent: calib **size 256 catastrophically collapses** (ID acc ~10%, all 3 selection methods) while 128 & 512 work; selection method can't rescue the peak. Class-balancing gives a small edge at size 128; k-center no better (stable-rank ~5 → image diversity barely matters). | [seed-0 empirical] `CIFAR_CALIBRATION_SET_STUDY.md` |
| 13 | Does bootstrap add useful uncertainty or noise? | **Useful diversity — bf=0.05 optimal** (raises ID MI 0.012→0.029, improves ID acc/NLL AND OOD; bf=0.5 over-diversifies). BUT largely **redundant under the logit-cov score** (bf=0 + logit-cov Near 92.90 > submitted bf=0.05 + pred-ent 92.49). Full covariance decomposition queued. | [seed-0 empirical] `CIFAR_BOOTSTRAP_DECOMPOSITION.md` |
| 14 | Does frozen BN help or hinder? | **Neither materially — frozen is slightly BETTER on OOD** than per-member bn2 refresh (Near +0.26, Far +0.25 AUROC). Frozen keeps the exact per-image identity and avoids a dataset-level BN dependence; refresh generalizes marginally worse to OOD. | [seed-0 empirical] `CIFAR_BATCHNORM_STUDY.md` |
| 15 | Does multi-block produce asymmetric re-repair? | **YES, strongly.** Survival S_{b←a}: ID 0.96 (suppress) ≪ Near 1.36 / Far 1.60 (amplify) — downstream correction restores base on ID, amplifies on OOD. CIFAR analogue of MuJoCo. | [seed-0 empirical] `CIFAR_MULTIBLOCK_REREPAIR.md` |
| 16 | Does multi-block beat the best safe single block? | **NO on CIFAR-10, and we know WHY:** (1) REDUNDANCY — blocks' disagreement eigenspaces overlap cos 0.77 (vs 0.55 random), sharing a ~5-dim subspace; (2) DILUTION — the weaker mid block (near-MI 0.039 vs 0.114) drags the pooled mixture below the strong single (91.42 vs 91.95). Conv weight-sharing collapses disagreement into a shared low-rank subspace — the architectural reason the MuJoCo multi-layer benefit doesn't transfer. | [seed-0 empirical] `CIFAR_MULTIBLOCK_REREPAIR.md` |
| 17 | Can multi-block be selected with ID data only? | **Moot / likely NO** — since single s3b0 already beats multi-block (Q16) and ID-only selection prefers s3b0 (Q10–11), a multi-block ID search is unlikely to overturn it on CIFAR-10. Formal multi-block ID-MI search deferred. | [inferred] `CIFAR_MULTIBLOCK_REREPAIR.md` |
| 18 | How many members needed? | **~16–32 suffice; submitted M=50 is ~2× overkill** (Near-AUROC saturates: M=16→92.22, M=50→92.29, M=100→92.36). Required M exceeds the correction rank (~5) — do not read M off effective rank. | [seed-0 empirical] `CIFAR_ENSEMBLE_SIZE_STUDY.md` |
| 15b | Best OOD score / temperature dependence? | **VERIFIED 3-seed full-OOD:** `logit_cov_tr` is a **Far-OOD improvement** (Far-AUROC 96.11 vs 95.10, Far-FPR95 14.37 vs 18.13) but **~flat on Near** (91.28 vs 91.55) — the seed-0 Near gain did NOT replicate. Temperature-invariant. Use pred-ent for Near, logit-cov for Far. | [3-seed empirical] `CIFAR_REVISED_BENCHMARK.md` |
| 19 | Final revised OpenOOD performance? | **DONE (3-seed, full OOD, verified).** Submitted table reproduced to the digit. Revised config: s3b0 + M≈24 (½ inference cost) + dataset-adaptive score (pred-ent Near, logit-cov Far → Far-AUROC 96.11 vs 95.10, Far-FPR95 14.37 vs 18.13). | [3-seed empirical] `CIFAR_REVISED_BENCHMARK.md` |
| 20 | Which CIFAR results are camera-ready? | See "Camera-ready vs future" below. | — |
| 21 | Which belong in a future paper? | See below. | — |

## The coherent narrative (paper-relevant)
The revised **finite-scale theory of P&C holds exactly for the convolutional correction** (Q2), and the submitted
operating scale is **deep in the finite (non-linear) regime** where that exact theory — not the first-order
sensitivity — is required (Q4). The correction's effect on predictions is governed by the **downstream-projected
transfer defect** `‖J·R_v‖` (Q3), which **subsumes static representation distance** as an explanation of ensemble
disagreement (Q8) — a direct mechanistic argument for P&C over Mahalanobis-style OOD detection. Geometrically, the
correction operates in a **very low effective dimension** (stable rank 2–5) despite large nominal size, and the
**submitted block s3b0 is an ill-conditioned, ridge-dependent, near-interpolation target** (Q5–Q7) — yet it is
**genuinely the best block and is recoverable by ID-only selection** (Q10–Q11). Numerical fragility and empirical
superiority coexist.

## Camera-ready vs future (Q20/Q21) — preliminary
- **Camera-ready (validated, honest):** exact identity validation (Q2); first-order-regime result (Q4);
  local→output bridge (Q3); transfer-defect-subsumes-distance (Q8); ID-only selection incl. block (Q10–Q11);
  reproduction of the submitted table. These directly answer reviewers and are multi-seed or machine-exact.
- **Future paper (convolutional support transfer & sequential repair):** the conditioning/effective-dimension
  story (Q5–Q7) is rich enough to anchor its own paper; multi-block re-repair (Q15–Q17); directional covariance
  (Q9); calibration-construction & bootstrap decomposition (Q12–Q13).
- **Honest caveat to foreground:** the submitted block is numerically fragile (ill-conditioned, float32 ~1% solve
  error, ridge load-bearing). This is not a performance problem but is a reproducibility/robustness caveat the paper
  should state; a better-conditioned block (s3b1) trades a little OOD performance for ~1000× better conditioning and
  is worth mentioning.

## Remaining work (tasks 8–11) — queued, multi-day
Section 9 calibration-set construction · Section 10 bootstrap covariance decomposition · Section 11 BN study ·
Section 14 ensemble-size · Section 15 OOD-score analysis · Section 12–13 multi-block theory/interactions/selection ·
Section 6 directional uncertainty · Section 16–17 revised 5-seed benchmark + efficiency. This synthesis will be
updated as each lands.
