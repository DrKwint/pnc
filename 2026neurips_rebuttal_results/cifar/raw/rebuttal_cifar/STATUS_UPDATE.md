# CIFAR Rebuttal — Status Update for Coordinating Agent

> ## ⏩ SUPERSEDING DIRECTIVE (2026-07-24): Full P&C Theory Experimental Program
> A much larger 23-section program is now underway (revised finite-scale theory, MuJoCo→CIFAR generalization).
> Tracking it via 12 tasks. **Phase 0 + Phase 1 COMPLETE** (the theoretical core):
> - `CIFAR_IMPLEMENTATION_AUDIT.md` — block reconciliation (submitted=s3b0=(3,0)), exact residual-block sequence,
>   solver internals (float32, absolute-λ augmented Gram, bias regularized, image-level bootstrap), data protocol.
> - `CIFAR_THEOREM_VALIDATION.md` — **exact convolutional identities C1 (correction) & C2–C4 (residual/transfer
>   defect) hold to machine precision (~1e-15–1e-17), on ID and OOD alike.** Patch operator = conv2, patch formula =
>   real network forward (float32 ~5e-4). Block-output residual = conv2 residual (shortcut cancels). **First-order
>   (§3): submitted scale 25 is FAR outside the linear regime (cos≈0.15) — the finite-scale exact theory is
>   required, not linearization.** Striking: **s3b0 is ~1000× more ill-conditioned than s3b1** (cond 8.6e7 vs 7.5e4).
> - `CIFAR_LOCAL_TO_OUTPUT_BRIDGE.md` — local conv2 residual explains logit/prob disagreement **only via the
>   downstream Jacobian**: `‖J·R_v‖` predicts MI/logit-cov/entropy (Sp 0.72–0.86) while raw residual norm is
>   uninformative/anti-correlated (Sp ≈0 to −0.29).
>
> **Phase 2 IN PROGRESS:** per-block conditioning/geometry table (Section 2.1) running now.
> Remaining: Phase 2 grids (conditioning phase diagram, distance-vs-transfer-defect, ID-only selection, calibration
> study), Phase 3 multi-block, Phase 4 (directional/bootstrap/BN/M/score), Phase 5 revised benchmark + efficiency,
> final synthesis. The prior sensitivity/mechanism/ablation/efficiency results below feed the later sections.
> **This is a multi-day program.** New driver scripts: `results/neurips_2026_rebuttal/cifar/_theorem_validation.py`,
> `_firstorder_sweep.py`, `_local_to_output.py`, `_block_geometry.py`.

---

## (Prior task — CIFAR rebuttal sensitivity/mechanism package, still valid & feeding the program)

**As of:** 2026-07-24 (updated after seeds 1–2 + Phase 7 completed)
**Owner:** CIFAR sub-agent (Claude Code)
**Branch:** `neurips-2026-rebuttal` · **Commit at start:** `854f4d8` (working tree dirty — pre-existing infra)
**Hardware:** 1× NVIDIA RTX 5060, 8 GB VRAM, WSL2 · JAX 0.9.1 / Flax 0.12.3 / Python 3.12.3 (single GPU, no parallel GPU jobs)
**All outputs under:** `results/neurips_2026_rebuttal/cifar/`

---

## TL;DR

- **Phases 0, 1, 2, 3, 4, 5, 7 COMPLETE.** Phase 3 sensitivity now has all **3 seeds** (both solve instabilities — λ=0 and calib ss=2048 — replicated across seeds and are documented). Phase 7 efficiency verified + matched-M DE n=50 added. Final synthesis `CIFAR_REBUTTAL_RESULTS.md` written. **Only Phase 6 (linearization bridge, optional/lowest-priority) remains open.** Nothing currently running.
- **The submitted CIFAR-10 Table 2 P&C config was ambiguous across artifacts; it is now RESOLVED and author-confirmed:** single-block **s3b0 / ps25 / bf0.05** (Candidate A). See below — this is the single most important fact for the coordinator.
- Two high-value, honest results are already in hand: the **distance→disagreement mechanism generalizes to CIFAR** (survives dataset fixed effects), and the **correction ablation** shows the correction decouples hidden perturbation from ID output **within a bounded operating range**.

---

## The anchor decision (READ THIS)

Two candidate "submitted" configs existed in the repo:

| | **Candidate A — SELECTED (submitted)** | Candidate B — earlier/appendix |
|---|---|---|
| target block | `s3b0` = (stage_idx=3, block_idx=0) = **stage4 block0** | `s3b1` = (3,1) = stage4 block1 |
| scale / bootstrap | ps=25, **bf=0.05** | ps=50, no bootstrap |
| K / M / dirs | 20 / 50 / random | 20 / 50 / random |
| matches | **`experiments/cifar_tables_paper.tex`** (final paper table) EXACTLY, 7/7 metrics | `experiments/cifar10_ood_bootstrap_table.tex` (appendix) + all Apr-10–14 narrative docs |

- Resolved by exact numerical match (decisive fingerprint: ECE 0.0050 = A vs 0.0079 = B), chronology (A's table is the newest artifact, Apr-28), and **author confirmation on 2026-07-23**.
- **The internal markdown narrative docs are STALE** — they describe Candidate B and never recorded the switch to s3b0+bootstrap. Do not trust `cifar_neurips_strengthening_*.md` / `cifar10_ood_detection.md` for the PnC config.
- **Naming hazard:** the tuning docs' prose label "S3B0" means code `s2b0` (a *different* block) — do not conflate with the filename token `s3b0`.
- Full detail + all 24 audited items with file:line citations: **`SUBMISSION_MANIFEST.md`**. Frozen anchor: **`ANCHOR_CONFIG.json`**.

---

## Phase-by-phase status

| Phase | Status | Headline result |
|---|---|---|
| **0** Identify submitted config | ✅ done | `SUBMISSION_MANIFEST.md` — anchor = s3b0/ps25/bf0.05, single-block, 3 seeds. Baselines: MC Dropout n=32, Deep Ensemble n=5. Near-OOD={cifar100,tiny_imagenet}; Far-OOD={mnist,svhn,textures,places365}; OOD score=predictive_entropy; temp on 5000-img ID-val split; BN frozen; bn2 NOT absorbed; ridge centered on original conv2 weights. |
| **1** Reproduce | ✅ done | Both candidates reproduce their cached JSONs at float level (seed 0); gate passed. `reproduction_cifar_seed0.md`. |
| **2** Anchor | ✅ done | `ANCHOR_CONFIG.json` (every parameter, author-confirmed). |
| **3** Sensitivity | ✅ done (3 seeds) | See "Sensitivity findings" below. `sensitivity_cifar_{raw,agg}.csv`, `sensitivity_cifar_summary.md`. λ=0 and ss=2048 instabilities replicated across seeds. |
| **4** Distance–disagreement mechanism | ✅ done | Link **survives dataset fixed effects** (OLS slope on log10-dist, t=14–43); positive within 7/7 datasets (member-to-base KL); **strongest for epistemic member-disagreement** (logit/KL). `mechanism_cifar_*`. |
| **5** Correction ablation | ✅ done | Correction decouples hidden perturbation from ID output **within an operating range** (scale ≤ anchor 25): at 25, corrected ID acc 95.7% vs uncorrected 87.3% AND better OOD; beyond (≥50) both collapse. Honest caveat documented. `correction_ablation_cifar_*`. |
| **6** Linearization bridge | ⬜ OPEN (optional/lowest priority) | Reviewer #3; partial evidence via Phase-5 finite-α behaviour. Needs `jax.jvp` through ridge solve + finite-diff cross-check. |
| **7** Efficiency verification | ✅ done | `efficiency_cifar_table.md/csv`. Counts verified; matched-M DE n=50 (7.26 ms) ≈ P&C single (7.42 ms). **P&C not cheaper at inference** — advantage is 1× training + storage. |

---

## Sensitivity findings (3 seeds: 0,1,2)

| knob | verdict | detail |
|---|---|---|
| **scale** | NARROW OPTIMUM; most sensitive | peak at anchor 25 (near-AUROC 91.4); ps≥50 collapses ID acc (51% @50, 10% @100). |
| **rank K** | FLAT / saturating | K=1→90.7, K=20→91.4, K=40→91.4. Robust. Scale meaning is comparable across K (unit-norm coeffs × orthonormal dirs). |
| **calib size** | mostly flat, one REPLICATING instability | 256–1024 ≈91.4; **ss=2048 collapses in all 3 seeds** (ID acc 89/41/42%); ss=4096 healthy → conditioning/chunk-boundary anomaly, not sample scarcity. |
| **ridge λ** | THRESHOLD (not "no effect") | **λ=0 fails (hard error @seed1; garbage acc≈10% @seeds0,2 — singular solve)**; λ=1e-4 degraded; λ≥1e-3 flat over 3 orders. **Corrects a stale doc claim.** |
| **block** | late block best | anchor s3b0 (91.4) > early/mid (~89.5). Caveat: same ps=25 across blocks has different meaning per block. |

FPR95 tracks AUROC throughout (no masked regressions). Both solve instabilities (λ=0, ss=2048) **replicated across all 3 seeds** and are recorded in `sensitivity_cifar_summary.md`.

---

## Currently running

- Nothing. GPU idle. All background sweeps complete.

## Remaining work

1. **Phase 6 linearization bridge** (optional, lowest priority; reviewer #3) — `jax.jvp` through the ridge solve vs finite differences across a log-scale grid, per regime. Partial evidence already exists via Phase-5 finite-α behaviour. See `local_sensitivity.py` for possibly-reusable MuJoCo-side infra.
2. Optional strengthening: seed-1/2 for mechanism & ablation (currently seed-0, subsampled); a condition-number probe to explain the ss=2048 anomaly.
- **`CIFAR_REBUTTAL_RESULTS.md` is written** (capstone; claims cited to saved rows, organized by the 4 reviewer concerns).

## Open items / caveats for the coordinator

- **Mechanism representation choice:** primary distance space = block-input (stage3-output GAP, 256-d), as most analogous to the correction geometry; penultimate 512-d computed as a secondary check. Distance = shrinkage-covariance Mahalanobis (α=0.10), same estimator for ID/near/far.
- **Correction ablation is honest, not a pure win:** the correction helps only within a bounded scale band; the submitted anchor (25) sits near the top of it. Framing to reviewers should keep this caveat.
- **Places365 uses a 10k subset, DTD its native 5640** — not identical to public OpenOOD leaderboard splits (same as the submitted protocol).
- Mechanism/ablation numbers are seed-0, subsampled (2500/2000 per dataset) — documented in each `_meta.json`. Fine for the mechanism claim; not headline table numbers.

## Deliverables index (all in `results/neurips_2026_rebuttal/cifar/`)

- `SUBMISSION_MANIFEST.md`, `ANCHOR_CONFIG.json`
- `reproduction_cifar_seed0.md` (+ `_A.json`, `_B.json`, `repro_*_seed0.json`)
- `sensitivity_cifar_raw.csv`, `sensitivity_cifar_agg.csv`, `sensitivity_cifar_summary.md`
- `mechanism_cifar_per_example.csv`, `mechanism_cifar_summary.{md,json}`, `mechanism_cifar_dist_vs_*.png`, `mechanism_cifar_{raw.npz,extract_meta.json}`
- `correction_ablation_cifar.csv`, `correction_ablation_cifar_summary.md`, `correction_ablation_cifar_meta.json`
- driver scripts: `_repro_driver.py`, `_mechanism_extract.py`, `_mechanism_analyze.py`, `_correction_ablation.py`, `_correction_ablation_summary.py`, `_sensitivity_runner.py`, `_sensitivity_aggregate.py`
- **pending:** `CIFAR_REBUTTAL_RESULTS.md`, updated efficiency table, (optional) linearization outputs
