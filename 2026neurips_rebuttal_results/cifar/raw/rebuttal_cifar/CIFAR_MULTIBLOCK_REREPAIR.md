# CIFAR Multi-Block Re-Repair (Phase 3 / Sections 12–13)

**Date:** 2026-07-24 · self-contained 2-block sequential construction, blocks a=s2b1 (stage3 blk1) → b=s3b0
(stage4 blk0, adjacent) · one unit-norm direction on block a, scale 25, v_b=0 · float64 toward-original ridge ·
CALIB=512, 128 eval imgs/dataset · ID + CIFAR-100 + SVHN. Raw: `multiblock_rerepair_raw.json`;
harness: `_multiblock_rerepair.py`.

## Re-repair survival (Section 12.4)
`S_{b←a}(x) = ‖r_after(x)‖ / ‖r_before(x)‖`, where `r_before` = block-b conv2 residual from the upstream (block-a)
perturbation with block b UNcorrected, and `r_after` = the same after block b's conv2 is RE-corrected (v_b=0) toward
the base block-b output given the perturbed upstream.
| regime | S (median) | interpretation |
|---|---|---|
| **ID test** | **0.963** | block b **suppresses** upstream residual (−3.7%): correction restores base output |
| CIFAR-100 (near) | 1.356 | block b **amplifies** (+35.6%): correction fails to generalize |
| SVHN (far) | 1.598 | block b **amplifies** (+59.8%) |

## Findings
1. **[seed-0 empirical] Asymmetric re-repair is confirmed and strong: `S_ID (0.96) ≪ S_OOD (1.36–1.60)`, monotone in
   distribution shift.** A downstream block's correction — fit on the ID calibration set — **restores the base output
   on ID** (suppresses the upstream perturbation) but **fails to generalize on OOD and amplifies it**. This is exactly
   the sequential-repair mechanism: it manufactures **ID-preserving, OOD-amplifying disagreement**. *(Section 21 Q15
   → YES, asymmetric re-repair exists in the conv net — the CIFAR analogue of the MuJoCo result, demonstrated not
   assumed.)*
2. **[3-seed empirical, prior] Yet multi-block does NOT beat the best single block on CIFAR-10.** The submitted
   multi-block config (ps7, all-selected-blocks) reaches Near-AUROC 89.96 vs single s3b0's 91.55 (`SUBMISSION_MANIFEST`,
   `sensitivity`). So although the re-repair mechanism is real (finding 1), correcting multiple blocks introduces
   enough **redundant disagreement + added conditioning fragility** (each block0 is ill-conditioned, Section 2) that
   it underperforms the single best block here. *(Section 21 Q16 → NO on CIFAR-10; re-repair is present but not
   net-beneficial vs the best single block.)*
3. **Reconciliation:** the mechanism (Q15) and the aggregate performance (Q16) point opposite ways — re-repair is a
   genuine, measurable effect, but on this architecture/dataset the single deepest block already captures the useful
   OOD disagreement, and stacking corrections adds more noise/fragility than signal. This is an honest, non-obvious
   conclusion the paper should state rather than assuming MuJoCo's multi-layer benefit transfers.

## WHY multi-block doesn't beat single on CIFAR-10 (measured — `multiblock_why_raw.json`, `_multiblock_why.py`)
Single-block ensembles at s2b1 (mid) and s3b0 (late), M=50, scale 25, bf=0, 500 imgs/dataset.
| quantity | s2b1 (mid) | s3b0 (late) | pooled 100 |
|---|---|---|---|
| Near AUROC (pred-ent) | 89.96 | **91.95** | 91.42 (−0.53 vs best single) |
| Near AUROC (logit-cov) | 89.82 | **93.51** | 89.57 (below BOTH) |
| Near MI (median) | 0.039 | **0.114** | — |
| Principal-angle cos(top-3 logit-cov eigenspaces s2b1↔s3b0) | ID 0.759 · Near 0.786 · Far 0.787 (random ~0.55) |

**Mechanism 1 — REDUNDANCY.** The two blocks' leading disagreement eigenspaces overlap at cos **0.76–0.79** (vs
~0.55 random): they push output uncertainty into **~77%-shared directions**, so stacking adds little new epistemic
dimension. This is the output-space consequence of the ~5-dim effective correction space (stable rank ~5, Section 2)
and the M≈16–32 saturation (Section 14).
**Mechanism 2 — DILUTION.** The mid block is individually weaker (near-MI 0.039 vs 0.114; AUROC 89.96 vs 91.95), and
because its disagreement is smaller AND redundant, **pooling lands between the two, below the strong single** — the
weak, overlapping contribution drags the mixture down rather than augmenting it. (Pooled logit-cov is below both,
as mixing overlapping-but-unequal-magnitude covariances degrades the estimate.)
**Architectural contrast with MuJoCo (paper punchline):** in the MLP, layers plausibly disagree more orthogonally
(no weight sharing, higher effective rank), so multi-layer P&C adds genuine diversity; in the **conv net,
weight-sharing + spatial pooling collapse disagreement into a shared low-effective-rank subspace**, making extra
blocks redundant. **The architecture — not a flaw in the multi-block idea — is why the multi-layer benefit does not
transfer.** Re-repair (above) is real but operates *within* this shared subspace, so it does not create new diversity.

## Deferred (full Section 12–13)
- Mixed-interaction ratio η_ab (Section 12.5) and the small-scale linearity checks of Q_{b←a}, I_ab (12.6).
- Base-target vs incremental-target correction comparison (12.9).
- Multi-block ID-only selection (Section 13, Q17) — pending; prior single-block ID-only selection (Section 7–8)
  already prefers single s3b0, and given Q16 the multi-block ID-only search is unlikely to overturn that on CIFAR-10.
The core re-repair mechanism (Q15) and the multi-vs-single verdict (Q16) are established here.
