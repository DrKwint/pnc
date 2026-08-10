# CIFAR Ensemble-Size Study (Phase 4 / Section 14)

**Date:** 2026-07-24 · anchor P&C s3b0, seed 0, scale 25, bf=0.05, CALIB=1024 · one **M=100 member pool**,
nested random subsets (5 per M except M=100), 800 imgs/dataset, Near={CIFAR-100,Tiny-ImageNet},
Far={MNIST,SVHN,Textures,Places365}, predictive-entropy score, T=0.752 (ID-fit).
Raw: `ensemble_size_ood_score_raw.json`; harness: `_ensemble_size_ood_score.py`.

## Convergence vs M
| M | Near AUROC | Far AUROC | ID MI | subset std (near) |
|---|---|---|---|---|
| 2 | 90.79 | 95.14 | 0.0140 | ±0.41 |
| 4 | 91.51 | 95.83 | 0.0218 | ±0.35 |
| 8 | 91.90 | 96.19 | 0.0265 | ±0.18 |
| 16 | 92.22 | 96.45 | 0.0293 | ±0.16 |
| 32 | 92.27 | 96.52 | 0.0311 | ±0.07 |
| **50 (submitted)** | 92.29 | 96.54 | 0.0318 | ±0.07 |
| 64 | 92.33 | 96.55 | 0.0319 | ±0.03 |
| 100 | 92.36 | 96.57 | 0.0321 | ±0.00 |

## Findings
1. **[seed-0 empirical] The ensemble saturates by M≈16–32.** From M=32 to M=100, Near-AUROC rises only +0.09
   (92.27→92.36) and Far-AUROC +0.05; M=16 is within 0.14 of M=100. **The submitted M=50 is ~2× more members than
   needed** — M=16–32 captures essentially all the OOD benefit. Even M=8 (91.90 Near) is close. *(Section 21 Q18:
   ~16–32 members suffice; 50 is comfortably converged but overkill.)*
2. **[seed-0 empirical] Member-subset variance vanishes by M≈32** (subset std ±0.41 at M=2 → ±0.07 at M=32 →
   ±0.00 at M=100). Which particular members are chosen stops mattering beyond ~32.
3. **[seed-0 empirical] ID mutual information converges on the same schedule** (0.014→0.032, saturating by ~32),
   confirming the epistemic-disagreement measure — not just the AUROC — is what plateaus.
4. **Efficiency implication:** halving to M≈24 would roughly halve P&C inference cost (Section 17: latency ∝ member
   forward passes) for <0.1 AUROC loss. This is an actionable, honest efficiency lever the paper can note (and does
   not require the full M=50 the submission used).

## Method note (per spec 14.3)
Required M was determined **empirically**, not inferred from effective rank. Note the effective (stable) rank of the
correction design is only ~4.8 (Section 2), which would naively suggest ~5 members — but convergence needs ~16–32,
i.e. **more than the correction rank**. Disagreement magnitude keeps accruing beyond the rank-implied count, exactly
the caution the spec flags: do not read required M off the effective rank.
