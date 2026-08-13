# Preservation-frontier P&C on ImageNet-1K ViT-B/16

## 1. Executive summary

> **How large can the final-block ViT perturbation become before P&C correction can no
> longer preserve ImageNet accuracy?**

Under a 0.50 pp top-1 budget, **r = 2.0** — passing through r = 2.0 and failing by
r = 2.125, a boundary bisected to a width of 0.125 (5.9 %). Under the stricter 0.25 pp
budget, r = 1.25; under 1.00 pp, r = 3.0. By r = 4.0 no ridge value preserves accuracy
under any of the three budgets.

> **Does correction increasingly separate P&C from uncorrected perturbations as that
> boundary is approached?**

Yes, and sharply. The two are indistinguishable up to r ≈ 1.25, cross near r ≈ 1.5–2, and
then diverge:

| r | P&C Δ top-1 | uncorrected Δ top-1 |
|---|---|---|
| 1.0 | −0.094 pp | −0.008 pp |
| 1.5 | −0.216 pp | −0.090 pp |
| 2.0 | −0.305 pp | −0.358 pp |
| 3.0 | −0.655 pp | −2.238 pp |
| 4.0 | −0.960 pp | **−10.754 pp** |

**The single most important result is that the original experiment's operating point was a
ridge artefact, not a scale limit.** The first protocol fixed λ = 1e-3 during its scale
sweep and concluded that r = 0.375 was the largest ID-stable perturbation. Searching ridge
at every scale, the *same* 0.25 pp accuracy criterion supports **r = 1.25 — 3.3× larger** —
and the primary 0.50 pp budget supports **r = 2.0, 5.3× larger**. At r = 1.0 the difference
is stark: λ = 1e-3 loses 0.346 pp; λ = 1000 loses 0.094 pp.

That larger operating point converts directly into OOD performance. Against the original
experiment's cleanly-selected result and the strongest baseline:

| | Near AUROC | Near FPR95 | Far AUROC | Far FPR95 |
|---|---|---|---|---|
| MSP | 73.52 | 81.84 | 86.04 | 51.74 |
| P&C, original protocol (r = 0.375) | 74.74 | 74.37 | 86.52 | 48.75 |
| **P&C, PRIMARY (r = 2.0)** | **76.49** | **67.65** | **87.89** | **44.97** |
| matched uncorrected at r = 2.0 | 72.53 | 71.66 | 84.40 | 46.51 |

All four of the protocol's "particularly strong outcome" criteria are met: ridge tuning
supports a substantially larger r; at that scale the matched uncorrected ensemble loses
materially more ID performance; corrected P&C ranks OOD better than the matched uncorrected
ensemble (+3.96 Near AUROC, where the original protocol's gap was +0.14, i.e. noise); and a
clear corrected preservation boundary exists at larger r.

## 2. Why the original selection rule was changed

The first ImageNet experiment gated on three conditions: corrected top-1 drop ≤ 0.25 pp,
base agreement ≥ 99 %, and the correction beating matched uncorrected perturbations on both
median and p99 logit MSE. The last two are the problem.

Requiring the correction to beat the uncorrected ensemble *on every deviation statistic*
only admits perturbations small enough that the correction is already winning — which is
precisely the regime where the perturbation barely moves the network and there is nothing
to correct. The original experiment's own attribution analysis showed the consequence:
member disagreement was under 1 % of the predictive entropy, corrected and uncorrected
rankings correlated at ρ = 0.998, and the two scored identically on OOD.

That gate answers "how large a perturbation leaves the *uncorrected* model benign". The
question P&C is about is "how large a perturbation can the *corrected* model repair". This
protocol asks the second one.

## 3. New preservation-budget definition

Full statement in [`ID_SELECTION_RULE.md`](ID_SELECTION_RULE.md), committed before any
follow-up OOD evaluation. In brief:

- Scale is chosen as the **largest r the corrected model supports** within a predeclared
  top-1 budget. The uncorrected ensemble is an ablation and enters no selection decision.
- Three budgets, fixed in advance: STRICT 0.25 pp, **PRIMARY 0.50 pp** (main operating
  point), RELAXED 1.00 pp.
- Preservation is decided by a **one-sided 95 % lower confidence bound** from a paired
  bootstrap over the 8,192 ID-selection examples (10,000 replicates, seed 20260814), not by
  a point estimate. Because the paired difference takes few distinct values, each replicate
  is a multinomial draw over the empirical counts — exactly equivalent to resampling
  examples, and cheap enough to run for all 88 configurations.
- Agreement, logit-MSE comparisons and calibration residual are **diagnostics only**.
- A `CALIBRATION_PATHOLOGY` safeguard rejects only catastrophic cases (NaN/Inf, NLL > 2×
  base, ECE > 3× base) rather than a narrow tolerance that would recreate the old gate.

## 4. Dataset and protocol reuse

Everything reusable is reused byte-for-byte from `../imagenet_vit/`: the class-stratified
disjoint training pools (correction 32,768, selection 8,192, temperature 8,192; manifests
and SHA-256 unchanged), the official 50,000-image validation set, the pinned checkpoint,
the CLS-only correction observation, K = 20, n_cal = 32,768, and the shared temperature
**T = 0.700** — not refitted per configuration, so ID NLL reflects the predictor rather
than each configuration recalibrating itself. The deterministic MSP / Energy / ReAct+Energy
baselines are reused verbatim (identical checkpoint, images, preprocessing and scoring
code).

## 5. Perturbation × ridge search

11 scales × 8 ridge values × 3 construction seeds at M = 10, i.e. 88 configurations and 264
ensembles. Within a seed the basis and member coefficient directions are drawn once and
reused at every scale, so scale comparisons are **paired**; only the scalar multiplier
changes. Realized median r matches the target to 3 decimal places at every point.

Two implementation choices make this affordable on one 12 GiB card: for a fixed
(scale, seed, member) the Gram statistics `G = XᵀX` and `C = Xᵀz₀` do not depend on λ, so
all eight ridge values share one GPU pass and differ only by a Cholesky solve; and the
bootstrap is multinomial rather than index-based. A whole ridge curve at one scale costs
91 s.

**Ridge matters more than the original protocol assumed, and the right value moves with
scale.** At r = 0.5 every ridge from 1e-3 to 1e4 passes all three budgets. At r = 2.0 only
λ ≥ 1000 passes anything. The old fixed λ = 1e-3 is adequate below r ≈ 1 and badly wrong
above it.

The safeguard earns its place: λ = 1e4 gives the best *raw accuracy* at every scale ≥ 2.0,
but is flagged `CALIBRATION_PATHOLOGY` there (ECE 0.057 → 0.181 against base 0.010, up to
17.7×; NLL up to 2.5× base). Selection therefore lands on λ = 1000 at all three operating
points. Without that guard the protocol would have chosen configurations that preserve
accuracy by destroying calibration.

## 6. Corrected ID preservation frontier

Full table in [`tables/table1_preservation_frontier.md`](tables/table1_preservation_frontier.md).
Each row uses that scale's ID-selected ridge.

| r | λ | corrected top-1 | Δ pp | 95 % LCB | NLL | ECE | agree | uncorr. Δ pp | S | P | R |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.5 | 1e4 | 93.840 | +0.016 | −0.041 | 0.2067 | 0.0080 | 0.9978 | +0.004 | ✓ | ✓ | ✓ |
| 0.75 | 1e4 | 93.787 | −0.037 | −0.110 | 0.2080 | 0.0066 | 0.9963 | −0.016 | ✓ | ✓ | ✓ |
| 1.0 | 1e3 | 93.730 | −0.094 | −0.191 | 0.2100 | 0.0058 | 0.9948 | −0.008 | ✓ | ✓ | ✓ |
| **1.25** | 1e3 | 93.730 | −0.094 | −0.203 | 0.2137 | 0.0055 | 0.9934 | −0.045 | **✓** | ✓ | ✓ |
| 1.5 | 1e3 | 93.608 | −0.216 | −0.354 | 0.2188 | 0.0086 | 0.9907 | −0.090 | · | ✓ | ✓ |
| **2.0** | 1e3 | 93.518 | −0.305 | −0.460 | 0.2321 | 0.0146 | 0.9876 | −0.358 | · | **✓** | ✓ |
| 2.125 | 1e3 | 93.424 | −0.399 | −0.562 | 0.2356 | 0.0153 | 0.9865 | −0.464 | · | · | ✓ |
| 2.25 | 1e3 | 93.408 | −0.415 | −0.582 | 0.2390 | 0.0164 | 0.9859 | −0.606 | · | · | ✓ |
| 2.5 | 1e3 | 93.359 | −0.464 | −0.639 | 0.2453 | 0.0184 | 0.9843 | −0.977 | · | · | ✓ |
| **3.0** | 1e3 | 93.168 | −0.655 | −0.854 | 0.2545 | 0.0187 | 0.9812 | −2.238 | · | · | **✓** |
| 4.0 | 1e3 | 92.863 | −0.960 | −1.188 | 0.2617 | 0.0122 | 0.9767 | −10.754 | · | · | · |

Note the ordering flip in the last two columns. Up to r ≈ 1.5 the *uncorrected* ensemble
preserves accuracy better than the corrected one — the correction's own estimation error
exceeds the damage it repairs when the perturbation is this small. They cross at r = 2.0,
and past that the corrected model is better by an order of magnitude.

## 7. Preservation ceiling (explicit)

| budget | passes through | fails by | selected r | λ |
|---|---|---|---|---|
| STRICT 0.25 pp | **r = 1.25** | r = 1.5 | 1.25 | 1000 |
| **PRIMARY 0.50 pp** | **r = 2.0** | **r = 2.125** | **2.0** | **1000** |
| RELAXED 1.00 pp | **r = 3.0** | r = 4.0 | 3.0 | 1000 |

The PRIMARY boundary was refined by bisection from the coarse bracket [2.0, 2.5] through
2.25 to [2.0, 2.125] — width 0.125, 5.9 % relative, satisfying both stopping criteria.

## 8. Final 50k ImageNet ID results (M = 20, 5 seeds)

Base: 81.068 % top-1, NLL 0.8482, ECE 0.0913 (T = 0.700).

| config | r | top-1 | top-5 | NLL | ECE | agreement | budget check on 50k |
|---|---|---|---|---|---|---|---|
| STRICT | 1.25 | 80.968 ± 0.022 | 95.211 | 0.8120 | 0.0725 | 0.9857 | Δ−0.100, LCB −0.153 vs −0.25 → **PASS** |
| PRIMARY | 2.0 | 80.830 ± 0.034 | 95.139 | 0.8045 | 0.0542 | 0.9736 | Δ−0.238, LCB −0.315 vs −0.50 → **PASS** |
| RELAXED | 3.0 | 80.598 ± 0.064 | 95.139 | 0.8173 | 0.0459 | 0.9612 | Δ−0.470, LCB −0.571 vs −1.00 → **PASS** |

All three re-validate at M = 20 on the untouched validation set, so no fallback to a smaller
scale was needed. Every configuration also **improves NLL and ECE over the base model**
(0.8045 vs 0.8482; 0.0542 vs 0.0913 at PRIMARY) — the ensemble is better calibrated than
the single network under the shared temperature, and increasingly so with scale.

Selection-pool estimates transfer conservatively to validation: the 8,192-image search
predicted −0.305 pp at PRIMARY and the 50k set delivered −0.238 pp.

## 9. Matched uncorrected ablation

Identical basis, coefficients, scale and seeds; only W2 differs
([`tables/table3_correction_ablation.md`](tables/table3_correction_ablation.md)).

| budget | r | P&C top-1 | uncorr. top-1 | P&C NLL | uncorr. NLL | median MSE ratio | p99 ratio |
|---|---|---|---|---|---|---|---|
| STRICT | 1.25 | 80.968 | 80.957 | 0.8120 | 0.8227 | 2.53× | 1.99× |
| PRIMARY | 2.0 | 80.830 | 80.619 | 0.8045 | 0.9048 | 4.37× | 2.43× |
| RELAXED | 3.0 | 80.598 | **79.065 ± 0.685** | 0.8173 | **1.4219** | 6.76× | 2.51× |

At RELAXED the uncorrected ensemble has lost 2.0 pp of top-1, 2.8 pp of top-5, its NLL has
risen 74 % above base and its ECE is 0.2646 against P&C's 0.0459. Its seed spread (±0.685)
is 11× P&C's — the uncorrected ensemble is not merely worse but unstable. Mean predictive
entropy tells the same story: 0.743 for P&C against 3.383 uncorrected, i.e. the uncorrected
members have largely stopped agreeing on anything.

## 10. OpenOOD results

| method | Near AUROC | Near FPR95 | Far AUROC | Far FPR95 |
|---|---|---|---|---|
| **P&C RELAXED (r = 3.0)** | **77.27 ± 0.14** | **66.94** | **88.06 ± 0.13** | **44.52** |
| **P&C PRIMARY (r = 2.0)** | **76.49 ± 0.09** | 67.65 | 87.89 ± 0.12 | 44.97 |
| P&C STRICT (r = 1.25) | 75.59 ± 0.09 | 69.41 | 87.38 ± 0.06 | 46.31 |
| uncorrected, r = 1.25 | 74.20 | 71.64 | 86.53 | 47.22 |
| uncorrected, r = 3.0 | 73.16 | 71.22 | 83.82 | 45.27 |
| uncorrected, r = 2.0 | 72.53 | 71.66 | 84.40 | 46.51 |
| MSP | 73.52 | 81.84 | 86.04 | 51.74 |
| ReAct + Energy | 69.21 | 84.23 | 85.61 | 53.90 |
| Energy | 62.39 | 93.16 | 78.96 | 85.29 |

P&C improves monotonically with scale on every aggregate. The matched uncorrected ensemble
does **not** — it peaks at the smallest scale and degrades, exactly as the ID frontier
predicts. At PRIMARY the corrected/uncorrected gap is **+3.96 Near AUROC**; the original
protocol's gap was +0.14, within seed noise.

Per dataset (PRIMARY, 5-seed mean AUROC, against MSP): SSB-hard 70.77 vs 68.93 (+1.84),
NINCO 82.21 vs 78.11 (+4.10), iNaturalist 89.69 vs 88.19 (+1.50), Textures 86.22 vs 85.06
(+1.16), OpenImage-O 87.77 vs 84.86 (+2.90). P&C is ahead on all five; the largest gain is
NINCO. FPR95 improves on all five as well, most sharply on NINCO (55.56 vs 77.27) and
OpenImage-O (44.43 vs 56.27).

## 11. Preservation–diversity trade-off

[`figures/preservation_frontier.png`](figures/preservation_frontier.png) shows the three
panels: corrected ID preservation against the budget lines; corrected vs uncorrected ID
accuracy (symlog, so both the crossover near r ≈ 1.75 and the r = 4 collapse are visible);
and Near AUROC at the three frozen points.

The trade-off is real but gentle over the tested range. Going from STRICT to RELAXED costs
0.37 pp of ID top-1 and buys 1.68 Near AUROC. Since OOD improves monotonically all the way
to the relaxed boundary, the binding constraint is the accuracy budget, not any internal
optimum — the experiment does not identify a scale at which OOD performance turns over
before ID preservation fails.

**Per §24 this observation does not move the primary budget.** PRIMARY remains 0.50 pp and
r = 2.0. That RELAXED scores better OOD is reported, not acted on.

## 12. Calibration / ridge diagnostics

Three effects are worth carrying into the manuscript:

1. **The optimal ridge grows with the perturbation.** λ = 1e-3 is fine below r ≈ 1 and
   loses 0.25 pp of preservation by r = 2. A protocol that fixes ridge while sweeping scale
   will mis-locate the boundary.
2. **Accuracy and calibration can be traded against each other by ridge.** λ = 1e4 buys
   the best top-1 at every large scale while inflating ECE up to 17.7× base. Accuracy alone
   is not a sufficient preservation criterion.
3. **The ensemble is better calibrated than the base network**, and more so at larger
   scale (ECE 0.0913 base → 0.0725 → 0.0542 → 0.0459). Correction plus averaging is doing
   real calibration work, not merely holding ground.

## 13. Runtime and memory

Construction after the reused (h, z₀) cache: **12 s per M = 20 ensemble** (0.6 s/member),
peak 2.16 GiB GPU. The whole 88-configuration search took ~17 min; the three final
five-seed ensembles plus their ablations ~4 min; OOD across three operating points, five
seeds and two variants ~35 min. Sustained throughput on this card is 131 img/s (12 % below
cold — thermal). No inference-time advantage is claimed: M members cost M tail evaluations.

## 14. Limitations and provenance

**Provenance.** This protocol is a follow-up, motivated by analysis of the earlier
`../imagenet_vit/` experiment — which had already inspected OOD performance at several
scales, including r = 1 and r = 2. **It is not a pristine preregistered OOD experiment and
no such claim is made.** What holds is narrower: the selection rule was written and
committed (`e25c8c9`) before the follow-up OOD ran, the frozen configurations were
committed (`74f6d56`) before it too, and those configurations were determined by ID data
alone. The original cleanly-selected result is preserved unchanged in `../imagenet_vit/`.

Other limitations:

- **Single target block, single token design, K = 20, n_cal = 32,768** — all fixed by
  design, none searched.
- **The boundary is budget-relative, not absolute.** "r = 2.0" means "the largest scale
  meeting a 0.50 pp top-1 budget on this checkpoint with this correction set", not a
  property of ViT-B/16 in general.
- **OOD improves monotonically to the relaxed boundary**, so the tested range does not
  contain an OOD optimum. A wider ID budget might score better still; that is a
  manuscript-level judgement about acceptable accuracy loss, not an empirical finding here.
- **Construction seeds are not training seeds.** All ensembles come from one checkpoint.
- The ridge grid tops out at 1e4, which is selected at the two smallest scales. Since those
  scales pass every budget regardless, the truncation does not affect any reported boundary.

## 15. Recommended manuscript treatment

1. **Replace the original operating point with the PRIMARY configuration** (r = 2.0,
   λ = 1000) as the headline ImageNet result, and say plainly why: the earlier r = 0.375
   was an artefact of fixing ridge during the scale sweep.
2. **Lead the mechanism argument with Table 3 and the frontier figure.** "The uncorrected
   ensemble loses 10.75 pp of ImageNet top-1 at r = 4 where P&C loses 0.96" is the clearest
   statement of what the correction buys, and it is exactly the claim the method makes.
3. **Report the boundary, not just the point** — passes through r = 2.0, fails by r = 2.125.
   Characterising the frontier is more informative than a single selected scale.
4. **Keep all three budgets in the paper.** They show the trade-off is smooth and that the
   conclusion is not an artefact of one threshold choice.
5. **Be explicit about the provenance caveat** (§14). The honest framing — a follow-up with
   an ID-only rule fixed before its own OOD evaluation, alongside a preserved clean original
   — is stronger than an overclaim a reviewer can puncture.
6. Retain the observation that ridge must scale with perturbation magnitude; it is a
   practical result other P&C applications will need.
