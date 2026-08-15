# ViT geometry follow-up: P&C vs Mahalanobis, SCOD, LLLA

Second diagnostic round on the frozen ImageNet ViT-B/16 P&C result. The previous
round proposed `PERTURBATION_SUBSPACE` as the bottleneck; this round treats that as a
hypothesis and tests it. Nothing here changes the frozen P&C headline (r=2, λ=1000,
K=20, M=20) or the Mahalanobis headline, and no OOD data was used to select any
hyperparameter.

## 1. Executive summary

This round tested the previous round's `PERTURBATION_SUBSPACE` verdict as a hypothesis. **It
does not survive.** The replacement explanation is `LOW_VARIANCE_MODE_MISALIGNMENT`, with a
separately-evidenced `BASE_CONFIDENCE_DOMINANCE` component.

Nothing here changes the frozen P&C headline (r=2, λ=1000, K=20, M=20), alters the
Mahalanobis headline, or uses OOD data to select any hyperparameter, rank, scope, layer,
prior, temperature or estimator.

**Is the Mahalanobis advantage caused by labels?** No. The previous round showed that
refitting on the ViT's own predicted labels matches ground truth (78.86 vs 78.82 Near). This
round adds that a *single unconditional* Gaussian already reaches 75.54 Near / 91.13 Far, so
most of the performance needs neither labels nor classes.

**Is it caused simply by 768 dimensions versus P&C's K=20?** No. A random 20-dimensional
view of the same CLS geometry gives Mahalanobis only **70.79 Near — below P&C's 76.43**. It
takes ~320 random dimensions for Mahalanobis to match P&C on Near. P&C's twenty directions
are better than twenty random ones; the deficit is not a dimension count.

**Which covariance modes carry the advantage?** The **low-variance** ones. Ordered by
decreasing ID variance and scored in bands of 96, Near AUROC rises monotonically 68.32 →
81.81 from the highest-variance band to the lowest, and the **lowest-variance 96 modes alone
(81.81) beat the full 768-mode detector (78.82)**. The high-variance modes actively dilute
the score.

**Does P&C's perturbation response cover those modes?** No — it covers the opposite ones.
Response energy across modes tracks ID variance at Spearman **+0.9925**, hence −0.9925
against the 1/λ weighting Mahalanobis applies and −0.822 against each mode's measured OOD
separation. P&C spends 26.7 % of its response energy on the top-96 modes (where Mahalanobis
is weakest) and 4.7 % on the bottom 96 (where it is strongest). This is what a *random*
low-rank weight perturbation must do — its response inherits the covariance's own shape,
which is precisely what Mahalanobis inverts — and it is why the rank sweep is flat: Near
AUROC 76.53 / 76.49 / 76.51 / 76.41 at K = 5 / 20 / 40 / 80, a 0.12 spread against a
0.11–0.18 seed standard deviation.

**Is P&C predictive entropy materially different from base entropy?** Yes, but thinly.
Against the same base model at the same temperature it gains **+1.90 Near / +1.32 Far**,
while sharing 88–94 % of its ranking (Spearman, computed within ID and within OOD
separately). The gain is real and reproducible; it is also small next to the 4.29 Near that
still separates P&C from Mahalanobis.

**Does P&C beat Mahalanobis at detecting actual ImageNet errors?** **Yes, decisively —
85.58 vs 78.00 AUROC.** The ranking inverts relative to OOD: Mahalanobis is the best OOD
detector tested and the *worst* of the confidence-based error detectors. Two methods, two
different quantities. But P&C does not beat the base model here either: temperature-scaled
base entropy leads at 86.81.

**How do memory-bounded SCOD variants compare?** The MC-Fisher route **failed its own
convergence check** (genuine across-seed Spearman ≈0.46 at q=8), so it was replaced by
upstream's exact categorical factorisation, made tractable by exploiting the gradient map's
linearity in the output residual. **SCOD-linear reaches 75.15 Near / 88.48 Far** — the best
Far AUROC of any method here except Mahalanobis, above P&C's 87.93. Enlarging the scope
*hurts*: **SCOD-FFN gets 72.02 / 84.63** at the same rank, uniformly worse on all five OOD
sets. SCOD-last-block was not run, and §11 states why rather than papering over it.

**Does ID-only temperature calibration materially strengthen LLLA?** No. It buys +1.59 Near
and costs −0.93 Far, and lands on **exactly** the temperature-scaled base model (74.54 vs
74.53 Near). That is the expected consequence of the posterior having collapsed onto the
MAP. P&C's advantage over LLLA is therefore not an artefact of P&C having been allowed a
calibration step that LLLA was denied.

## 2. Base confidence vs P&C ensemble uncertainty

The previous round argued that P&C's OOD signal is "mostly inherited base confidence" from
the ensemble's internals alone. Measuring base entropy directly changes that conclusion.

| score | Near AUROC | Far AUROC | Near FPR95 | Far FPR95 |
|---|---|---|---|---|
| MSP | 73.52 | 86.04 | 81.84 | 51.74 |
| base entropy, raw softmax | 72.95 | 87.42 | 85.93 | 55.66 |
| **base entropy, T = 0.7** | **74.53** | 86.42 | 75.74 | 49.23 |
| **P&C predictive entropy** | **76.43** | **87.74** | 67.63 | 45.73 |
| P&C expected member entropy | 76.55 | 87.85 | 67.60 | 45.74 |
| P&C mutual information | 73.97 | 84.30 | 68.30 | 42.51 |
| Mahalanobis | 78.82 | 92.55 | 66.36 | 30.23 |

`base_entropy_T` uses exactly the scalar temperature the frozen P&C classifier uses
(T = 0.7); the raw variant is reported alongside it and neither was chosen using OOD.

**P&C does change member confidence under shift, but modestly.** Against the fairest
control — the same base model at the same temperature — P&C gains **+1.90 Near** and
**+1.32 Far** AUROC, and improves Near FPR95 from 75.74 to 67.63. That is a real effect and
not a re-labelling of base entropy. It is also small relative to the 4.29 Near / 4.81 Far
that still separates P&C from Mahalanobis.

Rank correlations, computed separately inside ID and inside pooled OOD so that the
group-level split cannot inflate them:

| P&C score | vs base entropy (T), within ID | within pooled OOD |
|---|---|---|
| predictive entropy | +0.882 | +0.937 |
| expected member entropy | +0.888 | +0.944 |
| mutual information | +0.733 | +0.739 |

So roughly 88–94 % of P&C's ranking is shared with the base model's own entropy. The
ensemble adds a real but thin layer on top; the disagreement-only component (mutual
information) is both the least correlated with base entropy and the weakest detector.

One incidental but useful check: `base_entropy_raw` scores **72.95 Near / 87.42 Far**, which
is *identical* to the LLLA-Kron result from the previous round. That is an independent
confirmation that the fitted last-layer Laplace posterior collapsed exactly onto the MAP.

## 3. Does P&C rank K explain the Mahalanobis gap?

**No. Rank explains essentially nothing.**

Each K was given its own ID-only search over r ∈ {1.0, 1.5, 2.0, 2.5, 3.0, 4.0} ×
λ ∈ {100, 300, 1000, 3000, 10000} against the same primary preservation budget
(ΔAcc_ID ≥ −0.50 pp by paired-bootstrap LCB), coarse at M=10/seed 0 and confirmed at
M=10/seeds {0,10,42}. **All four ranks independently selected the same operating point,
r = 2.0 and λ = 1000** — which is also the frozen headline configuration. Final ensembles
are M=20 over five seeds.

| K | selected r | λ | realised r | ID top-1 | ΔAcc (pp) | ID NLL | Near AUROC | Far AUROC |
|---|---|---|---|---|---|---|---|---|
| 5 | 2.0 | 1000 | 2.0000 | 80.833 ± 0.031 | −0.235 | 0.8074 | 76.53 ± 0.15 | 87.82 ± 0.16 |
| **20** (headline) | 2.0 | 1000 | 2.0000 | 80.813 ± 0.030 | −0.255 | 0.8043 | **76.49 ± 0.11** | **87.93 ± 0.15** |
| 40 | 2.0 | 1000 | 2.0000 | 80.807 ± 0.055 | −0.261 | 0.8044 | 76.51 ± 0.18 | 87.93 ± 0.14 |
| 80 | 2.0 | 1000 | 2.0000 | 80.808 ± 0.041 | −0.260 | 0.8043 | 76.41 ± 0.14 | 87.93 ± 0.12 |

The entire Near range across a 16× change in rank is **0.12 AUROC** (76.41–76.53), smaller
than the seed standard deviation (0.11–0.18); Far is flat to 0.11. Mutual information
(73.73–73.97) and logit variance (68.36–69.08) are equally flat, and both *decrease* very
slightly with K. Against Mahalanobis at 78.82 / 92.55 the gap is untouched.

The diagnostic correlations do not move either:

| K | Spearman(Mahalanobis, hidden response) | Spearman(Mahalanobis, P&C entropy) |
|---|---|---|
| 5 | 0.6418 | 0.8790 |
| 20 | 0.6403 | 0.8823 |
| 40 | 0.6351 | 0.8817 |
| 80 | 0.6342 | 0.8824 |

**Therefore the previous round's `PERTURBATION_SUBSPACE` classification is withdrawn.** The
0.634 correlation it rested on is real, but it is not a rank artefact: it is identical at
K=5 and K=80. Something other than subspace dimensionality is limiting the response.

Two controls make this trustworthy. The bases are genuinely nested — one orthonormal K=80
basis per seed, smaller ranks are its first K rows (Gram–Schmidt/Cholesky is
lower-triangular, so row k depends only on draws 1..k), and the member coefficients are the
first K columns of one (M, 80) draw. And perturbation energy is matched, not merely
nominal: the *median* realised ‖ΔW₁‖_F/‖W₁‖_F is 2.0000 at every K. One caveat worth
stating: matching the median narrows the per-member dispersion as K grows (member-level
realised r spans [0.60, 3.09] at K=5 but [1.70, 2.31] at K=80), because ‖coefficients‖
concentrates. That is intrinsic to nesting at fixed median energy and cannot be removed
without unmatching the totals.

The K=20 arm reproduces the frozen headline to three decimals (76.49 ± 0.11 vs the frozen
76.49 ± 0.09), which validates the reimplementation. **K=80 is not proposed as a new P&C
configuration**; nothing here changes the headline.

## 4. Random-projection dimensionality control

Class-conditional and unconditional Mahalanobis fitted in a random K-dimensional projection
of the same CLS feature, on the same 32,768-image ID pool; 20 random orthonormal
projections per K, mean ± std.

| K | class-conditional Near | Far | unconditional Near | Far |
|---|---|---|---|---|
| 5 | 59.59 ± 1.15 | 66.23 ± 2.29 | 59.25 ± 1.19 | 66.12 ± 2.52 |
| 20 | 70.79 ± 0.62 | 81.51 ± 1.43 | 65.43 ± 1.15 | 75.59 ± 2.27 |
| 40 | 73.43 ± 0.38 | 86.07 ± 0.92 | 67.75 ± 0.82 | 79.80 ± 1.73 |
| 80 | 74.90 ± 0.18 | 88.72 ± 0.38 | 69.66 ± 0.54 | 82.91 ± 0.92 |
| 160 | 75.94 ± 0.15 | 90.34 ± 0.21 | 71.55 ± 0.32 | 85.75 ± 0.54 |
| 320 | 76.84 ± 0.08 | 91.22 ± 0.15 | 73.09 ± 0.25 | 87.89 ± 0.44 |
| 768 (full) | 78.82 | 92.55 | 75.54 | 91.13 |

**This rules out the simplest form of the dimensionality explanation.** A random 20-D view
of the CLS geometry reaches only **70.79 Near**, which is *below* P&C's 76.43 at K = 20.
Mahalanobis needs roughly 320 random dimensions before it merely matches P&C's Near AUROC.
So P&C at rank 20 is already extracting more Near-OOD signal than 20 random feature
directions contain — its directions are better than random — and "P&C only has 20
dimensions" is not, on its own, why it trails.

The curve does show that Mahalanobis's own advantage is genuinely spread across many
dimensions: it climbs monotonically all the way to 768 and never saturates.

## 5. Which covariance modes make Mahalanobis strong?

Eigendecomposition of the same regularised shared covariance the detector inverts
(λ₁ = 0.885, λ₇₆₈ = 1e-6, condition number 8.9e5), with each example's class fixed to the
one the headline detector selects. Modes sorted by decreasing ID variance, eight bands of 96,
each band scored **alone**:

| band (ID-variance rank) | λ range | Near AUROC | Far AUROC | share of ID d² |
|---|---|---|---|---|
| 1–96 | 0.885 → 0.252 | 68.32 | 83.64 | 11.56 % |
| 97–192 | 0.251 → 0.193 | 70.17 | 84.60 | 11.63 % |
| 193–288 | 0.193 → 0.155 | 72.30 | 86.32 | 12.01 % |
| 289–384 | 0.154 → 0.125 | 75.20 | 89.55 | 12.35 % |
| 385–480 | 0.125 → 0.100 | 77.65 | 91.23 | 12.70 % |
| 481–576 | 0.100 → 0.078 | 80.18 | 92.47 | 13.05 % |
| 577–672 | 0.078 → 0.058 | 81.63 | 92.18 | 13.40 % |
| **673–768** | 0.058 → 1e-6 | **81.81** | **92.44** | 13.30 % |

Monotone, with no exception, from the highest-variance band to the lowest. Cumulatively:

| subset | Near AUROC | Far AUROC |
|---|---|---|
| highest-variance 96 | 68.32 | 83.64 |
| highest-variance 192 | 69.38 | 84.36 |
| highest-variance 384 | 71.95 | 86.85 |
| **lowest-variance 96** | **81.81** | 92.44 |
| lowest-variance 192 | **82.02** | 92.67 |
| lowest-variance 384 | 81.33 | **93.07** |
| all 768 | 78.82 | 92.55 |

**Answer: yes, decisively.** Mahalanobis's advantage lives in the *low*-variance ID
directions, exactly where the inverse covariance puts large weight. The lowest-variance 96
modes alone (81.81 Near) beat the full 768-mode detector (78.82), and the lowest-variance
192 are better still (82.02) — the high-variance modes are actively diluting the score. The
d² mass is spread almost evenly across bands (11.6–13.4 %), so this is not a mass effect: it
is which directions carry ID-vs-OOD information.

## 6. Does P&C perturb those modes?

Exact final-CLS changes Δφ_m(x) = φ_m(x) − φ₀(x) under the frozen members (r=2, λ=1000,
K=20, M=20), on 8,192 ID and 5,000 per OOD set, projected onto the same eigenbasis;
E_j = E_{x,m}[⟨q_j, Δφ_m⟩²].

| relationship | Spearman |
|---|---|
| **E_j vs λ_j** | **+0.9925** |
| **E_j vs 1/λ_j** | **−0.9925** |
| E_j vs Mahalanobis mode separation | −0.8219 |
| E_j vs per-mode Mahalanobis AUROC | −0.6693 |
| Mahalanobis separation vs 1/λ_j | +0.8392 |

| band | mean λ | P&C response-energy share | whitened share | Mahalanobis mode AUROC |
|---|---|---|---|---|
| 1–96 | 0.330 | **26.70 %** | 10.61 % | 61.80 |
| 97–192 | 0.220 | 16.32 % | 10.21 % | 61.40 |
| 193–288 | 0.173 | 13.78 % | 10.96 % | 61.78 |
| 289–384 | 0.139 | 12.04 % | 11.88 % | 62.70 |
| 385–480 | 0.112 | 10.29 % | 12.61 % | 63.43 |
| 481–576 | 0.089 | 8.86 % | 13.70 % | 64.53 |
| 577–672 | 0.068 | 7.32 % | 14.82 % | 65.37 |
| **673–768** | 0.041 | **4.69 %** | 15.21 % | 64.67 |

**This is the direct geometric explanation.** P&C's perturbation response is almost
perfectly proportional to the ID variance of each direction (Spearman +0.9925 against λ_j),
which is exactly the ordering Mahalanobis inverts. P&C spends 26.7 % of its response energy
on the top-96 highest-variance modes — the band where Mahalanobis is *weakest* (61.80) — and
only 4.69 % on the lowest-variance 96, where the standalone detector reaches 81.81 Near.

Mechanistically this is what a *random* low-rank perturbation of W₁ must do: the induced
change in the final representation is filtered through the same activation statistics that
generate the covariance, so its energy inherits the covariance's own shape. Mahalanobis
deliberately undoes that shape. The two are, to a very good approximation, opposites.

Mean ‖Δφ‖² also rises on OOD (29.8 ID vs 38.8–64.9 across the OOD sets), so P&C does
respond more strongly out of distribution — but in the wrong subspace to compete with an
inverse-covariance detector.

## 7. High-Mahalanobis / low-P&C examples

Sets defined by ID-referenced empirical percentiles, so the two scales are comparable. At
the strict 95/50 threshold the P&C-only set is empty, so the 90/60 relaxed thresholds carry
the analysis (both are reported in `metrics/disagreement.json`).

| | geometry-only (M ≥ 90th, P&C ≤ 60th) | P&C-only (P&C ≥ 90th, M ≤ 60th) | all data |
|---|---|---|---|
| count | **282** | **16** | 135,908 |
| composition | 47 ID, 72 SSB-hard, 21 NINCO, 57 iNaturalist, 20 Textures, 65 OpenImage-O | 12 ID, 4 SSB-hard | — |
| **member top-1 agreement** | **1.000** | 0.900 | 1.000 |
| **mean pairwise KL** | **6.02** | 38.64 | 18.77 |
| mutual information | 0.0178 | 0.0573 | 0.0534 |
| MSP | 0.889 | 0.253 | 0.679 |
| base entropy (T) | 0.067 | 1.666 | 0.434 |
| Mahalanobis d² | 2233 | 857 | 1586 |
| P&C predictive entropy | 0.276 | 1.838 | 0.859 |
| base top-1 correct (ID subset) | 83.0 % | **41.7 %** | 81.1 % |

**Yes — geometry-only points are exactly the predicted case.** On all 282 of them every
member agrees on the top-1 (agreement 1.000), pairwise KL is a third of the background
(6.02 vs 18.77) and mutual information is a third (0.0178 vs 0.0534). These are inputs that
sit far from the training manifold and on which *all P&C members extrapolate in essentially
the same way*, so the ensemble has nothing to disagree about. They are also confidently
predicted by the base model (MSP 0.889 vs 0.679).

The reverse set barely exists: 16 examples, three quarters of them ID, with base top-1
accuracy 41.7 % against 81.1 % overall. P&C's distinctive high-uncertainty region is
therefore not an OOD region at all — it is *hard, misclassified in-distribution inputs*.
That directly anticipates §8.

The asymmetry (282 vs 16) says P&C's high-entropy set is close to a subset of Mahalanobis's
high-distance set on this benchmark, not a complement to it.

### Conditional performance

OOD examples binned by Mahalanobis percentile, then scored by P&C entropy — and the reverse.

OOD binned by **Mahalanobis** percentile, scored by **P&C entropy**:

| bin (percentile) | n OOD | Near AUROC | Far AUROC |
|---|---|---|---|
| 0–50 | 11,448 | 40.43 | 34.35 |
| 50–75 | 13,672 | 61.33 | 56.18 |
| 75–90 | 16,941 | 79.40 | 77.46 |
| 90–95 | 9,434 | 88.61 | 86.33 |
| 95–100 | 34,413 | 93.06 | 94.35 |

OOD binned by **P&C entropy** percentile, scored by **Mahalanobis**:

| bin (percentile) | n OOD | Near AUROC | Far AUROC |
|---|---|---|---|
| 0–50 | 11,787 | 43.77 | 56.55 |
| 50–75 | 16,739 | 68.54 | 80.69 |
| 75–90 | 21,716 | 84.88 | 91.36 |
| 90–95 | 13,787 | 93.79 | 95.95 |
| 95–100 | 21,879 | 97.11 | 97.99 |

The two methods **fail on the same region, not complementary ones**: where Mahalanobis is
weak, P&C is weak too (40.43 Near in the bottom bin), and vice versa (43.77). Mahalanobis
retains more resolution in every matched bin. This is descriptive only and no combined
detector is built from it.

## 8. OOD detection vs ID error detection

Positive class: the frozen base model's top-1 on the untouched 50k ImageNet validation set is
wrong (9,466 of 50,000; base top-1 81.068 %).

| score | ID error AUROC | ID error AUPR | Near AUROC (for contrast) |
|---|---|---|---|
| **base entropy (T = 0.7)** | **86.81** | **60.88** | 74.53 |
| MSP | 85.61 | 60.12 | 73.52 |
| P&C expected member entropy | 85.78 | 57.92 | 76.55 |
| **P&C predictive entropy** | 85.58 | 57.57 | **76.43** |
| P&C mean pairwise KL | 85.30 | 56.93 | — |
| base entropy (raw) = LLLA-Kron | 82.40 | 55.16 | 72.95 |
| P&C mutual information | 80.92 | 49.53 | 73.97 |
| **Mahalanobis** | **78.00** | **46.94** | **78.82** |
| P&C logit variance | 74.03 | 40.81 | 68.63 |
| P&C member agreement | 73.05 | 44.27 | — |
| Mahalanobis (unconditional) | 71.22 | 38.62 | 75.54 |

**The ranking inverts.** Mahalanobis is the best OOD detector here and one of the *worst*
error detectors (78.00, below every confidence-based score). The class-conditional Gaussian
knows where the feature lies relative to the training manifold; it does not know whether the
classifier got this particular example right.

Two honest qualifications. First, P&C does **not** beat the base model at error detection
either: base entropy at the same temperature scores 86.81 against P&C's 85.58, and MSP
scores 85.61. Second, the base-entropy row benefits from the scalar temperature that was
fitted for the P&C ensemble; applying it to the base model is still an ID-only choice, so
the comparison is fair, but the base model in the frozen headline table is reported
untempered.

So on this backbone P&C is not the more task-relevant uncertainty measure either — the base
model's own confidence, once temperature-scaled, is the strongest error detector tested.

## 9. Layerwise representation geometry

An independent class-conditional shared-covariance Mahalanobis detector fitted at each depth
on the *same* frozen 32,768-image ID training pool (never on the evaluation split), with the
identical `cov + 1e-6 I` convention everywhere. No layer was selected using OOD; all are
reported.

| layer | Near AUROC | Far AUROC | Near FPR95 | Far FPR95 | unconditional Near |
|---|---|---|---|---|---|
| block 8 | 58.43 | 69.83 | 86.79 | 66.76 | 53.04 |
| block 9 | 68.59 | 81.49 | 78.32 | 56.71 | 52.18 |
| block 10 | 69.80 | 78.47 | 76.03 | 56.31 | 24.91 |
| block 11 | 77.44 | 92.40 | 70.86 | 29.18 | 40.82 |
| **final LayerNorm** | **78.82** | **92.55** | **66.36** | 30.23 | 75.54 |

**Near AUROC rises monotonically with depth and is strongest at the final LayerNorm.** By
the criterion set out in the spec, that points away from target-layer scope and towards
perturbation-response alignment: the geometry Mahalanobis exploits is *not* already sitting
in an earlier layer that P&C fails to reach. It is built up gradually and peaks exactly at
the representation the final block produces — the block P&C perturbs.

Block 11's own output already reaches 77.44 Near / 92.40 Far, within 1.4 / 0.15 of the final
LayerNorm. So P&C is operating on the right representation. What differs is *which
directions of it* the perturbation excites (§6), not *where* it acts.

The unconditional column behaves very differently: it is non-monotone and collapses at
block 10 (24.91, i.e. strongly anti-correlated), then recovers only after the final
LayerNorm (75.54). A single global Gaussian is evidently sensitive to the un-normalised
scale of the intermediate residual stream; class-conditioning absorbs that. This is a
property of the representation, not of the detector, and it is reported as measured.

### MC-label convergence (spec §14) — the check that failed

SCOD-linear, 4,096 ID calibration examples, 4,096 held-out ID evaluation examples, 3 MC
seeds per q, k = 30, T = 184. ID data only. The rule was "smallest q with Spearman ≥ 0.99
against q = 8".

| q | Spearman vs q=8 (mean) | (min) | max principal angle (deg) | across-seed score CV |
|---|---|---|---|---|
| 1 | 0.5213 | 0.5109 | 89.75 | 0.3013 |
| 2 | 0.5439 | 0.4562 | 89.08 | 0.4077 |
| 4 | 0.5689 | 0.4323 | 89.46 | 0.4566 |
| 8 | 0.6415 | 0.4619 | 59.98 | 0.3917 |

**No q reaches 0.99, and the diagnosis is worse than the table suggests.** The q = 8 mean of
0.6415 includes seed 0 compared against *itself* (a 1.0); the genuine across-seed agreement
at q = 8 is (3 × 0.6415 − 1) / 2 ≈ **0.46**. Two runs of the same estimator, differing only
in the MC seed, rank the same 4,096 ID examples at Spearman 0.46, with a per-example score
CV of 0.39 and recovered eigenbases 60–90° apart.

This is the per-example noise Gate 3 predicted (relative error 1.12 at q = 1, still 0.36 at
q = 8): one label sample is a poor estimate of a 1000-class expectation, and the score is a
per-example quantity. Extrapolating the 1/√q rate, Spearman 0.99 would need q of order
several hundred, i.e. ~50× the cost of q = 8.

Spec §10 says not to proceed to the large sketch if these checks fail, and §12 says not to
weaken SCOD to make it run. **The MC path was therefore abandoned rather than reported**, and
replaced by the exact factorisation below — which costs the same as q = 1 and has no
sampling noise at all. `metrics/scod_qsweep.json` retains the full failed sweep.

### Gates 4–6 — the exact factorisation that replaced MC

Because the §14 sweep (below) showed the MC score is noise-dominated at every feasible q,
the Fisher factor was replaced by upstream's **exact** one,
L_F = (I − p1ᵀ)diag(√p) — `apply_sqrt_F(exact=True)`. Upstream cannot use it at this scale
(1000 backward passes per image; L is n_params × 1000, 3 GB per example for the head alone),
but for these scopes the gradient map is **linear in the output residual**, g(s) = A_x s, so
L = A L_F and every sketch quantity factors through A without forming L:

    Om L  = (Om A) L_F                                    T applications of A^T
    Y    += (1/M) A [ L_F (Om L)^T ]                      k applications of A
    W    += (1/M) [ A ( L_F (Psi L)^T ) ]^T               l applications of A

Per example that is 2T applications of A or Aᵀ — the same order as the q=1 MC update, but
exact. Right/left multiplication by L_F is O(T·C): X L_F = (X − (Xp)1ᵀ)⊙√p and
L_F Z = √p⊙Z − p·Σ(√p⊙Z). Norms use tr(A F Aᵀ) = Σ_c p_c‖Ae_c‖² − ‖Ap‖², which for the head
collapses to (‖φ‖²+1)(1−‖p‖²) and for the FFN decomposes block by block.

Three further gates check the derivation on the real ViT tail:

| gate | linear | ffn |
|---|---|---|
| adjointness ⟨Aq, v⟩ = ⟨q, Aᵀv⟩ | 1.27e-06 | 2.67e-06 |
| columns of A vs autograd (A e_c = ∇_w logits_c) | **0.00e+00** | 1.06e-07 |
| tr(A F Aᵀ) vs explicit 1000-class sum | 2.07e-06 | 4.89e-06 |

All at float32 round-off. The adjointness gate caught a real error during development: the
forward chain applies J_LNᵀ, so its adjoint applies J_LN, and for an affine LayerNorm those
are *not* the same matrix — the diag(γ/σ) sits on opposite sides of the symmetric core.

## 10. SCOD categorical implementation validation

The sketch (`RandomSymSketch`), the sketch operator and the `posterior_pred` projection are
vendored from `nn_ood/sketching.py` at commit `6a56973` essentially verbatim, and the SCOD
uncertainty definition is unchanged. Four adaptations were made, all permitted:

1. **Categorical MC Fisher.** Upstream's exact `CategoricalLogit.apply_sqrt_F` yields a
   k-column factor, so the weight Jacobian has k rows — 1000 backward passes per image on
   this head. Instead y ~ p(·|x) is sampled and g = ∇_w log p(y|x) used, with E_y[g gᵀ] =
   F(x), so q columns suffice. The SCOD supplement explicitly permits this.
2. **Restricted parameter scopes** — only the named block enters the sketch.
3. **Analytic per-example gradients** for the `linear` and `ffn` scopes, closed-form from
   the cached CLS residual (including the exact LayerNorm VJP and the erf-GELU derivative),
   which removes autograd and the per-example backward loop entirely.
4. **Blocked updates** — columns are accumulated in blocks and passed to one
   `low_rank_update`, algebraically identical to feeding them one at a time.

One deliberate departure from the paper's CIFAR SCOD-LL config is recorded: it uses
`sketch_type: 'srft'`, whose forward is an inverse DCT of length N per call. That cost is
O(N log N) *independent of how many columns are passed*, so with N ~ 10⁶ and ~10⁴ blocks it
dominates everything else. The **Gaussian** operator — upstream's own `base_config` default
(`sketch_type: 'random'`) — is used instead. It is pure GEMM, and costs a dense Ω and Ψ
(2·T·N floats rather than T·N). That memory is measured and reported in §13.

### Gate 1 — synthetic, exact enumeration vs MC

A 6-class, 54-parameter softmax classifier where the exact categorical Fisher can be
enumerated (256 training points, leading rank 5):

| q | Frobenius rel. error | max principal angle (deg) | SCOD score Spearman vs exact |
|---|---|---|---|
| 1 | 0.4742 | 78.15 | 0.9896 |
| 2 | 0.3213 | 71.34 | 0.9880 |
| 4 | 0.2517 | 57.81 | 0.9859 |
| 8 | 0.1794 | 53.70 | 0.9905 |
| 32 | 0.0846 | 38.25 | 0.9946 |

The Frobenius error falls as ~1/√q, as an unbiased MC estimator should. The principal
angles are large at every q because *this test problem's* leading eigenvalues are nearly
degenerate (0.143, 0.152, 0.162, 0.180, 0.206), so the leading-5 subspace is poorly
determined even for the exact matrix; the angle column is therefore uninformative here and
is reported rather than interpreted. What matters — the SCOD score itself — already
correlates ≥ 0.986 with the exact-Fisher score at q = 1, because the score is dominated by
the residual norm rather than by fine detail of the Fisher.

### Gate 2 — analytic vs autograd gradients on the real ViT tail

| scope | columns checked | max relative error | verdict |
|---|---|---|---|
| linear | 8 | 1.82e-06 | **OK** |
| ffn | 8 | 1.87e-06 | **OK** |

Float32 round-off. The analytic path is correct, including the transposition between the
code layout (`y = h W₁ + b₁`) and `nn.Linear.weight`.

### Gate 3 — the ViT 1000-class head, analytic vs MC

For a linear head, J_wᵀ u = [u ⊗ φ; u], so the exact Fisher-weighted norm has a closed
form with no sketch and no autograd:

  ‖L‖²_F = tr(J_wᵀ F J_w) = (‖φ‖² + 1)(1 − ‖p‖²).

Measured on 512 ImageNet examples, MC estimate vs that identity (3 seeds):

| q | pooled rel. error | per-example rel. error | per-example Spearman |
|---|---|---|---|
| 1 | 0.02784 | 1.1242 | 0.6701 |
| 2 | 0.00102 | 0.7939 | 0.6451 |
| 4 | 0.00871 | 0.5487 | 0.6934 |
| 8 | 0.00204 | 0.3629 | 0.7647 |
| 32 | 0.00246 | 0.1860 | 0.8927 |

The estimator is unbiased in aggregate (pooled error ≤ 3 % at q = 1, ≲ 0.3 % beyond), but
**per-example noise at low q is large** — a single sample estimating a 1000-class
expectation. This is the quantity that matters for a per-example detector, so it is what the
q sweep in §14 has to settle, and it is why the synthetic gate's optimism at q = 1 should
not be taken at face value.

All gates passed, so the large sketches were allowed to proceed.

## 11. SCOD-linear / FFN / last-block results

Both scopes that were run used the **exact** categorical Fisher (§10), the paper's own
CIFAR SCOD-LL sketch setting (k = 30, T = 184), the full 32,768-image ID calibration pool
matching P&C and Mahalanobis, and **no fallback** — neither needed k = 12 / T = 76.

| | SCOD-linear | SCOD-FFN | SCOD-last-block |
|---|---|---|---|
| parameters in scope | 769,000 | 5,491,432 | 7,858,408 |
| requested / actual k | 30 / **30** | 30 / **30** | — |
| requested / actual T | 184 / **184** | 184 / **184** | — |
| sketch k, l | 61, 123 | 61, 123 | — |
| recovered rank | 122 | 122 | — |
| n_eigs used for scoring | 30 | 30 | — |
| calibration N | 32,768 | 32,768 | — |
| fallback reason | none | none | — |
| **Near AUROC** | **75.15** | 72.02 | NA |
| **Far AUROC** | **88.48** | 84.63 | NA |
| Near FPR95 | 73.18 | 75.41 | NA |
| Far FPR95 | 43.37 | 46.38 | NA |
| **ID error AUROC** | **85.61** | 82.57 | NA |
| ID error AUPR | 58.75 | 52.79 | NA |
| construction time | 164 s + 1 s basis | 968 s + 9 s basis | — |
| evaluation time (135,908) | 96 s | 523 s | — |
| total | 4.4 min | 25.1 min | — |
| peak RSS | 3.43 GiB | **17.62 GiB** | — |
| peak VRAM | 0.83 GiB | 0.83 GiB | — |
| persistent storage | 358 MiB | 2,556 MiB | — |

Per-dataset AUROC:

| scope | SSB-hard | NINCO | iNaturalist | Textures | OpenImage-O |
|---|---|---|---|---|---|
| SCOD-linear | 69.26 | 81.03 | 90.21 | 87.33 | 87.91 |
| SCOD-FFN | 65.89 | 78.16 | 84.84 | 83.73 | 85.31 |

### How performance changes with scope

**It gets worse.** Moving from the head to the final FFN costs 3.13 Near, 3.85 Far and
3.04 ID-error AUROC, and is uniformly worse on all five OOD sets. The scope was not chosen
using OOD — both were run and both are reported.

The mechanism is visible in the spectra. At the head the top eigenvalues are ≈0.105–0.126;
at the FFN they are ≈0.215–0.434. The Fisher mass is larger and more concentrated, but it
lives in a 7.1× bigger space, so a fixed rank-30 basis explains a smaller fraction of it and
the `posterior_pred` residual — which is the score — becomes correspondingly less
discriminative.

**The honest caveat:** rank is held at k = 30 while the parameter count grows 7.1×, so this
comparison confounds *scope* with *rank adequacy*. k = 30 is the paper's own choice for a
last-layer scope, so applying it unchanged to a larger scope is the faithful reading of the
protocol — but a larger scope may simply need a larger k, and that was not tested. What can
be said is narrow and safe: **at the sketch rank the SCOD paper uses, enlarging the
parameter scope from the head to the final FFN makes the detector worse on this backbone.**

### Calibration size (spec §15)

Runtime was never the binding constraint for the exact estimator, so the full 32,768-image
pool — matching P&C and Mahalanobis — was used for both scopes. The convergence study is
reported for completeness, never used to pick N. SCOD-linear, k = 30, T = 184, ID data only,
scored on 4,096 held-out ID examples:

| n_cal | fit time | score Spearman vs previous n | max principal angle vs previous |
|---|---|---|---|
| 4,096 | 21 s | — | — |
| 8,192 | 42 s | **0.9999** | 87.6° |
| 16,384 | 82 s | **0.9999** | 84.7° |
| 32,768 | 163 s | **0.9999** | 80.4° |

**The score is converged by 4,096 examples.** The same pattern as the synthetic gate appears
again: the leading eigen*basis* rotates substantially between fits (80–88°) while the
*score* is invariant to four decimal places, because the spectrum is near-degenerate and the
`posterior_pred` residual depends on the subspace, not on the individual directions
spanning it.

This is also the sharpest available contrast with the rejected MC estimator. Changing the
calibration set size by 8× moves the exact score by 0.0001 in Spearman; changing only the MC
seed moved the MC score by 0.54.

### SCOD in context

SCOD-linear is a genuinely competitive baseline. Its **Far AUROC of 88.48 is the best of any
method in this round except Mahalanobis**, above P&C (87.93), base entropy at T (86.42) and
both LLLA variants. On Near it sits at 75.15 — above base entropy at T (74.53) and MSP
(73.52), below P&C (76.49) and Mahalanobis (78.82).

Its ID-error AUROC of 85.6128 lands within 0.0011 of MSP's 85.6139. That is a coincidence,
not an identity: the two scores correlate at Spearman 0.911, and SCOD-linear correlates 0.938
with temperature-scaled base entropy. The closed-form ‖L‖²_F = (‖φ‖²+1)(1−‖p‖²) makes the
relationship unsurprising — the head's Fisher norm is a confidence-like quantity modulated by
feature norm — but the scores are distinct.

### SCOD-last-block: not run

`LastBlockScope` and the `fitlast` stage are implemented in `scod_ll.py` / `scod_run.py`,
but no last-block number was produced, for two independent reasons:

1. **The exact route does not scale to it.** For the head and FFN the gradient map is linear
   in the output residual with a closed-form adjoint, so Aᵀω costs one small GEMM per sketch
   row. Self-attention has no such closed form here, so Aᵀω becomes a JVP — 184 forward-mode
   passes per example, ≈ 9 h for the calibration pass alone.
2. **The MC route is the one rejected in §10.** Producing a last-block number by the
   estimator that failed its own convergence check, after refusing it for the two smaller
   scopes, would be incoherent.

There is also now a *measured* memory obstacle. SCOD-FFN peaked at **17.62 GiB against the
18 GiB guard** with a 7.53 GiB sketch — roughly 10 GiB of transient overhead from the
operator splits and the (N, 2k) QR. Last-block's sketch is 10.77 GiB at k = 30, so the same
overhead pattern would put it near 25 GiB, above both the guard and the machine's 23 GiB.
It would therefore require the k = 12 / T = 76 fallback, making it non-comparable to the two
scopes reported here at k = 30.

Full-network SCOD (118.68 GiB sketch) was not attempted.

## 12. LLLA-Kron and temperature-calibrated LLLA

The completed LLLA-Kron result is preserved: the prior precision was **not** refitted. A
deterministic re-fit reproduced the previous round's ID-only selection exactly (grid
auto-extended to 1e10, optimum 1e9, ID selection NLL 0.3234), and the prior was then held
fixed while a single scalar probability temperature was fitted on the separate 8,192-image
temperature pool — the same pool and the same ID-NLL criterion P&C uses. Prior and
temperature were never tuned jointly, and no OOD data entered either.

Fitted **T = 0.7063** (temperature-pool NLL 0.3123 → 0.1936).

| variant | ID top-1 | ID NLL | ID ECE | Near AUROC | Far AUROC |
|---|---|---|---|---|---|
| LLLA-Kron | 81.068 | 0.8384 | 0.0560 | 72.95 | **87.42** |
| LLLA-Kron + Temp | 81.068 | 0.8438 | 0.0899 | **74.54** | 86.49 |
| *base entropy at T = 0.7 (for reference)* | *81.068* | — | — | *74.53* | *86.42* |
| *P&C (K=20, frozen)* | *80.813* | *0.8043* | *0.0534* | *76.49* | *87.93* |

**Answer: no, ID-only temperature calibration does not materially strengthen LLLA.** It buys
+1.59 Near and costs −0.93 Far, and the calibrated result lands on **exactly** the
temperature-scaled base model (74.54 vs 74.53 Near, 86.49 vs 86.42 Far). That is the
expected consequence of the posterior having collapsed onto the MAP: once the predictive is
the base model's, calibrating it is calibrating the base model. P&C still leads both
variants on Near and Far, and by a wide margin on ID NLL (0.8043 vs 0.8438).

So the P&C-over-LLLA gap is **not** an artefact of P&C having been allowed a calibration
step that LLLA was denied.

One honest observation about the protocol both methods share: the temperature pool is drawn
from ImageNet *train*, where the model is 93.8 % accurate, while evaluation is on val at
81.1 %. A temperature fitted on the easier pool over-sharpens on val — which is why
LLLA+Temp's val NLL (0.8438) and ECE (0.0899) are *worse* than the uncalibrated version
even though the pool NLL improved. This affects P&C identically, so the comparison is
matched, but it is a property of the frozen split protocol worth recording.

## 13. Timing and memory

Machine: NVIDIA TITAN X (Pascal, 12 GB, cc 6.1), 23 GiB system RAM, operational RSS ceiling
enforced at **18 GiB**. Persistent SCOD sketches are kept on the CPU per the memory guard;
only the model and the current batch live on the GPU.

| stage | wall | peak RSS | peak VRAM |
|---|---|---|---|
| base-entropy control (6 sets) | 4 s | — | — |
| disagreement pass (6 sets × 20 members, incl. pairwise KL) | 19 s | — | — |
| random-projection control (2 × 6 K × 20 projections) | 6.1 min | — | — |
| covariance eigenspectrum (768 per-mode AUROCs) | ~1 min | — | — |
| spectral alignment (33,192 examples × 20 members) | 5 s | — | — |
| K sweep — ID-only search (4 ranks, coarse + confirm) | ~28 min | — | — |
| K sweep — final ensembles (4 ranks × 5 seeds × M=20) | ~17 min | — | — |
| LLLA-Kron refit + temperature + evaluation | ~4 min | 1.3 GiB | 4.42 GiB |
| training-pool layer cache (32,768 images, 140 row groups) | 5.7 min | — | 0.46 GiB |
| layerwise Mahalanobis (5 depths × 2 variants) | ~1 min | — | — |

Two engineering notes that materially changed feasibility:

**The layer cache.** The previous round's attempt stalled indefinitely because
`ImageNetShards.load` sorts indices only *within* a 16-row batch, while `correction_rows.npy`
is ordered by class — so consecutive rows hit different parquet row groups and the
single-entry cache re-read a whole row group per image. Locating every row once and grouping
by (shard, row group) turns 32,768 scattered reads into **140 row-group reads** (234 images
per read) and the pass completes in 5.7 min at 96 img/s.

**The SCOD block size.** `A_apply` writes an (n_params, T) result whose size does not depend
on the example block — 0.55 GB for the head, 4 GB for the FFN — so at block=16 that write
dominated. Raising the block to 128 took SCOD-linear from 36 to **199 examples/s**, a 5.5×
speedup at identical output.

SCOD costs are in the table in §11. For reference, the sketch memory is 2·T·N floats with the
Gaussian operator (Ω, Ψ, Y and W all dense):

| scope | parameters | k=30, T=184 | k=12, T=76 |
|---|---|---|---|
| SCOD-linear | 769,000 | 1.05 GiB | 0.44 GiB |
| SCOD-FFN | 5,491,432 | 7.53 GiB | 3.11 GiB |
| SCOD-last-block | 7,858,408 | 10.77 GiB | 4.45 GiB |
| full network | 86,567,656 | 118.68 GiB | 49.02 GiB |

No fallback to k=12/T=76 was needed for either scope that was run: both fit at the paper's
own k=30, T=184 within the guard, and the measured peaks are reported in §11.

## 14. Revised explanation of the Mahalanobis gap

**Primary classification: `LOW_VARIANCE_MODE_MISALIGNMENT`.**
**Secondary, separately evidenced: `BASE_CONFIDENCE_DOMINANCE`.**

The previous round's `PERTURBATION_SUBSPACE` verdict is **withdrawn**. Four results rule
out each of the simpler explanations, and one identifies the mechanism positively.

*Ruled out — `LOW_RANK_LIMIT`.* The rank sweep is flat: Near AUROC 76.53 / 76.49 / 76.51 /
76.41 at K = 5 / 20 / 40 / 80, a 0.12 spread against a 0.11–0.18 seed standard deviation,
with all four ranks independently selecting r = 2.0, λ = 1000 and matched realised
perturbation. A 16× change in subspace dimension changes nothing (§3). The correlation the
previous verdict rested on — 0.634 between Mahalanobis and the hidden perturbation response —
is *identical* at K = 5 and K = 80, so it was never a rank artefact.

*Ruled out — dimensionality as such.* A random 20-dimensional view of the CLS geometry gives
Mahalanobis only 70.79 Near, **below** P&C's 76.43 at K = 20 (§4). P&C's twenty directions
are already better than twenty random ones; it takes ~320 random dimensions for Mahalanobis
to merely match P&C on Near.

*Ruled out — `TARGET_LAYER_SCOPE`.* Layerwise Mahalanobis rises monotonically with depth and
peaks at the final LayerNorm (58.43 → 68.59 → 69.80 → 77.44 → 78.82 Near), with block 11 —
the block P&C perturbs — already within 1.4 points of the peak (§9). The geometry is not
hiding in a layer P&C cannot reach.

*The mechanism.* Mahalanobis's discriminative power is concentrated in the **low-variance**
ID directions that the inverse covariance up-weights: the lowest-variance 96 of 768 modes
score 81.81 Near **on their own**, beating the full 768-mode detector (78.82), while the
highest-variance 96 reach only 68.32 (§5). P&C's perturbation response does the opposite. Its
energy across modes tracks the ID variance at Spearman **+0.9925**, hence **−0.9925** against
the 1/λ weighting Mahalanobis applies and −0.822 against each mode's actual OOD separation
(§6). Concretely, P&C spends 26.7 % of its response energy on the top-96 modes — where
Mahalanobis is weakest — and 4.7 % on the bottom 96, where it is strongest.

This is not a tuning failure; it is what a *random* low-rank perturbation of W₁ has to do.
The induced change in the final representation is filtered through the same activation
statistics that generate the covariance, so its energy inherits the covariance's shape.
Mahalanobis exists to invert that shape. Increasing K samples more directions from the same
badly-aligned distribution, which is exactly why the sweep is flat.

*Separately:* most of P&C's absolute detection level is inherited from the base model.
Against base entropy at the same temperature, P&C gains +1.90 Near / +1.32 Far, and shares
88–94 % of its ranking (§2). The gain is real and reproducible, but it is thin.

The two findings are consistent: P&C's uncertainty is mostly recalibrated base confidence,
plus a small ensemble-disagreement term whose geometric content is aimed at the wrong part
of the spectrum.

## 15. Implications for P&C

1. **P&C and feature-density methods measure different things, and the paper should say so.**
   §8 is the cleanest evidence: on ImageNet misclassification detection the ranking inverts —
   Mahalanobis is the *worst* confidence-based score (78.00) while it is the best OOD score
   (78.82 Near / 92.55 Far). Positioning P&C as a better detector of the same quantity is not
   supportable; positioning the two as complementary instruments is.
2. **But P&C should not be claimed as the better error detector either.** Temperature-scaled
   base entropy beats it (86.81 vs 85.58), as does MSP (85.61). What P&C *does* own in this
   comparison is predictive-distribution quality: ID NLL 0.8043 against 0.8482 for the base
   model and 0.8438 for temperature-calibrated LLLA.
3. **Rank is not a lever.** K = 5 performs as well as K = 80 under a matched preservation
   budget. Anyone tuning K for OOD gain on this backbone is tuning noise.
4. **The disagreement structure is one-sided.** 282 examples are high-Mahalanobis/low-P&C
   against 16 the other way, and on all 282 every member agrees on the top-1 (§7). Where the
   geometry says "unsupported", the perturbed members extrapolate *identically*, so the
   ensemble has nothing to disagree about. That is the failure mode in one sentence.

### A conditional proposal, not implemented here

Spec §22's two conditions both hold: K = 80 fails to close the gap, and the spectral analysis
shows P&C strongly under-exciting the low-variance modes that carry Mahalanobis's separation.
So, as a **proposal only**:

> Replace the isotropic random basis for ΔW₁ with one whitened by the ID activation
> statistics, so that the induced Δφ has energy roughly *flat* across covariance modes rather
> than proportional to λ_j. Concretely: estimate the second-moment matrix of the post-GELU
> activations on the ID pool, and draw the perturbation basis in the whitened coordinates, or
> equivalently precondition the drawn directions by Σ_y^{-1/2}. The ID-only preservation
> budget and the paired-bootstrap rule would be unchanged; only the proposal distribution for
> the basis changes.

This is **not implemented and not evaluated in this branch**, deliberately. It is a new P&C
variant, and adopting it after having seen OOD results would be exactly the post-hoc
optimisation the protocol forbids. It should be pre-registered and run as a separate
experiment with its own ID-only selection, and if it is run, the honest baseline to beat is
Mahalanobis at 78.82 / 92.55 — not P&C at 76.49.

## 16. What should and should not enter the manuscript

**Supportable claims.**
- P&C preserves ID accuracy within 0.5 pp while improving ID NLL over the base model
  (0.8043 vs 0.8482) — unchanged and independently reproduced here at K = 20.
- P&C's OOD detection exceeds the base model's own entropy at the same temperature by
  +1.90 Near / +1.32 Far.
- On this backbone, class-conditional Mahalanobis is a stronger **OOD detector**, and the
  advantage requires no labels (predicted labels match ground truth to 0.04 AUROC).
- P&C and Mahalanobis flag different inputs, and Mahalanobis is markedly worse at flagging
  the model's actual errors.

**Claims to avoid.**
- Do **not** write that `PERTURBATION_SUBSPACE` is the bottleneck. It is falsified here.
- Do **not** write "Mahalanobis is a better uncertainty estimator". Say "a better OOD
  detector on this benchmark" — §8 shows the broader statement is false.
- Do **not** attribute P&C's OOD signal to ensemble disagreement. Mutual information alone
  reaches 73.97 Near, below base entropy at T; expected member entropy (76.55) already
  matches the full predictive entropy (76.43).
- Do **not** report a larger K as an improvement. The sweep is flat within seed noise.
- Do **not** cite a SCOD number for the full network or for the final block; neither was run
  (§11).

**Framing suggestion.** The strongest honest story is mechanistic rather than competitive: a
random low-rank weight perturbation produces a representation change whose energy follows the
ID covariance, while feature-density detectors deliberately invert that covariance — so the
two methods are close to orthogonal by construction. That explains the gap, explains why rank
does not help, and motivates the geometry-aware variant above as future work.
