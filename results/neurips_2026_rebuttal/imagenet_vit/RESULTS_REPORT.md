# Final-block P&C on ImageNet-1K ViT-B/16 — results

## 1. Executive summary

**Verdict: `SUPPORTS_LARGE_SCALE_TRANSFER`.**

Perturb-and-Correct applies post hoc to a standard 86.6M-parameter ImageNet-1K ViT-B/16
with no retraining, preserves in-distribution behaviour to within 0.024 pp of the base
model, and produces the best OOD ranking of the five post-hoc scores tested on the
canonical OpenOOD ImageNet benchmark. Construction costs 0.50 s per member and 9.9 s for a
20-member ensemble after a one-off 5-minute activation cache, at 1.58 GiB peak GPU on a
12 GiB TITAN X.

Two qualifications keep this from being "strongly supports":

* The margin over the strongest baseline is real but modest: **+1.2 Near AUROC** and
  **+0.5 Far AUROC** over MSP, with larger gains on FPR95 (−7.5 pp Near, −3.0 pp Far).
* **At the ID-selected operating point the affine correction contributes almost nothing
  over the identical uncorrected perturbations** (74.74 vs 74.60 Near AUROC — within seed
  noise). The correction halves per-member logit deviation from the base model, which is
  what the ID gate rewards, but that ID fidelity does not convert into OOD separation at
  r = 0.375.

A post-hoc sweep explains why, and is the most scientifically interesting result here: the
correction's benefit **grows monotonically with perturbation size**, and past r ≈ 1 the
uncorrected ensemble collapses while P&C keeps improving (Near AUROC at r = 2: 76.32 for
P&C, 71.75 uncorrected). The correction is what makes a large perturbation usable. A
conservative ID-stability gate simply never lets the experiment get there. This is stated
as a post-hoc diagnostic and does **not** revise the frozen configuration.

## 2. Scientific motivation

The reviewer concern was evaluation at larger scale. This experiment asks whether P&C —
previously shown on CIFAR-scale classifiers, MuJoCo dynamics models and a Banking77
DistilBERT — transfers to an ImageNet-scale Vision Transformer under a protocol where
every construction and selection decision uses ID data only. It is not an attempt to
maximise ImageNet OOD performance or to claim state of the art.

## 3. Exact checkpoint and environment

| | |
|---|---|
| model | torchvision `vit_b_16`, `ViT_B_16_Weights.IMAGENET1K_V1` (pinned) |
| SHA-256 | `c867db91d3e12c6cbadabb610d73c24a546bf82d8c03a9fea34f43a712ddb0e9` |
| parameters | 86,567,656 |
| dtype | float32 throughout (fp16 is 44% slower on Pascal and was not used) |
| GPU | NVIDIA TITAN X (Pascal), 12 GiB, sm_61 |
| torch / torchvision | 2.7.1+cu126 / 0.22.1+cu126, in `.venv_vit` |

The repo's main `.venv` (torch 2.11+cu130) has no `sm_61` kernels and cannot run this card
at all; the preflight established the working environment reused here.

**Base parity.** On the untouched 50,000-image validation set the checkpoint scores
**81.068 % top-1 / 95.318 % top-5** against the published 81.072 / 95.318 — a 0.004 pp
difference, i.e. 2 images, with top-5 matching exactly.

Reaching that required the authentic gated ILSVRC JPEGs. The public
`evanarlian/imagenet_1k_resized_256` mirror used in the preflight re-encodes every image at
~0.26 bytes/pixel with the short side pre-scaled to 256, which costs **1.85 pp of top-1**
(79.22 %). Worse for an OOD study, it would have left ID images visibly more compressed
than the OOD sets — which are distributed as originals — allowing a detector to separate ID
from OOD partly on compression artefacts. All ID and OOD images now pass through one
identical pipeline.

## 4. Dataset and split provenance

All construction, selection and calibration draws from ImageNet **training** data; the
official validation set is touched only for final evaluation.

| pool | n | classes | per class | use |
|---|---|---|---|---|
| correction | 32,768 | 1000 | 32–33 | fit the corrected W2 |
| selection | 8,192 | 1000 | 8–9 | ID-only hyperparameter search |
| temperature | 8,192 | 1000 | 8–9 | shared scalar temperature |
| ID evaluation | 50,000 | 1000 | 50 | official validation split |

Pools are pairwise disjoint by construction and exactly class-stratified (per-class counts
allocated with the remainder spread over a seeded class permutation). Manifests with the
original ImageNet filenames and SHA-256 checksums are in `splits/`; split seed 20260813.

OOD membership comes from **OpenOOD's own `imglist` files**, not from archive contents:
SSB-hard 49,000 and NINCO 5,879 (Near); iNaturalist 10,000, Textures 5,160 and
OpenImage-O 15,869 (Far); 85,908 total. Notably the OpenImage-O archive holds 17,632
images, of which 1,763 form a validation split this experiment never reads.

## 5. Final-block CLS-conditioned P&C

The target is the final encoder block's FFN:
`encoder_layer_11.mlp.0` (W1, 768→3072) → GELU → `encoder_layer_11.mlp.3` (W2, 3072→768).

Exactly one correction row per calibration image: the CLS post-GELU activation entering W2,
with the base model's own W2 output for that row as target, fitted by the existing
original-centred ridge objective.

This is a design decision, not only a shortcut. After the final block's FFN every remaining
operation (`encoder.ln`, the CLS slice, `heads`) is token-wise, so **only the CLS output can
affect the classifier** — verified numerically in the preflight (CLS-only tail vs full-token
tail agree to 9.5e-07). Patch-token rows would add constraints that cannot influence the
prediction while pulling the least-squares fit away from the one direction that can; the
preflight measured that they make held-out task-relevant preservation *worse*. Earlier
blocks would need a different operator, because later attention mixes patch tokens into
CLS — which is why the primary experiment does not search them.

## 6. Hyperparameter-search protocol

Searched: perturbation scale, ridge strength, correction-set size. Fixed by design: target
layer, K = 20, M, token strategy, and the OOD score. Scale is reported as the **realized**
relative Frobenius norm r = ‖ΔW₁‖_F/‖W₁‖_F, not an implementation multiplier.

Stage A swept r ∈ {0.125, 0.25, 0.5, 1.0, 2.0} at n_cal = 32,768, λ = 1e-3, M = 5 on the ID
selection pool. The adaptive rule did not need to extend the grid: r = 2.0 already fails
and r = 0.125 already passes, so the stability boundary is bracketed inside the grid at
**r_boundary = 0.5**. Stage B then searched 3 scales × 2 correction sizes × 3 ridge values
(18 configurations) at M = 10.

## 7. ID-only configuration selection

The gate (frozen in `selection/ID_STABILITY_RULE.md` before any OOD data was read): top-1
drop ≤ 0.25 pp, base agreement ≥ 99 %, all metrics finite, and the correction must beat the
matched uncorrected ensemble on **both** median and p99 per-member logit MSE. Calibration
residual is deliberately not an acceptance criterion — the preflight showed it is lowest
exactly where the correction is most overfitted.

15 of 18 configurations passed; 8 tied on ID NLL within 0.002. Applying the frozen
tie-break (larger scale → n_cal = 16,384 → smaller ridge) selects:

```text
r = 0.375   lambda = 1.0   n_cal = 16,384   K = 20   M = 20
```

`selection/SELECTION_REPORT.md` records **OOD data accessed before configuration freeze:
NO**, and the configuration was committed to git before the OOD stage was run.

The gate earns its place: `r = 0.375, λ = 1e-3, n_cal = 16,384` fails on
`corrected_p99_not_better` (p99 1.77e-02 against the uncorrected 1.73e-02) — the preflight's
small-correction-set tail problem reproducing under the training-derived protocol.

**Local robustness (§14, diagnostic).** At 0.8× / 1.0× / 1.2× the selected scale, top-1
varies by 0.024 pp and agreement by 0.0004; 1.0× and 1.2× pass the gate and 0.8× fails only
on the p99 criterion. The selected point sits on a plateau, not on a spike.

## 8. ImageNet ID results (50,000 images, 5 construction seeds, M = 20)

| | top-1 | top-5 | NLL | ECE | base agreement |
|---|---|---|---|---|---|
| base (T-scaled) | 81.068 | 95.318 | 0.8482 | 0.0913 | — |
| **P&C** | **81.044 ± 0.006** | 95.308 ± 0.012 | **0.8431 ± 0.0004** | 0.0901 ± 0.0002 | 0.9954 |
| uncorrected | 81.047 ± 0.017 | 95.299 ± 0.011 | 0.8440 ± 0.0005 | 0.0896 ± 0.0004 | 0.9971 |

P&C loses 0.024 pp of top-1 and *improves* NLL over the base model. Per-member logit MSE
against the base model (mean over seeds of each seed's distribution statistic):

| | mean | median | p90 | p95 | p99 |
|---|---|---|---|---|---|
| P&C | 2.92e-03 | 1.22e-03 | 8.34e-03 | 1.15e-02 | 1.79e-02 |
| uncorrected | 4.17e-03 | 2.75e-03 | 9.42e-03 | 1.23e-02 | 1.85e-02 |

The correction improves every quantile, most strongly the median (2.3×); the gap narrows
in the tail (1.04× at p99), which is the same heavy-tailed behaviour the preflight found.

A shared temperature of **T = 0.700** was fitted on base logits over the training-derived
temperature pool, per the existing classification convention. Because training images are
much easier for the model than validation images (93.8 % vs 81.1 % top-1), NLL-optimal T on
that pool is < 1 and *sharpens*; base ECE consequently rises from 0.0560 untempered to
0.0913 tempered. This is a property of the mandated train-only calibration protocol, not of
P&C, and it is applied identically to every method, so comparisons are unaffected. Under
that shared protocol P&C is marginally better calibrated than the base model.

## 9. Uncorrected perturbation ablation

The ablation uses the identical basis, coefficients, scale and seeds; only W2 differs.

At the selected r = 0.375 the two are indistinguishable on ID accuracy (81.044 vs 81.047)
and on OOD (74.74 vs 74.60 Near, 86.52 vs 86.52 Far). The correction's measurable effect at
this scale is on ID fidelity: **2.3× lower median per-member logit deviation**.

The correction's importance is strongly scale-dependent (from Stage A, ratio of uncorrected
to corrected median logit MSE): 2.6× at r = 0.125, 2.7× at 0.25, 3.1× at 0.5, 4.6× at 1.0,
**11.7× at r = 2.0**. See §12.

## 10. OpenOOD Near/Far results

| Method | Near AUROC | Near FPR95 | Far AUROC | Far FPR95 |
|---|---|---|---|---|
| **P&C** | **74.74 ± 0.02** | **74.37 ± 0.14** | **86.52 ± 0.01** | 48.75 ± 0.08 |
| Uncorrected perturb. | 74.60 ± 0.02 | 74.63 ± 0.35 | 86.52 ± 0.02 | **48.70 ± 0.15** |
| MSP | 73.52 | 81.84 | 86.04 | 51.74 |
| ReAct + Energy | 69.21 | 84.23 | 85.61 | 53.90 |
| Energy | 62.39 | 93.16 | 78.96 | 85.29 |

Per dataset (P&C seed 0 vs MSP, AUROC):

| dataset | group | n | P&C | MSP |
|---|---|---|---|---|
| SSB-hard | near | 49,000 | 69.96 | 68.93 |
| NINCO | near | 5,879 | 79.50 | 78.11 |
| iNaturalist | far | 10,000 | 88.60 | 88.19 |
| Textures | far | 5,160 | 85.32 | 85.06 |
| OpenImage-O | far | 15,869 | 85.64 | 84.86 |

P&C is ahead on every one of the five datasets. Seed variability is tiny (±0.02 AUROC).

## 11. Baseline comparison

All baselines use the same checkpoint, preprocessing, datasets and evaluator; only the
score differs. ReAct's clipping threshold (0.6918) is the p90 of penultimate activations on
the ID *training* temperature pool — no OOD data and no validation data.

The notable result is that **Energy is much worse than MSP on this ViT** (62.4 vs 73.5 Near
AUROC), and ReAct recovers only part of the gap (69.2). Energy's advantage over MSP is a
CNN-era result; it does not carry over here. P&C beats all three.

## 12. Hyperparameter robustness and post-hoc OOD sensitivity

**Among ID-stable configurations, OOD performance is nearly flat**: the Near-AUROC spread
across all gate-passing scales is **0.31 pp**. ID-only selection therefore identifies the
OOD operating point only weakly — but it also cannot do much harm, because everything it
can choose performs similarly.

Extending the sweep past the gate (post hoc, `metrics/posthoc_ood_sensitivity/`, and
`figures/posthoc_ood_vs_scale.png`):

| r | gate | P&C Near | uncorr. Near | P&C Far | uncorr. Far |
|---|---|---|---|---|---|
| 0.125 | pass | 74.55 | 74.50 | 86.44 | 86.41 |
| 0.25 | pass | 74.61 | 74.49 | 86.48 | 86.42 |
| **0.375** | selected | **74.72** | 74.47 | **86.54** | 86.45 |
| 0.5 | pass | 74.85 | 74.46 | 86.60 | 86.49 |
| 1.0 | **fail** | 75.54 | 74.15 | 86.78 | 86.52 |
| 2.0 | **fail** | 76.32 | 71.75 | 86.43 | 83.78 |

Two things are visible. The corrected and uncorrected curves separate steadily with scale
(+0.25 Near at the selected point, +1.39 at r = 1, **+4.57 at r = 2**), and the uncorrected
ensemble *peaks and then collapses* while P&C keeps climbing. **The correction is the
mechanism that allows the perturbation to be scaled up at all.** The best Near AUROC seen
anywhere is 76.32 at r = 2 — 1.6 points above the selected configuration — but it violates
the ID gate (top-1 −0.842 pp, agreement 98.1 %).

Per §29 none of this revises the headline. It does identify the ID gate as the binding
constraint and suggests the ID/OOD trade-off deserves explicit treatment.

## 13. Correction-size / ridge diagnostic

At r = 0.375 (`tables/ncal_ridge_diagnostic.csv`), with the uncorrected held-out CLS
residual at 0.1922 throughout:

| n_cal | λ | calib residual | held-out residual | median MSE | p99 MSE | gate |
|---|---|---|---|---|---|---|
| 16,384 | 1e-3 | 0.0828 | 0.1470 | 1.03e-03 | 1.77e-02 | **fail** |
| 16,384 | 1 | 0.0838 | 0.1433 | 9.99e-04 | 1.65e-02 | pass |
| 16,384 | 100 | 0.1041 | 0.1396 | 1.18e-03 | 1.28e-02 | pass |
| 32,768 | 1e-3 | 0.0911 | 0.1333 | 9.04e-04 | 1.39e-02 | pass |
| 32,768 | 1 | 0.0915 | 0.1325 | 9.01e-04 | 1.36e-02 | pass |
| 32,768 | 100 | 0.1015 | 0.1331 | 1.03e-03 | 1.22e-02 | pass |

This answers Q7 affirmatively: **the preflight's finding reproduces under the training-derived
protocol.** The lowest calibration residual (0.0828) belongs to the one configuration that
*fails* the gate; doubling n_cal to 32,768 raises calibration residual while lowering the
held-out residual and the p99 tail. Ridge substitutes for calibration coverage — at 16,384
rows λ = 1e-3 is not enough and λ ≥ 1 is needed, while at 32,768 rows λ = 1e-3 is already
safe. Both routes reach a similar place, so this is a genuine interpolation/generalisation
effect, not a universal transformer result.

## 14. Runtime and memory

| quantity | measured |
|---|---|
| base checkpoint load | 0.8 s |
| model resident (fp32) | 0.340 GiB |
| perturbation basis (K = 20) | 0.76 s, 180 MiB CPU, 0 GiB GPU |
| (h, z₀) cache, 32,768 images | 5.0 min at 109 img/s, 289 MiB |
| ImageNet val cache, 50,000 images | 7.2 min at 116 img/s |
| OOD cache, 85,908 images | 12.3 min at 116 img/s |
| **per-member correction** | **0.50 s** |
| **M = 20 ensemble construction** | **9.9 s** |
| peak GPU, construction / inference | 1.584 / 1.584 GiB |
| compact ensemble on disk | 167 MiB per seed vs **6.45 GiB** for 20 ViT copies |
| ImageNet throughput, cold / **settled** | 149 / **131** img/s |
| thermal decay | 12.0 % (1708 → 1417 MHz, 88 °C) |

Throughput is reported settled, not cold: the card loses 12 % of its clock under sustained
load, so short timings overstate a long run.

**No inference-time advantage is claimed.** M = 20 members cost M tail evaluations. The
efficiency statement is narrow and specific: *P&C constructs its ensemble from one
pretrained checkpoint without training independent ViTs* — 9.9 s and 167 MiB per ensemble,
against the cost of training 20 ImageNet ViTs.

## 15. Failure modes and caveats

* **The correction is not doing the work at the selected operating point** (§9). Reported
  plainly rather than buried; the post-hoc sweep in §12 shows why and where it does.
* **The ID gate binds.** It admits only r ≤ 0.5, and the best OOD operating point found lies
  outside it. Whether ≤ 0.25 pp top-1 is the right threshold is a design question the
  manuscript should argue explicitly rather than assume.
* **Modest margins.** +1.2 Near AUROC over MSP is real and consistent across all five
  datasets and five seeds, but small.
* **Train-fitted temperature sharpens** (T = 0.700) and raises ECE relative to no
  temperature, because ImageNet training images are far easier than validation images. It
  is applied identically to every method.
* **Construction seeds are not training seeds.** All five ensembles come from one
  checkpoint; the spread measures construction-noise only and is correspondingly tiny
  (±0.02 AUROC). It must not be presented as ensemble-training variance.
* **Single target block, single token design.** Earlier blocks were not searched, by design.
* **28 of 294 training shards** were downloaded (122,024 of 1,281,167 rows); pools were
  drawn from those. Shards are shuffled, so this is a uniform sample, but it is not the
  whole training set.
* MC-Dropout was not run: torchvision's ViT-B/16 has dropout disabled (p = 0.0) in the
  released configuration, so it would require altering the architecture — explicitly out of
  scope per §21.

## 16. Recommended manuscript treatment

1. **Lead with feasibility and ID preservation.** "P&C constructs post hoc on an 86M
   ImageNet ViT in 9.9 s from one checkpoint, and preserves top-1 to within 0.024 pp" is a
   clean, defensible answer to the scale concern.
2. **Report the OOD table with the uncorrected ablation adjacent**, not in an appendix. The
   honest reading — the perturbation ensemble supplies most of the gain at the selected
   operating point — is better made by us than by a reviewer.
3. **Use the post-hoc scale sweep (Figure `posthoc_ood_vs_scale.png`) as the mechanism
   evidence.** It is the strongest result in this experiment: the uncorrected ensemble
   degrades past r ≈ 1 while P&C continues to improve, which is precisely the claim the
   method makes. Label it post hoc.
4. **State the ID/OOD trade-off explicitly** rather than presenting r = 0.375 as optimal. It
   is optimal under a chosen ID-stability gate, and a looser gate would score better OOD.
5. **Do not claim an inference-time win.** Claim construction cost and storage.
6. Keep Energy's underperformance on ViTs as a short remark — it is a useful observation for
   readers carrying CNN intuitions to transformers.
