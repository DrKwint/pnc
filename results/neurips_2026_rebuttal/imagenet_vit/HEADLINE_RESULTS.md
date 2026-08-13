# HEADLINE RESULTS — final-block P&C on ImageNet ViT-B/16

Written at the §32 checkpoint: selected hyperparameters, 5 construction seeds, full 50k
ImageNet validation, all five OpenOOD datasets, MSP, Energy, ReAct+Energy and the matched
uncorrected ablation are all complete. Optional analyses come after this file and do not
change anything in it.

## Configuration (selected from ID data only, frozen before OOD was read)

```text
model         torchvision vit_b_16, ViT_B_16_Weights.IMAGENET1K_V1 (sha256 c867db91…)
target        encoder_layer_11.mlp.0 (perturb W1) -> GELU -> .mlp.3 (correct W2)
observation   CLS token only, 1 correction row per calibration image
r             0.375   (realized ||dW1||_F/||W1||_F, median over members)
lambda        1.0
n_cal         16,384   (from the 32,768-image training correction pool)
K = 20        M = 20   seeds {0, 10, 42, 123, 2026}
temperature   0.700, shared, fit on base logits over the 8,192-image ID temperature pool
OOD score     predictive entropy of the temperature-scaled mean member softmax
```

Base parity: **81.068 % top-1 / 95.318 % top-5** on the untouched 50,000-image validation
set, against the published 81.072 / 95.318 — a 0.004 pp difference, i.e. 2 images.

## Main table

| Method | ID Acc | ID NLL | ID ECE | Near AUROC | Near FPR95 | Far AUROC | Far FPR95 |
|---|---|---|---|---|---|---|---|
| Base / MSP | 81.07 | 0.8482 | 0.0913 | 73.52 | 81.84 | 86.04 | 51.74 |
| Energy | 81.07 | 0.8482 | 0.0913 | 62.39 | 93.16 | 78.96 | 85.29 |
| ReAct + Energy | 81.07 | 0.8482 | 0.0913 | 69.21 | 84.23 | 85.61 | 53.90 |
| Uncorrected perturb. | 81.05 ± 0.02 | 0.8440 ± 0.0005 | 0.0896 ± 0.0004 | 74.60 ± 0.02 | 74.63 ± 0.35 | 86.52 ± 0.02 | 48.70 ± 0.15 |
| **P&C** | **81.04 ± 0.01** | **0.8431 ± 0.0004** | 0.0901 ± 0.0002 | **74.74 ± 0.02** | **74.37 ± 0.14** | **86.52 ± 0.01** | 48.75 ± 0.08 |

Accuracy/AUROC/FPR95 in %; P&C and the uncorrected ablation are mean ± std over 5
construction seeds at M=20. Near = SSB-hard, NINCO. Far = iNaturalist, Textures,
OpenImage-O, macro-averaged over datasets.

## The four things this establishes

**1. P&C constructs post hoc on an 86M-parameter ImageNet ViT.** No retraining, no
fine-tuning, no second checkpoint. After a one-off 5.0 min activation cache, a member costs
**0.50 s** and a 20-member ensemble **9.9 s**, at 1.58 GiB peak GPU on a 12 GiB card. The
compact state is 167 MiB per seed against 6.45 GiB for 20 ViT copies.

**2. ID behaviour is preserved.** 81.04 ± 0.01 % vs 81.068 % base — a 0.024 pp drop — with
99.54 % top-1 agreement, and NLL slightly *better* than the base model (0.8431 vs 0.8482).

**3. P&C is the best of the five scores on this benchmark**, though by a modest margin over
MSP: +1.2 Near AUROC, +0.5 Far AUROC, and the clearer gains on FPR95 (−7.5 pp Near,
−3.0 pp Far). Energy is markedly *worse* than MSP on this ViT (62.4 vs 73.5 Near AUROC),
and ReAct recovers only part of that gap.

**4. The affine correction is not what produces the OOD signal here.** The matched
uncorrected ensemble — identical basis, coefficients, scale and seeds, differing only in
whether W2 is corrected — scores 74.60 Near / 86.52 Far against P&C's 74.74 / 86.52. That
difference is within seed noise. The correction does measurably improve ID fidelity
(median per-member logit MSE 1.27e-03 vs 2.88e-03, a 2.3× reduction), which is what the
ID-stability gate rewards, but at the selected operating point it does not translate into
better OOD ranking.

## Reading of the headline

The experiment **supports large-scale transfer of the construction**: P&C builds cleanly on
a standard ImageNet ViT, preserves ID behaviour, and beats every post-hoc baseline tested.
It does **not**, at this operating point, isolate the correction as the source of that
benefit — the low-rank perturbation ensemble alone accounts for essentially all of it.

The likely reason is visible in the ID-stability frontier: the gate admits only
r ≤ 0.5, and the correction's advantage over doing nothing grows with r (median logit-MSE
ratio 2.6× at r=0.125, 3.1× at r=0.5, 4.6× at r=1.0, 11.7× at r=2.0). ID-only selection
therefore lands in the regime where the uncorrected members are already close to the base
model. Whether a larger perturbation with correction would separate the two on OOD is
examined post hoc in `metrics/posthoc_ood_sensitivity/` — and, per §29, cannot and does not
revise the frozen configuration above.
