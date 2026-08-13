# ID selection rule — preservation-budget protocol

Written and committed **before** the follow-up OOD evaluation was run. Everything below is
decided on ID data only.

## Principle

> **P&C scale = the largest perturbation supported by the *corrected* model's ID-preservation
> budget** — not the largest perturbation for which the *uncorrected* model is already benign.

The matched uncorrected ensemble is an explanatory control. Its accuracy, agreement, logit
MSE and OOD performance **may not enter any selection decision**.

## What is no longer required

The original protocol's gate is deliberately dropped. None of the following constrains
selection here; all are recorded as diagnostics only:

- base top-1 agreement ≥ 99 %
- corrected median logit MSE < uncorrected median logit MSE
- corrected p99 logit MSE ≤ uncorrected p99 logit MSE
- calibration residual (of any magnitude)

## Preservation budgets (predeclared)

| budget | maximum top-1 loss | ε |
|---|---|---|
| STRICT | 0.25 pp | 0.0025 |
| **PRIMARY** | **0.50 pp** | **0.0050** |
| RELAXED | 1.00 pp | 0.0100 |

**PRIMARY (0.50 pp) is the main operating point.** All three are reported so the
preservation/diversity trade-off is visible. These margins are fixed now and will not be
revised after seeing OOD results.

## Paired accuracy-equivalence test

For a configuration, corrected ensemble and base model are evaluated on the *same* 8,192
ID-selection examples (ImageNet **training** pool, disjoint from correction and temperature
pools, never the validation set).

Per example i, with correctness averaged over the construction seeds:

    d_i = mean_s[ 1(P&C_s correct on i) ] − 1(base correct on i)
    Δ_acc = mean_i d_i      (equals mean-over-seeds accuracy minus base accuracy)

A paired nonparametric bootstrap resamples examples with replacement, **10,000 replicates,
bootstrap seed 20260814**. Because d takes finitely many values, each replicate is drawn as
a multinomial over the empirical value counts — exactly equivalent to index resampling, and
fast enough to run for every configuration.

    LCB = 5th percentile of the bootstrap distribution of Δ_acc   (one-sided 95 % lower bound)

**A configuration passes budget ε iff `LCB ≥ −ε`.** The raw point estimate is reported
alongside but does not decide the outcome — this prevents a favourable fluctuation on one
8,192-image sample from being read as preservation.

## Ridge choice at a fixed scale (ID-only)

Among ridge values passing the same budget at that scale:

1. lowest corrected ID NLL;
2. ties within 0.002 NLL → lower ECE;
3. still tied → smaller ridge.

## Scale choice (per budget)

1. find every r for which at least one ridge passes;
2. take the **largest** such r;
3. use its ID-selected ridge from the rule above.

This yields three frozen operating points: `STRICT_CONFIG`, `PRIMARY_CONFIG`,
`RELAXED_CONFIG`.

## Calibration safeguard

NLL and ECE do **not** define a narrow tolerance — that would recreate the conservative gate
this protocol replaces. A configuration is flagged `CALIBRATION_PATHOLOGY` only for:

- any NaN/Inf in metrics or in the ridge solution;
- catastrophic NLL increase (> 2× the base model's NLL);
- catastrophic ECE increase (> 3× the base model's ECE);
- numerically unstable solve.

A configuration that preserves accuracy but *noticeably* worsens NLL or ECE is retained and
the trade-off reported explicitly.

## Fixed by design (not searched)

target layer (`encoder_layer_11.mlp.0` → GELU → `.mlp.3`), CLS-only correction observation,
K = 20, base checkpoint, preprocessing, **n_cal = 32,768**, shared temperature **T = 0.700**
(reused, not refitted per configuration, so ID NLL reflects the predictor rather than each
configuration recalibrating itself), and the OOD score (predictive entropy of the
temperature-scaled mean member softmax).

Search dimensions in the primary experiment are **only** perturbation magnitude r and ridge λ.

## Pairing across scales

Within a construction seed the perturbation basis and the member coefficient directions are
drawn once and reused at every scale; different scales are obtained by scaling those same
directions. Scale comparisons are therefore paired rather than confounded by different
random directions. Realized median r across members is reported, not only the requested
value.

## Provenance caveat

This protocol is a **follow-up**, motivated by analysis of the earlier `imagenet_vit/`
experiment — which had already inspected OOD performance at some scales, including r = 1 and
r = 2. It is not a pristine preregistered OOD experiment, and no such claim is made. What
holds is narrower and still worth having: **once this rule was fixed and committed, the
configurations it selects were determined by ID data alone.** The earlier experiment's
cleanly-selected result is preserved unchanged in `../imagenet_vit/`.
