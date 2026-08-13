# Matched post-hoc uncertainty baselines on ImageNet-1K ViT-B/16

## 1. Executive summary

**Verdict: `P&C weaker than established matched alternatives`.**

Against methods starting from the same frozen ImageNet ViT-B/16, P&C beats MSP, Energy,
ReAct+Energy, a KFAC last-layer Laplace posterior, and its own matched uncorrected
ablation — every interval in the paired bootstrap excludes zero. It is **decisively beaten
by Mahalanobis**, which wins on all five OOD datasets and both aggregates
(Near −2.39, Far −4.81 AUROC against P&C, CIs [−2.68, −2.09] and [−5.08, −4.57]) while
costing 0.3 s to fit, 5.2 MiB of state and one forward pass per image against P&C's 12 s,
167 MiB and 20 member evaluations.

The label is chosen on more than aggregate AUROC, as §33 requires. Consistency favours
Mahalanobis too — it wins on every individual dataset, so the gap is not one benchmark. ID
preservation is the one axis where P&C leads: it is the **best-calibrated row in the table**
(NLL 0.8045 and ECE 0.0542 against the base model's 0.8482 and 0.0913), whereas Mahalanobis
leaves predictions untouched and supplies no predictive distribution at all. That is a real
distinction, but it does not overturn a loss on the question this comparison asks.

Two headline methods could not be run and are reported with measured evidence rather than
omitted: **SCOD** (`SCOD_NOT_TRACTABLE_AT_VIT_SCALE`) and **dense LLLA**
(`MEMORY_INFEASIBLE`).

## 2. Frozen P&C reference result

Taken unchanged from `../imagenet_vit_preservation_frontier/`: the PRIMARY 0.50 pp
preservation-budget configuration, r = 2.0, λ = 1000, K = 20, M = 20, n_cal = 32,768, seeds
{0, 10, 42, 123, 2026}, shared temperature T = 0.700. Nothing was retuned, no scale
reselected, and the relaxed r = 3 result was **not** substituted despite its better OOD
numbers. Its saved metrics and per-example scores are referenced, not recomputed.

## 3. Common checkpoint, data and evaluator

One harness for every method: the same pinned checkpoint (SHA-256 `c867db91…`),
preprocessing, ImageNet class mapping, training-derived pools (32,768 / 8,192 / 8,192), the
official 50,000-image validation set, the same five OpenOOD datasets with OpenOOD's own
image lists, and the same metric and Near/Far aggregation code.

The §7 gate confirms it: **9/9 checks reproduce the completed experiment to < 1e-6** — base
top-1 (0.81068), top-5 (0.95318), NLL, ECE, the ReAct threshold (0.6918), and MSP/Energy
Near and Far AUROC, all bit-identical (`provenance/parity_gate.json`).

## 4. Baseline implementation audit

`BASELINE_AUDIT.md` records, per method, the source file, the parameters it acts on, the
data and hyperparameters it needs, how those were selected, the score, the sample count and
whether extra optimization is required. Every existing implementation is JAX/Flax, so
"port" always means re-expressing the same algorithm in torch; the audit states exactly what
that algorithm is in each case. Three methods were stopped there rather than approximated.

## 5. Frozen-checkpoint comparison

See the table in `CORE_RESULTS.md` and `tables/vit_frozen_baselines.{md,csv,tex}`.

Ordering by Near AUROC: Mahalanobis 78.82 > **P&C 76.49** > Laplace-KFAC 74.63 >
MSP 73.52 > Uncorrected 72.53 > ReAct+Energy 69.21 > Energy 62.39. Far AUROC gives the same
ordering except that Laplace (86.56) and MSP (86.04) swap places with the uncorrected
ensemble.

## 6. Per-dataset OOD results

`tables/vit_baselines_per_dataset.csv`. P&C beats MSP on all five and Mahalanobis beats P&C
on all five:

| Method | SSB-hard | NINCO | iNaturalist | Textures | OpenImage-O |
|---|---|---|---|---|---|
| Mahalanobis | **71.33** | **86.31** | **95.86** | **89.54** | **92.26** |
| P&C (r=2) | 70.77 | 82.21 | 89.69 | 86.22 | 87.77 |
| Laplace (KFAC) | 69.90 | 79.36 | 88.67 | 85.43 | 85.58 |
| MSP | 68.93 | 78.11 | 88.19 | 85.06 | 84.86 |
| ReAct + Energy | 63.01 | 75.41 | 85.99 | 86.61 | 84.23 |
| Energy | 58.78 | 66.00 | 79.28 | 81.15 | 76.46 |

Mahalanobis's margin is smallest on SSB-hard (+0.56), the hardest near-OOD set, and largest
on iNaturalist (+6.17). ReAct+Energy is the only method that beats P&C anywhere (Textures,
86.61 vs 86.22).

## 7. LLLA results

The repo's `LLLAEnsemble` builds a **dense** GGN and covariance over every last-layer
parameter. At 768 → 1000 that is 769,000 parameters and a 769,000² covariance —
**2.37 TB** in float32, 4.7 TB at the float64 the CIFAR code inverts in.
`MEMORY_INFEASIBLE`, reported rather than silently swapped for a cheaper covariance.

What runs instead is the repository's *other* existing Laplace baseline,
`LaplaceEnsemble`, which is Kronecker-factored — so this is a port of a paper method, not a
ViT-specific variant invented here. Its factors (A 769×769, S 1000×1000) fit in 6.1 MiB and
take 1.1 s. Prior precision was selected on **ID selection-pool NLL** over the §12 log grid
extended to 10⁸ (the optimum otherwise sat on the grid edge, which would have understated
the baseline); λ = 10⁴ is a genuine interior optimum, NLL rising again beyond it.

Result: Near 74.63 / Far 86.56 — only +1.11 Near over MSP. At its ID-optimal prior the
posterior is nearly degenerate (base agreement 0.9987), and smaller λ collapses accuracy
(top-1 1.2 % at λ = 10⁻⁴). **Approximate last-layer posteriors carry little usable epistemic
signal on this network.**

## 8. SCOD feasibility

`SCOD_NOT_TRACTABLE_AT_VIT_SCALE`, on four independent grounds, three of them measured
(`metrics/scod_preflight.json`):

1. **No categorical likelihood exists to port.** `scod_distribution.py` implements Case A/B
   diagonal Gaussians for `ProbabilisticRegressionModel`; the only SCOD results in the repo
   are MuJoCo. There is no CIFAR SCOD.
2. **Storage.** At P = 86,567,656 and the repo's own `num_samples = 604`, the Nyström test
   matrix and sketch are 195 GiB each — **390 GiB** — against 12 GiB VRAM and 24.6 GiB RAM.
   A rank-10 fallback still needs 41 GiB, and would no longer be the same method.
3. **Compute.** One Fisher matvec costs a measured **50.9 ms/image** (N = 1,024, peak
   5.08 GiB), i.e. 27.8 min over the 32,768-image pool and **≈ 280 GPU-hours** for the full
   604-matvec sketch.
4. **Autodiff support.** torchvision's fused attention implements neither forward-mode AD
   (`aten::_native_multi_head_attention`) nor a second derivative
   (`_scaled_dot_product_efficient_attention_backward`). Measuring anything required forcing
   the MATH SDPA backend and obtaining the JVP through a double-VJP identity.

To measure cost at all, a softmax Fisher factor was constructed and verified numerically
(`max |AᵀA − (diag(p) − ppᵀ)| = 4.2e-17`). That exists solely for the preflight and is
**not** offered as a SCOD baseline; per §13 no head-only approximation is substituted.

## 9. ReAct and Mahalanobis results

ReAct+Energy reuses the completed run's scores verbatim (threshold 0.6918 = p90 of ID
training-pool activations). It improves markedly on plain Energy (+6.8 Near, +6.7 Far) but
stays below MSP on Near — on a ViT, clipping recovers only part of Energy's deficit.

Mahalanobis is the single-layer class-conditional Gaussian with a shared covariance and the
canonical fixed 1e-6 ridge — **not** the OOD-tuned logistic-regression combination of the
original paper, and not multi-layer. It has no tunable hyperparameter, so nothing can leak.
Fitted on the 32,768-image pool in 0.3 s.

One difference deserves stating: Mahalanobis uses the calibration images' **labels**,
whereas P&C's correction regresses onto the base model's own outputs and uses none. Both
are ID-only and neither sees OOD data, so the comparison is fair, but they do not consume
the same information.

## 10. Secondary Laplace / SWAG / Subspace

`Laplace` and `LLLA` are **not** redundant here — see §7; the dense form is infeasible and
the KFAC form is what runs, so one row covers both and is labelled "Laplace (KFAC)".

SWAG and Subspace Inference are `METHOD_REQUIRES_RETRAINING` (category B). Both consume an
SGD trajectory; Subspace additionally needs SWAG's PCA directions. A faithful trajectory
means fine-tuning ImageNet on a TITAN X that sustains 131 img/s — order 2.7 h per epoch for
1.28 M images forward-only, several times that with backward — and §16/§17 forbid
substituting head-only fine-tuning or random subspaces.

## 11. MC Dropout / Epinet

`MC_DROPOUT_NOT_APPLICABLE`: the pinned checkpoint has 37 `nn.Dropout` modules, all p = 0.0,
and attention dropout 0.0 (verified programmatically). Stochastic forwards are identical to
the deterministic one, and §18 forbids injecting dropout the checkpoint never trained with.

`EPINET_NOT_PORTED_METHOD_CHANGE`: the epinet is a *trained* epistemic head; porting means
designing and training an ImageNet-scale one, which is method design rather than a port.

## 12. Matched uncorrected P&C ablation

Retained deliberately (§29) — it answers a different question from the external baselines.
At the same r = 2.0, identical basis, coefficients, scale and seeds:

| | ID Acc | ID NLL | Near AUROC | Far AUROC |
|---|---|---|---|---|
| P&C | 80.830 | 0.8045 | 76.49 | 87.89 |
| Uncorrected | 80.619 | 0.9048 | 72.53 | 84.40 |

The correction is worth +3.96 Near and +3.49 Far AUROC. Note the uncorrected ensemble scores
*below MSP* on Near — at r = 2 an uncorrected perturbation ensemble is worse than doing
nothing, and the correction is what makes the perturbation useful at all.

## 13. Timing, memory and storage

`tables/vit_baselines_compute.csv`, all MEASURED except the SCOD row (ESTIMATED from a
measured matvec, labelled as such).

| Method | fit | storage | evals/img | peak VRAM |
|---|---|---|---|---|
| MSP / Energy | 0 s | 0 | 1 | 0.47 GiB |
| ReAct + Energy | 0.1 s | 0 | 1 | 0.47 GiB |
| Mahalanobis | 0.3 s | 5.2 MiB | 1 | 0.47 GiB |
| Laplace (KFAC) | 1.1 s | 6.1 MiB | 20 | 0.47 GiB |
| P&C | 12 s | 167 MiB | 20 | 2.16 GiB |
| SCOD (not run) | ≈ 280 h | 390 GiB | 1 | 5.08 GiB measured |

Construction times exclude the shared one-off activation cache (5.0 min for 32,768 images),
which every fitted method here reuses. No method is called "more efficient" on one axis
alone: P&C's 12 s construction is trivial, but its 20 evaluations per image and 167 MiB of
state are the largest online cost of any method that ran.

## 14. Statistical comparisons

Paired bootstrap, 2,000 replicates, seed 20260815, ID subsampled to 10,000 per replicate,
using the saved per-example scores (`metrics/paired_bootstrap.json`). All four intervals
exclude zero: P&C − Mahalanobis Near −2.39 [−2.68, −2.09] and Far −4.81 [−5.08, −4.57];
P&C − Laplace +1.80 / +1.18; P&C − MSP +2.91 / +1.70; P&C − Energy +14.04 / +8.78. ReAct is
excluded because its per-example ID scores were not saved by the original run.

## 15. Failures and non-applicable methods

| Method | Classification | Evidence |
|---|---|---|
| SCOD | `SCOD_NOT_TRACTABLE_AT_VIT_SCALE` | measured: 390 GiB sketch, 280 GPU-h, no categorical likelihood, no forward-AD |
| LLLA (dense) | `MEMORY_INFEASIBLE` | 769,000² covariance = 2.37 TB |
| MC Dropout | `NOT_APPLICABLE` | all 37 dropout modules p = 0.0 |
| SWAG | `METHOD_REQUIRES_RETRAINING` | needs an SGD trajectory |
| Subspace Inference | `METHOD_REQUIRES_RETRAINING` | needs SWAG's trajectory + PCA |
| Epinet | `EPINET_NOT_PORTED_METHOD_CHANGE` | trained epistemic head |
| Deep Ensemble | not run by instruction | independent ImageNet training |

## 16. Recommended manuscript treatment

1. **Report the Mahalanobis result.** It is the strongest matched frozen-checkpoint method
   on this benchmark, on every dataset, at a fraction of the cost. A reviewer will run it;
   better that the paper has already done so.
2. **Reframe P&C's claim at ImageNet scale.** The defensible claims are that P&C is the
   best-calibrated predictive distribution among the matched methods, that it beats every
   approximate-posterior and confidence-score baseline tested, and that its correction is
   what makes a large perturbation usable. "Best OOD detector" is not supportable here.
3. **State what Mahalanobis does not provide** — no predictive distribution, no epistemic
   uncertainty on the prediction, only a detector score. That is the honest axis on which
   P&C is preferable, and it should be argued explicitly rather than implied by a table.
4. **Report SCOD's infeasibility as a finding.** "The paper's own sensitivity-based
   comparator needs 390 GiB of sketch at ImageNet scale" is informative about the method
   class, not an admission.
5. **Keep the uncorrected ablation adjacent to the baseline table** (§29). That the
   uncorrected ensemble scores below MSP while P&C scores above it is the cleanest statement
   of what the correction contributes.
6. Note Energy's collapse on ViTs (62.39 vs MSP's 73.52) — useful for readers carrying CNN
   intuitions to transformers.
