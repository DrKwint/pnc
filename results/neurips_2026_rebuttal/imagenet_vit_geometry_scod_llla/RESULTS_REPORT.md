# ViT diagnostic + memory-bounded LLLA — geometry, SCOD scoping, scalable Laplace

**Status: Parts A (core) and C are complete. Part B (SCOD-LL) is implemented up to the
scoping/feasibility groundwork but was not run — see §10. Sub-parts A6, A7 and A8 are also
not run.** Nothing below retunes P&C (frozen at r=2, λ=1000, K=20, M=20) or alters the
Mahalanobis headline.

## 1. Executive summary

**Mahalanobis's advantage is not class supervision.** Refitting the same class-conditional
Gaussian on the ViT's *own predicted labels* instead of ground truth changes nothing
(Near 78.86 vs 78.82, Far 92.47 vs 92.55). No ImageNet labels are needed to obtain the
result. What matters is class-conditional structure (+3.3 Near over a single global
Gaussian) and covariance whitening (+3.4 Near over nearest-centroid) — and, strikingly, a
single **unconditional** Gaussian on the CLS feature already beats P&C on Far OOD
(91.13 vs 87.74).

**P&C does not contain an attenuated copy of the Mahalanobis signal.** Its own mechanism
quantities are *weak* detectors — transfer/leverage terms reach only 55–62 Near AUROC where
P&C's predictive entropy reaches 76.43. The geometric signal enters the perturbation
response only weakly (Spearman 0.634 from Mahalanobis to hidden perturbation change) and is
then carried almost losslessly through correction and output projection (0.99, 0.95, 0.93)
before a second loss at the softmax (0.774). **Dominant bottleneck:
`PERTURBATION_SUBSPACE`.**

**P&C's OOD signal comes mostly from base confidence, not disagreement.** Expected member
entropy (76.55 Near) slightly *exceeds* the ensemble predictive entropy (76.43), and mutual
information alone reaches only 73.97 — the disagreement term is not what is doing the work.

**LLLA-Kron, fitted with the official `laplace-torch`, collapses to the MAP.** Its ID-NLL
optimum is at prior precision 10⁹ with base agreement **1.0000**; its ID metrics are exactly
the base model's untempered numbers. Near 72.95 / Far 87.42 — *weaker* on Near than the
earlier hand-rolled KFAC (74.63), which found a genuine interior optimum at λ=10⁴. A
standard scalable LLLA therefore does **not** give materially stronger uncertainty here; it
gives less.

## 2. Common representation and setup

Everything is computed on the same examples: the 50,000-image ImageNet validation set and
all five OpenOOD sets (85,908 images), using the final normalized CLS representation
φ(x) ∈ R⁷⁶⁸ that the successful Mahalanobis baseline uses. CLS representations after
encoder blocks 8/9/10/11 and after the final LayerNorm were cached for A8 (2.5 GiB), and
the W1-input, post-GELU and block-output representations are exact functions of the existing
block-11 residual cache, so no backbone pass was repeated for them.

## 3. Geometry vs P&C, on identical examples (A2, A3)

| score | Near AUROC | Far AUROC | Near FPR95 | Far FPR95 |
|---|---|---|---|---|
| **M1 Mahalanobis (true labels)** | **78.82** | **92.55** | 66.36 | **30.23** |
| M2 Mahalanobis (predicted labels) | **78.86** | 92.47 | 66.38 | 30.32 |
| M3 unconditional Gaussian | 75.54 | 91.13 | 72.68 | 36.39 |
| M4 nearest centroid (no covariance) | 75.39 | 90.25 | 68.79 | 33.95 |
| whitened global distance | 75.54 | 91.13 | 72.68 | 36.39 |
| feature norm | 29.12 | 14.65 | 98.50 | 99.64 |
| P&C expected member entropy | 76.55 | 87.85 | 67.60 | 45.74 |
| **P&C predictive entropy** (headline) | 76.43 | 87.74 | 67.63 | 45.73 |
| P&C mutual information | 73.97 | 84.30 | 68.30 | 42.51 |
| P&C mean probability variance | 73.25 | 81.83 | 68.01 | 45.01 |
| P&C mean logit variance | 68.63 | 79.36 | 74.43 | 47.64 |
| post-correction residual ‖r_S‖ | 62.37 | 77.70 | 82.76 | 54.39 |
| ridge leverage h_{v,S} | 58.13 | 74.13 | 89.34 | 65.69 |
| hidden perturbation change ‖Δy‖ | 57.12 | 69.45 | 88.50 | 69.04 |
| output-visible change ‖Δz‖ | 55.19 | 67.20 | 90.85 | 71.47 |

Feature norm is *anti*-correlated with OOD (29.12 Near) — OOD inputs have **larger** CLS
norms here, so norm-based scores would need the opposite sign; it is reported as measured.

## 4. Is the advantage labels, covariance, or generic unsupportedness? (A4)

Decisively **not labels**: M2, which assigns each calibration image to the frozen ViT's own
predicted class, matches M1 to within 0.04 Near AUROC. The base model is 94.13 % accurate on
the calibration pool, so predicted and true labels mostly agree — but the point stands that
the detector needs no ground truth.

Decomposing the remaining 3.3 Near / 1.4 Far that M1 holds over M3:

- **class-conditional structure** contributes +3.28 Near, +1.42 Far (M1 − M3);
- **covariance whitening** contributes +3.43 Near, +2.30 Far (M1 − M4);
- **generic unsupportedness** already accounts for most of the absolute performance: M3, one
  global Gaussian with no class information at all, reaches 75.54 Near / 91.13 Far — which
  is **above P&C on Far** and within 1 point of it on Near.

`whitened_global` is identical to M3 by construction (both are the single-Gaussian
Mahalanobis distance) and is reported as a consistency check; the two agree exactly.

## 5. Where is the signal lost? (A5)

Adjacent-stage Spearman correlations along
Mahalanobis → ‖Δy‖ → ‖Δz‖ → ‖r_S‖ → logit variance → predictive entropy, pooled over ID and
all OOD:

| transition | Spearman |
|---|---|
| **Mahalanobis → hidden perturbation change** | **+0.634** |
| hidden → output-visible change | +0.988 |
| output-visible → post-correction residual | +0.947 |
| post-correction residual → logit variance | +0.935 |
| **logit variance → predictive entropy** | **+0.774** |

The three middle stages are essentially lossless: whatever the perturbation does to the
hidden activation survives the correction and the output projection almost perfectly. The
losses are at the two ends — entering the perturbation subspace (0.634) and leaving through
the softmax (0.774).

Correlation of each score with M1 directly, computed separately within ID and within pooled
OOD (a score that merely tracks the ID/OOD split would show a high pooled and low within-group
correlation, so this separates real agreement from group-level confounding):

| score | vs M1 within ID | vs M1 within OOD |
|---|---|---|
| M2 predicted-label Mahalanobis | +0.999 | +0.999 |
| M4 nearest centroid | +0.992 | +0.922 |
| M3 unconditional | +0.934 | +0.958 |
| P&C logit variance | +0.939 | +0.734 |
| P&C mutual information | +0.877 | +0.763 |
| post-correction residual | +0.868 | +0.661 |
| P&C predictive entropy | +0.780 | +0.831 |
| hidden perturbation change | +0.810 | +0.543 |
| output-visible change | +0.763 | +0.489 |
| feature norm | −0.944 | −0.839 |

M2's agreement with M1 is 0.999 in both groups — the label ablation is not a coincidence of
aggregation. The P&C mechanism quantities agree with the geometry far less on OOD (0.49–0.73)
than on ID (0.76–0.94), i.e. precisely where the detector has to work, the perturbation
response stops tracking unsupportedness.

**Classification: `PERTURBATION_SUBSPACE`.** The first transition is both the weakest link
and the one that determines what the rest of the pipeline can possibly see. A K = 20 random
subspace of a 2.36 M-dimensional weight space responds to only a fraction of the geometric
unsupportedness that a 768-dimensional covariance model measures directly.

An important corollary, visible in §3: the mechanism quantities are *worse* detectors than
the final entropy (55–62 vs 76.43 Near). So this is **not** a case of a strong geometric
signal being attenuated on the way out. P&C's OOD performance is largely inherited from the
base model's own confidence — expected member entropy (76.55) already matches the full
predictive entropy (76.43), and the disagreement-only score is weaker still (73.97). The
answer to "does P&C contain the Mahalanobis signal?" is **largely no**: the two methods are
detecting substantially different things.

## 6. LLLA-Kron (Part C)

Fitted with official `laplace-torch` 0.2.2.2 (curvlinops 2.0.1),
`subset_of_weights="last_layer"`, `hessian_structure="kron"`, `last_layer_name="head"`, on
the 32,768-image ID calibration pool via a frozen-feature wrapper (52.1 s, peak 4.42 GiB
GPU). Prior precision selected on **ID selection-pool NLL** over the §C4 grid, extended to
10¹⁰ because the optimum sat on the boundary.

| λ | 10⁻⁴ | 1 | 10² | 10³ | 10⁴ | 10⁶ | **10⁹** |
|---|---|---|---|---|---|---|---|
| ID NLL | 6.6105 | 5.0258 | 1.0336 | 0.3742 | 0.3281 | 0.3234 | **0.3234** |
| ID ECE | 0.9372 | 0.9312 | 0.5287 | 0.1615 | 0.1243 | 0.1204 | **0.1203** |

The NLL is monotone decreasing to a plateau: **there is no interior optimum**. The selected
posterior is fully degenerate — base agreement **1.0000**, ID top-1 81.068 %, NLL 0.8384 and
ECE 0.0560, which are exactly the base model's untempered values.

Results: Near AUROC **72.95**, Far **87.42** (SSB-hard 68.08, NINCO 77.81, iNaturalist
89.73, Textures 86.58, OpenImage-O 85.94).

**Predictive.** `laplace-torch`'s own GLM predictive materialises the last-layer Jacobian —
`(batch, 1000, 769000)`, an **11,718 GiB** tensor for a 1000-class head — so it cannot run
here at any batch size. The library's fitted Kronecker factors are used unchanged and the
functional variance is computed in closed form instead, exploiting J = φ̂ ⊗ I:

    a = Q_Aᵀφ,  s_k = Σ_j a_j²/(λ_{S,k}λ_{A,j} + δ),  var_c = Σ_k Q_S[c,k]² s_k  (+ bias)

gated against the library's own `functional_variance` on a small head where the explicit
Jacobian fits: **max relative error 2.1e-07**. The GLM/probit predictive and an MC link
approximation agree to Spearman 1.0000 on entropy with 1.0000 top-1 agreement (C5).

**Answer to E-Q8: no.** A standard LLLA-Kron does not give materially stronger uncertainty
than the hand-rolled KFAC already reported — it is 1.7 points *worse* on Near (72.95 vs
74.63) and 0.9 better on Far (87.42 vs 86.56), and it is worse than P&C on both.

The two fits differ in a way worth recording. The earlier version's ID-NLL sweep has a
genuine interior minimum at λ = 10⁴ (0.2060, against 0.2065 at 10³ and 0.2062 at 3·10⁴) with
a mildly non-degenerate posterior (base agreement 0.9987); the official fit's sweep is
monotone to a plateau and selects a posterior that has collapsed entirely (agreement
1.0000). Their selection NLLs are not directly comparable — the earlier one averages 20 MC
softmax samples, this one uses the GLM/probit predictive — but the *shape* difference is
real, and it comes from the Fisher factors and damping convention rather than from the
sampling. "Last-layer KFAC Laplace" is not a single well-defined number, and on this
backbone neither version is competitive with a Gaussian fitted directly to the features.

## 7. Main table (Part D)

| Method | ID Acc | ID NLL | Near AUROC | Far AUROC | Fit RAM | Fit VRAM | Fit time |
|---|---|---|---|---|---|---|---|
| Mahalanobis | 81.068 | 0.8482 | **78.82** | **92.55** | 4.8 GiB | 0.47 GiB | 0.3 s |
| LLLA-Kron | 81.068 | 0.8384 | 72.95 | 87.42 | 1.35 GiB | 4.42 GiB | 52.1 s |
| SCOD-linear | — | — | — | — | 1.05 GiB† | — | — |
| SCOD-FFN | — | — | — | — | 7.53 GiB† | — | — |
| SCOD-last-block | — | — | — | — | 10.77 GiB† | — | — |
| Uncorrected P&C | 80.619 | 0.9048 | 72.53 ± 0.30 | 84.40 ± 0.27 | 4.8 GiB | 2.16 GiB | 12.0 s |
| **P&C** (r=2, frozen) | 80.830 | **0.8045** | 76.49 ± 0.09 | 87.89 ± 0.12 | 4.8 GiB | 2.16 GiB | 12.0 s |

† sketch arithmetic only — **not run, not measured**. The three SCOD rows are empty because no
SCOD number was produced; see §10.

Appendix rows (frozen, from `imagenet_vit_baselines/`): MSP 73.52 / 86.04, ReAct+Energy
69.21 / 85.61, Energy 62.39 / 78.96, Laplace KFAC (hand-rolled) 74.63 / 86.56.

Diagnostic variants measured here, all sharing the base model's ID metrics:

| variant | Near AUROC | Far AUROC |
|---|---|---|
| Mahalanobis M2 (predicted labels, no ground truth) | 78.86 | 92.47 |
| Mahalanobis M3 (unconditional) | 75.54 | 91.13 |
| Mahalanobis M4 (nearest centroid) | 75.39 | 90.25 |
| P&C predictive entropy, seed 0 (this run) | 76.43 | 87.74 |

The seed-0 recomputation agrees with the frozen 5-seed headline (76.49 ± 0.09 / 87.89 ± 0.12),
confirming that nothing in this directory perturbed the P&C configuration.

## 8. The nine required questions (Part E)

**1. Is Mahalanobis's advantage mostly class supervision, covariance geometry, or generic
unsupportedness?** Mostly **generic unsupportedness**, with a real but secondary contribution
from covariance geometry, and essentially **none** from class supervision. A single
unconditional Gaussian (M3) already reaches 75.54 / 91.13 — 96 % of M1's Near and 98 % of its
Far. Covariance whitening adds +3.4 Near over nearest-centroid; class-conditioning adds +3.3
Near over unconditional; ground-truth labels add +0.0 (M2 = 78.86 vs M1 = 78.82), inside ID
and inside OOD alike (Spearman 0.999 both).

**2. Does P&C contain a Mahalanobis-like geometric signal before it becomes predictive
entropy?** **Only weakly.** Its most upstream mechanism quantity, the hidden perturbation
change, correlates 0.634 with M1 pooled and only **0.543 within OOD**, and on its own reaches
57.12 Near AUROC. There is no strong geometric signal sitting inside P&C waiting to be
extracted.

**3. At what stage is the signal attenuated?** **`PERTURBATION_SUBSPACE`**, decisively. Entry
into the perturbation response costs the most (0.634); correction (0.947), output projection
(0.988) and the residual→logit-variance step (0.935) are near-lossless; the softmax costs a
second, smaller amount (0.774). A secondary observation: because the mechanism quantities
score *worse* than the final entropy (55–62 vs 76.43 Near), P&C's OOD performance is not an
attenuated version of the geometric signal at all — it is largely inherited base confidence
(expected member entropy 76.55 ≥ ensemble entropy 76.43; disagreement-only MI just 73.97).

**4. Does increasing P&C rank K close the gap?** **Not determined — A7 was not run.** §5 makes
this the highest-value remaining experiment, since Q3's answer points squarely at the
subspace. It requires per-K ID-only scale/ridge readjustment to hold the 0.5-pp budget and
realized ‖ΔW₁‖_F/‖W₁‖_F normalization so that larger K is not simply more perturbation energy.

**5. Does Mahalanobis detect points on which all P&C members extrapolate similarly?**
**Not determined — A6 was not run.** The within-OOD correlations (0.49–0.73) say the two
disagree substantially on OOD, so such points very likely exist, but this was not measured.
All per-example scores needed are saved in `predictions/geometry_scores_*.npz`, making A6 a
pure post-processing step.

**6. Can legitimate SCOD-LL variants run within the machine's memory budget?** **By
arithmetic, yes for all three declared scopes; not demonstrated.** Sketch sizes are 1.05 /
7.53 / 10.77 GiB at the paper's own k=30, T=184, against 24.6 GiB of system RAM — and 0.44 /
3.11 / 4.45 GiB at the k=12, T=76 fallback. Only full-network SCOD (118.68 GiB) is genuinely
out of reach. No sketch was built, so this is a feasibility argument, not a result.

**7. How does SCOD performance change from head → final FFN → final block?** **Unknown — no
SCOD scores were produced.** Nothing is claimed.

**8. Does a standard LLLA-Kron give materially stronger uncertainty than the earlier custom
KFAC?** **No** — see §6. It is 1.7 Near AUROC worse (72.95 vs 74.63), 0.9 Far better, and
worse than P&C on both. Its selected posterior is fully degenerate (base agreement 1.0000).

**9. Which method is strongest for pure OOD ranking, and which among methods producing
predictive distributions?** For pure OOD ranking, **Mahalanobis**, by 2.3 Near / 4.7 Far over
the next best — and, per Q1, without needing any labels. Among methods that produce a
predictive distribution, **P&C**: it has the best ID NLL in the table (0.8045 vs 0.8384 for
LLLA-Kron and 0.8482 for the base model) and beats LLLA-Kron on both OOD aggregates.

## 9. Not run in this session

- **A6** (disagreement-case analysis of high-Mahalanobis/low-P&C sets and vice versa). The
  per-example scores needed for it are saved in `predictions/geometry_scores_*.npz`, so this
  is now a pure analysis step on existing artefacts.
- **A7** (K ∈ {5, 20, 40, 80} rank sweep). This is the natural follow-up to §5's
  `PERTURBATION_SUBSPACE` verdict and is the single most valuable remaining experiment: it
  directly tests whether richer subspace coverage closes the gap. It needs a per-K ID-only
  scale/ridge readjustment to hold the 0.5-pp budget, so it is a real experiment rather than
  a re-analysis.
- **A8** (layerwise Mahalanobis at blocks 8/9/10/11). The val and OOD layer caches are built
  (2.5 GiB); only the 32,768-image *training-pool* layer pass is missing, which stalled on
  parquet row-group thrashing and was stopped. Fitting the layerwise Gaussians on the
  evaluation split instead would leak, so it was left undone rather than done wrongly.
- **Part B (SCOD-LL)**. See §10.

## 10. SCOD status and a correction to the earlier conclusion

**The earlier `SCOD_NOT_TRACTABLE_AT_VIT_SCALE` verdict was too broad and is corrected
here.** It was accurate for *full-network* SCOD using this repository's JAX port, but two of
its four stated blockers do not apply to last-layer SCOD:

1. **Categorical likelihood is supported upstream.** The official implementation
   (`StanfordASL/SCOD`, commit `6a569734d3e246e25c53c0dff97e4e83690087d4`) contains
   `Categorical` and `CategoricalLogit` distribution families in `nn_ood/distributions.py`.
   The absence was in *this repo's* MuJoCo-only port, not in SCOD.
2. **No second derivatives are needed.** With sampled-label score gradients
   `g = ∇_w log p(y|x)`, `E_y[ggᵀ] = F_w(x)` requires only ordinary backward passes, which
   sidesteps the fused-attention forward-AD and double-backward failures entirely.

The memory argument stands only for full-network scope. At the paper's own CIFAR SCOD-LL
setting (k = 30, T = 184 = 6k+4, from `experiments/cifar10/config.py`), the sketch
requirement per scope is:

| scope | parameters | sketch at k=30, T=184 | at k=12, T=76 |
|---|---|---|---|
| SCOD-linear (`heads.head`) | 769,000 | 1.05 GiB | 0.44 GiB |
| SCOD-FFN (W1, W2, head) | 5,491,432 | 7.53 GiB | 3.11 GiB |
| SCOD-last-block (block 11 + `encoder.ln` + head) | 7,858,408 | 10.77 GiB | 4.45 GiB |
| full network | 86,567,656 | 118.68 GiB | 49.02 GiB |

Sketch = Ω and Y, each p × T float32. All three restricted scopes fit in 24.6 GiB of system
RAM at k=30, and SCOD-linear also fits in the 12 GB of GPU. Only the full-network scope is out
of reach — which is what the original verdict should have said. **These are arithmetic, not
measurements** — the
sketch was not built and no SCOD scores were produced, so no SCOD row appears in any table
here. The upstream repository is cloned and the scoping decided; the implementation work
(categorical MC-Fisher factor, parameter-scope masking, sketch driver, the q ∈ {1,2,4,8}
convergence study of B6) remains.

## 11. Recommended next steps

1. **Run A7.** §5 identifies the perturbation subspace as the bottleneck; the K sweep is the
   direct test, and if performance saturates by K = 20 the explanation lies elsewhere.
2. **Run A6** — it is now pure analysis on saved per-example scores, and it tests whether
   Mahalanobis and P&C flag genuinely different inputs (§5 suggests they do).
3. **Complete Part B** at the three declared scopes, reporting each separately.
4. **For the manuscript**, the defensible framing from this diagnostic is that P&C and
   feature-density methods detect different things — P&C's OOD signal is largely inherited
   base confidence rather than perturbation-response geometry, which is worth saying plainly
   rather than positioning P&C as a better detector of the same quantity.
