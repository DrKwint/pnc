# CORE RESULTS — matched post-hoc uncertainty baselines on ImageNet ViT-B/16

Written at the §31 stop point: MSP, Energy, ReAct+Energy, Mahalanobis, Laplace (KFAC), a
measured SCOD infeasibility result, frozen P&C and the matched uncorrected ablation are all
complete. Secondary methods (SWAG, Subspace, MC Dropout, Epinet) are classified in
`BASELINE_AUDIT.md` and none is runnable — reasons below.

**Verdict: `P&C weaker than established matched alternatives`** — specifically weaker than
Mahalanobis, while beating every other matched method tested.

## Frozen-checkpoint comparison

All rows use the identical checkpoint, preprocessing, ImageNet split, OOD sets and
evaluator. The common harness reproduces the completed experiment **bit-exactly** on 9/9
parity checks (`provenance/parity_gate.json`).

| Method | Extra opt? | Cal N | ID Acc | ID NLL | ID ECE | Near AUROC | Near FPR95 | Far AUROC | Far FPR95 | Fit | Storage | Evals/img |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| MSP | no | — | 81.068 | 0.8482 | 0.0913 | 73.52 | 81.84 | 86.04 | 51.74 | 0 s | 0 | 1 |
| Energy | no | — | 81.068 | 0.8482 | 0.0913 | 62.39 | 93.16 | 78.96 | 85.29 | 0 s | 0 | 1 |
| ReAct + Energy | no | 8,192 | 81.068 | 0.8482 | 0.0913 | 69.21 | 84.23 | 85.61 | 53.90 | 0.1 s | 0 | 1 |
| **Mahalanobis** | no | 32,768 | 81.068 | 0.8482 | 0.0913 | **78.82** | **66.36** | **92.55** | **30.23** | **0.3 s** | 5.2 MiB | **1** |
| Laplace (KFAC) | no | 32,768 | 81.014 | 0.8446 | 0.0901 | 74.63 | 75.03 | 86.56 | 48.79 | 1.1 s | 6.1 MiB | 20 |
| Uncorrected perturb. | no | 32,768 | 80.619 | 0.9048 | 0.0466 | 72.53 ± 0.30 | 71.66 | 84.40 ± 0.27 | 46.51 | 12 s | 0 | 20 |
| **P&C** (r=2, λ=1000) | no | 32,768 | 80.830 | **0.8045** | 0.0542 | 76.49 ± 0.09 | 67.65 | 87.89 ± 0.12 | 44.97 | 12 s | 167 MiB | 20 |
| SCOD | — | — | — | — | — | — | — | — | — | — | — | — |
| LLLA (dense) | — | — | — | — | — | — | — | — | — | — | — | — |

SCOD: `SCOD_NOT_TRACTABLE_AT_VIT_SCALE`. LLLA (dense): `MEMORY_INFEASIBLE`. Both are
measured results, not omissions — see below.

## Paired bootstrap (2,000 replicates, seed 20260815, ID subsampled to 10,000)

Aggregate AUROC difference, P&C minus comparator; positive means P&C better.

| Comparison | Δ Near AUROC | 95 % CI | Δ Far AUROC | 95 % CI |
|---|---|---|---|---|
| P&C − Mahalanobis | **−2.39** | [−2.68, −2.09] | **−4.81** | [−5.08, −4.57] |
| P&C − Laplace (KFAC) | +1.80 | [+1.57, +2.03] | +1.18 | [+1.02, +1.34] |
| P&C − MSP | +2.91 | [+2.65, +3.19] | +1.70 | [+1.52, +1.88] |
| P&C − Energy | +14.04 | [+13.46, +14.61] | +8.78 | [+8.38, +9.16] |

Every interval excludes zero. P&C's advantage over MSP, Energy and the Laplace posterior is
real; so is Mahalanobis's advantage over P&C.

## What this establishes

**1. Mahalanobis beats P&C on this benchmark, and it is not close.** It wins on **all five**
OOD datasets — SSB-hard 71.33 vs 70.77, NINCO 86.31 vs 82.21, iNaturalist 95.86 vs 89.69,
Textures 89.54 vs 86.22, OpenImage-O 92.26 vs 87.77 — and on both aggregates, with the
largest margin on Far FPR95 (30.23 vs 44.97, a 14.7 pp gap). It also costs far less:
**0.3 s to fit, 5.2 MiB of state, and one forward pass per image** against P&C's 12 s,
167 MiB and 20 member evaluations. This is a genuine negative result for P&C on
ImageNet-scale OOD detection and is reported as such.

**2. P&C beats every other matched method**, including the approximate-posterior baseline.
Against MSP it is +2.91 Near / +1.70 Far AUROC; against Laplace (KFAC) +1.80 / +1.18;
against Energy +14.04 / +8.78. Energy, strong on CNNs, is markedly worse than MSP on this
ViT (62.39 vs 73.52 Near).

**3. The two methods answer different questions, which the table alone hides.** Mahalanobis
is a *pure OOD score*: it leaves predictions untouched, so its ID row is simply the base
model's, and it supplies no predictive distribution or epistemic uncertainty for the
prediction itself. P&C returns a full predictive distribution and is the **best-calibrated
row in the table** — NLL 0.8045 against the base model's 0.8482, ECE 0.0542 against 0.0913.
A practitioner who needs only "is this input OOD?" should use Mahalanobis here. One who
needs calibrated predictive uncertainty cannot use it at all.

**4. Approximate last-layer posteriors are weak at this scale.** Laplace (KFAC) is only
+1.11 Near AUROC over MSP. Its ID-optimal prior precision (λ = 10⁴, selected on ID NLL over
a grid extended to 10⁸) leaves the posterior nearly degenerate — base agreement 0.9987 —
so it behaves close to the deterministic model. Smaller λ collapses accuracy catastrophically
(top-1 1.2 % at λ = 10⁻⁴).

## Methods that could not be run, with evidence

**SCOD — `SCOD_NOT_TRACTABLE_AT_VIT_SCALE`.** Four independent blockers, measured:

1. the repo's SCOD implements only Gaussian Case A/B for MuJoCo regression — there is **no
   categorical likelihood to port**, and no CIFAR SCOD exists;
2. the Nyström sketch at the repo's own `num_samples = 604` needs
   **390 GiB** (Ω plus Y, at P = 86,567,656) against 12 GiB VRAM and 24.6 GiB RAM; even at
   `num_eigs = 10` it needs 41 GiB;
3. one Fisher matvec over the 32,768-image calibration pool takes **27.8 min**
   (measured 50.9 ms/image at N = 1,024, peak 5.08 GiB), so the full 604-matvec sketch is
   **≈ 280 GPU-hours**;
4. torchvision's fused attention supports neither forward-mode AD nor a second derivative;
   the measurement above required forcing the MATH SDPA backend and computing the JVP by a
   double-VJP identity.

Per §13 SCOD is **not** replaced by a head-only approximation.

**LLLA (dense) — `MEMORY_INFEASIBLE`.** The repo's `LLLAEnsemble` builds a dense GGN and
covariance over all last-layer parameters. For a 768 → 1000 head that is 769,000 parameters
and a 769,000² covariance = **2.37 TB** in float32 (the CIFAR code inverts in float64, so
4.7 TB). The Kronecker-factored `LaplaceEnsemble`, which is a *separate existing baseline*
in this repository, is what runs at this scale and is reported as "Laplace (KFAC)".

**MC Dropout — `MC_DROPOUT_NOT_APPLICABLE`.** The pinned checkpoint has 37 `nn.Dropout`
modules, all with p = 0.0, and attention dropout 0.0 (verified). Stochastic forwards equal
the deterministic forward. Injecting dropout is forbidden by §18.

**SWAG / Subspace Inference — `METHOD_REQUIRES_RETRAINING`.** Both consume an SGD
trajectory; Subspace additionally needs SWAG's PCA directions. Producing a faithful
trajectory means fine-tuning ImageNet, and §16/§17 forbid head-only substitutes.

**Epinet — `EPINET_NOT_PORTED_METHOD_CHANGE`.** The epinet is trained; porting means
designing and training an ImageNet-scale epistemic head.

**Deep Ensemble** — not run by instruction (§20); no literature number imported.

## Caveat on this comparison

Mahalanobis is fitted on the same 32,768 training images P&C uses for its correction, and
uses their **labels**, which P&C's correction does not (it regresses onto the base model's
own outputs). That is a legitimate difference in information used, not an unfairness — both
are ID-only and neither touches OOD data — but it is worth stating when reporting the gap.
