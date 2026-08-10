# Subnetwork Linearized Laplace (SLL) on CIFAR-10 — Report

**Headline (3 seeds, OpenOOD v1.5):** SLL-Backbone is **not competitive** on this frozen
PreActResNet-18. The *faithful* variance-based subnetwork selection **degenerates to the base
classifier** (≡ MSP); a principled *non-degenerate* selection (predictive-variance contribution,
S=2048) produces a working posterior but its predictive-entropy OOD score **underperforms even
MSP**. This is a robust negative result and a useful rebuttal point: a natural frozen-checkpoint
Laplace baseline does not rival P&C (or simple baselines) here.

Framework note: the SLL spec is written for PyTorch; this repo is JAX/Flax nnx, so the
implementation is the faithful JAX analogue (jacrev selected-parameter Jacobians, frozen BN via
`use_running_average=True`). Checkpoint SHA-256 hashes **match the SCOD manifest** for all 3 seeds.

## 1. Final comparison table (mean ± sample-std, 3 seeds)

| Method | Acc | Near AUROC | Near FPR95 | Far AUROC | Far FPR95 |
|---|---:|---:|---:|---:|---:|
| LLLA (n=50) | 95.77 | 88.97 | 54.10 | 93.04 | 28.40 |
| SCOD-1024 | 95.74 | 89.69 | 39.40 | 92.56 | 21.41 |
| **SLL-Backbone (S=2048)** | **95.46±0.09** | **79.05±2.80** | **80.70±5.21** | **86.44±3.04** | **56.08±8.79** |
| P&C s3b0 (M=50) | 95.59 | 91.55 | 33.08 | 95.09 | 18.15 |
| Standard Ensemble (n=5) | 96.56 | 91.10 | 40.40 | 94.63 | 19.50 |

ID (SLL-Backbone): Acc 95.46±0.09, NLL 0.183±0.014, ECE 0.010±0.003, Brier 0.074±0.003. Per-dataset
AUROC/FPR95 in `tables/sll_per_dataset.md`. SLL is the weakest row on every OOD metric, with high
seed variance (±2.8 Near AUROC).

## 2. Method

Full-covariance linearized Laplace over a selected subnetwork w_S of internal **backbone** weights
(fc excluded), all other parameters fixed at the checkpoint. Posterior
`q(w_S)=N(ŵ_S,(G_S+λ_S I)^{-1})`, `G_S=Σ_n J_{S,n}^T (diag(p_n)-p_np_n^T) J_{S,n}` (categorical GGN).
Predictions use the **locally linearized** network `f_lin(x)=f(x,ŵ)+J_S(x)(w_S-ŵ_S)`; the primary
score is predictive entropy of the **probit** rule `z̃_c = z_c/√(1+(π/8)v_c)`, `v_c=Var_m[J_S(x)Δw_m]`,
M=50 posterior samples `Δw_m=U diag((d+λ_S)^{-1/2})ε_m`. MC posterior-predictive
entropy / mutual information are saved as secondary diagnostics. Temperature and λ_S are fit on
ID-val only.

## 3. Subnetwork selection — the central finding

**The faithful SLL variance criterion degenerates here.** Ranking by largest marginal variance
(= smallest diagonal GGN) selects the *least data-constrained* backbone weights. On this network
those coincide with **predictively-inert** weights (near-zero Jacobian, `stage4.1.conv2` — the last
residual block). Consequences, measured:
- 5.3% of backbone candidates have *exactly* zero curvature; the top-S smallest are all zero/denormal.
- The selected posterior is **prior-dominated** (eigenvalues ~1e-42) and the probit adjustment leaves
  logits **bit-identical** to the base classifier (`max|Δ logits| = 0.0`), so SLL ≡ MSP (Near AUROC
  ≈ 88 on a CIFAR-100 probe). Identical results at S=512/1024/2048 confirm the degeneracy.

Because pure marginal variance is only a *diagonal approximation* to SLL's Wasserstein-predictive
selection objective, the principled fix (ID-only) is to rank by the **linearized predictive-variance
contribution** of each weight:

    score_j = R_j / (G_diag,j + λ0),   R_j = raw-logit Jacobian energy (influence),
              1/(G_diag,j+λ0) = marginal variance,   λ0 = median positive G_diag.

This selects weights that are both **uncertain and predictively influential** (a mix of
`stage4.1.conv1`, `stage4.0.conv2`, downsample — 4 leaves, 0.018% of candidates). It yields a
**non-degenerate** posterior (GGN eigenvalues 1.7–6.2, predictions that move) and is what the
reported SLL-Backbone uses. An ID-only comparison of selection rules (no OOD): smallest-G_diag
(variance) → inert (Δlogits=0); largest-G_diag (most influential) → moves predictions but low
variance; predictive-variance contribution → best ID-val NLL and non-degenerate. S=2048 chosen by
ID-val NLL (0.169 vs 0.177/0.180 at 512/1024).

**Even non-degenerate, SLL underperforms.** Probit predictive entropy gives Near ~79 (< MSP 88); the
MC epistemic signals (mc-entropy, mutual information) are **near chance** (~55) — the linearized
posterior injects OOD-*irrelevant* noise. So no faithful/principled ID-only configuration makes SLL
competitive on this benchmark.

## 4. Protocol & provenance
- Checkpoints seeds 0,1,2; SHA-256 verified == SCOD manifest. Split seed 99 (45000 train / 5000 val);
  1024 calib pool = `RandomState(seed).choice(45000,1024,replace=False)` (P&C/SCOD indices).
- Normalization mean/std [0.4914,0.4822,0.4465]/[0.247,0.2435,0.2616]; deterministic eval, frozen BN.
- Near {CIFAR-100, TinyImageNet}, Far {MNIST, SVHN, Textures, Places365}; macro-mean; ± sample std.
- **No OOD** entered subnetwork selection, curvature, prior, temperature, or stopping (gate 9).

## 5. Validation gates (`validation/validation_gates.json`)
1 frozen parity PASS · 2 Fisher factor PASS (1e-15/5e-7) · 3 selected-Jacobian PASS (exact 0.0 f64)
· 4 finite-difference PASS (converges) · 5 streamed-vs-explicit GGN PASS (8.6e-4) · 6 posterior
covariance PASS (1.5% MC) · 7 predictive covariance PASS (2.4% MC) · 8 last-layer sanity NOT_RUN
(LLLA uses a different last-layer approximation; deferred) · 9 OOD isolation PASS.

## 6. Efficiency
Post-hoc construction ≈ 86 s/seed (diagonal GGN selection + dense S×S GGN + eigendecomposition +
ID-only prior/temperature). Serialized posterior: eigensystem + S×50 perturbation matrix (small,
S=2048). Inference requires a **selected-parameter Jacobian** per example (not a plain forward):
~8.7 ms/image at the eval batching used (10k-image dataset ≈ 87 s). Full covariance is over the
S=2048 subnetwork only; no dense all-network covariance is ever formed.

## 7. Required statements
- **Subnetwork size:** S=2048 weights = **0.018%** of the 11,167,040 backbone candidate parameters.
- **Full covariance** was used within the selected subnetwork (dense (G_S+λ_S I)^{-1}, S×S).
- **All other parameters fixed** at their checkpoint values (fc and non-selected backbone weights).
- **Prediction used the linearized network** (probit rule), not nonlinear sampled-weight networks.
- **ID data only** selected the subnetwork, prior precision, and temperature.
- **Resource-adapted**, not an exact reproduction of the original 42k-weight SLL ResNet experiment
  (8 GB VRAM). Full covariance retained (never diagonal).
- **Costs:** ~86 s construction/seed; per-example test-time Jacobian (~8.7 ms/image).
- **Numerical compromise / deviation:** the faithful marginal-variance selection degenerates on this
  net; we use the predictive-variance-contribution selection (ID-only) to obtain a non-degenerate
  full-covariance SLL. This deviation is documented and motivated above.

## 8. Limitations & interpretation
SLL is a legitimate frozen-checkpoint, post-hoc, 1×-training comparator (like P&C and SCOD), but on
CIFAR-10 / PreActResNet-18 it does not produce a useful OOD signal: the variance criterion is
degenerate and principled non-degenerate variants underperform MSP with a near-chance epistemic
term. This strengthens the rebuttal: P&C's advantage is not an artifact of comparing only against
weak or last-layer methods — a full-covariance subnetwork Laplace, given the same checkpoints and ID
budget, is markedly weaker.
