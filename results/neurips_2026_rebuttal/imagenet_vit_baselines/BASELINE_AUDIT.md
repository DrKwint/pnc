# Baseline implementation audit

Written **before** implementing anything (spec §6). The goal is to port *the same method*,
not something with the same name. Where an existing implementation cannot transfer
faithfully to ViT-B/16, that is recorded and the baseline is stopped rather than replaced
with a convenient substitute.

Every existing implementation in this repository is **JAX/Flax**; the ViT experiment is
PyTorch. "Port" therefore always means re-expressing the *same algorithm and
hyperparameters* in torch, and each entry below states exactly what that algorithm is.

---

## Summary

| Method | Existing source | Verdict |
|---|---|---|
| MSP | `pnc_core/openood_eval.py`, `experiments/imagenet_vit_pnc/full_baselines.py` | **Reuse** (already computed, identical inputs) |
| Energy | same | **Reuse** |
| ReAct + Energy | same | **Reuse** |
| Mahalanobis | `pnc_core/openood_eval.py::_fit_mahalanobis` | **Port** — direct, ID-only, single-layer |
| LLLA (dense) | `pnc_core/ensembles.py::LLLAEnsemble` + `cifar_tasks.py:1917` | **MEMORY_INFEASIBLE** at ViT scale — see below |
| Laplace (KFAC) | `pnc_core/ensembles.py::LaplaceEnsemble` + `pnc_core/laplace.py::compute_kfac_factors` | **Port** — this is the scalable last-layer Laplace |
| SCOD | `experiments/posthoc/scod_*.py` | **Regression-only + sketch infeasible** — see below |
| SWAG | `pnc_core/ensembles.py::SWAGEnsemble` | METHOD_REQUIRES_RETRAINING (SGD trajectory) |
| Subspace Inference | `pnc_core/ensembles.py::SubspaceInferenceEnsemble` | METHOD_REQUIRES_RETRAINING (needs SWAG trajectory + PCA) |
| MC Dropout | `pnc_core/ensembles.py::MCDropoutEnsemble` | **NOT_APPLICABLE** — pinned checkpoint has every dropout p = 0.0 |
| Epinet | `pnc_core/ensembles.py::EpinetWithPrior` | METHOD_REQUIRES_RETRAINING (learned epistemic head) |
| Deep Ensemble | — | Not run by instruction (§20) |

---

## MSP / Energy / ReAct + Energy

* **Source**: `experiments/imagenet_vit_pnc/full_baselines.py`, itself matching the
  conventions in `pnc_core/openood_eval.py`.
* **Parameters acted on**: none — pure post-hoc scores on the frozen checkpoint.
* **Data required**: none for MSP/Energy. ReAct needs one ID-derived clipping threshold.
* **Hyperparameters**: ReAct clipping percentile p90, taken on the **ID training-derived
  temperature pool** (threshold 0.6918). MSP and Energy have none.
* **Score**: MSP `-max softmax`; Energy `-logsumexp(logits)`; ReAct+Energy = Energy after
  clipping the penultimate CLS feature at the ID p90.
* **Samples/image**: 1.
* **Extra optimization**: none.
* **Verdict**: reuse the saved scores verbatim. Inputs, preprocessing, checkpoint and
  scoring code are byte-for-byte identical, and §8/§9 explicitly say not to recompute
  merely to produce new numbers.

## Mahalanobis

* **Source**: `pnc_core/openood_eval.py::_fit_mahalanobis` / `_mahalanobis_scores`.
* **Variant**: **single-layer, class-conditional Gaussian with a shared covariance.** Not
  multi-layer, and *not* the OOD-tuned logistic-regression combination from the original
  paper — the repo's canonical version has no OOD-fitted component, which is exactly the
  form §11 asks for.
* **Fit**: per-class means over the feature set; `cov = centeredᵀ centered / N + 1e-6·I`;
  `precision = inv(cov)`. Labels required.
* **Score**: minimum squared Mahalanobis distance across classes (higher = more OOD).
* **Hyperparameters**: the fixed `1e-6` covariance ridge. Nothing is searched, so nothing
  can leak.
* **ViT transfer**: the CIFAR version uses the penultimate feature; the ViT analogue is the
  final normalized CLS representation φ(x) ∈ R⁷⁶⁸ (`encoder.ln(...)[:, 0]`) — the same
  representation ReAct clips and the classifier consumes. Class means (1000, 768) and a
  768×768 precision are trivial to fit and store.
* **Verdict**: **port directly.** Fit on the 32,768-image training calibration pool.

## LLLA — dense last-layer Laplace

* **Source**: `pnc_core/ensembles.py::LLLAEnsemble`, GGN built in
  `pnc_core/cifar_tasks.py:1917`.
* **Exact algorithm**: features φ augmented with a bias column, `x̂ ∈ R^{D+1}`; GGN
  `G = Σ_n (x̂_n x̂_nᵀ) ⊗ H_n` with `H_n = diag(p_n) − p_n p_nᵀ`, assembled as a dense
  `((D+1)K, (D+1)K)` matrix; `precision = G + prior_precision·I`;
  `covariance = inv(precision)`; sampling via a dense Cholesky of the covariance.
* **Why it does not transfer**: for CIFAR-10, `(512+1)·10 = 5,130` parameters, so the dense
  covariance is 5,130² ≈ 26 M entries — 105 MB, fine. For ViT-B/16 the head is
  768 → 1000, so `(768+1)·1000 = 769,000` parameters and the dense covariance is
  **769,000² = 5.91 × 10¹¹ entries = 2.37 TB in float32** (4.7 TB in float64, which is what
  the CIFAR code uses for the inverse). The GGN itself is the same size. This is not a
  tight-hardware problem; it is four orders of magnitude beyond the machine.
* **Verdict**: `MEMORY_INFEASIBLE`. Reported with the arithmetic above rather than silently
  swapped for a cheaper covariance — §12 anticipates exactly this ("a literal dense
  posterior over every classifier parameter is not practical") and §36 forbids inventing a
  ViT-specific variant and calling it the paper's baseline.

## Laplace — KFAC (the feasible last-layer Laplace)

* **Source**: `pnc_core/ensembles.py::LaplaceEnsemble` +
  `pnc_core/laplace.py::compute_kfac_factors`. This is a **distinct existing baseline** in
  the repo (used for MuJoCo/UCI/MNIST), not a variant invented here — which resolves §15
  in the opposite direction to the redundancy case: at ViT scale the KFAC form is the one
  that runs, and the dense LLLA form is the one that does not.
* **Exact algorithm**, per layer, transcribed from the source:
  * factors `A` (input side, `(D+1)×(D+1)`) and `S` (output side, `K×K`), each ridged by
    `1e-6·I`, then eigendecomposed by SVD with eigenvalues clamped at 0;
  * sample `Z ~ N(0,1)` shaped like `W_full = [W_map; b_map]`;
  * `eig_matrix = N · outer(eig_A, eig_S) + λ`, `std = 1/√eig_matrix`;
  * `ΔW = U_A (Z ⊙ std) U_Sᵀ`, `W_new = W_full + ΔW`.
* **Classification Fisher**: `compute_kfac_factors(..., is_classification=True)` uses the
  **MC Fisher** — `y_sampled ~ Categorical(p)`, `d_pre = p − onehot(y_sampled)` — with
  `A = E[â âᵀ]` and `S = E[d_pre d_preᵀ]`. Ported exactly.
* **ViT transfer**: applied to the classifier head only (`heads.head`, 768 → 1000), with the
  backbone frozen. `A` is 769×769 and `S` is 1000×1000 — 2.4 MB and 4 MB. Fully feasible.
* **Hyperparameters**: `prior_precision` (λ) and `data_size` (N). Selected by **ID
  validation NLL only**, on the log grid of §12.
* **Samples/image**: 20, matching P&C's M = 20 (§23).

## SCOD

* **Source**: `experiments/posthoc/scod_adapter.py`, `scod_distribution.py`,
  `run_scod.py`; config `configs/mujoco_posthoc.yaml`.
* **Two independent blockers**, both recorded rather than worked around:
  1. **The implementation is regression-only.** `scod_distribution.py` implements SCOD
     Case A (fixed diagonal noise) and Case B (heteroscedastic diagonal Gaussian) for
     `models.ProbabilisticRegressionModel`, deriving the Fisher factor `L` from a Gaussian
     likelihood. There is **no categorical/softmax likelihood** anywhere in the repo's SCOD
     code, and the only SCOD results are under `results/posthoc_mujoco/`. There is no
     CIFAR SCOD to port. Supplying a categorical Fisher would be new method code, not a
     port.
  2. **The parameter-space sketch does not fit.** SCOD sketches the dataset Fisher over the
     parameters it differentiates through — the whole network. The repo's own configuration
     is `num_eigs_max: 100`, `num_samples: 604` (= 6·100 + 4). For ViT-B/16,
     P = 86,567,656, so the Nyström test matrix Ω is `P × 604 × 4 B = 209 GB`, and the
     sketch `Y = FΩ` is another **209 GB** — against 12 GiB VRAM and 24.6 GiB system RAM.
     Even at `num_eigs = 10` (`num_samples = 64`) each factor is 22 GB, still beyond system
     RAM, and a rank-10 sketch of an 86.6 M-dimensional Fisher is not the same method.
* **Measured preflight** (§13 requires evidence, not arithmetic alone): see
  `metrics/scod_preflight.json` and §8 of the report.
* **Verdict**: `SCOD_NOT_TRACTABLE_AT_VIT_SCALE`. Per §13 it is **not** replaced by a
  head-only approximation called SCOD.

## SWAG

* **Source**: `pnc_core/ensembles.py::SWAGEnsemble` — low-rank-plus-diagonal sampling with
  BN refresh, consuming `swag_mean`, `swag_var`, `swag_cov_mat_sqrt` collected along an
  **SGD trajectory**.
* **Blocker**: there is no SGD trajectory for a pretrained torchvision ViT, and producing a
  faithful one means fine-tuning ImageNet — days of TITAN X time. §16 forbids substituting
  head-only or final-block fine-tuning and calling it full-model SWAG.
* **Verdict**: `METHOD_REQUIRES_RETRAINING`; category B, not run. A measured per-epoch
  estimate is in the report.

## Subspace Inference

* **Source**: `pnc_core/ensembles.py::SubspaceInferenceEnsemble` — takes `swag_mean` and
  `pca_components`, i.e. **PCA directions derived from the SWAG SGD trajectory**, then
  samples within that subspace (ESS by default).
* **Blocker**: strictly downstream of SWAG. §17 forbids substituting random directions.
* **Verdict**: `METHOD_REQUIRES_RETRAINING`; category B, not run.

## MC Dropout

* **Source**: `pnc_core/ensembles.py::MCDropoutEnsemble` — repeated stochastic forwards
  with `deterministic=False`, using the checkpoint's own trained dropout rates.
* **Measured**: the pinned `ViT_B_16_Weights.IMAGENET1K_V1` has **37 `nn.Dropout` modules,
  every one with p = 0.0**, and `MultiheadAttention` dropout 0.0. Stochastic forwards are
  therefore identical to the deterministic forward.
* **Verdict**: `MC_DROPOUT_NOT_APPLICABLE — pinned pretrained ViT uses zero dropout.`
  §18 explicitly prefers this over injecting dropout the checkpoint was never trained with.

## Epinet

* **Source**: `pnc_core/ensembles.py::EpinetWithPrior` / `EpinetEnsemble` — a trained
  epistemic head with a fixed prior network, over a frozen base.
* **Blocker**: the epinet itself is *trained*; porting means designing and training an
  ImageNet-scale epinet head, and choosing its architecture, index dimension and training
  schedule. That is method design, not a port.
* **Verdict**: `EPINET_NOT_PORTED_METHOD_CHANGE`; category C, not run.

## Deep Ensemble

Not run by instruction (§20): requires independent ImageNet-scale training, is not a
frozen-checkpoint construction, and the smaller-scale experiments already establish the
comparison. No metrics are imported from the literature.
