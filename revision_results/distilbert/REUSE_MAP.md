# REUSE_MAP — Banking77 DistilBERT P&C

How the transformer experiment reuses the existing JAX P&C implementation. The rule:
reuse unchanged first, thin transformer adapter second, never a second P&C solver.

| Component | Existing path / symbol | Shapes | Reused? | Adapter work |
|---|---|---|---|---|
| **Affine correction (ridge)** | `experiments/scripts/pnc_theory/linalg.py::ridge_solve(Xv, target, lam, w_prior)` | Xv (N, p+1), target (N, d_out), Theta (p+1, d_out) | **Unchanged** | Feed augmented post-GELU `[1, Yv]` and target z; Theta=[b2; W2] |
| Ridge centering (toward original) | same, `w_prior=Theta` | — | **Unchanged** | ridge=1e-3, center=original per config |
| Correction spectrum / residual | `linalg.py::Correction` | — | Available | Used only for diagnostics/validation |
| **OOD AUROC / AUPR** | `metrics.py::compute_ood_metrics` (sklearn roc_auc/AP) | (N_id,), (N_ood,) | **Unchanged** (wrapped) | `pnc_metrics.ood_metrics` adds FPR95 + both AUPR directions |
| NLL | `metrics.py::compute_nll` (Gaussian) — regression only | — | Not applicable | Classification NLL written fresh (`pnc_metrics.clf_metrics`) |
| Predictive mixture | `util.py::_predictive_mean_var` (regression tuple) | — | Pattern reused | Classification mixture (mean softmax) in `construct.uncertainty_scores` |
| Perturbation basis (random low-rank) | random dirs pattern in `pnc_theory/harness.py::build_pnc` | (K, D) unit rows | Convention reused | `construct.perturbation_basis`: rows on flattened lin1 (D=768*3072) |
| Member coefficients / seeding | `harness.py` latent sampling | (M, K) | Convention reused | `construct.member_coefficients` (seed+1) |
| Bootstrap sampling | full-bootstrap convention (bf=1.0, w/ replacement) | — | Convention reused | per-member `RandomState(seed*1000+m)` index draw |
| Member serialization (compact) | shared basis + coeffs + corrected head | — | Pattern reused | `members/seed_<s>/{basis,coefficients,corrected_lin2}.npz` |
| Scale selection (ID-val NLL) | `run_sensitivity.py::anchor_scale` (val-NLL over sizes) | — | Pattern reused | `evaluate.select_scale`, ID-only, constrained |
| Temperature scaling | (new) | — | New | `pnc_metrics.fit_temperature` on ID-val |
| Config / seeds / resume | yaml + per-seed dirs | — | New (thin) | `configs/banking77_distilbert_pnc.yaml` |
| GPU-idle queue | (new) | — | New | `scripts/wait_for_gpu_then_run.py` (+ unit test) |

**Transformer-specific new code (unavoidable):** `transformer_adapter.py` (cached-prefix +
final-FFN/head tail replicating DistilBERT layer-5 attention, residual, LayerNorm, GELU,
and the classification head; cached-tail parity vs full model < 1e-5), `hf_checkpoint.py`
(PT→Flax conversion), `data.py` (Banking77 + CLINC), `baselines.py` (MSP/Energy/MC-Dropout/
uncorrected/head-P&C).

**Not reimplemented:** the ridge/least-squares solver, the AUROC/AUPR computation, the
random-perturbation convention, and the bootstrap/member-storage scheme all come from the
existing code.
