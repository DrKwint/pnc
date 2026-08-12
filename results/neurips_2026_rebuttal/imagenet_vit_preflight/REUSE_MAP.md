# REUSE_MAP — ImageNet ViT-B/16 P&C preflight

How this preflight reuses the existing P&C implementation. Same rule as the Banking77
DistilBERT experiment: **reuse unchanged first, thin model adapter second, never a second
P&C solver.**

The Banking77 experiment is the right ancestor because its target FFN has the identical
geometry — `768 → 3072 → 768`, flattened perturbation dimension `D_flat = 2,359,296`,
correction dimension `3072 + 1 = 3073` — so every convention transfers exactly.

| Component | Existing path / symbol | Shapes | Reused? | Adapter work |
|---|---|---|---|---|
| **Ridge solver** | `experiments/scripts/pnc_theory/linalg.py::ridge_solve` | Xv (N, 3073), target (N, 768), Theta (3073, 768) | **Unchanged, imported** | Imported directly; used as the parity reference for the streaming form |
| **Ridge, streaming form** | same solver's normal equations | G (3073, 3073), C (3073, 768) | **Reformulated, gated bitwise** | `pnc_core.ridge_solve_from_stats`; `ridge_solve` already builds `Xv.T@Xv` and `Xv.T@target` internally, so this is the same linear system fed from streamed statistics. `validate.py::solver_parity` asserts **bitwise** equality |
| Ridge centering toward original | `ridge_solve(..., w_prior=Theta)` | — | **Unchanged** | `w_prior = [b2; W2]` of the pristine checkpoint; λ = 1e-3 |
| Shared factorisation across outputs | (new, required by spec §13) | — | New | `pnc_core.cho_solve_shared`: one `cho_factor` reused for all 768 columns; gated against `ridge_solve` to 1e-8 |
| **Perturbation basis (random low-rank)** | `experiments/banking77_pnc/construct.py::perturbation_basis` | U (K, 2359296) unit rows | **Convention reused, gated bitwise** | Re-expressed in pure numpy (Banking77's module imports jax, unavailable in the torch venv). `validate.py::basis_parity` asserts element-for-element equality with the Banking77 original |
| **Member coefficients / seeding** | `construct.py::member_coefficients` | (M, K) | **Convention reused, gated bitwise** | same gate |
| **Perturbation scaling** | `construct.py::base_scale` | — | **Convention reused, gated bitwise** | median ‖dW1‖/‖W1‖ = target_rel at multiplier 1.0 |
| On-demand member materialisation | `construct.py::member_dW1` | (768, 3072) | **Pattern reused** | `pnc_core.member_dW1`; 20 full models are never built (6.45 GiB avoided, §8/§16) |
| Compact member serialization | `members/seed_<s>/{basis,coefficients,corrected_lin2}.npz` | — | **Pattern reused** | `raw/compact_members_M20.npz`: shared basis + coefficients + corrected W2/b2 |
| **Cached-prefix + tail inference** | `banking77_pnc/transformer_adapter.py` (`capture_h0` / `tail`) | h (B, 768) | **Pattern reused** | `vit_adapter.ViTPnCAdapter.prefix`/`tail` for torchvision ViT; parity vs full model gated < 1e-4 |
| Bias-first augmented design `[1, y]` | `construct.build_members` | (N, 3073) | **Unchanged** | `SufficientStats.augment` |
| Predictive mixture (mean softmax) | `construct.uncertainty_scores` | (M, B, C) | **Pattern reused** | `stages_memory._ensemble_probs` accumulates mean softmax sequentially |
| Deterministic seeds | `RandomState(seed)` / `seed+1` convention | — | **Unchanged** | same offsets; token sampling adds `RandomState(seed*100003 + image_id)` |
| OOD AUROC / AUPR | `pnc_core/metrics.py`, `banking77_pnc/pnc_metrics.py` | — | **Not used here** | Spec §21 forbids OOD in this preflight |

## New code (unavoidable, and why)

| File | Why it could not be reused |
|---|---|
| `vit_adapter.py` | torchvision `VisionTransformer` ≠ Flax DistilBERT: different block algebra (pre-norm `ln_1`/`ln_2`, post-attention residual), different weight layout (`nn.Linear.weight` is (out, in), so code-layout `W1 = weight.T`), CLS slice after `encoder.ln`. Gated against the full model. |
| `pnc_core.SufficientStats` | Banking77 materialises its whole (N, 3073) design in memory. ImageNet all-token calibration is 197 rows/image, so the design must be streamed. Statistics only — the solve is the reused one. |
| `memprobe.py` | The spec requires CUDA peak-memory instrumentation and OOM-tolerant CSV recording; the JAX experiments had no equivalent. |
| `data.py` | ImageNet-1k val via parquet + pinned torchvision transform. ID only — it has no code path that can load OOD data. |
| `stages_memory.py`, `stages_correction.py`, `run_preflight.py` | The preflight protocol itself (spec §6–17). |

## Not reimplemented

The ridge/least-squares solve, the original-centred ridge convention, the random
low-rank perturbation convention, the coefficient/scale conventions, the compact
member-storage scheme, and the cached-prefix inference pattern all come from the
existing code. Where an exact import was impossible (jax dependency, streamed design),
the reformulation is **gated against the original in `validate.py`** rather than assumed
equivalent.

## Environment note

The repo's main `.venv` carries torch 2.11.0+cu130, whose kernels start at `sm_75`; the
TITAN X Pascal is `sm_61` and every CUDA call fails with
`no kernel image is available for execution on the device`. This preflight therefore uses
a sibling venv `.venv_vit` (torch 2.7.1+cu126, whose `sm_60` cubins are binary-compatible
with sm_61), matching the repo's existing `.venv_bank` convention and the `.venv_*/`
gitignore rule. The main `.venv` is left untouched.

`pnc_theory.linalg` is pure numpy and imports cleanly in `.venv_vit`, so the reused solver
needs no JAX. Only the Banking77 basis-parity gate needs JAX, and it is run separately:

```bash
JAX_PLATFORMS=cpu .venv/bin/python -m experiments.imagenet_vit_pnc.validate --only basis_parity
```
