# Final P&C protocols across all reported domains (Part K)

One row per domain, for the configuration that reaches the manuscript. The **ridge centre
column reads "original affine map" in every row** — that was the gate for finalising the
tables, and it passes.

| | MuJoCo | CIFAR-10 | ImageNet ViT | DistilBERT (appendix) |
|---|---|---|---|---|
| base architecture | 4x200 MLP dynamics model, ReLU, probabilistic head | PreAct ResNet, stage4/block1 | ViT-B/16, `IMAGENET1K_V1` | DistilBERT, Banking77 head |
| target layer(s) | hidden layers 1 and 3 (`perturb_indices=[0,2]`, multi-block sequential) | stage 4, block 1 | encoder block 11 (final), FFN | final FFN (`lin1`/`lin2`) |
| perturbed affine map | W of each perturbed hidden layer | `conv1` kernel | `mlp.0` (W1), 768x3072 | `lin1` |
| corrected affine map | W, b of the immediately following layer | `conv2` kernel + bias | `mlp.3` (W2), 3072x768 + b2 | `lin2` kernel + bias |
| correction observation rows | 4,096 ID states (per-member bootstrap) | 1,024 ID images, all spatial patches | 32,768 ID train images, **CLS row only** | 9,000 ID sentences |
| **ridge centre** | **original affine map** | **original affine map** | **original affine map** | **original affine map** |
| lambda | 1e-4 (summed objective) | 1e-3 | 1000 | 1e-3 |
| K (perturbation rank) | 20 | 10 | 20 | 20 |
| M (members) | 50 | 50 | 20 | 20 |
| correction N | 4,096 | 1,024 | 32,768 | 9,000 |
| bootstrap | 0.1 (Ant) / 0.3 (HalfCheetah, Hopper), per-member | none | none | full bootstrap with replacement, per member |
| selection criterion | perturbation size by lowest **ID-validation NLL**, per (env, seed) | lambda and scale by **best validation NLL subject to an accuracy-drop cap** | scale by the largest r whose **corrected** model holds ID top-1 within 0.50 pp (paired-bootstrap LCB); ridge by lowest ID NLL among passing | scale by ID validation NLL |
| predictive aggregation | mean of member Gaussians -> predictive mean and total variance | mean of member softmax probabilities | mean of member softmax probabilities at T = 0.7 | mean of member softmax probabilities |
| OOD / uncertainty score | total predictive variance | predictive entropy | predictive entropy | predictive entropy |

## Evidence for the ridge-centre row

Not asserted from prose. Each production solver was called on a synthetic problem and
compared against both closed forms (`ridge_center_verification.json`):

| domain | solver | rel. err vs original-centred | rel. err vs zero-centred |
|---|---|---|---|
| MuJoCo | `pnc_core.ensembles._ls_or_ridge_solve` | 1.7e-07 (float32) | 6.4e-02 |
| CIFAR | `pnc_core.pnc._ridge_regression_solve` | 9.0e-07 (float32) | 5.5e-02 |
| ImageNet | `imagenet_vit_pnc.pnc_core.cho_solve_shared` | ~0 | large |
| DistilBERT | `pnc_theory.linalg.ridge_solve(w_prior=Theta)` | 0.0 (exact) | 6.4e-02 |

CIFAR is the one that needs a word: it parameterises the *delta* rather than the corrected
map, solving `(H + lambda I) Delta = M^T (T - Y w2_orig)` and returning `w2_orig + Delta`.
Penalising `||Delta||^2` is penalising `||Theta_hat - Theta||^2`, and `conv2` is built with
`use_bias=False` so the implicit original bias is exactly zero. It is original-centred.

## What changed to get here

Only one manuscript-facing result was actually produced under the zero centre — the MuJoCo
headline table (lambda = 1e-4, no `_ridgeorig` token) — and it was rerun in Part C. The
implementation defaults were flipped so this cannot recur:

- `pnc_core/ensembles.py::PJSVDEnsemble(ridge_toward_orig=...)`: `False` -> **`True`**
- `pnc_core/gym_tasks.py::GymPJSVD.ridge_toward_orig`: `False` -> **`True`**
- `results/.../mujoco_sensitivity/scripts/run_sensitivity.py`: the `ridge_center` CSV column
  is now **derived from the configuration** instead of being the hardcoded string `"zero"`,
  which had mislabelled all 14,520 stored rows.

`experiments/pnc_protocol/ridge.py::RidgeSpecification` and
`experiments/scripts/pnc_theory/harness.py::build_pnc` already defaulted to the original
centre and were left alone.

## Remaining inconsistency

None on the ridge centre. Two lesser inconsistencies are recorded rather than fixed, because
fixing them would change reported numbers:

1. **lambda is not on a common scale across domains.** MuJoCo, CIFAR and DistilBERT use the
   *summed* reconstruction objective; the value that is comparable across row counts is the
   mean-objective lambda (`RidgeSpecification.lambda_mean`). ImageNet's 1000 and MuJoCo's
   1e-4 are not the same amount of shrinkage. The manuscript should either report
   `lambda / n_rows` or state the convention explicitly.
2. **Bootstrap differs by domain** (none for CIFAR/ImageNet, 0.1-0.3 for MuJoCo, full for
   DistilBERT). This is a genuine protocol difference, not a bookkeeping error.
