# Methods — finite-transfer analysis of Perturb-and-Correct

Companion to `FINITE_TRANSFER_REPORT.md`. Everything here is fixed by the code
in `analysis/finite_transfer/` and does not depend on any result.

## 1. The system under analysis

Single-layer P&C on the MuJoCo transition models: a 4x200 ReLU
`ProbabilisticRegressionModel` (Gaussian mean/variance heads) trained by NLL.
One perturb/correct interface — perturb hidden layer `li`, refit the affine map
of layer `li+1` — because the exact local interpretation is only clean at one
correction interface. Multi-layer P&C is deliberately out of scope for Tier 1.

Ant's target layer is `l1` (`li = 0`), which is what the validated ID-only layer
selector picks for Ant in `pnc_theory_reports/LAYER_SELECTION.md`. Part 5 sweeps
`li` over all three available interfaces.

Members: `K = 20` unit-norm Gaussian directions per layer, member latents scaled
so each member's coefficient vector has norm `pert_size`. This reproduces
`experiments/scripts/pnc_theory/harness.build_pnc`, which in turn reproduces the
shipped `ensembles.PJSVDEnsemble` build path.

## 2. Exactness of the reproduction

`analysis/finite_transfer/capture.py` re-derives the correction in float64. It is
not a re-implementation of the *idea*; it is checked against the object that
ships. `tests/test_implementation_parity.py` builds a real `PJSVDEnsemble` and
asserts agreement on

* the calibration subset indices (exact),
* the per-member weight perturbations (< 1e-6 relative),
* the fitted correction weights (< 2e-4 relative — the shipped solve runs in
  float32 on GPU, and the design has cond(G) ~ 1e5-1e6),
* the member forward predictions (< 2e-4 relative).

Conventions that had to match and are individually pinned by tests:

| convention | MLP interface | conv (CIFAR) interface |
|---|---|---|
| bias column | **last**: `[h, 1]` | **first**: `[1, Y]` |
| penalty | uniform `lam I`, bias included | uniform `lam I`, bias included |
| `lam = 0` | `lstsq` minimum-norm, *not* a ridge | n/a (conv uses `lam > 0`) |
| shrinkage target | zero by default; `Theta` if `ridge_toward_orig` | `Theta` always (solves for a delta) |
| correction target | original pre-activation of layer `li+1` | original conv2 output |

## 3. Quantities

For member `v` and evaluation point `x`, with `Gv = Xv^T Xv + lam I`:

```
PredictedDelta = Av Gv^-1 Xv^T (Xv - X0)
TransferDefect = (Av - A0) - PredictedDelta - B          B = ridge-shrinkage term
transfer_score(x)   = mean_v || Theta^T TransferDefect ||^2
raw_effect(x)       = mean_v || Theta^T (Av - A0) ||^2
predictable_effect  = mean_v || Theta^T PredictedDelta ||^2
final_disagreement  = trace Cov_v [ f_v(x) ]
```

### Every identity is tagged by its correction objective

The clean formula (`B = 0`) is not right or wrong in the abstract — it is the
exact residual for **one** objective. The three-way rule, verified in
`tests/test_identity_by_objective.py`, which takes the objective as an explicit
argument:

| objective | penalty | exact `B` | clean form exact? |
|---|---|---|---|
| **toward original** `min ||Xv W - X0 Theta||^2 + ||W - Theta||^2_Lambda` | any `Lambda`, uniform or not | `0` | **always** |
| **toward zero** `min ||Xv W - X0 Theta||^2 + lam||W||^2` | uniform `lam I` | `lam Av Gv^-1` | no — off by the shrinkage term |
| **minimum norm** (`lam = 0`, `lstsq`) | none | `Av (I - P_row(Xv))` | **iff `Xv` has full column rank** |

Two consequences worth stating explicitly:

* The toward-original identity holds for a **non-uniform** `Lambda` too (e.g. an
  unpenalised bias) — provided the *reconstruction weights* use the same
  `Lambda` the fit used. What a non-uniform `Lambda` costs is orthogonal
  coordinate invariance, not exactness.
* The toward-zero discrepancy scales with `lam`. At the headline benchmark's
  `lam = 1e-4` it is ~6e-7 on synthetic problems — negligible. The literal-formula
  error measured in the *real* headline regime (up to 52%, see
  `CANONICAL_CONFIG_AUDIT.md`) is therefore almost entirely a **rank-deficiency**
  effect from `bootstrap_frac = 0.1`, not a ridge effect.

Everything is computed through the SVD filter factors `f1 = s/(s^2+lam)`,
`f2 = s f1` so the `lam = 0` and rank-deficient regimes stay well defined.

### Which convention each part of this program uses

| context | ridge target | lambda | source |
|---|---|---|---|
| code default | toward zero | 0.0 (`lstsq`) | `gym_tasks.GymPJSVD` |
| **headline MuJoCo benchmark** | **toward zero** | **1e-4**, `bootstrap_frac=0.1` | `make_gym_paper_table._pnc_canonical` |
| finite-transfer diagnostics | toward original | 1.0 | `configs/mujoco_tier1.yaml` |
| CIFAR conv interface | toward original (solves a delta) | 1e-3 | `pnc.solve_chunked_conv2_correction` |

**Decomposing "finite".** Two things can be finite rather than infinitesimal —
the response vector and the design conditioned on. Three scores isolate them,
holding everything else fixed:

| score | response | design |
|---|---|---|
| `transfer_score` | finite | perturbed `Xv` |
| `lin_transfer_score` | first-order Taylor | perturbed `Xv` |
| `lin0_transfer_score` | first-order Taylor | unperturbed `X0` |

## 3a. A structural caveat on the Part-2 comparison

The prediction target is `T = Theta^T (Av - A0)`, and one candidate feature space
*is* `Av`. Since `Theta^T Av` is exactly linear in that feature, conditioning on
`Av` starts with half the target already solved, and the problem reduces to
recovering `Theta^T A0` from `Av`. `perturbed_hidden` therefore has an algebraic
head start that no other feature space gets, and its near-perfect ID `R^2` should
not be read as evidence on its own.

Two comparisons are not contaminated by this and carry the weight instead:

* `unperturbed_hidden` (`A0`) faces the mirror-image problem — recover `Av` from
  `A0` — with no head start in either direction, so the `Av` vs `A0` asymmetry is
  informative;
* `random_relu`, `raw_state`, `penultimate` and `base_output` contain no part of
  the target linearly, so their comparison against each other is clean.

## 4. Selection protocol

No OOD split is read by any selection step.

* **Perturbation scale**: gate on `|ID RMSE(ensemble) - ID RMSE(base)| <= 0.05`,
  then take the largest surviving scale from `{5, 10, 20, 50}` — the rule
  validated at near-zero OOD regret in `pnc_theory_reports/ID_ONLY_SELECTION.md`.
* **Part-2 ridge values**: chosen per feature space on an ID validation split
  drawn from the training pool and disjoint from both the calibration subset and
  the ID test set.
* **Target layer**: fixed a priori from the existing ID-only layer selector;
  Part 5 sweeps it rather than selecting it.

## 5. Statistical protocol

* **Seed is the replication unit.** Confirmatory comparisons are paired per-seed
  differences with a 95% t-interval over seeds, computed *within* an environment.
  Cross-environment lines average per-environment means; they never pool
  examples across environments.
* **Per-example correlations are exploratory** and are reported as effect sizes
  only. No p-values are computed on example counts — 8000 correlated evaluation
  points from one model are one experimental replicate, not 8000.
* Compared variants always see the same evaluation examples.
* Regimes: `ID` (held-out in-distribution), `Near`/`Mid`/`Far` (increasing shift,
  from the Minari policy tiers in `data.py`).

## 6. Data products

`examples.parquet` carries one row per (example, seed, regime, configuration)
with: environment, model seed, member count, target layer, perturbation scale,
calibration size, ridge value, regime, base and ensemble squared error, final
disagreement, raw / predicted / hidden-defect / affine-visible / actual-residual
energies, Mahalanobis distance, ridge leverage, reconstruction-weight norm,
activation flip rate, Gram condition number, and effective rank. `detail.parquet`
keeps member-level values for a fixed stratified subset of examples.
`provenance.json` carries the schema version, git commit, platform, per-run
configuration, calibration-index and perturbation digests, dtype and runtimes.

## 7. Reproduction

```bash
.venv/bin/python -m pytest analysis/finite_transfer/tests -q
.venv/bin/python scripts/run_finite_transfer.py \
    --config analysis/finite_transfer/configs/mujoco_tier1.yaml
.venv/bin/python scripts/make_finite_transfer_report.py \
    --artifacts artifacts/finite_transfer/mujoco_tier1
```

CIFAR-10 is prepared but not run here (it runs on the machine holding the CIFAR
checkpoints): `configs/cifar_tier1.yaml` plus `capture_conv.py`, whose conv
algebra is covered by `tests/test_conv_identity.py` and needs no CIFAR data.
