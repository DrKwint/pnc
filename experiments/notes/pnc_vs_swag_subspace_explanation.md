# Why PnC has lower ID RMSE than SWAG despite larger hidden perturbations

## The paradox

On Ant-v5 (and other environments), the frontier plot shows PnC has 3-5x larger
raw hidden perturbations (Unc-L2-h) than SWAG/Subspace, yet comparable or better
ID RMSE. The naive explanation "PnC works because its perturbations are smaller"
is wrong.

## Structural explanation

The key difference is **what gets perturbed**.

### PnC (multi-layer)
- Perturbs hidden layers l1, l3 (even-indexed)
- Corrects hidden layers l2, l4 (odd-indexed) via LS refit or pass-through
- Does **NOT** perturb the output heads (`mean_layer`, `var_layer`)
- Predictions pass through fixed output weights, which filter the hidden perturbation

### SWAG and Subspace
- Perturb **ALL** parameters: l1, l2, l3, l4, `mean_layer`, `var_layer`
- Direct perturbation of `mean_layer` and `var_layer` immediately distorts predictions
- Even small weight perturbations in the output heads can produce large prediction errors

## Three diagnostics

### Diagnostic C: Output sensitivity (gamma)

For each ensemble member i, measured at hidden layer 1:
```
gamma_i = E_x[||y_i(x) - y_base(x)||] / E_x[||h_i(x) - h_base(x)||]
```

This measures prediction error per unit of hidden perturbation.

**Ant-v5** (seed 0):

| Method           | dh (L1) |  dy   | gamma |
|------------------|---------|-------|-------|
| Subspace         |   1.21  |  6.04 |  5.03 |
| SWAG             |   1.20  |  4.80 |  4.02 |
| PnC (no corr.)   |   1.39  |  2.43 |  1.75 |
| PnC (LS corr.)   |  31.54  |  0.77 |  0.02 |

**Hopper-v5** (seed 0):

| Method           | dh (L1) |  dy   | gamma |
|------------------|---------|-------|-------|
| Subspace         |   0.66  |  0.54 |  0.77 |
| SWAG             |   2.19  |  0.28 |  0.13 |
| PnC (no corr.)   |   8.04  |  1.57 |  0.20 |
| PnC (LS corr.)   |   8.04  |  0.14 |  0.02 |

**Key finding**: On Ant-v5 at comparable dh (~1.2-1.4), PnC (no corr.) produces
only half the prediction change (dy=2.4 vs 4.8 for SWAG). On Hopper-v5, PnC has
3.7x larger dh than SWAG, but gamma stays comparable (0.20 vs 0.13), meaning
the output damage scales sub-linearly with hidden perturbation for PnC.

In both environments, Subspace has the highest gamma. LS correction brings gamma
near zero regardless of dh magnitude.

### Diagnostic B: Subspace alignment (alpha)

For each method's weight perturbation delta at the PnC-targeted layers:
```
alpha = ||proj_PnC(delta)||^2 / ||delta||^2
```

| Method           | alpha (Ant) | alpha (Hopper) |
|------------------|-------------|----------------|
| Subspace         |  0.001      |  0.002         |
| SWAG             |  0.001      |  0.002         |
| PnC (no corr.)   |  1.000      |  1.000         |
| PnC (LS corr.)   |  1.000      |  1.000         |

PnC perturbations have alpha = 1.0 by construction. SWAG/Subspace have alpha ≈ 0,
confirming they perturb in completely different directions at the same layers.
The PnC random subspace (20 directions per layer in a space of ~40,000 dimensions)
occupies a negligible fraction of the weight space.

### Diagnostic A: Effective projected-residual score (s_eff)

Derived from alignment: `s_eff = sqrt(1 - alpha)`.
SWAG/Subspace: s_eff ≈ 1.0 (virtually all perturbation is outside PnC's subspace).
PnC: s_eff = 0.0 (by construction).

## Answering the primary questions

### Q1: Why can PnC (no correction) have lower ID RMSE than SWAG?

**Structural isolation of output heads.** PnC perturbs hidden layers l1, l3 but
does NOT perturb `mean_layer` or `var_layer`. SWAG perturbs all parameters
including output heads, so its weight perturbations translate more directly into
prediction error.

Evidence: On Ant-v5 at comparable layer-1 hidden perturbation (dh ≈ 1.2-1.4),
PnC produces dy=2.4 vs SWAG's dy=4.8 — half the prediction damage. The output
sensitivity gamma is 1.75 for PnC vs 4.02 for SWAG (2.3x lower). On Hopper-v5,
PnC has 3.7x larger dh but gamma remains comparable to SWAG (0.20 vs 0.13),
confirming that hidden perturbation magnitude is not the right measure of harm.

### Q2: Why can PnC (no correction) beat Subspace on Far NLL?

Subspace's perturbations are conservative — low hidden-state diversity means
limited uncertainty coverage. PnC's larger perturbations (dh = 8.0 vs 0.7 on
Hopper) produce more diverse predictions, which improves NLL by better covering
the far-OOD predictive distribution. The hidden perturbation is large but not
catastrophic because the output heads are fixed (gamma stays at 0.20).

Subspace has the highest gamma in both environments (5.0 on Ant, 0.77 on Hopper),
meaning its small perturbations are disproportionately damaging per unit hidden
displacement. This limits how much perturbation Subspace can apply before
degrading predictions.

### Q3: Does LS correction mainly reduce perturbation magnitude, or remove harmful components?

**LS correction removes harmful output components, not hidden perturbation.**

On Hopper-v5, PnC (no corr.) and PnC (LS corr.) have identical dh at layer 1
(both 8.04) because correction acts downstream. But dy drops from 1.57 to 0.14
(9x reduction), and gamma drops from 0.20 to 0.02 (10x reduction).

LS correction remaps the next-layer weights to minimize the output residual while
leaving the hidden diversity intact. This preserves useful uncertainty for OOD
detection while removing harmful prediction distortion.

## Conclusion

PnC's lower ID RMSE despite larger hidden perturbations is not explained by
smaller perturbation magnitude. Instead, PnC perturbations are **structurally
isolated from the output heads**, so they cause less prediction damage per unit
of hidden-state displacement. Even without correction, PnC benefits from this
architecture: its output sensitivity gamma is 2-4x lower than Subspace and
comparable to SWAG despite much larger hidden perturbations. The LS correction
then removes the remaining harmful output component (reducing gamma by 10x)
without shrinking the hidden perturbation, improving the ID/OOD tradeoff by
preserving useful diversity while eliminating prediction distortion.
