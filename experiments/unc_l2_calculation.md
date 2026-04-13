# How Unc-L2-h and Unc-L2-z are calculated

This document explains the perturbation magnitude metrics reported in `report.txt`
and stored in the JSON result files as `uncorrected_l2_*_h` and `uncorrected_l2_*_z`.

## Core formula

Both metrics measure the L2 distance between ensemble members' intermediate
representations and the original (base/MAP) model's representations, averaged
over members and data points.

Given:
- `h_orig`: hidden activations from the base model, shape `(Batch, Dim)`
- `h_ens`: hidden activations from each ensemble member, shape `(N_members, Batch, Dim)`

```
Unc-L2 = mean over i,b of  ||h_ens[i,b,:] - h_orig[b,:]||_2
```

Implementation in `util.py:77-85`:
```python
def compute_l2_distance(perturbed_states, orig_state):
    return float(
        jnp.mean(jnp.sqrt(jnp.sum((perturbed_states - orig_state) ** 2, axis=-1)))
    )
```

## Two variants: h-space vs z-space

### Unc-L2-h (hidden space)

The L2 distance is computed at the post-activation hidden layer where perturbation
has its most direct effect. For methods that perturb layer weights, this is the
output of the perturbed layer after applying the activation function.

```
h_orig = activation(x @ W_orig + b_orig)        # shape (Batch, Hidden)
h_ens  = activation(x @ W_pert_i + b_pert_i)    # shape (N, Batch, Hidden)
Unc-L2-h = compute_l2_distance(h_ens, h_orig)
```

For full-network methods (SWAG, Subspace, Laplace), ALL layers are perturbed, so
the hidden state at any layer reflects perturbations from preceding layers as well.
By default, `layer_idx=1` is used (first hidden layer).

### Unc-L2-z (pre-activation / output space)

The L2 distance is computed in the next layer's pre-activation space, using the
**original** (unperturbed) next-layer weights applied to the perturbed hidden states:

```
z_orig = h_orig @ W_next_orig + b_next_orig
z_ens  = h_ens  @ W_next_orig + b_next_orig
Unc-L2-z = compute_l2_distance(z_ens, z_orig)
```

This measures how the hidden-layer perturbation propagates through the next linear
transformation *without* any correction, regardless of whether correction is later
applied. For probabilistic models with mean/variance heads, `z` is the concatenated
output of those heads.

## PnC/PJSVD also reports Corr-L2-z

PnC applies a correction step (least-squares or affine refit) that remaps the
perturbed hidden activations to better match the original model's next-layer output.

```
z_corr = correction(h_ens)
Corr-L2-z = compute_l2_distance(z_corr, z_orig)
```

The ratio `Corr-L2-z / Unc-L2-z` quantifies correction effectiveness.

When `correction_mode=none`, no correction is applied, so `Corr-L2-z = Unc-L2-z`.

## Per-method details

### PnC / PJSVD (gym_tasks.py:986-1060)

- **Base model**: MAP-trained network (probabilistic or standard)
- **Perturbation**: Directions found via SVD of the Jacobian (or random), applied to
  specific hidden layers. Perturbation magnitude controlled by `perturbation_scale`
  and `member_radius_multipliers`.
- **h reference**: `get_intermediate_state(model, x, layer_idx)` on the base model
- **h ensemble**: `ens.predict_intermediate_and_corrected(x)` returns both
  uncorrected hidden states and corrected next-layer activations
- **L2 computed**: `uncorrected_l2_*_h`, `uncorrected_l2_*_z`, `corrected_l2_*_z`
- **Regions**: `id`, `ood_near`, `ood_mid`, `ood_far`

### SWAG (gym_tasks.py:543, ensembles.py:SWAGEnsemble)

- **Base model**: MAP-trained network (before SWAG posterior sampling)
- **Perturbation**: Full parameter sampling from a Gaussian with diagonal + low-rank
  covariance estimated from the SGD trajectory. All layers perturbed jointly.
- **h reference**: `get_intermediate_state(model, x, layer_idx=1)` on the base model
- **h ensemble**: `ensemble.predict_intermediate(x, layer_idx=1)` -- for each of
  `n_models` samples, clones the model, sets sampled weights, extracts the hidden
  state at `layer_idx`
- **L2 computed**: `uncorrected_l2_*_h`, `uncorrected_l2_*_z`
- **Regions**: `id`, `ood_near`, `ood_mid`, `ood_far`

### Subspace Inference (gym_tasks.py:1393, ensembles.py:SubspaceInferenceEnsemble)

- **Base model**: MAP-trained network (mean of SWAG posterior)
- **Perturbation**: Parameters perturbed along top PCA directions of the SGD
  trajectory, with coefficients drawn via elliptical slice sampling (ESS).
  All layers perturbed jointly through the shared subspace.
- **h reference**: `get_intermediate_state(model, x, layer_idx=1)` on the base model
- **h ensemble**: `ensemble.predict_intermediate(x, layer_idx=1)` -- for each of
  `n_samples` ESS samples, updates model weights, extracts hidden state
- **L2 computed**: `uncorrected_l2_*_h`, `uncorrected_l2_*_z`
- **Regions**: `id`, `ood_near`, `ood_mid`, `ood_far`

### Laplace (gym_tasks.py:634, ensembles.py:LaplaceEnsemble)

- **Base model**: MAP-trained network
- **Perturbation**: K-FAC approximate Laplace posterior. Layer weights sampled
  independently from a Gaussian with K-FAC-factored covariance scaled by prior
  precision. All layers perturbed.
- **h reference**: `get_intermediate_state(model, x, layer_idx=1)` on the base model
- **h ensemble**: `ensemble.predict_intermediate(x, layer_idx=1)` -- for each of
  `n_models` samples, sets sampled weights, extracts hidden state
- **L2 computed**: `uncorrected_l2_*_h`, `uncorrected_l2_*_z`
- **Regions**: `id`, `ood_near`, `ood_mid`, `ood_far`

## Methods without Unc-L2 metrics

### Deep Ensemble

Independently trained models from different random initializations. There is no
shared base model, so the "distance from base" concept does not apply.

### MC Dropout

Uses a single trained model with dropout enabled at inference. While there is a
base model, the perturbation mechanism (random neuron masking) is fundamentally
different from weight perturbation and the hidden-state L2 metric is not directly
comparable.

## JSON keys reference

Each region `R` in `{id, ood_near, ood_mid, ood_far}` produces:

| Key                         | Description                                   |
|-----------------------------|-----------------------------------------------|
| `uncorrected_l2_{R}_h`     | L2 distance in hidden space (Unc-L2-h)        |
| `uncorrected_l2_{R}_z`     | L2 distance in pre-activation space (Unc-L2-z)|
| `corrected_l2_{R}_z`       | L2 distance after PnC correction (PnC only)   |
