# Gym PnC Tuning Plan (4x200 Architecture)

## Goal

Tune single-layer and multi-layer PnC on Gym dynamics prediction tasks (Ant-v5, HalfCheetah-v5, Hopper-v5, Humanoid-v5) so that PnC matches or beats MC Dropout, SWAG, Laplace, and Subspace Inference on **NLL** (minimize) and **ECE** (minimize) while maintaining competitive **RMSE** (within 5% of the single-model regressor). Deep Ensemble is reported as a reference but not a target to beat. Results are reported in two tables: one with post-hoc variance calibration (VCal) applied uniformly to all methods, and one without any posthoc calibration. OOD detection (AUROC) is reported if time permits.

## Architecture: 4x200

All methods use a 4-hidden-layer, 200-unit-wide feedforward network as the underlying architecture. This replaces the 2x64 TransitionModel used in prior experiments. The 4x200 architecture has ~130K parameters (environment-dependent) and provides more representational capacity and, critically for PnC, more layers to perturb and correct through.

| Environment | Input dim | Output dim | Layer sizes | Total params |
|-------------|-----------|------------|-------------|-------------|
| Ant-v5 | 35 | 27 | 35→200→200→200→200→27 | ~133K |
| HalfCheetah-v5 | 23 | 17 | 23→200→200→200→200→17 | ~128K |
| Hopper-v5 | 14 | 11 | 14→200→200→200→200→11 | ~125K |
| Humanoid-v5 | 393 | 376 | 393→200→200→200→200→376 | ~274K |

**Why 4x200:**
- 4 hidden layers provide 3 affine interfaces for PnC correction (l1→l2, l2→l3, l3→l4), plus the final l4→output interface
- Width 200 gives large enough weight matrices (200x200 = 40K params) for rich Lanczos subspaces
- Matches the classification model architecture used in the UCI experiments
- No memory concerns: all matrices fit comfortably in 8GB VRAM

---

## Fairness Protocol

1. **Shared architecture**: All methods use the same 4x200 feedforward network (except Deep Ensemble, which trains 5-10 separate 4x200 models; and MC Dropout, which adds dropout layers to the same architecture).
2. **Same training recipe**: Adam optimizer, lr=1e-3, batch_size=64, early stopping with patience=10, eval_freq=100. Training steps may need increasing from 2000 to 5000 for the larger architecture — calibrate in Phase 1.
3. **Post-hoc variance calibration** (`--posthoc-calibrate`): Both settings are reported. Every method is run once WITH and once WITHOUT posthoc variance calibration, and both sets of results are reported in separate tables. Variance scale (when used) is fit on the 10% validation split. This allows the reader to see both the raw uncertainty signal quality and the calibrated result.
4. **Hyperparameter selection**: For sweep parameters (perturbation_scale, Laplace prior, etc.), select by **best ID NLL** (in the VCal table, after posthoc calibration; in the raw table, without) **subject to the constraint that ID RMSE does not degrade by more than 5% relative to the single-model regressor**. This prevents selecting configurations that sacrifice accuracy for artificially good NLL. The same selected configuration is used in both tables.
5. **Same ensemble size**: n_perturbations=100 for PnC, MC Dropout, SWAG, Subspace Inference. n_models=5 for Deep Ensemble (standard). This is reduced from 1000 to match typical ensemble sizes in the literature and for computational tractability.
6. **Same data**: 10,000 transitions per regime, same seeds (0, 10, 200).
7. **Probabilistic base model**: All methods that can use a probabilistic base model (dual-head mean+variance) should do so. Methods that perturb parameters (PnC, SWAG, Laplace) then produce ensembles of heteroscedastic predictors whose mean and variance heads are both affected by the perturbation. The total predictive variance is `mean(vars) + var(means)` (law of total variance decomposition).

---

## Phase 0: Code Changes

The current code hardcodes `hidden_dims=[64, 64]` or uses `TransitionModel` (2x64) across most Gym tasks. The following changes are needed before any experiments run.

### 0a. Parameterize `hidden_dims` across all Gym tasks

Add `hidden_dims = luigi.ListParameter(default=[200, 200, 200, 200])` to every Gym task class that creates models:

- `GymStandardEnsemble`: Change `ProbabilisticRegressionModel(..., hidden_dims=[64, 64])` to use `self.hidden_dims`
- `GymMCDropout`: Replace `MCDropoutTransitionModel` with `MCDropoutRegressionModel(..., hidden_dims=self.hidden_dims)`. The `MCDropoutRegressionModel` already supports arbitrary `hidden_dims`.
- `GymSWAG`: Change `ProbabilisticRegressionModel(..., hidden_dims=[64, 64])` to use `self.hidden_dims`
- `GymLaplace`: Replace `TransitionModel` with `RegressionModel(..., hidden_dims=self.hidden_dims)` or `ProbabilisticRegressionModel` if probabilistic
- `GymSubspaceInference`: Change `ProbabilisticRegressionModel(..., hidden_dims=[64, 64])` to use `self.hidden_dims`
- `GymPJSVD`: Fix to actually use its existing `hidden_dims` parameter (currently hardcoded to `[64, 64]` in model construction despite accepting the parameter)
- `AllGymExperiments`: Add `hidden_dims` parameter and pass through to all sub-tasks
- `AllGymExperimentsMultiSeed`: Same

### 0b. Fix GymPJSVD model construction

In `GymPJSVD.run()`, the model construction ignores `self.hidden_dims`:
```python
# Current (broken):
model = TransitionModel(inputs_id.shape[1], targets_id.shape[1], ...)

# Fixed:
model = RegressionModel(inputs_id.shape[1], targets_id.shape[1], ..., hidden_dims=list(self.hidden_dims))
```

Similarly for `ProbabilisticRegressionModel` when `probabilistic_base_model=True`.

### 0c. Fix multi-layer weight extraction for arbitrary depths

`GymPJSVD.run()` currently hardcodes `model.l1`, `model.l2`, `model.l3` for weight extraction. For `RegressionModel` (which uses `model.layers`), update to:
```python
# Extract weights from RegressionModel
W_layers = [model.layers[i].kernel.get_value() for i in range(len(self.hidden_dims) + 1)]
b_layers = [model.layers[i].bias.get_value() for i in range(len(self.hidden_dims) + 1)]
```

### 0d. Generalize layer_scope for 4-layer models

The current `layer_scope` has only "first" (perturb l1) and "multi" (perturb l1+l2). For 4x200, we need:
- "first": perturb l1 only, correct at l2
- "multi": perturb l1+l2, correct at l3
- Add "deep": perturb l1+l2+l3, correct at l4
- Add "per_layer": configurable set of layers to perturb (most flexible)

Alternatively, add a `perturbed_layers` parameter that accepts a list of layer indices, e.g. `[0, 1, 2]` to perturb the first 3 hidden layers.

### 0e. Ensure posthoc calibration works for Gym

Verify that `--posthoc-calibrate` is passed and functions correctly for all Gym tasks. The current results don't appear to use posthoc calibration (no `_vcal` suffix in filenames).

### 0f. Implement sequential layerwise correction for multi-layer PnC

**This is the intended behavior of the algorithm.** When perturbing multiple layers, the correction should be applied at each layer interface independently rather than a single joint correction at the end. This must be implemented before running any multi-layer PnC experiments.

**Current behavior** (joint tail correction): perturb l1+l2 jointly via a single direction vector in the concatenated weight space, then solve a single least-squares correction at the l3 interface. The perturbation to l1 propagates through l2 (and is further perturbed there), creating a compound effect that the single tail correction must absorb. This leads to poor ID preservation at moderate perturbation scales.

**Correct behavior** (sequential layerwise correction): For each ensemble member:
1. Perturb l1 weights: `W1' = W1 + dW1`
2. Compute perturbed activations: `h1' = act(X @ W1' + b1)`
3. Solve least-squares at l2 interface: find `W2', b2'` such that `h1' @ W2' + b2' ≈ h1_orig @ W2 + b2`
4. Compute corrected-then-activated h2: `h2' = act(h1' @ W2' + b2')`
5. Perturb l2 weights: `W2'' = W2' + dW2` (perturb the CORRECTED weights, or equivalently add another perturbation direction on top)
6. Solve least-squares at l3 interface: find `W3', b3'` such that `h2'' @ W3' + b3' ≈ h2_orig @ W3 + b3`
7. Continue for remaining perturbed layers

**Implementation in `PJSVDEnsemble`:**
- In `_precompute_least_squares`, iterate over perturbed layers sequentially
- At each interface, the correction target is the original pre-activation of the next layer
- Each correction is a local 201×D_next least-squares problem
- The Lanczos directions can still be found in the joint weight space (or per-layer — see Phase 3)
- The key change is that `_forward_member` applies perturbation + correction at each interface, not just at the tail

**Why this matters**: Each correction operates on a smaller residual (only one layer's perturbation), and the corrections compound constructively. This expands the usable perturbation scale range significantly.

**Note on per-layer direction finding**: For sequential correction, it is natural (but not required) to find Lanczos directions per-layer rather than in the joint space. Per-layer Lanczos is also faster (200×200 = 40K per layer vs 120K for 3 layers jointly). The plan supports both approaches.

---

## Phase 1: Training Recipe Calibration

Before running any method comparisons, establish the training recipe for 4x200.

### 1a. Training steps sweep

The 2x64 model used 2000 steps. The 4x200 model has ~20x more parameters and may need longer training.

```bash
for STEPS in 2000 5000 10000 20000; do
  for ENV in HalfCheetah-v5 Hopper-v5 Ant-v5; do
    python -m luigi --module gym_tasks GymPJSVD \
      --env $ENV --steps 10000 --hidden-dims '[200,200,200,200]' \
      --n-directions 10 --n-perturbations 1 \
      --perturbation-sizes '[0.0]' \
      --layer-scope first --correction-mode least_squares \
      --pjsvd-family random --seed 0 --local-scheduler \
      --training-steps $STEPS
  done
done
```

(This runs PnC with zero perturbation to evaluate base model quality at different training lengths. Alternatively, use a standalone training task if available.)

**What to look for**: RMSE convergence — the point where more training steps stop improving validation RMSE significantly.

### 1b. Learning rate check

```bash
for LR in 1e-4 5e-4 1e-3 2e-3; do
  # Quick single-seed eval
done
```

Use the standard 1e-3 unless results show a clear winner.

### 1c. Record base model quality

For each environment and the selected training recipe, record:

| Environment | RMSE (train) | RMSE (val) | RMSE (ID eval) | Steps to converge |
|-------------|-------------|------------|----------------|------------------|
| Ant-v5 | | | | |
| HalfCheetah-v5 | | | | |
| Hopper-v5 | | | | |
| Humanoid-v5 | | | | |

---

## Phase 2: Baselines (4x200)

Run all baselines with the calibrated 4x200 training recipe, seeds 0, 10, 200. Each method is run **twice**: once with `--posthoc-calibrate` and once without, so both raw and calibrated results are available. (Luigi output paths differ by the `_vcal` suffix, so both runs coexist.)

### 2a. Single model (base reference)

No uncertainty — just the base model. Used to establish the RMSE ceiling and the 5% degradation threshold.

### 2b. Deep Ensemble

```bash
for VCAL in "" "--posthoc-calibrate"; do
  for SEED in 0 10 200; do
    for ENV in HalfCheetah-v5 Hopper-v5 Ant-v5 Humanoid-v5; do
      python -m luigi --module gym_tasks GymStandardEnsemble \
        --env $ENV --steps 10000 --n-baseline 5 \
        --hidden-dims '[200,200,200,200]' \
        --seed $SEED $VCAL --local-scheduler
    done
  done
done
```

### 2c. MC Dropout

```bash
for VCAL in "" "--posthoc-calibrate"; do
  for SEED in 0 10 200; do
    for ENV in HalfCheetah-v5 Hopper-v5 Ant-v5 Humanoid-v5; do
      python -m luigi --module gym_tasks GymMCDropout \
        --env $ENV --steps 10000 --n-perturbations 100 \
        --hidden-dims '[200,200,200,200]' \
        --seed $SEED $VCAL --local-scheduler
    done
  done
done
```

### 2d. SWAG

```bash
for VCAL in "" "--posthoc-calibrate"; do
  for SEED in 0 10 200; do
    for ENV in HalfCheetah-v5 Hopper-v5 Ant-v5 Humanoid-v5; do
      python -m luigi --module gym_tasks GymSWAG \
        --env $ENV --steps 10000 --n-perturbations 100 \
        --hidden-dims '[200,200,200,200]' \
        --seed $SEED $VCAL --local-scheduler
    done
  done
done
```

### 2e. Laplace

```bash
for VCAL in "" "--posthoc-calibrate"; do
  for SEED in 0 10 200; do
    for ENV in HalfCheetah-v5 Hopper-v5 Ant-v5 Humanoid-v5; do
      python -m luigi --module gym_tasks GymLaplace \
        --env $ENV --steps 10000 --n-perturbations 100 \
        --subset-size 4096 \
        --laplace-priors '[1.0,10.0,100.0,1000.0,10000.0,100000.0]' \
        --hidden-dims '[200,200,200,200]' \
        --seed $SEED $VCAL --local-scheduler
    done
  done
done
```

### 2f. Subspace Inference

```bash
for VCAL in "" "--posthoc-calibrate"; do
  for SEED in 0 10 200; do
    for ENV in HalfCheetah-v5 Hopper-v5 Ant-v5 Humanoid-v5; do
      python -m luigi --module gym_tasks GymSubspaceInference \
        --env $ENV --steps 10000 --n-perturbations 100 \
        --hidden-dims '[200,200,200,200]' \
        --seed $SEED $VCAL --local-scheduler
    done
  done
done
```

### 2g. Analysis

Two tables per environment: one raw, one with VCal.

| Method | Env | ID RMSE | ID NLL | ID ECE | Near NLL | Mid NLL | Far NLL | Far AUROC |
|--------|-----|---------|--------|--------|----------|---------|---------|-----------|
| Single model | HC | | | | | | | |
| Deep Ens (5) | HC | | | | | | | |
| MC Dropout | HC | | | | | | | |
| SWAG | HC | | | | | | | |
| Laplace (best) | HC | | | | | | | |
| Subspace | HC | | | | | | | |
| ... | ... | | | | | | | |

**Targets for PnC to beat** (from best of MC Dropout, SWAG, Laplace, Subspace):
- RMSE: maintain within 5% of the single-model regressor RMSE (from Phase 1c)
- NLL: < best baseline NLL across ID and all shift regimes, among configurations that pass the 5% RMSE gate
- ECE: < best baseline ECE among configurations that pass the 5% RMSE gate

---

## Phase 3: PnC Single-Layer Scale Discovery

For each of the 4 hidden layers, sweep perturbation scales to find the diversity threshold and accuracy limit. This phase uses `layer_scope=first` with different `target_layer_idx` values.

**Key question for 4x200**: Which layer(s) are best to perturb?
- Layer 1 (input→200): perturbations propagate through 3 more layers before output. More correction capacity but more compounding.
- Layer 2 (200→200): perturbations propagate through 2 more layers.
- Layer 3 (200→200): perturbations propagate through 1 more layer.
- Layer 4 (200→output): direct output effect, minimal correction capacity. Probably too direct for diversity.

### 3a. Layer 1 (first hidden layer)

```bash
for ENV in HalfCheetah-v5 Hopper-v5 Ant-v5; do
  python -m luigi --module gym_tasks GymPJSVD \
    --env $ENV --steps 10000 --hidden-dims '[200,200,200,200]' \
    --n-directions 20 --n-perturbations 100 \
    --perturbation-sizes '[1.0,5.0,10.0,20.0,50.0,100.0,200.0]' \
    --layer-scope first --target-layer-idx 0 \
    --correction-mode least_squares \
    --pjsvd-family low --safe-subspace-backend projected_residual \
    --subset-size 4096 --probabilistic-base-model \
    --seed 0 --posthoc-calibrate --local-scheduler
done
```

### 3b. Layer 2 (second hidden layer)

```bash
for ENV in HalfCheetah-v5 Hopper-v5 Ant-v5; do
  python -m luigi --module gym_tasks GymPJSVD \
    --env $ENV --steps 10000 --hidden-dims '[200,200,200,200]' \
    --n-directions 20 --n-perturbations 100 \
    --perturbation-sizes '[1.0,5.0,10.0,20.0,50.0,100.0,200.0]' \
    --layer-scope first --target-layer-idx 1 \
    --correction-mode least_squares \
    --pjsvd-family low --safe-subspace-backend projected_residual \
    --subset-size 4096 --probabilistic-base-model \
    --seed 0 --posthoc-calibrate --local-scheduler
done
```

### 3c. Layer 3 (third hidden layer)

```bash
for ENV in HalfCheetah-v5 Hopper-v5 Ant-v5; do
  python -m luigi --module gym_tasks GymPJSVD \
    --env $ENV --steps 10000 --hidden-dims '[200,200,200,200]' \
    --n-directions 20 --n-perturbations 100 \
    --perturbation-sizes '[1.0,5.0,10.0,20.0,50.0,100.0,200.0]' \
    --layer-scope first --target-layer-idx 2 \
    --correction-mode least_squares \
    --pjsvd-family low --safe-subspace-backend projected_residual \
    --subset-size 4096 --probabilistic-base-model \
    --seed 0 --posthoc-calibrate --local-scheduler
done
```

### 3d. Random directions ablation (per best layer)

```bash
for ENV in HalfCheetah-v5 Hopper-v5 Ant-v5; do
  python -m luigi --module gym_tasks GymPJSVD \
    --env $ENV --steps 10000 --hidden-dims '[200,200,200,200]' \
    --n-directions 20 --n-perturbations 100 \
    --perturbation-sizes '[BEST_SCALES_FROM_3abc]' \
    --layer-scope first --target-layer-idx $BEST_LAYER \
    --correction-mode least_squares \
    --pjsvd-family random --safe-subspace-backend projected_residual \
    --subset-size 4096 --probabilistic-base-model \
    --seed 0 --posthoc-calibrate --local-scheduler
done
```

### 3e. Analysis

For each layer and environment, record:

| Layer | Best scale | ID RMSE | ID NLL | ID ECE | Near NLL | Mid NLL | Far NLL | AUROC |
|-------|-----------|---------|--------|--------|----------|---------|---------|-------|
| L1 Lanczos | | | | | | | | |
| L1 Random | | | | | | | | |
| L2 Lanczos | | | | | | | | |
| L2 Random | | | | | | | | |
| L3 Lanczos | | | | | | | | |
| L3 Random | | | | | | | | |

**What to look for:**
- Which layer produces the best NLL improvement with minimal RMSE degradation?
- Does Lanczos consistently beat random? (Theory predicts yes — Lanczos finds the most correctable directions)
- What is the optimal perturbation scale per layer? (Expect deeper layers to need smaller scales due to shorter propagation path)
- Is there a clean U-shaped NLL curve or does accuracy collapse before NLL improves?

---

## Phase 4: Multi-Layer PnC Optimization

**Prerequisite**: Sequential layerwise correction (Phase 0f) must be implemented before running any experiments in this phase. All multi-layer runs use sequential correction at each layer interface.

Use the best single-layer findings from Phase 3 to build multi-layer PnC configurations. Because sequential correction applies at each interface, perturbation scales from single-layer sweeps transfer more directly — each layer's perturbation is corrected locally before affecting the next.

### 4a. Two-layer combinations

Try the top 2 layers identified in Phase 3:

```bash
for ENV in HalfCheetah-v5 Hopper-v5 Ant-v5; do
  # Perturb layers 1+2, sequential correction at each interface
  python -m luigi --module gym_tasks GymPJSVD \
    --env $ENV --steps 10000 --hidden-dims '[200,200,200,200]' \
    --n-directions 20 --n-perturbations 100 \
    --perturbation-sizes '[SCALES]' \
    --layer-scope multi --perturbed-layers '[0,1]' \
    --correction-mode least_squares --use-full-span \
    --pjsvd-family low --safe-subspace-backend projected_residual \
    --subset-size 4096 --probabilistic-base-model \
    --seed 0 --posthoc-calibrate --local-scheduler

  # Perturb layers 2+3, correct at layer 4
  python -m luigi --module gym_tasks GymPJSVD \
    --env $ENV --steps 10000 --hidden-dims '[200,200,200,200]' \
    --n-directions 20 --n-perturbations 100 \
    --perturbation-sizes '[SCALES]' \
    --layer-scope multi --perturbed-layers '[1,2]' \
    --correction-mode least_squares --use-full-span \
    --pjsvd-family low --safe-subspace-backend projected_residual \
    --subset-size 4096 --probabilistic-base-model \
    --seed 0 --posthoc-calibrate --local-scheduler
done
```

### 4b. Three-layer combination (deep multi-layer)

```bash
for ENV in HalfCheetah-v5 Hopper-v5 Ant-v5; do
  python -m luigi --module gym_tasks GymPJSVD \
    --env $ENV --steps 10000 --hidden-dims '[200,200,200,200]' \
    --n-directions 20 --n-perturbations 100 \
    --perturbation-sizes '[SCALES]' \
    --layer-scope multi --perturbed-layers '[0,1,2]' \
    --correction-mode least_squares --use-full-span \
    --pjsvd-family low --safe-subspace-backend projected_residual \
    --subset-size 4096 --probabilistic-base-model \
    --seed 0 --posthoc-calibrate --local-scheduler
done
```

### 4c. Scale adjustment for multi-layer

Multi-layer perturbations compound through the network. Start with per-layer best scales from Phase 3 divided by `sqrt(n_perturbed_layers)`:

```bash
# If best single-layer scale was S, try S/sqrt(2) for 2-layer and S/sqrt(3) for 3-layer
```

### 4d. Analysis

| Config | Env | ID RMSE | ID NLL | Near NLL | Mid NLL | Far NLL | AUROC |
|--------|-----|---------|--------|----------|---------|---------|-------|
| L1 only | HC | | | | | | |
| L1+L2 | HC | | | | | | |
| L2+L3 | HC | | | | | | |
| L1+L2+L3 | HC | | | | | | |

---

## Phase 5: Algorithm Improvement Hypotheses

These are changes to the algorithm that may improve Gym performance while remaining faithful to the paper's method and fair to baselines. Each should be tested as an ablation against the best configuration from Phases 3-4.

### Hypothesis 1: Per-Layer Perturbation Scales

**Problem**: In multi-layer mode, a single `perturbation_scale` is used for all perturbed layers. Phase 3 will likely show that different layers reach the diversity threshold at very different scales. Layer 1 (input→200) may tolerate large perturbations that are easily corrected, while layer 3 (200→200, close to output) may need much smaller perturbations.

**Implementation**: Accept a list of perturbation scales, one per perturbed layer. In `_scale_coefficients_with_member_radii`, split the direction vector into per-layer segments and apply the corresponding scale to each segment.

For Lanczos directions in the joint weight space: the direction `v` has components `[v_l1 | v_l2 | v_l3]`. The per-layer norm `||v_li||` indicates how much of the perturbation goes into each layer. With per-layer scales, we scale each segment independently:
```
delta_W = [scale_1 * v_l1 / ||v_l1|| * ||v||_l1_portion, ...]
```

Simpler approach: run Lanczos separately per layer, producing per-layer direction sets `V1`, `V2`, `V3`. Then each member samples from each layer's subspace independently with different scales.

**Fairness**: This is purely a hyperparameter-space expansion. The algorithm is unchanged.

**Expected effect**: Significant NLL improvement by allowing each layer to contribute optimally.

**Test**: Compare uniform scale (best from Phase 4) vs per-layer scales (best per layer from Phase 3).

### Hypothesis 2: Sigma-Squared Weighting

**Problem**: The current coefficient scaling divides by sigma (singular value): `coeffs = z / sigma`. This means directions with small sigma (most correctable) get amplified the most. The alternative `sigma_sq_weights=True` divides by sigma^2. This gives EVEN MORE weight to the most correctable directions.

**Theoretical motivation**: Theorem 1 shows the bottom singular vectors minimize the projected residual. If we trust the linear approximation, perturbations along these directions have the smallest post-correction residual. Sigma^2 weighting concentrates more perturbation energy along these safest directions, potentially allowing larger overall perturbation scale before accuracy degrades.

**Implementation**: Already implemented — set `--sigma-sq-weights` flag.

**Expected effect**: May allow larger perturbation scales (more diversity) while preserving ID accuracy. The tradeoff is that perturbation diversity concentrates on fewer directions.

**Test**:
```bash
for ENV in HalfCheetah-v5 Hopper-v5 Ant-v5; do
  # Standard (sigma weighting)
  python -m luigi --module gym_tasks GymPJSVD \
    --env $ENV --hidden-dims '[200,200,200,200]' \
    --perturbation-sizes '[BEST_SCALES]' \
    --sigma-sq-weights False --seed 0 --posthoc-calibrate --local-scheduler

  # Sigma-squared weighting
  python -m luigi --module gym_tasks GymPJSVD \
    --env $ENV --hidden-dims '[200,200,200,200]' \
    --perturbation-sizes '[BEST_SCALES]' \
    --sigma-sq-weights True --seed 0 --posthoc-calibrate --local-scheduler
done
```

### Hypothesis 3: Antithetic Pairing

**Problem**: Standard z-coefficient sampling (iid Gaussian) introduces noise in the ensemble mean relative to the base model's prediction. With finite ensemble size (100 members), the mean of the perturbed ensemble deviates slightly from the base model, worsening RMSE.

**Proposed change**: Use antithetic pairing: for each z_i, also include -z_i. This ensures the ensemble mean is exactly the base model mean for the linear component of the perturbation effect. The 100-member ensemble becomes 50 antithetic pairs.

**Implementation**: Already implemented — set `--antithetic-pairing`.

**Expected effect**: Strictly better RMSE preservation (ensemble mean = base model mean in the linear regime). May also improve NLL by reducing noise in the predictive mean while maintaining diversity (variance).

**Fairness**: This is a variance reduction technique, standard in Monte Carlo methods. No additional computation.

**Test**:
```bash
for ENV in HalfCheetah-v5 Hopper-v5 Ant-v5; do
  python -m luigi --module gym_tasks GymPJSVD \
    --env $ENV --hidden-dims '[200,200,200,200]' \
    --perturbation-sizes '[BEST_SCALES]' \
    --antithetic-pairing --seed 0 --posthoc-calibrate --local-scheduler
done
```

### Hypothesis 4: Affine vs Least-Squares Correction (with 4x200)

**Problem**: There are currently two correction modes:
- **Affine**: matches mean + std of perturbed activations to original via next layer rescaling. Fast but only captures first two moments.
- **Least-squares**: solves `[h_new, 1] @ W_aug ≈ Z_target` for the full weight matrix of the next layer. More expressive but may overfit with too few calibration samples.

With 4x200, the least-squares problem is 201×D_out which is well-conditioned for subset_size >= 1024. But the affine correction is O(200) parameters vs O(200 × D_out) for least-squares.

**Key insight**: Affine correction may actually PRODUCE more diversity because it corrects less (only 2 parameters per neuron), leaving more post-correction residual. Least-squares corrects more thoroughly, potentially killing diversity like the over-correction phenomenon seen in CIFAR.

**Test**: Run the best PnC config from Phase 3 with both correction modes and compare:
```bash
for MODE in affine least_squares; do
  for ENV in HalfCheetah-v5 Hopper-v5 Ant-v5; do
    python -m luigi --module gym_tasks GymPJSVD \
      --env $ENV --hidden-dims '[200,200,200,200]' \
      --perturbation-sizes '[BEST_SCALES]' \
      --correction-mode $MODE \
      --seed 0 --posthoc-calibrate --local-scheduler
  done
done
```

**Expected effect**: If over-correction is the bottleneck (NLL never improves before accuracy drops), affine correction should help. If under-correction is the problem (accuracy drops too fast), least-squares should help.

### Hypothesis 5: K (n_directions) Tuning

**Problem**: The number of Lanczos directions K controls the dimensionality of the perturbation subspace. Too few: limited diversity. Too many: perturbations spread across too many directions, some of which may be poorly correctable.

For 4x200 with 200×200 = 40K-dim weight matrices, the effective subspace may be much larger than for 64-dim layers. More directions may be needed.

**Theory**: Coverage probability (Theorem 3 corollary) depends on ensemble size M and subspace dimension K. For M=100, going from K=10 to K=40 should meaningfully improve angular coverage in the perturbation subspace.

**Test**:
```bash
for K in 5 10 20 40 80; do
  for ENV in HalfCheetah-v5 Hopper-v5 Ant-v5; do
    python -m luigi --module gym_tasks GymPJSVD \
      --env $ENV --hidden-dims '[200,200,200,200]' \
      --n-directions $K --perturbation-sizes '[BEST_SCALES]' \
      --seed 0 --posthoc-calibrate --local-scheduler
  done
done
```

### Hypothesis 6: Correction Target — Matching Final Output vs Next-Layer Preactivation

**Problem**: In least-squares correction, the target is the pre-activation of the layer immediately after the last perturbed layer (`target_act = h_old @ W_next + b_next`). But the real objective is to preserve the FINAL output of the network. With 4x200, perturbing l1 and correcting at l2 still leaves 2 more nonlinear layers (l3, l4) through which the corrected-but-imperfect residual propagates and potentially amplifies.

**Proposed change**: Set the correction target to the FINAL output rather than the next-layer pre-activation. For a 4x200 network perturbing l1:
```python
# Current: target_act = h1_orig @ W2 + b2  (next-layer preactivation)
# Proposed: target_act = f(x)  (full model output)
```

The correction then solves: find W2', b2' such that `tail(act(h1_new @ W2' + b2'))` ≈ f(x).

**Challenge**: This is a nonlinear least-squares problem (the tail of the network is nonlinear). Approximate by linearizing: set target_act = final_output and let the least-squares solver find the best linear map from corrected hidden state to the desired final output. This implicitly factors in the amplification through later layers.

**Implementation**:
```python
# In _precompute_least_squares:
if correction_target == "final_output":
    # Z = full model output (instead of next-layer pre-activation)
    Z = model(X_sub)  # shape: (N, output_dim)
    # h_new = last perturbed layer's activation
    # Solve: [h_new, 1] @ W_aug ≈ Z
    # This effectively learns a new mapping from perturbed hidden to output
```

**Caution**: This changes the correction from a local (layer-adjacent) correction to a global one, which may overfit or introduce artifacts. Only try if local correction proves insufficient.

**Test**: Compare next-layer target vs final-output target at the same perturbation scale.

### Hypothesis 7: Adaptive Lambda Regularization

**Problem**: The ridge regression in least-squares correction uses a fixed `lambda_reg` (currently not exposed as a parameter for Gym — uses default from lstsq). Different layers have different conditioning, and the optimal lambda may vary.

For 4x200, all hidden-to-hidden corrections involve a 201×201 Gram matrix. The conditioning depends on the activation distribution at that layer. Earlier layers (with ReLU) may have sparse activations (many zeros) leading to rank-deficient Gram matrices, while later layers may be better conditioned.

**Proposed change**: 
1. Expose `lambda_reg` as a sweep parameter for GymPJSVD
2. Try `lambda_reg` values: 1e-6, 1e-4, 1e-3, 1e-2, 1e-1, 1.0
3. Higher lambda = weaker correction = more residual diversity

**Key insight from CIFAR plan**: Lambda is a DIVERSITY CONTROL KNOB, not just regularization. Higher lambda leaves more correction error, which creates diversity. This may be MORE important than perturbation scale for tuning NLL.

**Test**:
```bash
for LAMBDA in 1e-6 1e-4 1e-3 1e-2 1e-1 1.0; do
  for ENV in HalfCheetah-v5 Hopper-v5 Ant-v5; do
    python -m luigi --module gym_tasks GymPJSVD \
      --env $ENV --hidden-dims '[200,200,200,200]' \
      --perturbation-sizes '[BEST_SCALES]' \
      --lambda-reg $LAMBDA \
      --seed 0 --posthoc-calibrate --local-scheduler
  done
done
```

### Hypothesis 8: Member Radius Distribution

**Problem**: All ensemble members currently have the same perturbation radius (fixed). Having a distribution of radii (some members close to the base model, some far) may create a better-calibrated uncertainty estimate.

**Options already implemented**:
- `fixed`: all members at perturbation_scale
- `lognormal`: radii sampled from lognormal distribution around perturbation_scale
- `two_point`: members split between two distinct radii

**Theoretical motivation**: Different OOD severity requires different scale sensitivity. Members at small radius detect near-OOD (subtle shift), members at large radius detect far-OOD (large shift). A mixture covers the full range.

**Test**:
```bash
for DIST in fixed lognormal; do
  for ENV in HalfCheetah-v5 Hopper-v5 Ant-v5; do
    python -m luigi --module gym_tasks GymPJSVD \
      --env $ENV --hidden-dims '[200,200,200,200]' \
      --perturbation-sizes '[BEST_SCALE]' \
      --member-radius-distribution $DIST \
      --member-radius-std 0.5 \
      --seed 0 --posthoc-calibrate --local-scheduler
  done
done
```

### Hypothesis 9: Per-Dimension Variance Calibration

**Problem**: The current posthoc variance calibration fits a SINGLE scalar multiplier across all output dimensions. But in Gym dynamics models, different state dimensions may have very different prediction uncertainty characteristics (e.g., joint positions vs velocities vs contact forces).

**Proposed change**: Fit a per-output-dimension variance scale: `var_scaled[d] = scale[d] * var_raw[d]`.

**Implementation**: In `_fit_posthoc_variance_scale`:
```python
# Current: scalar scale
raw_scale = jnp.mean(sq_err / (var + 1e-6))

# Proposed: per-dimension scale
raw_scale = jnp.mean(sq_err / (var + 1e-6), axis=0)  # shape: (output_dim,)
```

**Fairness**: Apply per-dimension calibration to ALL methods, not just PnC. This gives every method the same opportunity to benefit.

**Expected effect**: Better NLL and ECE for all methods, but PnC may benefit more because its perturbation-induced variance has structure that varies by output dimension.

---

## Phase 6: N_directions × Subset Size Interaction

### 6a. Subset size sweep

The calibration subset controls the quality of the geometry estimation and correction quality.

```bash
for SS in 512 1024 2048 4096 8000; do
  for ENV in HalfCheetah-v5 Hopper-v5 Ant-v5; do
    python -m luigi --module gym_tasks GymPJSVD \
      --env $ENV --hidden-dims '[200,200,200,200]' \
      --perturbation-sizes '[BEST_SCALES]' \
      --subset-size $SS --n-directions 20 \
      --seed 0 --posthoc-calibrate --local-scheduler
  done
done
```

**What to look for**:
- `subset_size < 201` (hidden dim + 1): least-squares will be underdetermined. Must avoid.
- `subset_size ≈ 500-1000`: may underfit correction → more diversity → possibly better NLL at cost of RMSE
- `subset_size ≈ 4000-8000`: stable correction → good RMSE preservation → may need larger perturbation scale for diversity
- There may be a sweet spot where correction is "good enough" for ID preservation but imperfect enough for OOD diversity.

---

## Phase 7: Final Runs

Once the best configuration is found (after Phases 3-6), run with multiple seeds.

### 7a. Best PnC configurations

```bash
for SEED in 0 10 200; do
  for ENV in HalfCheetah-v5 Hopper-v5 Ant-v5 Humanoid-v5; do
    # Best single-layer config (for ablation table)
    python -m luigi --module gym_tasks GymPJSVD \
      --env $ENV --steps 10000 --hidden-dims '[200,200,200,200]' \
      --n-directions $BEST_K --n-perturbations 100 \
      --perturbation-sizes '[$BEST_SCALE]' \
      --layer-scope first --target-layer-idx $BEST_LAYER \
      --correction-mode $BEST_MODE \
      --pjsvd-family low --safe-subspace-backend projected_residual \
      --subset-size $BEST_SS --probabilistic-base-model \
      --seed $SEED --posthoc-calibrate --local-scheduler

    # Best multi-layer config (main result)
    python -m luigi --module gym_tasks GymPJSVD \
      --env $ENV --steps 10000 --hidden-dims '[200,200,200,200]' \
      --n-directions $BEST_K --n-perturbations 100 \
      --perturbation-sizes '[$BEST_SCALES]' \
      --layer-scope multi --perturbed-layers '[$BEST_LAYERS]' \
      --correction-mode $BEST_MODE \
      --pjsvd-family low --safe-subspace-backend projected_residual \
      --subset-size $BEST_SS --probabilistic-base-model \
      --antithetic-pairing \
      --seed $SEED --posthoc-calibrate --local-scheduler

    # Random directions ablation
    python -m luigi --module gym_tasks GymPJSVD \
      --env $ENV --steps 10000 --hidden-dims '[200,200,200,200]' \
      --n-directions $BEST_K --n-perturbations 100 \
      --perturbation-sizes '[$BEST_SCALES]' \
      --layer-scope multi --perturbed-layers '[$BEST_LAYERS]' \
      --correction-mode $BEST_MODE \
      --pjsvd-family random --safe-subspace-backend projected_residual \
      --subset-size $BEST_SS --probabilistic-base-model \
      --antithetic-pairing \
      --seed $SEED --posthoc-calibrate --local-scheduler
  done
done
```

### 7b. Generate report

```bash
python report.py --results_dir results --env HalfCheetah-v5 --fmt md
python report.py --results_dir results --env Hopper-v5 --fmt md
python report.py --results_dir results --env Ant-v5 --fmt md
python report.py --results_dir results --env Humanoid-v5 --fmt md
```

---

## Decision Tree for Failures

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| 4x200 base model worse RMSE than 2x64 | Underfitting or overfitting | Increase training steps; tune weight decay; try lr=5e-4 |
| NLL flat across all scales (no diversity) | Correction too effective | Increase lambda_reg to 0.1 or 1.0; try affine instead of least_squares; reduce subset_size |
| Accuracy drops >5% before NLL improves | Perturbation destroys structure | Use Lanczos instead of random; reduce scale and increase lambda instead; try sigma_sq weighting |
| NLL improves but ECE bad | Variance calibration issue | Check posthoc_variance_scale value; if extreme, try per-dimension calibration (Hypothesis 9) |
| Lanczos ≈ Random for all layers | Subspace selection not important | Use random (cheaper). Focus on scale + lambda tuning |
| Multi-layer much worse than single-layer | Compounding perturbation effects | Verify sequential layerwise correction is active (Phase 0f); reduce scale by sqrt(n_layers) |
| Lanczos takes >30 min per layer | Large weight matrices (40K dims) | Reduce subset_size; use fewer Lanczos iterations; or use random as starting point |
| SWAG/Laplace OOM with 4x200 | KFAC factors or SWAG covariance too large | Reduce subset_size for KFAC; use diagonal SWAG approximation |
| Posthoc variance scale extreme (>100 or <0.01) | Raw variance scale mismatch | Normal for some methods; the posthoc step corrects it. But if residual NLL still bad, investigate raw predictions |
| One environment breaks while others work | Environment-specific dynamics | Tune per-environment; Humanoid (376 dims) may need different hyperparameters |

---

## Success Criteria

The final paper tables should show:

1. **ID RMSE**: PnC within 5% of base model RMSE (perturbation preserves accuracy)
2. **ID NLL**: PnC ≤ best baseline NLL (better or equal predictive uncertainty on ID)
3. **Shifted NLL (Near/Mid/Far)**: PnC ≤ best baseline NLL for each shift level (better calibration under distribution shift)
4. **ECE**: PnC ≤ best baseline ECE
5. **Far AUROC**: PnC competitive with or better than baselines (ideally best among non-Deep-Ensemble methods)
6. **Monotone uncertainty increase**: NLL and variance increase monotonically from ID → Near → Mid → Far
7. **Deep Ensemble**: Reported as reference but not required to beat

### Ablations reported:
- Single-layer vs multi-layer PnC
- Lanczos vs random directions
- Per-layer scale analysis (which layers matter most)
- Affine vs least-squares correction
- Effect of lambda_reg (diversity control)
- Antithetic pairing effect
- Raw vs VCal results (both tables)

---

## Priority Order

If time is limited, execute phases in this order:

1. **Phase 0** (code changes + sequential correction) — required for everything else
2. **Phase 1** (training calibration) — quick, 1-2 hours
3. **Phase 2b, 2c, 2e** (Deep Ensemble, MC Dropout, Laplace baselines) — these are the main comparisons
4. **Phase 3a-3c** (single-layer scale discovery) — core PnC tuning
5. **Phase 5, Hypothesis 3** (antithetic pairing) — free performance, no code changes
6. **Phase 5, Hypothesis 4** (affine vs least-squares) — easy comparison
7. **Phase 5, Hypothesis 7** (lambda sweep) — high impact, easy to run
8. **Phase 4a** (two-layer multi-layer) — core multi-layer result
9. **Phase 5, Hypothesis 5** (K tuning) — refine direction count
10. **Phase 2d, 2f** (SWAG, Subspace Inference) — secondary baselines
11. **Phase 5, remaining hypotheses** — if time permits
12. **Phase 7** (final multi-seed runs) — publication-ready
