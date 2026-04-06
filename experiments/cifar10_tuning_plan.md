# CIFAR-10 PnC Tuning Plan

## Goal

Tune single-block and multi-block PnC on CIFAR-10 so that the multi-block algorithm matches or beats MC Dropout, SWAG, and LLLA on **accuracy** (maintain), **NLL** (minimize), and **ECE** (minimize). Deep Ensemble is reported but not a target to beat. All comparisons use post-hoc temperature scaling uniformly.

## Hardware Constraint: 8GB VRAM

All commands and hyperparameter choices account for an 8GB GPU. OOM-critical numbers:

| Block | Conv1 params | K=10 dirs (MB) | K=20 dirs (MB) | H matrix (MB) | Max safe K |
|-------|-------------|----------------|----------------|---------------|-----------|
| Stage 1 (64ch) | 36,864 | 1.4 | 2.8 | 1.3 | 40 |
| Stage 2 (128ch) | 73K-147K | 2.8-5.6 | 5.6-11.2 | 5.1 | 40 |
| Stage 3 (256ch) | 295K-590K | 11-22 | 22-45 | 20.3 | 20 |
| Stage 4 (512ch) | 1.2M-2.4M | 45-90 | 90-180 | 81.0 | 10 |

**Rules of thumb for 8GB:**
- `chunk_size=64` is safe for all stages. Use 128 only for stages 3-4.
- `subset_size=1024` is safe. 2048 requires `chunk_size=64`. 4096 is risky for stage 4.
- For multi-block Lanczos (sequential over 8 blocks), call `jax.clear_caches()` between blocks (already done in `CIFARMultiBlockPnC.run()`).
- LLLA GGN is ~100MB for CIFAR-10 (feasible). CIFAR-100 LLLA will OOM (10GB).
- Deep Ensemble loads 5 models; evaluate one at a time if needed.

---

## Critical Fix: Patch Ordering (commit 4ed1527)

The original code had a **conv kernel flatten ordering bug**: `extract_patches` outputs in `(C_in, kh, kw)` order (channel-major), but W2 kernels were being flattened as `(kh, kw, C_in)` (spatial-major). The ridge regression solved in a scrambled basis — producing numerically small residuals that didn't correspond to meaningful corrections.

**Impact (S3B1, K=16, n=32, random, seed 0):**

| Scale | Acc (bug→fix) | NLL (bug→fix) | Temp (bug→fix) |
|-------|--------------|--------------|----------------|
| 10.0 | 92.70%→95.83% | 0.236→0.137 | 0.270→1.260 |
| 50.0 | 92.40%→95.61% | 0.335→0.139 | 0.126→0.999 |

The fix eliminates the 3% accuracy drop and 0.1 NLL penalty. **All PnC results generated before this fix are invalid.**

### Post-Fix Tuning Challenge

With correct patch ordering, the Conv2 correction is highly effective: at scale=50 the correction reduces block-output shift by 87%. This means ensemble members produce nearly identical predictions — PnC matches the base model NLL (~0.138) but doesn't improve it. The original plan's scale ranges (0.001–5.0 for Stage 4) are far too small to create meaningful diversity.

**New strategy**: Explore much larger scales to find the "diversity threshold" where residual correction error creates useful predictive diversity, and tune `lambda_reg` as a second diversity control knob (higher lambda = weaker correction = more diversity = potentially better NLL at some accuracy cost).

---

## Fairness Protocol

1. **Shared base model**: All methods except Deep Ensemble and MC Dropout use the same `CIFARTrainPreActResNet18` checkpoint (same seed, same recipe).
2. **Post-hoc temperature scaling** (`--posthoc-calibrate`): Applied uniformly to every method. Temperature fit on the 10% validation split (5000 samples). This isolates uncertainty signal quality from scale mismatch.
3. **Hyperparameter selection**: For sweep parameters (perturbation_scale, Laplace prior), select by **best test NLL after posthoc calibration**. The temperature is fit on the validation split, so test NLL is uncontaminated. Apply the same selection protocol to all methods.
4. **Training recipe**: 300 epochs, SGD+momentum 0.9, nesterov, lr=0.1, cosine decay with 5-epoch warmup, weight_decay=5e-4, batch_size=128, cutout=8. This matches standard PreActResNet-18 literature recipes.
5. **Same ensemble size**: n_perturbations=50 for PnC, MC Dropout, SWAG. n_models=5 for Deep Ensemble.

---

## Measured Baseline Performance

All baselines use post-hoc temperature scaling and 300-epoch PreActResNet-18 training.

| Method | Accuracy | NLL | ECE | Seeds | Notes |
|--------|----------|-----|-----|-------|-------|
| Single model | 95.74±0.18% | 0.144±0.005 | 0.010±0.001 | 3 | |
| Deep Ensemble (5) | 96.51% | 0.108 | 0.005 | 1 | Gold standard |
| MC Dropout (n=50) | 95.74±0.14% | 0.148±0.006 | 0.010±0.004 | 3 | |
| LLLA (prior=10, n=50) | 95.77±0.26% | 0.140±0.006 | 0.008±0.002 | 3 | Best prior from sweep of 0.01–1000 |
| SWAG (sws=240) | — | — | — | 0 | Not yet run; sws=160 was broken |

**Targets for PnC to beat** (best of MC Dropout and LLLA):
- Accuracy: maintain within 0.5% of 95.74%
- NLL: < 0.140 (LLLA's mean)
- ECE: < 0.008 (LLLA's mean)

---

## Phase 0: Baselines ✓ (mostly complete)

### Completed
- Single model eval: seeds 0,1,2 ✓
- LLLA prior sweep (0.01–1000): seeds 0,1,2 ✓ → **prior=10.0 is best**
- MC Dropout (n=50, dr=0.1): seeds 0,1,2 ✓
- Deep Ensemble (n=5): seed 0 ✓

### Remaining
```bash
# SWAG with corrected start epoch
for SEED in 0 1 2; do
  python -m luigi --module cifar_tasks CIFARPreActSWAG \
    --dataset cifar10 --epochs 300 --n-perturbations 50 \
    --swag-start-epoch 240 --swag-collect-freq 1 \
    --swag-use-bn-refresh --bn-refresh-subset-size 2048 \
    --seed $SEED --posthoc-calibrate --local-scheduler
done

# Deep Ensemble seeds 1,2 (need base models 5,6 first)
for SEED in 5 6; do
  python -m luigi --module cifar_tasks CIFARTrainPreActResNet18 \
    --dataset cifar10 --epochs 300 --seed $SEED --local-scheduler
done
for SEED in 0 1 2; do
  python -m luigi --module cifar_tasks CIFARStandardEnsemble \
    --dataset cifar10 --epochs 300 --n-models 5 --seed $SEED \
    --posthoc-calibrate --local-scheduler
done
```

---

## Phase 1: Code Audit ✓

Confirmed by re-running PnC S3B1 after the patch ordering fix (commit 4ed1527):
- Pipeline runs end-to-end ✓
- Accuracy preserved (95.83% at scale=10) ✓
- No NaN in metrics ✓
- diag_test_reduction > 0 (37–87%) ✓
- Memory under 8GB ✓

---

## Phase 2: Per-Block Scale Discovery (Single-Block Sweeps)

**Key insight (revised)**: With the patch fix, the correction is so effective that scales up to 50 produce almost no diversity. We need to explore scales well beyond the correction capacity to find useful NLL improvement. Scale ranges are shifted 10–100x higher than the original plan.

**What we're looking for**: The "sweet spot" where the correction starts failing enough that residual error creates predictive diversity (NLL improves), but not so much that accuracy collapses. Expect a U-shaped NLL curve: flat at small scales (no diversity), improving as diversity increases, then worsening as accuracy drops.

### 2a. Stage 4 blocks (largest, most direct effect on output)

```bash
# Stage 4 Block 1 (last block -- closest to output)
python -m luigi --module cifar_tasks CIFARPnC \
  --dataset cifar10 --epochs 300 --n-directions 10 --n-perturbations 50 \
  --perturbation-sizes '[1.0,5.0,10.0,50.0,100.0,200.0,500.0]' \
  --target-stage-idx 3 --target-block-idx 1 \
  --lambda-reg 1e-3 --subset-size 1024 --chunk-size 64 \
  --seed 0 --posthoc-calibrate --local-scheduler

# Stage 4 Block 0
python -m luigi --module cifar_tasks CIFARPnC \
  --dataset cifar10 --epochs 300 --n-directions 10 --n-perturbations 50 \
  --perturbation-sizes '[1.0,5.0,10.0,50.0,100.0,200.0,500.0]' \
  --target-stage-idx 3 --target-block-idx 0 \
  --lambda-reg 1e-3 --subset-size 1024 --chunk-size 64 \
  --seed 0 --posthoc-calibrate --local-scheduler
```

### 2b. Stage 3 blocks

```bash
for BLOCK in 0 1; do
  python -m luigi --module cifar_tasks CIFARPnC \
    --dataset cifar10 --epochs 300 --n-directions 15 --n-perturbations 50 \
    --perturbation-sizes '[5.0,10.0,50.0,100.0,200.0,500.0,1000.0]' \
    --target-stage-idx 2 --target-block-idx $BLOCK \
    --lambda-reg 1e-3 --subset-size 1024 --chunk-size 64 \
    --seed 0 --posthoc-calibrate --local-scheduler
done
```

### 2c. Stage 2 blocks

```bash
for BLOCK in 0 1; do
  python -m luigi --module cifar_tasks CIFARPnC \
    --dataset cifar10 --epochs 300 --n-directions 20 --n-perturbations 50 \
    --perturbation-sizes '[10.0,50.0,100.0,200.0,500.0,1000.0,2000.0]' \
    --target-stage-idx 1 --target-block-idx $BLOCK \
    --lambda-reg 1e-3 --subset-size 1024 --chunk-size 64 \
    --seed 0 --posthoc-calibrate --local-scheduler
done
```

### 2d. Stage 1 blocks

```bash
for BLOCK in 0 1; do
  python -m luigi --module cifar_tasks CIFARPnC \
    --dataset cifar10 --epochs 300 --n-directions 20 --n-perturbations 50 \
    --perturbation-sizes '[50.0,100.0,200.0,500.0,1000.0,2000.0,5000.0]' \
    --target-stage-idx 0 --target-block-idx $BLOCK \
    --lambda-reg 1e-3 --subset-size 1024 --chunk-size 64 \
    --seed 0 --posthoc-calibrate --local-scheduler
done
```

### 2e. Random directions ablation (one per stage, best block)

```bash
# Run on the block in each stage that had the best single-block NLL
# (fill in STAGE, BLOCK, SCALES after 2a-2d)
python -m luigi --module cifar_tasks CIFARPnC \
  --dataset cifar10 --epochs 300 --n-directions 10 --n-perturbations 50 \
  --perturbation-sizes '[SCALES]' \
  --target-stage-idx $STAGE --target-block-idx $BLOCK \
  --lambda-reg 1e-3 --subset-size 1024 --chunk-size 64 \
  --seed 0 --posthoc-calibrate --random-directions --local-scheduler
```

### 2f. Analysis

For each block, record:

| Block | Best scale | Accuracy | NLL | ECE | diag_calib_reduction% | diag_test_reduction% |
|-------|-----------|----------|-----|-----|----------------------|---------------------|
| S1B0 | | | | | | |
| S1B1 | | | | | | |
| S2B0 | | | | | | |
| S2B1 | | | | | | |
| S3B0 | | | | | | |
| S3B1 | | | | | | |
| S4B0 | | | | | | |
| S4B1 | | | | | | |

**What to look for (revised after fix):**
- The scale at which NLL starts improving over the base model (0.138) — this is the diversity threshold
- The scale at which accuracy drops more than 0.5% — this is the accuracy limit
- The sweet spot between these two thresholds (if it exists)
- If no sweet spot exists (NLL never improves before accuracy collapses), move to Phase 3 (lambda_reg tuning) immediately

---

## Phase 3: Diversity Control via Lambda Tuning

**Key insight**: `lambda_reg` controls the correction-diversity tradeoff. Higher lambda = worse correction = more residual diversity. This may be MORE important than perturbation scale for creating useful ensembles.

### 3a. Lambda sweep on best block from Phase 2

```bash
for LAMBDA in 1e-4 1e-3 1e-2 1e-1 1.0; do
  python -m luigi --module cifar_tasks CIFARPnC \
    --dataset cifar10 --epochs 300 --n-directions BEST_K --n-perturbations 50 \
    --perturbation-sizes '[BEST_SCALES]' \
    --target-stage-idx $BEST_STAGE --target-block-idx $BEST_BLOCK \
    --lambda-reg $LAMBDA --subset-size 1024 --chunk-size 64 \
    --seed 0 --posthoc-calibrate --local-scheduler
done
```

### 3b. Scale × lambda grid (if Phase 3a shows lambda matters)

Run a coarse 2D grid of (scale, lambda) to find the joint optimum:

```bash
for SCALE in PROMISING_SCALES; do
  for LAMBDA in PROMISING_LAMBDAS; do
    python -m luigi --module cifar_tasks CIFARPnC \
      --dataset cifar10 --epochs 300 --n-directions BEST_K --n-perturbations 50 \
      --perturbation-sizes "[$SCALE]" \
      --target-stage-idx $BEST_STAGE --target-block-idx $BEST_BLOCK \
      --lambda-reg $LAMBDA --subset-size 1024 --chunk-size 64 \
      --seed 0 --posthoc-calibrate --local-scheduler
  done
done
```

### 3c. Subset size variation

Also try `subset_size=512` (weaker correction → more diversity) and `subset_size=2048` (stronger correction → less diversity):

```bash
for SS in 512 2048; do
  python -m luigi --module cifar_tasks CIFARPnC \
    --dataset cifar10 --epochs 300 --n-directions BEST_K --n-perturbations 50 \
    --perturbation-sizes '[BEST_SCALES]' \
    --target-stage-idx $BEST_STAGE --target-block-idx $BEST_BLOCK \
    --lambda-reg BEST_LAMBDA --subset-size $SS --chunk-size 64 \
    --seed 0 --posthoc-calibrate --local-scheduler
done
```

---

## Phase 4: Multi-Block PnC

### 4a. First attempt with uniform scale

Use the best per-block scale from Phase 2, divided by sqrt(8) to account for compounding:

```bash
python -m luigi --module cifar_tasks CIFARMultiBlockPnC \
  --dataset cifar10 --epochs 300 --n-directions 10 --n-perturbations 50 \
  --perturbation-sizes '[SCALE/4, SCALE/2, SCALE, SCALE*2]' \
  --lambda-reg BEST_LAMBDA --subset-size 1024 --chunk-size 64 \
  --seed 0 --posthoc-calibrate --local-scheduler
```

**OOM note**: Multi-block runs Lanczos on all 8 blocks sequentially, then solves 8 * n_perturbations ridge regressions. With K=10 and chunk_size=64, this should fit in 8GB. If OOM, reduce subset_size to 512 first.

### 4b. Per-block scale support (code change -- see Hypothesis 2)

If uniform scale is insufficient because some blocks need very different scales, implement per-block `perturbation_scale` in `MultiBlockPnCEnsemble._coeffs_from_z()`. This is the most impactful code change for multi-block performance.

### 4c. Refine multi-block

```bash
# Fine sweep around best multi-block scale
python -m luigi --module cifar_tasks CIFARMultiBlockPnC \
  --dataset cifar10 --epochs 300 --n-directions 10 --n-perturbations 50 \
  --perturbation-sizes '[FINE_RANGE]' \
  --lambda-reg BEST_LAMBDA --subset-size 1024 --chunk-size 64 \
  --seed 0 --posthoc-calibrate --local-scheduler

# Try sigma_sq_weights
python -m luigi --module cifar_tasks CIFARMultiBlockPnC \
  --dataset cifar10 --epochs 300 --n-directions 10 --n-perturbations 50 \
  --perturbation-sizes '[BEST_SCALES]' \
  --lambda-reg BEST_LAMBDA --subset-size 1024 --chunk-size 64 \
  --sigma-sq-weights \
  --seed 0 --posthoc-calibrate --local-scheduler
```

---

## Phase 5: Code Improvement Hypotheses

These are changes to the algorithm/code that may improve CIFAR performance while remaining faithful to the paper and fair to baselines. Each should be tested as an ablation against the baseline PnC configuration from Phase 4.

### Hypothesis 1: BN2 Running-Stat Refit After Conv1 Perturbation

**Problem**: `make_cifar_block_get_Y_fn()` applies BN2 with frozen running stats (from training on the original Conv1). When Conv1 is perturbed, the distribution feeding BN2 changes, so the running mean/var are stale. The ridge regression must implicitly compensate for this mismatch via the delta formulation, wasting correction capacity.

**Post-fix reassessment**: After the patch ordering fix, the correction is *already* highly effective (87% reduction at scale=50). BN2 refit would make it even better — but the current problem is *too little* diversity, not too much correction error. BN2 refit may be **counterproductive for diversity** unless paired with much larger perturbation scales.

**Change**: After perturbing Conv1, run a forward pass through the calibration subset to compute new BN2 running stats (mean, var), then use those in `get_Y_fn` and in `_forward_member`. This matches what SWAG does (BN refresh) and is standard practice in weight-perturbation ensembles.

**When to try**: Only if Phase 2–3 show that correction quality (not diversity) is the bottleneck. If the bottleneck is diversity, skip this hypothesis.

**Test**: Compare diag_reduction_pct and NLL with vs without BN2 refit at the same perturbation_scale.

### Hypothesis 2: Per-Block Perturbation Scale in Multi-Block

**Problem**: `MultiBlockPnCEnsemble` uses a single `perturbation_scale` for all 8 blocks. Phase 2 will likely show different blocks reach the diversity threshold at very different scales. A uniform scale forces a compromise.

**Change**: Accept a list of perturbation scales (one per block) or a dict mapping block index to scale. In `_coeffs_from_z()`, use the block-specific scale instead of the global one.

**Fairness**: This is purely a hyperparameter-space expansion. The algorithm is unchanged; we're just tuning more carefully.

**Implementation**: Add `perturbation_scales: list[float]` parameter to `MultiBlockPnCEnsemble.__init__()`. In `_coeffs_from_z()`, accept a `block_index` argument and use `self.perturbation_scales[block_index]`. Update `CIFARMultiBlockPnC` to accept a list.

**Expected effect**: Significant NLL improvement by allowing each block to contribute optimally.

**Test**: Compare multi-block with uniform scale (best from Phase 4a) vs per-block scales (best per block from Phase 2).

### Hypothesis 3: Regularization-Weighted Ridge (Lambda Per Block)

**Problem**: A single `lambda_reg` is used for all blocks, but the condition number of H = M^T M varies dramatically: Stage 1 H is 577x577 and well-conditioned (many spatial patches), Stage 4 H is 4609x4609 and may be ill-conditioned (few spatial patches per image at 4x4 resolution).

**Post-fix context**: Lambda is now understood as a **diversity control knob**, not just a regularization parameter. Per-block lambda would allow fine-grained control of how much correction error (= diversity) each block contributes.

**Change**: Allow per-block `lambda_reg`, or auto-scale lambda proportional to trace(H)/dim(H) (the mean eigenvalue), making regularization adapt to each block's conditioning.

**Implementation**: In `solve_chunked_conv2_correction()`, after accumulating H, optionally compute `lambda_auto = trace(H) / D_M * lambda_relative` where `lambda_relative` is a user-specified fraction. Or simply accept a list of lambdas in `CIFARMultiBlockPnC`.

**Test**: Compare fixed lambda_reg=1e-3 vs auto-scaled lambda vs per-block lambda sweep.

### Hypothesis 4: Ensemble Prediction via Log-Probability Averaging

**Problem**: `_evaluate_cifar()` averages softmax probabilities across ensemble members (`mean = jnp.mean(preds, axis=0)`). For classification, geometric averaging in log-space (averaging logits, then softmax) can produce better-calibrated predictions, especially when ensemble members have different confidence levels.

**Change**: Add a `logit_averaging` option to `_evaluate_cifar()` that averages logits before applying softmax, instead of averaging probabilities after softmax.

**Fairness**: Both probability averaging and logit averaging are standard in the ensemble literature. The choice should be reported. To be fair, test both modes for all methods (not just PnC).

**Implementation**: In `_evaluate_cifar()`, add a flag. If True: `mean_logits = jnp.mean(logits / temperature, axis=0); mean = jax.nn.softmax(mean_logits)`. If False: current behavior.

**Expected effect**: May improve NLL and ECE for PnC specifically because PnC members have perturbation-induced logit shifts that average more naturally in logit space. May help or hurt baselines depending on their diversity characteristics.

**Test**: Run the best PnC config and all baselines with both averaging modes. Report whichever mode is better for each method (or pick one mode for all -- fairer but may disadvantage some methods).

---

## Phase 6: Final Runs

Once the best configuration is found (after Phases 2-5), run with multiple seeds:

```bash
for SEED in 0 1 2; do
  # Best single-block config (for ablation table)
  python -m luigi --module cifar_tasks CIFARPnC \
    --dataset cifar10 --epochs 300 \
    --n-directions $BEST_K --n-perturbations 50 \
    --perturbation-sizes '[$BEST_SCALE]' \
    --target-stage-idx $BEST_STAGE --target-block-idx $BEST_BLOCK \
    --lambda-reg $BEST_LAMBDA --subset-size $BEST_SS --chunk-size 64 \
    --seed $SEED --posthoc-calibrate --local-scheduler

  # Best multi-block config (main result)
  python -m luigi --module cifar_tasks CIFARMultiBlockPnC \
    --dataset cifar10 --epochs 300 \
    --n-directions $BEST_K --n-perturbations 50 \
    --perturbation-sizes '[$BEST_SCALE]' \
    --lambda-reg $BEST_LAMBDA --subset-size $BEST_SS --chunk-size 64 \
    --seed $SEED --posthoc-calibrate --local-scheduler

  # Best multi-block with random directions (ablation)
  python -m luigi --module cifar_tasks CIFARMultiBlockPnC \
    --dataset cifar10 --epochs 300 \
    --n-directions $BEST_K --n-perturbations 50 \
    --perturbation-sizes '[$BEST_SCALE]' \
    --lambda-reg $BEST_LAMBDA --subset-size $BEST_SS --chunk-size 64 \
    --seed $SEED --posthoc-calibrate --random-directions --local-scheduler
done
```

### Final Report

```bash
python report.py --results_dir results --env cifar10 --fmt md
```

---

## Decision Tree for Failures

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| OOM during Lanczos | K too large for this stage | Reduce K; for Stage 4 use K<=10 |
| OOM during ridge solve | chunk_size or subset_size too large | Reduce chunk_size to 32; reduce subset_size |
| OOM during Deep Ensemble eval | 5 models loaded | Eval one model at a time, stack predictions |
| NLL flat across all scales (no diversity) | Correction too effective at current lambda | Increase lambda_reg by 10x; try lambda=0.1 or 1.0 |
| Accuracy drops >1% before NLL improves | Perturbation destroys structure too fast | Try Lanczos directions instead of random; reduce scale and increase lambda instead |
| NLL improves but accuracy drops >1% | Diversity threshold crossed | Use slightly smaller scale; or increase lambda_reg instead of scale |
| diag_calib_reduction_pct < 50% | Ridge underfitting | Reduce lambda_reg by 10x; ensure chunk_size isn't too small |
| diag_test_reduction << diag_calib_reduction | Correction overfitting | Increase lambda_reg or subset_size |
| NLL good but ECE bad | Temperature scaling edge case | Check posthoc_temperature; if extreme (>5 or <0.1), try Hypothesis 4 (logit averaging) |
| Lanczos ~ Random for all blocks | Subspace selection not important here | Fine -- use random (cheaper). Focus on scale and lambda tuning. |
| Multi-block much worse than single-block | Compounding perturbation effects | Implement per-block scale (Hypothesis 2); reduce scale by sqrt(n_blocks) |
| Multi-block Stage 4 ridge fails | H matrix ill-conditioned | Increase lambda_reg for Stage 4 specifically (Hypothesis 3) |
| SWAG NLL much worse than expected | BN refresh not working | Verify swag_use_bn_refresh=True and bn_refresh_subset_size >= 2048 |

---

## Success Criteria

The final paper table should show:

1. **Accuracy**: Multi-block PnC within 0.5% of MC Dropout, SWAG, and LLLA
2. **NLL**: Multi-block PnC <= min(MC Dropout NLL, SWAG NLL, LLLA NLL) = 0.140
3. **ECE**: Multi-block PnC <= min(MC Dropout ECE, SWAG ECE, LLLA ECE) = 0.008
4. **Deep Ensemble**: Reported as a reference point but not a target to beat
5. **Ablations reported**: Single-block vs multi-block, Lanczos vs random, per-block scale effect
