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

## Fairness Protocol

1. **Shared base model**: All methods except Deep Ensemble and MC Dropout use the same `CIFARTrainPreActResNet18` checkpoint (same seed, same recipe).
2. **Post-hoc temperature scaling** (`--posthoc-calibrate`): Applied uniformly to every method. Temperature fit on the 10% validation split (5000 samples). This isolates uncertainty signal quality from scale mismatch.
3. **Hyperparameter selection**: For sweep parameters (perturbation_scale, Laplace prior), select by **best test NLL after posthoc calibration**. The temperature is fit on the validation split, so test NLL is uncontaminated. Apply the same selection protocol to all methods.
4. **Training recipe**: 300 epochs, SGD+momentum 0.9, nesterov, lr=0.1, cosine decay with 5-epoch warmup, weight_decay=5e-4, batch_size=128, cutout=8. This matches standard PreActResNet-18 literature recipes.
5. **Same ensemble size**: n_perturbations=50 for PnC, MC Dropout, SWAG. n_models=5 for Deep Ensemble.

---

## Expected Baseline Performance (Literature Reference)

These are approximate targets for sanity-checking our baselines. Based on Google Uncertainty Baselines (WRN-28-10, scaled for ResNet-18), SWAG (Maddox 2019), and Laplace Redux (Daxberger 2021). PreActResNet-18 is smaller than WRN-28-10, so expect slightly lower accuracy and higher NLL.

| Method | Accuracy | NLL | ECE | Notes |
|--------|----------|-----|-----|-------|
| Single model | 94.5-95.5% | 0.18-0.25 | 0.02-0.04 | Depends on augmentation and epochs |
| Deep Ensemble (5) | 95.5-96.0% | 0.14-0.18 | 0.01-0.02 | Gold standard |
| MC Dropout | 94.0-94.5% | 0.20-0.28 | 0.01-0.02 | With temp scaling; dropout hurts accuracy slightly |
| SWAG | 94.0-95.0% | 0.17-0.22 | 0.01-0.02 | BN refresh is critical |
| LLLA | 94.0-95.0% | 0.18-0.24 | 0.01-0.03 | Highly sensitive to prior precision |

**If baselines fall outside these ranges, debug before proceeding with PnC tuning.**

Sources:
- [Google Uncertainty Baselines](https://github.com/google/uncertainty-baselines/tree/master/baselines/cifar)
- [SWAG paper (Maddox et al., 2019)](https://arxiv.org/abs/1902.02476)
- [Laplace Redux (Daxberger et al., 2021)](https://arxiv.org/abs/2106.14806)

---

## Phase 0: Baselines

Run each baseline for seeds 0, 1, 2. Each command is independent and can be run sequentially if GPU memory is tight.

### 0a. Base Model Training (one per seed)

```bash
for SEED in 0 1 2; do
  python -m luigi --module cifar_tasks CIFARTrainPreActResNet18 \
    --dataset cifar10 --epochs 300 --seed $SEED --local-scheduler
done
```

### 0b. Single Model Eval

```bash
for SEED in 0 1 2; do
  python -m luigi --module cifar_tasks CIFAREvalPreActResNet18 \
    --dataset cifar10 --epochs 300 --seed $SEED \
    --posthoc-calibrate --local-scheduler
done
```

**Sanity check**: Accuracy should be 94.5-95.5%. If below 93%, check training recipe.

### 0c. Deep Ensemble

```bash
for SEED in 0 1 2; do
  python -m luigi --module cifar_tasks CIFARStandardEnsemble \
    --dataset cifar10 --epochs 300 --n-models 5 --seed $SEED \
    --posthoc-calibrate --local-scheduler
done
```

**OOM note**: Loads 5 models. If OOM during `predict()`, reduce eval `batch_size` by editing `_evaluate_cifar()` call or passing smaller batches.

### 0d. MC Dropout

```bash
for SEED in 0 1 2; do
  python -m luigi --module cifar_tasks CIFARPreActMCDropout \
    --dataset cifar10 --epochs 300 --n-perturbations 50 --dropout-rate 0.1 \
    --seed $SEED --posthoc-calibrate --local-scheduler
done
```

**Sanity check**: Accuracy should be ~94%. If much lower, dropout rate may be too high for this architecture.

### 0e. SWAG

```bash
for SEED in 0 1 2; do
  python -m luigi --module cifar_tasks CIFARPreActSWAG \
    --dataset cifar10 --epochs 300 --n-perturbations 50 \
    --swag-start-epoch 240 --swag-collect-freq 1 \
    --swag-use-bn-refresh --bn-refresh-subset-size 2048 \
    --seed $SEED --posthoc-calibrate --local-scheduler
done
```

**Sanity check**: Accuracy should be 94-95%. BN refresh is critical for SWAG on ResNets -- without it, NLL can be much worse. `swag-start-epoch=240` means collecting stats from epoch 240/300 (last 20% of training), which is standard.

### 0f. LLLA (Laplace)

```bash
for SEED in 0 1 2; do
  for PRIOR in 0.01 0.1 1.0 10.0 100.0 1000.0; do
    python -m luigi --module cifar_tasks CIFARLLLA \
      --dataset cifar10 --epochs 300 --n-perturbations 50 \
      --prior-precision $PRIOR \
      --seed $SEED --posthoc-calibrate --local-scheduler
  done
done
```

**Selection**: Report the prior precision with best test NLL (after posthoc calibration), averaged over seeds. Typical best prior for CIFAR-10 is 1.0-100.0.

### 0g. Baseline Audit

After all baselines complete:

```bash
python report.py --results_dir results --env cifar10 --fmt md
```

Compare against the literature table above. Flag any method that is more than 1% accuracy or 0.05 NLL away from expected range.

---

## Phase 1: Code Audit

Before tuning hyperparameters, audit the PnC code for correctness and potential improvements. Run a single small experiment to verify the pipeline works end-to-end:

```bash
# Smoke test: single block, tiny config, should complete in <5 min
python -m luigi --module cifar_tasks CIFARPnC \
  --dataset cifar10 --epochs 300 --n-directions 5 --n-perturbations 10 \
  --perturbation-sizes '[0.1,1.0,10.0]' \
  --target-stage-idx 3 --target-block-idx 1 \
  --lambda-reg 1e-3 --subset-size 512 --chunk-size 64 \
  --seed 0 --posthoc-calibrate --local-scheduler
```

**Check in the output:**
1. `diag_calib_reduction_pct` > 0 (correction is doing something)
2. Accuracy is not 10% (random chance for CIFAR-10 -- would mean forward pass is broken)
3. No NaN in metrics
4. Memory usage stays under 8GB

---

## Phase 2: Per-Block Scale Discovery (Single-Block Sweeps)

**Key insight**: Different blocks have very different parameter counts and activation geometries. Stage 4 Block 1 has 2.4M conv1 params; Stage 1 Block 0 has 37K. A perturbation_scale of 1.0 means very different things for each. We run single-block PnC on each of the 8 blocks to find the appropriate scale per block.

### 2a. Stage 4 blocks (largest, most direct effect on output)

```bash
# Stage 4 Block 1 (last block -- closest to output)
python -m luigi --module cifar_tasks CIFARPnC \
  --dataset cifar10 --epochs 300 --n-directions 10 --n-perturbations 50 \
  --perturbation-sizes '[0.001,0.005,0.01,0.05,0.1,0.5,1.0,5.0]' \
  --target-stage-idx 3 --target-block-idx 1 \
  --lambda-reg 1e-3 --subset-size 1024 --chunk-size 64 \
  --seed 0 --posthoc-calibrate --local-scheduler

# Stage 4 Block 0
python -m luigi --module cifar_tasks CIFARPnC \
  --dataset cifar10 --epochs 300 --n-directions 10 --n-perturbations 50 \
  --perturbation-sizes '[0.001,0.005,0.01,0.05,0.1,0.5,1.0,5.0]' \
  --target-stage-idx 3 --target-block-idx 0 \
  --lambda-reg 1e-3 --subset-size 1024 --chunk-size 64 \
  --seed 0 --posthoc-calibrate --local-scheduler
```

### 2b. Stage 3 blocks

```bash
for BLOCK in 0 1; do
  python -m luigi --module cifar_tasks CIFARPnC \
    --dataset cifar10 --epochs 300 --n-directions 15 --n-perturbations 50 \
    --perturbation-sizes '[0.01,0.05,0.1,0.5,1.0,5.0,10.0]' \
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
    --perturbation-sizes '[0.05,0.1,0.5,1.0,5.0,10.0,50.0]' \
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
    --perturbation-sizes '[0.1,0.5,1.0,5.0,10.0,50.0,100.0]' \
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

**What to look for:**
- Blocks where `diag_calib_reduction_pct` > 80% = correction working well
- Blocks where accuracy barely changes = safe to perturb
- Blocks where NLL improves most = most useful for uncertainty
- If all blocks show similar NLL improvement, multi-block will compound the benefit

---

## Phase 3: Single-Block Refinement

Pick the single best block from Phase 2 and refine lambda_reg:

```bash
for LAMBDA in 1e-4 1e-3 1e-2 1e-1; do
  python -m luigi --module cifar_tasks CIFARPnC \
    --dataset cifar10 --epochs 300 --n-directions BEST_K --n-perturbations 50 \
    --perturbation-sizes '[NARROW_RANGE_AROUND_BEST]' \
    --target-stage-idx $BEST_STAGE --target-block-idx $BEST_BLOCK \
    --lambda-reg $LAMBDA --subset-size 1024 --chunk-size 64 \
    --seed 0 --posthoc-calibrate --local-scheduler
done
```

Also try `subset_size=2048`:

```bash
python -m luigi --module cifar_tasks CIFARPnC \
  --dataset cifar10 --epochs 300 --n-directions BEST_K --n-perturbations 50 \
  --perturbation-sizes '[BEST_SCALES]' \
  --target-stage-idx $BEST_STAGE --target-block-idx $BEST_BLOCK \
  --lambda-reg BEST_LAMBDA --subset-size 2048 --chunk-size 64 \
  --seed 0 --posthoc-calibrate --local-scheduler
```

---

## Phase 4: Multi-Block PnC

### 4a. First attempt with uniform scale

Use the median of per-block best scales from Phase 2, divided by sqrt(8) to account for compounding:

```bash
python -m luigi --module cifar_tasks CIFARMultiBlockPnC \
  --dataset cifar10 --epochs 300 --n-directions 10 --n-perturbations 50 \
  --perturbation-sizes '[MEDIAN_SCALE/4, MEDIAN_SCALE/2, MEDIAN_SCALE, MEDIAN_SCALE*2]' \
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

**Change**: After perturbing Conv1, run a forward pass through the calibration subset to compute new BN2 running stats (mean, var), then use those in `get_Y_fn` and in `_forward_member`. This matches what SWAG does (BN refresh) and is standard practice in weight-perturbation ensembles.

**Fairness**: This is the same BN refresh that SWAG already uses. It improves the correction without changing the core PJSVD algorithm.

**Implementation**: In `PnCEnsemble._precompute_corrections()`, after computing `w1_pert`, run the calibration chunks through BN1 -> ReLU -> Conv1(w1_pert) to get perturbed activations, compute channel-wise mean/var, and store them. Then modify `get_Y_fn` and `_forward_member` to use per-member BN2 stats instead of the original running average.

**Expected effect**: Better correction quality (higher diag_reduction_pct), especially for larger perturbation scales. May allow using larger scales without accuracy loss.

**Test**: Compare diag_reduction_pct and NLL with vs without BN2 refit at the same perturbation_scale.

### Hypothesis 2: Per-Block Perturbation Scale in Multi-Block

**Problem**: `MultiBlockPnCEnsemble` uses a single `perturbation_scale` for all 8 blocks. But Phase 2 will show that different blocks have different optimal scales (possibly 100x difference between Stage 1 and Stage 4). A uniform scale forces a compromise where some blocks are under-perturbed (no diversity) and others are over-perturbed (accuracy loss).

**Change**: Accept a list of perturbation scales (one per block) or a dict mapping block index to scale. In `_coeffs_from_z()`, use the block-specific scale instead of the global one.

**Fairness**: This is purely a hyperparameter-space expansion. The algorithm is unchanged; we're just tuning more carefully.

**Implementation**: Add `perturbation_scales: list[float]` parameter to `MultiBlockPnCEnsemble.__init__()`. In `_coeffs_from_z()`, accept a `block_index` argument and use `self.perturbation_scales[block_index]`. Update `CIFARMultiBlockPnC` to accept a list.

**Expected effect**: Significant NLL improvement by allowing each block to contribute optimally.

**Test**: Compare multi-block with uniform scale (best from Phase 4a) vs per-block scales (best per block from Phase 2).

### Hypothesis 3: Regularization-Weighted Ridge (Lambda Per Block)

**Problem**: A single `lambda_reg` is used for all blocks, but the condition number of H = M^T M varies dramatically: Stage 1 H is 577x577 and well-conditioned (many spatial patches), Stage 4 H is 4609x4609 and may be ill-conditioned (few spatial patches per image at 4x4 resolution).

**Change**: Allow per-block `lambda_reg`, or auto-scale lambda proportional to trace(H)/dim(H) (the mean eigenvalue), making regularization adapt to each block's conditioning.

**Fairness**: Standard ridge regression practice. Does not change the algorithm.

**Implementation**: In `solve_chunked_conv2_correction()`, after accumulating H, optionally compute `lambda_auto = trace(H) / D_M * lambda_relative` where `lambda_relative` is a user-specified fraction. Or simply accept a list of lambdas in `CIFARMultiBlockPnC`.

**Expected effect**: Better correction for poorly-conditioned blocks (stage 4), potentially allowing larger subset_size or perturbation_scale without overfitting.

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
| Accuracy drops >1% from baseline | perturbation_scale too large | Halve the scale |
| NLL worse than single model at all scales | Correction not working | Check diag_reduction_pct; if low, reduce lambda_reg or increase subset_size |
| diag_calib_reduction_pct < 50% | Ridge underfitting | Reduce lambda_reg by 10x; ensure chunk_size isn't too small |
| diag_test_reduction << diag_calib_reduction | Correction overfitting | Increase lambda_reg or subset_size |
| NLL good but ECE bad | Temperature scaling edge case | Check posthoc_temperature; if extreme (>5 or <0.1), try Hypothesis 4 (logit averaging) |
| Lanczos ~ Random for all blocks | Subspace selection not important here | Fine -- correction quality matters more. Focus on Hypotheses 1-3 |
| Multi-block much worse than single-block | Compounding perturbation effects | Implement per-block scale (Hypothesis 2); reduce scale by sqrt(n_blocks) |
| Multi-block Stage 4 ridge fails | H matrix ill-conditioned | Increase lambda_reg for Stage 4 specifically (Hypothesis 3) |
| Baseline accuracy below 93% | Training recipe issue | Check 300 epochs is completing; verify cosine schedule and warmup |
| SWAG NLL much worse than expected | BN refresh not working | Verify swag_use_bn_refresh=True and bn_refresh_subset_size >= 2048 |

---

## Success Criteria

The final paper table should show:

1. **Accuracy**: Multi-block PnC within 0.5% of MC Dropout, SWAG, and LLLA
2. **NLL**: Multi-block PnC <= min(MC Dropout NLL, SWAG NLL, LLLA NLL)
3. **ECE**: Multi-block PnC <= min(MC Dropout ECE, SWAG ECE, LLLA ECE)
4. **Deep Ensemble**: Reported as a reference point but not a target to beat
5. **Ablations reported**: Single-block vs multi-block, Lanczos vs random, per-block scale effect
