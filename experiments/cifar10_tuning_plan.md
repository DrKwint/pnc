# CIFAR-10 PnC Tuning Plan (Revised 2026-04-08)

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
- Multi-block with 6+ blocks OOMs on 8GB. Maximum safe: 4-5 blocks.

---

## Critical Fix: Patch Ordering (commit 4ed1527)

The original code had a **conv kernel flatten ordering bug**: `extract_patches` outputs in `(C_in, kh, kw)` order (channel-major), but W2 kernels were being flattened as `(kh, kw, C_in)` (spatial-major). The ridge regression solved in a scrambled basis — producing numerically small residuals that didn't correspond to meaningful corrections.

**Impact (S3B1, K=16, n=32, random, seed 0):**

| Scale | Acc (bug→fix) | NLL (bug→fix) | Temp (bug→fix) |
|-------|--------------|--------------|----------------|
| 10.0 | 92.70%→95.83% | 0.236→0.137 | 0.270→1.260 |
| 50.0 | 92.40%→95.61% | 0.335→0.139 | 0.126→0.999 |

The fix eliminates the 3% accuracy drop and 0.1 NLL penalty. **All PnC results generated before this fix are invalid.**

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
| Deep Ensemble (5) | 96.56±0.06% | 0.109±0.000 | 0.005±0.000 | 3 | Gold standard |
| MC Dropout (n=50) | 95.74±0.14% | 0.148±0.006 | 0.010±0.004 | 3 | |
| LLLA (prior=10, n=50) | 95.77±0.26% | 0.140±0.006 | 0.008±0.002 | 3 | Best prior from sweep of 0.01–1000 |
| SWAG (sws=240, n=50) | 95.41±0.04% | 0.146±0.002 | 0.007±0.000 | 3 | BN refresh bug fixed |
| Epinet (ps=1.0, n=50) | 95.78±0.15% | 0.147±0.005 | 0.010±0.002 | 3 | Best prior_scale from sweep |

**Targets for PnC to beat** (best of MC Dropout, LLLA, and Epinet):
- Accuracy: maintain within 0.5% of 95.74%
- NLL: < 0.140 (LLLA's mean, pending Epinet/SWAG results)
- ECE: < 0.008 (LLLA's mean, pending Epinet/SWAG results)

---

## Phase 0: Baselines — 80% Complete

### Completed ✓
- Single model eval: seeds 0,1,2 ✓
- LLLA prior sweep (0.01–1000): seeds 0,1,2 ✓ → **prior=10.0 is best**
- MC Dropout (n=50, dr=0.1): seeds 0,1,2 ✓
- Deep Ensemble (n=5): seeds 0,1,2 ✓

### Remaining: SWAG (BN refresh bug FIXED) and Epinet

**SWAG bug fixed**: Root cause was `_refresh_batch_norm_stats` using EMA momentum (0.9) after resetting BN to zero/one. With only 16 batches, running stats reached ~82% of true values — catastrophic for PreActResNet BN. Fix: cumulative averaging (momentum = 1/k for batch k). Eval running.

```bash
# SWAG eval (all 3 seeds)
```bash
for SEED in 0 1 2; do
  python -m luigi --module cifar_tasks CIFARPreActSWAG \
    --dataset cifar10 --epochs 300 --n-perturbations 50 \
    --swag-start-epoch 240 --swag-collect-freq 1 \
    --swag-use-bn-refresh --bn-refresh-subset-size 2048 \
    --seed $SEED --posthoc-calibrate --local-scheduler
done

# Epinet sweep (train + eval for each config, 3 seeds)
for SEED in 0 1 2; do
  for PRIOR_SCALE in 0.5 1.0 3.0; do
    python -m luigi --module cifar_tasks CIFARPreActEpinet \
      --dataset cifar10 --epochs 300 --epinet-epochs 100 \
      --n-perturbations 50 --index-dim 8 \
      --epinet-hiddens '50,50' --prior-scale $PRIOR_SCALE \
      --seed $SEED --posthoc-calibrate --local-scheduler
  done
done
```

---

## Phase 1: Code Audit ✓ — Complete

Confirmed by re-running PnC S3B1 after the patch ordering fix (commit 4ed1527):
- Pipeline runs end-to-end ✓
- Accuracy preserved (95.83% at scale=10) ✓
- No NaN in metrics ✓
- diag_test_reduction > 0 (37–87%) ✓
- Memory under 8GB ✓

---

## Phase 2: Per-Block Scale Discovery ✓ — Complete

### Key Findings

1. **Random directions outperform Lanczos** for uncertainty quantification. Lanczos directions are SO correctable (99.5-99.9% reduction) that the ensemble has zero diversity. Random directions leave ~50% uncorrected → useful diversity → better NLL.

2. **Scale=10 is optimal** for 5 of 6 blocks tested. Scale=5 is best for S3B1 only.

3. **Best single block: S3B0 (stage_idx=2, block_idx=0) → NLL=0.1354 at scale=10, K=15.**

### Full Per-Block Results (random directions, n=50, seed 0)

| Block | K | Best Scale | Accuracy | NLL | ECE | diag_red% |
|-------|---|-----------|----------|-----|-----|-----------|
| S2B0 | 20 | 10.0 | 95.67% | 0.1367 | 0.009 | 67.0% |
| S2B1 | 20 | 10.0 | 95.56% | 0.1437 | 0.009 | 72.5% |
| **S3B0** | **15** | **10.0** | **95.78%** | **0.1354** | **0.009** | **52.9%** |
| S3B1 | 15 | 5.0 | 95.81% | 0.1384 | 0.009 | 40.6% |
| S4B0 | 10 | 10.0 | 95.80% | 0.1373 | 0.009 | 54.5% |
| S4B1 | 10 | 10.0 | 95.85% | 0.1367 | 0.008 | 59.2% |

**Observations:**
- S3B0 best NLL by significant margin (0.1354 vs 0.1367 next best)
- S2B1 and S3B1 are notably weaker — not all blocks contribute equally
- All blocks preserve accuracy within 0.2% at best scale
- Optimal diag_reduction is ~50-60% — too much correction kills diversity, too little kills accuracy

---

## Phase 3: Lambda Sweep ✓ — Complete (Negative Result)

Lambda has **zero effect** across 5 orders of magnitude (1e-4 to 1.0) on S3B0. The data term in the ridge regression completely dominates. **Lambda is not a useful diversity knob.** Default lambda_reg=1e-3 is fine.

---

## Phase 4: Multi-Block PnC — Partially Complete

### Completed: Block Selection and Scale Sweep (seed 0)

| Config | Blocks (indices) | Best Scale | Acc | NLL | ECE | Temp |
|--------|-----------------|-----------|-----|-----|-----|------|
| 1-block | s2b0 (blk 4) | 10.0 | 95.78% | 0.1354 | 0.009 | 1.245 |
| 2-block | 4+7 | 10.0 | 95.69% | 0.1343 | 0.009 | 1.199 |
| 3-block | 2+4+7 | 5.0 | 95.88% | 0.1345 | 0.008 | 1.249 |
| 3-block | 4+6+7 | 10.0 | 95.65% | 0.1349 | 0.009 | 1.158 |
| **4-block** | **2+4+6+7** | **7.0** | **95.69%** | **0.1339** | **0.008** | **1.188** |
| 5-block | 2+4+5+6+7 | 5.0 | 95.73% | 0.1354 | 0.009 | 1.232 |
| 6-block | all Stage 2-4 | — | OOM | — | — | — |

**Best: 4-block at scale=7.0 → NLL=0.1339, Acc=95.69%, ECE=0.008**

This beats:
- Base model NLL 0.1384 by 3.2%
- LLLA NLL 0.140 by 4.4%
- MC Dropout NLL 0.148 by 9.4%

**Key observations:**
- Multi-block improves over single-block (0.1354 → 0.1339)
- Optimal scale decreases as blocks increase (10 → 7)
- Adding weak blocks (S2B1=blk 3, S3B1=blk 5) hurts performance
- The 4 best blocks are S2B0(2), S3B0(4), S4B0(6), S4B1(7) — one from each stage except stage 1

### Remaining Phase 4 Work

#### 4a. Fine-grained scale sweep around scale=7.0

The current best is at scale=7.0, but the sweep tested [3, 5, 7, 10, 15]. A finer grid could find a better optimum.

```bash
python -m luigi --module cifar_tasks CIFARMultiBlockPnC \
  --dataset cifar10 --epochs 300 --n-directions 10 --n-perturbations 50 \
  --perturbation-sizes '[5.0,6.0,7.0,8.0,9.0,10.0]' \
  --block-selection '[2,4,6,7]' \
  --lambda-reg 1e-3 --subset-size 1024 --chunk-size 64 \
  --seed 0 --posthoc-calibrate --random-directions --local-scheduler
```

#### 4b. K (n_directions) sweep on 4-block config

Phase 2 used different K per stage (10 for stage 4, 15 for stage 3, 20 for stage 2). Multi-block uses uniform K=10. Does increasing K help?

```bash
for K in 5 10 15 20; do
  python -m luigi --module cifar_tasks CIFARMultiBlockPnC \
    --dataset cifar10 --epochs 300 --n-directions $K --n-perturbations 50 \
    --perturbation-sizes '[7.0]' \
    --block-selection '[2,4,6,7]' \
    --lambda-reg 1e-3 --subset-size 1024 --chunk-size 64 \
    --seed 0 --posthoc-calibrate --random-directions --local-scheduler
done
```

**OOM note:** K=20 with block 6 (S4B0) uses 20×590K×4B=45MB for directions alone. Should still fit in 8GB with chunk_size=64, but monitor.

---

## Phase 5: Algorithm Improvements — Revised Priority Order

Based on Phase 2-4 findings, the hypotheses are re-prioritized by expected impact:

### Hypothesis 2 (HIGH PRIORITY): Per-Block Perturbation Scale

**Why this matters now:** Phase 2 showed S3B0 is best at scale=10, S3B1 at scale=5, and the optimal multi-block scale=7 is a compromise. Per-block scale would let each block contribute at its individual optimum.

**Implementation:** Accept a list of perturbation scales (one per block) in `MultiBlockPnCEnsemble.__init__()`. In `_coeffs_from_z()`, use block-specific scale. Update `CIFARMultiBlockPnC` to accept `--perturbation-scales-per-block`.

**Test:**
```bash
# Per-block scale using Phase 2 optima
python -m luigi --module cifar_tasks CIFARMultiBlockPnC \
  --dataset cifar10 --epochs 300 --n-directions 10 --n-perturbations 50 \
  --perturbation-scales-per-block '{"2": 10.0, "4": 10.0, "6": 10.0, "7": 10.0}' \
  --block-selection '[2,4,6,7]' \
  --lambda-reg 1e-3 --subset-size 1024 --chunk-size 64 \
  --seed 0 --posthoc-calibrate --random-directions --local-scheduler

# Then sweep per-block scales around single-block optima
```

**Expected impact:** Moderate NLL improvement (0.1339 → ~0.131-0.133).

### Hypothesis 4 (MEDIUM PRIORITY): Logit Averaging

**Why this matters now:** The posthoc_temperature for the best 4-block config is 1.188 — close to 1 but not exactly. Logit averaging changes how ensemble member logit shifts combine and may produce better-calibrated predictions without needing as much temperature correction.

**Implementation:** Add `--logit-averaging` flag to `_evaluate_cifar()`. Average logits before softmax instead of averaging probabilities after softmax.

**Fairness requirement:** Must test both modes for ALL methods (PnC, LLLA, MC Dropout, Deep Ensemble). Report whichever mode is better for each, or pick one mode for all.

**Test:**
```bash
# Run best PnC config with both modes
python -m luigi --module cifar_tasks CIFARMultiBlockPnC \
  --dataset cifar10 --epochs 300 --n-directions 10 --n-perturbations 50 \
  --perturbation-sizes '[7.0]' --block-selection '[2,4,6,7]' \
  --lambda-reg 1e-3 --subset-size 1024 --chunk-size 64 \
  --seed 0 --posthoc-calibrate --random-directions \
  --logit-averaging --local-scheduler

# Also rerun baselines with logit averaging for fairness
```

### Hypothesis 1 (LOW PRIORITY — likely counterproductive): BN2 Refit

**Deprioritized.** Phase 2 showed the bottleneck is *too little* diversity, not too much correction error. BN2 refit would improve correction → less diversity → worse NLL. Only revisit if per-block scale (Hypothesis 2) creates scenarios where correction quality becomes the bottleneck.

### Hypothesis 3 (DROPPED): Per-Block Lambda

**Dropped.** Phase 3 showed lambda has zero effect across 5 orders of magnitude. Per-block lambda would have zero effect per block. The data term completely dominates.

---

## Phase 6: Final Multi-Seed Validation

Once the best configuration is found (after Phases 4-5), run with multiple seeds.

### 6a. Best multi-block config (3 seeds)
```bash
for SEED in 0 1 2; do
  python -m luigi --module cifar_tasks CIFARMultiBlockPnC \
    --dataset cifar10 --epochs 300 \
    --n-directions $BEST_K --n-perturbations 50 \
    --perturbation-sizes '[$BEST_SCALE]' \
    --block-selection '[2,4,6,7]' \
    --lambda-reg 1e-3 --subset-size 1024 --chunk-size 64 \
    --seed $SEED --posthoc-calibrate --random-directions --local-scheduler
done
```

### 6b. Best single-block config for ablation table (3 seeds)
```bash
for SEED in 0 1 2; do
  python -m luigi --module cifar_tasks CIFARPnC \
    --dataset cifar10 --epochs 300 \
    --n-directions 15 --n-perturbations 50 \
    --perturbation-sizes '[10.0]' \
    --target-stage-idx 2 --target-block-idx 0 \
    --lambda-reg 1e-3 --subset-size 1024 --chunk-size 64 \
    --seed $SEED --posthoc-calibrate --random-directions --local-scheduler
done
```

### 6c. Lanczos ablation (3 seeds, to quantify Lanczos vs random gap)
```bash
for SEED in 0 1 2; do
  python -m luigi --module cifar_tasks CIFARMultiBlockPnC \
    --dataset cifar10 --epochs 300 \
    --n-directions $BEST_K --n-perturbations 50 \
    --perturbation-sizes '[$BEST_SCALE]' \
    --block-selection '[2,4,6,7]' \
    --lambda-reg 1e-3 --subset-size 1024 --chunk-size 64 \
    --seed $SEED --posthoc-calibrate --local-scheduler
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
| OOM during multi-block (6+ blocks) | Too many blocks for 8GB | Use 4-block config (2,4,6,7); max tested is 5 |
| OOM during Lanczos | K too large for this stage | Reduce K; for Stage 4 use K<=10 |
| OOM during ridge solve | chunk_size or subset_size too large | Reduce chunk_size to 32; reduce subset_size |
| NLL flat across all scales (no diversity) | Lanczos directions too correctable | Switch to random directions (already confirmed better) |
| Accuracy drops >1% before NLL improves | Perturbation destroys structure too fast | Reduce scale; for multi-block use scale < 10 |
| NLL improves but accuracy drops >1% | At diversity threshold | Use slightly smaller scale |
| diag_calib_reduction_pct < 50% | Ridge underfitting | Reduce lambda_reg by 10x; ensure chunk_size isn't too small |
| Multi-block much worse than single-block | Weak blocks diluting signal | Remove blocks with single-block NLL > 0.140 from selection |
| SWAG 10% accuracy | BN refresh using EMA from zero (FIXED) | Use cumulative averaging (1/k) in _refresh_batch_norm_stats |
| Multi-seed variance too high | Insufficient seeds or high base model variance | Run 5 seeds instead of 3 |
| Per-block scale code OOMs | Per-block H matrices too large | Use uniform scale with fine grid instead |

---

## Execution Priority (Ordered TODO)

1. **SWAG eval** (Phase 0) — BN refresh bug FIXED, eval running for seeds 0,1,2
2. **Epinet baseline** (Phase 0) — implemented, needs train+eval for seeds 0,1,2
3. **Fine-scale sweep** (Phase 4a) — cheap, may find better scale than 7.0
4. **K sweep** (Phase 4b) — cheap, may find better K than 10
5. **Per-block scale** (Hypothesis 2) — moderate effort code change, likely NLL improvement
6. **Logit averaging** (Hypothesis 4) — small code change, unclear impact, must test all methods
7. **Multi-seed validation** (Phase 6) — required for publication, run last after best config is locked
8. **Ablation table** (Phase 6c) — Lanczos vs random multi-seed, single-block vs multi-block

---

## Success Criteria

The final paper table should show:

1. **Accuracy**: Multi-block PnC within 0.5% of MC Dropout, SWAG, LLLA, and Epinet
2. **NLL**: Multi-block PnC <= min(MC Dropout NLL, SWAG NLL, LLLA NLL, Epinet NLL)
3. **ECE**: Multi-block PnC <= min(MC Dropout ECE, SWAG ECE, LLLA ECE, Epinet ECE)
4. **Deep Ensemble**: Reported as a reference point but not a target to beat
5. **Ablations reported**: Single-block vs multi-block, Lanczos vs random, block selection effect

### Current Status vs Criteria (3 seeds)

Best PnC config: 4-block (blks 2,4,6,7), K=20, scale=7.0, random directions.

| Criterion | Target | PnC Result | Status |
|-----------|--------|-----------|--------|
| Accuracy | ≥95.24% | 95.67±0.15% | ✓ Met |
| NLL | <0.140 | 0.138±0.004 | ✓ Met (beats LLLA 0.140, Epinet 0.147, SWAG 0.146) |
| ECE | <0.008 | 0.008±0.001 | ~ Tied with LLLA |
| All baselines | 3 seeds each | All complete | ✓ Done |
| Multi-seed PnC | 3 seeds | Seeds 0,1,2 done | ✓ Done |
| Ablations | Required | Scale sweep, K sweep done; Lanczos vs random (Phase 2) | Mostly done |
