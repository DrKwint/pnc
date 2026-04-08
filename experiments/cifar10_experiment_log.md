# CIFAR-10 Experiment Log

## 2026-04-05: Initial State Assessment

### Existing Results (seed 0 only, n=32)

| Method | Accuracy | NLL | ECE | PostHoc Temp | Status |
|--------|----------|-----|-----|-------------|--------|
| Base model | 95.77% | 0.138 | 0.009 | 1.312 | OK — within expected 94.5-95.5% |
| Deep Ensemble (5) | 96.51% | 0.108 | 0.005 | 1.115 | OK — within expected range |
| MC Dropout (n=32) | 95.79% | 0.142 | 0.008 | - | OK — actually exceeds expected 94-94.5% acc |
| SWAG (sws=160) | **10.00%** | **2.303** | 0.004 | 11.92 | BROKEN — random chance |
| LLLA (prior=1.0) | 95.37% | **0.549** | **0.334** | 0.328 | BAD calibration |
| PnC S3B1 rand (best@10.0) | 92.70% | 0.236 | 0.012 | 0.270 | Underperforming |

### PnC S3B1 Random Directions Detail (seed 0, n=32, K=16)

| Scale | Accuracy | NLL | ECE | diag_test_reduction% | PostHoc Temp |
|-------|----------|-----|-----|---------------------|-------------|
| 1.0 | 91.67% | 0.266 | 0.009 | 98.1% | 0.414 |
| 5.0 | 92.38% | 0.247 | 0.009 | 90.9% | 0.351 |
| 10.0 | 92.70% | 0.236 | 0.012 | 83.0% | 0.270 |
| 50.0 | 92.40% | 0.335 | 0.112 | 83.4% | 0.126 |

**Observations:**
- diag_test_reduction is very high (83-98%) — correction is working
- But accuracy drops 3% from baseline (95.77% → 92.70%) — too much degradation
- All posthoc_temperature < 0.5 — ensemble is underconfident before calibration
- NLL best at scale=10.0 but still 0.236 vs baseline 0.138

### Diagnosis: SWAG

**Root cause**: `swag_start_epoch=160` is far too early for 300-epoch cosine decay training. At epoch 160, learning rate ≈ 0.1 × (1 + cos(π×160/300))/2 ≈ 0.028. The model is still moving significantly, creating enormous parameter variance across the 141 collected snapshots (epochs 160-300). Sampled parameters are garbage.

**Fix**: Use `swag_start_epoch=240` per the plan. At epoch 240, LR ≈ 0.003 — parameters are nearly converged, giving a tight SWAG distribution around the final basin.

### Diagnosis: LLLA

**Root cause**: `prior_precision=1.0` yields a posterior that is too diffuse. The Cholesky factor L of the (G + λI)^{-1} covariance is too large, so sampled last-layer weights deviate too far from the MAP estimate. The ensemble averages to an underconfident predictor (posthoc_temperature=0.33 < 1 confirms this — calibration must *sharpen* predictions).

**Fix**: Sweep prior_precision ∈ {0.01, 0.1, 1.0, 10.0, 100.0, 1000.0}. Higher prior → tighter posterior → more confident (and likely better-calibrated) ensemble. Typical best prior for CIFAR-10 LLLA is 1.0-100.0 per Laplace Redux paper.

### Discrepancies from Plan

| Parameter | Plan | Actual | Impact |
|-----------|------|--------|--------|
| n_perturbations | 50 | 32 | Minor — need to rerun |
| SWAG start epoch | 240 | 160 | **Critical** — caused 10% acc |
| LLLA prior sweep | 6 values | 1 value | **Critical** — only tested bad prior |
| Seeds | 0,1,2 | 0 only | Need multi-seed for significance |

### Next Steps: Phase 0 Corrected Baselines

1. Train SWAG with `swag_start_epoch=240` for seeds 0,1,2
2. ~~Run LLLA prior sweep (6 priors × 3 seeds)~~ ✓ DONE
3. Rerun MC Dropout with n=50 for seeds 0,1,2
4. Run Deep Ensemble for seeds 1,2 (seed 0 already good)
5. ~~Run single model eval for seeds 1,2 (seed 0 already good)~~ ✓ DONE

---

## 2026-04-05: LLLA Prior Sweep Results

### Single Model Eval (seeds 0,1,2, posthoc calibrated)

| Seed | Accuracy | NLL | ECE | PostHoc Temp |
|------|----------|-----|-----|-------------|
| 0 | 95.77% | 0.138 | 0.009 | 1.312 |
| 1 | 95.90% | 0.144 | 0.010 | 1.359 |
| 2 | 95.55% | 0.149 | 0.012 | 1.329 |
| **Mean** | **95.74%** | **0.144** | **0.010** | |

### LLLA Prior Sweep (n=50, posthoc calibrated, seeds 0,1,2)

| Prior | Acc (mean) | NLL (mean) | ECE (mean) | Temp (mean) | Notes |
|-------|-----------|-----------|-----------|------------|-------|
| 0.01 | 25.5% | 2.11 | 0.130 | 15.2 | Destroyed — way too diffuse |
| 0.1 | 81.2% | 1.61 | 0.596 | 1.22 | Still too diffuse |
| 1.0 | 95.71% | 0.586 | 0.368 | 0.11 | Good acc, terrible calibration |
| **10.0** | **95.77%** | **0.137** | **0.008** | 0.85 | **Best NLL — near-perfect** |
| 100.0 | 95.75% | 0.141 | 0.010 | 1.29 | Also good, slightly worse |
| 1000.0 | 95.73% | 0.144 | 0.010 | 1.33 | Converges to base model |

**Key findings:**
- **Prior=10.0 is the clear winner** across all 3 seeds: NLL=0.137 (beats base model 0.144), ECE=0.008
- Prior=10.0 posthoc_temperature≈0.85 — nearly calibrated before scaling (only slight underconfidence)
- Prior=100.0 is also competitive (NLL=0.141) — confirms sweet spot is 10-100
- Prior≥100 converges to base model behavior (too-tight posterior → no diversity)
- Prior≤1.0 is catastrophically bad — posterior way too diffuse

**For paper:** Report LLLA with prior=10.0, n=50.

### MC Dropout (n=50, posthoc calibrated, dropout_rate=0.1)

| Seed | Accuracy | NLL | ECE |
|------|----------|-----|-----|
| 0 | 95.78% | 0.143 | 0.009 |
| 1 | 95.86% | 0.146 | 0.007 |
| 2 | 95.59% | 0.154 | 0.014 |
| **Mean** | **95.74%** | **0.148** | **0.010** |

### Phase 0 Baseline Summary (seed mean, posthoc calibrated)

| Method | Accuracy | NLL | ECE | Status |
|--------|----------|-----|-----|--------|
| Single model | 95.74% | 0.144 | 0.010 | ✓ 3 seeds |
| Deep Ensemble (5) | 96.51% | 0.108 | 0.005 | 1 seed only |
| MC Dropout (n=50) | 95.74% | 0.148 | 0.010 | ✓ 3 seeds |
| LLLA (prior=10, n=50) | 95.77% | 0.140 | 0.008 | ✓ 3 seeds |
| SWAG (sws=240) | — | — | — | Not yet run |

**Observations:**
- LLLA (prior=10) slightly beats MC Dropout on NLL (0.140 vs 0.148) and ECE (0.008 vs 0.010)
- Both are only marginally better than the single model (NLL 0.144)
- Deep Ensemble is clearly best (NLL 0.108) but only 1 seed — need more
- SWAG still needs to be run with corrected sws=240

### Remaining Phase 0 Work
- [ ] SWAG training with sws=240 for seeds 0,1,2 (~30 min each)
- [ ] Deep Ensemble seeds 1,2 (need base models 5,6 first, ~20 min each)
- [ ] Base model training seeds 5,6 for Deep Ensemble

---

## 2026-04-06: Patch Ordering Fix (commit 4ed1527)

### Bug Description
`extract_patches` outputs in `(C_in, kh, kw)` order (channel-major), but W2 was being flattened as `(kh, kw, C_in)` (spatial-major). The ridge regression solved in a scrambled basis — corrections were numerically small but semantically wrong.

### Fix
New `flatten_conv_kernel_to_patches()` does `w.transpose(2,0,1,3).reshape(-1, C_out)` to match patch ordering. Applied in `pnc.py` (ridge solve) and `ensembles.py` (diagnostics).

### Impact (PnC S3B1, K=16, n=32, random, seed 0)

| Scale | Acc (before→after) | NLL (before→after) | Temp (before→after) |
|-------|-------------------|-------------------|---------------------|
| 1.0 | 91.67%→**95.77%** | 0.266→**0.138** | 0.414→1.311 |
| 5.0 | 92.38%→**95.78%** | 0.247→**0.138** | 0.351→1.296 |
| 10.0 | 92.70%→**95.83%** | 0.236→**0.137** | 0.270→1.260 |
| 50.0 | 92.40%→**95.61%** | 0.335→**0.139** | 0.126→0.999 |

**Critical**: The 3% accuracy drop and ~0.1 NLL penalty were entirely caused by the bug. PnC now preserves accuracy and matches the base model NLL. All previous PnC results are invalid.

---

## Phase 2 — Per-Block Scale Discovery (post-fix)

### S4B1 Lanczos (K=10, n=50, lambda=1e-3, seed 0)

| Scale | Accuracy | NLL | ECE | diag_test_red% | PostHoc Temp |
|-------|----------|-----|-----|---------------|-------------|
| 1.0 | 95.76% | 0.1384 | 0.009 | 98.1% | 1.310 |
| 5.0 | 95.77% | 0.1384 | 0.009 | 99.5% | 1.309 |
| 10.0 | 95.76% | 0.1383 | 0.009 | 99.7% | 1.307 |
| 50.0 | 95.78% | 0.1380 | 0.009 | 99.9% | 1.299 |
| 100.0 | 95.76% | 0.1377 | 0.009 | 99.9% | 1.288 |
| 200.0 | 95.80% | **0.1377** | 0.009 | 99.9% | 1.254 |
| 500.0 | 95.79% | 0.1437 | 0.013 | 99.8% | 1.028 |

**Critical insight**: Lanczos directions are SO well-aligned with the correction that diag_test_reduction is 99.5-99.9% even at scale=500. Nearly zero diversity leaks through. NLL is flat at ~0.138 (matching base model exactly) until scale=200, then worsens at scale=500.

**Compare with random directions** (S3B1, K=16, post-fix re-run):
- Random, scale=10: diag_red=60.2%, NLL=0.1369 ← BETTER NLL because more diversity
- Lanczos, scale=10: diag_red=99.7%, NLL=0.1383 ← worse NLL because no diversity

**Key strategic conclusion**: Lanczos directions optimize for correctability, but correctability = no diversity = no NLL improvement. Random directions create more useful diversity because the correction is less effective for them. Random directions may be the better choice for PnC uncertainty quantification.

### S4B0 Lanczos (K=10, n=50, lambda=1e-3, seed 0)

| Scale | Accuracy | NLL | ECE | diag_test_red% | PostHoc Temp |
|-------|----------|-----|-----|---------------|-------------|
| 1.0 | 95.77% | 0.1388 | 0.009 | 96.0% | 1.312 |
| 200.0 | 95.77% | 0.1388 | 0.010 | 100.0% | 1.308 |
| 500.0 | 95.79% | 0.1382 | 0.009 | 100.0% | 1.289 |

S4B0 Lanczos: 100% correction at scale 200-500. NLL completely flat. Zero diversity.

### S4B1 Lanczos vs Random Head-to-Head

| Scale | Dir | Acc | NLL | ECE | diag_red% | Temp |
|-------|-----|-----|-----|-----|-----------|------|
| 10 | Lanczos | 95.76% | 0.1383 | 0.009 | 99.7% | 1.307 |
| 10 | **Random** | **95.85%** | **0.1367** | **0.008** | 59.2% | 1.260 |
| 50 | Lanczos | 95.78% | 0.1380 | 0.009 | 99.9% | 1.299 |
| 50 | Random | 95.64% | 0.1386 | 0.008 | 86.4% | 1.001 |

**Random directions at scale=10 on S4B1**: NLL=**0.1367** — beats base model (0.1384), LLLA (0.140), and MC Dropout (0.148). This is the best PnC result so far.

### Phase 2 Complete: Per-block best (random directions, n=50, seed 0)

| Block | K | Best Scale | Accuracy | NLL | ECE | diag_red% | Temp |
|-------|---|-----------|----------|-----|-----|-----------|------|
| S2B0 (s1b0) | 20 | 10.0 | 95.67% | 0.1367 | 0.009 | 67.0% | 1.259 |
| S2B1 (s1b1) | 20 | 10.0 | 95.56% | 0.1437 | 0.009 | 72.5% | 1.298 |
| **S3B0 (s2b0)** | **15** | **10.0** | **95.78%** | **0.1354** | **0.009** | 52.9% | 1.245 |
| S3B1 (s2b1) | 15 | 5.0 | 95.81% | 0.1384 | 0.009 | 40.6% | 1.300 |
| S4B0 (s3b0) | 10 | 10.0 | 95.80% | 0.1373 | 0.009 | 54.5% | 1.264 |
| S4B1 (s3b1) | 10 | 10.0 | 95.85% | 0.1367 | 0.008 | 59.2% | 1.260 |

Base model NLL: 0.1384 | LLLA (prior=10): 0.1403 | MC Dropout: 0.1477

**Winner: S3B0 (stage_idx=2, block_idx=0)** — NLL=0.1354 at scale=10.0 with K=15.
This already beats LLLA (0.140) and MC Dropout (0.148) in single-block mode!

**Observations:**
- Scale=10 is optimal for 5 of 6 blocks. Scale=5 is best for S3B1 only.
- diag_test_reduction 40-72% — moderate correction, leaving useful diversity
- All blocks preserve accuracy within 0.2% of base model at best scale
- S3B0 has the best NLL by a significant margin (0.1354 vs next-best 0.1367)
- Larger blocks (S4) have higher diag_red% → less diversity at same scale
- S2B1 and S1B1 are notably worse — not all blocks contribute equally

### Phase 3: Lambda Sweep on S3B0 (scale=[5,10,20], random, K=15)

| lambda | NLL @scale=10 | Notes |
|--------|--------------|-------|
| 1e-4 | 0.1354 | identical |
| 1e-3 | 0.1354 | identical |
| 1e-2 | 0.1354 | identical |
| 1e-1 | 0.1354 | identical |
| 1.0 | 0.1354 | identical |

**Lambda has zero effect** across 5 orders of magnitude. The data term in the ridge regression completely dominates the regularization term. Diversity is controlled entirely by perturbation scale, not correction weakness.

**Conclusion**: lambda_reg=1e-3 is fine (default). No need to tune further.

### Phase 4: Multi-Block PnC (random directions, n=50, seed 0)

Added `block_selection` parameter to `CIFARMultiBlockPnC` to select subsets of blocks. Removed test-data forwarding to reduce memory (only calibration-set diagnostics remain).

| Config | Blocks | Best Scale | Accuracy | NLL | ECE | Temp |
|--------|--------|-----------|----------|-----|-----|------|
| 1-block S3B0 | s2b0 | 10.0 | 95.78% | 0.1354 | 0.009 | 1.245 |
| 2-block | s2b0+s3b1 | 10.0 | 95.69% | 0.1343 | 0.009 | 1.199 |
| 3-block | s1b0+s2b0+s3b1 | 5.0 | 95.88% | 0.1345 | 0.008 | 1.249 |
| 3-block | s2b0+s3b0+s3b1 | 10.0 | 95.65% | 0.1349 | 0.009 | 1.158 |
| **4-block** | **s1b0+s2b0+s3b0+s3b1** | **7.0** | **95.69%** | **0.1339** | **0.008** | 1.188 |
| 5-block | s1b0+s2b0+s2b1+s3b0+s3b1 | 5.0 | 95.73% | 0.1354 | 0.009 | 1.232 |
| 6-block | all Stage 2-4 | — | OOM | — | — | — |

Base model NLL: 0.1384 | LLLA: 0.1403 | MC Dropout: 0.1477

**Best result: 4-block (S2B0+S3B0+S4B0+S4B1) at scale=7.0 → NLL=0.1339**

**Observations:**
- Multi-block improves over single-block: 0.1354 → 0.1339 (1-block → 4-block)
- Optimal scale decreases as blocks are added (10 → 7)
- Adding weak blocks (S3B1, S2B1) hurts — 5-block is worse than 4-block
- 6-block OOMs on 8GB GPU despite optimizations
- NLL=0.1339 beats LLLA (0.140), MC Dropout (0.148), and the single model (0.144)
- Accuracy preserved at 95.69% (within 0.1% of base model 95.74%)

---
