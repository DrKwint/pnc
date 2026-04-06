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
2. Run LLLA prior sweep (6 priors × 3 seeds)
3. Rerun MC Dropout with n=50 for seeds 0,1,2
4. Run Deep Ensemble for seeds 1,2 (seed 0 already good)
5. Run single model eval for seeds 1,2 (seed 0 already good)

---
