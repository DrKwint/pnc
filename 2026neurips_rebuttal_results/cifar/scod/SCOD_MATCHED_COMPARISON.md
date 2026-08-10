# SCOD ↔ P&C matched comparison — recovered & verified

Same 3 base checkpoints (seeds 0,1,2), same 1024 ID calibration pool, ID-only selection.

## SCOD (verified EXACT vs quoted)
Source: `../scod/seed{0,1,2}_metrics.json` variant `tempered_k10_Meps5000`; also `../scod/tables/scod_aggregate.json`.
| seed | Acc | Near AUROC | Near FPR95 | Far AUROC | Far FPR95 |
|---|---|---|---|---|---|
| 0 | 95.77 | 89.74 | 38.31 | 93.08 | 20.18 |
| 1 | 95.90 | 89.56 | 40.64 | 92.31 | 21.94 |
| 2 | 95.55 | 89.76 | 39.25 | 92.31 | 22.10 |
| **mean±std** | **95.74±0.18** | **89.69±0.11** | **39.40±1.17** | **92.56±0.45** | **21.41±1.06** |
Quoted SCOD (95.74/89.69/39.40/92.56/21.41) → **exact match**.

## P&C — the quoted 90.99/94.83 is s3b1, NOT the s3b0 anchor  ⚠ CONFLICT
The quoted P&C (95.69 / **90.99** / 37.39 / **94.83** / 19.52) reproduces exactly as the 3-seed predictive-entropy mean of the **s3b1** submitted JSONs (`results/cifar10/openood_v1p5_pnc_single_block_vcal_s3b1_..._bf0.05_..._chunksize1024_seed{0,1,2}_random.json`).
| P&C config (3-seed mean) | Acc | Near AUROC | Near FPR95 | Far AUROC | Far FPR95 |
|---|---|---|---|---|---|
| **s3b1** (Candidate B) | 95.69 | **90.99** | 37.39 | **94.83** | 19.52 | ← quoted P&C
| **s3b0** (Candidate A, submitted anchor) | 95.59 | 91.55 | 33.08 | 95.09 | 18.15 | ← manuscript + canonical SCOD pairing

- The **canonical** matched comparison in `SCOD_CIFAR_REPORT.md` §6/§9 and the manuscript pairs SCOD against **P&C s3b0** (91.55/95.09), corroborated by `revised_benchmark_raw.json` `summary_3seed.pe` (91.55/33.07/95.10/18.13, per-seed `cached_pe_check` matches the s3b0 JSONs to the digit).
- **Recommendation:** when quoting SCOD-vs-P&C, use the **s3b0** anchor (91.55/95.09) to stay consistent with the manuscript headline. The s3b1 numbers (90.99/94.83) are a valid alternative block but a different config — do not mix.

## SCOD provenance (verified)
- Variant SCOD-1024 (resource-matched to P&C's 1024 pool). Sketch rank **k=10** reported (k_max=20; saturates by k=10). num_samples **T=124**, **Meps=5000** (Meps-invariant over {500,5000,50000}; eigenvalues O(1–9) ≫ 1/(2Meps)). Score = tempered `posterior_pred` local Fisher-curvature (higher=more OOD).
- Calibration **1024** = `RandomState(seed).choice(45000,1024,replace=False)`, no bootstrap. Selection **ID-only** (rank/Meps ID-predeclared; temperature on ID-val: seed0=1.3120, seed1=1.3595, seed2=1.3291). No OOD in build/selection — CONFIRMED.
- Checkpoints (PreActResNet18, P=11,172,170) SHA-256: seed0=`b4c45628adaa…`, seed1=`2e942d691915…`, seed2=`9b5d9de691ba…` (== manifest; weights bit-unchanged after sketching). Sketch seed = 100000+checkpoint_seed.
- Runtime (`../scod/timing/profile.json`): build ≈342s/seed (sketch 167s), serialized sketch 894MB, peak GPU ≈4.6GB; online B=1 21.4ms/img vs base 2.0ms (10.7×). Git 854f4d847c, JAX 0.9.1/Flax 0.12.3 (JAX/Flax reimpl of PyTorch SCOD, validated by exact tiny-network agreement).
