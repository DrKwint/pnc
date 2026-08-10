# CIFAR Correction Geometry & Conditioning (Phase 2 / Section 2.1 + 2.5)

**Date:** 2026-07-24 · seed 0 · CALIB=1024 ID images · perturbation scale 25, one unit-norm direction ·
augmented conv2 design `X_v` (streaming Gram) · λ=1e-3. Raw: `block_geometry_raw.json`; harness: `_block_geometry.py`.

## 2.1 Per-block geometry table (all 8 residual blocks)
| block | shortcut | p=9·Cin+1 | patch rows | rows/p | num-rank | rank/p | **stable rank** | cond(G) | **cond(G+λ)** | λ_rel |
|---|---|---|---|---|---|---|---|---|---|---|
| s0b0 | identity | 577 | 1,048,576 | 1817 | 577 | 1.00 | **1.6** | 2.7e5 | 2.7e5 | 1.7e-9 |
| s0b1 | identity | 577 | 1,048,576 | 1817 | 577 | 1.00 | **1.5** | 4.9e5 | 4.9e5 | 1.1e-9 |
| s1b0 | projected | 1153 | 262,144 | 227 | 1153 | 1.00 | 2.6 | 5.3e4 | 5.3e4 | 6.7e-8 |
| s1b1 | identity | 1153 | 262,144 | 227 | 1153 | 1.00 | 2.7 | 2.1e4 | 2.1e4 | 5.3e-8 |
| s2b0 | projected | 2305 | 65,536 | 28.4 | 2301 | 1.00 | 3.7 | **3.7e18** | **7.3e8** | 8.6e-7 |
| s2b1 | identity | 2305 | 65,536 | 28.4 | 2305 | 1.00 | 4.0 | 1.2e4 | 1.2e4 | 6.8e-7 |
| **s3b0** (submitted) | **projected** | 4609 | 16,384 | 3.6 | 4609 | 1.00 | **4.8** | **7.3e10** | **8.1e7** | 1.2e-5 |
| s3b1 | identity | 4609 | 16,384 | 3.6 | 4609 | 1.00 | 4.1 | 1.0e5 | 1.0e5 | 7.1e-6 |

`stable rank = ‖X_v‖_F²/‖X_v‖₂²` (effective # significant directions). `cond(G)` is without ridge; `cond(G+λ)` with
the submitted λ=1e-3. `λ_rel = λ / (tr(G)/p)`.

## 2.5 Rank vs #images (spatial redundancy)
Numerical rank at 64/128/256/512/1024 calibration images:
- **s0, s1: FULL rank at 64 images** (577, 1153) — adding images does not raise rank. Rank is saturated by the many
  spatial patches within a handful of images.
- **s2: near-full at 64 images** (s2b0 2297/2305, a persistent tiny deficiency → near-singular).
- **s3 (deep, 4×4=16 patches/img): rank GROWS with images** — s3b0 2651(64)→3184(128)→4294(256)→4609(512). The
  interpolation threshold (rank=p) is crossed between **256 and 512 images**; the submitted 1024 is just past it.

## Findings (evidence class in brackets)
1. **[exact/deterministic] Effective dimensionality ≪ nominal, everywhere.** Despite nominal `rows/p` up to 1817×
   and full numerical rank, **stable rank is only 1.5–4.8**. Spatial patches from an image are extremely correlated,
   so the correction effectively fits a ~2–5 dimensional design regardless of p. **Any "double-descent by nominal
   patch count" reasoning is misleading** — the effective problem is always low-dimensional; the informative axis is
   the singular-value tail, not `rows/p`. *(Section 21 Q5.)*
2. **[exact/deterministic] The downsampling block0s (projected shortcut) are catastrophically worse conditioned than
   the identity block1s.** cond(G) s3b0=7.3e10 vs s3b1=1.0e5; s2b0=**3.7e18 (numerically singular)** vs s2b1=1.2e4.
   The **submitted anchor s3b0 is a block0** — inherently among the worst-conditioned targets. Likely because the
   strided, channel-doubling conv1 produces post-bn2-relu features with a sharper spectrum. This is the structural
   root of the s3b0-vs-s3b1 gap seen in theorem validation (float32 error) and the sensitivity instabilities.
3. **[exact/deterministic] Ridge λ=1e-3 rescues but does NOT fully stabilize the block0s.** It turns s3b0 from
   cond 7.3e10 → 8.1e7 and s2b0 from 3.7e18 → 7.3e8 — i.e. from (near-)singular to merely ill-conditioned. Block1
   blocks are well-conditioned with or without ridge (~1e4–1e5). **So the ridge is load-bearing specifically for the
   downsampling blocks**, directly explaining the **λ=0 catastrophic failure** observed in the sensitivity sweep
   (`sensitivity_cifar_summary.md`: λ=0 → acc 10%, singular solve). *(Section 21 Q7 → NO: λ=1e-3 leaves s2b0/s3b0 at
   cond ~1e8, only marginally stabilized.)*
4. **[exact/deterministic] The submitted block sits just past its interpolation threshold** (rank=p at ~512 imgs;
   submitted 1024). At CALIB=128 (used in an early theorem smoke test) s3b0 is **underdetermined** (rank 3184 < 4609)
   → the float32 solve was O(1)-wrong there. *(Section 21 Q6: the calibration size, not the bootstrap fraction, is
   what sits near the interpolation threshold for the deep blocks; bootstrap bf=0.05 resamples ~51 images, which is
   deep in the underdetermined regime for s3 — a likely contributor to bootstrap-induced diversity AND to the
   ss=2048 instability. Full calib×ridge×OOD phase diagram is the next step.)*

## Connections to prior results
- **λ=0 sensitivity failure** ← finding 3 (s3b0/s2b0 singular without ridge).
- **s3b0 float32 correction error (theorem validation, ~1%)** ← finding 2 (cond 8.1e7).
- **ss=2048 sensitivity instability** ← plausibly a conditioning/chunk-interaction near the deep-block threshold
  (finding 4); the calib×ridge phase diagram (Section 2.2–2.4) will test this directly with a condition-number probe.

## Next (Section 2.2–2.4, compute-heavier)
Calibration-size × ridge phase diagram with OOD outcomes and a double-descent probe, focused on s3b0 (submitted) and
a well-conditioned contrast (s3b1). Given finding 1, the phase-diagram x-axis will be **stable rank / singular-value
tail**, not nominal rows/p. Spectral-intervention comparison (ridge vs truncated-SVD vs PCR) near the s3 threshold.
