# CIFAR Theorem Validation — exact convolutional finite-scale identities (Phase 1)

**Date:** 2026-07-24 · seed 0 · blocks s3b0 (submitted anchor) & s3b1 · CALIB=1024 (submitted operating point)
· λ=1e-3 · scale=25 · K=20 · 3 members checked · eval on ID test + CIFAR-100 + SVHN (64 imgs each).
Feature extraction in the network's **native float32**; reference linear algebra in **numpy float64**.
Raw: `theorem_validation_raw.json`. Harness: `_theorem_validation.py`.

## Convolutional finite-scale form (as implemented)
Target block: `bn1→relu→conv1(w1)→bn2→relu → U(x)`; corrected layer = `conv2` (+ new bias). Patch operator
`Φ(x)=extract_patches(U(x),k=3,stride=1)` (im2col via `conv_general_dilated_patches`, feature order
`(C_in,kh,kw)`); `Θ=flatten(w2_orig)` (`p−1=9·C_in` rows). Augmented `X_v=[1,Φ_v]`, `p=9·C_in+1`.
- The shipped ridge regresses `[1,Φ_v]` onto `R = XΘ − X_vΘ = −ΔX_v·Θ` and sets `w2_new=w2_orig+Δ`,
  which is **exactly the toward-original ridge C1**:  `Θ_v = Θ − G_v⁻¹ X_vᵀ ΔX_v Θ`, `G_v=X_vᵀX_v+λI`
  (λ absolute, on the full augmented Gram incl. the bias coordinate). Derivation in `CIFAR_IMPLEMENTATION_AUDIT.md` §0.3.
- Local residual **C2–C4**: `R_v(x) = [1,Φ_v(x)]Θ_v − Φ(x)Θ = G_v(x)Θ`, with transfer defect
  `G_v(x) = ΔΦ_v(x) − Φ_v(x)G_v⁻¹X_vᵀΔX_v` (`ΔΦ_v=[1,Φ_v]−[1,Φ]`).
- Because the block's shortcut and `relu(bn1(h))` are computed **before** the perturbed conv1, they are identical
  for the original and any corrected member ⇒ **the block-output residual (after the residual add) equals the
  conv2 residual `R_v(x)` exactly** — no shortcut/BN leakage. (Used by the Section-4 output bridge.)

## 1.2 Patch operator (im2col) reproduces the real conv2
`Φ(x)Θ` vs the framework's `conv2(U(x))` (pre-residual), relative error:
- s3b0: **8.6e-5**, max-abs 5.5e-5 · s3b1: **9.5e-5**, max-abs 5.6e-5.
Consistent with ordinary float32 convolution reduction-order error (the two use different XLA kernels). The
patch operator is correct. *(A float64 re-extraction would drive this to ~1e-12; the shipped path is float32,
so this is the faithful figure for what the network actually computes.)*

## 1.3 Exact correction identity (C1)
| | s3b0 | s3b1 |
|---|---|---|
| **ridge normal-eq == closed-form C1** (f64, self-consistent) | **2.4e-17** | **4.2e-17** |
| implemented (float32) vs f64 solve — kernel | 1.6e-3 … 1.2e-2 | 1.1e-3 … 1.3e-3 |
| implemented (float32) vs f64 solve — bias | 1.4e-2 | 3.1e-2 |
| cond(G_v) | **8.6e7** | 7.5e4 |

- **C1 is EXACT**: the closed-form `Θ_v = Θ − G_v⁻¹X_vᵀΔX_vΘ` matches the normal-equations solve to **machine
  precision (~1e-17)**. The manuscript's convolutional correction identity holds.
- The **shipped float32 solve** recovers the correction to ~2–3 significant digits on the kernel and ~1–2 digits
  on the **bias** (the intercept coordinate is the least-well-conditioned). Error **grows with cond(G_v)** and
  with under-determination (at CALIB=128, n/p=0.44, the float32 solve differs from f64 by **O(1)** — see note).

## 1.4 Exact residual identity (C2–C4) — holds on ID and OOD alike
Per dataset (member 0), relative error:
| dataset | s3b0 patch-vs-network | s3b0 C4 algebra | s3b1 patch-vs-network | s3b1 C4 algebra |
|---|---|---|---|---|
| ID test | 5.4e-4 | **1.7e-15** | 6.9e-4 | 1.9e-15 |
| CIFAR-100 (near) | 4.8e-4 | 1.5e-15 | 4.5e-4 | 1.3e-15 |
| SVHN (far) | 4.3e-4 | 1.3e-15 | 3.9e-4 | 1.2e-15 |

- **C4 is EXACT** (`G_v(x)Θ` == patch formula to ~1e-15) and — as the theory requires — **identically exact on
  ID, Near-OOD, and Far-OOD**. The finite-scale residual identity is not an ID-only phenomenon.
- The **patch formula reproduces the real network conv2 residual** to float32 precision (~4–7e-4, max-abs ~5e-5,
  per-image max <1.2e-3) uniformly across ID/OOD — the local-residual bridge to the actual network is validated.

## Findings (evidence class in brackets)
1. **[exact identity]** The convolutional finite-scale identities **C1 (correction)** and **C2–C4 (residual /
   transfer defect)** hold to machine precision, on ID and OOD, for both candidate blocks. The revised theory's
   convolutional form is confirmed. *(Section 21 Q2 → YES.)*
2. **[exact identity]** Block-output residual == conv2 residual (shortcut/BN cancel) — the propagation target for
   Section 4 is exact.
3. **[multi-seed-pending empirical]** The **submitted block s3b0 is ~1000× more ill-conditioned than s3b1**
   (cond 8.6e7 vs 7.5e4). The shipped **float32** correction therefore carries ~0.2–1.2% kernel / ~1.4% bias
   error at s3b0 (vs ~0.13% at s3b1). Exactness is a property of the *math*; the *shipped numerics* are float32
   and degrade with conditioning — worst at the submitted block. This motivates the Section-2 conditioning phase
   diagram and connects to the observed λ=0 / calib-size instabilities.

## Section 3 — First-order validity range (submitted scale 25 is FAR outside the linear regime)
Single unit-norm direction `v` in conv1 param space, scale `t` swept; correction refit (float64 toward-original
ridge) at each `t`; corrected residual `R_{tv}(x)` via the validated patch formula; first-order prediction
`t·A_S(x)v` with `A_S(x)v` estimated by finite difference at the smallest scale. s3b0, seed 0, ID+CIFAR100+SVHN
(64 imgs). Raw: `firstorder_sweep_raw.json`.

| scale t | s_rel=‖Δw₁‖/‖w₁‖ | cosine(R_{tv}, t·A) [ID] | relerr / ‖linear‖ [ID] |
|---|---|---|---|
| 0.01 | 0.001 | 0.92 | 0.38 |
| 0.1 | 0.014 | 0.90 | 0.44 |
| 1.0 | 0.139 | 0.86 | 0.51 |
| 2.5 | 0.347 | 0.78 | 0.63 |
| 5.0 | 0.69 | 0.64 | 0.79 |
| 10 | 1.39 | 0.41 | 0.97 |
| **25 (submitted)** | **3.47** | **0.15** | **1.06** |
| 50 | 6.94 | 0.07 | 1.03 |
| 100 | 13.9 | 0.03 | 1.01 |

- **[multi-seed-pending empirical] The submitted scale 25 is deeply outside the first-order regime.** At s_rel=3.47
  the linearization has near-zero alignment (cos≈0.15) and 100%+ relative error — the actual corrected residual is
  essentially orthogonal to its first-order prediction. Linearization degrades **monotonically** and is already
  poor (cos 0.86, relerr 0.51) at scale 1 (s_rel 0.14). Near/Far OOD track ID closely (SVHN slightly less nonlinear,
  cos 0.20 at 25). **This is exactly why the finite-scale EXACT identities (C1–C4) — not the linearization — are the
  right object at the operating scale**; the first-order "corrected sensitivity" does not predict finite behaviour
  at scale 25. Qualitatively matches MuJoCo, now verified for the CIFAR conv correction (not assumed).
- **Honest caveat:** the very-small-scale reference is limited by ReLU activation flips + float32 feature extraction
  (t=0.005→0.01 finite-difference cross-check disagrees ~38%; small-scale remainder slope ≈12, not the quadratic 2).
  So the *precise* first-order boundary and a quadratic-remainder claim are **not cleanly resolvable on the shipped
  float32 path** — the linear regime is narrower than scale 0.01 (≥3500× below submitted). The *large-scale*
  nonlinearity conclusion is precision-independent and robust. A float64 feature-extraction re-implementation would
  be required to resolve the sub-0.01 regime; deferred as low-value (the operating-scale answer is unambiguous).

## Summary of Phase-1 theorem status
| identity | status | evidence |
|---|---|---|
| Patch operator = conv2 | ✅ (float32 ~1e-4; exact in f64) | 1.2 |
| C1 correction identity | ✅ EXACT (~1e-17) | 1.3 |
| C2–C4 residual identity (ID & OOD) | ✅ EXACT (~1e-15) | 1.4 |
| Patch formula = real network forward | ✅ (float32 ~5e-4) | 1.4 |
| Block-output residual = conv2 residual | ✅ EXACT (shortcut cancels) | audit + 1.4 |
| First-order predicts finite behaviour @ scale 25 | ❌ NO (cos≈0.15) — finite-scale theory required | §3 |

> Note on precision: exact-algebra checks (C1 normal-eq vs closed-form; C4 vs patch formula) are numpy-float64 and
> self-consistent → ~1e-15. Network-consistency checks (patch-op; residual vs real forward; code float32 vs f64)
> reflect the shipped float32 path and are the honest operating-point figures. CALIB=128 (underdetermined)
> produced O(1) float32 solve error — the identity is numerically trustworthy only where G_v is adequately
> conditioned, which the Section-2 diagram will map per block.
