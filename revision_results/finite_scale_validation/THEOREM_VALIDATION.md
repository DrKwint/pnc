# THEOREM_VALIDATION.md — Round 1 (revised P&C mathematics)

Status of the exact-algebra claims of the revised P&C account, validated on real
MuJoCo activations. **This file covers the exact-identity core (1.1–1.3),
confirmed; the approximation/spectral sub-experiments (1.4–1.11) are in
progress** (tracked at the bottom).

Instrumentation: `experiments/scripts/pnc_theory/` (float64, SVD-based; core
unit-tested 189/189 to ≤1e-8, ≤1e-13 well-conditioned). Data: Ant-v5, Hopper-v5,
HalfCheetah-v5, seed 0, λ∈{0,1e-2,1,100}, members {0,1,3,5}, both perturbed
blocks, 1024 test points/regime. Raw: `artifacts/pnc_theory/round1/*.json`.

## Theorem-by-theorem status table

| Claim | Statement | Status | Evidence (rel-err, float64) |
|---|---|---|---|
| **Correction identity (Eq 1)** | `Θ̂_v = Θ − Θ ΔX_vᵀ X_v G_v⁻¹` (toward-Θ); default is toward-0, exact at λ=0 | **CONFIRMED** (under stated ridge convention) | SVD-pinv form median **9.7e-14**, max 2.6e-11; G⁻¹ closed form (λ>0) median **1.4e-12** |
| **Test-point residual (Eq 2/4)** | `r_v(x)=Θ[Δh̄_v − ΔX_vᵀ X_v G_v⁻¹ h̄_v] = Θ g_v` | **CONFIRMED — exact, all regimes** | ID median **9.4e-14**; Near/Mid/Far medians 7.3e-14 / 6.9e-14 / 5.2e-14 (max ≤2.3e-11) |
| **Transfer defect (Eq 3)** | `g_v = Δh̄_v − ΔX_vᵀ α_v`, `α_v = X_v G_v⁻¹ h̄_v`, `r=Θg` | **CONFIRMED** | same rows as Eq 2/4 (identical decomposition) |
| **Normal equations (1.1)** | `Xvᵀ(Xv Θ̂ᵀ−XΘᵀ)+λ(Θ̂ᵀ−Θᵀ)=0` | **CONFIRMED** | median **5.6e-16**, max 3.5e-14 |
| **Calibration hat identity (Eq 5)** | `R_{S,v}=(I−H_{v,λ})ΔX_vΘ` | **CONFIRMED** | median **1.1e-13**, max 6.0e-12 |
| **λ=0 interpolation (1.3)** | `n≤p` full row rank ⇒ `R_{S,v}≈0` | **CONFIRMED in synthetic**; real MuJoCo is n≫p (no interpolation) | unit test; real R_S≠0 (10–15% of ‖target‖) |
| **Ridge convention (audit)** | default shrinks toward **0**, not Θ; differ by exact `−λ G⁻¹Θ` | **CONFIRMED & DERIVED** | `AUDIT_IMPLEMENTATION.md §3`; unit-tested |
| **λ=0 weight non-identifiability** | cond(G)≈1e32 ⇒ fitted weights not reproducible; residual is | **CONFIRMED (numerical)** | float32-vs-float64 weight rel-err 0.6–11 at λ=0 → 2e-5 at λ=100 |
| Spectral gains (1.4) | correction governed by `s/(s²+λ)` filter | instrumented, **run pending** | — |
| First-order limit (1.6) | `E₁(t)∝t²` below activation-flip scale; `A_S(x)` valid model | **CONFIRMED locally, BROKEN at operating scale** | slope 1.56–1.78 (≈2, ReLU-kink contaminated); valid s≲0.04–0.16; **benchmark s=5–50 ⇒ Erel≈1.0** |
| Stability bounds (1.5/1.7) | Eq (6),(7) hold with `‖X_vG_v⁻¹‖≤1/(2√λ)` | **run pending** | — |
| Random-sketch (1.8) | `E‖A_Sv‖²=σ²‖A_S‖_F²`; var law | **run pending** | — |
| Local→final bridge (1.9/B.1) | `Δy≈J_down·r_1`; local residual controls final disagreement | **CONFIRMED — local residual is a monotone, near-linear driver of final Δy** | cos(Δy,J_down r_1)≈**1.00** all regimes/scales; Spearman(‖r_1‖,‖Δy‖) 0.63–0.87; Taylor relerr ID 0–0.04, Far 0.14–0.25 |
| Phase diagram (1.10) | n/p × λ error surface, interpolation peak | **CONFIRMED — double-descent peak at n≈p; ridge removes it** | held-out ID residual peaks at n/p≈1 (Ant 171, HC 12, Hopper 2.8) with train residual ~1e-13; small λ cuts it 1–3 orders |
| Spectral intervention (1.11) | TSVD/PCR removes ID pollution, keeps OOD signal | **NOT SUPPORTED at n≫p** (local-residual); truncation worsens OOD/ID separation | Far/ID ratio falls with truncation (Ant 70→4.5, Hopper 4.7→1.4, HC 4.0→2.1); full-rank OLS best |

## Key confirmed results (1.1–1.3)

1. **The exact residual identity is exact.** Across all three environments, both
   perturbed blocks, every λ, and every regime (correction set, held-out ID,
   Near, Mid, Far), the reconstructed residual `Θ g_v(x)` matches the direct
   residual `Θ̂_v h̄_v(x) − Θ h̄(x)` to **~1e-13 relative** (float64). This is a
   genuine exact algebraic identity of the implemented correction — **safe for
   the camera-ready paper** as an exact statement.

2. **It holds out-of-distribution.** The identity is not an ID phenomenon: it is
   exact on Near/Mid/Far shifts too (the residual *magnitude* grows with shift —
   Ant |r|_rms: corr 0.022 → ID 0.031 → Near 0.082 → Mid 0.116 → Far 0.405 — but
   the *identity* stays exact). So the decomposition `r = Θ g_v` is a valid lens
   for OOD disagreement, not just calibration fit.

3. **The default (λ=0) correction is a numerically singular pseudoinverse.**
   cond(G) ≈ 1e31–1e32 on all envs; the fitted weights are not reproducible
   across precisions (see audit §4). Exact identities still hold because they are
   evaluated in float64 via the stable SVD form; but any *weight-space* claim at
   λ=0 must be treated as ill-defined. Ridge (λ≳1e-2) restores weight
   reproducibility.

## 1.6 — First-order sensitivity is valid only far below the operating scale

The exact operator `A_S(x)=Θ[J_x−Σ_i w_i^λ(x)J_{x_i}]` was validated against the
refit residual over a log grid of perturbation magnitudes `s=‖dW‖_F` (block l1,
3 directions, canonical toward_orig; λ∈{0,1}):

| env | E₁ log-log slope (small-s) | first-order valid (Erel≤10%) | ReLU flips reach 1% | benchmark s | Erel at benchmark |
|---|---|---|---|---|---|
| Ant-v5 | 1.78 | s ≲ **0.16** | s≈1.6 | 5–50 | **≈1.0 (broken)** |
| Hopper-v5 | 1.61 | s ≲ 0.04 | s≈0.40 | 5–50 | ≈1.0 (broken) |
| HalfCheetah-v5 | 1.56 | s ≲ 0.06 | s≈0.63 | 5–50 | ≈1.0 (broken) |

**Finding (answers synthesis Q2).** The first-order/linearized picture is
quantitatively accurate only for `‖dW‖_F ≲ 0.04–0.16`, whereas the benchmark
operates at `‖dW‖_F = 5–50` (empirically ‖dW‖_F ≡ perturbation_scale) — **30–300×
beyond the valid range**, with a large fraction of ReLU units flipped. So the
implemented P&C at its operating point is a **strongly nonlinear** map; the
first-order sensitivity `A_S(x)` and anything built on it (e.g. the random-sketch
theorem 1.8) are **not** a quantitatively valid model there. The breakdown is
λ-independent (driven by ReLU flips, not correction conditioning). The slope <2 is
the ReLU non-smoothness bleeding into the remainder even at small s. **Camera-ready
implication: present `A_S`/random-sketch as an infinitesimal-limit intuition, not
as a quantitative account of the deployed correction.**

## 1.10 — Double-descent interpolation peak, removed by ridge

Sweeping calibration size n/p × ridge λ (block l1, benchmark scale P=10,
canonical toward_orig), the held-out ID **local residual** shows the classic
interpolation peak, while the **training** residual confirms interpolation:

| env | held-out peak location | peak RMS (λ=0) | train RMS at peak | best λ (peak RMS) |
|---|---|---|---|---|
| Ant-v5 | n/p ≈ 0.75–1.1 | 171 | ~5e-13 | 1e-2 → **0.06** |
| Hopper-v5 | n/p = 1.0 | 2.8 | ~1e-13 | 1 → **0.13** |
| HalfCheetah-v5 | n/p = 1.0 | 12.3 | ~2e-12 | 1 → **0.35** |

Training residual is machine-zero for **all** n/p≤1 (the correction interpolates
whenever n≤p) and rises smoothly for n/p>1; the held-out residual explodes
precisely at that n≈p boundary — the variance signature of double descent. **A
small ridge removes the peak entirely (1–3 orders of magnitude).** This is the
direct generalization-theory case for a ridge floor over the λ=0 default
(consistent with `VERDICT_conditioning_selection.md`), and it is safe to state as
an empirical regularity. Heatmaps: `artifacts/pnc_theory/round1/round1_phase_*.png`.

## 1.11 — Weak-mode suppression does NOT improve OOD/ID separation at n≫p

Comparing OLS / ridge / truncated-SVD / PCR on the local residual (block l1,
P=10, full n=4096 calibration), suppressing the weakest correction modes **raises
ID residual faster than OOD residual**, so the OOD/ID separation gets worse:

| method | Ant Far/ID | Hopper Far/ID | HalfCheetah Far/ID |
|---|---|---|---|
| OLS (k=201, λ=0) | **70.0** | **4.74** | **4.04** |
| TSVD k=100 | 11.6 | 3.29 | 3.33 |
| TSVD k=20 | 5.9 | 1.71 | 2.54 |
| ridge λ=100 | (≈OLS until λ≫1) | 2.63 | 3.43 |

**Finding (contradicts Insight-11 hypothesis at the operating point).** The
premise "weak correction modes are ID pollution; remove them to keep only
geometric OOD disagreement" is **not** borne out at the benchmark's calibration
size (n=4096 ≫ p=201). There the correction is over-determined and *not* overfit,
so weak modes carry useful ID signal — truncating them degrades ID more than OOD
and *shrinks* the Far/ID ratio; full-rank OLS is best. Ridge with λ≲1 is nearly a
no-op (consistent with prior "λ barely matters at full calibration"). Spectral
suppression is beneficial **only near the interpolation regime n≈p** (Round 1.10),
where it acts as the double-descent cure. **Camera-ready implication: do not claim
TSVD/ridge improves OOD detection in general; scope any such claim to scarce
calibration (n≈p).** (Caveat: measured on single-direction local residual; the
ensemble-AUROC version is queued with the Round-5/6 pipeline — but the residual
ratio is a direct separation proxy and the trend is unambiguous.)

## 1.9 / B.1 — The exact local residual cleanly drives the final disagreement

The last correction (l3→l4) yields a corrected l4-preactivation whose exact local
residual `r_1(x)` is the *entire* deviation entering the head `F_down=mean∘relu`.
So `Δy(x)=F_down(z_0+r_1)−F_down(z_0)`, first-order `≈J_down(z_0)r_1`:

| env | cos(Δy, J_down r_1) | Spearman(‖r_1‖,‖Δy‖) | Taylor relerr ID / Far (P=10) | gain (median) |
|---|---|---|---|---|
| Ant-v5 | **1.00** (all) | 0.79–0.87 | 0.00 / 0.14 | 3.4–4.0 (amplifies) |
| Hopper-v5 | 0.99–1.00 | 0.66–0.72 | 0.04 / 0.18 | ~0.31 (attenuates) |
| HalfCheetah-v5 | 0.97–1.00 | 0.63–0.87 | 0.02 / 0.25 | ~1.0 |

**Finding (answers synthesis Q2, supports amendment A.1).** Unlike the first-order
*sensitivity* `A_S` (which linearizes correction-in-perturbation and is broken at
benchmark scale, 1.6), the *downstream* map linearizes cleanly in the **exact**
local residual: `Δy` is essentially collinear with `J_down r_1` (cos≈1.0) at every
scale, and `‖r_1‖` monotonically ranks `‖Δy‖` (Spearman 0.63–0.87). The linear
magnitude approximation is excellent on ID/Near/Mid and degrades only on Far at
large scale (head-ReLU flips), yet the *direction* survives. Counterexamples
(large-local/small-final and vice versa) are <1.5%. **So the exact finite-scale
local residual is a monotone, quantitatively useful explanation of final P&C
disagreement — the right primary object, as the amendment directs.** The env-specific
downstream gain (Ant amplifies ~4×, Hopper attenuates ~0.3×) is a stable per-env
constant. Raw: `artifacts/pnc_theory/bridge/`.

## Per-environment consistency (seed 0)

| env | Eq2/4 residual (ID median) | Eq5 (median) | |r|_rms ID→Far | λ=0 cond(G) | λ=0 weight rel-err |
|---|---|---|---|---|---|
| Ant-v5 | 9.4e-14 | 1.1e-13 | 0.031 → 0.405 | 1.2e32 | 11 |
| HalfCheetah-v5 | 5.4e-13 | 5.4e-13 | 0.160 → 0.707 | 6.1e31 | 0.60 |
| Hopper-v5 | 8.2e-13 | 8.2e-13 | 0.057 → 0.224 | 8.2e31 | 0.79 |

## Statements safe for the camera-ready paper (from 1.1–1.3)

- The P&C correction is exactly a bias-augmented finite-sample ridge/least-squares
  fit of perturbed→original-next-preactivation; the post-correction residual
  admits the **exact** closed form `r_v(x)=Θ[Δh̄_v(x) − ΔX_vᵀ X_v G_v⁻¹ h̄_v(x)]`
  (default λ=0; add `−λ h̄_vᵀG_v⁻¹Θ` for λ>0 toward-0).
- This residual decomposes exactly as `Θ g_v` with `g_v` a purely
  representation-level transfer defect — separating the *geometry* of the repair
  from the *following weights* Θ.

## Statements that must remain hypotheses / empirical (pending 1.4–1.11)

- Any first-order (`A_S(x)`) approximation quality, spectral-gain governance,
  local→final relevance, and the causal "TSVD removes ID pollution" claim — **not
  yet run**; do not state as validated.

## Reproduce
```
.venv/bin/python experiments/scripts/pnc_theory/validate_round1.py --env Ant-v5 --seed 0
.venv/bin/python experiments/scripts/pnc_theory/validate_round1.py --env Hopper-v5 --seed 0
.venv/bin/python experiments/scripts/pnc_theory/validate_round1.py --env HalfCheetah-v5 --seed 0
```
