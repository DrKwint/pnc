# AUDIT_IMPLEMENTATION.md — Section 0.2 implementation audit (MuJoCo P&C)

**Scope.** Audit of the *implemented* P&C correction against the revised
ridge-regression account (spec Eq 1–4) **before** running experiments, as
required by Section 0.2. Everything below is read from code and confirmed
numerically on real Ant/Hopper/HalfCheetah activations; nothing is assumed.

Code under audit: `ensembles.py` (`PJSVDEnsemble`, `_ls_or_ridge_solve` l.146,
`_precompute_sequential_ls` l.411, `_forward_member_sequential` l.697); build
path `experiments/scripts/measure_effective_rank.py:_build_pnc_ensemble` (l.162).
Instrumentation reproducing it in float64: `experiments/scripts/pnc_theory/`.

---

## 0. One-paragraph verdict

The MuJoCo P&C correction **is exactly the finite-sample ridge/least-squares
problem** the revised account describes: for each perturbed layer it fits an
affine map from the *perturbed* post-activation to the *original next-layer
pre-activation* on an ID calibration set, bias-augmented. The spec's Eq (1)–(4)
hold **exactly** (confirmed to ~1e-13 in float64) with **one substantive
caveat**: the default ridge shrinks the correction **toward zero**, whereas
Eq (1)–(4) as written assume shrinkage **toward the original weights Θ**. The two
coincide at **λ=0** (the default configuration, a pseudoinverse solve) and differ
by an exact, derived term `−λ·h̄_v(x)ᵀ G_v⁻¹ Θ` for λ>0. Both formulations are
implemented and cross-checked. A second, numerical finding: at λ=0 the
correction Gram is **numerically singular (cond(G)≈10³¹–10³²)**, so the fitted
*weights* are not reproducible across float32/float64 — only their in-distribution
*action* (the residual) is stable.

---

## 1. Answers to the Section 0.2 audit checklist

| # | Question | Finding | Evidence |
|---|---|---|---|
| a | Is the **bias** included in the correction design? | **Yes.** Design is `h_aug = [h_pert, 1]` (bias is the **last** column, not first as in the spec's `[1;h]` — a harmless permutation). | `ensembles.py:563-564` |
| b | Is ridge **centered at the original affine layer**? | **No, by default.** `_ls_or_ridge_solve` shrinks toward **zero** (`W=(HᵀH+λI)⁻¹Hᵀt`). Shrink-toward-Θ exists but is **opt-in** (`ridge_toward_orig=True`, off by default). **This is the discrepancy vs Eq (1).** | `ensembles.py:165-172`, `211-215`, `566-574` |
| c | Exact meaning & scaling of **λ**? | **Absolute** Tikhonov weight on `HᵀH+λI`; **not** normalized by `tr(HᵀH)/p`. Default **λ=0** (⇒ pseudoinverse via `lstsq`). Paper grid `{0,1e-4,1e-2,1,100}`. A scale-free `λ_rel = λ/(tr(XᵀX)/p)` is used only in our sweeps. | `ensembles.py:166-167`; `inventory.md` |
| d | **Solver** (normal eq / QR / SVD / pinv)? | **λ=0:** `jnp.linalg.lstsq` (SVD min-norm pseudoinverse). **λ>0:** **normal equations** `jnp.linalg.solve(HᵀH+λI, …)` (LU), **in float32**. No QR; no SVD for the ridge path. | `ensembles.py:165-172` |
| e | Are conv spatial positions treated as independent regression rows? | **N/A** for the MuJoCo/MLP path (dense features, one row per state). Applies only to the CIFAR conv path (separate track). | — |
| f | Are activations **centered / standardized**? | **No.** Raw `h_pert` used directly; only bias augmentation. | `ensembles.py:562-564` |
| g | Correction **target** = next-layer preactivation? | **Yes.** `target = ref_h @ W_next_orig + b_next_orig` = the **original pre-activation** at the next layer, i.e. `X Θ`. | `ensembles.py:543-546` |
| h | Is the calibration set **shared across members**? | **Yes** by default (all members use `X_sub`, ≤4096 ID points). Per-member **bootstrap** resampling is opt-in (`bootstrap_frac>0`). | `ensembles.py:515-528` |
| i | Perturbation applied **before or after** the nonlinearity? | It is a **weight** perturbation `dW` on the perturbed layer's pre-activation: `h_pert = σ(h(W+dW)+b)`. So `h_v(x)` is the **post-nonlinearity** representation of a pre-nonlinearity weight perturbation. | `ensembles.py:538` |
| j | How is perturbation **scale normalized** across layers? | Per-layer unit random directions (`v_opts`, K=20) × `sigmas` (=1) × global `perturbation_scale` × optional per-member radius multiplier. Not activation-normalized. | `measure_effective_rank.py:180-196`; `ensembles.py:453-465` |

**Which layers.** Even hidden layers are perturbed (`l1`=layer0, `l3`=layer2 of a
4×200 ReLU MLP); each is corrected at the **following** layer (corr_idx = 1, 3).
Correction is **sequential**: block 1's design is built on the *already
perturbed-and-corrected* upstream activations (`_precompute_sequential_ls`
l.525-582), not the original ones. n=4096, p=201 (200 hidden + bias), d=200.

---

## 2. Mapping the implementation to the spec notation

Per member, per perturbed block (bias-augmented, **bias last**, row-vector layout):

```
X   (n,p)  rows [h_orig(x_i), 1]     original post-activation at the perturbed layer
Xv  (n,p)  rows [h_pert(x_i), 1]     perturbed post-activation (incl. upstream corr.)
Θ   (p,d)  [W_next_orig ; b_next_orig]   the following affine layer; preact = X Θ
target = X Θ                          the regression target (orig next-layer preact)
G   = XvᵀXv + λI                      the correction Gram (spec G_v)
```

The implemented solve is `Θ̂ᵀ ≡ W_aug = ridge_solve(Xv, XΘ, λ, w_prior)`.

- **λ=0:** `W_aug = pinv(Xv) (XΘ)` — min-norm least squares.
- **λ>0, default (toward 0):** `Θ̂ = G⁻¹ Xvᵀ X Θ`.
- **λ>0, `ridge_toward_orig`:** `Θ̂ = G⁻¹ (Xvᵀ X Θ + λ Θ)`.

## 3. The exact implemented identity (derived, since impl ≠ Eq 1 for λ>0)

Spec **Eq (1)** `Θ̂_v = Θ − Θ ΔX_vᵀ X_v G_v⁻¹` is the **toward-Θ** solution.
Transposing to code layout it equals `Θ − G⁻¹ Xvᵀ ΔX Θ`. The **default**
(toward-0) solve differs by exactly:

```
Θ̂_zero  =  Θ̂_orig  −  λ G⁻¹ Θ                                (weights)
r_zero(x) = r_orig(x)  −  λ · h̄_v(x)ᵀ G⁻¹ Θ                    (test residual)   [Eq 2']
g_zero(x) = g_orig(x)  −  λ G⁻¹ h̄_v(x)                         (transfer defect)
```

so the spec's transfer-defect chain `r_v = Θ g_v`, `g_v = Δh̄_v − ΔX_vᵀ α_v`,
`α_v = X_v G_v⁻¹ h̄_v` holds **verbatim only for the toward-Θ (ridge_toward_orig)
variant**. For the default variant it holds with the extra `−λ G⁻¹ h̄_v` bias
term. **At λ=0 (the default run) the term vanishes and the spec's Eq (1)–(4) are
exact as written.** Both variants are implemented in
`experiments/scripts/pnc_theory/linalg.py` (`mode="toward_orig"` / `"toward_zero"`).

Derivation is recorded in `linalg.py` docstrings and unit-tested
(`test_linalg.py`, 189/189 exact identities to ≤1e-8, well-conditioned ≤1e-13).

## 4. Numerical findings that change how results must be read

1. **Default λ=0 ⇒ numerically singular Gram.** cond(G) ≈ **10³¹–10³²** on all
   three envs (post-ReLU designs are rank-deficient: numerical rank ≈ 160–200 of
   201). Consequences:
   - The fitted **weights `Θ̂` are not a well-defined numerical object**: float32
     (implementation) vs float64 (reference) weights differ by **rel-err ≈ 0.6–11
     (Frobenius)** — *not* float32 round-off, but genuine weak-mode
     amplification. Both still (i) satisfy the normal equations to ~1e-6, (ii)
     leave the identical calibration residual, and (iii) yield an **ID residual
     `r(x)` that is stable**. The instability lives entirely in the weak singular
     directions of `Xv`.
   - Adding ridge conditions the problem and makes the weights reproducible:
     weight rel-err falls monotonically with λ (Ant: 11 → 4.6e-3 → 5.8e-4 → 1.8e-5
     for λ=0, 1e-2, 1, 100; cond(G): 1e32 → 2e6 → 2e4 → 2e2). This is the direct
     numerical motivation for the ridge/TSVD interventions of Rounds 1.10–1.11.
2. **float64 is mandatory for identity checks** (`jax_enable_x64` is **off** by
   default here). All theory quantities are computed in float64 via the SVD
   filter-factor form `s/(s²+λ)`, which stays exact and stable across the
   well-/ill-/under-determined regimes (`linalg.py`).
3. **The instrumentation faithfully mirrors the pipeline.** The extracted
   `(Θ̂_impl, reps)` reproduce the ensemble's own corrected pre-activation
   (`predict_intermediate_and_corrected`) to **~2e-7** (float32) on all envs — so
   downstream analyses operate on exactly what the benchmark computes.

## 5. Consequences for the experimental program

- Report results under **both** ridge conventions; state clearly that the
  camera-ready P&C default is **λ=0 (pseudoinverse)**, for which Eq (1)–(4) are
  exact. For any λ>0 claim, use Eq (2′) (toward-0) or switch on
  `ridge_toward_orig` to make Eq (1)–(4) literal.
- Treat **weight-space** correction norms at λ=0 as unreliable; prefer
  **residual / action-space** quantities, or measure norms only under a
  conditioning floor (λ≳1e-4) — consistent with the prior
  `VERDICT_conditioning_selection.md` recommendation of a small ridge floor.
- The λ=0 weak-mode instability is itself a **first-class object of study** (it
  is a source of test-time disagreement independent of geometric transfer
  failure) — Rounds 1.11, 4.8, 5.5.

## 5b. RESOLUTION of the ridge-convention question (amendment B.2)

The toward-zero vs toward-orig discrepancy is **empirically irrelevant at the
submitted λ**. Decomposing the default (toward-zero) residual as
`r_v^(0) = Θ g_v − λΘG_v⁻¹h̄_v` on all three envs, at λ∈{1e-4,1e-2}:

| quantity | λ=1e-4 | λ=1e-2 |
|---|---|---|
| ‖transfer defect‖ (median) | 0.7–8.7 | 0.7–8.6 |
| ‖ridge-bias term‖ (median) | ~1e-6–1e-5 | ~4e-4–3e-3 |
| ridge-bias **energy fraction** | **0.0%** (p95 0%) | **0.0%** (p95 0%) |
| Δ(toward_orig−toward_zero) AUROC / RMSE / NLL | **±0.000** | **±0.000** |

**Why:** the convention difference `−λG⁻¹Θ` lives in the weak/near-null weight
modes (cond(G)≫1 at small λ), but test points `h̄_v(x)` lie in the well-supported
subspace, so `λΘG⁻¹h̄_v` is tiny — predictions are convention-invariant at the
operating λ even though the weights differ. **Verdict: the manuscript may state
the toward-orig Eq (1)–(4) form directly; it is numerically identical to the
shipped toward-zero code at submitted λ. No implementation change and no dual
description are needed.** (The distinction only matters at λ≳1, which is not a
submitted setting.) Raw: `artifacts/pnc_theory/ridge_center/`.

## 6. Reproduce

```
.venv/bin/python experiments/scripts/pnc_theory/test_linalg.py            # 189/189 exact identities
.venv/bin/python experiments/scripts/pnc_theory/validate_round1.py \
    --env Ant-v5 --seed 0 --lams 0,1e-2,1,100 --members 0,1,3,5           # real-activation validation
# artifacts: artifacts/pnc_theory/round1/round1_validation_<env>_seed0.json
```
