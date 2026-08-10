# CIFAR Local-Residual → Output Bridge (Phase 1 / Section 4)

**Date:** 2026-07-24 · anchor P&C s3b0, seed 0, scale 25, CALIB=1024 · M=8 members for the bridge, M=50 for the
disagreement targets · ID test + CIFAR-100 + SVHN (64 imgs each). Raw: `local_to_output_raw.json`;
harness: `_local_to_output.py`.

Setup: exact conv2 residual `R_v(x)` (validated patch formula) = block-output deviation (shortcut cancels).
Downstream `J_down(x)` = Jacobian of logits w.r.t. the block output, computed by `jax.jvp` through the remaining
blocks (frozen BN) + global pool + fc. Actual logit change `Δℓ_v(x) = ℓ_v(x) − ℓ_0(x)`.

## 4.2 Residual propagation to logits
| dataset | cos(Δℓ, J·R_v) median | relative-magnitude error | Spearman(‖R_v‖_F, ‖Δℓ‖) |
|---|---|---|---|
| ID test | 0.82 | 0.66 | −0.27 |
| CIFAR-100 (near) | 0.83 | 0.68 | 0.00 |
| SVHN (far) | 0.89 | 0.57 | 0.21 |

- **[multi-seed-pending empirical] The downstream Jacobian propagation of `R_v` captures the DIRECTION of the actual
  logit change well** (cos 0.82–0.89) — even at the finite operating scale 25, the map from block output to logits
  is directionally near-linear (the downstream perturbation is milder than the correction itself).
- **Magnitude is only ~40% accurate** (relmag 0.57–0.68): the later ReLU blocks are nonlinear, so first-order
  downstream under/over-shoots magnitude. The bridge is directional, not a magnitude predictor.
- **Raw local-residual norm does NOT predict logit-change magnitude** (Spearman −0.27 on ID, ~0 near, +0.21 far).

## 4.3 Which local statistic predicts disagreement? — only the Jacobian-projected residual
Spearman of each per-image local statistic (member-averaged) vs the M=50 ensemble disagreement, on ID:
| statistic | Sp(MI) | Sp(logit-cov-tr) | Sp(pred-entropy) |
|---|---|---|---|
| Frobenius ‖R_v‖ | −0.16 | −0.29 | −0.08 |
| mean-patch norm | −0.14 | −0.27 | −0.07 |
| max-patch norm | −0.14 | −0.27 | −0.06 |
| p95-patch norm | −0.14 | −0.27 | −0.06 |
| channelwise norm | −0.16 | −0.29 | −0.08 |
| GAP-residual norm | −0.03 | −0.16 | +0.02 |
| **J-projected ‖J·R_v‖** | **+0.80** | **+0.86** | **+0.72** |

Near/Far confirm the ordering (jproj leads): CIFAR-100 jproj +0.54/+0.77/+0.52 (raw ≈ 0); SVHN jproj
+0.48/+0.53/+0.19 (on far-OOD the raw norms gain some signal, +0.33/+0.55, but jproj still leads for MI/entropy).

## Findings
1. **[multi-seed-pending empirical] The exact local conv2 residual explains final logit/probability disagreement —
   but ONLY through the downstream Jacobian.** `‖J·R_v‖` predicts per-image MI / logit-covariance-trace / predictive
   entropy strongly (Sp 0.72–0.86 on ID), whereas the **raw local residual norm is uninformative or
   anti-correlated** (Sp ≈ 0 to −0.29). *(Answers Section 21 Q3: YES, in the Jacobian-projected sense.)*
2. This is a necessary correction to any "large local residual ⇒ large disagreement" intuition: the downstream
   Jacobian re-weights the residual heavily (different spatial/channel directions have very different output
   leverage), so raw magnitude at the block is a poor proxy. Disagreement is set by the residual's **component in
   the output-sensitive subspace**, not its size.
3. Consistent with §3: the correction is far outside first-order, but the map from block-output to logits is
   directionally near-linear (cos ≈ 0.85), so `J·R_v` is a good *direction* model even when magnitude is off.

## Downstream for later sections
- Section 5 (distance vs transfer defect) should use the **downstream-projected residual `‖J·R_v‖`** (or its
  ensemble variance) as the transfer-defect output variable, not the raw local norm — the raw norm would understate
  the transfer defect's explanatory power.
- Section 15 (OOD score): jproj's strong link to logit-covariance-trace suggests logit-covariance-based scores may
  track the mechanism more directly than predictive entropy.
