# LAYER_SELECTION.md — Round 3 (ID-only layer selection)

**Question (amendment D).** Can the P&C perturbation layer be chosen using ID-only
signals, matching an OOD oracle?

**Answer: Yes — ID epistemic disagreement (L3) selects the OOD-AUROC-best layer
with near-zero regret across all environments and seeds, adapting to the
env-dependent best layer.** The transfer-amplification and max-safe-P scores fail
(biased to early layers); effective rank is anti-correlated.

Setup: candidate single perturb-correct layers l1→l2, l2→l3, l3→l4; float64
`SingleLayerPnC` engine (shipped forward only supports the first layer);
M=24, λ=1e-2, ID gate |ΔRMSE|≤0.05; seeds 0/10/42 × 3 envs. Utility = mean
Near/Mid AUROC (feasible). Raw: `artifacts/pnc_theory/round3/`.

## The best layer is environment-dependent — and L3 tracks it

| env | best AUROC layer (oracle) | L3(epi) top-1 pick | AUROC regret |
|---|---|---|---|
| Ant-v5 (s0/10/42) | l1 / l2 / l1 | l1 / l1 / l1 | 0.000 / 0.017 / 0.000 |
| Hopper-v5 (s0/10/42) | l3 / l3 / l3 | l3 / l3 / l3 | 0.000 / 0.000 / 0.000 |
| HalfCheetah-v5 (s0/10/42) | l3 / l3 / l3 | l3 / l2 / l2 | 0.000 / 0.007 / 0.003 |

**L3 mean AUROC regret ≈ 0.003, max 0.017** over 9 combos — effectively
oracle-matching. Crucially the oracle layer differs by environment (Ant favors the
**early** layer, Hopper/HalfCheetah the **late** l3), and L3 adapts correctly with
no per-env tuning.

## Score comparison (AUROC top-1 regret, mean over 9 combos)

| score | what it is | mean regret | verdict |
|---|---|---|---|
| **L3 = max safe ID epistemic disagreement** | Round-6's winning signal | **≈0.003** | **selects oracle layer** |
| L4 = safe transfer amplification E‖Δh‖/E‖r‖ | | ~0.03 | biased to l1 (early), fails when late layers win |
| L1 = max safe P | | n/a | non-discriminating (all layers feasible at P=50) |
| NC srank (effective rank) | negative control | ~0.09 | **anti-correlated — picks the WORST layer on Ant & Hopper** |
| NC best ID-NLL | negative control | — | unreliable |

## Findings

1. **One ID-only signal selects both knobs.** ID epistemic disagreement chooses
   the best perturbation *scale* (Round 6) **and** the best perturbation *layer*
   (here) — "gate on ID preservation, then maximize ID epistemic disagreement" is
   a single rule for scale and layer, with near-zero OOD regret.
2. **No universal best layer.** Ant detects OOD best when perturbing an early
   hidden layer; Hopper/HalfCheetah prefer the last hidden layer. So a fixed
   "always use layer X" rule is wrong; an adaptive ID-only score is needed — and
   L3 supplies it.
3. **Effective rank fails again (negative control).** Higher member effective rank
   picks the *worst* AUROC layer on Ant and Hopper — a third independent
   confirmation (Round 6, VERDICT_step0, here) that diversity/rank is
   anti-correlated with OOD utility and must not drive selection.
4. **Utility-dependence persists.** For Far-NLL the oracle layer sometimes differs
   from the AUROC oracle (e.g. Ant l2 has the best Far-NLL while l1 has the best
   AUROC); L3 tracks the AUROC oracle. A camera-ready claim should state the
   utility it targets.

## Camera-ready implication
The P&C perturbation layer can be chosen **without OOD data** by maximizing ID
epistemic disagreement under an ID-preservation gate; this matches the AUROC
oracle across environments and adapts to the env-specific best layer. Do not use
effective-rank/diversity to choose the layer (anti-correlated).

## Pending
Layer-pair coverage (H.8: l1→l2,l2→l3 overlapping etc.); does the headline
two-layer l1/l3 config beat the best single layer (H.12/D.3); leave-one-env-out
threshold freezing; multi-layer normalized-budget selection.

Reproduce: `.venv/bin/python experiments/scripts/pnc_theory/validate_round3_layer.py --env <env> --seed <s>`
