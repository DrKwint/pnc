# MULTILAYER_THEORY_VALIDATION.md — Round H (theory validation)

Validation of the finite-scale multi-layer P&C mathematics on the headline
two-stage MuJoCo config (perturb l1→correct l2; perturb l3→correct l4).
Engine: `experiments/scripts/pnc_theory/multilayer.py` (float64), validated
F_ab(0,0)=F_0 to ≤2e-13 at λ=1. Data: Ant/Hopper/HalfCheetah seed 0.

## Claim status

| Claim (multi-layer) | Status | Evidence |
|---|---|---|
| **H.2 Stage-local exact identity under cumulative upstream perturbation** | **CONFIRMED** | Eq2/4 at stage b (l3→l4, carries stage-a perturbation): 1e-13–1e-12 rel-err, ID & Far, all envs |
| **H.6 Re-repair is first-order** `‖Q_{b←a}(t·v)‖∝t` | **CONFIRMED** | log-log slope 1.00–1.01, all envs |
| **H.6 Mixed interaction is second-order** `‖I_ab(t·v,s·w)‖∝ts` | **CONFIRMED** | log-log slope 2.03–2.06, all envs |
| F_ab(0,0)=F_0 (zero-perturbation ⇒ identity correction) | **CONFIRMED at λ>0**; **held-out gen-gap at λ=0** | ≤2e-13 (λ=1); 3.8e-3 gen-gap (λ=0, Ant) — the min-norm correction does not reproduce Θ off the calibration row space |

## Findings

1. **The exact transfer-defect identity is a genuine multi-layer statement.** It
   holds stage-by-stage even though stage b's design `X_b^v` is built on the
   already perturbed-and-corrected stage-a activations — the cumulative upstream
   change does not break the per-stage algebra (rel-err ~1e-13). So the exact
   finite-scale account extends verbatim to the deployed two-stage system.
2. **The infinitesimal picture is clean and matches theory** — re-repair grows
   linearly and the cross-stage interaction grows bilinearly in the small-scale
   limit. This is the correct limiting behavior; it is a *mathematical* validation
   and (per 1.6) must not be extrapolated to the benchmark scale, where the
   interaction is O(1) (see `MULTILAYER_INTERACTIONS.md`).
3. **λ=0 non-identifiability recurs at the network level.** With the default
   pseudoinverse, even the zero-perturbation two-stage forward does not reproduce
   the base network on held-out inputs (3.8e-3 gap on Ant) — the min-norm
   correction generalizes the identity imperfectly off the calibration row space.
   A small ridge floor (λ≥1e-2) removes this (gap ≤2e-13) and makes every
   multi-layer quantity reproducible and solver-independent. **Use λ≥1e-2 for all
   multi-layer measurements.**

## Answers to multi-layer synthesis questions (partial)
- **Q1 (identity exact under cumulative upstream?)** — Yes, ≤1e-12, all stages/regimes.
- Remaining (re-repair asymmetry, additive vs interaction, conditioning-driven
  interactions, incremental targeting, selection) in `MULTILAYER_INTERACTIONS.md`.

Reproduce: `.venv/bin/python experiments/scripts/pnc_theory/validate_multilayer.py --env <env> --seed 0 --lam 0.01`
