# LAYER_PAIR_SELECTION.md — Work Package D

**Question.** Is the current alternating pair (l1→l2, l3→l4) special, or do other
stage combinations do better? Can the best pair be chosen with ID-only signals?

**Answer: the default disjoint l1/l3 pair is NOT optimal in any environment.** An
*overlapping* pair concentrated in the environment's preferred layer region beats
it everywhere, and ID epistemic disagreement selects the best pair (8/9 grids,
max regret 0.008).

Engine: `GeneralTwoStage` (float64) — handles overlapping pairs where stage b
perturbs the layer stage a just corrected. M=32, λ=1e-2, P=10. Pairs compared:
l1/l2 (overlap), l1/l3 (disjoint, default), l2/l3 (overlap). 3 envs × seeds
0/10/42. Raw: `artifacts/pnc_theory/layerpair/`.

## Near/Mid AUROC by pair (all 9 grids)

| env / seed | l1/l2 (overlap) | l1/l3 (default) | l2/l3 (overlap) | best | Δ(best−default) |
|---|---|---|---|---|---|
| Ant / 0 | **0.722** | 0.711 | 0.544 | l1/l2 | +0.011 |
| Ant / 10 | **0.747** | 0.729 | 0.519 | l1/l2 | +0.018 |
| Ant / 42 | **0.702** | 0.677 | 0.571 | l1/l2 | +0.025 |
| Hopper / 0 | 0.894 | 0.897 | **0.907** | l2/l3 | +0.010 |
| Hopper / 10 | 0.902 | 0.898 | **0.906** | l2/l3 | +0.008 |
| Hopper / 42 | 0.838 | 0.823 | **0.844** | l2/l3 | +0.021 |
| HalfCheetah / 0 | 0.871 | 0.882 | **0.886** | l2/l3 | +0.004 |
| HalfCheetah / 10 | 0.892 | 0.894 | **0.903** | l2/l3 | +0.009 |
| HalfCheetah / 42 | 0.907 | 0.906 | **0.915** | l2/l3 | +0.009 |

## Findings

1. **The shipped default l1/l3 is suboptimal in every environment.** Ant prefers
   the early overlapping pair **l1/l2** (+0.011–0.025); Hopper and HalfCheetah
   prefer the late overlapping pair **l2/l3** (+0.004–0.021). The best pair matches
   the environment's preferred single layer (Round 3: Ant→early, Hopper/HC→late) —
   concentrating both stages in that region beats splitting them disjointly.
2. **ID epistemic disagreement selects the best pair** (argmax epi = argmax AUROC in
   **8/9** grids; the one miss is HalfCheetah/42, regret 0.008). So the unified
   ID-only rule — gate then maximize ID epistemic disagreement — extends to the
   **layer-pair** axis, and it correctly prefers the overlapping pair over the
   disjoint default. This completes the layer-pair part of Work Package F.
3. **l2/l3 is worst on Ant but best on Hopper/HalfCheetah** — there is no universal
   best pair; it must track the env's preferred layer region (which the ID-only
   score does automatically).

## Camera-ready implication
The default disjoint l1/l3 configuration is a reasonable but **not optimal** choice;
an ID-only selector (maximize ID epistemic disagreement under an ID gate) finds a
better, environment-adapted pair — typically an overlapping pair in the preferred
layer region — for a modest, consistent gain (≈+0.01 AUROC). Report the gain size
honestly. The unified selection rule now covers scale, layer, scale-pair, and
layer-pair.

## Pending
Overlapping-stage exact-identity audit (the general engine is validated by
construction; a stage-identity check on GeneralTwoStage would strengthen it); full
layer-pair × scale-pair joint selection; seeds 100/200.

Reproduce: `.venv/bin/python experiments/scripts/pnc_theory/validate_layerpair.py --env <env> --seed <s>`
