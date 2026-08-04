# Why Random-Proj PnC beats Low-Proj PnC — hypotheses and experiment plan

## Observed phenomenon

Across Ant-v5, HalfCheetah-v5, Hopper-v5, Humanoid-v5, the Random-Proj family
beats the Low-Proj family on OOD NLL, OOD AUROC, and often on ID RMSE at
matched configuration (`k=20` directions, `n=50` members, LS correction,
ReLU MLP 200-200-200-200, 5 seeds). Concretely on Ant-v5 seed 0, ps=50.0:

| Metric                | Low     | Random  |
| --------------------- | ------- | ------- |
| `var_id`              | 0.384   | 0.387   |
| `var_ood_far`         | 0.385   | 0.727   |
| `var_ratio` (ood/id)  | 1.00    | 1.88    |
| `auroc_ood_far`       | 0.503   | 0.927   |
| `nll_ood_far`         | 9.55    | 4.54    |
| `rmse_id`             | 0.543   | 0.542   |

Low's predictive variance is flat across ID→OOD, Random's expands. That is the
phenomenon to explain.

## Minimal mechanistic background

Both methods perturb hidden layers `l1, l3` (indices 0, 2 in a 4-hidden MLP)
and apply least-squares correction at the next-layer interface (`l2, l4`). They
leave `mean_layer` and `var_layer` untouched. The difference is the
direction set at each perturbed layer:

- **Low-Proj**: bottom-K eigenvectors of `J^T (I - P_{span M}) J` computed on
  ID data (`pnc.find_pnc_subspace_lanczos`). These are directions with minimal
  residual output impact after the LS correction — "maximally correctable" by
  design. σ_i = bottom eigenvalues (near zero).
- **Random-Proj**: K i.i.d. Gaussian unit vectors, per layer, per seed. σ_i = 1.
  No data dependence.

The paradox: directions that are *by design* most-correctable should produce
an ensemble where the LS correction works best, i.e. perturb-and-correct gives
small deviation from the base model on ID and still reasonable diversity OOD.
Instead, Low-Proj collapses OOD.

## Candidate hypotheses (H1–H5)

Each hypothesis makes a prediction about a measurable quantity. A hypothesis is
"strongly supported" if:
(a) the predicted direction holds at 5/5 seeds on at least 2 environments
(Ant, Hopper);
(b) the effect size is monotone with the Low↔Random swap holding all else
fixed.

### H1 — OOD variance collapse (output-space diversity)
**Claim**: Low-Proj members make nearly identical predictions at any input;
Random-Proj members produce predictions whose ensemble variance grows as the
input moves OOD. The OOD-detection gap (AUROC) is a direct consequence.

**Prediction**: `var_ratio = var_ood_far / var_id` is ≥1.5 for Random and ≤1.1
for Low across all seeds/envs. Equivalently, the per-sample predictive std is
≤ half for Low vs Random on `ood_far`, even though ID std is similar.

**Test**: Re-aggregate existing JSON results across seeds/envs; compute mean
(std, σ/√n) of `var_ratio`. No new runs needed — this pins the phenomenon.

**Caveat**: This is partly the phenomenon itself, not a mechanism. Treat H1 as
the *empirical consolidation step* that the other hypotheses must explain.

### H2 — "Flat" Low directions fail to perturb the activation, even uncorrected
**Claim**: Lanczos bottom eigenvectors have, by construction, a tiny residual
Jacobian effect at the correction interface. On ID, the perturbed
post-correction layer-2 pre-activation `z = h @ W_next + b_next` is nearly
identical to the base model. Since Low perturbations barely shift activations,
the correction doesn't need to do anything — and the final predictions
essentially ignore the perturbation.

**Prediction**:
- The *uncorrected* L2 difference at the correction-layer pre-activation
  `||(h_new @ W_next_orig + b) − (h_old @ W_next_orig + b)||` is much smaller
  for Low than Random (≥2× factor), even though the L1 hidden perturbation
  magnitude is comparable.
- Setting `correction_mode=none` (the "uncorrected_pnc" variant already logged)
  already produces low OOD diversity for Low.

**Test**:
1. Write a diagnostic script that loads a trained Ant model, builds both Low
   and Random subspaces per layer, and reports
   `||Δz||₂ = ||h(W+ΔW) W_next − h(W) W_next||` on ID and OOD-far inputs.
2. Check existing `uncorrected_pnc_seeds.log` runs — if they already ran the
   uncorrected version of Low-Proj we can use that data directly.

### H3 — σ-rescaling collapses Low to effective rank ≪ K
**Claim**: The code scales direction-coefficients by `z / σ` and then
normalises. Low's σ's span orders of magnitude (bottom eigenvalues are near
zero and cluster at different scales); after `z / σ` the smallest-σ direction
dominates the scaled vector, so after L2-normalisation each member's
*perturbation direction in weight space* effectively lies on a 1–3 dimensional
manifold instead of 20. Random's σ_i = 1 ⇒ uniform weighting ⇒ full rank.

**Prediction**: The 20×20 per-member coefficient matrix `C = z_coeffs / σ`
has participation ratio (`(Σ c_i²)² / Σ c_i⁴`) ≤ 3 for Low and ≥ 15 for Random.
Equivalently, the rank of the `(n_members × D)` stack of weight-deltas has
rapid singular-value decay for Low.

**Test**: Python script that builds both subspaces for seed 0 Ant, computes
coefficients, prints participation ratio and SVD spectrum. If H3 is true, we
also test: does setting `sigma_sq_weights=False` or using a flat `σ = 1` for
Low (but with its Lanczos directions) recover performance? That is an
intervention experiment.

### H4 — LS correction overfits Low's perturbation on ID (train-test leakage in correction)
**Claim**: Low directions are selected such that `Δh = h(W+ΔW) − h(W)` lives
in a subspace that the next-layer LS solve can *exactly* compensate on
`X_sub` (by definition of projected residual = 0). So the LS fit absorbs the
perturbation completely on ID. But on OOD, the same `W_next_new` is applied to
different activations, and the cancellation no longer holds — yet because
Low's perturbation is small, the OOD residual is small too. Net effect: no
diversity ID, no diversity OOD.

Random's perturbations have a non-trivial projected residual that is present
on both ID and OOD, producing robust diversity on both; the residual scales
sub-linearly with how OOD the input is, giving growing `var_ratio`.

**Prediction**: Measure, per member, the ID and OOD ensemble prediction
std vs base model. For Low: both near zero. For Random: small but non-zero
ID, 1.5–3× larger OOD. The LS-correction "leakage" → train residual tiny for
Low, moderate for Random.

**Test**: Run the diagnostic script to compute predictive std of each ensemble
at ID and OOD inputs, along with the L2 of `W_new - W_orig` (how much the
correction had to move). Low's correction magnitude should be small in
weight-space L2 too, because the underlying perturbation was already tiny.

### H5 — Functional (output) rank of Random is much higher than Low
**Claim**: Random's perturbations span a higher-rank subspace in
**predictive-output space** (not just weight space), giving better function
coverage. Low's members live in a 1–3 dim output subspace, so their
predictions are essentially along a single axis — the ensemble can only
disagree in one direction.

**Prediction**: Let `ΔY_i = y_i(X) − y_base(X)` for each member i, stacked
into `(n_members, n_test × n_out)`. The effective rank (participation ratio
of singular values) of `ΔY` is ≥ 10 for Random, ≤ 3 for Low on OOD-far.

**Test**: Same diagnostic script as H3/H4; add the output-delta SVD. Also
compute the inter-member cosine similarity in output-delta space — Low should
cluster near 1.0, Random near 0.

## Concrete experiment roadmap

1. **Aggregate existing JSON metrics** (fast, no GPU). Produce a table:
   mean±std of `var_ratio`, `auroc_ood_far`, `nll_ood_far`, `rmse_id` for
   Low vs Random across 5 seeds × 3 envs. Pins H1.
2. **Diagnostic script** (`experiments/scripts/random_vs_low_diagnostic.py`)
   that loads `results/<env>/data_*.npz`, trains a single probabilistic model
   per seed (cached via the training pipeline), builds Low and Random
   multi-layer PJSVD ensembles, and reports for each:
    - per-layer σ spectrum (`sigmas` of Low vs trivial 1's of Random) → H3
    - participation ratio of coefficient matrix `C = z / σ` → H3
    - hidden-activation Δh at perturbed layers, correction-layer Δz (ID, OOD) → H2
    - predictive variance and ensemble std ID vs OOD → H1, H4
    - correction magnitude `||W_new - W_orig||` → H4
    - output-delta SVD spectrum + participation ratio → H5
    - inter-member cosine similarity in Δy → H5
   Run one GPU job at a time (per user memory).
3. **Intervention**: run Low with `σ = 1` flat weights (keeping directions)
   and Random with `σ = low_sigmas` (keeping coefficients) to disentangle
   direction vs weighting as the cause. → tests H3 directly.
4. **Write-up**: `experiments/random_vs_low_evidence.md` with verdicts per
   hypothesis, figures, tables. For each hypothesis, clearly state whether
   the experimental evidence is (a) strong, (b) weak-but-suggestive, or
   (c) disproven. If disproven, re-plan.

## Caveats / what would falsify each hypothesis

- H1 is not a mechanism — if all four other hypotheses fail, H1 alone is
  just a restatement of the effect.
- H2 is falsified if Low's Δh and Δz are of the same order as Random's — that
  would mean the perturbation does reach activations but is still filtered
  out somewhere.
- H3 is falsified if the participation ratio is similar between Low and
  Random, or if the Low+σ=1 intervention does **not** recover Random-level
  performance.
- H4 is falsified if Low's correction magnitude `||W_new − W_orig||` is
  comparable to Random's — that would mean the LS solver is working hard
  on Low too.
- H5 is falsified if output-delta effective rank and inter-member cosine
  similarity are comparable between Low and Random.

## Priorities

H2 and H3 are the most mechanistically precise. H4 and H5 are consequences
of them at different abstraction levels. H1 is the empirical anchor. If H2
and H3 are both supported, we have a complete story (Low directions are flat
*and* the σ-scaling collapses them further), and H4/H5 follow.

## Post-experiment verdicts (2026-04-17)

- **H1 phenomenon confirmed at aggregate level** — but my framing of
  "members make nearly identical predictions" was wrong. Low actually has
  *larger* ID predictive std than Random on Ant (1.37 vs 0.68). The
  correct statement is: Low's predictive variance does *not expand* OOD
  the way Random's does (var_ratio 1.0 vs 1.6 on Ant; 1.6 vs 2.3 on
  Hopper). See ``random_vs_low_evidence.md`` H1.
- **H2 as written was wrong but the mechanism is right in the opposite
  direction** — Low's Δh is *larger* than Random's on ID (by 2-8×), not
  smaller. What I missed is that Low's Δh *shrinks* as inputs move OOD
  (42-80% smaller on OOD vs ID), while Random's stays flat or grows.
  Lanczos directions are ID-data-fit and lose their effect off-ID. The
  revised H2 in the evidence doc reflects this.
- **H3 (σ-rescaling) DISPROVEN by Low(σ=1) intervention** — removing
  σ-rescaling does *not* recover Random-like behaviour; Low(σ=1) has
  eff_rank_ood as bad as Low (Ant: 2.4 vs 4.7 — actually worse).
  Direction selection, not coefficient weighting, is the cause. This
  is now stated as H4 (negative control) in the evidence doc.
- **H4 (LS overfits on ID) partially supported but direction-confused** —
  the predicted failure mode (Low's ID residual ≈ 0, OOD residual grows)
  is observed, but the correction-magnitude signature is environment-
  specific (Low's ||ΔW_corr|| is 3.4× Random's on Ant l3, but 0.5× on
  Hopper l1). So "LS overfits" is real but not the cleanest framing.
- **H5 (output rank collapse) fully supported and promoted** — this is
  the strongest mechanistic finding. Low's effective output rank is
  2-6× smaller than Random's, with 2-4× higher member-pair cosine
  similarity. Now H3 in the evidence doc.

Final hypotheses in ``random_vs_low_evidence.md`` (H1-H5) differ from
this plan by promoting output-rank collapse to H3, folding the
ID-selectivity mechanism into H2 in the corrected direction, recording
the disproven σ-rescaling as H4 (explicit negative control), and adding
a new H5 that the entire OOD-NLL gap is UQ-quality-driven (ID RMSE is
within 0.4%). The original plan's H5 (σ-rescaling) is preserved here
as documentation of the hypothesis we ruled out.
