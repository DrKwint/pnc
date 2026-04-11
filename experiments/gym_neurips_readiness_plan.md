# Gym NeurIPS Readiness Plan

## Purpose

The current gym dynamics results (PJSVD wins Far NLL across HC/Hopper/Ant) have a real qualitative story, but a careful NeurIPS reviewer will reject the quantitative comparison for several reasons. This plan enumerates each concern, the concrete fix, and the experiments needed to close it. The plan is organized so the highest-impact / most-likely-to-be-rejected issues are tackled first.

## Status of current results (2026-04-09)

- 3 envs (HC, Hopper, Ant), 3 seeds (0, 10, 200), 6 methods (Deep Ens, MC Dropout, SWAG, Subspace, Laplace, PJSVD-multi).
- All methods now use a probabilistic dual-head base model (mean + variance, Gaussian NLL training).
- Both raw and `--posthoc-calibrate` (VCal) variants run for every (env, seed, method) combination.
- Best PJSVD config: multi-layer (perturb 0,2 / correct 1,3), random projection, K=20, n=50, scale=20.

The headline numbers from `gym_tables.txt`:

| Env | PJSVD Far NLL | Best baseline Far NLL | PJSVD Far AUROC |
|-----|---------------|----------------------|-----------------|
| Ant | **0.20** | 2.71 (SWAG) | **0.899** |
| HC  | **2.82** | 3.51 (Subspace) | **0.9995** |
| Hopper | **0.36** | 7.69 (DE)  | **0.958** |

But several issues prevent these from being publishable as-is. The rest of this document describes what needs to change.

---

## Tier 1 — Blockers (will get the paper rejected)

### 1.1 Fix the Deep Ensemble ensemble-size mismatch

**Problem:** Deep Ensemble uses 5 members; PJSVD uses 50. The variance estimate from 5 samples is fundamentally noisier than from 50. A reviewer will immediately ask "what does Deep Ensemble look like at n=50, or PJSVD at n=5?"

**Fix:** Report two comparisons:
1. **Matched ensemble size**: PJSVD at n=5, DE at n=5 (current default).
2. **Matched inference cost**: PJSVD at n=50, DE at n=50 (i.e., a "Deep Ensemble" of 50 independently trained models). Acknowledge this is computationally expensive but it's the strongest possible baseline.

Optionally, also show PJSVD at n∈{5,10,20,50,100} as a sample-size sweep, plotted alongside DE at the same n values. Inference cost on the x-axis, Far NLL on the y-axis.

**New experiments needed:**

```bash
# (a) PJSVD at n=5
for SEED in 0 10 200; do
  for ENV in HalfCheetah-v5 Hopper-v5 Ant-v5; do
    .venv/bin/python -m luigi --module gym_tasks GymPJSVD \
      --env $ENV --steps 10000 --hidden-dims '[200,200,200,200]' \
      --n-directions 20 --n-perturbations 5 \
      --perturbation-sizes '[20.0]' \
      --layer-scope multi --correction-mode least_squares \
      --pjsvd-family random --safe-subspace-backend projected_residual \
      --subset-size 4096 --probabilistic-base-model \
      --seed $SEED --local-scheduler
  done
done

# (b) Deep Ensemble at n=50 (very expensive — 50 model trainings per seed)
for SEED in 0 10 200; do
  for ENV in HalfCheetah-v5 Hopper-v5 Ant-v5; do
    .venv/bin/python -m luigi --module gym_tasks GymStandardEnsemble \
      --env $ENV --steps 10000 --n-baseline 50 \
      --hidden-dims '[200,200,200,200]' \
      --seed $SEED --local-scheduler
  done
done

# (c) Sample-size sweep for PJSVD: n in {5, 10, 20, 50, 100}
for N in 5 10 20 50 100; do
  for SEED in 0 10 200; do
    for ENV in HalfCheetah-v5 Hopper-v5 Ant-v5; do
      .venv/bin/python -m luigi --module gym_tasks GymPJSVD \
        --env $ENV --steps 10000 --hidden-dims '[200,200,200,200]' \
        --n-directions 20 --n-perturbations $N \
        --perturbation-sizes '[20.0]' \
        --layer-scope multi --correction-mode least_squares \
        --pjsvd-family random --safe-subspace-backend projected_residual \
        --subset-size 4096 --probabilistic-base-model \
        --seed $SEED --local-scheduler
    done
  done
done
```

**Output:** A new sub-table or figure in `gym_tables.txt` showing the matched-n and matched-cost comparisons.

### 1.2 Fix the test leakage in hyperparameter selection

**Problem:** The current protocol selects PJSVD's perturbation scale by best `nll_id`, where `nll_id` is computed on `id_eval` (the evaluation split, which we report as the test set). This is hyperparameter selection on test data — a leak that reviewers will flag.

**Fix:** Split `id_train` into `id_train_actual` (90%) and `id_val` (10%), already partially done by `_split_data`. Compute a `nll_val` metric for every method/config and add it to the JSON output. Use `nll_val` for all hyperparameter selection (PJSVD scale, Laplace prior, K, etc.). Re-generate `gym_tables.txt` with the val-selected configs and report only those on `id_eval`.

**Code changes:**
1. In `_evaluate_gym` (`util.py`), add a `validation_data=(x_va, y_va)` argument and compute `nll_val`, `rmse_val`, `ece_val` on it. Add these to the metrics dict.
2. In `json_to_tex_table.py`, change `selection_metric` from `nll_id` to `nll_val` (and similarly for the RMSE gate).
3. Re-run `report.py` and `json_to_tex_table.py`.

**No new experiments needed** if validation metrics can be computed from the existing sidecar `.npz` files (they currently store `pred_var_id`, `sq_error_id` etc.; we'd need to also store the same on the val split). Otherwise, re-run all methods with the updated `_evaluate_gym`.

### 1.3 Investigate and fix the Hopper seed-10 training instability

**Problem:** The experiment log already documents that Hopper seed 10 has anomalously high RMSE (0.448 vs ~0.13 for seeds 0, 200) because the probabilistic base model converges to a suboptimal solution. This is training instability, not noise. With n=3 seeds and one bad run, the means/stds are unreliable.

**Fix path A — diagnose the root cause:**
1. Train Hopper seed 10 with `train_probabilistic_model` and log the loss curve. Compare to seeds 0, 200.
2. Try fixes: gradient clipping (`optax.clip_by_global_norm(1.0)`), LR warmup (linear warmup over first 500 steps), longer training (10k steps), smaller initial LR (5e-4).
3. Report the loss curves to verify the suboptimal convergence is fixed.

**Fix path B — use more seeds:**
1. Run 5–10 seeds and report **median** and **IQR** instead of mean ± std. With n=10, one outlier doesn't dominate the statistics.

**Recommended:** Do both. Diagnose the instability AND run more seeds.

```bash
for SEED in 0 10 42 100 200 314 500 777 1000 2000; do
  for ENV in HalfCheetah-v5 Hopper-v5 Ant-v5; do
    # Re-run all methods at these seeds (or at least PJSVD-multi + DE + SWAG)
  done
done
```

### 1.4 Investigate MC Dropout HC non-monotone NLL

**Problem:** HC MC Dropout shows ID NLL -2.10, Near NLL **10.07**, Mid NLL 6.02, Far NLL 3.71. NLL goes UP at Near and DOWN at Far. Variance should grow monotonically with shift; this is not monotone, which suggests a calibration bug or an OOD-construction artifact.

**Fix:**
1. Print per-input NLL distributions for ID, Near, Mid, Far for HC MC Dropout. Identify whether the high Near NLL comes from a few outliers or the whole distribution.
2. Check the variance: if Near has lower variance than Mid/Far, that's the bug — variance head is mispredicting Near.
3. Check whether MC Dropout's variance head is essentially constant across all inputs (model collapse).
4. Compare with PJSVD's per-input variance distribution to see if PJSVD is monotone where MC Dropout is not.

If the bug is in the variance head training, increase MC Dropout training steps or reduce dropout rate (currently 0.1). If the bug is in OOD construction, see Issue 1.5.

### 1.5 Document and validate the OOD construction

**Problem:** "Near/Mid/Far" come from `OODPolicyWrapper` and `id_policy_random`. Reviewers don't know what these mean, can't reproduce them, and can't verify they form a monotone shift sequence.

**Fix:**
1. Read `data.py` and write a clear textual description of how each OOD regime is constructed (action perturbation magnitude? state noise? different policies?).
2. Add this description to the paper's experimental section and to the experiment log.
3. **Validate monotonicity**: compute a distance metric between ID and each OOD distribution (e.g., MMD on inputs, Wasserstein on actions, KL on next-state distributions) and verify Near < Mid < Far in distribution distance.
4. If the OOD construction is non-standard, also show results on a standard OOD setup (e.g., perturbed mass/friction in MuJoCo) for at least one environment.

---

## Tier 2 — Important (will weaken the paper)

### 2.1 ID RMSE gap on HC and Hopper

**Problem:** PJSVD's ID RMSE on HC (0.290) is 2.1× worse than MC Dropout (0.137); on Hopper (0.132) it's 1.4× worse than DE (0.092). This violates the plan's "5% RMSE gate" criterion. The defense ("the probabilistic base model has higher RMSE") is weak because Subspace/SWAG/Laplace use the same base model and have different RMSEs.

**Investigation:**
1. Train the probabilistic base model alone (no PJSVD perturbation). Record its ID RMSE. This is the true "5% gate" baseline — not the deterministic regressor.
2. Compare to PJSVD at scale=0 (which should equal the base model). If PJSVD scale=0 ≠ base model RMSE, something is off in the inference path.
3. Sweep PJSVD scale ∈ {0, 0.1, 0.5, 1, 5, 10, 20} and plot RMSE vs scale. Find the largest scale that satisfies RMSE-within-5%-of-base-model.

**Fix:** Either find a smaller scale that meets the gate, or change the gate criterion to "within 5% of the matched probabilistic base model" and document this clearly.

### 2.2 Add Humanoid-v5

**Problem:** Humanoid was in the original plan and never run. 3 envs is the bare minimum; 4 is more credible.

**Fix:** Run all methods (raw + VCal) on Humanoid-v5 with seeds 0, 10, 200. Humanoid has 393-dim input and 376-dim output, so it stress-tests the architecture.

```bash
for SEED in 0 10 200; do
  for METHOD in GymStandardEnsemble GymMCDropout GymSWAG GymSubspaceInference GymLaplace GymPJSVD; do
    # Add the appropriate flags per method
    .venv/bin/python -m luigi --module gym_tasks $METHOD \
      --env Humanoid-v5 --steps 10000 --hidden-dims '[200,200,200,200]' \
      --seed $SEED --local-scheduler
  done
done
```

Watch out: Humanoid may need a larger architecture (the plan mentions ~274K params at 4×200) and longer training.

### 2.3 Add a no-correction ablation

**Problem:** Reviewers will ask "what does the LS correction actually contribute? How is this different from random weight noise?" The current results don't isolate the correction.

**Fix:** Add an ablation that runs PJSVD with the LS correction disabled (just random weight perturbation, no correction). The same `--correction-mode` parameter exists but currently only allows `affine` or `least_squares`. Add a `--correction-mode none` option that skips correction entirely, then compare:

| Config | ID RMSE | Far NLL | Far AUROC |
|--------|---------|---------|-----------|
| Random weight noise (no correction) | ? | ? | ? |
| PJSVD with affine correction | ? | ? | ? |
| PJSVD with LS correction | ? | ? | ? |

Expected: LS correction should preserve ID RMSE much better than no correction, while still producing OOD diversity. If no-correction is competitive, the method's novelty is in question.

**Code change:** Add `none` to `correction_mode` choices in `GymPJSVD`. In `PJSVDEnsemble._precompute_least_squares`, if mode is `none`, skip the correction step entirely (use the perturbed weights as-is).

### 2.4 Explain and possibly fix the VCal-hurts-baselines result

**Problem:** VCal makes Far NLL **worse** for most baselines (Subspace HC 3.51 → 12.56, SWAG Hopper 19.22 → 43.37, MC Hopper 22.46 → 106.06). This is the opposite of what VCal is supposed to do. Reviewers will think VCal is implemented wrong.

**Investigation:**
1. Print the learned `posthoc_variance_scale` for each method/env. If it's > 1, the val-fit is amplifying the variance, which makes ID NLL better (calibrated) but worsens OOD where variance was already too small.
2. Verify the implementation matches a standard reference (e.g., Kuleshov et al. 2018).

**Fix options:**
- If the implementation is correct, **explain it explicitly** in the paper: "VCal is fit on val to minimize ID NLL; this fit can amplify variance in a way that hurts OOD when the variance was already underestimated everywhere." This is a real finding.
- Try **per-dimension VCal** (Hypothesis 9 in `gym_tuning_plan.md`): fit a scale per output dimension instead of one global scale.
- Try **temperature scaling** instead of variance scaling (analogous to classification).

### 2.5 Report median + IQR with more seeds

**Problem:** With n=3 seeds, "mean ± std" is misleading — the std is itself estimated from 3 numbers. One outlier dominates.

**Fix:**
1. Run 5–10 seeds total (combine with Tier 1.3).
2. Change `report.py` to compute and display median and IQR (25th–75th percentile) instead of mean ± std.
3. Optionally also show min/max as error bars in plots.

---

## Tier 3 — Polish (paper looks more thorough)

### 3.1 Compute matched-cost Pareto frontier

For each method, compute `inference_FLOPs_per_prediction × n_members` and plot Far NLL vs inference cost. PJSVD at n=5, n=10, n=50 vs DE at n=5, n=10, n=50 all on the same axes. This is the cleanest possible comparison.

### 3.2 Add a second architecture

Run PJSVD + best baseline (DE) at 2×100 (smaller) and 6×400 (larger) on at least one environment. Show the method works at multiple scales.

### 3.3 Unify the K and scale ablations into a single figure

Currently the K sweep and Lanczos vs random ablations are separate tables in the experiment log. For the paper, fold them into a single ablation figure: bars for {K=20, K=80} × {Lanczos, Random} × {single-layer, multi-layer}, showing the contribution of each design choice.

### 3.4 Add a per-dimension variance analysis

For PJSVD vs the best baseline on one environment, plot per-output-dimension variance under ID, Near, Mid, Far. If PJSVD's variance scales differently per dimension while baselines don't, that's a publishable insight about the method's behavior.

### 3.5 Real-world or large-scale dataset

If time permits: run PJSVD on a CIFAR-10 / CIFAR-100 image classification task with corruption shift (CIFAR-10-C). The method already has CIFAR support per the codebase. This dramatically broadens the empirical claim.

---

## Order of operations

If executed sequentially:

1. **Tier 1.2** (test leakage fix) — code change, then re-generate tables. Cheapest, highest priority.
2. **Tier 1.3** (Hopper instability) — diagnose first, then either fix or run more seeds.
3. **Tier 1.4** (MC Dropout non-monotone bug) — diagnose first; the fix may be small.
4. **Tier 1.5** (OOD documentation) — read code, write description, validate monotonicity.
5. **Tier 1.1** (DE n-mismatch) — run new experiments. Most expensive but unavoidable.
6. **Tier 2.1** (RMSE gap investigation) — small ablation.
7. **Tier 2.3** (no-correction ablation) — small code change + new runs.
8. **Tier 2.2** (Humanoid) — runs.
9. **Tier 2.4** (VCal explanation) — investigation + writing.
10. **Tier 2.5** (more seeds + median/IQR) — runs + small code change.
11. **Tier 3** items as time permits.

---

## What success looks like

After this plan is executed:

- **Tables are reproducible from scratch** with documented OOD construction and val-based hyperparameter selection.
- **DE comparison is fair** at multiple ensemble sizes; PJSVD still wins at matched-n or matched-cost (the qualitative story holds).
- **Hopper seed 10 anomaly is gone** (training fix) or absorbed into a 10-seed median (statistical fix).
- **MC Dropout HC NLL is monotone** under shift, or the non-monotonicity is explained.
- **The 5% RMSE gate is satisfied** with the corrected baseline (probabilistic base model RMSE), or the gate is replaced with a more honest criterion.
- **The no-correction ablation shows LS correction is essential**, justifying the method's novelty.
- **4+ environments**, 5+ seeds, median + IQR.

If any of these fail — for example, if PJSVD does NOT win at matched-n vs DE — the paper's claims need to be scaled back accordingly. Better to know that now than after a desk reject.
