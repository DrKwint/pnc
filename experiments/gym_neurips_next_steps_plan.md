# NeurIPS Next Steps Plan — moving from defensible to impressive

This plan picks up after `gym_neurips_readiness_plan.md` was closed (Tier 1–2 done, Tier 3 partial). The readiness pass made the paper's quantitative claims defensible. This plan is about making them *impressive*. It prioritizes four items, in the user's requested order: **(1) PJSVD + Deep Ensemble hybrid → (5) Modern baseline (evidential regression) → (4) Theory extensions with empirical backing → (3) Risk-aware MPC downstream task.** CIFAR is covered on another machine and is not in scope here.

Progress is tracked in `experiments/gym_neurips_next_steps_log.md`.

---

## Context: what the current results can and can't claim

After the readiness pass, the n=5 median-IQR table shows:

| Env | PJSVD Far NLL | DE Far NLL | PJSVD Far AUROC | DE Far AUROC |
|-----|---------------|------------|-----------------|--------------|
| Ant-v5 | **2.87** | 3.79 | **0.922** | 0.623 |
| HalfCheetah-v5 | 2.07 | **1.60** | 0.980 | 0.983 |
| Hopper-v5 | 1.18 | **1.17** | 0.873 | **0.912** |
| Humanoid-v5 | (negative result — Minari ladder is inverted on this env) | | | |

PJSVD wins Ant decisively, ties HC, loses Hopper. A careful reviewer reads that as "this method matches the strongest baseline on 2 of 3 envs, beats it on 1." The current state is defensible but not impressive.

## What a careful reviewer will still object to

1. Only one decisive win out of three environments. Contribution looks marginal on HC and Hopper.
2. All empirical results are MuJoCo dynamics prediction. No classification, no vision, no downstream task showing the uncertainty is *useful*.
3. Theory is local and all upper bounds. No statement connecting ensemble variance to distance-from-calibration, no lower bound on OOD disagreement, no rate on coverage, no theoretical justification for composing PnC with DE.
4. All baselines are 2016–2020 work.

This plan closes objections 1, 3, and 4, and takes a substantive swing at 2.

---

## Priority 1 — PJSVD + Deep Ensemble hybrid

### Hypothesis
Applying PnC perturbations to each member of a small deep ensemble strictly dominates both parent methods on Far NLL and Far AUROC across all three envs. The hybrid inherits DE's RMSE advantage (averaging over multiple trained models) and PnC's shift-aware variance (per-member safe-subspace perturbations). The law of total variance decomposes:

```
Var_hybrid(x)  =  E_i[Var_k(f_{θ_i + v_{i,k}}(x))]   +   Var_i[E_k(f_{θ_i + v_{i,k}}(x))]
              =  mean within-DE-member PnC variance  +  between-DE mean variance
```

Both terms are non-negative so `Var_hybrid(x) ≥ max(V_PnC, V_DE)` pointwise. This is the theoretical hook that motivates the experiment; empirically we want to verify the hybrid dominates *both* parents.

### Method
Let `f_{θ_1}, …, f_{θ_M}` be a Deep Ensemble of `M` independently-trained probabilistic regressors. For each member `i`:

1. Compute its own projected-residual operator `R_i = (I − P_{A^{(i)}}) J^{(i)}` on the shared calibration set.
2. Derive its safe subspace `C_k^{(i)}`.
3. Draw `K` PnC perturbations `{v_{i,1}, …, v_{i,K}}` uniformly from the unit sphere in `C_k^{(i)}` (scale `r`).
4. LS-correct each perturbed member against its own base.

Total ensemble size is `MK`; inference cost is `MK` forward passes; *training* cost is only `M` base models instead of `MK`.

### Implementation tasks

1. **`ensembles.py::HybridPnCDEEnsemble`** — accepts a list of `M` trained probabilistic base models, computes per-member safe subspace, samples `K` perturbations per member, stores `MK` corrected variants, exposes `.predict_mean_var(x)`. Reuses existing `PJSVDEnsemble` internals per member.
2. **`gym_tasks.py::GymHybridPnCDE`** — new luigi task. Parameters: `n_de` (M), `n_pjsvd_per_de` (K), perturbation scale sweep, layer scope, correction mode, calibration toggle, rest from `GymPJSVD`. Requires `CollectGymData` only; trains `M` bases internally. Output filename pattern includes `nDE{M}_nPnC{K}`.
3. **`json_to_tex_table.py::gym_friendly_name`** — recognize `hybrid_pnc_de_` stems, add row class.
4. **Table generator integration** — include in `method_order`.

### Experiments (sequential, single GPU)

Matched-cost sweep at total budget `MK = 50`:

| Config | M (DE) | K (PnC per DE) | Runs |
|--------|--------|----------------|------|
| Hybrid-2x25 | 2 | 25 | 3 envs × 5 seeds × {raw, VCal} = 30 |
| Hybrid-5x10 | 5 | 10 | 30 |
| Hybrid-10x5 | 10 | 5 | 30 |

**Total: 90 runs.** Plus 6 runs for a DE-n=50 reference (3 envs × 2 seeds — we only need 0/200 for a spot-check, since the comparison is already in the matched-n table at n=20). At ~80s–120s per run that is about 2–3 hours sequential.

**Pilot first (Day 1):** a minimal hybrid (`M=2, K=10`) on Hopper seed 0 only — ~3 min. Gate criterion: hybrid Far NLL ≤ min(PnC-only-Hopper, DE-only-Hopper) = min(1.18, 1.17) = 1.17. If the pilot meets the gate, commit to the full 90-run sweep. If it doesn't, reanalyze before spending the GPU time.

### Deliverables
- `gym_tables.txt` and `gym_tables_seeds_common5.txt` regenerated with hybrid rows.
- A new section in `gym_neurips_next_steps_log.md` decomposing each hybrid variant's variance into within-DE and between-DE components per env.
- A paper-ready paragraph summarizing the hybrid result.

### Risks
- **Null result** (hybrid = max(parents)): still acceptable, reframe paper as "PnC is a cheaper alternative to DE, and the hybrid is the best of both". Not a story *win*, but not a loss either.
- **Hybrid is worse than both parents** (extremely unlikely given T7; would imply a bug). Low risk.

### Estimated wall-clock: 3–4 days (code + runs + analysis).

---

## Priority 2 — Modern baseline: Evidential Deep Regression

### Rationale
Every current baseline (MC Dropout, DE, SWAG, Subspace, Laplace) predates 2020. A 2026 reviewer will ask about post-2020 methods. **Evidential regression** (Amini et al. NeurIPS 2020) is the right fit because it's (a) a direct regression method, (b) single-model, (c) well-known and well-critiqued, (d) has official code, (e) philosophically aligned with PnC (single model + richer output head).

### Method
Replace the `(mean, log_var)` head of `ProbabilisticRegressionModel` with a 4-dimensional `(γ, ν, α, β)` head parameterizing a Normal-Inverse-Gamma. Predictive mean = `γ`; predictive variance = `β/(α−1)`; epistemic variance = `β/(ν(α−1))`.

Loss (per output dim):
```
L_NLL = 0.5·log(π/ν) − α·log(2β(1+ν)) + (α+0.5)·log((y−γ)²·ν + 2β(1+ν)) + log(Γ(α)/Γ(α+0.5))
L_reg = |y − γ| · (2ν + α)
L    = L_NLL + λ · L_reg
```

### Implementation tasks

1. **`evidential.py::EvidentialRegressionModel`** — `flax.nnx` module with 4 output heads, softplus activations on `ν`, `α` (+1), `β`.
2. **`util.py` training loop** — add evidential path (`train_evidential_model`) or extend `train_probabilistic_model` with a flag. Gradient clipping at 1.0 (evidential losses can blow up).
3. **`gym_tasks.py::GymEvidential`** — luigi task. Parameters: `steps`, `hidden_dims`, `seed`, `policy_preset`, `posthoc_calibrate`, `lam` (λ regularizer).
4. **`json_to_tex_table.py::gym_friendly_name`** — recognize `evidential_` stem.

### Experiments
- Hyperparameter sweep on Ant seed 0: `λ ∈ {0.001, 0.01, 0.1, 1.0}`, pick by `nll_val`.
- Full runs: 3 envs × 5 seeds × {raw, VCal} = **30 runs**.

### Risks
- Evidential regression has known instability. Mitigation: grad clipping, LR warmup, fallback to λ=0.01.

### Estimated wall-clock: 2–3 days.

---

## Priority 3 — Theoretical extensions with empirical backing

### What the current theory gives us
The draft (`neurips_draft/theory.tex`, `theory_improved.tex`) proves three results:

- **T1** (Optimal suppressible subspace): `C_k` minimizes worst-case residual.
- **T2** (Local transfer): `ρ(x;v) ≤ C₀‖Rv‖_F + C₁·dist(y(x), G(Y))·‖v‖ + C₂‖v‖²`.
- **T3** (Oracle OOD direction): best safe direction is a right-singular vector of `S_U P_{C_k}`.

Plus a Coverage corollary (no rate).

### What's missing

1. **No connection between ensemble variance and distance.** T2 bounds per-direction residual, not variance. The paper's empirical hook (variance tracks shift) isn't theoretically grounded.
2. **No lower bound on OOD disagreement.** All three theorems are upper bounds.
3. **No rate on coverage.** "Monotone in M" is weaker than a concentration bound.
4. **No theoretical justification for the hybrid.** Nothing ties PnC and DE together.

### Proposed new results

#### T4 — Expected variance is linear in squared distance from geometry

**Statement.** Let `v ~ Unif(r·S(C_k))`. Under Assumptions `local_cal`, `local_x`, and stable refit:

```
E_v[ρ(x;v)²]  =  r² · (1/k) · ‖(I − P_A(x)) J_x‖²|_{C_k}  +  r² · η(x)  +  O(r³)
```

where `η(x) ≤ γ · dist(y(x), G(Y))²` for a geometry-dependent constant `γ`.

**Proof sketch.** Square T2's bound and take the expectation over `v` drawn uniformly from the unit sphere in `C_k`. Cross terms vanish by symmetry; the `‖Rv‖²` term produces the first summand (via `E[v vᵀ] = (1/k)·P_{C_k}`); the `dist·‖v‖` term produces the `η(x)` summand.

**What it buys us.** A direct link from *expected* ensemble variance to distance-from-calibration-geometry. This is the bridge from T2's per-direction bound to the actual AUROC story.

#### T5 — OOD disagreement lower bound

**Statement.** Let `J_x` be the local Jacobian at test point `x`, let `P_{C_k}` project onto `C_k`, and define the safe-subspace Jacobian projection

```
Q_k(x) := ‖J_x P_{C_k}‖_F
```

Then for `v ~ Unif(r·S(C_k))` and under mild non-degeneracy:

```
E_v[‖f_{θ+v}(x) − f_θ(x)‖²]  ≥  (r²/k) · Q_k(x)²  −  O(r⁴)
```

**Interpretation.** For any test point whose local Jacobian projects substantially onto `C_k`, the ensemble disagreement is bounded *below* by a quantity proportional to that projection. This is the complement of T2/T4 and is what turns "ID is suppressed" into "OOD is surfaced".

**AUROC corollary.** Define the shift separability
```
Δ(x) := Q_k(x)² − γ · dist(y(x), G(Y))²
```
If `E[Δ | OOD] > E[Δ | ID]`, the PnC variance ranking separates ID and OOD, giving a quantitative AUROC lower bound via a Chebyshev-style inequality on the `Δ` distributions.

#### T6 — Exponential coverage rate

**Statement.** Strengthen the existing Corollary. For `v_1, …, v_M ~ iid Unif(S(C_k))` and any `θ ∈ (0, π/2)`:

```
Pr(max_m |⟨v_m, u⟩| ≥ cos θ)  ≥  1 − exp(−M · I_k(θ))
```

where `I_k(θ)` is the spherical cap mass in dimension `k` (explicit formula from Ball 1997 or any geometry-of-high-d reference).

**Practical implication.** At `k = 20`, `θ = π/4`, `I_k(θ) ≈ 0.003`, so `M = 300` suffices for 60% coverage and `M = 1000` for 95%. Gives the paper a concrete "how many perturbations do I need" answer.

#### T7 — Hybrid variance decomposition

**Statement.** Let `{f_1, …, f_M}` be independently trained probabilistic regressors, each with its own safe subspace. Let `{v_{i,k}}` be PnC perturbations with `v_{i,k} ∈ r·S(C_k^{(i)})`. The hybrid predictive variance satisfies the law of total variance:

```
Var_hybrid(x)  =  E_i[Var_k(f_{i + v})]  +  Var_i[E_k(f_{i + v})]
              =  (mean within-model PnC variance)  +  (between-model mean variance)
```

Both summands are non-negative, so `Var_hybrid(x) ≥ max(V_PnC, V_DE)` pointwise.

**Proof.** One-line application of the law of total variance. The value is that it gives the hybrid a principled justification rather than "we tried composing them and it worked".

### Empirical validation experiments (all use existing trained models)

#### E-T4 — Expected variance ∝ squared distance
`experiments/scripts/variance_distance_decomposition.py`. For each env, method, and regime, compute per-sample ensemble variance `V(x)` and distance `dist(y(x), G(Y))`. Scatter plot `V(x)` vs `dist²(x)`. Fit linear model; report `R²` and slope. Hypothesis: PnC has high R² (variance strongly explained by distance); DE has moderate R² (different mechanism); Hybrid has R² between the two with higher absolute variance on OOD.

**Needs:** per-dim variance arrays in the sidecar npz — currently only per-sample scalars. Either (a) extend `_evaluate_gym` to dump per-sample vectors and re-run on one env, or (b) write a one-off script that re-loads the base model and computes `V(x)` from scratch. Option (b) is cheaper.

#### E-T5 — Safe-subspace Jacobian projection
`experiments/scripts/safe_subspace_jacobian_projection.py`. For Ant seed 0, compute `Q_k(x) = ‖J_x P_{C_k}‖_F` for each test sample. Plot:
- `Q_k(x)` histogram for ID vs. each OOD regime.
- Scatter `Q_k(x)` vs. observed `V(x)`. Expect positive correlation.
- Predicted AUROC from `Δ(x)` separation vs. observed AUROC from ensemble variance.

#### E-T6 — Coverage rate curve
`experiments/scripts/coverage_rate.py`. For Ant (best-PnC env), train a PnC ensemble with `M = 200` members at `k = 20`. Subsample to `M' ∈ {5, 10, 20, 50, 100, 200}` and measure Far AUROC. Plot against theoretical `1 − exp(−M' · I_k(θ))` for several `θ`. Confirm empirical curve dominates the theoretical lower bound.

#### E-T7 — Hybrid variance decomposition
Part of Priority 1. For each hybrid config, explicitly report:
- `V_within_DE` = mean over `i` of per-PnC variance at `x`
- `V_between_DE` = variance over `i` of per-PnC means at `x`
- Sum = `V_hybrid` (should match exactly — not an approximation).

Bar chart per env decomposing `V_hybrid` into the two components. Confirms the decomposition empirically and shows *which* part dominates per env (Ant: within-DE expected; Hopper: between-DE expected).

### Deliverables
- `neurips_draft/theory_improved.tex` extended with T4–T7 + proofs.
- 3 validation scripts in `experiments/scripts/`.
- 3–4 figures in `experiments/figures/`.
- Log section summarizing empirical vs. theoretical predictions.

### Estimated wall-clock: 3 days (can run in parallel with Priority 1 runs, since most of the work is non-GPU writing + small ad-hoc scripts).

---

## Priority 4 — Risk-aware MPC on Ant (downstream task)

### Rationale
Current results are metric-only. A reviewer will ask: *does the higher AUROC matter?* A downstream capability result converts "here's a better number" into "here's a useful method". Ant is the natural env because PnC's AUROC advantage there is the largest (0.92 vs DE's 0.62).

### Task
Use a learned dynamics model + a simple Cross-Entropy Method (CEM) planner on Ant-v5 with an env-parameter perturbation (e.g., +30% gravity or perturbed body mass). The planner evaluates candidate action sequences via `H`-step rollouts in the learned model, and the uncertainty estimate penalizes trajectories through high-variance regions.

Reward during planning:
```
r_plan(s, a)  =  env_reward(s, a)  −  β · ‖predicted_variance(s, a)‖
```

The env-parameter perturbation is what pushes rollouts into OOD regions of the dynamics model.

### Comparisons
1. No-uncertainty baseline (`β = 0`).
2. DE variance penalty.
3. PnC variance penalty.
4. Hybrid variance penalty (from Priority 1).

### Metrics
- **Average episode return.** Higher = better.
- **Failure rate.** Fraction of episodes where the robot falls before `t_max`.
- **Uncertainty calibration.** Correlation between predicted variance and actual one-step prediction error during rollouts.

### Implementation tasks

1. **`mpc.py`** — batched CEM planner. Takes any object with `.predict_mean_var(x)` as the simulator. Planning horizon `H`, `n_candidates`, `n_elites`, `n_iters`.
2. **`experiments/scripts/mpc_eval.py`** — loads a trained model, wraps as dynamics simulator, runs `N_episodes` of CEM-controlled Ant under a chosen shift, logs per-episode return + failure + calibration.
3. **Env-parameter perturbation** — set `env.unwrapped.model.opt.gravity[2]` after `gym.make("Ant-v5")` (mujoco bindings).
4. **Luigi integration (optional)** — wrap as `GymMPCEval` task for reproducibility.

### Experiments
- 4 methods × 3 shift severities (none, +15%, +30% gravity) × 10 episodes × 5 seeds = **600 episodes**.
- ~60s per episode (200 steps × 32 candidates × H=10 ≈ 64k dynamics passes at ~1ms each). Total ~10 GPU-hours.
- Could halve to 5 episodes × 3 seeds for a pilot.

### Pilot gate
Before running the full sweep, verify CEM with the learned dynamics model can achieve >100 mean return on *unperturbed* Ant-v5 with no uncertainty penalty. If it can't, the downstream task isn't viable with this dynamics model and we need a different controller or to skip the task.

### Deliverables
- `mpc.py`, `mpc_eval.py`.
- `experiments/mpc_downstream_log.md` — per-method per-shift return, failure, calibration metrics.
- A paper section with a 2-panel figure: (a) return vs. shift severity per method, (b) failure rate vs. shift severity per method.

### Estimated wall-clock: 5 days (scaffolding + pilot + runs + analysis).

### Risks
- **Learned dynamics model might be too inaccurate** for CEM to work even without shift. This is the largest risk. The pilot gate catches it.
- **Downstream gain might be small** even if the uncertainty is better. This would be an honest negative result for the capability claim but wouldn't hurt the main paper.

---

## Timeline

| Days | Item | Notes |
|------|------|-------|
| 1 | (1) pilot + hybrid code + theory draft start | gate decides full (1) commitment |
| 2–3 | (1) full hybrid sweep + (5) evidential code | runs in background, code work in parallel |
| 3–4 | (5) evidential sweep + (4) theorem proofs | sequential GPU |
| 4–5 | (4) empirical validation scripts E-T4, E-T5, E-T6 | non-GPU if using re-load trick |
| 5–6 | (3) MPC scaffolding + pilot | non-GPU for scaffolding, 1 GPU episode for pilot |
| 6–9 | (3) MPC full sweep | runs in background |
| 9–10 | Final writeup, figures, log closure | non-GPU |

Total wall-clock: ~9–10 working days at 1 GPU-bound job at a time.

---

## Rules of engagement

1. **Never run more than 1 GPU-bound job at a time.** Luigi tasks or batch scripts must be launched sequentially; background tasks are fine as long as only one has the GPU.
2. **Always use `.venv/bin/python`**, never bare `python` or `python3`.
3. **Progress lands in `experiments/gym_neurips_next_steps_log.md`**, not scattered across commit messages.
4. **Every pilot has a kill criterion.** If a pilot doesn't meet its gate, stop and re-plan before spending compute.
5. **No destructive git actions.** The readiness work's outputs (tables, figures, logs) are the ground truth to build on, not to overwrite.
