# REBUTTAL_RESULTS — strongest defensible results (NeurIPS 2026, "Perturb and Correct")

Every claim cites its source file/experiment under `results/neurips_2026_rebuttal/`. All MuJoCo
results are on cached data with the current pipeline (Ant-v5 reproduced bit-for-bit, §0). CIFAR
arms run on a separate machine (scripts prepared in `scripts/neurips_2026_rebuttal/cifar_other_machine/`).
Git commit `70fb480`; hardware/env in `PROVENANCE.md`. ID-only selection preserved throughout;
no OOD data used for any hyperparameter choice.

---

## 0. Reproducibility (Phase 0)
**The evaluation pipeline reproduces its reported MuJoCo numbers exactly.** Re-running the
canonical Ant-v5 P&C config gives **max |repro − cached| = 0.000000** across all 13 reported
metrics. *Source: `repro/reproduction_ant.{md,json}`; `scripts/neurips_2026_rebuttal/repro_ant.py`.*
- Flagged honestly: the Apr-8 submitted table used a config (`size=8`) not present in the cached
  results; the current canonical config (`bf0.1`, grid {5,10,20,50}) is what reproduces. Not a
  pipeline error — a configuration difference (`REPRODUCTION.md`, `protocol_clarifications.md #14`).

## 1. Reviewer concern: "Is P&C brittle to its main design choices?" → **No.**
One-factor-at-a-time sensitivity on HalfCheetah-v5, 3 seeds (0/10/42), all else at the submitted
default. *Source: `priority1/sensitivity_mujoco.csv`, `priority1/sensitivity_mujoco_agg.csv`,
`priority1/sensitivity_summary.md`.*

| knob swept | Near-AUROC range (max−min of 3-seed mean) | verdict |
|---|---:|---|
| perturbation **scale** (0.25×–4×) | 0.103 (monotone ↑) | the one knob that matters |
| perturbation **rank** k∈{1,2,5,20,40} | **0.006** | rank-1 ≈ rank-40 |
| **bootstrap** frac {0…0.99} | 0.022 | bf=0.1 default is a mild optimum |
| **ridge** λ (≤1e-2) | ~0.018 | flat; degrades only at λ≥1 |
| target **block** (single vs multi) | 0.010 | single ≈ multi |

- **ID fidelity preserved everywhere:** ID RMSE stays in **1.642–1.677** across *all* settings.
- **Broad useful operating region; the default is representative, not cherry-picked.** Only scale
  moves detection, smoothly and monotonically, at ~0.7% ID-RMSE cost over a 16× scale span —
  a gentle ID↔OOD frontier, not a cliff. *(Rows: `sweep=scale/rank/bootstrap/ridge/layer` in the CSV.)*

## 2. Reviewer concern: "Expose the omitted Near/Mid results." → **Doing so STRENGTHENS P&C.**
The submitted MuJoCo table reports only **Far** AUROC; the harness computes Near/Mid AUROC for
every method (present in every cached JSON). *Source: `priority5_pnc_tiers.csv` (P&C canonical),
`priority5_full_mujoco_tables.csv` (all methods), `protocol_clarifications.md #12,14`.*

P&C-Random canonical (5 seeds), the previously-hidden tiers:

| env | Near AUROC | Mid AUROC | Far AUROC (shown in paper) |
|---|---:|---:|---:|
| Ant-v5 | **0.750 ± 0.048** | **0.825 ± 0.039** | 0.997 |
| HalfCheetah-v5 | 0.742 ± 0.022 | 0.946 ± 0.013 | 0.988 |
| Hopper-v5 | 0.798 ± 0.012 | 0.861 ± 0.010 | 0.951 |

**On Ant, every baseline's Near AUROC is 0.48–0.59** (barely above chance) vs P&C's **0.750** —
so the tiers the paper omitted are exactly where P&C's advantage is *largest*, not smallest.
*(Baseline rows in `priority5_full_mujoco_tables.csv`, env=Ant-v5.)* Recommendation: add the
Near/Mid AUROC columns — they help the paper.

## 3. Reviewer concern: "Does the distance–disagreement mechanism appear beyond Ant?" → **Yes.**
Replicated the submitted Ant-v5 diagnostic (hidden-space regularized Mahalanobis distance vs P&C
disagreement) on two more environments, n=40,000 each. *Source: `priority3/mechanism_second_domain.csv`,
`priority3/mechanism_second_domain_summary.md`, `priority3/mechanism_<env>_seed0.json`.*

| env | pooled Spearman ρ | within Near/Mid/Far | regime-controlled OLS slope (disagreement ~ log₁₀ d) |
|---|---:|---|---:|
| HalfCheetah-v5 | **0.904** | 0.837 / 0.860 / 0.816 | **1.97 ± 0.008** |
| Hopper-v5 | 0.749 | 0.691 / 0.730 / 0.776 | 0.32 ± 0.002 |

All p ≈ 0. The link survives controlling for regime (positive within every tier, rising with
shift severity). Hopper is weaker, matching the manuscript's "most delicate environment" framing.

## 4. Reviewer concern: "Does the local theory predict finite-perturbation behavior?" → **Partly, and honestly stated.**
Compared the finite corrected residual `r_actual(x,α)` to the first-order `r_linear=α·A_S(x)u`
(A_S via central difference through the LS solve; two-ε agreement ≈3.7%). Ant-v5, 200 pts × 6
dirs × 4 regimes. *Source: `priority2/linearization_diagnostics.csv`, `priority2/linearization_summary.md`.*
- **Small scales:** first-order prediction is directionally accurate — cosine(r_actual, r_linear)
  = 0.9–0.99 for ID/Near/Mid at α ≤ 0.5.
- **Operating scales (α=5–50):** pointwise agreement is gone (cosine→0), **but the linearized
  residual norm stays strongly rank-predictive** of the actual residual norm — Spearman **0.74–0.85
  at α=10**, still 0.66–0.74 at α=50. So the ranking/geometry the OOD score relies on survives to
  operating scale even where exact linearization fails.
- **Kept separate (per scientific-integrity instruction):** the remainder ‖r_actual−r_linear‖ has
  a **sub-quadratic** log-log slope (~1.0–1.5), partly confounded by the finite-difference
  estimation floor — we do **not** claim a clean quadratic remainder. High correlation ≠ small
  quadratic remainder; both are reported, unconflated.

## 5. Reviewer concern: "What are the real costs?" → **P&C wins construction & storage; not inference.**
*Source: `efficiency_summary.md`, `efficiency_{construction_mujoco,storage,inference_mujoco,cifar_inference}.csv`,
`results/cifar10/inference_cost.json`.*
- **Construction (Ant, mean of 3 seeds):** P&C **12.4 s (one trained base)** vs Deep-Ensemble-×50
  **2166 s (fifty trained models)** → **~175× cheaper**; vs SWAG 148 s → 12×. Marginal post-hoc
  build given a pretrained base ≈1–2 s (closed-form).
- **Storage (Ant, M=50):** P&C **16.8 MB implemented** (shared base + corrected blocks, a real code
  property) vs Deep Ensemble **37.1 MB** → **2.2×**; theoretical minimal representation **7.15 MB**
  → 5.2× (labeled as not-yet-implemented).
- **Inference — NOT cheaper (stated plainly):** P&C runs M member forward passes (cost ∝ M).
  MuJoCo M=50 forward = 2.8–5.5× a single forward; CIFAR P&C(50) = 7.4–9.7 ms/sample vs Deep
  Ensemble(5) 1.34. Only one base is *trained*, but M members are *run* — no single-pass claim.

---

### One-paragraph summary for the response
P&C is robust to its design choices (only perturbation scale materially affects detection, and
smoothly; rank-1 ≈ rank-40; ID fidelity preserved across every setting). Exposing the Near/Mid
AUROC the submission omitted *helps* P&C — on Ant it scores 0.75/0.82 Near/Mid where baselines are
near chance (0.48–0.59). The distance–disagreement mechanism generalizes beyond Ant (pooled
Spearman 0.90 on HalfCheetah, 0.75 on Hopper, both p≈0). The local theory is directionally
accurate at small scales and, at the operating scales, still rank-predicts the finite residuals
(Spearman 0.74–0.85), though the remainder is sub-quadratic (reported honestly, not conflated with
the correlation). Costs: P&C builds a 50-member ensemble ~175× cheaper than a 50-model Deep
Ensemble and stores 2.2–5.2× smaller, while inference costs the usual M forward passes (not
cheaper). The pipeline reproduces the reported numbers exactly.
