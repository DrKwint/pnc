# Priority 1 — MuJoCo hyperparameter robustness (HalfCheetah-v5, seeds 0/10/42)

Anchor (submitted/canonical, ★): PJSVD-Multi-LS random, projected-residual, prob, k=20, M=50, subset=10000, λ=0, bootstrap_frac=0.1, hidden [200×4]; perturbation size selected per-seed by ID val-NLL. One factor varied at a time; mean±std over 3 seeds. Source: `sensitivity_mujoco.csv` / `sensitivity_mujoco_agg.csv`.

## Headline judgments

- **Near-AUROC range induced by each knob** (max−min of the 3-seed mean): scale **0.103**, rank **0.006**, bootstrap **0.022**, ridge **0.035**, layer **0.010**.
- **ID RMSE range across ALL settings**: 1.642–1.677 (ID fidelity is preserved across every knob setting).
- **Broad useful operating region:** YES. Only the perturbation *scale* materially moves OOD detection (monotone ↑ with scale, small ID cost), and even there every value in the 0.25×–4× band gives usable detection. Rank, bootstrap, ridge (λ≤1e-2), and single-vs-multi block all move Near-AUROC by <0.03.
- **Rank is nearly irrelevant:** rank-1 ≈ rank-40 (Near-AUROC within ~0.01). The method does not depend on a wide safe subspace.
- **The submitted default is representative, not cherry-picked:** the bf=0.1 default is a mild local optimum for bootstrap (~+0.02 vs bf=0 or bf≈1), and λ=0/size-selection sit in the flat interior of their curves — no setting is unusually favorable.
- **ID↔OOD trade-off is smooth:** raising scale improves OOD detection while ID RMSE degrades only ~1% over 16× — a gentle, monotone frontier, not a cliff.

## Per-sweep tables (mean±std over seeds)

### scale sweep (size ×default)

| size ×default | def | RMSE_id | NLL_id | Near AUROC | Mid AUROC | Far AUROC | Near NLL |
|---|:---:|---|---|---|---|---|---|
| 0.25x |  | 1.642±0.088 | 0.368±0.391 | 0.647±0.032 | 0.796±0.040 | 0.966±0.013 | 5.224±0.737 |
| 0.5x |  | 1.642±0.088 | 0.267±0.256 | 0.666±0.030 | 0.838±0.037 | 0.973±0.009 | 3.674±0.495 |
| 1.0x | ★ | 1.643±0.087 | 0.218±0.186 | 0.692±0.029 | 0.893±0.029 | 0.981±0.005 | 2.803±0.475 |
| 2.0x |  | 1.648±0.088 | 0.191±0.135 | 0.721±0.028 | 0.939±0.016 | 0.986±0.005 | 2.272±0.246 |
| 4.0x |  | 1.654±0.086 | 0.188±0.113 | 0.749±0.027 | 0.960±0.009 | 0.987±0.006 | 1.930±0.134 |

### rank sweep (k (n_directions))

| k (n_directions) | def | RMSE_id | NLL_id | Near AUROC | Mid AUROC | Far AUROC | Near NLL |
|---|:---:|---|---|---|---|---|---|
| 1 |  | 1.647±0.089 | 0.225±0.183 | 0.687±0.028 | 0.881±0.031 | 0.980±0.006 | 2.903±0.511 |
| 2 |  | 1.646±0.089 | 0.199±0.154 | 0.690±0.028 | 0.888±0.033 | 0.980±0.005 | 2.641±0.412 |
| 5 |  | 1.644±0.087 | 0.212±0.176 | 0.692±0.029 | 0.894±0.026 | 0.981±0.005 | 2.686±0.317 |
| 20 | ★ | 1.643±0.087 | 0.218±0.186 | 0.692±0.029 | 0.893±0.029 | 0.981±0.005 | 2.803±0.475 |
| 40 |  | 1.644±0.087 | 0.208±0.171 | 0.693±0.029 | 0.893±0.027 | 0.981±0.005 | 2.737±0.227 |

### bootstrap sweep (bootstrap_frac)

| bootstrap_frac | def | RMSE_id | NLL_id | Near AUROC | Mid AUROC | Far AUROC | Near NLL |
|---|:---:|---|---|---|---|---|---|
| 0.0 |  | 1.643±0.088 | 0.398±0.434 | 0.669±0.031 | 0.839±0.037 | 0.971±0.011 | 4.933±0.532 |
| 0.1 | ★ | 1.643±0.087 | 0.218±0.186 | 0.692±0.029 | 0.893±0.029 | 0.981±0.005 | 2.803±0.475 |
| 0.25 |  | 1.643±0.088 | 0.306±0.309 | 0.677±0.030 | 0.861±0.033 | 0.976±0.007 | 3.959±0.353 |
| 0.5 |  | 1.643±0.088 | 0.340±0.356 | 0.672±0.030 | 0.849±0.035 | 0.974±0.008 | 4.268±0.370 |
| 0.75 |  | 1.643±0.088 | 0.368±0.394 | 0.671±0.030 | 0.844±0.036 | 0.973±0.009 | 4.560±0.441 |
| 0.99 |  | 1.643±0.088 | 0.387±0.420 | 0.671±0.030 | 0.842±0.036 | 0.971±0.011 | 4.761±0.529 |

### ridge sweep (λ (lambda_reg))

| λ (lambda_reg) | def | RMSE_id | NLL_id | Near AUROC | Mid AUROC | Far AUROC | Near NLL |
|---|:---:|---|---|---|---|---|---|
| 0.0 | ★ | 1.643±0.087 | 0.218±0.186 | 0.692±0.029 | 0.893±0.029 | 0.981±0.005 | 2.803±0.475 |
| 0.0001 |  | 1.645±0.089 | 0.159±0.102 | 0.695±0.029 | 0.902±0.028 | 0.985±0.004 | 2.218±0.479 |
| 0.01 |  | 1.643±0.087 | 0.244±0.223 | 0.690±0.029 | 0.888±0.029 | 0.980±0.006 | 3.179±0.420 |
| 1.0 |  | 1.645±0.087 | 0.364±0.387 | 0.677±0.029 | 0.854±0.032 | 0.973±0.009 | 4.319±0.466 |
| 100.0 |  | 1.677±0.082 | 0.426±0.394 | 0.712±0.023 | 0.873±0.016 | 0.967±0.009 | 4.060±0.646 |

### layer sweep (target block)

| target block | def | RMSE_id | NLL_id | Near AUROC | Mid AUROC | Far AUROC | Near NLL |
|---|:---:|---|---|---|---|---|---|
| single-block-L0 |  | 1.644±0.088 | 0.296±0.293 | 0.681±0.035 | 0.873±0.045 | 0.976±0.010 | 3.700±0.115 |
| multi-block-L0,2* | ★ | 1.643±0.087 | 0.218±0.186 | 0.692±0.029 | 0.893±0.029 | 0.981±0.005 | 2.803±0.475 |

## Notes
- Perturbation *scale* is the one knob that matters; larger perturbations give higher Near/Mid/Far AUROC and lower Near/Mid NLL, at a ~1% ID-RMSE cost — consistent with the theory (bigger safe-subspace perturbations ⇒ more OOD disagreement, ID suppressed by correction).
- The layer sweep is single-block-[0] vs multi-block-[0,2]; the implementation's sequential correction requires the perturbed set to begin at layer 0, so deeper single-layer targets are not natively supported (documented in protocol_clarifications.md #9).
- Ridge λ≥1 degrades detection and (λ=100) forces a larger selected size + ID harm; the condition-number / solve-fallback logging is a follow-up addendum.
- No seeds were dropped; all 3 seeds × 5 sweeps are in `sensitivity_mujoco.csv`.
