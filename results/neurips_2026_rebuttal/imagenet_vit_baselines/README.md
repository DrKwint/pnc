# Matched post-hoc uncertainty baselines — ImageNet ViT-B/16

Extends the completed ViT experiments with the uncertainty/OOD baselines used elsewhere in
the P&C paper, all starting from the **same frozen checkpoint**. P&C itself is frozen at the
preservation-frontier PRIMARY result (r = 2.0, λ = 1000) and was not retuned.

**Verdict: `P&C weaker than established matched alternatives`** — beaten by Mahalanobis,
ahead of everything else tested. See [`CORE_RESULTS.md`](CORE_RESULTS.md) then
[`RESULTS_REPORT.md`](RESULTS_REPORT.md).

| Method | ID Acc | ID NLL | Near AUROC | Far AUROC | Fit | Storage | Evals/img |
|---|---|---|---|---|---|---|---|
| **Mahalanobis** | 81.068 | 0.8482 | **78.82** | **92.55** | **0.3 s** | 5.2 MiB | **1** |
| **P&C** (r=2) | 80.830 | **0.8045** | 76.49 | 87.89 | 12 s | 167 MiB | 20 |
| Laplace (KFAC) | 81.014 | 0.8446 | 74.63 | 86.56 | 1.1 s | 6.1 MiB | 20 |
| MSP | 81.068 | 0.8482 | 73.52 | 86.04 | 0 s | 0 | 1 |
| Uncorrected perturb. | 80.619 | 0.9048 | 72.53 | 84.40 | 12 s | 0 | 20 |
| ReAct + Energy | 81.068 | 0.8482 | 69.21 | 85.61 | 0.1 s | 0 | 1 |
| Energy | 81.068 | 0.8482 | 62.39 | 78.96 | 0 s | 0 | 1 |

Paired bootstrap: every difference against P&C excludes zero. Mahalanobis wins on **all
five** OOD datasets; P&C beats MSP, Energy, Laplace and its own uncorrected ablation.

## Not runnable, with measured evidence

| Method | Classification |
|---|---|
| SCOD | `SCOD_NOT_TRACTABLE_AT_VIT_SCALE` — 390 GiB sketch, ≈280 GPU-h, no categorical likelihood in the repo's implementation, fused attention has no forward-AD |
| LLLA (dense) | `MEMORY_INFEASIBLE` — 769,000² covariance = 2.37 TB |
| MC Dropout | `NOT_APPLICABLE` — all 37 dropout modules have p = 0.0 |
| SWAG · Subspace | `METHOD_REQUIRES_RETRAINING` — need an SGD trajectory |
| Epinet | `EPINET_NOT_PORTED_METHOD_CHANGE` |
| Deep Ensemble | not run by instruction |

## Layout

| path | contents |
|---|---|
| `CORE_RESULTS.md` | the §31 core comparison |
| `RESULTS_REPORT.md` | full report, 16 sections |
| `BASELINE_AUDIT.md` | per-method audit done **before** porting anything |
| `MANIFEST.md` | provenance, versions, sources, selection grids |
| `tables/` | frozen-baseline table (md/csv/tex), per-dataset, compute |
| `metrics/` | baseline results, SCOD preflight, paired bootstrap |
| `id_selection/` | Laplace prior-precision sweep + selected config |
| `predictions/` | per-example ID and OOD scores |
| `provenance/` | the 9/9 parity gate against the completed experiment |

## Fairness

Every fitted baseline selected its hyperparameters on **ID data only**
(`id_selection/*_selected.json` records `OOD data accessed before selection: NO`).
Mahalanobis has no tunable hyperparameter. The common harness reproduces the completed
experiment bit-exactly on 9/9 checks before any new method was run.
