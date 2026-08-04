# CIFAR Paper Tables (NeurIPS)

Paper-ready markdown tables for the CIFAR section of the submission. All numbers
are mean ± std over **3 seeds (0, 1, 2)** unless noted. All OOD results use
the paper-clean OpenOOD v1.5 protocol — no OOD data for tuning, model selection,
threshold, or temperature fitting. Temperature is fit on the ID validation split.

Source-of-truth data lives in `results/cifar10/` and `results/cifar100/`; these
tables are regenerated via `scripts/aggregate_ood_results.py` and
`scripts/make_ood_paper_table.py`.

---

## Table 1 — CIFAR-10 OOD detection

| Method | Train cost | N | Acc% ↑ | NLL ↓ | ECE ↓ | Near AUROC ↑ | Near FPR95 ↓ | Far AUROC ↑ | Far FPR95 ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Post-hoc OOD baselines (frozen single model)** |  |  |  |  |  |  |  |  |  |
| PreAct ResNet-18 | 1× | 3 | 95.74 ± 0.18 | 0.144 ± 0.005 | 0.010 ± 0.001 | 87.85 ± 0.08 | 66.30 ± 1.07 | 91.86 ± 1.20 | 38.30 ± 6.66 |
| MSP | 1× | 3 | 95.74 ± 0.18 | 0.144 ± 0.005 | 0.010 ± 0.001 | 87.66 ± 0.06 | 65.32 ± 1.13 | 91.53 ± 1.11 | 37.47 ± 6.21 |
| Energy | 1× | 3 | 95.74 ± 0.18 | 0.144 ± 0.005 | 0.010 ± 0.001 | 87.00 ± 0.12 | 73.20 ± 1.17 | 91.38 ± 1.51 | 44.99 ± 8.78 |
| Mahalanobis ★ | 1× | 3 | 95.74 ± 0.18 | 0.144 ± 0.005 | 0.010 ± 0.001 | 87.98 ± 0.13 | 42.63 ± 1.11 | 93.25 ± 0.32 | **21.00 ± 0.42** |
| ReAct+Energy | 1× | 3 | 95.77 ± 0.20 | 0.148 ± 0.005 | 0.012 ± 0.001 | 88.63 ± 0.18 | 59.49 ± 1.69 | 92.50 ± 1.47 | 33.55 ± 6.58 |
| **Post-hoc UQ methods (single model)** |  |  |  |  |  |  |  |  |  |
| LLLA (prior=10) | 1× | 3 | 95.77 ± 0.25 | 0.140 ± 0.006 | 0.008 ± 0.002 | 88.97 ± 0.84 | 54.12 ± 6.98 | 93.04 ± 1.17 | 28.44 ± 8.23 |
| Epinet (ps=3.0) | 1.05× | 3 | 95.76 ± 0.22 | 0.148 ± 0.006 | 0.010 ± 0.003 | 88.07 ± 0.05 | 63.37 ± 1.72 | 92.16 ± 1.25 | 35.10 ± 6.82 |
| PnC scale=25 (single-block) | 1× | 3 | 95.70 ± 0.19 | 0.141 ± 0.005 | 0.0077 ± 0.001 | 91.11 ± 0.13 | 34.99 ± 0.64 | 94.04 ± 0.86 | 21.10 ± 1.78 |
| **PnC scale=50** (single-block) | 1× | 3 | 95.59 ± 0.12 | 0.144 ± 0.006 | 0.0079 ± 0.001 | **91.51 ± 0.21** | **32.17 ± 1.23** | 94.10 ± 1.00 | 20.54 ± 2.27 |
| Multi-block PnC scale=7 | 1× | 3 | 95.09 ± 0.18 | 0.152 ± 0.004 | 0.0077 ± 0.002 | 89.96 ± 0.09 | 38.82 ± 0.99 | 92.86 ± 1.14 | 24.41 ± 1.70 |
| **Train-time UQ methods (single model)** |  |  |  |  |  |  |  |  |  |
| MC Dropout (n=32, dr=0.1) | 1× | 3 | 95.76 ± 0.14 | 0.148 ± 0.006 | 0.010 ± 0.004 | 87.25 ± 0.18 | 71.04 ± 0.85 | 91.34 ± 0.27 | 42.53 ± 1.42 |
| SWAG (n=50) | 1× | 3 | 95.37 ± 0.07 | 0.146 ± 0.004 | **0.0075 ± 0.001** | 90.03 ± 0.19 | 44.71 ± 0.97 | 94.19 ± 1.26 | 22.09 ± 6.01 |
| **Reference: 5×-cost** |  |  |  |  |  |  |  |  |  |
| Standard Ensemble (n=5) | 5× | 3 | 96.56 ± 0.06 | 0.109 ± 0.000 | 0.0046 ± 0.000 | 91.10 ± 0.04 | 40.39 ± 0.47 | 94.63 ± 0.17 | 19.49 ± 0.32 |

**Bold = best in column among 1×-training-cost methods.**
★ = paper-clean Mahalanobis variant (single penultimate layer, no OOD combiner;
Lee et al. 2018 numbers use OOD validation data and are not directly comparable).

---

## Table 2 — CIFAR-100 OOD detection (frozen CIFAR-10 hyperparameters)

CIFAR-100 is a transferability test: hyperparameters are **frozen** at their
CIFAR-10-tuned values (no CIFAR-100-specific retuning on ID or OOD data).
LLLA and multi-block PnC are omitted because their memory profile does not
fit an 8 GB GPU at 100 classes (details in `cifar100_ood_detection.md`).

| Method | Train cost | N | Acc% ↑ | NLL ↓ | ECE ↓ | Near AUROC ↑ | Near FPR95 ↓ | Far AUROC ↑ | Far FPR95 ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Post-hoc OOD baselines (frozen single model)** |  |  |  |  |  |  |  |  |  |
| PreAct ResNet-18 | 1× | 3 | 78.31 ± 0.13 | 0.865 ± 0.007 | 0.042 ± 0.002 | 80.61 ± 0.20 | 57.89 ± 0.43 | 78.64 ± 0.68 | 57.11 ± 1.19 |
| MSP | 1× | 3 | 78.30 ± 0.12 | 0.865 ± 0.007 | 0.042 ± 0.002 | 79.99 ± 0.18 | 57.78 ± 0.52 | 77.69 ± 0.73 | 57.37 ± 1.15 |
| Energy | 1× | 3 | 78.31 ± 0.13 | 0.865 ± 0.007 | 0.042 ± 0.002 | 80.44 ± 0.24 | 58.27 ± 0.34 | 78.91 ± 0.71 | 56.32 ± 1.60 |
| Mahalanobis ★ | 1× | 3 | 78.30 ± 0.12 | 0.865 ± 0.007 | 0.042 ± 0.002 | 73.60 ± 0.58 | 72.85 ± 0.37 | **82.41 ± 0.62** | **52.18 ± 0.78** |
| ReAct+Energy | 1× | 3 | 75.56 ± 0.14 | 0.949 ± 0.013 | 0.011 ± 0.001 | 78.30 ± 0.49 | 61.73 ± 0.48 | 73.86 ± 0.85 | 68.67 ± 3.29 |
| **Post-hoc UQ methods (single model)** |  |  |  |  |  |  |  |  |  |
| LLLA | 1× | — | *omitted — 10 GB GGN at K=100 exceeds VRAM* |  |  |  |  |  |  |
| Epinet (ps=3.0) | 1.05× | 3 | 78.31 ± 0.24 | 0.880 ± 0.005 | 0.040 ± 0.004 | 80.49 ± 0.26 | 58.26 ± 0.57 | 79.23 ± 0.85 | 55.41 ± 1.30 |
| PnC scale=50 (single-block) | 1× | 3 | 75.47 ± 0.31 | 0.898 ± 0.002 | **0.028 ± 0.001** | 79.51 ± 0.30 | 61.34 ± 0.93 | 75.42 ± 0.71 | 63.17 ± 3.03 |
| Multi-block PnC | 1× | — | *omitted — GPU memory fragmentation in OOD score phase* |  |  |  |  |  |  |
| **Train-time UQ methods (single model)** |  |  |  |  |  |  |  |  |  |
| MC Dropout (n=32, dr=0.1) | 1× | 3 | 78.36 ± 0.22 | 0.868 ± 0.005 | 0.042 ± 0.003 | **80.64 ± 0.31** | 58.48 ± 1.01 | 78.63 ± 1.58 | 56.14 ± 3.39 |
| SWAG (n=50) | 1× | 3 | 77.23 ± 0.28 | 0.874 ± 0.014 | 0.031 ± 0.004 | 80.32 ± 0.21 | 58.88 ± 0.21 | 81.28 ± 1.12 | 53.33 ± 2.36 |
| **Reference: 5×-cost** |  |  |  |  |  |  |  |  |  |
| Standard Ensemble (n=5) | 5× | 3 | 81.25 ± 0.13 | 0.706 ± 0.003 | 0.021 ± 0.001 | 82.62 ± 0.03 | 54.98 ± 0.11 | 80.61 ± 0.20 | 53.67 ± 0.32 |

**Bold = best in column among 1×-training-cost methods.**

Note: Under frozen CIFAR-10 hyperparameters, **Standard Ensemble (5× cost) leads
every 1×-cost method on CIFAR-100 Near-AUROC by > 2 points** — a larger gap
than on CIFAR-10. PnC scale=50 underperforms, consistent with the CIFAR-10
scale being too aggressive for 100 classes. Per-dataset retuning would likely
recover competitiveness and is left as future work.

---

## Table 3 — Inference cost (CIFAR-10, PreActResNet-18, batch_size=256)

Wall-clock per-sample latency on a single CUDA device, warm cache. Source:
`results/cifar10/inference_cost.json` (5000-sample benchmark, seed 0).

| Method | Train cost | Forward passes / sample | Warm ms / sample | Cold warmup (s) |
|--------|---:|---:|---:|---:|
| PreAct ResNet-18 / MSP / Energy | 1× | 1 | 0.62 | 3.61 |
| ReAct + Energy | 1× | 1 | 0.61 | 0.39 |
| Mahalanobis | 1× | 1 | 0.73 | 3.30 |
| Deep Ensemble n=5 | **5×** | 5 | 1.34 | 3.73 |
| Epinet n=50 | 1.05× | 50 | 1.62 | 4.95 |
| MC Dropout n=32 | 1× | 32 | 5.18 | 4.97 |
| SWAG n=50 (cached) | 1× | 50 | 7.33 | 59.42 |
| **PnC single-block s=25** | 1× | 50 | 7.42 | 5.74 |
| LLLA n=50 | 1× | 50 | 7.68 | 6.00 |
| PnC multi-block s=7 | 1× | 50 | 9.69 | 35.77 |

**Trade-off summary.** PnC matches Deep Ensemble's Near-OOD quality on CIFAR-10
at 1× training cost, but at ~5.5× inference latency (50 perturbed forward
passes vs 5 ensemble passes). PnC, SWAG, and LLLA all cluster at ≈ 7–8 ms /
sample because all use n=50 stochastic passes. For latency-sensitive deployment
the ordering favors Deep Ensemble (5-pass); for training-cost-sensitive
deployment it favors PnC (1× train, 1× model storage).

---

## Table 4 — ID UQ metrics (CIFAR-10, 3 seeds)

This is the pre-existing multi-seed UQ table; included for completeness.
See `experiments/cifar10_tuning_plan.md` for the full tuning history.

| Method | Acc% ↑ | NLL ↓ | ECE ↓ |
|--------|---:|---:|---:|
| PreAct ResNet-18 | 95.74 ± 0.18 | 0.144 ± 0.005 | 0.010 ± 0.001 |
| MC Dropout (n=32, dr=0.1) | 95.76 ± 0.14 | 0.148 ± 0.006 | 0.010 ± 0.004 |
| SWAG (n=50) | 95.37 ± 0.07 | 0.146 ± 0.004 | 0.0075 ± 0.001 |
| LLLA (prior=10, n=50) | 95.77 ± 0.25 | 0.140 ± 0.006 | 0.008 ± 0.002 |
| Epinet (ps=3.0, n=50) | 95.76 ± 0.22 | 0.148 ± 0.006 | 0.010 ± 0.003 |
| PnC scale=25 (single-block) | 95.70 ± 0.19 | 0.141 ± 0.005 | 0.0077 ± 0.001 |
| Multi-block PnC scale=7 | 95.09 ± 0.18 | 0.152 ± 0.004 | 0.0077 ± 0.002 |
| Standard Ensemble n=5 | 96.56 ± 0.06 | 0.109 ± 0.000 | 0.0046 ± 0.000 |

---

## Table 5 — Architectural generality (planned)

The WideResNet-28-10 row group is **not yet run**. The current submission
makes the architectural-generality claim only implicitly (PnC's mechanism is
architecture-agnostic). WRN-28-10 results with PnC / SWAG / Deep Ensemble /
Mahalanobis / LLLA on CIFAR-10 are listed in P4 of
`cifar_neurips_strengthening_plan.md` as recommended but optional strengthening;
they would be added here if the compute is available before the submission
deadline.

---

## Per-dataset OOD appendix

Full per-dataset breakdowns (AUROC / AUPR / FPR95 for every
{method × dataset × seed} cell) are available via
`python scripts/aggregate_ood_results.py results/cifar10` and
`python scripts/aggregate_ood_results.py results/cifar100`. They are
deliberately not inlined here — the headline claims in §3 use aggregate
near/far AUROC and FPR95; per-dataset numbers belong in the paper's appendix.
