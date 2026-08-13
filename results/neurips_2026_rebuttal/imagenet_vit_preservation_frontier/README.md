# Preservation-frontier P&C on ImageNet-1K ViT-B/16

Follow-up to [`../imagenet_vit/`](../imagenet_vit/), re-running hyperparameter selection
around a different question:

> **How far can the perturbation go before the *corrected* model can no longer preserve
> ImageNet accuracy?**

The original protocol required the correction to beat matched uncorrected perturbations on
every deviation statistic, which only admits perturbations small enough that there is
nothing to correct. Here the uncorrected ensemble is demoted to an ablation and the scale is
chosen by the corrected model's own accuracy budget.

**The original operating point turned out to be a ridge artefact, not a scale limit.** The
first experiment fixed λ = 1e-3 while sweeping scale and selected r = 0.375. Searching ridge
at every scale, the *same* 0.25 pp accuracy criterion supports r = 1.25 (3.3× larger) and the
primary 0.50 pp budget supports **r = 2.0 (5.3× larger)**.

## Headline

| | Near AUROC | Near FPR95 | Far AUROC | Far FPR95 | ID top-1 |
|---|---|---|---|---|---|
| MSP | 73.52 | 81.84 | 86.04 | 51.74 | 81.068 |
| P&C, original protocol (r = 0.375) | 74.74 | 74.37 | 86.52 | 48.75 | 81.044 |
| **P&C PRIMARY (r = 2.0)** | **76.49** | **67.65** | **87.89** | **44.97** | 80.830 |
| matched uncorrected at r = 2.0 | 72.53 | 71.66 | 84.40 | 46.51 | 80.619 |

The corrected/uncorrected OOD gap is **+3.96 Near AUROC** at PRIMARY, where the original
protocol's gap was +0.14 — inside seed noise. The mechanism the first experiment could not
see is now unmistakable.

## The preservation ceiling

| budget | passes through | fails by | selected r | λ |
|---|---|---|---|---|
| STRICT 0.25 pp | r = 1.25 | r = 1.5 | 1.25 | 1000 |
| **PRIMARY 0.50 pp** | **r = 2.0** | **r = 2.125** | **2.0** | **1000** |
| RELAXED 1.00 pp | r = 3.0 | r = 4.0 | 3.0 | 1000 |

Boundary bisected to width 0.125 (5.9 %). At r = 4.0 nothing passes any budget.

Correction value grows with scale — ID top-1 change, P&C vs matched uncorrected:
−0.094/−0.008 pp at r = 1, −0.305/−0.358 at r = 2, −0.655/−2.238 at r = 3,
**−0.960/−10.754 at r = 4**.

## Layout

| path | contents |
|---|---|
| `RESULTS_REPORT.md` | the full report, 15 sections |
| `ID_SELECTION_RULE.md` | the selection rule, committed before any OOD work |
| `MANIFEST.md` | provenance: commits, reused artefacts, commands |
| `selection/` | `frontier_grid.csv` (88 configs), `best_lambda_by_scale.csv`, `frozen_configs.json` |
| `metrics/` | final ID (5 seeds × 3 configs), OOD, budget re-validation |
| `tables/` | Table 1 frontier · Table 2 operating points · Table 3 ablation |
| `figures/` | `preservation_frontier.png` — the three-panel frontier |
| `predictions/` | per-example ID scores per config and seed |
| `raw/` | member weights (gitignored, regenerable) |

## Provenance — read this before citing

This is a **follow-up**, motivated by analysis of the earlier experiment, which had already
inspected OOD at some scales including r = 1 and r = 2. **It is not a pristine preregistered
OOD experiment.** What holds:

- the selection rule was committed in `e25c8c9` **before** the follow-up OOD ran;
- the frozen configurations were committed in `74f6d56`, also before it;
- those configurations were determined by **ID data only** — the uncorrected ensemble,
  agreement, logit-MSE comparisons and calibration residual were all excluded from
  selection by construction;
- the budgets (0.25 / 0.50 / 1.00 pp) were predeclared and were **not** revised after seeing
  OOD, even though RELAXED scores best.

The original cleanly-selected result is preserved unchanged in `../imagenet_vit/`.

## Reuse

Splits, caches, checkpoint, temperature (T = 0.700) and the deterministic MSP / Energy /
ReAct+Energy baselines are reused byte-for-byte from `../imagenet_vit/`. Only perturbation
magnitude r and ridge λ were searched.
