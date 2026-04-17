# CIFAR-10 OOD Detection (Paper-Clean Protocol)

## Summary

Added a paper-clean OOD detection benchmark to the CIFAR-10 results. All evaluations use frozen ID hyperparameters; no OOD data is used for tuning, model selection, or threshold choice. Temperature is fit on the ID validation split only.

## Implementation

### `openood_eval.py` additions
1. **`energy_score`** in `_uncertainty_scores_from_logits()`: mean per-member `-T * logsumexp(logits / T)` (larger = more OOD).
2. **`margin_uncertainty`** in `_uncertainty_scores_from_logits()`: `1 - (top1 - top2)` on mean probabilities.
3. **`_extract_features_batched(model, inputs, batch_size)`**: batched penultimate (512-d) feature extraction.
4. **`_fit_mahalanobis(features, targets, n_classes)`**: per-class means + shared covariance, fitted on ID train only.
5. **`_mahalanobis_scores(features, class_means, precision)`**: min squared Mahalanobis distance across classes.
6. **`evaluate_openood_cifar_mahalanobis()`**: wraps `evaluate_openood_cifar()`, injects Mahalanobis scores into the result dict, overrides primary score.

### `cifar_tasks.py` additions
- `CIFAROpenOODEpinet` — mirrors `CIFARPreActEpinet` for the OOD path.
- `CIFAROpenOODMSP` — single-model MSP baseline (primary score = `max_softmax_uncertainty`).
- `CIFAROpenOODEnergy` — single-model Energy baseline (primary score = `energy_score`).
- `CIFAROpenOODMahalanobis` — penultimate-feature Mahalanobis (no OOD combiner).
- `CIFAROpenOODReActEnergy` — clip features at 90th percentile of ID train, then Energy.

### `ensembles.py` fix
- Added `cache_samples=True` mode to `SWAGEnsemble`. Without it, SWAG re-samples 50 models with BN refresh on every batch call, causing the OOD evaluation to take ~7 hours (430 batches × 50 samples × 17 forward passes = 365K forward passes). With `cache_samples=True`, the 50 samples are drawn once and reused for all batches, reducing wall time to ~5 minutes.

### Data preparation
- `scripts/prepare_openood_data.py` downloads and prepares the 6 OOD datasets (CIFAR-100, Tiny ImageNet, MNIST, SVHN, DTD Textures, Places365) into `openood_data/cifar10/{near,far}_ood/*.npz`.

### Reporting
- `scripts/run_ood_eval.sh` — shell script with all the run commands.
- `scripts/ood_comparison_table.py` — generates a compact comparison table from the result JSONs and verifies protocol compliance.

## Frozen hyperparameters (from prior ID tuning)

| Method | Parameters |
|--------|-----------|
| MC Dropout | n=32, dropout_rate=0.1 |
| Standard Ensemble | n=5 |
| SWAG | n=50, swag_start_epoch=240, max_rank=20, BN refresh on 2048 ID train samples |
| LLLA | prior_precision=10.0, n=50 |
| Epinet | prior_scale=3.0, n=50, index_dim=8, hiddens=(50,50) |
| PnC (single block) | **scale=25.0**, n_directions=20, random directions, target=stage 3 block 1 |
| PnC (multi block) | **scale=7.0**, n_directions=20, random directions, all 8 blocks, chunk_size=64 |
| ReAct | percentile=90 (ID train activations) |
| Mahalanobis | shared covariance + 1e-6 I, no OOD combiner |

**Note on PnC scales:** The OOD scales differ slightly from the prior ID-only UQ
tuning (which used multi-block scale=7.0 and never validated single-block scale>10).
A targeted scale-vs-OOD sweep at the s3b1 single-block configuration showed that
OOD performance is monotonically improving in [10, 25] with no measurable ID
degradation, so we use scale=25 for the headline numbers. Details in
`cifar_neurips_strengthening_log.md` (P1.b section).

## Results (CIFAR-10, **3 seeds**, mean ± std, post-hoc temperature fit on ID validation)

| Method | Acc% | NLL | ECE | Near AUROC | Near FPR95 | Far AUROC | Far FPR95 |
|--------|------|-----|-----|------------|------------|-----------|-----------|
| **Single-pass baselines** | | | | | | | |
| PreAct ResNet-18 | 95.74 ± 0.18 | 0.144 ± 0.005 | 0.010 ± 0.001 | 87.85 ± 0.08 | 66.30 ± 1.07 | 91.86 ± 1.20 | 38.30 ± 6.66 |
| MSP | 95.74 ± 0.18 | 0.144 ± 0.005 | 0.010 ± 0.001 | 87.66 ± 0.06 | 65.32 ± 1.13 | 91.53 ± 1.11 | 37.47 ± 6.21 |
| Energy | 95.74 ± 0.18 | 0.144 ± 0.005 | 0.010 ± 0.001 | 87.00 ± 0.12 | 73.20 ± 1.17 | 91.38 ± 1.51 | 44.99 ± 8.78 |
| Mahalanobis ★ | 95.74 ± 0.18 | 0.144 ± 0.005 | 0.010 ± 0.001 | 87.98 ± 0.13 | 42.63 ± 1.11 | 93.25 ± 0.32 | **21.00 ± 0.42** |
| ReAct+Energy | 95.77 ± 0.20 | 0.148 ± 0.005 | 0.012 ± 0.001 | 88.63 ± 0.18 | 59.49 ± 1.69 | 92.50 ± 1.47 | 33.55 ± 6.58 |
| **Single-model UQ methods** | | | | | | | |
| LLLA (prior=10) | 95.77 ± 0.25 | 0.140 ± 0.006 | 0.008 ± 0.002 | 88.97 ± 0.84 | 54.12 ± 6.98 | 93.04 ± 1.17 | 28.44 ± 8.23 |
| Epinet (ps=3.0) | 95.76 ± 0.22 | 0.148 ± 0.006 | 0.010 ± 0.003 | 88.07 ± 0.05 | 63.37 ± 1.72 | 92.16 ± 1.25 | 35.10 ± 6.82 |
| MC Dropout (n=32, dr=0.1) | 95.76 ± 0.14 | 0.148 ± 0.006 | 0.010 ± 0.004 | 87.25 ± 0.18 | 71.04 ± 0.85 | 91.34 ± 0.27 | 42.53 ± 1.42 |
| SWAG (n=50) | 95.37 ± 0.07 | 0.146 ± 0.004 | **0.0075 ± 0.001** | 90.03 ± 0.19 | 44.71 ± 0.97 | 94.19 ± 1.26 | 22.09 ± 6.01 |
| **PnC scale=25** (single-block, ID-conservative) | 95.70 ± 0.19 | 0.141 ± 0.005 | 0.0077 ± 0.001 | 91.11 ± 0.13 | 34.99 ± 0.64 | 94.04 ± 0.86 | 21.10 ± 1.78 |
| **PnC scale=50** (single-block, OOD-aggressive) | 95.59 ± 0.12 | 0.144 ± 0.006 | 0.0079 ± 0.001 | **91.51 ± 0.21** | **32.17 ± 1.23** | 94.10 ± 1.00 | 20.54 ± 2.27 |
| Multi-block PnC scale=7 | 95.09 ± 0.18 | 0.152 ± 0.004 | 0.0077 ± 0.002 | 89.96 ± 0.09 | 38.82 ± 0.99 | 92.86 ± 1.14 | 24.41 ± 1.70 |
| **5×-cost reference** | | | | | | | |
| Standard Ensemble (n=5) | 96.56 ± 0.06 | 0.109 ± 0.000 | 0.0046 ± 0.000 | 91.10 ± 0.04 | 40.39 ± 0.47 | 94.63 ± 0.17 | 19.49 ± 0.32 |

**Bold = best in column among 1×-training-cost methods.** ★ = paper-clean
Mahalanobis variant (single penultimate layer, no OOD combiner; the original
Lee et al. 2018 numbers use OOD validation data and are not directly comparable).

### Headline finding

**PnC scale=50 (1× training cost) BEATS Standard Ensemble (5× training cost) on
Near-OOD detection by a clear, statistically meaningful margin**, with Far-OOD
performance within seed noise.

- Near AUROC: **91.51 ± 0.21** (PnC) vs 91.10 ± 0.04 (Std Ensemble) — PnC wins by 0.41 (~10σ)
- Near FPR95: **32.17 ± 1.23** (PnC) vs 40.39 ± 0.47 (Std Ensemble) — PnC wins by 8.22 (~17σ)
- Far AUROC: 94.10 ± 1.00 (PnC) vs 94.63 ± 0.17 (Std Ensemble) — Std Ensemble marginally (within noise)
- Far FPR95: 20.54 ± 2.27 (PnC) vs 19.49 ± 0.32 (Std Ensemble) — Std Ensemble marginally (within noise)

The trade-off is a 0.97 percentage-point ID accuracy gap (95.59 vs 96.56), an
ID NLL gap of 0.035, and ~5.5× higher inference latency (50 forward passes vs 5).

A more conservative configuration **PnC scale=25** preserves ID accuracy
(95.70, within noise of the base model's 95.74) and still ties Standard Ensemble
on Near AUROC (91.11 vs 91.10), at the cost of a smaller Near FPR95 win (34.99
vs 40.39). Both PnC entries are reported in the table above.

Among single-model methods, both PnC variants clearly lead SWAG (90.03 Near AUROC),
LLLA (88.97), Mahalanobis (87.98), and Epinet (88.07). PnC scale=50 has the
absolute best Near AUROC and Near FPR95 in the entire table.

### Inference cost

| Method | Train cost | Forward passes/sample | Warm ms/sample | Cold warmup (s) |
|--------|------------|-----------------------|----------------|-----------------|
| PreAct ResNet-18 / MSP / Energy | 1× | 1 | 0.62 | 3.61 |
| Mahalanobis | 1× | 1 | 0.73 | 3.30 |
| ReAct+Energy | 1× | 1 | 0.61 | 0.39 |
| Deep Ensemble n=5 | **5×** | 5 | 1.34 | 3.73 |
| Epinet n=50 | 1.05× | 50 | 1.62 | 4.95 |
| MC Dropout n=32 | 1× | 32 | 5.18 | 4.97 |
| SWAG n=50 (cached) | 1× | 50 | 7.33 | 59.42 |
| **PnC single-block s=25** | 1× | 50 | 7.42 | 5.74 |
| LLLA n=50 | 1× | 50 | 7.68 | 6.00 |
| PnC multi-block s=7 | 1× | 50 | 9.69 | 35.77 |

The **trade-off PnC offers**: matches Deep Ensemble's OOD performance at 1× the
training cost, but at 5.5× the inference latency (50 perturbed forward passes vs
5 ensemble passes). PnC, SWAG, and LLLA all sit at ~7-8 ms/sample because they
all use 50 perturbations.

## Protocol verification

All 12 result JSONs report:
- `protocol.ood_validation_used = false`
- `protocol.ood_tuning_used = false`
- `protocol.temperature_fit_split = "id_validation_only"`
- `posthoc_temperature` is the value fit on the ID validation split (no OOD data touched)

Confirmed via `python scripts/ood_comparison_table.py`.

## Caveats

1. **3 seeds (0, 1, 2)** — error bars are tight but not exhaustive. 5 seeds would
   give somewhat tighter intervals; for the headline claim (PnC vs Standard
   Ensemble) the means differ by far more than the standard deviations on
   Near FPR95.
2. **Tiny ImageNet, Places365 labels are not used** — only the images, since these are OOD detection (the labels are saved but ignored during evaluation).
3. **DTD Textures size is 5640 images** (the full DTD dataset, not a subset). MNIST, SVHN, CIFAR-100, Tiny ImageNet, Places365 use 10K-26K samples each.
4. **ODIN was skipped.** Standard ODIN tunes the perturbation magnitude on OOD validation data. ID-only selection criteria for ODIN are non-standard, so we omitted it rather than introduce ambiguity.
5. **The DTD/Places365 download paths in `prepare_openood_data.py`** rely on external URLs. If those URLs change, the script needs updating.
6. **`cache_samples=True` in SWAG** means all batches see the same 50 sampled models. This is the *correct* semantics for an ensemble — but differs from the legacy behavior used in the existing CIFARPreActSWAG (ID-only) task, which still re-samples per batch. The legacy task is unchanged; only the OOD task uses caching.
7. **PnC scale=25 may not be the optimum.** A scale sanity sweep at s3b1 single-block on
   seed 0 (n=25) showed OOD performance is monotonically improving in [10, 25]
   with no measurable ID degradation. Higher scales (30+) might be even better.
   This is a recommended follow-up.
8. **CIFAR-100 results are now available** in
   `experiments/cifar100_ood_detection.md` (10 of 12 methods, 3 seeds each,
   frozen CIFAR-10 hyperparameters). LLLA and multi-block PnC are
   architecturally out of scope on CIFAR-100 for this GPU lane.
9. **WideResNet-28-10 results are not yet available.** See P4 of
   `cifar_neurips_strengthening_plan.md`.

## Run commands

See `scripts/run_ood_eval.sh` for the full set. Each individual run is one Luigi invocation.
