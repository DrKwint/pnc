# raw/ — self-contained raw result files

This directory now contains the **actual raw result files** (not just pointers), so the whole
`2026neurips_rebuttal_results/cifar/` bundle is self-contained and resolves standalone after a
manual transfer. Contents (result data only — model weights / sketches / datasets are excluded, see
below):

| subdir | what | size |
|---|---|---|
| `cifar10_openood/` | all submitted per-method OpenOOD JSONs (12 methods × 3 seeds) + full original P&C sensitivity JSONs (ps/bf/λ/K sweeps) + `inference_cost*.json` | 1.3 MB |
| `pnc_full_grid/` | complete 162-run joint grid (candidate `metrics.json`, tables, selected/, GRID_SPEC, reports, per-seed OpenOOD) — minus per-candidate `val_predictions.npz` | 4.3 MB |
| `rebuttal_cifar/` | all rebuttal reports (`CIFAR_*.md`), raw sweep JSON/CSV, sensitivity, efficiency, peak_memory, `sweeps/`, and driver/aggregation `_*.py` scripts | 5.2 MB |
| `scod/` | SCOD metrics, tables, spectra, timing/profile, plots, MANIFEST, report — minus 5 GB sketches and 94 MB per-example parquet | 0.7 MB |
| `sll/` | SLL (related comparator) metrics, tables, report, validation gates — minus large posterior `.npz`/parquet | 0.2 MB |

## Excluded large blobs (NOT in bundle — recorded for optional retrieval)
Model weights / intermediates / datasets, not "results." On the source machine `/home/elean/pnc/`
(checksums for checkpoints in `../MANIFEST_large_artifacts.md`):

| blob | path | size |
|---|---|---|
| Base checkpoints (7 seeds; 3 used) | `results/cifar10/preact_resnet18_train_e300_..._seed{0..6}.pkl` | 42.7 MiB each (~4.5 G total) |
| SCOD Fisher sketches (6) | `results/scod_cifar/sketches/*.npz` | ~852 MiB each (5 G) |
| SCOD per-example predictions | `results/scod_cifar/predictions/**/*.parquet` | 94 MB |
| Joint-grid per-candidate val logits | `results/neurips_2026_rebuttal/cifar/pnc_full_grid/candidates/**/val_predictions.npz` | 63.5 MB |
| SLL posterior eigensystems | `results/neurips_2026_rebuttal/cifar/sll/posterior/*.npz` | (part of 529 M) |
| OOD input datasets | `openood_data/_cache/{tiny-imagenet-200.zip,dtd.tar.gz,places365_val_256.tar}`, `cifar10_persist/cifar-10-binary.tar.gz` | ~1.5 G |

**To make the bundle reproducible-from-scratch**, add only the 3 used base checkpoints (seeds 0,1,2 =
~128 MiB): `results/cifar10/preact_resnet18_train_e300_..._seed{0,1,2}.pkl`. Everything numeric in
the reports/tables recomputes from the JSON/CSV already in this bundle.
