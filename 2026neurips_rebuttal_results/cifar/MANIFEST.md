# MANIFEST — CIFAR-10 results bundle

**Bundle:** `2026neurips_rebuttal_results/cifar/` (compact, committable). **Source machine:** CIFAR machine `/home/elean/pnc`, repo **DrKwint/pnc**, branch `neurips-2026-rebuttal`, commit **854f4d8** (rebuttal artifacts git-ignored/untracked at collection time). Collected on branch **`agent/collect-cifar-results`**. Env: Python 3.12, JAX 0.9.1, Flax 0.12.3, NVIDIA RTX 5060 (8 GB), WSL2. Paper repo `EtherealEq/perturb_and_correct` **not accessible** here (see RESULTS_AUDIT.md).

## Source trees (recorded, mostly NOT copied — too large)
| tree | size | copied into bundle? |
|---|---|---|
| `results/cifar10/` (submitted per-method OpenOOD JSONs + base ckpts + inference/peak) | 4.5 G | pointers only (JSONs are small but numerous); checksums in `MANIFEST_large_artifacts.md` |
| `results/scod_cifar/` (SCOD: sketches 5G, metrics, tables, report) | 5.1 G | compact only → `scod/` (tables, metrics, report, timing; sketches referenced) |
| `results/neurips_2026_rebuttal/cifar/` (reports, sensitivity, efficiency, peak_memory, scripts) | 605 M | compact docs/CSVs → `sensitivity/`, `efficiency/`, `inference/`, `storage_memory/`, `block_scope/`, `headline/`, `configs/` |
| `results/neurips_2026_rebuttal/cifar/pnc_full_grid/` (162-run joint grid) | 69 M | compact → `joint_selection/` (tables, report, all_162_candidates.csv, selected, final_metrics) |
| `results/neurips_2026_rebuttal/cifar/sll/` (SLL, related) | 529 M | not in scope of this table set; referenced in RESULTS_AUDIT |

## Provenance (common to all CIFAR groups)
- **Checkpoints:** PreActResNet18 e300, seeds **0,1,2** used (seeds 0–6 exist). SHA-256 in `MANIFEST_large_artifacts.md` (verified == SCOD manifest: seed0 `b4c45628…`, seed1 `2e942d69…`, seed2 `9b5d9de6…`).
- **Splits:** `_split_data` seed 99, 45,000 train-fit / 5,000 val; correction subset 1024 = `RandomState(seed).choice(45000,1024,replace=False)`. See `configs/CIFAR_DATA_SPLITS.md`.
- **P&C construction seed = checkpoint seed** (nested). K=20, M=50, λ=1e-3, calib 1024. Anchor block **s3b0** (stage_idx3=stage4.0), ps25, bf0.05.
- **Temperature:** one scalar per config on ID-val, applied to all members' logits before softmax; predictive-entropy OOD score. See `configs/CIFAR_INFERENCE_PROTOCOL.md`.
- **OpenOOD v1.5:** Near {cifar100, tiny_imagenet}, Far {mnist, svhn, textures, places365}; macro-mean; no OOD in construction/selection.
- **Aggregation scripts** (copied → `scripts/`): `experiments/pnc_grid/` (joint grid), `experiments/scod_cifar/` (SCOD), plus `results/neurips_2026_rebuttal/cifar/_*.py` drivers (sensitivity, efficiency, peak_memory, de_matched_timing, swag_construction_time, revised_benchmark).

## Bundle contents
`RESULTS_AUDIT.md` (main), `STATUS.csv`, `MANIFEST.md` (this), `MANIFEST_large_artifacts.md` (checksums), and the 11 group dirs: `headline/ scod/ block_scope/ sensitivity/ joint_selection/ efficiency/ inference/ storage_memory/ configs/ scripts/ raw/`. Each group carries its own verified table + provenance doc. `raw/` holds a pointer note to the large uncopied raw trees.

## Not committed (per instructions)
Base checkpoints (44 MB × 7), SCOD sketches (~852 MB × 6), prediction Parquet caches, OOD input datasets (tiny-imagenet/dtd/places365 archives), TensorBoard/W&B. All recorded by path+size(+SHA-256 for checkpoints) in `MANIFEST_large_artifacts.md`.
