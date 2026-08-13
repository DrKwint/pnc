# Manifest — geometry / SCOD scoping / LLLA follow-up

Branch `agent/vit-geometry-scod-llla`, repo commit at time of writing `8d0e10c`.

## Environment

| | |
|---|---|
| interpreter | `/home/elean/pnc/.venv_vit/bin/python` (**not** the main `.venv`, which ships CUDA 13 wheels with no sm_61 kernels) |
| torch / torchvision | 2.7.1+cu126 / 0.22.1+cu126, CUDA 12.6 |
| laplace-torch | 0.2.2.2 (pinned with `curvlinops-for-pytorch` 2.0.1; 2.1+ breaks the `curvlinops._base` import) |
| numpy / scipy / scikit-learn | 1.26.4 / 1.17.1 / 1.9.0 |
| GPU | NVIDIA TITAN X (Pascal), compute capability 6.1, 12 GB |
| checkpoint | `torchvision` `ViT_B_16_Weights.IMAGENET1K_V1`, frozen |

## Code

| file | role |
|---|---|
| `experiments/imagenet_vit_pnc/geom_cache.py` | multi-layer CLS cache (A1); `derived_from_residual` reconstructs h / post-GELU y / z / block-out / φ exactly from the existing block-11 residual cache, so no backbone pass is repeated for those |
| `experiments/imagenet_vit_pnc/geom_analysis.py` | Part A — M1–M4 Gaussians, P&C mechanism quantities, OOD metrics, A5 chain |
| `experiments/imagenet_vit_pnc/llla_kron.py` | Part C — official laplace-torch fit + closed-form KFAC predictive and its validation gate |
| `experiments/imagenet_vit_pnc/geom_run.py` | stage driver (`--stage llla` / `analysis` / `scod`) |

`geom_run.py --stage scod` dispatches to `experiments/imagenet_vit_pnc/scod_ll.py`, **which does
not exist** — Part B was scoped but not implemented, and the stage will raise `ImportError`. It is
left wired rather than removed so the missing piece is visible.

## Inputs consumed (read-only, unmodified)

| path | what |
|---|---|
| `results/neurips_2026_rebuttal/imagenet_vit/raw/cache_val50k.npz` | block-11 CLS residual cache, 50,000 ImageNet val |
| `results/neurips_2026_rebuttal/imagenet_vit/raw/cache_ood_*.npz` | same for the five OpenOOD sets |
| `results/neurips_2026_rebuttal/imagenet_vit/raw/cache_correction.npz` | 32,768-image ID training pool |
| `results/neurips_2026_rebuttal/imagenet_vit/splits/correction_rows.npy` | frozen ID pool row addresses |
| `results/neurips_2026_rebuttal/imagenet_vit/final/` | frozen P&C members (r=2, λ=1000, K=20, M=20) |
| `results/neurips_2026_rebuttal/imagenet_vit_baselines/` | frozen Mahalanobis headline, unchanged |

## Outputs

`sha256` truncated to 12 hex chars.

| path | MiB | sha256 |
|---|---|---|
| `metrics/geometry_analysis.json` | 0.02 | `1b6b46557437` |
| `metrics/llla_results.json` | <0.01 | `53b591b2b556` |
| `metrics/cls_layer_cache.json` | <0.01 | `cc34811b6c71` |
| `id_selection/llla-kron_selected.json` | <0.01 | `ff5a9d814046` |
| `id_selection/llla-kron_selection.csv` | <0.01 | `c115b29cb059` |
| `predictions/geometry_scores_val50k.npz` | 2.55 | `e99c12213451` |
| `predictions/geometry_scores_ssb_hard.npz` | 2.49 | `08e5cfbbb228` |
| `predictions/geometry_scores_openimage_o.npz` | 0.79 | `7d171064c0da` |
| `predictions/geometry_scores_inaturalist.npz` | 0.50 | `8738a506f563` |
| `predictions/geometry_scores_ninco.npz` | 0.30 | `2349fb9b1ed9` |
| `predictions/geometry_scores_textures.npz` | 0.26 | `27cf97a2ac5d` |
| `predictions/llla-kron_scores.npz` | 0.46 | `2acc83ecd402` |

Each `geometry_scores_*.npz` holds one float32 vector per example for all 15 scores in
report §3 (`M1_true_label`, `M2_pred_label`, `M3_unconditional`, `M4_nearest_centroid`,
`whitened_global`, `feature_norm`, `predictive_entropy`, `expected_member_entropy`,
`mutual_information`, `logit_variance`, `prob_variance`, `hidden_perturbation_change`,
`output_visible_change`, `post_correction_residual`, `ridge_leverage`). These are the inputs
A6 needs, so the disagreement analysis is now pure post-processing.

### git-ignored (regenerable)

`raw/cls_layers_*.npz`, ~2.0 GB total, rebuilt by `geom_cache.py`. Each holds
`cls_block8/9/10/11` and `cls_final_ln` as float32.

| set | images | MiB | sha256 | img/s |
|---|---|---|---|---|
| val50k | 50,000 | 732.4 | `c8262d2bc3da` | 110.8 |
| ssb_hard | 49,000 | 717.8 | `3090a99e2755` | 127.4 |
| openimage_o | 15,869 | 232.5 | `183681aaab84` | 125.3 |
| inaturalist | 10,000 | 146.5 | `9198e6006429` | 106.2 |
| ninco | 5,879 | 86.1 | `c5dab5a986e9` | 117.5 |
| textures | 5,160 | 75.6 | `25e72ad3bc5f` | 110.4 |

Peak GPU 0.46 GiB throughout. **`cls_layers_correction.npz` is absent** — the 32,768-image
training-pool pass stalled on parquet row-group thrashing (GPU at ~1 %, >20 min with no
progress) and was stopped. A8 needs it, because fitting layerwise Gaussians on the evaluation
split instead would leak; A8 is therefore not reported.

## Selection provenance

LLLA's prior precision is the only hyperparameter chosen in this directory. It was selected on
**ID selection-pool NLL** over the declared grid `1e-4 … 1e8`, auto-extended to `1e9, 1e10,
1e11` because the optimum sat on the grid boundary; selected `1e9`. `llla-kron_selected.json`
records `"OOD data accessed before selection": "NO"`. No other method's hyperparameters were
touched — P&C and Mahalanobis are consumed frozen.

## Runtimes

| stage | wall |
|---|---|
| CLS layer caches (6 sets, 135,908 images) | 19.2 min |
| LLLA-Kron fit (32,768 images) | 52.1 s, peak 4.42 GiB GPU / 1.35 GiB CPU |
| LLLA prediction (50,000 ID) | ~11 s |
| Part A scoring (all 6 sets, 15 scores) | 56 s |

## Reproduce

```bash
V=/home/elean/pnc/.venv_vit/bin/python
cd /home/elean/pnc
$V -m experiments.imagenet_vit_pnc.geom_cache
$V -m experiments.imagenet_vit_pnc.geom_run --stage llla
$V -m experiments.imagenet_vit_pnc.geom_run --stage analysis
```

## Not produced

No SCOD scores, tables or figures exist in this directory — Part B was scoped (upstream clone
audited, categorical-Fisher route established, per-scope sketch sizes computed) but never run,
so nothing downstream of it is reported. Likewise A6, A7 and A8 produce no artifacts here.
Report §8–§9 state what each would require.
