# Manifest — ImageNet ViT-B/16 geometry follow-up

Branch `agent/vit-geometry-scod-complete`.

## Environment

| | |
|---|---|
| interpreter | `/home/elean/pnc/.venv_vit/bin/python` (**not** the main `.venv`, whose CUDA 13 wheels have no sm_61 kernels) |
| torch / torchvision | 2.7.1+cu126 / 0.22.1+cu126, CUDA 12.6 |
| laplace-torch | 0.2.2.2 (pinned with `curvlinops-for-pytorch` 2.0.1) |
| numpy / scipy / scikit-learn | 1.26.4 / 1.17.1 / 1.9.0 |
| GPU | NVIDIA TITAN X (Pascal), cc 6.1, 12 GB |
| system RAM | 23 GiB total; operational ceiling enforced at **18 GiB RSS** |
| checkpoint | `torchvision` `ViT_B_16_Weights.IMAGENET1K_V1`, frozen |
| SCOD reference | `StanfordASL/SCOD` @ `6a569734d3e246e25c53c0dff97e4e83690087d4` |

## Code

| file | role |
|---|---|
| `fu_common.py` | shared paths, dataset lists, OOD metrics, Gaussian fit/score, member loading |
| `fu_base.py` | §1 base-entropy control (raw and at the frozen P&C temperature) |
| `fu_disagree.py` | §6–7 disagreement sets and conditional bins; one-pass mean pairwise KL |
| `fu_randproj.py` | §3 random-projection Mahalanobis control |
| `fu_spectrum.py` | §4 covariance eigenspectrum decomposition |
| `fu_align.py` | §5 P&C perturbation response projected onto the eigenbasis |
| `fu_ksweep.py` | §2 nested-orthonormal-basis rank sweep, ID-only search + final ensembles |
| `fu_iderror.py` | §8 ImageNet misclassification detection |
| `fu_llla_temp.py` | §17–18 frozen LLLA prior + ID-only temperature calibration |
| `fu_layercache.py` | §20 row-group-sorted training-pool multi-layer CLS cache |
| `fu_layerwise.py` | §21 layerwise Mahalanobis at blocks 8–11 and the final LayerNorm |
| `fu_tables.py` | §23 main diagnostic table |
| `fu_figures.py` | §24 figures |
| `scod_ll.py` | vendored SCOD sketch (`RandomSymSketch`, `GaussianSketchOp`, `Projector`), MC-Fisher path, analytic per-example gradients, `LastBlockScope` |
| `scod_exact.py` | **exact** categorical factorisation that replaced MC, with the factored sketch update |
| `scod_run.py` | driver: `validate`, `qsweep`, `nsweep`, `fit`, `fitlast`, `exactvalidate`, `exactfit` |

All under `experiments/imagenet_vit_pnc/`.

## Inputs consumed (read-only, unmodified)

| path | what |
|---|---|
| `imagenet_vit/raw/cache_{val50k,ood_*,correction,selection,temperature}.npz` | frozen CLS residual caches |
| `imagenet_vit/raw/base_val_logits.npy` | frozen base-model validation logits |
| `imagenet_vit/splits/correction_rows.npy` | frozen 32,768-image ID pool addresses |
| `imagenet_vit/metrics/temperature.json` | frozen P&C temperature T = 0.7 |
| `imagenet_vit_preservation_frontier/` | frozen P&C members and configs (r=2, λ=1000, K=20, M=20) |
| `imagenet_vit_baselines/` | frozen Mahalanobis / MSP / Energy / ReAct / KFAC headline |
| `imagenet_vit_geometry_scod_llla/predictions/geometry_scores_*.npz` | per-example scores from the previous round |
| `imagenet_vit_geometry_scod_llla/raw/cls_layers_{val50k,ood}.npz` | evaluation-side layer caches |

## Selection provenance

Every hyperparameter chosen in this directory was chosen on ID data only.

| choice | criterion | outcome |
|---|---|---|
| P&C scale r and ridge λ, per K | ID selection-pool accuracy, paired-bootstrap LCB ≥ −0.50 pp; lowest NLL among passing | r = 2.0, λ = 1000 at **every** K ∈ {5,20,40,80} |
| LLLA prior precision | **not refitted** — held at the previous round's ID-NLL choice | 1e9 |
| LLLA temperature | ID NLL on the separate 8,192-image temperature pool, after freezing the prior | T = 0.7063 |
| SCOD Fisher estimator | §14 MC convergence check on ID data | MC **rejected** (across-seed Spearman ≈0.46 at q=8); exact factorisation used |
| SCOD sketch rank | the paper's CIFAR SCOD-LL setting, subject to the RAM guard | k = 30, T = 184 — **no fallback needed** for either scope |
| SCOD calibration N | largest feasible, matching the P&C/Mahalanobis pool | 32,768 (score converged by 4,096) |
| SCOD scope | none — both scopes run and both reported | — |
| layerwise Mahalanobis layer | none — all five reported | — |
| random-projection K | none — all six reported | — |

No OOD data was used to select a scale, ridge, rank, prior, temperature, scope, layer or
estimator anywhere in this round.

## Runtimes and memory

| stage | wall | peak |
|---|---|---|
| base-entropy control (6 sets) | 4 s | — |
| disagreement pass (6 sets × 20 members) | 19 s | — |
| random-projection control (2 variants × 6 K × 20 projections) | 6.1 min | — |
| covariance eigenspectrum (768 per-mode AUROCs) | ~1 min | — |
| spectral alignment (33,192 examples × 20 members) | 5 s | — |
| K sweep — ID-only search (4 K, coarse + confirm) | ~28 min | — |
| K sweep — final ensembles (4 K × 5 seeds × M=20) | ~17 min | — |
| LLLA-Kron refit + temperature + evaluation | ~4 min | 4.42 GiB GPU |
| training-pool layer cache (32,768 images, 140 row groups) | 5.7 min | 0.46 GiB GPU |
| layerwise Mahalanobis (5 depths × 2 variants) | ~1 min | — |
| SCOD gates (MC + exact) | ~6 min | — |
| SCOD MC q-sweep (12 fits; **failed**, discarded) | ~13 min | — |
| SCOD-linear (exact) | 4.4 min | 3.43 GiB RSS / 0.83 GiB VRAM |
| SCOD-FFN (exact) | 25.1 min | **17.62 GiB RSS** / 0.83 GiB VRAM |
| SCOD calibration-size sweep (4 fits, linear) | ~5 min | 3.8 GiB RSS |

## Not produced

- **SCOD-last-block** — implemented (`scod_ll.LastBlockScope`, `scod_run --stage fitlast`)
  but **not run**. Report §11 gives three reasons: the exact route needs Aᵀω per sketch row,
  which for self-attention is a JVP (~184 forward-mode passes per example, ≈9 h); the MC
  route is the one rejected in §10 for the smaller scopes; and SCOD-FFN's *measured* 17.62
  GiB peak against the 18 GiB guard implies last-block would need the k=12/T=76 fallback,
  making it non-comparable to the two scopes reported at k=30.
- **Full-network SCOD** — 118.68 GiB sketch against an 18 GiB ceiling. Not attempted.
- **The geometry-aware perturbation basis** proposed in §15 — deliberately not implemented in
  this branch.

## Reproduce

See [`README.md`](README.md) for the ordered command list. `raw/` is git-ignored and
regenerable.
