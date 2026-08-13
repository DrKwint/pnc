# ImageNet ViT-B/16 — geometry diagnostic, SCOD scoping, scalable LLLA

Follow-up to `imagenet_vit_baselines/`, asking **why** class-conditional Mahalanobis beats
P&C on this backbone, and whether a properly-scoped SCOD or an official-package LLLA changes
the picture.

**Read [`RESULTS_REPORT.md`](RESULTS_REPORT.md).** It carries all findings and the honest
status of what was and was not run.

## Headline findings

- Mahalanobis's advantage is **not** label supervision — refitting on the ViT's own predicted
  labels matches ground truth (Near 78.86 vs 78.82).
- P&C's mechanism quantities (perturbation transfer, post-correction residual, ridge leverage)
  are **weak** detectors, 55–62 Near AUROC against 76.43 for its predictive entropy. P&C's OOD
  signal is largely inherited base confidence, not perturbation-response geometry.
- Signal-flow bottleneck is `PERTURBATION_SUBSPACE` (Spearman 0.634 into the perturbation
  response; 0.93–0.99 through every subsequent stage).
- Official `laplace-torch` LLLA-Kron collapses to the MAP (base agreement 1.0000) and is
  *weaker* on Near (72.95) than this repo's earlier hand-rolled KFAC (74.63).

## Frozen inputs

Nothing here retunes P&C — it is used at the frozen headline configuration r=2, λ=1000, K=20,
M=20 — and the Mahalanobis headline from `imagenet_vit_baselines/` is unchanged. No OOD data
was used to select any hyperparameter: LLLA's prior precision was chosen on ID selection-pool
NLL only.

## Reproducing

```bash
V=/home/elean/pnc/.venv_vit/bin/python      # NOT the main .venv (no sm_61 kernels)
cd /home/elean/pnc
$V -m experiments.imagenet_vit_pnc.geom_cache            # multi-layer CLS caches (~2.0 GB)
$V -m experiments.imagenet_vit_pnc.geom_run --stage llla       # Part C
$V -m experiments.imagenet_vit_pnc.geom_run --stage analysis   # Part A
```

`raw/` is git-ignored and regenerable; see [`MANIFEST.md`](MANIFEST.md) for every artifact.

## Not run

Sub-parts A6 (disagreement sets), A7 (K sweep), A8 (layerwise Mahalanobis) and all of Part B
(SCOD-linear / SCOD-FFN / SCOD-last-block). §8–§9 of the report state exactly what is missing
and what each would take. No SCOD number appears anywhere here, because none was measured.
