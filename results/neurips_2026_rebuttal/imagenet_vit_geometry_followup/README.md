# ImageNet ViT-B/16 — geometry follow-up: why Mahalanobis wins, and what P&C actually measures

Second diagnostic round on the frozen ViT-B/16 P&C result. The previous round
(`imagenet_vit_geometry_scod_llla/`) proposed `PERTURBATION_SUBSPACE` as the bottleneck.
This round treats that as a hypothesis and tests it directly — and it does not survive in
that form.

**Read [`RESULTS_REPORT.md`](RESULTS_REPORT.md).** [`MANIFEST.md`](MANIFEST.md) lists every
artifact and how to regenerate it.

## What this round establishes

- **Not a rank limit.** A random 20-dimensional view of the CLS geometry gives Mahalanobis
  only 70.79 Near AUROC — *below* P&C's 76.43 at K=20. P&C's 20 directions are already
  better than 20 random ones.
- **Mahalanobis lives in low-variance directions.** The lowest-variance 96 covariance modes
  alone score 81.81 Near, beating the full 768-mode detector (78.82); the highest-variance
  96 score only 68.32.
- **P&C excites the opposite end of the spectrum.** Its perturbation-response energy tracks
  ID variance at Spearman **+0.9925** — that is −0.9925 against the 1/λ weighting
  Mahalanobis applies. 26.7 % of P&C's response energy lands in the top-96 modes; 4.7 % in
  the bottom 96.
- **P&C is not a relabelled base model, but it is close to one.** Against base entropy at
  the same temperature it gains +1.90 Near / +1.32 Far, while sharing 88–94 % of its ranking
  (Spearman, computed within ID and within OOD separately).
- **The two methods measure different things.** On ImageNet misclassification detection the
  ranking inverts: Mahalanobis is worst of the confidence scores (78.00) and temperature-
  scaled base entropy is best (86.81).
- **Disagreement is one-sided.** 282 examples are high-Mahalanobis/low-P&C; only 16 are the
  reverse. On the former, all 20 members agree on the top-1 — geometrically unsupported
  inputs on which every member extrapolates identically.

## Frozen inputs

The headline P&C configuration (r=2, λ=1000, K=20, M=20) is unchanged and the Mahalanobis
headline is untouched. The K sweep is a diagnostic and does **not** redefine P&C. LLLA's
prior precision stays at the value the previous round selected by ID-only NLL. No OOD data
was used to select any hyperparameter, scope, rank, layer or temperature anywhere here.

## Reproducing

```bash
V=/home/elean/pnc/.venv_vit/bin/python      # NOT the main .venv (no sm_61 kernels)
cd /home/elean/pnc
$V -m experiments.imagenet_vit_pnc.fu_base                       # §1  base entropy control
$V -m experiments.imagenet_vit_pnc.fu_disagree                   # §6-7 disagreement
$V -m experiments.imagenet_vit_pnc.fu_randproj                   # §3  projection control
$V -m experiments.imagenet_vit_pnc.fu_spectrum                   # §4  eigenspectrum
$V -m experiments.imagenet_vit_pnc.fu_align                      # §5  spectral alignment
$V -m experiments.imagenet_vit_pnc.fu_ksweep --stage search      # §2  ID-only selection
$V -m experiments.imagenet_vit_pnc.fu_ksweep --stage final       # §2  final ensembles
$V -m experiments.imagenet_vit_pnc.scod_run --stage validate     # §10 gates
$V -m experiments.imagenet_vit_pnc.scod_run --stage qsweep       # §14 MC convergence
$V -m experiments.imagenet_vit_pnc.scod_run --stage fit --scopes linear,ffn   # §11-16
$V -m experiments.imagenet_vit_pnc.scod_run --stage nsweep       # §15 calibration size
$V -m experiments.imagenet_vit_pnc.fu_llla_temp                  # §18 LLLA + temperature
$V -m experiments.imagenet_vit_pnc.fu_layercache                 # §20 train-pool layer cache
$V -m experiments.imagenet_vit_pnc.fu_layerwise                  # §21 layerwise Mahalanobis
$V -m experiments.imagenet_vit_pnc.fu_iderror                    # §8  ID error detection
$V -m experiments.imagenet_vit_pnc.fu_tables                     # §23 main table
$V -m experiments.imagenet_vit_pnc.fu_figures                    # §24 figures
```

`raw/` is git-ignored and regenerable. See the report for anything not run.
