# Final-block P&C on ImageNet-1K ViT-B/16

Full-scale run of Perturb-and-Correct on a standard pretrained ImageNet ViT-B/16,
motivated by the TITAN X feasibility preflight in
[`../imagenet_vit_preflight/`](../imagenet_vit_preflight/).

**Verdict: `SUPPORTS_LARGE_SCALE_TRANSFER`** — see
[`RESULTS_REPORT.md`](RESULTS_REPORT.md) for the full argument and
[`HEADLINE_RESULTS.md`](HEADLINE_RESULTS.md) for the §32 checkpoint written before any
optional analysis.

## The result in one table

| Method | ID Acc | ID NLL | Near AUROC | Near FPR95 | Far AUROC | Far FPR95 |
|---|---|---|---|---|---|---|
| Base / MSP | 81.07 | 0.8482 | 73.52 | 81.84 | 86.04 | 51.74 |
| Energy | 81.07 | 0.8482 | 62.39 | 93.16 | 78.96 | 85.29 |
| ReAct + Energy | 81.07 | 0.8482 | 69.21 | 84.23 | 85.61 | 53.90 |
| Uncorrected perturb. | 81.05 ± 0.02 | 0.8440 | 74.60 ± 0.02 | 74.63 ± 0.35 | 86.52 ± 0.02 | 48.70 ± 0.15 |
| **P&C** | **81.04 ± 0.01** | **0.8431** | **74.74 ± 0.02** | **74.37 ± 0.14** | **86.52 ± 0.01** | 48.75 ± 0.08 |

5 construction seeds, M = 20. Base parity: 81.068 % top-1 against the published 81.072 %.

Three things worth knowing before reading further:

1. P&C is ahead of every baseline on **all five** OOD datasets, but the margin over MSP is
   modest (+1.2 Near AUROC), with the clearer win on FPR95 (−7.5 pp Near).
2. At the ID-selected operating point the **correction contributes almost nothing over the
   matched uncorrected perturbations** — this is reported in the headline, not buried.
3. The post-hoc scale sweep shows why, and is the most interesting result here: the
   correction's benefit grows with perturbation size, and past r ≈ 1 the uncorrected
   ensemble collapses while P&C keeps improving. See
   [`figures/posthoc_ood_vs_scale.png`](figures/posthoc_ood_vs_scale.png).

## Layout

| path | contents |
|---|---|
| `RESULTS_REPORT.md` | the full report; answers Q1–Q8 of the spec |
| `HEADLINE_RESULTS.md` | core result, frozen before optional analyses (§32) |
| `MANIFEST.md` | provenance: commit, checkpoint SHA, versions, splits, commands |
| `configs/` | the exact configuration as run |
| `splits/` | class-stratified train pool manifests + SHA-256; `SPLIT_PROTOCOL.md` |
| `selection/` | Stage A/B sweeps, frozen gate, selected config, selection report |
| `metrics/` | base ID, final ID, OOD results, temperature, post-hoc sensitivity |
| `baselines/` | MSP / Energy / ReAct per-dataset metrics |
| `predictions/` | per-example scores for every method and seed (§24) |
| `tables/` | main table (tex/md/csv), hyperparameter search, n_cal×ridge diagnostic |
| `figures/` | ID-stability frontier; post-hoc OOD vs scale |
| `timing/` | measured runtime and memory (§31) |
| `provenance/` | correctness-gate results |

**Not committed** (regenerable; see `MANIFEST.md` for sources and checksums):
`raw/` activation caches and `construction/` member weight matrices, ~1.8 GB. No ImageNet
or OOD images, and no model checkpoints, are stored here.

## Reproducing

Code lives in [`../../../experiments/imagenet_vit_pnc/`](../../../experiments/imagenet_vit_pnc/).
Use `.venv_vit`, not `.venv` — the main environment's torch has no `sm_61` kernels and
cannot run this GPU at all. Stages are listed in order in `MANIFEST.md`; run them
sequentially (one GPU job at a time).

Correctness gates first:

```bash
.venv_vit/bin/python -m experiments.imagenet_vit_pnc.full_validate
JAX_PLATFORMS=cpu .venv/bin/python -m experiments.imagenet_vit_pnc.full_validate \
    --only ood_metric_parity
```

Twelve gates, including the load-bearing one: logits read out of the cached CLS residual
match a full model forward to 1.24e-05 on real validation images, and the OOD metrics match
`pnc_core/openood_eval.py` to <1e-12.

## Protocol guarantees

- Construction, selection and calibration use **ImageNet training data only**; the official
  50,000-image validation set is untouched until final evaluation.
- **No OOD data was read before the configuration was frozen.** `selected_config.json` and
  `ID_STABILITY_RULE.md` were committed to git in a separate commit before the OOD stage
  ran (`git log` on this directory shows the ordering).
- The OOD score was fixed in advance (predictive entropy) and was not changed after seeing
  the benchmark. Post-hoc analyses live in `metrics/posthoc_ood_sensitivity/` and are
  labelled as such; they do not revise the headline (§29).
