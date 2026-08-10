# CIFAR-10 Submission Manifest (NeurIPS 2026 Rebuttal — Phase 0)

**Purpose:** identify the exact P&C run that produced the submitted CIFAR-10 Table 2 row,
resolving conflicting artifacts, before any reproduction or sweeps are launched.

**Audit metadata**
- Repo: `/home/elean/pnc`
- Branch: `neurips-2026-rebuttal`
- Git commit at audit time: `854f4d847ce9ce0d723e7b49a707ebe2f5fcb95c`
- Working tree: **DIRTY** (322 changed/untracked paths; pre-existing experiment infra, not created by this audit)
- Hardware: NVIDIA GeForce RTX 5060, 8 GB VRAM (8151 MiB), driver 580.88, WSL2
- Env: `.venv`, Python 3.12.3, JAX 0.9.1, Flax 0.12.3, single CUDA device
- Audit date: 2026-07-23

---

## 0. Executive summary — the central conflict and its resolution

There are **two candidate "submitted" P&C configurations**, backed by different artifacts:

| | **Candidate A (SELECTED)** | Candidate B |
|---|---|---|
| Target block | `s3b0` = (stage_idx=3, block_idx=0) = stage4 block0 | `s3b1` = (stage_idx=3, block_idx=1) = stage4 block1 |
| Perturbation scale | ps = **25.0** | ps = 50.0 |
| Bootstrap | **bf = 0.05** (per-member resample) | none |
| K / M / directions | 20 / 50 / random | 20 / 50 / random |
| Reproduces which table | **`experiments/cifar_tables_paper.tex`** "PnC" row (exact, 7/7 metrics) | `experiments/cifar10_ood_bootstrap_table.tex` "PnC single (sc=50)" row |
| Supported by | the final paper-named `.tex` (latest artifact); numerical match | every internal markdown narrative doc (Apr 10–14) |

**Resolution — Candidate A (`s3b0` / ps25 / bf0.05) is selected as the submitted config**, on three independent grounds:

1. **Exact numerical match.** The 3-seed aggregate of
   `openood_v1p5_pnc_single_block_vcal_s3b0_k20_n50_ps25.0_lr0.001_bf0.05_..._seed{0,1,2}_random.json`
   reproduces the `cifar_tables_paper.tex` "PnC" row to every printed digit:

   | metric | paper `cifar_tables_paper.tex` | s3b0/ps25/bf0.05 (recomputed) |
   |---|---|---|
   | Acc % | 95.59 ± 0.20 | 95.59 ± 0.20 |
   | NLL | 0.138 ± 0.003 | 0.138 ± 0.003 |
   | ECE | 0.0050 ± 0.0008 | 0.0050 ± 0.0008 |
   | Near AUROC | 91.55 ± 0.13 | 91.55 ± 0.13 |
   | Near FPR95 | 33.08 ± 0.63 | 33.08 ± 0.63 |
   | Far AUROC | 95.09 ± 0.50 | 95.09 ± 0.50 |
   | Far FPR95 | 18.15 ± 1.07 | 18.15 ± 1.07 |

   The **ECE (0.0050)** is the decisive fingerprint: Candidate B (`s3b1`/ps50) has ECE 0.0079 and cannot
   produce it. Per the task tie-break rule ("select the candidate that matches the submitted values numerically"),
   this alone selects Candidate A.

2. **Chronology.** `cifar_tables_paper.tex` (the file literally named for the paper) is the **latest** table
   artifact, written 2026-04-28 16:08 — after the s3b0/bf0.05 result JSON was produced (2026-04-28 14:25),
   and well after the appendix bootstrap table (2026-04-20) and the narrative docs (2026-04-10–14). The final
   table was regenerated last, from the s3b0/bf0.05 run.

3. **Provenance.** The s3b0 config originates from the block-position re-sweep
   `scripts/sweep_pnc_block_position_seeds_1_2.sh`, which re-ranked all 6 single-block positions × 3 seeds at
   bf=0.05/ps=25/K=20 and **switched the pick from s3b1 (seed-0-only choice) to s3b0**, as recorded in
   `experiments/cifar10_swept_hyperparameters.txt:100-104`.

> **✅ RESOLVED — author-confirmed 2026-07-23.** The author confirmed against the submitted PDF's Table 2 that
> **Candidate A (`s3b0`/ps25/bf0.05) is the submitted configuration**. It is frozen as the rebuttal anchor in
> `ANCHOR_CONFIG.json`. Candidate B (`s3b1`/ps50) is retained as the earlier/appendix lineage. Phase 1 seed-0
> reproduction passed the gate for BOTH candidates at floating-point level (see `reproduction_cifar_seed0.md`).

> **⚠ NAMING-COLLISION HAZARD.** The internal tuning docs (`cifar10_tuning_plan.md:129,162`) use a **1-indexed
> human label "S3B0" that they define as code `stage_idx=2, block_idx=0`** (filename token `s2b0`). This is a
> *different block* from the OOD filename token `s3b0` (code stage_idx=3, block_idx=0). Do not conflate them.

---

## 1–24. Itemized Phase-0 answers (for the SELECTED config, Candidate A)

Citations are paths under `/home/elean/pnc`. "builder_metadata"/"protocol" refer to fields stored inside the
result JSON `openood_v1p5_pnc_single_block_vcal_s3b0_..._seed0_random.json`.

**1. Single-block or multi-block?** — **Single-block.** builder='single_block' (JSON `builder_metadata.builder`;
`cifar_tasks.py:387-464` `_build_single_block_pnc_ensemble`). Multi-block (all-8-block, scale=7) exists only as a
secondary, lower-performing row (`cifar_paper_tables.md:29`, `cifar_neurips_strengthening_log.md:251`).

**2. Exact task class.** — `CIFAROpenOODPnC` Luigi task (`cifar_tasks.py`, defaults at `:1240-1241`, filename
build `:2258-2305`), which builds a `PnCEnsemble` (`ensembles.py:1535-1774`) via `_build_single_block_pnc_ensemble`
and evaluates through `evaluate_openood_cifar` (`openood_eval.py:98-205`).

**3. Exact result JSON consumed.** — The 3 seeds of
`results/cifar10/openood_v1p5_pnc_single_block_vcal_s3b0_k20_n50_ps25.0_lr0.001_bf0.05_e300_optsgd_lr1e-01_wd5e-04_bs128_wu5_mom0p9_n1_augfcco8_ls0_subsetsize1024_chunksize1024_seed{0,1,2}_random.json`
→ table `experiments/cifar_tables_paper.tex` (PnC row). (Candidate B's files feed
`experiments/cifar10_ood_bootstrap_table.tex` via `scripts/make_cifar_bootstrap_tables.py:76`.)

**4. Target stage and block.** — `target_stage_idx=3, target_block_idx=0` (JSON `builder_metadata`;
`cifar_tasks.py:409-410`, `ensembles.py:1723-1731`). Maps to `stages=[stage1,stage2,stage3,stage4][3][0]` =
**first block of stage4** (`models.py:417-419`).

**5. Indexing convention (3,0) vs (3,1)?** — Code convention is **(stage=3, block=0), both 0-indexed = "(3,0)"**.
The corrected conv2 is 3×3, 512→512 (`models.py:333-341`); block0 has a downsample/shortcut (1×1, 256→512,
stride 2; `models.py:343-355`, forward `ensembles.py:1757-1758`). Candidate B is (3,1). Note the prose-"S3B0"
collision in §0.

**6. Direction family.** — **Random** (random_directions=True). K Gaussian directions QR-orthonormalized in
conv1 parameter space; `find_random_directions` (`pnc.py:159-172`), selected at `cifar_tasks.py:424-427`.

**7. Perturbation rank K.** — **20** (`n_directions=20`; JSON filename `k20`; `cifar_tasks.py:440`).

**8. Ensemble size M.** — **50** members (`n_perturbations=50`; `z_coeffs` shape (50,20), `cifar_tasks.py:440`;
member loop `ensembles.py:1623,1772`).

**9. Perturbation-scale grid evaluated.** — Phase-2 val grid {1,5,10,50,100,200,500,1000,2000,5000}; OOD-eval grid
{25, 50} (`cifar10_swept_hyperparameters.txt:49-52`).

**10. Scale selected.** — **25.0** (`ps25`). Selection criterion = lowest ID-validation NLL after temperature
scaling; no OOD used (`cifar10_swept_hyperparameters.txt:4-5,49`). Candidate B used 50.

**11. Calibration subset size.** — **1024** images (`subset_size=1024`; JSON `builder_metadata.subset_size_used`;
`cifar_tasks.py:404-407`). Distinct from: the temperature/ID-val split = **5000** images (first 10% of the
50000-image CIFAR-10 train set, `_split_data` seed=99, `util.py:404-415`).

**12. Ridge value.** — **λ = 1e-3** (`lambda_reg=1e-3`; filename `lr0.001`; `cifar_tasks.py:2262`). Doc note:
λ had no measurable effect across 5 orders of magnitude (`cifar10_swept_hyperparameters.txt:57-60`).

**13. Ridge centered on original parameters?** — **Yes.** The solve fits a *delta* to the original conv2 weights;
the design residual is `R = T − Y_pert · w2_orig` and `(H+λI)Δθ = b` with `w2_new = w2_orig + Δθ`, so `λ‖Δθ‖²`
penalizes deviation from the **original** conv2 params (`pnc.py:110-132, 246-259`).

**14. Temperature scaling enabled?** — **Yes** (`posthoc_calibrate=True`; JSON `posthoc_calibrate`;
`openood_eval.py:117`). Fitted T≈0.756 (seed0).

**15. ID split for temperature.** — **ID validation only** (`protocol.temperature_fit_split="id_validation_only"`).
Golden-section NLL minimization on the 5000-image held-out ID val split; no OOD data
(`openood_eval.py:121`, `util.py:142-190`; `cifar_tasks.py:2306-2313`).

**16. Seeds used.** — **0, 1, 2**.

**17. Exact OOD score for AUROC and FPR95.** — **predictive_entropy** for both (`protocol.primary_score`;
`openood_eval.py:29,106,203-204`). AUROC and FPR95 are computed from the same score array per dataset
(`openood_eval.py:62-72`). Six scores are stored per dataset (predictive_entropy, max_softmax_uncertainty,
energy_score, margin_uncertainty, mutual_information, variation_ratio); predictive_entropy is the headline.

**18. Member probabilities averaged after softmax?** — **Yes**, probability-space mean: `probs=softmax(logits)`
then `mean_probs=probs.mean(axis=0)` (`openood_eval.py:26-28`; classification path `util.py:543-545`). Members
return raw logits; softmax precedes averaging.

**19. BatchNorm statistics frozen?** — **Yes**, `use_running_average=True` throughout construction and eval
(`cifar_tasks.py:55,65,136,145`; `ensembles.py:1735-1765`; OOD path `openood_eval.py:221` etc.).

**20. Was BN2 absorbed into the affine correction?** — **No.** `bn2` (between conv1 and conv2, `models.py:332`)
is applied with frozen stats *before* the ridge design features are formed
(`y_bn2=bn2(...); relu; extract_patches`, `cifar_tasks.py:145-150`); the correction is the conv2 weight delta
plus a new per-channel bias only. bn2 scale/shift are untouched.

**21. 3, 5, or other seed count?** — **3 seeds** (`cifar_paper_tables.md:4`,
`cifar_neurips_strengthening_aggregate.md:2-18`). "5 seeds" appears only as a hypothetical future step.

**22. MC Dropout — 32 or 50 passes?** — **32** stochastic passes in the submitted OOD tables
(`openood_v1p5_mc_dropout_vcal_n32_dr0.1_...`; `cifar_paper_tables.md:31`; `cifar10_ood_detection.md:38`;
`cifar10_swept_hyperparameters.txt:22`). dropout_rate=0.1. (An earlier ID-only UQ pass used n=50,
`cifar10_experiment_log.md:93`; superseded for OOD.)

**23. Deep Ensemble — 5 members?** — **Yes**, n=5, labeled 5× training cost
(`openood_v1p5_standard_ensemble_vcal_n5_...`; `cifar_paper_tables.md:34`; `cifar10_ood_detection.md:39`).

**24. OpenOOD Near/Far definitions.** — Near-OOD = **CIFAR-100, Tiny-ImageNet-200**; Far-OOD = **MNIST, SVHN,
Textures (DTD), Places365** (`data.py:626-651`). Full official test sets, no subsampling by default
(`benchmark_metadata.max_examples_per_dataset=None`); per-file counts: cifar100=10000, tiny_imagenet=10000,
mnist=10000, svhn=26032, textures=5640, places365=10000. Near/Far AUROC = unweighted **macro-mean over datasets**
(`openood_eval.py:75-95,203-204`); `concat_auroc` (pooled) also stored but not headline. Caveat: Places365 uses a
10k subset and DTD its native 5640 — not identical to public OpenOOD leaderboard splits
(`cifar_paper_narrative.md:117-123`).

---

## Additional submitted-baseline settings (for the full Table 2)

| Method | Members / passes | Key hyperparams | OOD result file prefix |
|---|---|---|---|
| PreAct ResNet-18 / MSP / Energy / Mahalanobis / ReAct+Energy | 1 | derived scores; ReAct p=90 | `openood_v1p5_{preact_resnet18,msp,energy,mahalanobis,react_energy}_vcal_*` |
| LLLA | 50 | prior_precision = 10.0 | `openood_v1p5_llla_vcal_n50_prec10.0_*` |
| Epinet | 50 | prior_scale = 3.0, index_dim=8, hiddens=[50,50] | `openood_v1p5_epinet_vcal_n50_*_ps3.0_*` |
| MC Dropout | **32** | dropout_rate = 0.1 | `openood_v1p5_mc_dropout_vcal_n32_dr0.1_*` |
| SWAG | 50 | start=240, freq=1, rank=20, bn_refresh=2048 | `openood_v1p5_swag_vcal_n50_*_sws240_*` |
| Deep Ensemble | **5** | — (5× train cost) | `openood_v1p5_standard_ensemble_vcal_n5_*` |

---

## Exact reproduction command (SELECTED config, from luigi log `experiments/logs/p2_s3b0_ood_20260428_141145.log`)

```bash
cd /home/elean/pnc && source .venv/bin/activate
python -m luigi --module cifar_tasks CIFAROpenOODPnC --local-scheduler \
  --dataset cifar10 --epochs 300 \
  --n-directions 20 --n-perturbations 50 --perturbation-sizes "[25.0]" \
  --target-stage-idx 3 --target-block-idx 0 \
  --lambda-reg 1e-3 --subset-size 1024 --chunk-size 1024 --random-directions \
  --bootstrap-frac 0.05 --seed 0 --posthoc-calibrate
```

Full stored task signature (seed 0):
`CIFAROpenOODPnC(batch_size=128, lr=0.1, weight_decay=0.0005, optimizer=sgd, momentum=0.9, nesterov=True,
warmup_epochs=5, cutout_size=8, label_smoothing=0.0, openood_root=openood_data,
openood_max_examples_per_dataset=0, dataset=cifar10, epochs=300, n_directions=20, n_perturbations=50,
perturbation_sizes=[25.0], subset_size=1024, chunk_size=1024, target_stage_idx=3, target_block_idx=0,
random_directions=True, seed=0, lambda_reg=0.001, posthoc_calibrate=True, bootstrap_frac=0.05)`

**Prerequisites verified present:**
- Base checkpoints (seeds 0,1,2): `results/cifar10/preact_resnet18_train_e300_optsgd_lr1e-01_wd5e-04_bs128_wu5_mom0p9_n1_augfcco8_ls0_seed{0,1,2}.pkl` ✓
- OpenOOD data (`openood_data/`, 3.3 GB): near `{cifar100, tiny_imagenet}.npz`, far `{mnist, svhn, textures, places365}.npz` ✓
- Cached submitted result JSONs (seeds 0,1,2) present for byte-comparison against reproduction ✓

`chunk_size` is a memory-only knob (results invariant); the submitted OOD run used 1024, the block-position
val sweep used 16. On this 8 GB GPU either is workable for stage4.

---

## Conflicting / superseded artifacts (retained, not the submitted config)

- `experiments/reproduce_pnc_baseline.py` — a **different** reproduction attempt: s3b1, K=16, n=32, scale=50, no
  bootstrap, and a **different checkpoint** (`results_newer/cifar10/preact_resnet18_e300_bs128_lr1e-03_wd1e-04_seed0.pkl`,
  i.e. lr1e-03/wd1e-04, unlike the submitted seed's lr1e-01/wd5e-04). Not the submitted lineage.
- All `experiments/cifar*_*.md` narrative docs (Apr 10–14) describe s3b1/ps50, no bootstrap — **stale**; they
  predate and do not record the Apr-28 switch to s3b0 + bootstrap.
- `experiments/cifar10_ood_bootstrap_table.tex` (Apr 20) — appendix superset; its "PnC single (sc=50)" row is
  Candidate B (s3b1/ps50), a distinct lineage from the main paper row.
