# MANIFEST — rebuttal-era artifacts collected for the P&C paper revision

Collected on branch `agent/collect-rebuttal-results`, cut from `neurips-2026-rebuttal`
at commit **`70fb480fadcc4ee8e9d8ae0b8a060de431007aa0`** ("evidence").

Machine-readable companions:

| file | contents |
|---|---|
| `STATUS.csv` | one row per requested artifact, with status and notes (89 rows) |
| `provenance/artifact_inventory.csv` | every copied file: source path, destination, SHA-256, size, source mtime, git state (146 rows) |
| `provenance/large_artifacts.csv` | the 6.22 GB deliberately **not** committed, with per-tree checksum manifests |
| `provenance/large_artifact_checksums/*.sha256` | per-file SHA-256 for each large tree (1,141 files) |
| `provenance/protocol_configuration_record.csv` | §10 protocol facts, each with a `file:line` code reference |

Per-file SHA-256 values are in `provenance/artifact_inventory.csv` and are not
repeated here; this document records the *experimental* provenance that a checksum
cannot express.

---

## 0. Repository situation (read this first)

**The paper repository `https://github.com/EtherealEq/perturb_and_correct.git` is not
reachable from this machine.** `git ls-remote` over HTTPS fails on credentials and over
SSH returns `ERROR: Repository not found`; the `gh` CLI is not installed. The only git
repository anywhere on the filesystem is `/home/elean/pnc` (remote
`git@github.com:DrKwint/pnc.git`), which contains both the experiment code and an April
copy of the manuscript under `neurips_draft/`. `revision_results/` was therefore created
here. **No manuscript `.tex` file was modified.**

**Two artifact-location facts shape everything below.**

1. **`results/cifar10/` is deleted in the working tree but intact in git HEAD.** All 137
   result JSONs exist as blobs at `70fb480`. They are recovered non-destructively by
   `provenance/extract_cifar_from_git.py` (`git show HEAD:<path>`); nothing in the working
   tree was touched.
2. **The rebuttal-era CIFAR experiments were run on a different machine and never copied
   back** (user-confirmed; the split is recorded in
   `scripts/neurips_2026_rebuttal/cifar_other_machine/README.md`). This machine has no
   PreActResNet-18 checkpoints and no `openood_data/`. Every CIFAR-specific rebuttal
   number — the sensitivity cross-product, the SCOD comparison, the 8,010 s training time —
   is `MISSING` **here** and should be retrieved from that machine rather than rerun.

Global protocol, hardware and software provenance: `provenance/PROVENANCE_original.md`
(NVIDIA TITAN X Pascal 12,288 MiB; i7-12700KF; 15 GiB RAM; WSL2; Python 3.12.3; JAX 0.9.1;
Torch 2.11.0+cu130; GPU jobs strictly sequential).

---

## 1. `mujoco_sensitivity/`

| field | value |
|---|---|
| **Original path** | `results/neurips_2026_rebuttal/mujoco_sensitivity/aggregates/far_sensitivity_raw.csv` |
| **Repo / commit** | `DrKwint/pnc` @ `70fb480` (file itself is untracked) |
| **Modified** | 2026-07-25 23:30:39 UTC (6,279,792 B, SHA-256 `cef0a039…`) |
| **Configuration** | per-env anchors in `anchors.json`: K=20, M=50, correction subset 4,096, multi-block (perturb `[0,2]`), random directions, hidden `[200,200,200,200]`; per-env (λ, bootstrap fraction, scale) |
| **Checkpoints** | base MLPs retrained per (env, seed) by `script_run_sensitivity.py`; not separately archived |
| **Seeds** | 27: 0–11, 13, 17, 19, 23, 29, 31, 37, 41, 42, 100, 123, 200, 314, 404, 500 |
| **Dataset / split** | Minari MuJoCo; ID = `expert-v0`, Near = `medium-v0`, Mid = `simple-v0`, Far = uniform-random actions; `val_split=0.1`, `seed=99` |
| **Selection criterion** | anchor configuration is fixed per environment; the swept factor is varied one at a time around it |
| **OOD used for tuning** | **No** |
| **Aggregation script** | `script_aggregate_far.py` (+ `script_run_sensitivity.py`, `script_gen_data.py`, `script_merge_bootfull.py`) |
| **Artifact kind** | raw (per env-seed-value rows), plus aggregated `far_by_environment.csv` / `far_across_envs.csv` and a plotted table |

**Coverage.** 14,520 rows; 11 environments; 7 factors (`scale`, `rank`, `bootstrap`,
`bootfull`, `calib`, `ridge`, `layer`); `status == 'ok'` on every row. The 573 per-run
JSONs that feed it are indexed in `provenance/large_artifact_checksums/mujoco_sensitivity_raw_json.sha256`.

**Column definitions** (the requested documentation of every column):

| column | meaning |
|---|---|
| `factor`, `factor_value`, `factor_value_numeric` | which knob was swept and to what value |
| `anchor_value`, `relative_to_anchor` | that environment's submitted setting, and value ÷ anchor (`1.0` marks the anchor row) |
| `scale`, `subspace_dimension_K`, `ensemble_size_M`, `bootstrap_fraction`, `layer_scope`, `direction_family`, `ridge_center` | the realized configuration of the run |
| `calibration_pool_size` | size of the clean ID pool the correction set is drawn from |
| `per_member_calibration_size` | rows each member's least-squares system actually uses = `bootstrap_fraction × pool` (a **with-replacement** draw) |
| `unique_calibration_rows` | *distinct* rows in that draw — the with-replacement analysis lives in the gap between these two columns |
| `feature_dimension_p` | correction design width **including the bias column** (201 = 200 hidden + 1) |
| `nominal_n_over_p`, `unique_n_over_p` | interpolation ratio computed from the nominal and the distinct row counts |
| `id_rmse`, `id_nll`, `id_total_variance`, `id_residual` | in-distribution metrics |
| `near_*`, `mid_*`, `far_*` (`rmse`, `nll`, `auroc`) | per-tier metrics; **`mid_*` is populated only for Ant, HalfCheetah, Hopper, Humanoid** |
| `far_spearman` | rank correlation of predictive uncertainty against squared error on Far. **No Near or Mid Spearman is computed anywhere in the harness.** |
| `gram_rank`, `gram_min_eig`, `gram_condition`, `normalized_ridge` | conditioning diagnostics of the correction Gram matrix |
| `construction_time_sec`, `mean_delta_w_norm`, `git_commit`, `status`, `error_message` | run bookkeeping |

**Inconsistencies with the quoted values.**

* The brief describes a **20-seed** study; the data has **27** seeds and no 20-seed snapshot
  exists (checked: the current file, both `.bak` snapshots, and both archived snapshots).
* The **sharp / flat / materially-worse counts cannot be reproduced.** The cell structure is
  confirmed (44 cells per factor = 11 environments × 4 metrics; 176 total), but no script in
  the repository produces the counts and none of six candidate rules comes close — best total
  L1 error 122 of 176 cells. See §1 of `RESULTS_AUDIT.md` and
  `mujoco_sensitivity/classification_rule_sweep.csv`.
* **`ridge_center` is `zero` in all 14,520 rows**, while `anchors.json` and
  `MUJOCO_FAR_SENSITIVITY_RESULTS.md` both state the ridge shrinks *toward original*.
* The `bootfull` double-descent sweep ran at `calibration_pool_size = 10000`; every other
  factor used 4,096. The curves are therefore not directly comparable to the rest of the study.

**Derived here** (scripts in `provenance/`): `cell_classification.csv`,
`cell_classification_counts_vs_quoted.csv`, `classification_rule_sweep.csv`,
`seed_paired_differences_ci95.csv`, `bootstrap_double_descent_curves.csv`,
`calibration_pool_curves.csv`, `interpolation_threshold_peaks.csv`.

**Superseded versions kept:** `SUPERSEDED_wrong_anchor_MUJOCO_SENSITIVITY_RESULTS.md`
(λ=0 / toward-zero / bf=0.1 anchor, 4 envs × 3 seeds) and the earlier HalfCheetah-only
3-seed study (`priority1_halfcheetah_3seed_sensitivity.csv`) — the latter is the source of
the §9 layer-scope numbers, see `layer_scope/`.

---

## 2. `cifar_sensitivity/`

| field | value |
|---|---|
| **Original path** | `results/cifar10/openood_v1p5_*.json` — **deleted in the working tree, recovered from git HEAD `70fb480`** |
| **Modified** | n/a (git blobs); the deletion is unstaged in the working tree |
| **Configuration** | `CIFARPnC` single-block stage 4 / block 1 (filename tag `s3b1`), K=20, M=50 (some M=25), `subset_size=1024`, `lambda_reg=1e-3`; multi-block variant also present |
| **Checkpoints** | `preact_resnet18_train_e300…_seed{0,1,2}.pkl` — **absent from this machine** |
| **Seeds** | 0, 1, 2 (coverage varies by configuration; single-block scale=25 has 3, scale=20 has 1) |
| **Dataset / split** | CIFAR-10 train/test; OpenOOD v1.5 Near = {CIFAR-100, Tiny-ImageNet-200}, Far = {MNIST, SVHN, Textures, Places365} |
| **Selection criterion** | ID validation only — every JSON records `ood_validation_used=false`, `ood_tuning_used=false`, `temperature_fit_split='id_validation_only'` |
| **OOD used for tuning** | **No** (recorded per run, not merely asserted) |
| **Aggregation script** | `provenance/extract_cifar_from_git.py` (written for this collection) |
| **Artifact kind** | raw result JSONs (byte-identical to HEAD, in `raw_json_from_git/`) + a normalized derivative |

**What is here:** the *submission-era* CIFAR OpenOOD results — 42 JSONs, 48
(file × scale) rows, with accuracy, ID NLL/ECE/Brier, Near/Far AUROC and FPR95, fitted
temperature and protocol flags.

**What is missing:** every rebuttal-era CIFAR arm. The prepared design
(`OTHER_MACHINE_run_cifar_sensitivity.sh`) is explicitly **coordinate descent** — one factor
at a time around the anchor — so the promised **target block × scale × bootstrap fraction
cross-product was never specified, let alone run**. It is marked `MISSING`, not approximated
by the coordinate sweeps.

**Inconsistency with the quoted values.** The nearest available comparison — single-block
scale = 25 over the 3 checkpoints — gives accuracy 95.70, Near AUROC 91.11, Near FPR95 34.99,
Far AUROC 94.04, Far FPR95 21.10, against a quoted 95.69 / 90.99 / 37.39 / 94.83 / 19.52.
Accuracy and Near AUROC are close; the Far columns are not. The quoted row is from the other
machine.

---

## 3. `scod/`

| field | value |
|---|---|
| **Original paths** | `results/posthoc_mujoco/scod/predictions/<env>/<seed>/_metrics.json` (33 files); report `results/posthoc_mujoco/scod/SCOD_REPORT.md` |
| **Repo / commit** | `70fb480-posthoc` (recorded as `git_tag` inside every `_metrics.json`) |
| **Modified** | report 2026-07-27 03:32:28 UTC (SHA-256 `3c3ccfb0…`) |
| **Configuration** | frozen probabilistic MLP `[200,200,200,200]` with heteroscedastic Gaussian head; randomized single-pass Nyström sketch of the dataset Fisher on the **4,096-point ID calibration set**; `num_eigs ≤ 100`, `num_samples = 604`, sketch seed `100000 + model_seed`; Case-B heteroscedastic Fisher-weighted Jacobian |
| **Checkpoint identifier** | regenerated canonical batch, archived at `archive/artifacts_backup/base_models_posthoc_canonical.tar.gz` (SHA-256 in `provenance/large_artifact_checksums/posthoc_base_models_backup.sha256`) |
| **Seeds** | SCOD 3 (0, 10, 200); the P&C column it is compared against uses 27 |
| **Dataset / split** | same Minari tiers, same ID data and splits as P&C |
| **Selection criterion** | (k, Mεps, α) by **ID validation NLL**; frozen before any OOD read |
| **OOD used for tuning** | **No** — `used_ood_for_selection = false` in all 33 files |
| **Aggregation script** | `provenance/verify_scod_mujoco.py`; run scripts `script_run_scod.py`, `script_scod_adapter.py`, `script_validate_scod.py` |
| **Artifact kind** | raw per-seed metrics → normalized `scod_mujoco_per_seed.csv` and `scod_vs_pnc_mujoco_by_environment.csv` |

**Verification.** All quoted values reproduce (`scod_quoted_value_checks.csv`): SCOD mean Far
AUROC 0.5121 vs 0.512; Ant 0.1102 / Hopper 0.0276 / Humanoid 0.2667 / Pusher 0.9946; Far/ID
score ratios Hopper 0.3203 and Ant 0.4999; **P&C exceeds SCOD on 11/11 environments** under
both the 27-seed and the seed-matched {0,10,200} P&C aggregation.

**Provenance discovery.** The quoted P&C column comes from an **earlier snapshot** of the
sensitivity data, not the current file:

| P&C source | mean Far AUROC | HumanoidStandup | InvertedPendulum | Humanoid |
|---|---|---|---|---|
| `far_sensitivity_raw.csv` (current, 14,520 rows) | 0.9197 | 0.9134 | 0.9130 | 0.3427 |
| **`…pre_bootfull_merge.bak` (11,720 rows)** | **0.9215** | **0.9277** | **0.9169** | **0.3443** |
| rebuttal quoted | 0.922 | 0.928 | 0.919 | 0.344 |

The 11,720-row snapshot is the canonical source for every quoted P&C Far AUROC and is copied
as `mujoco_sensitivity/far_sensitivity_raw_11720rows_CANONICAL_FOR_QUOTED_NUMBERS.csv`
(SHA-256 `84714879…`).

**Inconsistencies.** (a) The comparison is **not seed-matched** (SCOD 3 seeds vs P&C 27) —
a seed-matched column is provided alongside and changes the mean by 0.0002. (b) `SCOD_REPORT.md`
§5 prose says Ant's Far/ID energy ratio is "0.22×" while its own §3 table says 0.50; the table
value is the one quoted. (c) The report states the **original** base checkpoints were lost to a
filesystem event, so SCOD and the P&C sensitivity anchor do not share bit-identical
checkpoints. (d) **No CIFAR SCOD artifact exists on this machine.**

---

## 4. `efficiency/`

| field | value |
|---|---|
| **Original paths** | `results/neurips_2026_rebuttal/efficiency_v2/efficiency_v2_rows.csv` (+ 3 per-env JSONs); `efficiency_{construction_mujoco,storage,inference_mujoco,cifar_inference}.csv`; `results/cifar10/inference_cost.json` @ git HEAD |
| **Modified** | `efficiency_v2_rows.csv` 2026-07-27 04:19:10 UTC (SHA-256 `12333e93…`) |
| **Configuration** | MuJoCo seed 0, 5,000 training steps, `[200,200,200,200]`; P&C M=50 vs a Deep Ensemble of **50 actually-trained networks** |
| **Seeds** | efficiency_v2: seed 0 only. The older `efficiency_construction_mujoco.csv`: seeds 0, 10, 42 |
| **Hardware** | NVIDIA TITAN X (Pascal), recorded per row in the `device` column |
| **Aggregation script** | `provenance/verify_efficiency_and_distilbert.py`; producers `script_priority4_efficiency.py`, `script_priority4_inference_bench.py`, `script_efficiency_benchmark.py` |
| **Artifact kind** | measured (timing), derived (storage from parameter counts) |

**Label per the required MEASURED / ESTIMATED / UPPER_BOUND / DERIVED taxonomy:**

| quantity | label | value / note |
|---|---|---|
| MuJoCo base training, P&C post-hoc build, total construction | **MEASURED** | Ant 16.61 + 1.75 = 18.35 s; HalfCheetah 25.47 + 1.52 = 26.99 s; Hopper 25.60 + 1.73 = 27.34 s |
| Matched 50-member Deep Ensemble construction | **MEASURED** | Ant 1,131.94 s; HalfCheetah 1,321.85 s; Hopper 1,279.33 s |
| "≈47–62× cheaper to construct" | **DERIVED — confirmed** | recomputed 61.7× (Ant), 49.0× (HalfCheetah), 46.8× (Hopper) |
| MuJoCo SWAG / Laplace construction | **MEASURED** | in the same table |
| MuJoCo storage (`storage_mb`, `minimal_ckpt_mb`) | **DERIVED** | from parameter counts; the report separates *implemented* 16.8 MB from *theoretical-minimal* 7.15 MB and labels the latter not-yet-implemented |
| CIFAR inference SWAG 7.33 / Laplace 7.68 ms/sample | **MEASURED — confirmed** | `warm_per_sample_ms` 7.3253 and 7.6797 over 5,000 samples |
| CIFAR inference P&C 7.53 / Deep Ensemble 7.26 ms/sample | **not reproducible here** | this file gives P&C 7.4186 (single-block) / 9.6881 (multi-block) and Deep Ensemble n=5 = 1.3380. A matched M=50 CIFAR ensemble was never benchmarked on this machine |
| CIFAR base training 8,010 s | **absent** | the string does not occur in any artifact |
| CIFAR P&C construction ≈100 s / upper-bound status | **absent** | no CIFAR construction-time record exists here |
| MuJoCo reconstruction time; materialized and peak GPU memory (both benchmarks); CIFAR persistent disk storage | **absent** | never instrumented |
| Repetitions per timing measurement | **partially recorded** | `inference_cost.json` separates `cold_warmup_s` from `warm_total_s` at `n_bench_samples = 5000`; the MuJoCo runs record no repetition count |

---

## 5. `distilbert/`

| field | value |
|---|---|
| **Original path** | `results/banking77_distilbert_pnc/` |
| **Modified** | `TRANSFORMER_FULL_REPORT.md` 2026-07-27 18:46:51 UTC (SHA-256 `145fbdf2…`); `banking77_pnc.csv` 2026-07-27 18:13:20 UTC |
| **Checkpoint identifier** | `optimum/distilbert-base-uncased-finetuned-banking77` @ commit **`89565753ee34239b4ab70c9c90317085cc0f4f8e`**, 67,012,685 float32 parameters, 77 labels; converted PyTorch→Flax once (msgpack). Parity gate: max logit diff **7.2e-06**, top-1 agreement 100%, Flax test accuracy 0.925 |
| **Configuration** | target = final DistilBERT FFN (layer 5): perturb `lin1.kernel [768,3072]` with a random rank-20 basis; correct `(lin2.kernel, lin2.bias)` by the shared ridge solver, **ridge 1e-3 toward original, full bootstrap**; M=20 members, K=20; scale selected by ID-validation NLL under constraints (ID accuracy drop ≤ 0.25 pp, base agreement ≥ 99%) |
| **Seeds** | 5 P&C construction seeds: 0, 10, 42, 123, 2026 (no model fine-tuning) |
| **Dataset / split** | Banking77 10,003 train → 9,000 calibration / 1,003 ID-val (stratified, seed 20260726); 3,080 test; `max_length` 64 (0.2% truncated). OOD = CLINC-OOS: Near = banking + credit_cards, Cross = non-financial in-scope, Far = official `oos` |
| **Selection criterion** | ID-validation NLL only; one base temperature (T ≈ 0.80) fit on ID-val and shared across methods |
| **OOD used for tuning** | **No** |
| **Aggregation script** | `script_aggregate.py`; command `script_run_queue.sh` |
| **Artifact kind** | raw per-seed (`metrics_raw_per_seed.csv`, 176 rows) → aggregated table (`banking77_pnc.{csv,md,tex}`) |

**Verification — all 15 quoted values reproduce exactly** (`distilbert_quoted_value_checks.csv`):
corrected P&C 0.925 / 0.301 / 0.031; uncorrected 0.921 / 1.079 / 0.525; CLINC Near/Cross/Far
for P&C 0.907 / 0.967 / 0.981, MC Dropout 0.917 / 0.969 / 0.982, Energy 0.929 / 0.980 / 0.990.
Additional baselines present but not quoted: MSP, Entropy, Head-P&C. Seed SE ≤ 0.001 on all
cells. **No inconsistency found in this group.**

Large artifacts held out of git: the Flax checkpoint (269 MB) and the per-seed corrected
`lin2` weights (876 MB) — checksummed in `provenance/large_artifact_checksums/`.

---

## 6. `mechanism/`

| field | value |
|---|---|
| **Original path** | `results/neurips_2026_rebuttal/priority3/` |
| **Modified** | 2026-07-23 16:56–16:58 UTC; `mechanism_second_domain.csv` SHA-256 `bf150d67…` |
| **Configuration** | canonical P&C: multi-block, random directions, bootstrap fraction 0.1, K=20, M=50, size selected by ID val-NLL; hidden layer = the P&C calibration layer (`layer_idx=2`) |
| **Seeds** | seed 0 (single seed per environment) |
| **Dataset / split** | 10,000 points each from ID, Near, Mid, Far → **40,000 per environment** |
| **Selection criterion** | perturbation size by ID validation NLL |
| **OOD used for tuning** | **No** |
| **Aggregation script** | `script_priority3_mechanism.py` |
| **Artifact kind** | point-level raw (40,000 rows/env: `env, seed, regime, distance_mahal, disagreement_sqrt_predvar`) → summary markdown |

**Verification — all eight quoted values reproduce exactly.** HalfCheetah pooled ρ = 0.904,
Near/Mid/Far 0.837 / 0.860 / 0.816, regime-controlled slope 1.968 ± 0.008. Hopper pooled
0.749, 0.691 / 0.730 / 0.776, slope 0.324 ± 0.002. Distance is a **regularized hidden-space
Mahalanobis** to the calibration distribution; disagreement is √(total predictive variance);
the slope is an OLS of disagreement on log₁₀ distance with regime fixed effects.

**Gap:** no plot or plotting script was produced for this replication (the Ant original has
one; the replication does not).

**Random vs Low (Figure 3(C) broadening).** `random_vs_low_evidence.md` *does* broaden the
comparison beyond Ant — Ant / HalfCheetah / Hopper, 5 seeds, with Far AUROC, Far NLL, variance
ratios and mechanism diagnostics (Δh shrinkage, output-delta effective rank, member-pair
cosine). "Low" is the Lanczos bottom-eigenvector subspace of `Jᵀ(I − P_M)J` on a 4,096-point
ID subset, coefficients rescaled by the bottom eigenvalues (`pnc_core/gym_tasks.py:921`).
**However** its scale axis is a *val-NLL-selected bucket per seed*, not the swept scale grid
Figure 3(C) uses, so it is recorded `FOUND_UNVERIFIED`, not as a drop-in extension.

The `priority2` linearization diagnostics are filed here too: they are a *different* claim
(first-order sensitivity `A_S`, which breaks at operating scale) from the exact finite-scale
identity in §7, and the two must not be conflated.

---

## 7. `finite_scale_validation/`

| field | value |
|---|---|
| **Original paths** | `pnc_theory_reports/THEOREM_VALIDATION.md`; `pnc_theory_reports/MULTILAYER_THEORY_VALIDATION.md`; `artifacts/finite_transfer/mujoco_tier1_v2/identity.csv` |
| **Modified** | THEOREM_VALIDATION 2026-07-24 16:53:53 UTC (`769e28f5…`); MULTILAYER 2026-07-24 17:05:06 UTC (`18f0854f…`); identity.csv 2026-07-29 23:31:12 UTC (`df1d382a…`) |
| **Configuration** | float64, SVD-based; λ ∈ {0, 1e-2, 1, 100}; members {0,1,3,5}; **both perturbed blocks**; 1,024 test points per regime. `identity.csv`: 4 variants (canonical / shipped-default λ=0 lstsq / shipped ridge toward zero / small-calibration dual form) × float32 and float64 |
| **Checkpoints** | `artifacts/pnc_theory/base_models/*.npz` — **present** (41 files, 28.97 MB, checksummed) |
| **Seeds** | THEOREM_VALIDATION and MULTILAYER: **seed 0 only**. `identity.csv`: seeds 0, 10, 42, 100, 200 |
| **Dataset / split** | correction set, held-out ID, Near, Mid, Far |
| **Aggregation script** | `script_finite_identity.py`, `script_validate_bridge.py`, `script_multilayer.py`, `script_validate_multilayer.py`; unit tests `unittest_test_linalg.py` (189/189 to ≤1e-8) |
| **Artifact kind** | **manually transcribed** report tables (the raw JSONs are gone — see below) + raw `identity.csv` |

**The relative-error number lost to OpenReview formatting is recovered: ~1e-13 relative
(float64).** Specifically, the Eq 2/4 test-point residual identity
`r_v(x) = Θ[Δh̄_v − ΔX_vᵀ X_v G_v⁻¹ h̄_v]`:

| environment | ID median | Near / Mid / Far median (Ant) | max |
|---|---|---|---|
| Ant-v5 | 9.4e-14 | 7.3e-14 / 6.9e-14 / 5.2e-14 | ≤ 2.3e-11 |
| HalfCheetah-v5 | 5.4e-13 | — | ≤ 2.3e-11 |
| Hopper-v5 | 8.2e-13 | — | ≤ 2.3e-11 |

The independent `identity.csv` distribution (800 rows, per env × seed × member × variant ×
dtype) gives float64 `exact_rel_fro` median **5.97e-12**, max **8.31e-11** — the *distribution*
the brief asks for, not only an aggregate.

**Definitions** (from `script_finite_identity.py` and `METHODS.md`): the relative error is
Frobenius-relative, ‖predicted − actual‖_F ⁄ ‖actual‖_F, computed in two forms that are
reported side by side — `literal_*` (the brief's formula as written, which is exact only when
the ridge shrinks toward the original map or λ = 0) and `exact_*` (the regime-correct SVD form
`ActualResidual = (ΔA − Alphaᵀ ΔX − (Av − Av V diag(f₂) Vᵀ)) Θ` with `f₂ = s²/(s²+λ)`). Rank
deficiency is handled by the SVD form rather than by inverting a singular Gram matrix; the
`residual_scale` column carries the denominator so near-zero residuals are visible rather than
silently regularized.

**Cosine alignment and the downstream Jacobian.** `THEOREM_VALIDATION.md` §1.9/B.1 computes
`Δy(x) = F_down(z₀ + r₁) − F_down(z₀)` against the first-order `J_down(z₀) r₁`:

| environment | cos(Δy, J_down r₁) | Spearman(‖r₁‖, ‖Δy‖) | Taylor rel-err ID / Far |
|---|---|---|---|
| Ant-v5 | 1.00 (all regimes and scales) | 0.79–0.87 | 0.00 / 0.14 |
| Hopper-v5 | 0.99–1.00 | 0.66–0.72 | 0.04 / 0.18 |
| HalfCheetah-v5 | 0.97–1.00 | 0.63–0.87 | 0.02 / 0.25 |

The union is exactly the quoted **0.97–1.00**. Cosine is computed **per example**, then
summarized per (environment, regime, scale); the quoted range is the min–max across
environments.

**Two inconsistencies with the quoted coverage.**

1. **`artifacts/pnc_theory/bridge/` and `artifacts/pnc_theory/round1/` do not exist** — not on
   disk and never committed (`git log --all -- 'artifacts/pnc_theory/*'` is empty). The cosine
   and relative-error tables survive only as transcribed markdown. `script_validate_bridge.py`
   would regenerate them and the required base models are present, so this is a cheap
   regeneration, not a lost experiment.
2. **The coverage is smaller than claimed.** The rebuttal describes "all MuJoCo environments,
   three seeds, ID/Near/Mid/Far, both sequential correction stages". What exists: the
   cosine/Jacobian evidence covers **3 environments at seed 0**; `identity.csv` covers
   **4 environments × 5 seeds but only correction stage 0 and carries no regime column**;
   `MULTILAYER_THEORY_VALIDATION.md` covers **stage b under cumulative upstream perturbation
   for 3 environments at seed 0** (rel-err 1e-13 to 1e-12, ID and Far). No single artifact
   spans 11 environments × 3 seeds × 4 regimes × 2 stages.

---

## 8. `shift_tiers/`

| field | value |
|---|---|
| **Original paths** | `results/neurips_2026_rebuttal/priority5_pnc_tiers.csv` (P&C, 3 envs); `priority5_full_mujoco_tables.csv` (44 methods, 3 envs); `pnc_repro/figures/appendix_per_env_table_paper.txt` (6 methods, 11 envs) |
| **Modified** | `priority5_pnc_tiers.csv` 2026-07-23 16:37:39 UTC (`8bd75d25…`); appendix tables 2026-05-01 07:50:04 UTC |
| **Configuration** | `priority5_pnc_tiers.csv`: PJSVD-Multi-LS random, projected residual, probabilistic base, **bf=0.1, λ=0, subset 10000**, K=20, M=50, grid {5,10,20,50} |
| **Seeds** | `priority5_*`: 0, 10, 42, 100, 200 (5). Appendix table: 10–28 per environment |
| **Selection criterion** | perturbation size by **minimum ID validation NLL** (`nll_val`), fallback `nll_id` (`pnc_core/json_to_tex_table.py:49-51`) |
| **OOD used for tuning** | **No** |
| **Aggregation script** | `script_priority5_full_mujoco_tables.py`, `script_json_to_tex_table.py` |
| **Artifact kind** | per-seed raw rows plus a `mean+/-std` summary row per environment |

**Verification — all nine quoted illustrative AUROC values reproduce exactly:** Ant
0.7500 ± 0.0478 / 0.8246 ± 0.0391 / 0.9973 ± 0.0014; HalfCheetah 0.7421 / 0.9461 / 0.9880;
Hopper 0.7984 / 0.8611 / 0.9506.

**Conflicting newer aggregation (the brief explicitly asks for this).** The later 11-environment,
27-seed corrected-anchor study gives Far AUROC **Ant 0.998, HalfCheetah 0.987, Hopper 0.980**
against the quoted 0.997 / 0.988 / **0.951**. Hopper differs by 0.029. The two use different
anchors (bootstrap fraction, ridge, correction-pool size), different seed sets and different
base checkpoints. Recommended canonical source: see `RESULTS_AUDIT.md` §"Recommended canonical
sources".

**Mid tier.** Five environments lack a `simple` Minari dataset and so have no distinct Mid
tier: **Swimmer, Reacher, Pusher, InvertedPendulum, InvertedDoublePendulum**. Caveat recorded
in `mid_tier_availability.csv`: HumanoidStandup and Walker2d *do* have Mid available but were
never run with it, so **seven** environments have empty Mid columns in the sensitivity data.

**Selected-hyperparameter conflict.** `appendix_selected_hparams_paper.txt` and `anchors.json`
agree on 10 of 11 environments. **HumanoidStandup-v5 disagrees**: the appendix says
λ=1e-4, bf=0.3, ps=32.0; `anchors.json` and every executed sweep row say λ=1e-2, bf=0.20, ps=8.0.

**Coverage gap.** No artifact has all methods × all 11 environments × Near/Mid/Far. The
44-method table covers 3 environments; the 11-environment table reports only ID RMSE, Far NLL
and Far AUROC. Near/Mid **Spearman** is not computed anywhere for any method.

---

## 9. `layer_scope/`

| field | value |
|---|---|
| **Original path (source of the quoted numbers)** | `results/neurips_2026_rebuttal/priority1/sensitivity_mujoco.csv`, rows `sweep == 'layer'` |
| **Modified** | 2026-07-23 16:50:24 UTC (SHA-256 `6cd19a59…`) |
| **Configuration** | earlier priority1 canonical anchor: K=20, M=50, random directions, probabilistic base, **λ=0, bf=0.1, subset_size=10000**, hidden `[200,200,200,200]` |
| **Seeds** | **0, 10, 42 — and one environment only (HalfCheetah-v5)** |
| **Aggregation script** | `provenance/verify_layer_scope.py` |
| **Artifact kind** | raw per-seed rows |

**Verification — all six quoted values reproduce to three decimals:** single block
0.6815 / 0.8733 / 0.9756 → 0.681 / 0.873 / 0.976; multi block 0.6917 / 0.8932 / 0.9813 →
0.692 / 0.893 / 0.981.

**Inconsistency.** The rebuttal presents these as **"average AUROC"**, which reads as an average
over environments. It is a **three-seed average on HalfCheetah-v5 alone**. The 11-environment,
27-seed layer sweep exists (`sourceB_11env_27seed_layer_by_environment.csv`, 586 rows) and gives
single 0.764 / 0.815 / 0.904 versus multi 0.799 / 0.843 / 0.922 — it does not reproduce the
quoted values under any environment subset or aggregation order tried (all-11 macro and pooled,
the 4 Mid-tier environments, the 3 headline environments, the 6 Minari-Mid environments).
The qualitative conclusion — multi ≥ single on every tier — holds in both sources.

**CIFAR side:** `MISSING`. The early/middle/late block comparison is prepared as sweep 1D in
`cifar_sensitivity/OTHER_MACHINE_run_cifar_sensitivity.sh` but was run elsewhere. Block identity
of the default (stage 4 / block 1, filename tag `s3b1`) is established from code and filenames;
correction dimensions and per-block compute cost were never measured here.

---

## 10. `provenance/`

Protocol facts, each with a verified `file:line` reference, are in
`protocol_configuration_record.csv` (25 rows). Verified against source, not assumed:

* MuJoCo correction pool **4,096**, sampled **without** replacement
  (`pnc_core/gym_tasks.py:635/727/1245`, `:679/879/1311`) — with the caveat that the
  `bootfull` sweep and the multi-seed wrapper both use 10,000.
* Per-member bootstrap draws are **with** replacement, independently per member
  (`pnc_core/ensembles.py:374-375`, `:519-525`), so `bootstrap_frac` sets both the resample
  size and the number of draws and is *not* a count of distinct examples.
* CIFAR correction pool **1,024** from the training-fit split (`pnc_core/cifar_tasks.py:1230`,
  `:1281/1449`) — which **overlaps base-model training data**; only the 10% validation split
  is strictly held out.
* MuJoCo predictive distribution: moment-matched diagonal Gaussian,
  `Σ = mean_m(var_m) + Var_m(mean_m)` (`pnc_core/util.py:112-120`), NLL with a 1e-6 variance
  floor (`pnc_core/metrics.py:7-15`); OOD score = per-point total predictive variance
  (`pnc_core/util.py:256`, `:304-305`).
* CIFAR: mean of per-member **softmax probabilities** (`pnc_core/util.py:560`,
  `pnc_core/openood_eval.py:24-28`); score = predictive entropy of that mean
  (`pnc_core/openood_eval.py:24-29`, `:106`); a **single scalar** temperature applied to each
  member's logits before softmax, fit by golden-section on ID-validation NLL
  (`pnc_core/util.py:159-207`).
* Modified parameter sets for single- and multi-block P&C on both benchmarks
  (`pnc_core/ensembles.py:538/575-576/695/1651`, `pnc_core/pnc.py:246-259`).
* **Ridge-centring conflict:** `anchors.json` specifies `ridge_center = "original"` but all
  14,520 executed sensitivity rows record `ridge_center = "zero"`.

Also copied: `PROVENANCE_original.md`, `INVENTORY_original.md`, `REPRODUCTION_original.md`,
`REBUTTAL_RESULTS_original.md`, `protocol_clarifications.md`,
`NON_EXPERIMENTAL_REBUTTAL_AUDIT.md` (1,124-line code-and-manuscript audit),
`pnc_experiment_protocol.md`, `RESULTS_CATALOG.md`, and the Ant-v5 reproduction gate
(`reproduction_ant.{md,json}` — max |repro − cached| = 0.000000 across all 13 reported fields).

---

## Reproducing everything in this directory

```bash
.venv/bin/python revision_results/provenance/verify_mujoco_sensitivity.py
.venv/bin/python revision_results/provenance/sweep_classification_rules.py
.venv/bin/python revision_results/provenance/verify_scod_mujoco.py
.venv/bin/python revision_results/provenance/verify_efficiency_and_distilbert.py
.venv/bin/python revision_results/provenance/verify_layer_scope.py
.venv/bin/python revision_results/provenance/extract_cifar_from_git.py
.venv/bin/python revision_results/provenance/collect_artifacts.py
.venv/bin/python revision_results/provenance/index_large_artifacts.py
```

All are read-only with respect to the source tree: they copy and derive, never modify or
overwrite an original artifact.
