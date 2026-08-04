# INVENTORY.md — P&C harness, defaults, protocol, and reproduction targets

**Purpose.** Phase-0 reproducibility inventory for the NeurIPS-2026 rebuttal of
*"Perturb and Correct: Post-Hoc Ensembles using Affine Redundancy."* Everything below is
read from code (file:line) or from the submitted manuscript artifacts, not assumed.

- **Repo root is the harness** (package = root modules). Runner: Luigi
  (`python -m luigi --module {gym,cifar}_tasks <TaskClass> ... --local-scheduler`), with a
  thin CLI wrapper `run_experiment.py`.
- **Branch for this work:** `neurips-2026-rebuttal` (in-place; a git *worktree* would omit
  the 25 GB of untracked cached data under `results/`).
- **Environment:** `.venv/bin/python` = 3.12; JAX 0.9.1 on `CudaDevice(id=0)`;
  Torch 2.11.0+cu130 (CUDA available). GPU jobs run strictly sequentially.
- **Git commit at inventory time:** see `git rev-parse HEAD` recorded in
  `results/neurips_2026_rebuttal/PROVENANCE.md`.

---

## 0. Where the code lives

| Concern | File | Key symbols |
|---|---|---|
| P&C ensemble (gym + CIFAR) | `ensembles.py` | `PJSVDEnsemble` (176), `_ls_or_ridge_solve` (146), `_precompute_sequential_ls` (411/530), `predict` (857); CIFAR block members (1651–1782), `SWAGEnsemble` (1023), `LLLAEnsemble` (2119) |
| Gym task / CLI | `gym_tasks.py` | `GymPJSVD` (724); wrappers `AllGymExperiments` (1595+), `AllGymExperimentsMultiSeed` (1707+) |
| CIFAR tasks | `cifar_tasks.py` | `CIFARPnC` (1223), `CIFARMultiBlockPnC` (1369), `CIFARSelectPnCPaperProtocol` (1606), `CIFAROpenOODPnC` (2237), baselines `CIFARStandardEnsemble` (604), `CIFARPreActSWAG` (830), `CIFARLLLA` (1872), `CIFARPreActMCDropout` (705) |
| Conv P&C math kernels | `pnc.py` | `extract_patches` (9), `flatten_conv_kernel_to_patches` (23), `find_pnc_subspace_lanczos` (175), `_ridge_regression_solve` (246) |
| Metrics | `metrics.py` | `compute_ood_metrics` (48), `compute_nll` (7) |
| Eval drivers | `util.py` | `_evaluate_gym` (210), `_evaluate_cifar` (607), `_predictive_mean_var` (112), `_fit_posthoc_temperature` (159), `_split_data` (421) |
| OpenOOD eval | `openood_eval.py` | `evaluate_openood_cifar` (106), `_uncertainty_scores_from_logits` (12) |
| Data / OOD tiers | `data.py` | Minari loaders; OpenOOD spec `_OPENOOD_CIFAR_SPEC` (627) |
| Theory (submitted) | `neurips_draft/theory.tex` | projected-residual `R=(I-P_A)J`, local-transfer bound, Mahalanobis variance decomposition |
| Reusable diagnostics | `local_sensitivity.py`, `geometry.py`, `experiments/scripts/measure_effective_rank.py` | `jax.jvp` directional derivative; `CalibrationGeometry.distance` |

---

## 1. The P&C method as coded

### 1a. MuJoCo / gym path (`GymPJSVD`, main table method = "PJSVD-Multi-LS (Full)+Prob+VCal")
Base = deterministic MLP `hidden_dims=[200,200,200,200]`, ReLU (gym base is
**deterministic** by default: `probabilistic_base_model=False`, `gym_tasks.py:759`; the
manuscript's "+Prob" variant enables a Gaussian aleatoric head separately).

- **Perturbation scale** — param `perturbation_sizes` (a *grid*). Standalone default
  `[20,40,80,160]` (`gym_tasks.py:730`); `AllGymExperiments` grid `[2,4,8,16,32,64,128]`
  (`:1598`); `AllGymExperimentsMultiSeed` `[1,2,4,8,16,32,64]` (`:1712`). The **operative
  scale is selected per (env,seed) by best validation NLL** `nll_val` (held-out 10% of ID
  train; `util.py:264-269`), fallback `nll_id` (`json_to_tex_table.py:49-51`).
- **Perturbation rank / directions** — param `n_directions` (=K), default **40**
  (`gym_tasks.py:728`). Family `pjsvd_family` default **`"low"`** = learned bottom-singular
  subspace of the layer map via `find_pnc_subspace_lanczos` (`gym_tasks.py:921`); `"random"`
  = unit-norm Gaussian directions (`:900`); `"random_full"` = identity basis (working-tree
  only).
- **Target layer/block** — `layer_scope="multi"` (wrapper default, `:1684`):
  `perturb_indices = range(0, n_hidden, 2)` = **even hidden layers `[0,2]`** for the 4-layer
  net; each perturbed layer `j` is corrected at layer `j+1`; **corrections applied
  sequentially in ascending order** (0→correct@1, 2→correct@3; `ensembles.py:530-582`).
  **Output head is never perturbed** (tail evaluated unperturbed, `ensembles.py:695`).
  `layer_scope="first"` = single-block (perturb `[0]`, correct @1).
- **Members M** — `n_perturbations`: standalone 1000; `AllGymExperiments` **128** (`:1595`);
  MultiSeed **100** (`:1707`).
- **Correction subset (X_sub)** — `subset_size`: standalone 4096, wrappers **10000**;
  capped `min(len(id_train), subset_size)`, sampled without replacement (`:872-874`).
- **Bootstrap fraction** — `bootstrap_frac`, default **0.0** (`gym_tasks.py:774`). When
  `0<frac<1`: each member draws `size_b = max(8, int(frac·n_X))` indices **with
  replacement**, independently per member (`ensembles.py:515-521`). So "fraction" sets **both
  the resample size and the number of with-replacement draws** — it is *not* a count of
  distinct examples. `frac=0.0` ⇒ all members share the full X_sub.
- **Ridge λ** — `lambda_reg`, default **0.0** (`gym_tasks.py:770`). `_ls_or_ridge_solve`
  (`ensembles.py:165`): **λ=0 ⇒ min-norm `jnp.linalg.lstsq(h_aug, target)`** (pseudoinverse);
  **λ>0 ⇒ ridge normal equations `solve(hᵀh+λI, hᵀtarget)`** (regularizes all rows incl.
  bias). Working-tree flag `ridge_toward_orig` (default False) shrinks toward original
  weights instead of zero (inert at λ=0).
- **Correction target** = the *unperturbed* model's pre-activation at the next layer,
  computed on X_sub (`ensembles.py:542-546`). Design matrix = `[h_pert, 1]` (perturbed
  post-activation + bias col).
- **Predictive distribution / OOD score** — `predict()` returns per-member `(means,vars)`
  or `means`, shape `(M,B,D)` (`ensembles.py:857`). Aggregation `_predictive_mean_var`
  (`util.py:112`): probabilistic ⇒ law of total variance `mean(vars)+var(means)`;
  deterministic ⇒ `var(means)`. **OOD score = per-point predictive variance** (mean over
  output dims), thresholded ID-vs-OOD for AUROC (`util.py:256,304`). Default gym base is
  deterministic ⇒ score = **variance over member means (epistemic disagreement)**.
- **Storage** — only member-specific params stored: shared `v_opts`/`sigmas`, tiny
  `z_coeffs (M,K)`, one base-model copy, per-member correction weights
  (`seq_w_effs/seq_b_effs`); perturbed weights recomputed on-the-fly. **No full per-member
  model is serialized in the gym path** (`ensembles.py:134-144,584-588`). Only `all_metrics`
  JSON + npz of per-point `sq_error`/`pred_var` written (`gym_tasks.py:1214`).

### 1b. CIFAR path (`CIFARPnC` single-block; `CIFARMultiBlockPnC` multi-block)
Base = **PreActResNet18** (4 stages × 2 basic blocks; penultimate `D=512`).

- **Perturbed block** — default **stage 4, block 1** (last residual block):
  `target_stage_idx=3`, `target_block_idx=1` (`cifar_tasks.py:1232-1233`). Perturbation is
  added to **conv1's kernel** `w1_pert = w1_orig + p.reshape(...)` (`ensembles.py:1651`);
  the **correction re-solves conv2** as an affine (weight+bias) ridge regression to match the
  unperturbed block output.
- **Conv-tensor representation** — im2col patch space: inputs
  `conv_general_dilated_patches → (N·H·W, C_in·kh·kw)` (`pnc.py:9`); kernel flattened
  `w.transpose(2,0,1,3).reshape(-1, C_out)` (`pnc.py:23`). Ridge appends a bias column
  (`pnc.py:275`); solve `Θ=(H+λI)⁻¹b` (`pnc.py:249`).
- **BatchNorm handling** — **frozen/absorbed, not refit.** The correction design runs
  `bn1(running_avg) → relu → conv1(perturbed) → bn2(running_avg) → relu → extract_patches`
  (`cifar_tasks.py:133-152`); BN is inference-mode and **BN2 is folded into the conv2 ridge
  fit** (comment `:1302` "W2 ridge absorbs BN2 + perturbation"). Residual/downsample path
  and head `final_bn` unchanged.
- **Defaults (`CIFARPnC`)** — `perturbation_sizes=[10,50,100,200]`, `n_directions=10`,
  `n_perturbations=50`, `subset_size=1024`, `lambda_reg=1e-3`, `random_directions=False`
  (Lanczos bottom-K eigvecs of AᵀA), `posthoc_calibrate=False` (`:1225-1237`).
- **Calibration data** — subset drawn from **train split** `x_tr`
  (`idx=rng.choice(len(x_tr), 1024)`, `:1279`); base model trains on the *same* `x_tr`
  (identical `_split_data(val_split=0.1, seed=99)`). **⇒ correction calibration overlaps
  base-model training data.** Held-out `x_va` (10%) used only for temperature.
- **OOD score (headline)** — `predictive_entropy` of the **ensemble-mean softmax**
  (`openood_eval.py:28`, default `primary_score` `:106`). Also computed: MSP, energy,
  margin, mutual information, variation ratio.
- **Member combination** — **mean of softmax probabilities** (not mean logit;
  `util.py:560`, `openood_eval.py:24`).
- **Temperature** — a **single scalar** applied to per-member logits *before* softmax, fit
  by golden-section on **ID-val NLL** only (`util.py:159-207`). `posthoc_calibrate` default
  False in `CIFARPnC` (T=1.0), True in the paper-protocol selector (`:1626`).

---

## 2. Data, OOD tiers, and calibration↔training overlap

*(Confirmed from `data.py` / `util.py`; dims verified by loading cached npz — see §5.)*

- **MuJoCo policy-shift ladder** (submitted `experiments.tex`): ID=Expert, OOD-Near=Medium,
  OOD-Mid=Simple, OOD-Far=Random. Base dynamics model trained on **expert data only**;
  shifted regimes used strictly for evaluation, never for ensemble construction.
- **CIFAR OpenOOD groups** (`data.py:627`, `_OPENOOD_CIFAR_SPEC`): **Near-OOD** = {CIFAR-100,
  Tiny-ImageNet-200}; **Far-OOD** = {MNIST, SVHN, Textures, Places365}. **No "Mid" group** —
  `evaluate_openood_cifar` iterates exactly `near_ood`/`far_ood` (`openood_eval.py:143`).
  Confirms Priority-5 item 13.
- **Calibration↔training overlap** — YES on both benchmarks: the P&C correction subset is
  drawn from the ID/train pool that the base model was trained on (gym `id_train`; CIFAR
  `x_tr`). The only strictly-held-out split is the 10% validation used for size selection
  (gym) and temperature (CIFAR).

---

## 3. Baselines (for Priority-4 efficiency accounting)

| Baseline | CIFAR task | Notes on cost boundaries |
|---|---|---|
| Deep Ensemble | `CIFARStandardEnsemble` (604) | `n_models=5`; **reuses pretrained base checkpoints** via `requires()`; reported `train_time` = checkpoint-load only (excludes the 5 base trainings). |
| SWAG | eval `CIFARPreActSWAG` (830), train `CIFARTrainSWAGPreActResNet18` (764) | **SGD-trajectory collection is a separate task and is NOT in the eval `train_time`** (which covers load + BN refresh + build only). `swag_start_epoch=160` (240 in `run_phase0.sh`), `max_rank=20`, 50 samples, BN refresh on. |
| Laplace (LLLA) | `CIFARLLLA` (1872) | **Last-layer only**, **full dense GGN** over `fc` (513×K), posterior `inv(GGN+prior·I)`; prior added to GGN diagonal; 50 samples. |
| MC Dropout | `CIFARPreActMCDropout` (705) | `dropout_rate=0.1`, 50 stochastic forward passes. |

**MuJoCo baselines** (submitted): MC Dropout, Deep Ensemble, Subspace Inference, SWAG,
Laplace (prior sweep), plus Hybrid PnC+DE. Base MLP train ≈ tens of seconds (cheap); the
LS correction is closed-form (sub-second) — so P&C's marginal post-hoc cost is small
relative to base training. **Existing timing logs:** *(agent-3 pending — see §5).*

---

## 4. Discrepancies between manuscript and implementation (flagged)

1. **Omitted Near/Mid AUROC (the central Priority-5 concern).** The submitted MuJoCo main
   table (`neurips_draft/gym_tables.tex`) reports ID RMSE, Near/Mid/Far **NLL**, and **only
   Far AUROC** — Near-AUROC and Mid-AUROC are computed by the harness (`util._evaluate_gym`
   produces `auroc_ood_near/mid/far`) but **not shown**. These can be extracted from cached
   outputs without new training.
2. **Version drift in tables.** `neurips_draft/` (Apr 8, the submission) has *filled* PnC
   rows (e.g. Ant PJSVD Random size=8, PJSVD Low size=8); the later `gym_tables_paper.txt`
   (Apr 25) has **empty `--` PnC rows** and a different size grid ({5,10,20,50} per
   `gym_settings_appendix.txt`). Reproduction is pinned to the **Apr-8 submission
   artifacts**.
3. **Uncommitted working-tree default change (reproduction-relevant).** `hidden_dims`
   default changed `[64,64]→[200,200,200,200]` on every gym task (working tree vs HEAD).
   The submitted P&C runs used `[200,200,200,200]` **explicitly** (their filenames carry the
   `_h200-200-200-200` tag), so reproduction is safe **if `hidden_dims` is pinned**. All
   other working-tree P&C edits (`ridge_toward_orig`, `random_full`, `orthogonal_members`)
   are **inert at default flags**. Bottom line: pin `--hidden-dims 200 200 200 200`.
4. **CIFAR uses both single- and multi-block P&C** (`cifar_tables.txt`: "PnC Single Block
   scale=20", "PnC Multi Block scale=6"), whereas the rebuttal premise says CIFAR is
   single-block. Both exist; which is "headline" is resolved in `protocol_clarifications.md`.

---

## 5. Reproduction targets, data availability, and the reproduction result

### 5a. MuJoCo — cached data & dims (verified by loading npz)
- Path: `results/<env>/data_{id_train,id_eval,ood_near,ood_mid,ood_far,ood}_seed{S}_steps10000.npz`
  (keys `inputs`,`targets`). Tiers (from npz sidecar `.json`): id=`expert-v0`, ood_near=`medium-v0`,
  ood_mid=`simple-v0`, ood_far=`pure_random` (uniform action sampling; `data.py:56-78`).
- All 5 tiers present for **Ant-v5 seeds {0,1,2,3,4,5,10,42,100,200,314}** (11) and
  **Hopper-v5 seeds {0,1,2,10,42,100,200,314}** (8). Dims: **Ant 113→105**, **Hopper 14→11**.
- **Hardest vs saturated:** ood_near hardest (AUROC ≈0.72–0.75), ood_mid mid (≈0.77–0.83),
  ood_far saturated (≈0.997–0.998). (NLL magnitude is inverted: far largest.)

### 5b. Submitted P&C numbers (targets)
- **Ant-v5 (Apr-8 manuscript, `neurips_draft/gym_tables.tex`):** PJSVD Low (size=8) =
  RMSE 0.6269 / NearNLL 2.5265 / MidNLL 1.3379 / FarNLL 1.4271 / **Far AUROC 0.8962**;
  PJSVD Random (size=8) = 0.6236 / 1.4500 / 0.7631 / 1.2357 / **0.8330**. Single-seed,
  size selected by ID-NLL. **This config (grid incl. size 8) is NOT in the cached results.**
- **Ant-v5 current canonical (cached, Apr-18/25):** `pjsvd_multi_least_squares_random_
  projected_residual_prob_bf0.1_k20_n50_ps5.0-10.0-20.0-50.0_h200-200-200-200_..._seed{0,10,42,100,200}.json`.
  JSON keyed by size `"5.0".."50.0"`; fields incl. `rmse_val,nll_val,rmse_id,nll_id,
  {rmse,nll,ece,var}_ood_{near,mid,far}, auroc_ood_{near,mid,far}, aupr_*, train_time,
  eval_time, uncorrected_l2_*, corrected_l2_*`. 5-seed paper-protocol aggregate (select size
  by min nll_val): auroc_near 0.750±0.053, auroc_mid 0.825±0.044, auroc_far 0.997±0.002.
- **CIFAR-10 (submitted):** `cifar_tables.txt` is calibration-only (Acc/NLL/ECE/Brier/Temp),
  **no AUROC/FPR95** and no CIFAR OOD table in `neurips_draft/`. OOD targets live in cached
  `results/cifar10/openood_v1p5_pnc_{single,multi}_block_..._seed0_random.json` (primary
  score `predictive_entropy`; fields `id_metrics.accuracy`, `near_ood_auroc`, `far_ood_auroc`,
  `{near,far}_ood.aggregate.predictive_entropy.mean_fpr95`). Multi-block ps6 seed0:
  near AUROC 0.899 / far 0.942; single-block ps20 seed0: near 0.908 / far 0.948.

### 5c. Timing logs already present
- **CIFAR inference:** `results/cifar10/inference_cost.json` (batch 256, ms/sample, #fwd):
  DE(5) 1.34, SWAG(50) 7.33, LLLA(50) 7.68, MC-Dropout(32) 5.18, PnC-multi 9.69,
  PnC-single 7.42, Epinet(50) 1.62; MSP/Energy/Mahalanobis ≈0.6–0.7 (1 fwd).
- **MuJoCo:** each result JSON has `train_time`/`eval_time`. Ant seed0: PnC(bf0.1) train
  11.9 s / eval 0.2–0.5 s; DE(×5,vcal) 136 s; SWAG 44 s; Laplace 17 s; MC-Dropout(100) 32 s.
- Raw stdout logs: `logs/p1_baselines.log`, `logs/p2_de_m50.log`, `experiments/logs/*.log`.

### 5d. Existing MuJoCo theory-bridge & mechanism assets (Priorities 2–3 head start)
`pnc_repro/artifacts/` holds, for **11 MuJoCo envs** (Ant, HalfCheetah, Hopper, Humanoid,
HumanoidStandup, Walker2d, Swimmer, Reacher, Pusher, Inverted{,Double}Pendulum) seed0:
- `panel_c_diagnostic_<env>_seed0.csv` — columns `sensitivity_sketch` (first-order/linearized
  corrected sensitivity), `finite_disagreement` (actual finite residual), `distance_to_calibration`,
  `probe_eps`, `n_probes`, `pnc_perturbation_scale`, `target_layer`, `bootstrap_frac`,
  `correction_lambda`, `calibration_size`, per regime (id/ood). **Directly seeds Priority 2
  (linearized-vs-finite) and Priority 3 (distance-vs-disagreement).**
- `pnc_bridge_hidden_mahal_<env>_seed0.npz` — hidden-activation Mahalanobis distances.
- `pnc_bridge_panel_c_<env>_seed0_lreg*_bf*_ps32_J16_epsfrac0.01.npz` — full bridge panels.
- `pnc_repro/figures/*.txt` — aggregated Spearman tables & cross-env summaries.
(These are scalar sketches; Priority-2's vector metrics — cosine, relative error, log-log
remainder slope — still need the per-example residual *vectors*, so new probing is required,
but the tooling/《config》is established.)

### 5e. Reproduction result (Phase-0 gate)
- **Ant-v5: PASS — EXACT (bit-for-bit).** Re-ran the canonical cached config
  (`scripts/neurips_2026_rebuttal/repro_ant.py`, output redirected to
  `results/neurips_2026_rebuttal/repro/`) → **max |repro − cached| = 0.000000** across all 13
  reported fields (RMSE/NLL id+val, RMSE/NLL/AUROC near/mid/far). Selected size 10 by ID
  val-NLL (matches). Base train 12–13 s; total ≈40 s. The current evaluation pipeline
  reproduces its reported MuJoCo numbers exactly. See `repro/reproduction_ant.{md,json}`.
- **CIFAR-10: BLOCKED (cannot reproduce from existing artifacts).**
  1. **No base ResNet-18 checkpoints exist** (repo-wide search: none;
     `CIFARTrainPreActResNet18` saves `preact_resnet18_train_e300..._seed{0,1,2}.pkl`, all
     absent). Reproduction would require retraining PreActResNet18 for 300 epochs ×3 seeds
     (~hours each on the 12 GB TITAN X Pascal).
  2. **No OpenOOD datasets cached** (`openood_data/` absent) — needs downloading +
     preprocessing CIFAR-100, TinyImageNet-200, MNIST, SVHN, Textures, Places365.
  3. No cached CIFAR train/test arrays either.
  → Decision required before committing GPU-hours + downloads (see gate report / next steps).

### 5f. Flagged discrepancies (see also §4)
- **Manuscript↔cached config drift (Ant):** Apr-8 submitted numbers (size=8, Far AUROC
  0.833/0.896) are from a config not present in the cached results; the current canonical
  config (bf0.1, grid {5,10,20,50}) gives Far AUROC ≈0.998. Not a pipeline error — a
  different experimental configuration. The pipeline reproduces the cached config exactly.
- **Omitted Near/Mid AUROC** in the submitted MuJoCo table (only Far AUROC shown) — the
  harness computes near/mid AUROC (present in every cached JSON); trivially extractable.
