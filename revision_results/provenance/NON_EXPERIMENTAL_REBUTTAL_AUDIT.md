# NON_EXPERIMENTAL_REBUTTAL_AUDIT.md

Audit of the P&C codebase and manuscript (`12747_Perturb_and_Correct_Post.pdf`, 33 pp)
against the reviewer question list. Answers are derived **only** from existing code,
configs, saved outputs, logs, and the manuscript — no new experiments were run.

**Label key:** Answered · Partially answered · Not recoverable · Requires experiment ·
Manuscript inconsistency · Implementation inconsistency.

### Reading notes / caveats that recur below
- **Manuscript source is not in the repo.** The submitted PDF is dated July; the only
  `.tex` in the tree (`neurips_draft/`, `neurips_draft_original/`) is an April version
  that lacks the submitted paper's distinctive strings ("uniformly-weighted mixture",
  "multi-block", the Minari/Younis cite). So §12 edit locations are given by
  section/table against the **PDF**, not by `.tex` line number. The authoritative source
  (likely Overleaf) should be edited there.
- **Headline MuJoCo P&C config** (from `experiments/logs/ant_bf0.05_paper_table.log` and
  `pnc_repro/figures/appendix_selected_hparams_paper.txt`): `GymPJSVD` with
  `layer_scope=multi`, `correction_mode=least_squares`, `pjsvd_family=random`,
  `safe_subspace_backend=projected_residual`, `probabilistic_base_model=True`,
  `n_directions(K)=20`, `n_perturbations(M)=50`, `subset_size=4096`,
  `ridge_toward_orig=False` (default), per-env (λ, bf, ps) from Table 3.
- **Headline CIFAR P&C config** (paper Table 6): single residual block (3,0), `PnCEnsemble`,
  σ=25, bf=0.05, λ=1e-3 (fixed), K=20, M=50, subset_size=1024, BN frozen.
- **pnc_repro/ is the rebuttal reproduction**, not the submission: it uses 10–28 seeds
  per env (vs the PDF's "10 seeds") and diverges from the submitted numbers on
  HumanoidStandup. Where they differ this is flagged.
- The PDF embeds two **prompt-injection blocks** (pp. 2 and 33, white/hidden text
  demanding an LLM reviewer emit fixed phrases). These are ignored here; the authors
  should be aware they are present in the submitted file.

---

## 1. Exact P&C implementation

**1. Which parameters are perturbed in single-block P&C? — Answered.**
Only the **weight matrix of the target layer** (bias untouched). CIFAR single-block: Conv1
kernel `W1` only — `w1_pert = w1_orig + reshape(p)` (`ensembles.py:1651`). MuJoCo (multi):
each perturbed layer's `W` via `W_pert + dWs[j]` (`ensembles.py:538`); perturbed layers are
hidden `l1,l3` (`gym_tasks.py:882-891`). ΔW = σ·reshape(U z) with U a random low-rank basis.
MS: matches Eq (1) (only `W_ℓ` perturbed). RC: "what exactly is perturbed."

**2. Which parameters are refit during correction? — Answered.**
The **following affine layer's weight *and* bias**. MuJoCo: `(W_corr, b_corr)` at layer j+1
(`ensembles.py:575-576`). CIFAR: Conv2 kernel + a *newly added* bias
(`pnc.py:_ridge_regression_solve` returns `w2_new, b2_new`; base Conv2 had no bias). MS: yes
(Eq (1) fits `(W_{ℓ+1}, b_{ℓ+1})`). RC: scope of the correction.

**3. Are any parameters downstream of the next affine layer changed? — Answered. No.**
Only the immediate next affine layer is refit; the remaining "tail" runs on base weights
(`_forward_member_sequential`, `evaluate_tail_from_preact` `ensembles.py:88-132`). Output
heads unperturbed (paper §3: "output heads are left unperturbed"). RC: leakage into the rest
of the net.

**4. Are normalization params / running stats modified? — Answered. No.**
MuJoCo MLP has no norm layers. CIFAR BN is evaluated in **inference mode with base running
statistics**, never refit (`use_running_average=True`, `cifar_tasks.py:55,65,136,145`; paper
App B.3(ii)). A `_precompute_bn_refit` mode exists (`ensembles.py:590`) but is **not** used
for headline P&C. MS: stated in App B.3(ii). RC: hidden BN leakage / stats shift.

**5. Are biases perturbed, corrected, both, or neither? — Answered.**
**Corrected, not perturbed.** The perturbation hits `W` only; the correction refits the next
layer's bias (CIFAR *adds* a bias to a previously bias-free Conv2). MS: consistent with
Eq (1). RC: bias handling.

**6. Does the implementation match Equation (1) exactly? — Partially / Implementation inconsistency.**
Structure matches (perturb `W_ℓ`; refit next `(W,b)` by ridge-LS onto the *original*
next-layer pre-activation). **One deviation:** Eq (1) centers the ridge penalty on
`(W_{ℓ+1}, b_{ℓ+1})`; the **MuJoCo** headline centers it on **zero** (`ridge_toward_orig=False`,
`ensembles.py:215,396,566-574`). **CIFAR** matches Eq (1) exactly (centers on the original
kernel, `pnc.py:_ridge_regression_solve`). Effect is small (λ≤0.01) but the FC/main-text
equation does not match the MuJoCo runs' regularizer center. See Q13, Q118–120.

**7. Correction target: next-layer preactivation / postactivation / logits / output? — Answered.**
The **original next-layer pre-activation**. MuJoCo: `target = ref_h @ W_corr_orig + b_corr_orig`
where `ref_h` is the *unperturbed* base activation (`ensembles.py:540-546`); for the last
block the target is the original post-activation feeding the heads
(`gym_tasks.py:1028`). CIFAR: original Conv2 output `T_orig` (`pnc.py`, `cifar_tasks.py`
`t_orig_chunks`). Not logits/final output. MS: yes (Eq (1)/(3)/(30)). RC: what the LS fits to.

**8. Solved independently per member? — Answered. Yes.**
Per-member loop, each solving its own LS system (`ensembles.py:525-579`;
`ensembles.py:1649-1658` CIFAR). RC: member independence.

**9. Closed-form LS, iterative, or other? — Answered. Closed-form.**
λ=0 → minimum-norm pseudoinverse via `jnp.linalg.lstsq(rcond=None)`; λ>0 → ridge normal
equations `jnp.linalg.solve(HᵀH+λI, ·)` (`ensembles.py:165-173`). No iterative optimizer.

**10. What solver? — Answered.**
`jnp.linalg.lstsq` (SVD min-norm) at λ=0; `jnp.linalg.solve` (LU on normal equations) at λ>0
(`ensembles.py:167-172`). CIFAR: chunk-accumulated normal equations then `jnp.linalg.solve`
(`pnc.py:262-289`).

**11. Numerical fallback if singular / ill-conditioned? — Answered.**
λ=0 relies on `lstsq`'s SVD min-norm solution (rank-deficiency handled, `rcond=None`); λ>0
adds `λI` guaranteeing nonsingularity. **No explicit try/except pseudoinverse fallback and no
condition-number guard** in the headline path. (Condition numbers are computed only in the
separate self-tuning tooling, `experiments/scripts/selftune_lambda.py`.) RC: robustness of the
solve.

**12. Ridge always / optional / disabled in submitted runs? — Answered. Optional, enabled.**
Default λ=0, but all submitted headline configs use λ>0: MuJoCo per-env λ∈{1e-4, 0.01}
(Table 3), CIFAR λ=1e-3 fixed (Table 6). So ridge is (weakly) enabled everywhere in the
submission.

**13. Is the ridge centered on the original affine params (as the paper states)? — Implementation inconsistency (MuJoCo).**
**CIFAR: yes** — `_ridge_regression_solve` solves for the *delta* and adds it to `w2_orig`, i.e.
`min‖YW−T‖²+λ‖W−W2_orig‖²` (`pnc.py:279-289`), matching Eq (30). **MuJoCo: no** — headline
`ridge_toward_orig=False` shrinks toward **zero**. Confirmed empirically: **0** headline-family
files carry the `_ridgeorig` token; every `_ridgeorig` result is a separate `_vcal` exploratory
run. So the main-text Eq (1)/(3) shrink-toward-Θ statement holds for CIFAR but not for the
MuJoCo runs.

**14. Does the code regularize the bias term as well as the weight? — Answered. Yes.**
`_ls_or_ridge_solve` adds `λI` over *all* rows including the ones/bias row, and the prior
(when used) includes the bias row (`ensembles.py:162-170,402`). CIFAR: the ones column is the
first feature (index 0) and is inside the ridge, shrunk toward zero-bias (Θ₀ bias = 0),
matching Eq (30). MS: Eq (1) penalizes both `‖W−W_{ℓ+1}‖²` and `‖b−b_{ℓ+1}‖²`.

**15. Are perturbation directions normalized before the scale is applied? — Answered. Yes.**
"random" family: unit-normalized Gaussian directions (`gym_tasks.py:903-904, 972-973`
`rand_dirs /= ‖·‖`). "low"/Lanczos and CIFAR `find_random_directions`: orthonormal. Magnitude
enters via `z∼N(0,I_K)` scaled by σ (`_scale_coefficients_with_member_radii`
`ensembles.py:70-86`). (`random_full` uses an identity basis and is not the headline.)

**16. Basis sampled once per experiment / layer / seed / member? — Answered.**
Once **per layer, per seed**: multi-block draws a per-layer basis with `RandomState(seed+li)`
(`gym_tasks.py:971`); one basis is reused for all members within a run. RC: source of member
diversity.

**17. Same basis shared across members? — Answered. Yes.**
`v_opts` is stored per layer and reused for every member; only the coefficient `z` varies
(paper §3; `gym_tasks.py:971-975, 1002-1007`).

**18. Are member coefficient vectors sampled independently? — Answered. Yes.**
`z^{(m)}∼N(0,I_K)` i.i.d. per member (`_sample_member_latents` `gym_tasks.py:1054`; CIFAR
`rng_z.normal(size=(M,K))` `cifar_tasks.py:445`). Optional `antithetic_pairing` /
`orthogonal_members` reduce independence but are off by default.

**19. Is "perturbation rank" the # basis directions, a matrix rank, or something else? — Answered.**
The **number of random basis directions K** (`n_directions`); MuJoCo K=20, CIFAR K=20. ΔW =
σ·reshape(Uz) is a low-rank direction drawn from a K-dimensional subspace; "rank" = K, not a
computed matrix rank.

**20. Are seeds controlled separately for base training / directions / bootstrap / eval? — Partially.**
Base training, direction basis, and coefficient sampling all derive from one master `seed`
(`Rngs(params=seed)`; directions & latents both `RandomState(seed+li)`; `X_sub` via global
`seed_everything(seed)` then `np.random.choice`). The train/val split uses a **fixed** seed=99
(`util.py:421-423`), independent of the experiment seed. `bootstrap_seed` and
`member_radius_seed` are **separate** parameters (default 0). Evaluation is deterministic (no
eval seed). So: not fully independent — base/direction/coeff share the master seed.

---

## 2. Multi-block versus single-block P&C

**21. Which layers/blocks are perturbed in MuJoCo? — Answered.**
Hidden layers **l1 and l3** (0-based indices 0,2) of the 4×200 MLP, with corrections at l2 and
l4: `perturb_indices = range(0, n_hidden, 2) = [0,2]`, `corr = idx+1` (`gym_tasks.py:884-888`).
Two perturb–correct pairs; output/variance heads unperturbed.

**22. In what order are the MuJoCo blocks perturbed/corrected? — Answered.**
Ascending network order (l1 then l3), each perturbed and corrected before the next
(`ensembles.py:525-582`, loop `j in range(n_layers)`).

**23. Does each later correction operate on a network that already contains earlier perturbations+corrections? — Answered. Yes.**
The loop threads the corrected output forward: `h = activation(h_pert @ W_corr + b_corr)`
(`ensembles.py:582`) becomes the next block's perturbation input. RC: chaining semantics.

**24. Is each correction fitted against the original base preact or the previously modified output? — Answered. Original base preactivation.**
`target = all_orig_h[corr_idx] @ W_corr_orig + b_corr_orig`, where `all_orig_h` is a clean walk
through the **unperturbed** base model (`ensembles.py:487-500, 540-546`). The *design matrix*
uses the perturbed/corrected upstream, but the *target* is the original. Matches paper §3
("against the original pre-activations") and App B.6 (CIFAR).

**25. Which block is used for CIFAR-10? — Answered / Manuscript inconsistency.**
A **single residual block**. Paper Table 6 says the selected block is **(3,0)** from grid
`{1,2,3}×{0,1}`. But the code default is `target_stage_idx=3, target_block_idx=1`
(`cifar_tasks.py:1232-1233`, comments "stage4 index 3 / block1 index 1") and **every local result
filename uses `s3b1`** (e.g. `openood_v1p5_pnc_single_block_vcal_s3b1_k20_n50_ps25.0...`) — i.e.
stage 4 / block 1, the **last** residual block. So the block actually run (s3b1 = (3,1), 512-ch)
does **not** match the paper's stated "(3,0)". Reconcile which block the submitted CIFAR numbers
used (the two differ in channel width if (3,0) means stage 3). Perturb Conv1, correct Conv2, BN
frozen.

**26. Why multi-block for MuJoCo and single-block for CIFAR? — Not recoverable (rationale).**
Factually: MuJoCo `layer_scope=multi` (l1,l3); CIFAR uses single-block P&C. **No comment,
commit message, or note states the reason.** Plausible (small MLP needs multiple layers for
diversity; conv multi-block is costlier) but undocumented. RC: design-choice justification.

**27. Is the choice documented anywhere? — Answered. No.**
The candidate machinery exists (CIFAR `candidate_block_modes`; MuJoCo `layer_scope` choices)
but nothing documents *why* the two benchmarks differ.

**28. Was the block choice fixed, ID-val-selected, OOD-selected, or manual? — Answered.**
CIFAR: **ID val-NLL selected** from the 6-position grid (Table 6, coordinate descent; paper
D.1/D.2 "selected using ID validation"). MuJoCo: `layer_scope=multi` is **fixed a priori** (not
swept in the headline). No OOD-based selection anywhere.

**29. Were alternative blocks evaluated previously? — Answered. Yes.**
CIFAR: all 6 grid positions were swept (Table 6). MuJoCo: both `layer_scope=first` and `multi`
exist in `results/` filenames.

**30. Are unpublished layer-selection results stored? — Partially.**
MuJoCo `first`-scope result JSONs exist in `results/*/pjsvd_first_*`. CIFAR block-sweep results
are produced by the grid but CIFAR runs on a separate machine (see project memory), so local
availability is limited. RC: whether the selection is auditable.

---

## 3. Calibration data and data leakage

**31. MuJoCo calibration split? — Answered.**
A random **4096-example subset (without replacement) of `id_train`** — the expert-Minari
transitions (`gym_tasks.py:872-874`, `subset_size=4096`). Not the val split, not OOD.
(`posthoc_calibrate=False` ⇒ no `_vcal`; calibration is id_train, not validation.)

**32. CIFAR calibration split? — Answered.**
A random **1024-example subset (without replacement) of `train_fit_inputs`**
(`cifar_tasks.py:402-405`; Table 6 `subset_size=1024`). `train_fit` is the training remainder
after removing the 10% val-select and 10% BN-refresh splits (`_make_cifar_protocol_splits`,
`util.py:435-491`).

**33. Is the calibration set part of the base model's training set? — Answered. Yes.**
MuJoCo: `X_sub` is sampled from `id_train`, the base training pool (base trains on 90% of
id_train; `X_sub` overlaps it). CIFAR: the calibration subset comes from `train_fit`, which the
base ResNet trained on. Paper explicitly permits this (§3: "D_cal … which may be the training
data"). MS: consistent. RC: leakage — this is intended, not a bug.

**34. Is it held out from base-model training? — Answered. No** (see Q33). Consistent with the paper.

**35. Is it the same set used for validation / temperature scaling? — Answered. No.**
MuJoCo has no temperature scaling; validation (`x_va`, 10% of id_train) is a different split
from `X_sub`. CIFAR: temperature is fit on the 10% **val-select** split, which is **disjoint**
from the P&C calibration subset (`train_fit`). RC: double-use of a split.

**36. How many calibration examples per benchmark? — Answered.** MuJoCo 4096; CIFAR 1024.

**37. Sampled once and shared across seeds, or resampled? — Answered.**
Resampled per seed (`np.random.choice` after `seed_everything(seed)`), so each seed uses a
different calibration subset. Within a run, one `X_sub` is shared across members (then
per-member bootstrap if bf>0).

**38. With or without replacement? — Answered.**
Base subset: **without** replacement (`replace=False`). Per-member bootstrap resample (bf>0):
**with** replacement (`ensembles.py:520`).

**39. What does "bootstrap fraction" mean operationally? — Answered.**
The fraction of `|X_sub|` each member resamples **with replacement** for its own LS fit:
`size_b = max(8, int(bf·n_X))` (`ensembles.py:519`). bf=0.1 ⇒ each member fits on a
0.1·|X_sub| bootstrap draw.

**40. If bootstrap fraction is 100%, does each member get a full-size sample with replacement? — Manuscript inconsistency.**
**No.** The bootstrap branch triggers only for `0.0 < bf < 1.0` (`ensembles.py:516`); **bf=1.0
falls through to the non-bootstrap branch** (all members share the full `X_sub`, no
resampling). Paper §3 ("sampling with replacement at the full calibration size recovers
nontrivial decorrelation") implies bf=1.0 = full bootstrap; the code does the opposite at
exactly 1.0. Headline bf∈{0.05…0.3}, so results are unaffected, but the sentence is misleading.

**41. Can the same example appear multiple times in a member's subset? — Answered. Yes** when
bootstrap is active (with replacement).

**42. Does every member use an independently sampled correction subset? — Answered.**
Yes when bf∈(0,1) — per-member indices `[rng_boot.choice(...) for _ in range(M)]`
(`ensembles.py:520`). When bf=0, all members share `X_sub`.

**43. Is there an option for all members to share one subset? — Answered. Yes** — bf=0.0 (the
default). Headline uses bf>0, so headline members use independent bootstrap subsets.

**44. Is any OOD/Near/Mid/Far data used in correction / HP selection / temp scaling / early stop / model select? — Answered. No, to all five.**
Correction uses `X_sub⊂id_train` (`gym_tasks.py:874`); MuJoCo selection is val-NLL on
`x_va⊂id_train`; CIFAR selection is val-NLL post-temperature on the 10% ID val; temperature is
fit on ID val; there is no early stopping on OOD. OOD tiers enter only `_evaluate_gym` at final
test (`util.py:289-318`). Paper §5.3: "No OOD data was used for construction, selection, or
calibration." ✓

**45. Assertions/split checks preventing OOD entering calibration/tuning? — Partially.**
No explicit `assert`, but structural separation guarantees it: OOD regimes are loaded only into
the eval dictionary and read only for final metrics; no code path feeds `ood_*` into
calibration or selection. RC: reviewers may want an explicit guard — adding one is cheap.

**46. Does "hyperparameters selected using ID data only" match the implementation? — Answered. Yes.**
MuJoCo val-NLL on ID `x_va`; CIFAR val-NLL post-temp on ID val (paper C.2, D.1). Matches the
setup claim (p8). ✓

---

## 4. Hyperparameter-selection protocol

**47. Submitted values — Answered.**
| knob | MuJoCo (headline) | CIFAR (headline) |
|---|---|---|
| perturbation scale ps/σ | per-env {8,16,32} (Table 3) | 25 |
| perturbation rank K | 20 | 20 |
| target layer/blocks | multi (l1,l3) | single block (3,0) |
| ensemble size M | 50 | 50 |
| correction subset size | 4096 | 1024 |
| bootstrap fraction | per-env {0.05,0.1,0.2,0.3} | 0.05 |
| ridge λ | per-env {1e-4,0.01} | 1e-3 (fixed) |

**48. Stored in committed config files? — Partially.**
Selected values: `pnc_repro/figures/appendix_selected_hparams_paper.txt` + paper Tables 3/6.
Grids/launch: `experiments/scripts/run_bootstrap_ls_lreg_paper_table.sh`,
`scripts/neurips_2026_rebuttal/repro_ant.py`. Task defaults: `gym_tasks.py`/`cifar_tasks.py`.
No single YAML — config = luigi params + run scripts.

**49. Same values across all MuJoCo envs? — Answered. No** (per-env, Table 3). Fixed across
envs: K=20, M=50, subset 4096, scope=multi, family=random.

**50. Different for Ant/Hopper/HalfCheetah? — Answered. Yes.**
Ant (1e-4, 0.1, 8); Hopper (1e-4, 0.3, 32); HalfCheetah (1e-4, 0.3, 32) — Table 3.

**51. Different across CIFAR seeds? — Answered. No** — one selected config across the 5 seeds
(Table 6).

**52. Objective used to select each hyperparameter? — Answered.**
Validation **NLL** (MuJoCo: min val-NLL, ID-NLL fallback; CIFAR: min val-NLL after temperature
scaling). Paper C.2, D.1.

**53. Based only on ID NLL / RMSE / accuracy / conditioning? — Answered.**
ID validation **NLL** only (not RMSE, accuracy, or conditioning).

**54. Coordinate descent manual or automated? — Answered.**
CIFAR: **coordinate descent** in Table-6 row order (paper D.2), on top of the
`itertools.product` grid machinery (`cifar_tasks.py:273`). MuJoCo: **full cross-product** of
{ps}×{bf}×{λ} then val-NLL argmin (paper C.2).

**55. Order hyperparameters were tuned? — Answered.**
CIFAR: block → direction method → K → M → ps → λ (Table-6 row order; K/M/λ held fixed, so
effectively block → ps → bf). MuJoCo: cross-product (no order).

**56. How many values per hyperparameter? — Answered.**
MuJoCo: ps 3, bf 4, λ 2 ⇒ 24 configs/env. CIFAR: block 6, σ 3, bf 3, λ fixed (1), K fixed,
M fixed.

**57. Are all tried configs and their results available? — Partially.**
MuJoCo: full sweep JSONs per (env,seed,config) exist in `results/` for the 4 local envs
(Ant, HalfCheetah, Hopper, Humanoid). The other 7 envs' raw JSONs are absent (see §11). CIFAR:
sweep on the other machine — partial locally.

**58. Were unstable/unsuccessful configs retained in logs? — Partially.**
The full sweep JSONs (all configs, not just the selected) are retained for the 4 local envs;
`logs/`, `gym_tables_*.txt`, and the `VERDICT_*.md` scoping artifacts retain additional
exploratory runs.

**59. Were any hyperparameters selected after observing Near/Mid/Far? — Answered. No** —
selection is ID val-NLL (Q52). (The `VERDICT_*.md` OOD-aware-λ studies are separate scoping,
not the headline selection.)

**60. Is "plug-and-play" used, and is it supported? — Answered.**
**Not used** in the submitted PDF (grep clean). The actual procedure is a per-env val-NLL grid
search (24 configs) — *not* zero-tuning — so the term would be unsupported if introduced. No
current manuscript issue; just don't add it.

---

## 5. Ensemble prediction and OOD inference rules
*(from the evaluation/metrics audit)*

**61. MuJoCo predictive distribution? — Answered.** Per-member Gaussian: `ProbabilisticRegressionModel`
has two heads, `mean` and `var = softplus(var_logits)+1e-6` (`models.py:207-219`).

**62. Does each member predict a Gaussian mean and variance? — Answered. Yes** (`models.py:216-218`;
`predict` stacks `(means, vars)` each `(M,B,D)`, `ensembles.py:863-867`).

**63. How are members combined? — Answered. Law of total variance (moment matching).**
`_predictive_mean_var` (`util.py:112-120`): `mean = mean_m(means)`;
`var = mean_m(vars) + var_m(means)`. Uniform weighting. Matches the Deep-Ensembles recipe.

**64. NLL = exact mixture likelihood or moment-matched Gaussian? — Answered. Moment-matched single Gaussian.**
`compute_nll` is a single-Gaussian NLL (`metrics.py:7-15`) applied to the moment-matched
`(mean, var)` (`util.py:247,251`). Consistent with "as in Deep Ensembles."

**65. Is variance decomposed into aleatoric/epistemic? — Answered.**
Computed (`util.py:116`: aleatoric=`mean(vars)`, epistemic=`var(means)`) but **summed, not
reported separately** in the main eval. (A separate script splits them.)

**66/67. MuJoCo AUROC uncertainty score? — Answered. Total predictive variance.**
`pred_var_per_pt = mean over output dims of (mean(vars)+var(means))` (`util.py:256`), passed to
`compute_ood_metrics` (`util.py:304-305`). It is **total** (aleatoric+epistemic) variance —
**not** epistemic-disagreement-only, and not NLL/entropy/MI. (Note: this corrects an internal
note in `inventory.md` that called it epistemic disagreement.)

**68. CIFAR-10 P&C OOD score? — Answered. Predictive entropy of the mixture.**
`mean_probs = probs.mean(0); score = −Σ mean_probs·log mean_probs`
(`openood_eval.py:28-29`); `primary_score="predictive_entropy"` (`openood_eval.py:106-107`;
`make_ood_paper_table.py:32-34`; confirmed in saved JSON).

**69. Are member logits averaged before softmax? — Answered. No** — softmax is applied
per-member first (`openood_eval.py:26`).

**70. Are member probabilities averaged after softmax? — Answered. Yes** (`openood_eval.py:28`).

**71. Is predictive entropy from the mixture probabilities? — Answered. Yes** (`openood_eval.py:29`).

**72. Mutual information / expected entropy used? — Answered.** Computed (`openood_eval.py:53-57`)
but **not** the reported P&C score (secondary columns only).

**73. Is the P&C OOD score identical across Near and Far? — Answered. Yes** — same
`_uncertainty_scores_from_logits` and same primary key for both families
(`openood_eval.py:143-166,192-193`).

**74/75. Temperature scaling — where and shared? — Answered.**
A **single shared scalar T** is applied to **each member's logits before softmax**, so it flows
into the mixture probs and thus the OOD entropy score (`openood_eval.py:21-29`; `util.py:159-207`).
Not per-member, not on averaged logits only.

**76. Which split fits the temperature? — Answered. The 10% ID validation split** (`val_model_select`,
`cifar_tasks.py:1624,1660`; metadata `temperature_fit_split: id_validation_only`). Disjoint
from test.

**77. Same inference rule for P&C and Deep Ensemble? — Answered. Yes** — both go through the
identical aggregation code (MuJoCo `_evaluate_gym`/`_predictive_mean_var`; CIFAR
`evaluate_openood_cifar` → `predictive_entropy`).

**78. Does it match "uniformly-weighted mixture combined as in Deep Ensembles"? — Answered. Yes.**
Uniform weighting is the `jnp.mean(...,axis=0)` (`util.py:115-116`, `openood_eval.py:28`); NLL
uses the DE moment-matched Gaussian. Nuance (not a contradiction): the *reported score* differs
by domain — total variance (MuJoCo) vs predictive entropy (CIFAR).

**79. Where to add the missing CIFAR OOD inference-rule definition? — Manuscript inconsistency (omission).**
Currently defined only in code (quotable: `openood_eval.py:12-18,28-29`). Add to §5.4 / App B:
"per-member softmax at shared temperature T (fit on 10% ID val) → average to mixture probs →
score = Shannon entropy of the mixture; T also scales the OOD score."

---

## 6. Near, Mid, and Far reporting
*(from the evaluation/metrics audit)*

**80. Are Near/Mid MuJoCo results in saved files? — Answered. Yes.**
`_evaluate_gym` writes `nll_ood_{near,mid,far}`, `auroc_ood_{near,mid,far}`, `rmse_/var_/ece_`
per tier into each `results/<env>/*.json` (verified in a repro JSON). Present only for the
4 envs that have local JSONs.

**81. Available for every method/env/metric/seed? — Partially.**
Complete for all methods/seeds in the 4 envs with JSONs. For the 11-env paper set, only
*summaries* exist; **Mid is `--` for 5 envs** (Swimmer, Reacher, Pusher, Inverted{,Double}Pendulum —
those Minari datasets lack a Mid level). Near/Mid NLL/AUROC are not in the per-env appendix at all.

**82. Same eval script for Near/Mid and Far? — Answered. Yes** — all tiers in one loop
(`util.py:289-318`; regimes from `data.py:342-370`).

**83. Why only Far in the main table? — Not recoverable (rationale).** No code/notes rationale.
(Note: Table 1 actually reports ID RMSE **and** Far NLL/AUROC/Spearman — "ID + Far", not "Far
only".)

**84. Documented as space / severity / other? — Answered. Not documented** — the paper only
says per-env results are in the appendix.

**85. Does the appendix contain complete Near/Mid results? — Answered. No.**
Spearman Table 5 = ID/Near/Mid/Far (Mid `--` for 5 envs); per-env Table 4 = **Far only**
(ID RMSE, Far NLL, Far AUROC). No Near/Mid NLL/AUROC per env anywhere.

**86. Differences between aggregate claims and per-env results? — Answered. Yes.**
Envs where P&C is **not** best on Far NLL: HumanoidStandup, InvertedDoublePendulum, Ant (SWAG
wins); not best on Far AUROC: Swimmer, InvertedDoublePendulum. So "best on Far" holds on
aggregate rank but not per-env everywhere. RC: over-broad dominance wording (see §11).

**87. Does the CIFAR/OpenOOD config define only Near and Far? — Answered. Yes** — keys
`near_ood`/`far_ood` only (`data.py:626-651`; `openood_eval.py:143-146`).

**88. Any "Mid" OOD group in CIFAR code? — Answered. No.**

**89. Exact Near/Far dataset names (CIFAR-10)? — Answered.**
Near = {CIFAR-100, Tiny-ImageNet-200}; Far = {MNIST, SVHN, Textures, Places365}
(`data.py:626-637`). Matches OpenOOD v1.5.

**90. Inherited from OpenOOD config or manually recreated? — Answered / flag.**
**Manually recreated** — a hardcoded dict `OPENOOD_CIFAR_BENCHMARKS` (`data.py:626-651`);
datasets sorted into `near_ood/far_ood` dirs by `scripts/prepare_openood_data.py`. No OpenOOD
imglist/config is parsed to define the groups. Worth stating in the reproducibility text.

---

## 7. Existing mechanism diagnostics
*(from the figure/mechanism audit)*

**91. Code producing Figure 2? — Answered.**
`experiments/scripts/make_pnc_teaser_composite.py` (`main()` L193 → `pnc_teaser_ant_composite.{png,pdf}`).
Panel A frontier via `collect_method_points` (imported from `plot_pnc_correction_frontier.py`),
x=`rmse_id`, y=`nll_ood_far`, seeds {0,10,42,100,200}, selected by `nll_val`. Panels B/C bars
read `experiments/figures/pnc_subspace_analysis_ant_size20_bf0.1_data.json`, produced by
`experiments/scripts/analyze_perturbation_subspaces.py`.

**92. Code producing Figure 3? — Not recoverable (generator).**
The *current* Figure 3 is `pnc_repro/figures/pnc_bridge_Ant-v5.pdf`; its generating `.py` is
**not in the tree or git history** (grep for its unique artifact tokens matches no `.py`;
`git log -S` empty). A committed **predecessor**, `scripts/plot_ant_bridge_q123_bf0.1.py`,
produces panels A/B identically and a *different* panel C; the current panel-C definitions live
in `pnc_repro/figures/notes.txt`. The saved per-example artifacts (below) let the figure be
described exactly, but the script must be re-authored to regenerate it.

**93. Same base-model checkpoint for Fig 2 and Fig 3? — Answered. No.**
No gym base model is ever serialized; each figure retrains its own (Fig 2 at 5000 steps; Fig 3
predecessor at default 2000 steps). Same architecture/seed determinism, different training runs.

**94. Same calibration split? — Answered. No.**
Fig 2 B/C: 4096-subset of id_train. Fig 3: Mahalanobis μ,Σ from the **full** id_train; LS
calibration size 1024. Fig 2 aggregates 5 seeds; Fig 3 is seed 0 only.

**95. Same target layer and P&C hyperparameters? — Partially.**
Same: even-index multi-layer (l1,l3), K=20, M=50, [200×4] ReLU prob base. Differ: **ps** (Fig 2
= 20, Fig 3 = 32); **λ** (Fig 2 min-norm/bootstrap, Fig 3 λ=0.1); **measured layer** (Fig 2 Δh
at layer 1; Fig 3 Mahalanobis at layer 2); Fig 3 adds probe params J=16, ε=0.01·ps.

**96. How ‖Δh‖ is computed? — Answered.**
`analyze_perturbation_subspaces.compute_gamma_and_dy` (L280): per-example Euclidean norm in
hidden space `sqrt(Σ_dim (h_member−h_base)²)`, then mean over examples ⇒ one scalar per member.
Hidden = layer-1 post-activation.

**97. How ‖Δy‖ is computed? — Answered.**
Same function (L281): per-example L2 output-space distance to base `sqrt(Σ_out (y_member−y_base)²)`,
mean over examples, per member.

**98. Averaged over examples / members / seeds? — Answered. All three.**
mean over examples → median over members (L559-563) → median over seeds (`aggregate_seeds`
L588-590); the "median" field is what the teaser plots.

**99. How is Mahalanobis distance computed? — Answered.**
`plot_ant_bridge_q123_bf0.1.py:772-781`: `μ=mean(h_train)`, `Σ=cov(h_train)` (full D×D),
`λ=1e-4·mean(diag Σ)`, `Σ_reg=Σ+λI`, `Σ⁻¹=inv(Σ_reg)`, `d=sqrt(max((H−μ)ᵀΣ⁻¹(H−μ),0))`.
Confirmed numerically in the saved npz.

**100. Covariance form? — Answered.** Full-rank empirical covariance, diagonally
ridge-regularized, then plain `np.linalg.inv` — not diagonal-only, not Ledoit-Wolf, not
pseudoinverse.

**101. Regularization in the Mahalanobis calc? — Answered.** Diagonal ridge `λ=1e-4·mean(diag Σ)`
(≈1.2e-5 for Ant); no shrinkage coefficient.

**102. Original or perturbed representation space? — Answered. Original (unperturbed) hidden
space** at layer 2 (no perturbation applied when computing features).

**103. "Prediction disagreement to base" definition? — Answered.**
Predecessor plots `sqrt(pred_var)` (total predictive spread), not `‖mean_member−base‖` (the
latter is computed and saved as `mean_member_l2_from_base` but *not* plotted). Current Fig 3
uses `finite_disagreement` D(x) (the finite P&C ensemble's residual disagreement to base),
saved per point.

**104. Is the regime-controlled slope β₁ from an existing regression script? — Answered.**
Predecessor `_panel_b_stats` (L851-911): OLS with **regime fixed effects** via `np.linalg.lstsq`.
Current figure absent, but `notes.txt` documents the identical form; SE and t=β₁/SE reported.

**105. Covariates in that regression? — Answered.**
Intercept; the continuous predictor (`log₁₀(Mahalanobis)` in panel B; `log₁₀(ŝ+ε)` in panel C);
regime dummies (near/mid/far, ID = dropped baseline). No others.

**106. Per-example diagnostics saved, or only aggregates? — Answered.**
Fig 3: **per-example saved** — `pnc_repro/artifacts/panel_c_diagnostic_Ant-v5_seed0.csv`
(4000 rows: `sensitivity_sketch`, `finite_disagreement`, `distance_to_calibration`) plus npz.
Fig 3 predecessor also saved per-sample CSVs. Fig 2: **only aggregates** (per-seed scalar
medians/IQR).

**107. Have equivalent diagnostics been run on Hopper/HalfCheetah/CIFAR? — Answered.**
**MuJoCo: yes for all 11 envs** — `panel_c_diagnostic_<env>_seed0.csv`,
`pnc_bridge_hidden_mahal_<env>_seed0.npz`, and `pnc_bridge_<env>.pdf` exist in `pnc_repro/`.
Only Ant appears as Figure 3. **CIFAR: no** bridge/mahal/panel-c artifacts.

**108. Cached hidden representations for a cheap second-domain diagnostic? — Answered.**
Raw activations are **not** cached (only derived Mahalanobis distances and probe sketches, for
all 11 envs). Raw base inputs/targets and per-point `sq_error`/`pred_var` npz are cached. Base
MLP retrains in ~12 s, so a second-domain diagnostic is cheap but would recompute activations.

*(Effective-rank artifacts `experiments/figures/effective_rank_ant.*` and `effrank_vs_nll_ant.*`
come from `experiments/scripts/measure_effective_rank.py` / `plot_effrank_vs_performance.py`
and are **unrelated** to Fig 2/3.)*

---

## 8. Theory–implementation correspondence

**109. λ=0 or λ>0 in submitted experiments? — Answered. λ>0.**
MuJoCo per-env λ∈{1e-4,0.01} (Table 3); CIFAR λ=1e-3 (Table 6). (Legacy default 0, but the
headline selection is always >0.)

**110. Does Prop 3's diagnostic use the ridge hat weights? — Answered. No.**
The Fig-3C diagnostic uses a **finite-difference random-probe sketch** ŝ(x) (ε=0.01·ps, J=16;
`pnc_repro/figures/notes.txt`), not the analytic ridge hat weights of Eq (8). The correction
itself uses ridge (λ>0), but Prop 3's quantities are validated empirically, not computed.

**111. Is there code that explicitly computes … — Partially.**
- Corrected residual: **yes** — `PnCEnsemble.compute_shift_diagnostics` raw vs corrected RMS
  shift (`ensembles.py:1664`); `panel_c` `finite_disagreement`.
- Ridge leverage h^λ_S(x): **no** (not in headline code).
- Calibration hat weights w^λ_S(x): **no**.
- Corrected sensitivity operator A_S(x): only via the finite-difference probe sketch, **not
  analytically**.
- Calibration-predicted Jacobian T^λ_S(x): **no**.

**112. Corrected sensitivity: analytic / autodiff / finite differences? — Answered. Finite
differences** (random probes, ε=0.01·ps; `notes.txt`, confirmed by the figure audit).

**113. First-order vs finite-perturbation comparisons already present? — Answered. Yes.**
Panel C is exactly that: finite-scale ensemble disagreement D(x) (full ps) vs the small-ε
linear probe sketch ŝ(x); β₁ (power-law exponent, expected ≈1) quantifies the gap; `notes.txt`
discusses β₁<1 as finite-scale saturation. Saved per env in `panel_c_diagnostic_*.csv`.

**114. Does the code assume the local correction design has full column rank? — Answered. No.**
λ=0 uses `lstsq` min-norm (handles rank-deficiency); λ>0 adds `λI` (nonsingular regardless). So
the code is more permissive than Assumption 1(ii)'s λ=0 full-rank requirement.

**115. Are correction-system condition numbers logged? — Answered. No** in the headline pipeline;
only computed in `experiments/scripts/selftune_lambda.py` (separate scoping tooling).

**116. Are correction failures / pseudoinverse fallbacks logged? — Answered. No** — `lstsq`
returns a min-norm solution silently; no failure diagnostic.

**117. Does the code support differentiating through the least-squares correction? — Answered.**
The solves (`jnp.linalg.solve`/`lstsq`) are differentiable in JAX in principle, but P&C is
post-hoc and never backprops through the correction — no gradient path is exercised.

**118. Does ridge-form Prop 3 agree with its proof and with the implementation? — Answered / Implementation inconsistency (MuJoCo).**
Prop 3 ⇄ proof: **yes** — App A.3 keeps `G=XᵀX+λI` throughout and uses ridge hat weights
`w^λ_S=XG⁻¹y`; `A_S=Θ(J_x−T^λ_S)` is the ridge-generalized identity. Implementation: **CIFAR
matches** (ridge toward Θ₀); **MuJoCo does not** (ridge toward zero), so Thm 2's Eq (16) remark
"no λΘ ridge-bias term appears" (which relies on shrink-toward-Θ) does not strictly hold for the
MuJoCo runs.

**119. Does the proof specialize to λ=0? — Answered. No (as written in the submitted appendix).**
App A.3 is valid for general λ≥0. The only explicit λ=0 specializations are Lemma 1's
orthogonality clause and Thm 2's interpolation remark, both flagged as λ=0. So: the proposition
is intended for λ≥0; the proof is complete for ridge; the implementation is ridge-generalized
for CIFAR (toward Θ₀) but ridge-toward-zero for MuJoCo. If a reviewer reads Eq (16)'s "no λΘ"
remark as a λ=0 specialization, clarify that it is a consequence of *centering the ridge at Θ*
— and align the MuJoCo code (`ridge_toward_orig=True`) to match.

**120. Additional theorem-notation vs implementation mismatches — Answered.**
1. **Ridge center** (main one): theory & CIFAR shrink toward Θ₀; MuJoCo shrinks toward 0.
2. **Bootstrap at bf=1.0**: paper §3 implies full-size-with-replacement; code disables bootstrap
   at exactly 1.0 (`ensembles.py:516`).
3. **OOD score vs theory object**: the theory analyzes the corrected residual ρ_S; MuJoCo AUROC
   uses **total** predictive variance (aleatoric+epistemic), not the pure corrected-residual
   disagreement — a notation-vs-metric gap worth a sentence.

---

## 9. Efficiency information available without new benchmarking
*(from the baseline/efficiency audit)*

**121. Independent base networks trained per method — Answered.**
MuJoCo (each method trains its *own* base net — no shared backbone): P&C 1 (then 50 post-hoc
members, `gym_tasks.py:1495-1512`); Deep Ensemble `n_baseline` (`gym_tasks.py:313-321`);
MC Dropout 1; SWAG 1; Laplace 1; Subspace 1. CIFAR (dependency graph): base/MSP/Energy/
Mahalanobis/ReAct+Energy/**LLLA**/**P&C** all reuse the **same one** `CIFARTrainPreActResNet18`
checkpoint (`cifar_tasks.py:1234,1382,2533,…`); Deep Ensemble 5 independent trainings
(`cifar_tasks.py:612-616`); MC Dropout 1; SWAG 1; Epinet base(frozen)+1 head.

**122. Full training runs for the submitted Deep Ensemble — Answered / mismatch.**
CIFAR: **5** full 300-epoch runs (`cifar_tasks.py:608,612-616`; `openood_..._standard_ensemble_..._n5`).
MuJoCo: the *submitted* Table 3 lists **DE n=5**, but local run records disagree —
`gym_settings_appendix.txt` shows selected DE sizes of **10–20**, and the **rebuttal** headline
(`gym_tables_paper.txt` "Deep Ensemble (x50)"; `logs/p2_de_m50.log`) uses **50**. So the DE size
actually used for MuJoCo needs reconciliation (see Q147). Each DE member = one 5000-step run.

**123. Does SWAG require an extra SGD trajectory after base training? — Answered.**
No *separate* post-base trajectory: SWAG collects its moments **during a single training run** via
a `step_hook` firing after `swag_start` (`training.py:234-316`; MuJoCo
`train_swag_model(...,steps=5000,swag_start=1000)` `gym_tasks.py:572`). But it is its own fresh
training run, **not** post-hoc on a shared base. CIFAR: `swag_start_epoch=240`, `swag_max_rank=20`,
BN refresh on 2048 (`cifar_tasks.py:768-838`).

**124. Which parameters are in the Laplace approximation? — Answered.**
MuJoCo Laplace = **full-network KFAC** over every linear layer plus both probabilistic heads
(`laplace.py:19-56,107-117,166-173`), sampled across all blocks (`ensembles.py:1192`). CIFAR uses
a distinct **last-layer** Laplace (LLLA) over `fc` only (`cifar_tasks.py:1872-1873`;
`benchmark_inference_cost.py:150-175`). MS: paper says "Laplace" generically; the MuJoCo vs CIFAR
scope difference (all-layer KFAC vs last-layer) is not stated.

**125. Wall-clock training/construction times available? — Answered. Yes.**
`train_time`/`setup_time` are written into result JSONs (`gym_tasks.py:346,419,514,598,713,1116,
1433,1552`; **942** gym JSONs carry them) and printed to `logs/p1_baselines.log`, `logs/p2_de_m50.log`.
CIFAR JSONs carry `train_time` too, plus `inference_cost.json` (below).

**126. GPU types recorded in logs? — Not recoverable.** No GPU/CUDA/device string anywhere in
`logs/*.log` (only hostname/username). The paper states GTX 1080ti in App C.1, but logs don't
record it — a device-provenance claim can't be sourced from logs.

**127. Peak memory measurements? — Not recoverable.** No memory instrumentation exists
(no tracemalloc/RSS/peak anywhere). Requires new measurement.

**128. Checkpoint file sizes? — Not recoverable.**
Almost all checkpoints are deleted: repo-wide only 3 legacy `.pkl` (`old_results/mnist` 958,753 B ×2;
`old_results/uci/boston` 3,727 B). No CIFAR/MuJoCo model checkpoints on disk; MuJoCo never persists
a checkpoint (Q129). Sizes for base / P&C member / full P&C ensemble / SWAG / Laplace / DE member on
the paper's models require re-serialization.

**129. Does P&C serialization save one full model per member? — Answered.**
There is **no on-disk serialization of any P&C ensemble** (`GymPJSVD`/`CIFARPnC` save only metrics
JSON + a geometry `.npz`, `gym_tasks.py:1210`). In memory it is a **shared base + per-member
blocks**: `PnCEnsemble` holds one `base_model` and per-member `members_w1[i]`/`members_w2[i]`
(`ensembles.py:1657-1658`); `PJSVDEnsemble` stores per-member `seq_dWs`/`seq_w_effs`/`seq_b_effs`
(`ensembles.py:578-585`). Untouched layers are shared by reference, but perturbed/corrected blocks
are **materialized full-size** per member (not kept as the K coefficients).

**130. Could it be shared-base + per-member perturbed/corrected blocks? — Answered. Yes.**
That is essentially the current runtime object — the only per-member tensors are the target block's
perturbed `conv1`/`W` and corrected `conv2`+`b2`/`W_next`+`b_next` (`ensembles.py:1657-1658,530-582`);
everything else is identical across members. A shared-base + per-member-block layout is exactly
representable.

**131. Fraction of parameters that is member-specific — Answered.**
- **MuJoCo (submitted scope=`multi`)**: perturbs l1,l3 + corrects l2,l4, so **all four hidden layers
  are member-specific**; only the two heads stay shared. Weight fraction ≈ (200·D_in+120,000) /
  (200·D_in+120,000+400·D_out) ≈ **0.75–0.9** (materialized view).
- **CIFAR single-block (s3b1 = stage 4 / block 1, the code default and every local result file)**:
  member-specific = conv1 (3·3·512·512) + conv2 (3·3·512·512) + b2 (512) ≈ **4.72M of ≈11.17M ≈ 42%**.
  ⚠️ The paper's Table 6 says the selected block is **(3,0)**; the code/results use **s3b1 = (3,1)**
  — reconcile (Q25).
- **CIFAR multi-block** (also run): perturbs conv1 + corrects conv2 across **all 8 blocks** → most conv
  params become member-specific.
Caveat: these are for the *materialized* representation; the *intrinsic* member-specific DOF is only
the K coefficients (Q132).

**132. Theoretical minimal P&C storage (shared-base) — Answered.**
One shared base + shared direction bases (K vectors/layer) + shared calibration targets + **M×K
scalar coefficients** per perturbed layer (`z_coeffs`). Per-member *marginal* cost = **K floats**
(K=20 → 20/member/layer; MuJoCo multi = 40/member). Corrections are deterministic functions of
(base, coeffs, X_sub) and need not be stored. **This minimal path is not implemented.**

**133. One full forward pass per member at inference? — Answered. Yes.**
`predict` loops members (`ensembles.py:857-867`, `1784-1788`); CIFAR `_forward_member` recomputes the
**entire** ResNet with only the target block swapped (`ensembles.py:1732-1782`). `inference_cost.json`
records P&C `n_forward_passes: 50`.

**134. Can prefix/suffix be shared given the target-layer placement? — Answered. Not currently.**
For CIFAR, the stem + all stages/blocks **before** the target block are member-independent
(`ensembles.py:1734-1776`) yet recomputed for every member — cacheable once. (MuJoCo multi perturbs
l1 first, so the shareable prefix is just `x`; the untouched tail is shared but recomputed.)

**135. Vectorized or sequential? — Answered. Sequential Python loop** in every P&C predict path
(`ensembles.py:861-862,1786-1787`); DE inference is also a list comprehension. No vmap/scan over
members.

**136. Inference timing logs available? — Answered. CIFAR yes, MuJoCo no.**
`results/cifar10/inference_cost.json` (per method: `warm_per_sample_ms`, `n_forward_passes`,
`train_cost_factor`). Real numbers: Deep Ensemble n=5 = **1.34 ms/sample** (5 passes); P&C
multi-block = **9.69 ms/sample** (50 passes); P&C single-block ≈ 7.4 ms; MC Dropout n=32 = 5.18 ms;
LLLA n=50 = 7.68 ms; Mahalanobis = 0.73 ms (1 pass). MuJoCo has **no** inference-latency log.

**137. Construction-cost claims supportable from existing logs? — Answered. Partially.**
`train_cost_factor` (`inference_cost.json`): DE=5.0, Epinet=1.05, LLLA/Maha/MCD/SWAG/P&C=1.0, plus 942
gym `train_time`/`setup_time` values. These support "P&C/SWAG/Laplace/Subspace/MC-Dropout each ≈1×
base training; DE = M× (5× CIFAR); Epinet ≈1.05×." Absolute cross-method wall-clock on identical
hardware is only partial (gym timings mix train+construct; CIFAR DE `train_time` is load-only).

**138. Efficiency quantities still requiring NEW measurement — Requires experiment.**
(a) peak GPU/host memory; (b) GPU-type provenance in logs; (c) checkpoint/storage sizes for the
paper's actual models; (d) MuJoCo inference latency; (e) any speedup from the storage-minimal /
prefix-shared representation (not implemented); (f) apples-to-apples DE-vs-P&C wall-clock at matched
hardware/batching.

**139. Manuscript sentences that could imply single-pass / low-cost inference — Answered.**
- "requiring only a single pretrained model" (abstract) and "constructed entirely post hoc from a
  single pretrained model" (§2) describe **construction**, but could be misread as cheap **inference**.
  Inference is **50 sequential forward passes** (`inference_cost.json`; ~7× a DE-n5 per sample here).
- "strong training cost/quality tradeoff" (§1 contributions) — see Q180.
No sentence literally claims single-pass inference, but the framing should explicitly separate
construction from inference cost (Q188).

**140. Factual distinction the paper should draw — Answered.**
- **Avoiding multiple independent training runs:** TRUE and the real advantage — P&C needs 1 base
  training vs Deep Ensemble's 5 (CIFAR) / 5–50 (MuJoCo). Same 1× as the other post-hoc baselines.
- **Post-hoc construction cost:** M closed-form ridge solves on the calibration subset — cheap
  relative to training (`train_cost_factor=1.0`), comparable to other post-hoc methods.
- **Model storage:** currently materialized full-size per member (Q129/131); *could* be shared-base +
  K coefficients/member (Q132), but that is not implemented — so no realized storage win today.
- **Ensemble inference cost:** M=50 forward passes, i.e. **ensemble-like, not reduced** — the CIFAR
  timing shows P&C is the slowest method measured (9.69 ms/sample).

---

## 10. Baseline comparability and existing support
*(from the baseline/efficiency audit)*

**141. Baselines implemented directly in the repo? — Answered. All of them.**
Deep Ensemble, MC Dropout, SWAG (`training.py:234,348,823`), full-net KFAC Laplace (`laplace.py`),
LLLA, Subspace (`training.py:427`), Epinet (`training.py:1021`), Evidential (`evidential.py`), and the
OpenOOD scalar scorers MSP/Energy/Mahalanobis/ReAct+Energy (`openood_eval.py`,
`cifar_tasks.py:2524-2662`).

**142. External implementations? — Answered. None** for the methods. Only generic frameworks
(JAX/Flax-nnx/optax/grain) and `sklearn.metrics` for AUROC/AUPR. No laplace-torch, no OpenOOD library.

**143. Same base architecture across baselines? — Answered. Yes** — MuJoCo `[200,200,200,200]` ReLU
MLP; CIFAR PreActResNet-18 (dropout variant has identical topology), same `_e300_` recipe.

**144. Same training data? — Answered. Yes** — all methods share the same `id_train`/CIFAR splits and
the same OOD benchmark loaders.

**145. Same number of seeds? — Answered. Within a domain, yes.**
MuJoCo rebuttal = 5 seeds {0,10,42,100,200} (`run_bootstrap_ls_lreg_paper_table.sh:15`); the original
submission likely 3 {0,10,200} (`make_gym_paper_table.py:455`). CIFAR = 3 seeds {0,1,2}. (The PDF
tables say "10 seeds" for MuJoCo and "5 seeds" for CIFAR — reconcile against the run records; see Q167.)

**146. Equivalent ID-only calibration for all probabilistic methods? — Answered. Yes** — every method
gets the same ID-val post-hoc calibration + val-NLL selection (`vcal` variant uniformly;
`gym_tasks.py:594-596`; CIFAR temperature on ID val).

**147. DE = 5 (CIFAR) and 10 (MuJoCo)? Is "10" seeds or members? — Answered / mismatch.**
CIFAR DE = **5 members** (`n5`, matches paper). MuJoCo: the submitted Table 3 says **DE n=5**, but the
"10" the reviewer saw in Table 4 is **seeds, not members**. Separately, local records show MuJoCo DE
was run/selected at **10–20** (`gym_settings_appendix.txt`) and **50** in the rebuttal
(`gym_tables_paper.txt`, `logs/p2_de_m50.log`). So: reviewer's "10 members" = a seed/member confusion;
the *actual* MuJoCo DE membership used for the numbers must be pinned down (submitted Table 3 = 5;
rebuttal = 50).

**148. Why different ensemble sizes (5 DE / 100 variance / 50 P&C)? — Not recoverable (rationale).**
Variance methods run at `n_perturbations=100`, P&C at 50, DE at 5–50; the numbers are luigi params /
run-script flags with **no documented justification**.

**149. Is P&C size matched to Deep Ensemble? — Answered. No** — P&C M=50; DE 5 (CIFAR) / 5–50 (MuJoCo).
(A matched-cost `EnsemblePJSVDHybrid` and `experiments/scripts/matched_cost_pareto.py` exist for
separate analyses.)

**150. MC Dropout stochastic samples = P&C members? — Answered. No / mismatch.**
MuJoCo MC Dropout **100** vs P&C 50. CIFAR MC Dropout **n=32** in the OOD-eval files
(`openood_v1p5_mc_dropout_vcal_n32`) — paper Table 6 says the fixed value is **50**. (Note MC Dropout
is in the CIFAR *sweep* table but not in CIFAR Table 2.) Reconcile the n=32-vs-50 discrepancy.

**151. SWAG posterior samples = P&C? — Answered.** MuJoCo SWAG 100 vs P&C 50; CIFAR SWAG **50**
(matches CIFAR P&C's 50). `swag_start=240` matches Table 6.

**152. Laplace samples = P&C? — Answered.** MuJoCo Laplace 100 (KFAC fit on 10000 pts) vs P&C 50;
CIFAR LLLA 50.

**153. Is Subspace Inference post-hoc from one pretrained model, or does it need a trajectory? — Answered.**
It **requires an SGD trajectory**: `GymSubspaceInference` trains its own 2000-step model, snapshots
steps 1000–2000, and builds a rank-20 PCA subspace by SVD (`gym_tasks.py:1502-1512`;
`training.py:427-519`). Not post-hoc on the shared/P&C base.

**154. Is SNGP discussed but not evaluated? — Answered. Yes** — no SNGP code anywhere (grep 0 hits);
appears only in `related_work.tex`. Excluded (needs spectral-norm/arch change).

**155. Is Epinet on CIFAR but not MuJoCo? — Answered. Yes** — Epinet tasks exist only in
`cifar_tasks.py` (`CIFARTrainEpinet`, `CIFAROpenOODEpinet`); no Epinet in `gym_tasks.py`.

**156. Does Epinet require training an extra epistemic head? — Answered. Yes** — trained on a **frozen**
base via `stop_gradient` (`training.py:1021,1038-1074`), `epinet_epochs=100`, `index_dim=8`,
`train_cost_factor=1.05`.

**157. Does LLLA modify only the final layer? — Answered. Yes** — last-layer Laplace over `fc` only
(`cifar_tasks.py:1872-1873`; `benchmark_inference_cost.py:150-175`).

**158. Frozen-backbone post-hoc methods supported but omitted? — Answered.**
- **Evidential regression** (`evidential.py`, `GymEvidential`) — implemented, excluded from main tables.
- **OpenOOD scalar scorers MSP/Energy/Mahalanobis/ReAct+Energy** — implemented and evaluated (scalar
  OOD scores, Q161).
- Temperature scaling exists only as the shared calibration step, not a standalone comparator.
- **Not present:** ODIN, KNN, SNGP.

**159. Existing results for omitted comparators? — Answered. Yes.**
Evidential has **141** result JSONs (`results/*/evidential_*.json`); excluded per
`gym_settings_appendix.txt` ("evaluated but excluded due to NLL instability"). OpenOOD
MSP/Energy/Mahalanobis/ReAct results also present in `results/cifar10/`.

**160. Baselines that most directly match "uncertainty from one pretrained model, no backbone retraining"? — Answered.**
CIFAR: **LLLA, Epinet (frozen backbone + tiny head), and the OpenOOD scorers** reuse the identical
base checkpoint — same category as P&C. Not in this category: Deep Ensemble, MC Dropout, SWAG,
Subspace (each trains its own net). MuJoCo: **no** baseline is truly post-hoc on a shared base — only
P&C is single-train (full-net Laplace at least doesn't retrain but samples all layers).

**161. Baselines NOT comparable (scalar OOD score, not an ensemble of predictors)? — Answered.**
**MSP, Energy, Mahalanobis, ReAct+Energy** — one scalar OOD score per input (`n_forward_passes:1`), no
predictive distribution. Usable for OOD-AUROC only, not NLL/predictive-variance comparison.

**162. Which novelty claims to narrow — Answered.**
- The frozen-backbone-post-hoc-and-predictive comparators are LLLA, Epinet, and P&C; DE/MCD/SWAG/
  Subspace each need their own training run here (MuJoCo especially). So "post-hoc from one pretrained
  model" is a real but **shared** property (LLLA/Epinet also qualify) — frame P&C's novelty as the
  *affine-redundancy* construction with per-member cost = K coefficients, not as "the only post-hoc
  method."
- Comparisons are **not sample-count-matched** (variance methods 100, P&C 50, DE 5–50) — state the
  budgets.
- Present MSP/Energy/Mahalanobis/ReAct on OOD-detection metrics only (Q161).
- SNGP/ODIN/KNN are not implemented — any "outperforms" phrasing about them is unsupported here.

---



## 11. Variance, aggregation, and presentation
*(from the evaluation/ranking audit)*

**163. How is mean rank computed? — Not recoverable (code).**
The Table-1 generator is **absent from the repo**; only its outputs survive
(`pnc_repro/figures/cross_env_summary_paper.{txt,tex}`). Per the caption: methods ranked 1–6
within each env per metric, then mean rank (±stddev across envs). The computation is not
reproducible from code.

**164. Ranked independently by env / shift / metric? — Answered. Yes** — per-env, per-metric
(ID RMSE, Far NLL, Far AUROC, Far Spearman), averaged across envs. Only Far is ranked (plus ID
RMSE); Near/Mid are not ranked.

**165. How are ties handled? — Answered (from caption).** Average ranks ("ties get average
ranks", `cross_env_summary_paper.tex` caption). Not independently verifiable (no code).

**166. Ranks averaged across seeds before or after ranking? — Answered. Rank the seed-mean.**
Per-env cells are "mean ± stddev across seeds"; Table-1 stddev is "across envs" — i.e. one rank
per (env,method), so seeds are averaged before ranking. Inferred from output headers.

**167. Which raw tables underlie the headline mean-rank claims? — Answered / gap.**
`appendix_per_env_table_paper.txt` (ID RMSE, Far NLL, Far AUROC) + `spearman_table_paper.txt`
(Far Spearman). **Raw per-seed JSONs exist only for Ant/HalfCheetah/Hopper/Humanoid**; the
other 7 envs have no metric JSONs locally — Table 1 cannot be regenerated from repo code+data.
Seeds per env are 10–28, not a uniform 10.

**168. Are the HalfCheetah-variance observations supported by seed-level results? — Partially.**
The paper's own appendix shows P&C HalfCheetah ID RMSE = **1.779 ± 0.800** (largest ID-RMSE
std there). The raw 27-seed values are **not** in the repo; the 5 canonical seeds present
(`results/HalfCheetah-v5/`) give **1.672 ± 0.198** — much tighter. So the ±0.800 is real in the
paper's data source but not recoverable from repo JSONs. ⚠️ submitted-PDF Table 4 shows
1.779±0.800; the rebuttal `pnc_repro` reproduction is broadly consistent here.

**169. Does P&C underperform on ID in specific environments? — Answered. Yes.**
In the rebuttal per-env table P&C is **not best on ID RMSE in 6/11 envs** (worst on HalfCheetah
1.779, Hopper 0.219, Reacher 0.120; also Swimmer, Pusher). ⚠️ **Submitted-vs-rebuttal
discrepancy:** the rebuttal `appendix_per_env_table_paper.txt` shows a **HumanoidStandup ID RMSE
blow-up of 37515 ± 56733** (a numerical instability, ~1000× baselines), whereas the **submitted
PDF Table 4 shows 25.13 ± 5.226**. The reproduction surfaced an instability not in the
submission — worth investigating before citing rebuttal numbers. (The per-env table has no ID
NLL column, so the reviewer's "ID NLL" cannot be checked there.)

**170. Are those cases visible in the appendix? — Answered. Yes** — all in
`appendix_per_env_table_paper.txt` (submitted PDF Table 4 for the clean version).

**171. Are CIs / stddevs consistently reported? — Answered. Yes** — every table reports
mean ± stddev (rank±std across envs for Table 1; ±std across seeds elsewhere; CIFAR Table 2 ±std
across 5 seeds). These are standard deviations, not formal CIs.

**172. Does rank aggregation obscure practically large differences? — Answered. Yes.**
Far NLL absolute spreads are enormous but compress to 1-rank steps: Ant MC Dropout 1374 vs P&C
1.5; Pusher Subspace 2268 vs P&C 2.0; Reacher Subspace 1016 vs P&C 0.9. Conversely a P&C
catastrophe (HumanoidStandup ID RMSE, rebuttal) costs only ~1 rank step.

**173. Cases where raw means favor one method but ranks favor another? — Answered. Yes.**
On ID RMSE, rank favors **Subspace** (mean rank 3.00) while win-count favors **P&C** (5/11 vs
3/11); P&C and Laplace tie at 3.18. P&C's raw ID mean is dragged by the HumanoidStandup outlier
that the rank hides.

**174. Which claims should be rewritten as metric-specific? — Answered (from Wilcoxon-Holm).**
- **Far AUROC**: P&C significantly beats all 5 baselines (Holm p≤0.0098) — dominance claim fully
  supported (Table 1 shows ↑ on all rows).
- **Far NLL**: best *mean rank* but **not** Holm-significant vs Deep Ensemble/MC Dropout/SWAG/
  Laplace (p≈0.055) — significant only vs Subspace (Table 1 arrow already reflects this).
- **Far Spearman**: best mean rank but **not** significant vs Deep Ensemble (p≈0.054; Table 1
  has no ↑ on the DE row).
- **ID RMSE**: no pairwise comparison significant — "competitive, most wins" is the correct
  framing.

**175. Are Table 1's textual interpretations fully supported? — Partially.**
Table 1's *arrows* correctly encode the Holm significance, and the Far-AUROC and
"competitive-on-ID / most-ID-wins" wording is supported. But the §5.3 prose "obtains the best
mean rank on Far NLL … Far Spearman" reads as dominance while those are best-rank-not-significant
vs the strongest baselines; the per-env tables also show P&C is not per-env-best on Far NLL/AUROC
everywhere (Q86). Soft manuscript mismatch: caption "10 seeds" vs rebuttal "10–28 seeds/env".

---

## 12. Manuscript corrections and clarity issues

**176. Unresolved cross-references ("Appendix ??")? — Answered. None** in the submitted PDF
(grep clean). If a newer Overleaf draft has them, re-check there.

**177. Incorrect table references? — Answered. None found** in the submitted PDF — cross-refs
are internally consistent (CIFAR §5.4→Table 2; MuJoCo §5.3→Table 1; App→Tables 3/4/5/6).

**178. Does the CIFAR section refer to Table 4 instead of Table 2? — Answered. No.**
Submitted §5.4 says "Table 2 shows…" correctly (p9); "Table 4" appears only in the MuJoCo
appendix. This issue is **not present** in the submitted PDF — but confirm the current Overleaf
source, where the reviewer may have seen it.

**179. Does Table 5 lack the bold/italic promised by its caption? — Partially / needs source check.**
The caption promises bold (best) + italic (negative), but text extraction can't reveal
formatting. Verify the table body in the source (check whether `\textbf`/`\textit` are emitted;
the regenerated `pnc_repro/figures/spearman_table_paper.tex` is a good place to confirm the
macro logic). Plausible real issue.

**180. "cost" words used without specifying training/construction/storage/inference — Answered.**
- Intro contribution (p2): "**strong training cost/quality tradeoff** relative to standard
  post-hoc baselines" — misleading: the other post-hoc baselines (MC Dropout/SWAG/Laplace/
  Subspace) also need only **one** base training, so P&C's training-cost edge is *only* over Deep
  Ensemble. Scope it.
- CIFAR §5.4 (p9): "compares favorably to the **higher-cost** Deep Ensemble" — unqualified;
  should be "higher **training** cost" (inference cost is comparable — P&C runs M=50 passes).

**181. Broad claim words ("general/strong/matches/exceeds/outperforms") — Answered.**
- Abstract "matching or outperforming standard post-hoc baselines" — broad; supported for **Far
  OOD AUROC** but not for ID RMSE/NLL (P&C middling in several envs) nor fully for Far
  NLL/Spearman (best rank, not Holm-significant vs strongest baselines). Scope to OOD detection.
- "plug-and-play" / "general" — **not present** (good; don't add).
- §5.3 "best mean rank on Far NLL … Spearman" — accurate as *mean rank*; add that these are not
  Holm-significant vs Deep Ensemble/others.

**182. Where to clarify the calibration-set definition — Answered.**
§5.1 / §3. State that the calibration set is a subset of the ID **training** pool (4096 MuJoCo /
1024 CIFAR), overlaps base training (as §3's "may be the training data" allows), and — for CIFAR
— is disjoint from the temperature-scaling val split.

**183. Where to explain single- vs multi-block — Answered.**
§5.1 or §3. State: MuJoCo perturbs hidden layers l1 & l3 (two pairs); CIFAR perturbs one
residual block (stage 3 / block 0); scope was fixed a priori for MuJoCo and ID-val-selected for
CIFAR.

**184. Where to define the CIFAR OOD inference rule — Answered.** §5.4 / App B (see Q79).

**185. Where to reference Near/Mid MuJoCo results — Answered.**
§5.3 — point to Spearman Table 5 for Near/Mid and note Near/Mid NLL/AUROC are available (add an
appendix table if space permits).

**186. Where to explain the absent CIFAR Mid group — Answered.**
§5.4 / App D — OpenOOD v1.5 CIFAR-10 defines only Near ({CIFAR-100, TIN}) and Far ({MNIST, SVHN,
Textures, Places365}); there is no Mid tier (unlike MuJoCo).

**187. Where to state ensemble size — Answered.**
Stated in App C.2 (P&C M=50) and Table 6, but not near Tables 1/2. Add M=50 (and DE M=5,
variance methods n=100) inline near the headline tables.

**188. Where to acknowledge ensemble-like inference cost — Answered.**
Limitations / efficiency. P&C inference is **M=50 sequential forward passes**
(`ensembles.py:861`) — ensemble-like cost, not single-pass; only *construction* (no extra
training) is cheap.

**189. Where to add sensitivity limitations — Answered.**
Add a real limitations paragraph (currently elided "due to page limit", p27) covering: layer/
block choice, calibration coverage/size, BN-frozen normalization handling, and the affine-layer
architecture requirement.

**190. Caption/numbering/text mismatches — Answered.**
Reconcile seed counts: submitted PDF says "10 seeds" (Table 1/4) but the rebuttal `pnc_repro`
uses 10–28; ensure whichever numbers ship match the caption. Table 4's HumanoidStandup value
(25.13 submitted vs 37515 rebuttal) must be reconciled before any table update (Q169).

**191. Typographical issues (e.g. "representitive") — Answered.**
"representitive" is **not** in the submitted PDF (correct "representative" is used). Note: "SWAG"
renders as "SW AG" throughout the extraction — likely a kerning/ligature artifact, but verify
it's not a literal space in the source. Do a fresh typo pass on the current source.

**192. List of manuscript edits doable without new results — see summary block 2 below.**

---

## Summary 1 — Facts ready for rebuttal (no new work)

- **Calibration is ID-only and no OOD leaks into any tuning decision.** Correction uses a
  4096 (MuJoCo) / 1024 (CIFAR) subset of the ID training pool; selection is ID val-NLL; CIFAR
  temperature is fit on a disjoint 10% ID split; OOD tiers enter only at final test
  (`gym_tasks.py:872-874`, `cifar_tasks.py:402-405,1660`, `util.py:289-318`). (Q31–46)
- **Exact algorithm is auditable and matches the paper structurally**: perturb target-layer `W`
  (bias untouched), refit next layer's `(W,b)` by closed-form ridge-LS onto the *original*
  next-layer preactivation, per member, tail unchanged, BN frozen. (Q1–11, Q21–24)
- **Multi-block MuJoCo semantics**: perturb l1 & l3, correct l2 & l4, sequentially; each later
  correction sees earlier perturbations+corrections upstream but targets the original base
  preactivation. (Q21–24)
- **Prop 3 and its proof are the ridge-generalized identity** (valid for λ≥0); the proof does
  *not* specialize to λ=0. (Q118, Q119)
- **Inference/aggregation is the Deep-Ensembles recipe** (uniform mixture; moment-matched
  Gaussian NLL; total predictive variance for MuJoCo AUROC; mixture predictive entropy for
  CIFAR), and P&C and Deep Ensemble share the identical inference path. (Q61–78)
- **Mechanism diagnostics already exist for all 11 MuJoCo envs** (not just Ant), saved per
  example. (Q106, Q107)
- **Far-AUROC dominance is Holm-significant vs all 5 baselines** (p≤0.0098); Table 1's
  significance arrows are correctly placed. (Q174)
- **Selected hyperparameters are fully documented** (Tables 3, 6;
  `appendix_selected_hparams_paper.txt`). (Q47–56)

## Summary 2 — Clarifications requiring manuscript edits (no new results)

1. Fix Eq (1)/(3) ↔ MuJoCo mismatch: either state the ridge is centered at Θ (matching CIFAR)
   *and* set `ridge_toward_orig=True` for MuJoCo, or footnote that MuJoCo used ridge-toward-zero
   with tiny λ. (Q6, Q13, Q118–120)
2. Define the **CIFAR OOD inference rule** explicitly (§5.4/App B). (Q79, Q184)
3. Clarify the **calibration set** (subset of ID training pool; overlaps training; disjoint from
   CIFAR temp-scaling split). (Q182)
4. Explain the **single- vs multi-block** choice and which layers/blocks. (Q26, Q183)
5. Reference **Near/Mid** MuJoCo results and explain the **absent CIFAR Mid** group. (Q185, Q186)
6. State **ensemble sizes** near the tables (P&C 50, DE 5, variance methods 100). (Q187)
7. Acknowledge **ensemble-like inference cost** and qualify every "cost" as
   training/construction/storage/inference — especially the "training cost/quality tradeoff" and
   "higher-cost Deep Ensemble" lines. (Q180, Q188, and see Summary 4/§9–10)
8. **Scope the dominance claims** to specific metrics (Far AUROC significant; Far NLL/Spearman
   best-rank-not-significant; ID competitive). (Q174, Q175, Q181)
9. Add a real **limitations** paragraph (layer choice, calibration coverage, BN, architecture).
   (Q189)
10. Correct the **bf=1.0** description (§3) to match the code. (Q40)
11. Note the CIFAR near/far groups are **hardcoded**, not parsed from OpenOOD configs. (Q90)
12. Reconcile **seed counts** and the **HumanoidStandup** value between submission and rebuttal
    reproduction; fresh typo/formatting pass (Table 5 bold/italic). (Q168–170, Q179, Q190, Q191)
13. **Reconcile stated vs actual numbers (implementation ≠ manuscript):** (a) CIFAR single-block
    target is **s3b1 = (3,1)** in code/results, not the paper's **(3,0)** (Q25, Q131); (b) CIFAR
    **MC-Dropout OOD runs use n=32**, not the Table-6 "fixed 50" (Q150); (c) **MuJoCo Deep-Ensemble
    size** — Table 3 says n=5, but local/rebuttal records use 10–50 members (Q122, Q147); (d) MuJoCo
    Laplace is **full-network KFAC** while CIFAR Laplace is **last-layer** — state the difference
    (Q124). These are the numbers most likely to draw a reviewer "does the code match the paper?"
    challenge.

## Summary 3 — Unknown historical decisions (not recoverable from the repo)

- **Why multi-block for MuJoCo but single-block for CIFAR** — no rationale in code/notes/commits.
  (Q26, Q27)
- **Why only Far in the main MuJoCo table** — no documented reason. (Q83, Q84)
- **The Table-1 mean-rank + Wilcoxon-Holm generator script is absent** — ranks/ties/aggregation
  known only from the caption; not reproducible from repo code. (Q163, Q165)
- **The current Figure 3 generator is absent** from tree and git history (predecessor + saved
  artifacts remain). (Q92)
- **Raw per-seed metric JSONs for 7 of the 11 MuJoCo envs are absent** locally — Table 1 cannot
  be regenerated from repo code+data. (Q167)
- **The HalfCheetah ±0.800 / HumanoidStandup rebuttal blow-up** — the underlying 27-seed raw
  values are not in the repo. (Q168, Q169)

## Summary 4 — Questions that genuinely require new experiments

Only a handful of the 192 require new runs — almost all efficiency provenance:
- **Peak GPU/host memory** for any method — no instrumentation exists (Q127, Q138a).
- **GPU-type provenance** — the GTX 1080ti claim (App C.1) is not in any log (Q126, Q138b).
- **Checkpoint / storage sizes** for the paper's actual CIFAR/MuJoCo models — all checkpoints
  deleted; must re-serialize to measure (Q128, Q138c).
- **MuJoCo inference latency** — only CIFAR has `inference_cost.json`; MuJoCo has none (Q136, Q138d).
- **Any speedup from the storage-minimal / prefix-shared representation** — not implemented, so the
  claim cannot be measured without building it (Q130, Q132, Q134, Q138e).
- **Apples-to-apples DE-vs-P&C wall-clock at matched size/hardware/batching** (Q138f).

Everything else in the 192 is answerable from existing materials (with the "not recoverable"
historical items in Summary 3 requiring no *experiment*, just author memory / archived data).
