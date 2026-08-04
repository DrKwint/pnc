# protocol_clarifications.md — factual protocol answers (Priority 5)

All answers are read from code (file:line) or submitted artifacts. Where the logs do not
establish a rationale, that is stated explicitly rather than inferred.

---

### 1. Exact predictive distribution for the MuJoCo ensemble
A **diagonal-Gaussian mixture, moment-matched**. Each of the M members returns
`(mean_m, var_m)` (the probabilistic base has a Gaussian aleatoric head), shape `(M,B,D)`
(`ensembles.py:857`). The predictive distribution is `N(mu, Sigma)` with
(`util._predictive_mean_var`, `util.py:112-120`):
- `mu = (1/M) Σ_m mean_m`
- `Sigma_diag = (1/M) Σ_m var_m  +  Var_m(mean_m)`  (aleatoric mean + epistemic disagreement; law of total variance).
NLL is the diagonal-Gaussian NLL of that `(mu, Sigma)`, averaged over points and output dims,
with a `1e-6` variance floor (`metrics.py:7-15`). **OOD score** = per-point total predictive
variance `mean_d(Sigma_diag)` (`util.py:256`), thresholded ID-vs-OOD for AUROC
(`metrics.compute_ood_metrics`, `util.py:305`). (The default gym base is deterministic; the
submitted "+Prob" main method uses the probabilistic head above.)

### 2. Exact scalar OOD score for CIFAR-10
**Predictive entropy of the ensemble-mean softmax** (default `primary_score`,
`openood_eval.py:106`; used by `CIFAROpenOODPnC`):
`mean_probs = (1/S) Σ_s softmax(logits_s / T)`;
`score = − Σ_c mean_probs_c · log(mean_probs_c + 1e-8)` (`openood_eval.py:24-29`).
Also computed and stored (not the headline): MSP `1 − max_c mean_probs`, energy
`mean_s(−T·logsumexp(logits_s/T))`, margin, mutual information, variation ratio
(`openood_eval.py:30-58`). Higher score ⇒ more OOD; AUROC/AUPR/FPR95 via sklearn
(ID=0, OOD=1).

### 3. How member logits/probabilities are combined
**Mean of per-member softmax probabilities** (not mean logit): `probs = softmax(logits/T);
mean = mean_s(probs)` (`util.py:560`, `openood_eval.py:24-28`). Per-member logits are stacked
`(S,N,C)`.

### 4. Temperature scaling — per-member or combined?
A **single scalar T** applied to **each member's logits before softmax** (`logits_s / T`),
i.e. inside the per-member softmax that is then averaged — not a separate calibration of the
pooled predictor. T is fit by golden-section search minimizing the **mean-softmax NLL on the
ID validation split only** (`util._fit_posthoc_temperature`, `util.py:159-207`). Enabled by
`posthoc_calibrate` (default False in `CIFARPnC`/OpenOOD tasks ⇒ T=1.0; True in the
paper-protocol selector). For MuJoCo, the analogous step is **variance calibration** ("VCal"):
a single scalar `variance_scale` fit on the held-out ID split; it rescales the predictive
variance and therefore **does not change AUROC ranking** (monotone), only NLL/ECE.

### 5. Data used for base training / P&C correction / HP selection / temperature
| Stage | MuJoCo (gym) | CIFAR-10 |
|---|---|---|
| Base training | `x_tr` = 90% of ID `id_train` (`_split_data`, val_split=0.1, seed=99) | `x_tr` = 90% of CIFAR-10 train (`_split_data`, val_split=0.1, seed=99) |
| P&C correction (X_sub) | random subset of the **full `id_train`** (default ≤4096/10000) — **overlaps `x_tr`** | random subset (1024) of **`x_tr`** — **overlaps base training** |
| HP selection (size / λ / prior) | held-out `x_va` (10% ID) **validation NLL** | ID val (paper-protocol selector: ID metrics) |
| Temperature / VCal | `x_va` (same 10% ID slice, dual-used) | `x_va` (10% ID) NLL |
| Test metrics | `id_eval` = a **separate** expert-v0 Minari draw (seed+1) | CIFAR-10 test set |
Overlap summary: the correction calibration data overlaps the base-training pool on **both**
benchmarks; the only strictly held-out ID split is the 10% used for size/λ selection and
temperature/VCal. **No OOD data enters construction or selection** (see item 6).

### 6. Is any OOD data used for hyperparameter selection?
**No.** MuJoCo sizes/λ are selected by **ID validation NLL** (`nll_val` on the 10% ID split;
`json_to_tex_table.py:49-51`). CIFAR temperature is fit on ID val NLL; the paper-protocol
selector ranks candidates by ID metrics. The OOD/shift tiers are used strictly for
evaluation (`experiments.tex:6`; `CollectGymData` OOD tiers never feed construction). The
ID-only selection protocol is preserved throughout this rebuttal.

### 7. Single-block P&C — exactly which parameters change?
- **CIFAR (`CIFARPnC`, block = stage4/block1):** the perturbation is added to that block's
  **conv1 kernel** (`w1_pert = w1_orig + p`, `ensembles.py:1651`); the block's **conv2 weight
  AND bias** are re-solved by ridge to match the unperturbed block output
  (`_ridge_regression_solve`, `pnc.py:246-259`). BN1/BN2 stay in inference mode and BN2 is
  absorbed into the conv2 fit. Nothing else in the block or network changes.
- **MuJoCo single-block ([0]):** perturb hidden-layer-0 weights `W0` (`W0 + Σ_k z_k v_k`);
  re-solve the next affine layer's weight+bias `(W1', b1')` by LS/ridge on X_sub. Only
  `{W0(perturbed), W1(refit), b1(refit)}` differ per member.

### 8. Are all downstream parameters frozen?
**Yes.** Only the perturbed layer and its immediately-following affine correction layer
change; the entire tail after the correction layer is evaluated with the original base-model
parameters (`ensembles.py:695`, `evaluate_tail_from_preact`; CIFAR: residual/downsample path
and head `final_bn`/`fc` unchanged). For multi-block, each perturb/correct pair is applied and
everything after the last corrected layer is frozen.

### 9. Multi-block P&C — order of corrections
**Sequential, ascending layer order** (`_precompute_sequential_ls`, `ensembles.py:530-582`):
perturb layer 0 → solve correction at layer 1 → propagate the corrected activation forward →
perturb layer 2 → solve correction at layer 3 → … Each correction target is the *unperturbed*
model's pre-activation at the correction layer, computed on X_sub (`ensembles.py:542-546`).
Architectural note: the implementation feeds `X_sub` directly to the first perturbed layer
(`h = self.X_sub`, `ensembles.py:528`), so the **perturbed set must begin at the input layer
(layer 0)**; the natively supported target-layer configs are single-block `[0]` and
multi-block `[0,2]` (see Priority-1 layer sweep).

### 10. Why multi-block on MuJoCo but single-block on CIFAR?
**The code does not encode a stated rationale.** What the code shows:
- MuJoCo: the headline wrapper sets `layer_scope="multi"` (perturb layers [0,2]); single-block
  is available (`layer_scope="first"`) but not the default.
- CIFAR: **both** single-block (`CIFARPnC`) and multi-block (`CIFARMultiBlockPnC`) exist and
  **both appear in the submitted CIFAR table** (`cifar_tables.txt`: "PnC Single Block
  scale=20.0", "PnC Multi Block scale=6.0"). So CIFAR is **not** single-block-only in the
  submission; both are reported.
The premise "MuJoCo multi-block vs CIFAR single-block" is therefore only partly accurate: the
MuJoCo main method is multi-block, but CIFAR reports both. No experiment log states *why* the
MuJoCo default is multi.

### 11. Was the block-mode decision fixed in advance / by ID-val / engineering?
**Cannot be established from the logs, so stated as such.** For CIFAR, block_mode is a
**swept candidate** in `CIFARSelectPnCPaperProtocol` (`block_modes=["single","multi"]`,
`:1611`), i.e. it *can* be chosen by the ID-only paper-protocol selector — but the submitted
table reports both single and multi as fixed named rows, not a single selected mode. For
MuJoCo, `layer_scope="multi"` is a hard-coded wrapper default, not selected. There is no log
establishing whether multi was fixed in advance or picked by ID validation; we do not invent
one. (Priority-1's layer sweep shows single vs multi block differ negligibly on HalfCheetah,
so the choice is not performance-critical there.)

### 12. Complete MuJoCo Near/Mid/Far tables (all methods, seeds)
Extracted to `results/neurips_2026_rebuttal/priority5_full_mujoco_tables.csv` (companion
extraction). Key point: **the harness computes Near-AUROC and Mid-AUROC for every method**
(`auroc_ood_near/mid` in every result JSON), but the submitted table
(`neurips_draft/gym_tables.tex`) reports **only Far AUROC**. The full Near/Mid/Far NLL **and
AUROC** for all methods and seeds are therefore recoverable from cached artifacts with no new
training — see the companion CSV.

### 13. Does OpenOOD CIFAR-10 provide Near/Far (not Mid)?
**Confirmed.** `evaluate_openood_cifar` iterates exactly `near_ood` and `far_ood`
(`openood_eval.py:143`); there is **no Mid group** on CIFAR (Mid exists only in the MuJoCo
path). Groups (`data.py:627`): **Near** = {CIFAR-100, Tiny-ImageNet-200}; **Far** = {MNIST,
SVHN, Textures, Places365}. So the MuJoCo 3-tier (Near/Mid/Far) and CIFAR 2-tier (Near/Far)
structures are genuinely different by benchmark design, not an omission.

### 14. Manuscript table/caption/cross-reference issues
- **Omitted Near/Mid AUROC (main issue):** `neurips_draft/gym_tables.tex` reports ID RMSE,
  Near/Mid/Far NLL, and only **Far AUROC** — Near/Mid AUROC exist in the data but are not
  shown. The `experiments.tex` text says "AUROC for distinguishing ID from the corresponding
  shifted regime … (Table~\ref{tab:gym-main})", implying per-regime AUROC, but the table gives
  only Far AUROC — a text/table mismatch.
- **Table-vs-appendix config drift:** the submitted `gym_tables.tex` PnC rows (size=8) do not
  match the later working tables (`gym_tables_paper.txt`, PnC rows blank `--`) or the appendix
  sizes ({5,10,20,50}); the cached canonical results use a different config than the Apr-8
  table (see REPRODUCTION.md).
- **Humanoid-v5** appears in `gym_tables.tex` (4th table) but the `experiments.tex` prose lists
  only "Ant-v5, HalfCheetah-v5, and Hopper-v5" — a scope mismatch between text and tables.
- (No LaTeX cross-reference/label errors detected in the provided `.tex`; `tab:gym-main` and
  `fig:geometry-transfer` are referenced in text — verify the corresponding `\label`s resolve
  in the full build.)
