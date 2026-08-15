# NeurIPS revision closure — experimental report

Branch `agent/revision-experiment-closure`. All artifacts in
`2026neurips_rebuttal_results/revision_experiment_closure/`. No manuscript TeX was edited.
No P&C hyperparameter was tuned on OOD data. The frozen ImageNet K=20, r=2, λ=1000 result is
unchanged, no covariance-aware perturbation was introduced, and no new target block was
searched.

---

## 1. Executive summary

**Do all final reported P&C experiments now use original-centered ridge?** **Yes.** Every
production solver was verified numerically against both closed forms rather than read from
prose (`ridge_center_verification.json`): DistilBERT 0.0 relative error vs the
original-centered form, ImageNet ~0, MuJoCo 1.7e-07 and CIFAR 9.0e-07 (both float32). Exactly
one manuscript-facing result had actually been produced under the zero centre — the MuJoCo
headline table — and it was rerun. Implementation defaults were flipped so it cannot recur.

**Did switching MuJoCo from zero- to original-centered ridge materially change any scientific
conclusion?** **No.** Over the 15 (env, seed) pairs of the three headline environments, every
median paired difference is ≈ 0 (largest |median| = 0.0036), Far AUROC changes by −0.0018 on
average with a CI spanning zero, and only `nll_ood_mid` has an interval excluding zero — a
small *improvement* of −0.0355. Ant and HalfCheetah agree to four decimals.

**Were any hyperparameters reselected?** **No, on the headline environments: 0 of 15 cells
changed the ID-validation-selected perturbation size.** One of 20 changed when Humanoid-v5 is
included (seed 200, 10 → 50), and Humanoid is not in the headline table: P&C sits at or below
chance there (Far AUROC 0.22–0.47) under *both* centres.

**Final canonical ImageNet results**, all from one evaluator over the same 50,000 validation
predictions and the same five OpenOOD sets:

| Method | Near AUROC | Far AUROC | ID error AUROC |
|---|---|---|---|
| **P&C** (r=2, λ=1000, K=20, M=20, 5 seeds) | **76.49 ± 0.09** | **87.89 ± 0.12** | 85.50 |
| Mahalanobis | 78.82 | 92.55 | 78.00 |
| LLLA-Kron | 72.95 | 87.42 | 82.40 |
| SCOD-linear | 75.15 | 88.48 | 85.61 |

**Are all headline ImageNet metrics now computed by the same evaluator?** **Yes** —
`experiments/imagenet_vit_pnc/final_eval.py`, one metric implementation, one score sign, one
family aggregation, with a sha256 of all 35 per-example files it consumed recorded in
`imagenet_final_provenance.json`.

**Measured cost of constructing and evaluating the ImageNet ensemble.** Full post-hoc
construction from a loaded checkpoint to 20 materialised members: **284.7 s**, of which the
single backbone pass over 32,768 images is 276.3 s (97%); the basis costs 0.86 s, all 20 sets
of ridge sufficient statistics 3.04 s, and all 20 ridge solves 4.48 s. Peak 1.49 GiB GPU /
5.12 GiB CPU. From a persisted feature cache the same work is **8.4 s**. Storage: 180.1 MiB
of P&C state on top of a 330.3 MiB checkpoint, i.e. **7.7%** of 20 full checkpoints (derived,
6605.8 MiB — the 20 checkpoints were never written). Inference at batch 64: base
100.4 img/s, P&C M=20 98.4 img/s (**1.020×**), marginal cost **0.0046 ms/image per member**.

---

## 2. Canonical original-centered P&C definition

    Theta_hat = argmin_{Theta'} ||Y_v Theta'^T − Y Theta^T||_F^2 + lambda ||Theta' − Theta||_F^2

The ridge shrinks toward the **original affine map**, so λ is *repair conservatism*: λ → ∞
returns the uncorrected perturbed member, not a collapsed one. Defaults changed:

- `pnc_core/ensembles.py::PJSVDEnsemble(ridge_toward_orig)`: `False` → **`True`**
- `pnc_core/gym_tasks.py::GymPJSVD.ridge_toward_orig`: `False` → **`True`**

Both flags survive for reproducing historical runs, and new MuJoCo runs emit a `_ridgeorig`
filename token so historical outputs are never overwritten.
`experiments/pnc_protocol/ridge.py` and `pnc_theory/harness.py` already defaulted to the
original centre.

**Regression tests** (`experiments/pnc_protocol/tests/test_canonical_ridge_center.py`,
30 passing): closed-form parity ≤ 1e-10 in float64 for both the summed and mean objectives;
λ = 0 makes the centre irrelevant when the LS solution is unique; λ → ∞ drives
‖Θ̂ − Θ‖ → 0 monotonically; the exact residual identity closes **without a β ridge-centre
term**, and inserting any nonzero β provably breaks it. A contrast test pins that
zero-centered ridge drives Θ̂ → 0 instead, so the two conventions can never again be
confused silently.

---

## 3. Ridge-centre provenance audit

Full table in `RIDGE_CENTER_AUDIT.md`. Result: **13 manuscript-facing results audited, one
needed a rerun.** Two findings deserve highlighting.

**The MuJoCo sensitivity CSV's `ridge_center` column is false.** All 14,520 rows read
`"zero"`, but that is a hardcoded string literal at `run_sensitivity.py:163`, never derived
from configuration — as is the `git_commit` column. The configuration actually used is
`ANCHOR = dict(..., toward_orig=True)` (line 50, whose comment reads *"ridge toward
ORIGINAL"*), and every factor including `ridge` dispatches through `cell()`, which sets
`toward_orig=A["toward_orig"]`. The single `toward_orig=False` is inside `anchor_scale()` at
λ = 0, where the centre is mathematically irrelevant. **The sweep was already
original-centered.** The literal has been replaced with a value derived from the config.

**MuJoCo efficiency was measured under the zero centre, and that is fine.** The two centres
solve the same linear system and differ only by `+ λΘ` on an already-formed right-hand side —
an O(p·d) vector add. Construction cost, memory and storage are identical by construction,
and the benchmark reports no predictive metric.

---

## 4. MuJoCo original-centered rerun

Byte-for-byte the historical invocation with one flag changed
(`experiments/scripts/run_mujoco_original_centered_ridge.sh`): same base checkpoints
(`neurips_minari`), splits, seeds, K=20, M=50, bootstrap, target layers, perturbation-size
grid, λ=1e-4 and evaluation data. 4 environments × 5 seeds = **20 runs, all rc=0**.

Selection followed the manuscript's own ID-only rule — perturbation size by lowest
ID-validation NLL, applied independently under each centre, so no shifted data entered the
comparison.

---

## 5. MuJoCo old-vs-new paired comparison

15 pairs over Ant-v5, HalfCheetah-v5, Hopper-v5; 10,000 bootstrap replicates over pairs,
seed 20260815, percentile intervals. Direction is **original minus zero**.

| metric | mean | median | max abs | 95% CI |
|---|---|---|---|---|
| ID RMSE | −0.0144 | −0.0002 | 0.2355 | [−0.0468, +0.0026] |
| ID NLL | −0.0191 | +0.0000 | 0.4026 | [−0.0777, +0.0141] |
| Near NLL | −0.0313 | +0.0013 | 0.2591 | [−0.0819, +0.0081] |
| **Mid NLL** | **−0.0355** | −0.0004 | 0.3434 | **[−0.0854, −0.0037]** |
| Far NLL | −0.0073 | −0.0036 | 0.0614 | [−0.0195, +0.0048] |
| Far AUROC | −0.0018 | +0.0000 | 0.0351 | [−0.0077, +0.0025] |
| Near AUROC | +0.0017 | +0.0001 | 0.0447 | [−0.0054, +0.0098] |
| Mid AUROC | −0.0003 | +0.0000 | 0.0307 | [−0.0064, +0.0055] |

Only Mid NLL's interval excludes zero, and it favours the new convention. Per-cell, Ant and
HalfCheetah are identical to four decimals except HalfCheetah seed 10; Hopper moves a few
points either way, consistent with the manuscript already describing it as the most delicate
environment. Including Humanoid inflates every mean (its ID RMSE is 42–140 and its Far AUROC
is below chance) without changing any median; both scopes are tabulated in
`mujoco_original_centered_summary.md`.

---

## 6. Theory / mechanism validation

`validate_closure_mechanism.py`, float64, Ant/HalfCheetah/Hopper × {ID, Near, Mid, Far} ×
8 members × 2 blocks. The identity checked is the exact decomposition that holds at **every**
λ under the original centre:

    r_S(x; v) = g_S(x; v) Theta + X_v(x) C,     C = Theta_hat − Theta

| env | worst max rel. error | worst median |
|---|---|---|
| Ant-v5 | 8.49e-15 | 4.83e-15 |
| HalfCheetah-v5 | 1.81e-14 | 1.10e-14 |
| Hopper-v5 | 2.36e-14 | 1.56e-14 |

**Worst case 2.36e-14 — three orders inside the 1e-11 target.** No β ridge-centre term
appears anywhere.

**A caveat the manuscript should absorb.** At the operating λ = 1e-4 the correction term
*dominates*: ‖X_v C‖/‖r_S‖ is 1.17–15.74 depending on env and regime. So the simpler
statement `r_S ≈ Theta g_S` is **not** valid at the operating point; it is the λ → ∞ limit.
Measured on Hopper: ‖C‖/‖Θ‖ falls 1.36 → 7.96e-04 as λ goes 1e-4 → 1e8, and the `Theta g_S`
approximation's error falls 2.56 → 8.05e-03 over the same range. **The theory section should
state the exact identity, not the limit.**

---

## 7. Sensitivity reruns

**Ridge sweep (required).** Regenerated under the canonical convention across all 11
environments × 5 seeds × 9 λ = **495 cells, all completed**
(`mujoco_ridge_sweep_original_centered.csv`, table in
`mujoco_ridge_sensitivity_original_centered.md`). Far AUROC is 0.9196 at λ=0, peaks at
**0.9267 at the operating λ=1e-4**, and stays within 0.010 across λ = 1e-5…1e-2 before
degrading gently and monotonically (0.8967 at λ=0.1, 0.8832 at λ=1) — the graceful behaviour
expected when λ shrinks toward the original map.

**Other factors: no rerun, with evidence.** Per the audit they were already
original-centered, so the stated trigger ("nonzero *zero-centered* ridge materially entered
the configuration") does not fire. Two supporting measurements:

- Over the reported λ range the centre is empirically immaterial: ID RMSE differs by < 0.02%
  between centres for λ ≤ 1, and the curves separate only above λ ≈ 100 (1.07× vs 1.26× at
  λ=1e4). `sensitivity_center_shape.json`.
- The stored sweep is **not bit-reproducible** from the current tree. The runner had been
  broken by the `pnc_core/` reorganisation (flat `util` / `metrics` imports, now repaired),
  and once repaired it does not reproduce stored values *even at λ = 0*, where the centre
  provably cannot matter. So the mismatch is unrelated to the ridge question — but it means
  rerunning the remaining 11,883 rows would produce a fresh dataset that is still not
  comparable to the stored one, without changing any centre. **Flagged for review rather than
  run.** `sensitivity_center_check.json`.

---

## 8. CIFAR audit

**No rerun.** CIFAR looks zero-centered at a glance because it parameterises the *delta*:
it accumulates `b = M^T (T − Y w2_orig)` and solves `(H + λI) Δ = b`, returning
`w2_orig + Δ`. Penalising `‖Δ‖²` **is** penalising `‖Θ̂ − Θ‖²`. `conv2` is built with
`use_bias=False` (`pnc_core/models.py:339`), so the implicit original bias is exactly 0 and
the bias row is centred correctly too. The numerical probe confirms it: 9.0e-07 relative
error against the original-centered closed form versus 5.5e-02 against the zero-centered one.
This covers the headline 162-run grid, the ridge sensitivity, the SCOD comparison and the
efficiency run, all of which share this solver.

---

## 9. ImageNet audit

**No rerun.** Every ViT solve passes `w_prior=Theta0 = ad.theta() = [b2; W2]`, the original
map — verified across `frontier.py`, `frontier_final.py`, `full_pnc.py`, `fu_ksweep.py`,
`stages_correction.py` and `validate.py`. This covers the preservation frontier, the
r=2/λ=1000 primary result, the matched uncorrected ablation, the K sweep and the geometry
follow-up.

**λ = 1000 is strong shrinkage toward the original FFN second affine map**, recorded
explicitly in `imagenet_final_provenance.json`. One caveat for the manuscript: λ is on the
*summed* objective, so 1000 over 32,768 correction rows is a mean-objective λ of ≈ 0.031 —
not comparable at face value to MuJoCo's 1e-4. See `FINAL_PNC_PROTOCOLS.md`.

---

## 10. DistilBERT audit

**No rerun.** `experiments/banking77_pnc/construct.py:58` calls
`ridge_solve(Yv_aug, Z[idx], ridge, w_prior=Theta)` — original-centered, exactly (0.0
relative error in the probe). Scope unchanged; it remains appendix evidence.

---

## 11. Final ImageNet baseline comparison

Frozen rows, all through one evaluator (`imagenet_final_scores.csv`,
`imagenet_final_per_dataset.csv`, `tables_final/imagenet_vit.tex`):

| Method | ID err. AUROC | Near AUROC | Far AUROC | Near FPR95 | Far FPR95 |
|---|---|---|---|---|---|
| MSP | 85.61 | 73.52 | 86.04 | 81.84 | 51.74 |
| Base entropy (T=0.7) | 86.81 | 74.53 | 86.42 | 75.74 | 49.23 |
| Mahalanobis | 78.00 | **78.82** | **92.55** | 66.36 | 30.23 |
| LLLA-Kron | 82.40 | 72.95 | 87.42 | 85.93 | 55.66 |
| SCOD-linear | 85.61 | 75.15 | 88.48 | 73.18 | 43.37 |
| Uncorrected perturbation | 81.33 | 72.53 ± 0.30 | 84.40 ± 0.27 | 71.66 | 46.51 |
| **P&C** | 85.50 | 76.49 ± 0.09 | 87.89 ± 0.12 | 67.65 | 44.97 |

Appendix rows (Energy, base entropy raw, LLLA-Kron+Temp, SCOD-FFN, P&C mutual information
and expected member entropy) are in the same CSV. **SCOD scope was not selected using OOD** —
both `linear` and `ffn` were run and both are reported; SCOD-linear is the main comparator
because it is the method-faithful restricted scope that ran at the paper's own sketch rank.
Scope labels are kept explicit everywhere.

**Metric discrepancy resolved (§15 of the brief).** The three circulating P&C Far AUROC
values were three different evaluation artifacts, not disagreements about the model:

| value | what it actually was |
|---|---|
| 87.74 | a seed-0-only recomputation inside the geometry round |
| **87.89** | **the frozen 5-seed preservation-frontier primary — canonical** |
| 87.93 | the K-sweep's own K=20 arm, rebuilt from a nested orthonormal basis |

One trap the canonical evaluator pins down: the frontier stores its OOD scores as a single
pooled 85,908-vector concatenated in **alphabetical** dataset order (inaturalist, ninco,
openimage_o, ssb_hard, textures), while most other artifacts use the DS order with ssb_hard
first. Splitting with the wrong order silently scrambles every per-dataset number.

---

## 12. Final ImageNet statistical comparisons

Paired bootstrap, 10,000 replicates, seed 20260815, percentile intervals. Each replicate
draws one set of example indices shared by every method, and multi-seed methods are averaged
over seeds *within* the replicate so the interval is on exactly the quantity the table
reports. `imagenet_bootstrap.csv`; per-dataset point differences in
`imagenet_bootstrap_per_dataset.csv`.

| comparison | Near ΔAUROC | Far ΔAUROC |
|---|---|---|
| P&C − Mahalanobis | −2.33 [−2.51, −2.15] | −4.66 [−4.81, −4.52] |
| P&C − LLLA-Kron | +3.54 [+3.31, +3.77] | +0.48 [+0.34, +0.62] |
| P&C − SCOD-linear | +1.34 [+1.23, +1.45] | **−0.59 [−0.67, −0.51]** |
| P&C − MSP | +2.97 [+2.78, +3.16] | +1.86 [+1.73, +1.98] |
| P&C − Uncorrected | +3.95 [+3.81, +4.10] | +3.49 [+3.37, +3.61] |

**Every interval excludes zero.** Verbal claims that are safe: P&C beats its uncorrected
ablation, MSP and LLLA-Kron on both families; Mahalanobis beats P&C on both; SCOD-linear
beats P&C on Far while P&C beats it on Near. A claim that is **not** safe: that P&C is
competitive with Mahalanobis on Far — the gap is 4.66 points with a 0.29-point-wide interval.

---

## 13. Final ImageNet mechanism summary

`imagenet_mechanism_summary.csv`, regenerated through one script with per-row provenance. No
new search, no new P&C variant. Contents: P&C Near/Far AUROC at K ∈ {5, 20, 40, 80};
random-projection Mahalanobis at K ∈ {5, 20, 40, 80, 160, 320}; highest- vs lowest-variance
96-mode band AUROC (68.32 vs 81.81 Near); ρ(response energy, covariance eigenvalue) = +0.9925
and ρ vs inverse eigenvalue = −0.9925; geometry-only (282) and P&C-only (16) case counts;
base, P&C-entropy and P&C-mutual-information AUROCs.

---

## 14. ImageNet construction / storage / inference accounting

Frozen primary config, TITAN X (Pascal), `imagenet_efficiency_full_construction.json`,
`imagenet_inference_m_scaling.json`.

**Construction**, from a loaded checkpoint and prepared manifests to 20 materialised members,
using the cached implementation recommended for practical use (one backbone pass, sufficient
statistics reused across members) — not the naive preflight:

| stage | time |
|---|---|
| feature cache (one backbone pass, 32,768 images) | 276.3 s |
| perturbation basis + coefficients | 0.86 s |
| ridge sufficient statistics, 20 members | 3.04 s |
| ridge solves, 20 members | 4.48 s |
| **total** | **284.7 s** |
| peak GPU / peak CPU RSS | 1.49 GiB / 5.12 GiB |

97% is the single backbone pass. Re-running from a persisted feature cache takes **8.4 s**,
which is the marginal cost of building a *different* ensemble on the same base model. Dataset
download is excluded.

**Storage**

| item | size |
|---|---|
| base checkpoint | 330.3 MiB |
| P&C compact representation | 180.1 MiB |
| — per-member coefficients | 0.002 MiB |
| — per-member corrected W2, b2 | 180.1 MiB |
| base + P&C | 510.3 MiB |
| 20 full checkpoints (**derived**, never written) | 6605.8 MiB |
| **ratio** | **7.7%** |

The shared basis (180.0 MiB if materialised) is regenerable from (seed, K) and is not part of
the shipped representation. Essentially all P&C state is the 20 corrected W2 matrices.

**Inference**, batch 64 after warm-up:

| config | img/s | ms/image | ratio | peak VRAM |
|---|---|---|---|---|
| base ViT-B/16 | 100.4 | 9.96 | 1.000× | 0.85 GiB |
| P&C M=1 | 99.0 | 10.10 | 1.014× | 0.79 GiB |
| P&C M=20 | 98.4 | 10.16 | **1.020×** | 0.96 GiB |
| P&C M=50 | 96.8 | 10.33 | 1.037× | 1.22 GiB |

**P&C retains O(M) member inference, and no inference-time advantage is claimed.** The
marginal cost is 0.0046 ms/image per member (0.05% of a base forward) because members act on
the **CLS row only at the final block**, which is tiny next to the shared 12-block,
197-token backbone. That slope is a property of this target layer: an earlier target, or a
correction over all tokens, would make it much steeper.

*Correction to an intermediate measurement:* a first pass compared base and P&C across
separate processes and reported P&C at 0.976× base, i.e. apparently faster. The
single-process M-scaling measurement above supersedes it; the ratio is 1.02×, which is the
physically sensible answer.

---

## 15. Cross-experiment protocol consistency

`FINAL_PNC_PROTOCOLS.md` tabulates all four domains against base architecture, target layers,
perturbed and corrected affine maps, correction rows, ridge centre, λ, K, M, correction N,
bootstrap, selection criterion, predictive aggregation and uncertainty score.

**The ridge-centre column reads "original affine map" in all four rows.** That was the gate
for finalising tables; it passes.

Two lesser inconsistencies are recorded rather than fixed, because fixing them would change
reported numbers: λ is not on a common scale across domains (summed vs mean objective — see
§9), and the bootstrap fraction differs by domain (a genuine protocol difference).

---

## 16. Canonical manuscript numbers

Generated programmatically by `experiments/scripts/make_closure_tables.py` into
`tables_final/`. Nothing was transcribed by hand.

| table | status | source |
|---|---|---|
| `imagenet_vit.tex` | generated | `imagenet_final_scores.csv` |
| `imagenet_vit_per_dataset.tex` | generated | `imagenet_final_per_dataset.csv` |
| `mujoco_per_env.tex` | generated | `mujoco_original_centered_per_env.csv` |
| `mujoco_cross_env.tex` | generated | `mujoco_center_paired.csv` |
| `mujoco_ridge_sensitivity.tex` | generated | `mujoco_ridge_sensitivity_original_centered.csv` |
| `efficiency.tex` | generated | `imagenet_efficiency_full_construction.json`, `imagenet_inference_m_scaling.json` |
| `cifar_ood.tex` | **`\PARTIAL`** | no canonical CIFAR result file exists in this round |
| `distilbert.tex` | **`\PARTIAL`** | no canonical DistilBERT result file exists in this round |

The two incomplete tables emit a `\PARTIAL{...}` marker rather than a plausible number, so a
missing input cannot be mistaken for a result. Their ridge centres are audited and correct
(§8, §10); what is missing is a canonical *result* artifact in this round's directory, since
neither needed rerunning and neither has a per-example score file here to re-evaluate from.

---

## 17. Remaining unresolved issues

1. **The MuJoCo scope in the brief does not match the repository.** The brief refers to an
   "11-env headline" and a "10-seed headline experiment". Neither exists here: P&C MuJoCo
   data covers 4 environments, and the manuscript-locked configuration covers Ant,
   HalfCheetah and Hopper × 5 seeds (0, 10, 42, 100, 200). The 11-environment / 27-seed
   artifact is the *sensitivity* sweep, not the headline. I reran the 4 × 5 that actually
   back the manuscript rather than inventing seven new environments. **Needs your
   confirmation that this is the intended scope.**

2. **The remaining sensitivity factors were not rerun** (§7). They were already
   original-centered, the centre is immaterial over the reported λ range, and the stored
   artifact is not bit-reproducible anyway. Rerunning 11,883 rows would create a fresh
   dataset without changing a centre. **Flagged for review, per the brief's own instruction
   to ask before spending large compute.**

3. **Most result artifacts are untracked by git** — `results/{env}-v5/`,
   `results/banking77_distilbert_pnc/`, `results/neurips_2026_rebuttal/{mujoco_sensitivity,
   priority1,priority3}/`, and all of `revision_results/`. Their "git commit" provenance is
   the commit of the *code*, not of the artifact. Only the ImageNet rows have artifact-level
   git provenance. Pre-existing, but it limits reproducible-by-checkout claims.

4. **`cifar_ood.tex` and `distilbert.tex` are `\PARTIAL`** (§16). Producing them needs either
   a canonical per-example score file for each, or an explicit decision to carry the existing
   numbers forward with a provenance note.

5. **λ is not on a common scale across domains** (§9, §15). Worth a sentence in the
   manuscript rather than a rerun.

6. **The theory statement needs adjusting**: at the operating λ the correction term dominates
   the transfer defect, so the exact identity `r_S = g_S Θ + X_v C` should be stated rather
   than the λ → ∞ limit `r_S ≈ Θ g_S` (§6).

7. **Two scripts were repaired in place** to run at all: `run_sensitivity.py` (flat
   `util`/`metrics` imports broken by the `pnc_core/` reorg, plus the hardcoded
   `ridge_center` literal). The repairs are behaviour-preserving for the ridge path but they
   do mean the file now differs from the one that produced the stored sweep.
