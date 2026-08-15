# Ridge-centre provenance audit (Part B)

The canonical P&C correction is now

    Theta_hat = argmin_{Theta'} ||Y_v Theta'^T - Y Theta^T||_F^2 + lambda ||Theta' - Theta||_F^2

i.e. the ridge shrinks toward the **original affine map**. This audit establishes, for every
manuscript-facing P&C result, which centre was *actually* used.

**Method.** The centre was never read off prose, and in one case not even off the recorded
metadata — see the MuJoCo sensitivity row, where the `ridge_center` column turned out to be a
hardcoded literal that contradicts the code that wrote it. Evidence is, in order of
preference: (1) a numerical probe of the production solver
(`experiments/pnc_protocol/verify_ridge_center.py` →
`ridge_center_verification.json`), (2) the code path that constructed the ensemble, (3) the
filename/config tokens that encode the flag.

## Solver-level verification

Each production solver was called on a synthetic problem and compared against both closed
forms. Exactly one matched in every case.

| solver | used by | called as | rel. err vs original | rel. err vs zero | verdict |
|---|---|---|---|---|---|
| `pnc_theory.linalg.ridge_solve` | DistilBERT / Banking77 | `w_prior=Theta` | **0.0** | 6.4e-2 | original |
| `pnc_theory.linalg.ridge_solve` | (library default) | `w_prior=None` | 6.4e-2 | **0.0** | zero |
| `pnc_core.ensembles._ls_or_ridge_solve` | MuJoCo / PJSVD | `ridge_toward_orig=True` | **1.7e-07** | 6.4e-2 | original (float32) |
| `pnc_core.ensembles._ls_or_ridge_solve` | MuJoCo / PJSVD | `ridge_toward_orig=False` | 6.4e-2 | **1.6e-07** | zero (float32) |
| `pnc_core.pnc._ridge_regression_solve` | CIFAR conv block | production path | **9.0e-07** | 5.5e-2 | original (float32) |
| `imagenet_vit_pnc.pnc_core.cho_solve_shared` | ImageNet ViT | `w_prior=Theta0` | **~0** | large | original |

CIFAR deserves a note because it looks zero-centred at a glance: it parameterises the
*delta*, accumulating `b = M^T (T - Y w2_orig)` and solving `(H + lambda I) Delta = b`, then
returning `w2_orig + Delta`. Penalising `||Delta||^2` **is** penalising
`||Theta_hat - Theta||^2`. `conv2` is built with `use_bias=False`
(`pnc_core/models.py:339`), so the implicit original bias is exactly 0 and the bias row is
centred correctly too. The probe above confirms this numerically.

## Per-result audit

`lambda` is the value that actually entered the reported configuration. "Centre" is what the
run used, not what any label says.

| # | Experiment | Result / table | Source path | Git commit | lambda | Ridge centre actually used | Needs rerun? |
|---|---|---|---|---|---|---|---|
| 1 | MuJoCo headline (3 env x 5 seeds) | `reports/tables/gym_tables_paper.tex`, `gym_tables.txt` | `results/{Ant,HalfCheetah,Hopper}-v5/pjsvd_multi_least_squares_random_projected_residual_prob_lreg0.0001_bf0.1_k20_*.json` | untracked | **1e-4** | **zero** | **YES — done, Part C** |
| 2 | MuJoCo 11-env sensitivity, all 7 factors | `revision_results/mujoco_sensitivity/far_sensitivity_raw.csv` (14,520 rows, 11 env, 27 seeds) | same | untracked | anchor 1e-4; `ridge` factor sweeps 0 … 1.0 | **original** | **NO** — but the CSV's `ridge_center` column is **wrong**, see below |
| 3 | MuJoCo graded-shift / layer scope | `results/neurips_2026_rebuttal/priority1/sensitivity_mujoco.csv` | untracked | 1e-4 | **original** (same runner, `layer` factor) | NO |
| 4 | MuJoCo mechanism validation | `results/neurips_2026_rebuttal/priority3/mechanism_*.json` | untracked | per `pnc_theory.harness` | **original** (`harness.build_pnc` defaults `ridge_toward_orig=True`) | NO — revalidated in Part D |
| 5 | MuJoCo efficiency | `results/neurips_2026_rebuttal/efficiency_*.csv` | untracked | `cfg.lambda_reg` | zero at run time | **NO** — centre-invariant, see below |
| 6 | CIFAR headline 162-run grid | `revision_results/cifar_sensitivity/cifar10_ood_paper_table.md` | untracked | 1e-3 / 1e-2 | **original** | NO |
| 7 | CIFAR ridge sensitivity | `revision_results/cifar_sensitivity/` | untracked | swept | **original** | NO |
| 8 | CIFAR SCOD comparison | `revision_results/scod/` | untracked | 1e-3 | **original** | NO |
| 9 | CIFAR efficiency | `results/neurips_2026_rebuttal/efficiency_cifar_inference.csv` | untracked | 1e-3 | **original** | NO |
| 10 | DistilBERT (appendix) | `results/banking77_distilbert_pnc/` | untracked | 1e-3 | **original** | NO |
| 11 | ImageNet ViT preservation frontier | `imagenet_vit_preservation_frontier/` | `ccab71b` | **1000** | **original** | NO |
| 12 | ImageNet ViT baseline comparison | `imagenet_vit_baselines/` | `8d0e10c` | 1000 | **original** | NO |
| 13 | ImageNet ViT geometry follow-up (K sweep etc.) | `imagenet_vit_geometry_followup/` | `d65634e` | 1000 | **original** | NO |

## Two findings that needed resolving

**(a) The MuJoCo sensitivity CSV's `ridge_center` column is a hardcoded lie.**
All 14,520 rows read `ridge_center = "zero"`. That column is written as a literal string at
`results/neurips_2026_rebuttal/mujoco_sensitivity/scripts/run_sensitivity.py:163`:

```python
status=status, error_message=err, git_commit=GIT, ridge_center="zero",
```

It is never derived from the configuration. The configuration that was actually used is
`ANCHOR = dict(K=20, M=50, n_cal=4096, layers=[0, 2], toward_orig=True)` (line 50, whose own
comment reads *"ridge toward ORIGINAL"*), and every factor — including the `ridge` factor —
is dispatched through `cell()`, whose base config sets `toward_orig=A["toward_orig"]`
(line 210). The only `toward_orig=False` in the file is inside `anchor_scale()` (line 151),
which selects the perturbation scale by ID-val NLL at `lam=0.0`, where the centre is
mathematically irrelevant because the solver takes the minimum-norm least-squares branch and
ignores `w_prior` entirely.

So the sweep is original-centred and **does not need rerunning**; the column needs correcting.
This is verified empirically in `sensitivity_center_check.json` (Part E) by re-running cells
with the current code and comparing against the stored rows.

**(b) MuJoCo efficiency was measured under the zero centre, and that is fine.**
`experiments/scripts/efficiency_benchmark.py:294` constructs `PJSVDEnsemble` without passing
`ridge_toward_orig`, so it used the then-default zero centre. The two centres solve the *same*
linear system `(X^T X + lambda I) Theta_hat = rhs` and differ only in the right-hand side by
`+ lambda * Theta`, an O(p*d) vector add on an already-formed RHS. Construction cost,
peak memory and storage are therefore identical by construction, and the benchmark reports no
predictive metric. Marked no-rerun with that reason rather than silently.

## Untracked-provenance caveat

Rows 1–10 sit in directories that are **not tracked by git** (`results/{env}-v5/`,
`results/banking77_distilbert_pnc/`, `results/neurips_2026_rebuttal/{mujoco_sensitivity,
priority1,priority3}/`, `revision_results/`). Their "git commit" column is therefore the
commit of the *code*, not of the artifact, and the artifacts cannot be pinned by hash from
git history. Only the ImageNet rows (11–13) have artifact-level git provenance. This is a
pre-existing repository condition, not something this round introduced, but it limits how
strongly any of these rows can be said to be reproducible-by-checkout.

## Conclusion

Exactly one manuscript-facing result was produced with zero-centred ridge at a nonzero
lambda: **the MuJoCo headline table**. It is rerun in Part C. Every other reported P&C
result already uses the canonical original-centred correction, verified at the solver level.
