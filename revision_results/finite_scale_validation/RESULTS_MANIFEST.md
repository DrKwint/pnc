# Results manifest

Code commit `70fb480`. Every row is either a finished artifact on disk or an
explicit statement that the work has not been done. Statuses here are the
single source of truth; `FINITE_TRANSFER_REPORT.md` defers to this file.

All analysis is float64 on CPU. "New training?" records whether the row
required fitting new networks (as opposed to reusing the cached base models in
`artifacts/pnc_theory/base_models/`).

| experiment | status | artifact path | environments | seeds | members | headline output | new training? |
|---|---|---|---|---|---|---|---|
| Objective-specific finite residual identities | complete | artifacts/finite_transfer/mujoco_tier1_v2/identity.csv | Ant, HalfCheetah, Hopper, Humanoid | 5 | 5/variant | REPORT §3 + tables/part0_identity.md | no |
| Transfer defect vs disagreement / error | complete | artifacts/finite_transfer/mujoco_tier1_v2/part1.csv | Ant, HalfCheetah, Hopper, Humanoid | 5 | 50 | REPORT §4 + tables/part1_paired.md | no |
| Finite vs linearized vs unperturbed-design response | complete | artifacts/finite_transfer/mujoco_tier1_v2/part1.csv | Ant, HalfCheetah, Hopper, Humanoid | 5 | 50 | REPORT §4 ladder table | no |
| Hidden-code feature-space comparison | complete | artifacts/finite_transfer/mujoco_tier1/part2.csv | Ant, HalfCheetah, Hopper, Humanoid | 5 | 20 | REPORT §8 + tables/part2_feature_spaces.md | no |
| Calibration-response shuffle (Part 3A) | complete | artifacts/finite_transfer/mujoco_tier1/part3a.csv | Ant, HalfCheetah, Hopper, Humanoid | 5 | 20 | REPORT §5 + tables/part3a_shuffle.md | no |
| Layer x scale transfer-gap sweep | complete | artifacts/finite_transfer/mujoco_tier1/part5.csv | Ant, HalfCheetah, Hopper, Humanoid | 5 | 24 | REPORT §6 + tables/part5_transfer_gap.md | no |
| Calibration size and conditioning | complete | artifacts/finite_transfer/tier1_conditioning/conditioning.csv | Ant, Hopper | 5 | 24 | REPORT §7 | no |
| Synthetic transfer-failure causes (Part 7) | complete | generated in-process by synthetic.py | n/a (synthetic) | n/a | n/a | REPORT §9 + figures/fig9 | no |
| Geometry vs response-novelty decomposition | complete | artifacts/finite_transfer/geometry_novelty/summary.csv | Ant, HalfCheetah, Hopper, Humanoid | 5 | 50 | REPORT §10a + tables/p1_geometry_novelty.md | no |
| Novelty shuffle placebo | complete | artifacts/finite_transfer/geometry_novelty_shuffle/summary.csv | Ant, HalfCheetah, Hopper, Humanoid | 5 x 5 shuffles | 50 | REPORT §10a shuffle table | no |
| Common conditioning — Ant | complete | artifacts/finite_transfer/conditioning/methods.csv | Ant | 5 | 50 (DE 10) | REPORT §10 + tables/p1_conditioning_methods.md | yes (SWAG/Subspace/MCD/DE) |
| Common conditioning — beyond Ant | complete | artifacts/finite_transfer/conditioning_more_envs/methods.csv | HalfCheetah, Hopper, Humanoid | 5 | 50 (DE 10) | CONDITIONING_REPLICATION.md | yes |
| Directional covariance alignment (top-k) | complete | artifacts/finite_transfer/conditioning*/methods.csv | Ant, HalfCheetah, Hopper, Humanoid | 5 | 50 (DE 10) | tables/p3_directional_alignment.md | no (reuses conditioning) |
| Perturbation-site comparison | complete | artifacts/finite_transfer/perturb_sites/sites.csv | Ant, HalfCheetah, Hopper, Humanoid | 5 | 50 | REPORT §10b + tables/p5_perturb_sites.md | no |
| Canonical configuration audit | complete | source inspection + in-process measurement | Ant, HalfCheetah, Hopper, Humanoid | 0 (measured) | 6 | CANONICAL_CONFIG_AUDIT.md | no |
| Two-score analysis | complete | artifacts/finite_transfer/{mujoco_tier1_v2,geometry_novelty} | Ant, HalfCheetah, Hopper, Humanoid | 5 | 50 | TWO_SCORE_ANALYSIS.md + figE | no |
| Causal data completion — synthetic | complete | artifacts/finite_transfer/causal_synthetic/rows.csv | synthetic 2-D teacher-student | 5 | 40 | CAUSAL_DATA_COMPLETION.md + figF | yes (25 students) |
| Causal data completion — MuJoCo | complete | artifacts/finite_transfer/causal_mujoco/rows.csv | Ant | 5 | 40 | CAUSAL_DATA_COMPLETION.md Part B | yes (15 base models) |
| Canonical-regime rerun | complete | results/{env}/*ridgeorig* (80 luigi runs) | Ant, HalfCheetah, Hopper, Humanoid | 5 | 50 | CANONICAL_REGIME_RERUN.md | yes (shipped luigi path) |
| Continuation-family comparison | not started (optional) | - | - | - | - | - | yes (20+ models) |
| CIFAR confirmation | prepared, not run | configs/cifar_tier1.yaml + capture_conv.py | CIFAR-10 | - | - | conv algebra unit-tested only | no |
| Predictor-capacity sweep | deferred | - | - | - | - | - | - |
| GP-style covariance conditioning | deferred | - | - | - | - | - | - |

| Theory synthesis | complete | - | - | - | - | THEORY_SYNTHESIS.md | no |
| communication_v1 freeze | complete | reports/finite_transfer/communication_v1/ | - | - | - | README.md + MANIFEST.json | no |
| Communication figures A-F | complete | reports/finite_transfer/figures + tables/fig*_source.csv | as per source experiment | - | - | FIGURES.md | no |

## Frozen tables

`reports/finite_transfer/tables/` holds the numerical tables the report quotes.
They are frozen: `make_finite_transfer_report.py` refuses to overwrite an
existing table unless `--version SUFFIX` is passed, which writes
`<name>.<SUFFIX>.md` alongside the frozen original.

## Statistical protocol

* Seed is the replication unit; confirmatory claims are paired per-seed
  differences with 95% t-intervals computed within an environment.
* Pooled per-example correlations are labelled exploratory and carry no p-values.
* Within-regime results are reported separately from pooled regime detection.
* No OOD example or label is read by any selection step.
* Rank-R² increments are **not** used alone to validate novelty — the shuffle
  control showed sign-inverted noise can raise R². Signed partial correlations,
  paired AUROC differences and interventions are used instead.
* Native P&C is always reported separately from common-conditioned P&C, and
  every conditioned baseline is a mechanism diagnostic, not the published method.
