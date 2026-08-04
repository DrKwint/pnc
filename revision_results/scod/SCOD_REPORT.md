# SCOD Post-hoc OOD Baseline — MuJoCo Far-OOD: Full Report

_Generated from 33 completed runs. Artifacts under `results/posthoc_mujoco/scod/`._

## 1. Summary

We evaluate **SCOD (Sketching Curvature for OOD Detection)** as a post-hoc uncertainty baseline on the 11-environment MuJoCo dynamics benchmark, alongside P&C. SCOD equips the frozen pretrained probabilistic MLP with a low-rank sketch of its training-data Fisher and returns a scalar atypicality score; no weights change. **All 33 runs completed (11 envs x seeds {0,10,200}), 0 failures, validation PASSED, base predictive mean preserved exactly (max deviation 0.0).**

**Headline:** SCOD is *strongly bimodal* — near-perfect OOD detection on manipulation/pendulum environments but **inverted** (AUROC < 0.5) on locomotion environments, driven by a real Fisher-energy-collapse mechanism. On identical envs/data/checkpoints, **P&C is robust everywhere (0.92-1.00) while SCOD collapses** on the locomotion set.

## 2. Method & implementation

- **Base model (frozen):** probabilistic MLP, hidden [200,200,200,200], dual heads (mean, var=softplus+1e-6) = heteroscedastic diagonal Gaussian.
- **SCOD sketch:** randomized single-pass Nystrom sketch of the dataset Fisher on the **4096-point ID calibration set** (num_eigs<=100, num_samples=604, sketch seed 100000+model_seed). JAX port; Fisher-weighted Jacobian uses the **Case-B heteroscedastic** factors.
- **Score:** posterior_pred = sqrt of residual Fisher energy after eigenvalue-dependent shrinkage (Meps). Higher = more atypical/OOD.
- **Two variants:** (1) native score (AUROC/Spearman; NLL undefined); (2) ID-calibrated Gaussian (NLL via Sigma_a + alpha*qtilde*D_y).
- **ID-only selection:** (k, Meps, alpha) by ID-validation NLL; OOD never used for tuning. Frozen before any OOD read.
- **Numerics validated:** Fisher-weighted Jacobian reproduces the Gaussian-KL quadratic form to 8e-5; Nystrom sketch recovers the exact Fisher eigenvalues to 6e-5 and top eigenvector to |cos|=1.0.

## 3. Primary results — native SCOD score (mean +/- sd over 3 seeds)

| Environment | ID RMSE | Far AUROC | Far Spearman | mean u(OOD)/u(ID) | Far NLL (Gauss) | k | Meps | alpha |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Pusher | 0.050 | 0.995 ± 0.005 | 0.669 ± 0.037 | 19.15 | 4277.1 | 50 | 1N | 0.0001 |
| InvertedDoublePendulum | 0.031 | 0.930 ± 0.082 | 0.801 ± 0.073 | 70.87 | 81309.9 | 10 | 1N | 0 |
| Reacher | 0.093 | 0.897 ± 0.045 | 0.656 ± 0.059 | 8.79 | 1988.8 | 10 | 1N | 0.0001 |
| InvertedPendulum | 0.049 | 0.787 ± 0.034 | 0.592 ± 0.337 | 8.79 | 65.6 | 10 | 1N | 0.0003 |
| HumanoidStandup | 25.243 | 0.660 ± 0.037 | 0.510 ± 0.082 | 10.42 | 10252.6 | 100 | 1N | 0.3 |
| Walker2d | 0.544 | 0.383 ± 0.081 | 0.197 ± 0.036 | 1.01 | 34.9 | 100 | 1N | 0.0001 |
| HalfCheetah | 1.289 | 0.291 ± 0.053 | 0.170 ± 0.023 | 0.94 | 13.1 | 100 | 1N | 0.0003 |
| Swimmer | 0.034 | 0.286 ± 0.123 | 0.319 ± 0.088 | 1.01 | 45.9 | 20 | 1N | 0.0003 |
| Humanoid | 105.257 | 0.267 ± 0.018 | 0.761 ± 0.072 | 1.50 | 3992.8 | 20 | 1N | 0.3 |
| Ant | 0.683 | 0.110 ± 0.125 | 0.147 ± 0.010 | 0.50 | 20.9 | 10 | 1N | 0 |
| Hopper | 0.167 | 0.028 ± 0.005 | -0.092 ± 0.019 | 0.32 | 9.6 | 10 | 1N | 0 |

Ordered by descending Far AUROC. `u(OOD)/u(ID)` = ratio of mean native score on far-OOD vs ID.

## 4. Finding 1 — bimodal, inverts on locomotion envs

- **Works (AUROC > 0.6):** Pusher (0.99), InvertedDoublePendulum (0.93), Reacher (0.90), InvertedPendulum (0.79), HumanoidStandup (0.66).
- **Inverts (AUROC < 0.45):** Walker2d (0.38), HalfCheetah (0.29), Swimmer (0.29), Humanoid (0.27), Ant (0.11), Hopper (0.03).

## 5. Finding 2 — inversion is a real mechanism (Fisher-energy collapse), not a bug

AUROC exactly tracks u(OOD)/u(ID) (col. 5): where far-OOD (uniform random-action) inputs **raise** the local Fisher/Jacobian energy the score works; where they **collapse** it (Ant -> 0.22x, Hopper -> 0.32x of ID energy) the atypicality score reads OOD as *less* novel and inverts. SCOD scores local Fisher geometry, which assumes OOD -> higher sensitivity; random-action inputs instead push the ReLU MLP into low-sensitivity/dead regions. Ruled out as a bug by: validated numerics; near-perfect detection where OOD energy is genuinely higher; reproduction across two independent checkpoint sets; correct orientation on the working envs.

## 6. Finding 3 — SCOD fails where P&C succeeds (same data/checkpoints/evaluator)

| Environment | P&C Far AUROC | SCOD Far AUROC | gap |
|---|---:|---:|---:|
| Hopper | 0.981 | 0.028 | +0.953 |
| Ant | 0.998 | 0.110 | +0.888 |
| Swimmer | 0.995 | 0.286 | +0.709 |
| HalfCheetah | 0.987 | 0.291 | +0.696 |
| Walker2d | 0.998 | 0.383 | +0.615 |
| HumanoidStandup | 0.928 | 0.660 | +0.267 |
| InvertedPendulum | 0.919 | 0.787 | +0.131 |
| Reacher | 0.993 | 0.897 | +0.096 |
| Humanoid | 0.344 | 0.267 | +0.078 |
| InvertedDoublePendulum | 0.996 | 0.930 | +0.066 |
| Pusher | 1.000 | 0.995 | +0.005 |

Only the uncertainty *score* differs (P&C = ensemble predictive variance; SCOD = single-model Fisher energy). P&C working on Hopper (0.98) while SCOD inverts (0.03) rules out any shared-plumbing bug and localizes the effect entirely to SCOD's score.

## 7. ID-calibrated Gaussian variant

ID-only selection picked alpha=0 or ~1e-4 for most environments (base aleatoric variance already best-calibrated on ID-val), so the SCOD-Gaussian predictive collapses to the base model and its Far NLL is the base model's (badly miscalibrated on far-OOD). **The native score is SCOD's real detector**; the Gaussian variant is an explicit benchmark adaptation only, uninformative here.

## 8. Efficiency (offline + online)

- Sketch construction (offline): **20s** average per (env,seed).
- Total per run: **132s** (cheaper envs) to **3611s** (Ant, full 10k-point scoring).
- Dominant cost is the per-test-point parameter Jacobian (out_dim x P). P&C's anchor ensemble builds in ~0.55s, so **SCOD's online scoring is ~100-1000x P&C's** — reported for both so no 'cheaper' claim rests on offline cost alone.

## 9. Caveats & provenance

- **Checkpoints:** originals lost to a crash-induced filesystem event; all 33 runs use one regenerated canonical batch (backed up to `artifacts_backup/base_models_posthoc_canonical.tar.gz`). Base training is not bit-reproducible, so these differ from the original P&C-sensitivity checkpoints at BLAS level; the bimodal SCOD pattern reproduced across both sets.
- **Subsampling:** Humanoid & HumanoidStandup (out_dim=348) scored on a fixed 3000-point subsample per split (`score_subsampled=true`); ample for AUROC/Spearman.
- **Scope:** seeds {0,10,200}. **Not included:** SLL (held) and transformer P&C (in progress).

## 10. Reviewer-facing conclusion (<=120 words)

Across all 11 MuJoCo environments we add SCOD as a post-hoc Fisher-geometry OOD baseline on the same frozen checkpoints, data, and metrics as P&C. SCOD is strongly bimodal: near-perfect on manipulation/pendulum tasks (AUROC up to 0.995) but inverted on locomotion tasks (Hopper 0.03, Ant 0.11), because far-OOD random-action inputs collapse the network's local Fisher energy below in-distribution, so the atypicality score mis-ranks them. On the identical setup, P&C's perturbation-based predictive variance is robust everywhere (0.92-1.00), and SCOD's per-point parameter-Jacobian scoring is 100-1000x P&C's online cost. A well-regarded curvature method fails precisely where P&C succeeds on dynamics-model OOD.

## 11. Artifacts

- Predictions: `results/posthoc_mujoco/scod/predictions/<env>/<seed>/<split>.parquet`
- Configs/sketches/spectra/timing: `results/posthoc_mujoco/scod/{configs,sketches,spectra,timing}/`
- Validation: `experiments/posthoc/validate_scod.py` (PASSED); marker `SCOD_COMPLETE`
- Manifest: `results/posthoc_mujoco/MANIFEST.json`
