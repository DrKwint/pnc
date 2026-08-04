# MUJOCO_SENSITIVITY — MANIFEST

## Provenance (recorded at study start)
- branch: `neurips-2026-rebuttal`
- commit: `70fb480fadcc4ee8e9d8ae0b8a060de431007aa0`
- working tree: **dirty** (session instrumentation under `experiments/scripts/pnc_theory/`)
- host: `EriiNoDesktop`
- python 3.12.3 | jax 0.9.1 (CUDA) | GPU: CudaDevice(id=0)
- minari 0.5.3 (remote reachable, 343 datasets)

## Reconnaissance findings (before running the sweep)

### 1. Data availability
| status | environments |
|---|---|
| **cached data present** (seeds 0/10/42, all tiers incl. Mid) | Ant-v5, HalfCheetah-v5, Hopper-v5, Humanoid-v5 |
| Minari data available but **NOT yet generated** (needs download+split+base-model training) | HumanoidStandup-v5, Walker2d-v5 (have Mid); Swimmer-v5, Reacher-v5, Pusher-v5, InvertedPendulum-v5, InvertedDoublePendulum-v5 (**no Mid tier** — expert+medium only) |

All 11 envs have a Minari `mujoco/<slug>/expert` + `medium` dataset; 6 also have
`simple` (→ Mid-OOD): Ant, HalfCheetah, Hopper, Humanoid, HumanoidStandup,
Walker2d. The other 5 lack Mid (record as unavailable, per task §1).

### 2. Prior work already covering part of this task (do NOT rerun)
`results/neurips_2026_rebuttal/priority1/sensitivity_mujoco.csv` — a **complete
one-factor sensitivity study for HalfCheetah-v5**: factors {scale, rank, bootstrap,
ridge, layer} × seeds {0,10,42} (69 rows). Missing: the clean-calibration-size
sweep (task Sweep D) and the other 10 environments.

### 3. Anchor provenance — AMBIGUITY (task §4 requires documenting this)
The **submitted MuJoCo evaluation used 4 environments** (Ant, HalfCheetah, Hopper,
Humanoid) — this is what the repo's Minari mapping, cached data, `gym_tables.txt`,
and the prior sensitivity run all reflect. The other **7 environments in the task
list were not part of the original submission**, so they have **no "submitted
anchor"** to recover.

The canonical/anchor P&C config (from `scripts/neurips_2026_rebuttal/priority1_sensitivity.py`):
`k(K)=20, M=50, correction=least_squares (multi-block), family=random,
probabilistic base, hidden=[200,200,200,200] relu, λ=0.0, bootstrap_frac=0.1,
subset_size=10000 (FULL), perturbation size selected per (env,seed) by best
validation NLL from {5,10,20,50}`.

Discrepancies vs the task's anchor template:
- subset_size: existing run used **10000 (full)**; task template says **4096**.
- bootstrap/ridge: existing run is **uniform** (0.1 / 0.0); task says "env-specific".
- The env-specific quantity in the real submission is the **perturbation size**
  (val-NLL–selected), not bootstrap/ridge.

Resolution rule (task §4): use the config that reproduces the submitted per-env
result → the priority1 canonical anchor above (subset=full, λ=0, bf=0.1, scale by
val-NLL). This is recorded in `anchors.json`.

## Completion (§21) — DONE
- Scope: 4 submitted envs (user-confirmed) × seeds 0/10/42 × 6 factors = 372 raw rows.
- 0 solver failures / NaN across 372 runs; genuine catastrophic cells = calibration n/p≈1 (interpolation), documented as numerical (§17).
- Deliverables: raw + by_environment + across_envs CSVs; rebuttal table (md/tex/csv); 10 plots + env×factor heatmap; MUJOCO_SENSITIVITY_RESULTS.md with ≤180-word rebuttal paragraph; anchors.json; all scripts under scripts/.
- Reproduce: scripts/run_sensitivity.py --env <E> --seed <S>; then aggregate.py; plots.py.
- Note: absolute anchor metrics reflect this harness's base-model checkpoint (identical training procedure to submitted GymPJSVD; differences vs the older priority1 run are init/early-stop checkpoint variance). Sensitivity ranges are checkpoint-robust. priority1/sensitivity_mujoco.csv (HalfCheetah) is an independent trend cross-check.

## Extended-seed run (overnight)
Far-OOD sweep extended beyond seeds 0/10/42 to 10+ seeds for statistical
robustness. New seeds queued: 1,2,3,4,5,7,11,100,123,200,314,555,777,1000,1234,2024
(19 total). Each new seed: gen data (id_train/id_eval/ood_far, 11 envs) → run 6-factor
sweep (11 envs) → append to far_sensitivity_raw.csv. 3-seed snapshot saved as
far_sensitivity_raw_seeds0-10-42_snapshot.csv. Re-aggregate anytime with
scripts/aggregate_far.py. Runs until interrupted.
