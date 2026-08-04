# Phase-0 Reproduction Gate

Goal: confirm the current evaluation pipeline reproduces reported metrics within tolerance,
using existing artifacts, before launching any sweeps.

## Ant-v5 (MuJoCo) — PASS, exact
- **Config reproduced:** `PJSVD-Multi-LS random, projected_residual, prob, bf0.1, k20, n50,
  ps[5,10,20,50], hidden[200×4], seed0` — the current canonical cached config
  (producing script `experiments/scripts/run_ant_bf0.1_fullxsub_all_seeds.sh`).
- **Method:** `scripts/neurips_2026_rebuttal/repro_ant.py` subclasses `GymPJSVD` and redirects
  output to `results/neurips_2026_rebuttal/repro/` (cached file never touched); reuses cached
  Minari data (no download); base MLP retrained (12–13 s, seeded).
- **Result:** `max |repro − cached| = 0.000000` across all 13 reported fields
  (RMSE/NLL id+val, RMSE/NLL/AUROC × near/mid/far). Size selected by ID val-NLL = 10 (match).
  Detail: `repro/reproduction_ant.{md,json}`.
- **Verdict:** the pipeline is deterministic on this GPU and reproduces its reported MuJoCo
  numbers bit-for-bit. ✅

## CIFAR-10 — BLOCKED (cannot reproduce from existing artifacts)
Two hard blockers, both requiring new resource commitments:
1. **No base PreActResNet-18 checkpoints exist** anywhere in the repo/home (all
   `preact_resnet18_train_e300..._seed{0,1,2}.pkl` are absent). → would require retraining
   3×300-epoch ResNets (~hours each on the 12 GB TITAN X Pascal).
2. **No OpenOOD datasets cached** (`openood_data/` absent). → would require downloading +
   preprocessing CIFAR-100, TinyImageNet-200, MNIST, SVHN, Textures, Places365 (several GB).
Cached CIFAR *result JSONs* remain (they are the reproduction targets), but the inputs to
regenerate them do not. **Decision required** before spending GPU-hours + downloads.

## Discrepancies surfaced (kept, not hidden)
- **Manuscript↔cached config drift (Ant):** the Apr-8 submitted table numbers
  (`neurips_draft/gym_tables.tex`, PJSVD size=8, Far AUROC 0.833/0.896) come from a config
  **not present** in the cached results; the current canonical config (bf0.1, grid
  {5,10,20,50}) gives Far AUROC ≈0.998. This is a *configuration* difference, not a pipeline
  fault — the pipeline reproduces the cached config exactly. To be documented in
  `protocol_clarifications.md`.
- **Omitted Near/Mid AUROC** in the submitted MuJoCo table (only Far AUROC shown); near/mid
  AUROC are computed and cached in every P&C JSON and are trivially extractable.

## What is unblocked and ready now (MuJoCo track)
- Cached data + fast (~40 s/config) reproducible pipeline for Ant (11 seeds) and Hopper
  (8 seeds); HalfCheetah/Humanoid data also present.
- `pnc_repro/artifacts/` already holds theory-bridge (`panel_c_diagnostic_*`) and mechanism
  (`pnc_bridge_hidden_mahal_*`) data across 11 MuJoCo envs — a strong head start for
  Priorities 2 & 3 on the MuJoCo side.
