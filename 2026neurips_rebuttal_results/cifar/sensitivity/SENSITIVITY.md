# CIFAR P&C one-factor sensitivity — recovered

Anchor: s3b0 / ps25 / bf0.05 / K20 / M50 / λ1e-3 / calib1024. Two rebuttal sources (both 3-seed):
`sensitivity_cifar_{raw,agg}.csv` + `sensitivity_cifar_summary.md` (scale, rank, calib, ridge, block factors) and
`sweep_scalex_boot_{summary.md,agg.json}` (scale-multiplier + bootstrap-fraction refinement). Original submitted single-JSON sweeps also exist in `results/cifar10/pnc_single_block_*.json` (ps10–2000, bf0.05–0.8, λ1e-4–1.0, K5–40) — pointers in `../MANIFEST.md`.

## Findings vs manuscript qualitative claims (each labelled by evidence)
| claim | verdict | evidence |
|---|---|---|
| scale stable through ~25, degrades at larger | **SUPPORTED** | ps25 peak (Near 91.55); ps≥50 collapses ID (acc→35% @50, →10% @100); scale-multiplier: 1.5× (ps37.5) collapses (acc 9.99). Most sensitive knob. Corroborated by the full grid (s3b0 ps50→1.55, ps100→3.90 val NLL). |
| rank K flat above a low-rank threshold | **SUPPORTED** | rank sweep K∈{1,2,5,20,40}: flat above ~K5 (see `sensitivity_cifar_agg.csv` rank rows). |
| bootstrap exhibits interpolation/double-descent | **PARTIAL / see conflict** | bf∈{0.01,0.03,0.05}≈flat (Near 91.5); bf0.07 mild degrade; **bf0.1 catastrophic in the scalex_boot harness (acc 69.84±15.50)** — but the full joint grid has s3b0/ps25/bf0.1 stable (val NLL 0.1357). ⚠ Replicating instability in one harness only — a conditioning artifact, not a clean double-descent. The clearer double-descent is in **calib size** (below). |
| ridge stable over a broad nonzero range | **SUPPORTED (threshold form)** | λ=0 catastrophic (hard error seed1; acc→10% seeds0,2 — singular solve); λ=1e-4 degraded; **λ≥1e-3 flat over 3 orders (1e-3…1.0)**. Corrects any 'λ has no effect' phrasing. |
| early/middle/late blocks all viable | **SUPPORTED** | block sweep s1b0…s3b1 all functional (acc 94.6–95.6). Full per-block 3×3 tables in `../joint_selection/tables/validation_nll_by_block.md`. |

## Key instabilities (task-required record)
- **calib ss=2048:** severe replicating collapse in all 3 seeds (ID acc 89/41/42%) while ss∈{256,512,1024,4096} healthy (~95%) — a conditioning anomaly at the 2-chunk boundary (not sample scarcity). Recommended follow-up: condition-number probe.
- **bootstrap bf=0.1 (scalex_boot):** acc 69.84±15.50 collapse, contradicted by the stable full-grid bf0.1 — flagged as a harness/conditioning instability, not a robust effect.
- **λ=0:** singular-solve failure (recorded, not discarded).

## Scale-multiplier & bootstrap tables (3-seed) — from `sweep_scalex_boot_summary.md`
Perturbation scale ×anchor(ps25): 0.5×→Near 89.82, 0.75×→90.89, **1.0×→91.55**, 1.25×→90.86, 1.5×→64.08 (collapse).
Bootstrap fraction: bf0.01→91.54, bf0.03→91.55, **bf0.05→91.55**, bf0.07→90.73, bf0.1→69.25 (unstable, see conflict).
