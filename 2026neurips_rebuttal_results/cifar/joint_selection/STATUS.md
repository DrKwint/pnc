# Joint block × scale × bootstrap experiment — STATUS

## FULL_CROSS_PRODUCT_FOUND

A complete Cartesian product over **target block × perturbation scale × bootstrap fraction** was
run this rebuttal cycle. It is distinct from coordinate descent (the submitted appendix describes
coordinate descent in order block → scale → bootstrap).

### The grid (from the code/output, not assumed)
- **6 target blocks:** `s1b0, s1b1, s2b0, s2b1, s3b0, s3b1` (stage_idx∈{1,2,3} = Flax stage2/3/4, block∈{0,1}). See `BLOCK_PATH_AUDIT.md`.
- **3 perturbation scales:** 25.0, 50.0, 100.0
- **3 bootstrap fractions:** 0.05, 0.10, 0.20
- **Fixed:** K=20 directions, M=50 members, λ=1e-3, calib subset 1024, random directions, frozen BN, toward-original ridge. (chunk_size 64 for stage_idx≥3 else 16 — memory only, math-invariant.)
- **6 × 3 × 3 = 54 configs/checkpoint × 3 checkpoints (seeds 0,1,2) = 162 runs.** All 162 completed OK (0 failures) — see `all_162_candidates.csv`.

### Selection (ID-validation only — evidence)
- Selection metric = **mean temperature-scaled ID-validation NLL** across the 3 checkpoints; tie-break: std, then lex(stage,block,scale,frac). No OOD loaded during any candidate run (val-only builder; OOD benchmark imported only in the post-freeze OOD eval). See `GRID_SPEC.json`, `tables/full_grid_aggregate.csv`.
- **Global optimum = `s3b0 / ps25 / bf0.05`** (mean val NLL 0.1305 ± 0.0049), `selected/global_config.json` (SHA-256 recorded), frozen at `selected/FREEZE_TIMESTAMP`.
- **This is identical to the coordinate-descent config** (s3b0/ps25/bf0.05), which ranks **1/54**. Val-NLL regret = **0.0000**; well below the median seed std (0.0067). See `tables/coordinate_vs_full_grid.md`.
- Both forward (block→scale→bootstrap) and reverse (bootstrap→scale→block) coordinate-descent path reconstructions converge to the same global optimum. See `tables/cd_paths.json`, `interaction_summary.md`.

### Interactions (main-effect dominated, no reversal)
- ps=25 is best in every block; bf=0.05 best in 5/6; s3b0 best overall. Large scale is catastrophic at deep blocks (s3b0: ps25→0.13, ps50→1.55, ps100→3.90) — a scale *main effect* whose magnitude varies by block, not an interaction that changes the optimal value. Per-block 3×3 tables: `tables/validation_nll_by_block.md`.

### Post-freeze OOD eval (frozen config only, after selection)
- `final_metrics_summaries/` — 3-seed OpenOOD of the frozen s3b0/ps25/bf0.05 config: Near AUROC 91.55±0.16 / FPR95 33.12, Far AUROC 95.10±0.62 / FPR95 18.14 (matches the submitted anchor).

## Provenance
- **Original path:** `results/neurips_2026_rebuttal/cifar/pnc_full_grid/` (repo DrKwint/pnc, branch neurips-2026-rebuttal, commit 854f4d8, working-tree/uncommitted at run time).
- **Launchers/aggregation:** `experiments/pnc_grid/{setup,run_grid,candidate,aggregate_select,ood_eval,report}.py` (copied to `../scripts/`).
- **Base checkpoints:** `results/cifar10/preact_resnet18_train_e300_..._seed{0,1,2}.pkl` (hashes verified == SCOD manifest; see `checkpoint_manifest.json`).
- **Conclusion for the manuscript:** the full cross-product confirms the coordinate-descent selection reached the true ID-validation optimum — coordinate descent did **not** miss a material interaction. This closes the search-procedure gap; the one-factor sweeps remain a separate robustness claim.
