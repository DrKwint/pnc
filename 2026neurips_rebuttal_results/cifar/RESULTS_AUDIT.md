# CIFAR-10 Results Audit — Perturb-and-Correct manuscript

Forensics + provenance recovery for the CIFAR-10 experiments. Working machine: the CIFAR machine
(`/home/elean/pnc`, repo DrKwint/pnc, branch neurips-2026-rebuttal, commit `854f4d8`). No experiments
were rerun. Everything below is recovered from disk and re-verified by recomputation.

## ⚠ Access blocker (affects delivery, not the audit)
The paper repo **`EtherealEq/perturb_and_correct` is not accessible from this machine** (SSH: repository-not-found; HTTPS: no credentials; `gh` not installed; no local clone anywhere under `/home/elean` or `/mnt`). The deliverable bundle was therefore built **locally** at `/home/elean/pnc/2026neurips_rebuttal_results/cifar/` on a new branch **`agent/collect-cifar-results`** in the experiment repo. **The commit + draft-PR into the paper repo must be completed by someone with access** — see "Handoff" at the end.

## ⚠ Preservation risk
All rebuttal-era CIFAR artifacts (`results/scod_cifar/` 5.1G, `results/neurips_2026_rebuttal/cifar/` 605M, `results/cifar10/` 4.5G) are **git-ignored / untracked** — protected by no commit. Only report-copies exist off-machine (Windows OneDrive Desktop). This compact bundle is now the first tracked snapshot; the raw trees remain loss-exposed if the working tree is cleaned.

---

## Executive summary

**What was found (nearly everything):**
- **Headline OpenOOD table** (12 methods × 3 seeds) — reproduces **exactly** from `results/cifar10/openood_v1p5_*_seed{0,1,2}.json`. → `headline/`.
- **SCOD-1024** matched comparison — recovered and **verified exact** (95.74/89.69/39.40/92.56/21.41). → `scod/`.
- **The full block × scale × bootstrap cross-product EXISTS** — a complete 6×3×3×3 = 162-run grid selected by ID-val NLL. `joint_selection/STATUS.md` = **FULL_CROSS_PRODUCT_FOUND**. The global ID-val optimum equals the coordinate-descent choice (s3b0/ps25/bf0.05, rank 1/54, regret 0). → `joint_selection/`.
- **Sensitivity** (scale, rank, ridge, calib, block, bootstrap) — recovered (rebuttal 3-seed CSVs + original submitted JSONs). → `sensitivity/`, `block_scope/`.
- **Efficiency** — construction/inference/storage/memory recovered and **each number classified** MEASURED/DERIVED/ESTIMATED/UPPER_BOUND. GPU peak/resident memory **was measured**. → `efficiency/`, `inference/`, `storage_memory/`.
- **Protocol** — temperature/inference and correction-split provenance verified from code with line refs. → `configs/`.

**What can go straight into the manuscript:** the headline table (already matches), the SCOD-vs-P&C(s3b0) comparison, the joint-grid "closes the search-procedure gap" result, the sensitivity story (with the ridge/scale corrections), and the efficiency/inference/memory numbers with their honest classifications.

**Claims needing modification (see Conflicts):** (1) quote the **s3b0** P&C numbers in the SCOD comparison, not the s3b1 90.99/94.83; (2) the efficiency table's inference row is stale (7.419 s3b1 vs 7.529 s3b0) and its memory footer wrongly disclaims measured GPU memory; (3) P&C "~100 s" is a derived upper bound, not a measured construction; (4) present bootstrap as stable for bf≤0.05 with a flagged instability, not a clean double-descent.

---

## Canonical sources (recommended per manuscript table/claim)
| Manuscript element | Canonical source |
|---|---|
| CIFAR-10 headline OpenOOD table | `results/cifar10/openood_v1p5_<method>_..._seed{0,1,2}.json`; verified in `headline/HEADLINE_TABLE.md`; manuscript `experiments/cifar_tables_paper.tex` |
| P&C headline row | `..._pnc_single_block_vcal_s3b0_k20_n50_ps25.0_lr0.001_bf0.05_..._chunksize1024_seed{0,1,2}_random.json` |
| SCOD comparison | `results/scod_cifar/tables/scod_aggregate.json` + `scod/SCOD_MATCHED_COMPARISON.md` (pair vs P&C **s3b0**) |
| Block-scope | `sensitivity_cifar_agg.csv` (block factor) + `joint_selection/tables/validation_nll_by_block.md`; `block_scope/BLOCK_SCOPE.md` |
| Sensitivity curves | `sensitivity_cifar_{raw,agg}.csv`, `sweep_scalex_boot_summary.md`; `sensitivity/SENSITIVITY.md` |
| Joint selection | `pnc_full_grid/tables/` + `selected/global_config.json`; `joint_selection/STATUS.md` |
| Construction timing | `CIFAR_EFFICIENCY_ACCOUNTING.md` + `efficiency/CONSTRUCTION_TIMING_AUDIT.md` (base 8010s: log:534) |
| Inference timing | `inference_cost.json` + `inference_cost_deep_ensemble_n50_matched.json`; `inference/INFERENCE_AUDIT.md` |
| Storage/memory | `peak_memory/` + `storage_memory/STORAGE_MEMORY_AUDIT.md` |
| Temperature/splits | `openood_eval.py`,`util.py`,`cifar_tasks.py`,`pnc.py`; `configs/CIFAR_INFERENCE_PROTOCOL.md`,`CIFAR_DATA_SPLITS.md` |

---

## Conflicts (every disagreement)
1. **P&C matched numbers — s3b1 vs s3b0.** Quoted P&C (95.69/**90.99**/37.39/**94.83**/19.52) = the **s3b1** 3-seed predictive-entropy mean; the manuscript headline and canonical SCOD pairing use **s3b0** (95.59/**91.55**/33.08/**95.09**/18.15). Both are real, same checkpoints/seeds — different block. **Fix:** standardize on s3b0. (`scod/SCOD_MATCHED_COMPARISON.md`.)
2. **Inference latency block mismatch.** `efficiency_cifar_table.md` ships **7.419 (s3b1)**; the s3b0 anchor is **7.529** (`peak_memory_pnc_anchor_s3b0.json`). Quoted "7.53" = s3b0. **Fix:** re-render the row with s3b0. Conclusion (no matched-M advantage) unaffected.
3. **Bootstrap bf=0.1 instability.** `sweep_scalex_boot` shows bf0.1 catastrophic (acc 69.84±15.50); the full joint grid has s3b0/ps25/bf0.1 stable (val NLL 0.1357). Replicating conditioning artifact in one harness — **do not present bf0.1 as a robust degradation**; flag as instability. (`sensitivity/SENSITIVITY.md`.)
4. **Std convention.** Manuscript uses population std (ddof=0); recomputations here use sample std (ddof=1, ~1.22× larger). Means identical. No value discrepancy — just state the convention.
5. **Efficiency-table memory footer** claims "peak GPU memory not captured," but the separate `peak_memory/` run (2026-07-26) measured it for 11 methods. **Fix:** the footer is stale.
6. **calib ss=2048 anomaly** (all 3 seeds collapse while 256/512/1024/4096 are healthy) — recorded instability, needs a condition-number probe.

---

## Missing / not-recovered
- **The joint cross-product is NOT missing** — it exists (162 runs). **Memory profiling is NOT missing** — measured in `peak_memory/`.
- **MISSING:** P&C construction **line-item breakdown** (activation extraction / perturbation gen / member build are not separately timed — only Gram/eig/ridge on a probe harness); **LLLA** internal GGN/inverse/Cholesky split; a **standalone Deep-Ensemble training timer** (DE training is DERIVED as n× base). Multi-sketch-seed SCOD sensitivity and SLL gate-8 were deferred (documented in their reports).
- **Base training time** exists only in `experiments/logs/cifar10_finish_20260427_223643.log:534`, NOT in the checkpoint pickle.

---

## Reproduction of quoted values
| Claim | Quoted | Recomputed | Match? | Canonical source |
|---|---|---|---|---|
| P&C headline (s3b0) Near/Far AUROC | 91.55 / 95.09 | 91.55 / 95.09 | ✅ | s3b0 seed{0,1,2} JSONs |
| SCOD Acc/Near/Far AUROC | 95.74 / 89.69 / 92.56 | 95.74 / 89.69 / 92.56 | ✅ | scod_aggregate.json |
| P&C "matched" 90.99 / 94.83 | 90.99 / 94.83 | 90.99 / 94.83 (**= s3b1**) | ✅ but wrong block | s3b1 seed{0,1,2} JSONs |
| Base training time | 8010 s | 8010 s | ✅ (MEASURED) | log:534 |
| P&C construction | ≤100 s | 102.5 s residual (~65 s netted) | ⚠ DERIVED/UPPER_BOUND | EFFICIENCY_ACCOUNTING §2.2 |
| Inference P&C / DE / SWAG / Laplace | 7.53 / 7.26 / 7.33 / 7.68 | 7.529(s3b0) / 7.261 / 7.325 / 7.680 | ✅ (7.42=s3b1 in table) | inference_cost*.json |
| Block Acc / Near / Far ranges | 94.76–95.59 / 89.3–91.6 / 92.0–95.1 | identical | ✅ | sensitivity block rows |
| Joint optimum vs coordinate descent | (follow-up) | s3b0/ps25/bf0.05 == CD, rank 1/54, regret 0 | ✅ | pnc_full_grid |
| Near/Far tiers | {c100,tin} / {mnist,svhn,tex,pl365} | identical (12 methods×3 seeds) | ✅ | per_dataset keys |

**Aggregation axes (explicit):** all headline/SCOD/P&C aggregates are the mean over the **3 checkpoint×construction seeds** (nested: P&C construction seed = checkpoint seed), macro-averaged over OOD datasets **within** Near / Far. No averaging over hyperparameters. Std = across the 3 seeds.

---

## Manuscript recommendations (safe to use)
1. Headline table — use as-is (state ddof=0). ✅
2. SCOD vs P&C — use **s3b0** (91.55/95.09); note SCOD < P&C, same checkpoints/1024 pool, ID-only selection. ✅
3. Joint grid — "coordinate descent reached the global ID-val optimum (s3b0/ps25/bf0.05, rank 1/54); no material interaction missed." Distinct from the one-factor robustness sweeps. ✅
4. Efficiency — quote base 8010 s (MEASURED), P&C construction "≤100 s (derived upper bound; ~65 s netted)", inference "no matched-M advantage" (P&C 7.53 s3b0 ≈ DE 7.26). Re-render the stale s3b1 inference row and memory footer. ✅ with the fixes.
5. Memory — use the measured `peak_memory/` numbers. ✅
6. Sensitivity — scale narrow-optimum + cliff, rank flat, ridge threshold (λ=0 fails), blocks all viable; bootstrap stable ≤0.05 with a flagged instability. ✅ with the bootstrap caveat.

## Handoff (paper-repo steps I could not perform)
- Clone `EtherealEq/perturb_and_correct`, branch `agent/collect-cifar-results`, copy this `2026neurips_rebuttal_results/cifar/` tree in, commit compact artifacts (not the 5G raw trees), open the draft PR "Collect CIFAR results for P&C manuscript". Do **not** edit manuscript TeX. This branch/bundle in DrKwint/pnc is ready to transplant.
