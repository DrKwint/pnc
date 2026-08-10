# Phase 1 — CIFAR-10 Reproduction (seed 0)

**Date:** 2026-07-23 · **Branch:** `neurips-2026-rebuttal` · **Commit:** `854f4d8` (tree dirty)
**Hardware:** NVIDIA RTX 5060 8 GB, WSL2 · JAX 0.9.1 / Flax 0.12.3 / Python 3.12.3

Both candidate "submitted" configs (see `SUBMISSION_MANIFEST.md` §0) were recomputed **from scratch**
(no Luigi cache) at seed 0 via `_repro_driver.py`, writing to fresh paths so no existing checkpoint or
result file was overwritten. Each was diffed against its own cached result JSON.

Base checkpoint (both): `results/cifar10/preact_resnet18_train_e300_optsgd_lr1e-01_wd5e-04_bs128_wu5_mom0p9_n1_augfcco8_ls0_seed0.pkl`

## Reproduction gate (per task spec)
- same perturbation scale selected ✓ (fixed-scale eval; A=25, B=50)
- ID accuracy Δ ≤ 0.1 pp
- AUROC Δ ≤ 0.002 (abs, i.e. ≤0.0002 on the 0–1 scale / ≤0.02 on the ×100 scale)
- FPR95 Δ ≤ 1 pp

## Candidate A — `s3b0 / ps25 / bf0.05` (matches `cifar_tables_paper.tex`)
Runtime 4898 s (~82 min). Cached ref: `openood_v1p5_pnc_single_block_vcal_s3b0_k20_n50_ps25.0_lr0.001_bf0.05_..._seed0_random.json`

| metric | reproduced | cached | abs diff | gate |
|---|---|---|---|---|
| Acc % | 95.8500 | 95.8500 | 0.0000 | ✓ |
| NLL | 0.1342 | 0.1342 | 0.0000 | ✓ |
| ECE | 0.0049 | 0.0049 | 0.0000 | ✓ |
| posthoc T | 0.7608 | 0.7607 | +0.0001 | ✓ |
| Near AUROC×100 | 91.4134 | 91.4133 | +0.0001 | ✓ |
| Far AUROC×100 | 95.6913 | 95.6911 | +0.0002 | ✓ |
| Near FPR95 | 33.99 | 33.975 | +0.015 | ✓ |
| Far FPR95 | 17.1625 | 17.1625 | 0.0000 | ✓ |

**PASS** — all diffs at floating-point/nondeterminism level.

## Candidate B — `s3b1 / ps50 / no bootstrap` (matches appendix `cifar10_ood_bootstrap_table.tex` + narrative docs)
Runtime 892 s (~15 min). Cached ref: `openood_v1p5_pnc_single_block_vcal_s3b1_k20_n50_ps50.0_lr0.001_e300_..._seed0_random.json`

| metric | reproduced | cached | abs diff | gate |
|---|---|---|---|---|
| Acc % | 95.6500 | 95.6500 | 0.0000 | ✓ |
| NLL | 0.1384 | 0.1384 | 0.0000 | ✓ |
| ECE | 0.0079 | 0.0078 | +0.0001 | ✓ |
| posthoc T | 0.9971 | 0.9972 | −0.0001 | ✓ |
| Near AUROC×100 | 91.6950 | 91.6951 | −0.0000 | ✓ |
| Far AUROC×100 | 95.1915 | 95.1915 | −0.0000 | ✓ |
| Near FPR95 | 30.75 | 30.75 | 0.0000 | ✓ |
| Far FPR95 | 17.97 | 17.97 | 0.0000 | ✓ |

**PASS** — all diffs at floating-point/nondeterminism level.

## Interpretation
Both cached result files are faithfully reproducible; the eval pipeline is effectively deterministic on
this GPU. Reproduction alone does **not** disambiguate which config was in the submitted PDF — each
reproduces *its own* cache. The disambiguating evidence is table-level (3-seed aggregates):

| 3-seed aggregate | Cand A (s3b0/ps25/bf05) | Cand B (s3b1/ps50) |
|---|---|---|
| Acc % | 95.59 | 95.59 |
| NLL | **0.138** | 0.144 |
| ECE | **0.0050** | 0.0079 |
| Near AUROC | 91.55 | 91.51 |
| Near FPR95 | 33.08 | 32.17 |
| Far AUROC | **95.09** | 94.10 |
| Far FPR95 | **18.15** | 20.54 |
| Reproduces | `cifar_tables_paper.tex` (final paper table, Apr 28) | `cifar10_ood_bootstrap_table.tex` (appendix, Apr 20) + narrative docs |

**Candidate A** exactly matches the final paper-named table; **Candidate B** matches the appendix table and
the (stale) narrative docs. Cleanest discriminators in a submitted-PDF Table 2 PnC row: **Far-AUROC
(95.09 vs 94.10), NLL (0.138 vs 0.144), ECE (0.0050 vs 0.0079)**. Final anchor selection pending author
confirmation of the submitted PDF's numbers (see `SUBMISSION_MANIFEST.md` §0 outstanding-confirmation note).

Artifacts: `reproduction_cifar_seed0_A.json`, `reproduction_cifar_seed0_B.json`,
`repro_A_s3b0_ps25_bf0.05_seed0.json`, `repro_B_s3b1_ps50_nobf_seed0.json`.
