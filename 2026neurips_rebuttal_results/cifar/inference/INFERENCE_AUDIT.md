# Matched-M CIFAR inference latency — recovered & verified

Protocol: batch=256, N=5000 warm samples, seed 0, **1 repetition (mean only, no std/median)**, cold/JIT batch excluded, `block_until_ready()`-synced, models preloaded, member reconstruction included, BN refresh excluded, data transfer included.

| Method | ms/sample | M / fwd passes | Class | Source |
|---|---|---|---|---|
| PreAct ResNet-18 (single) | 0.617 | 1 | MEASURED | `submitted_inference_cost.json` |
| ReAct+Energy | 0.614 | 1 | MEASURED | same |
| Mahalanobis | 0.735 | 1 | MEASURED | same |
| Deep Ensemble n=5 | 1.338 | 5 | MEASURED | same |
| Epinet n=50 | 1.617 | 50 | MEASURED | same |
| MC Dropout n=32 | 5.177 | 32 | MEASURED | same |
| **Deep Ensemble n=50 (matched-M)** | **7.261** | 50 | MEASURED | `inference_cost_deep_ensemble_n50_matched.json` |
| SWAG n=50 | 7.325 | 50 | MEASURED | `submitted_inference_cost.json` |
| **P&C single s3b1 (as-shipped table)** | **7.419** | 50 | MEASURED | `submitted_inference_cost.json`, `../efficiency/efficiency_cifar_table.{md,csv}` |
| **P&C single s3b0 (SUBMITTED anchor)** | **7.529** | 50 | MEASURED | `../storage_memory/peak_memory/peak_memory_pnc_anchor_s3b0.json` (`warm_per_sample_ms=7.5288`) |
| LLLA n=50 | 7.680 | 50 | MEASURED | `submitted_inference_cost.json` |
| P&C multi-block | 9.688 | 50 | MEASURED | same |

## Reconciliation & verdict
- **Quoted 7.53 = P&C s3b0 anchor; 7.42 = P&C s3b1 as-shipped.** The efficiency table (`efficiency_cifar_table.md`) still shows the stale **7.419 (s3b1)** row and was never re-rendered with the s3b0 anchor. ⚠ Use **7.53 (s3b0)** to match the manuscript P&C block.
- Quoted comparators: P&C 7.53, Deep Ensemble 7.26, SWAG 7.33, Laplace 7.68 — all MEASURED and reproduce.
- **Claim "P&C has NO matched-M inference advantage over Deep Ensembles": SUPPORTED.** P&C s3b0 7.529 vs DE n=50 7.261 = +3.7% (s3b1 = +2.2%), within run-to-run machine variance (DE n=5 drifted 14.8% between runs per `CIFAR_EFFICIENCY_ACCOUNTING.md` §4.1). Both do 50 forward passes; P&C buys nothing at inference over an equal-member ensemble.
- **Caveat:** single-run means only (no std/median); one repetition.
