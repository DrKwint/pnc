# CIFAR construction / training timing — recovered & classified

Classes: MEASURED / DERIVED / ESTIMATED / UPPER_BOUND / MISSING.

| Item | Value | Class | Source |
|---|---|---|---|
| Base ResNet-18 train (300 ep) | **8,010 s** | **MEASURED** | `experiments/logs/cifar10_finish_20260427_223643.log:534` (`base_training_time_evidence.txt`). NOT in the checkpoint pickle (pickle `metrics` has only eval_time 6.59s). |
| P&C total build+eval, seed0/1/2 | 826/847/844 s | MEASURED | `experiments/logs/p2_s3b0_ood_20260428_141145.log` |
| P&C OpenOOD eval (subtracted) | 723.5 s | MEASURED | submitted s3b0 seed0 JSON eval_time |
| **P&C construction "~100 s"** | 102.5 s residual (~65 s netting the untimed ID-val pass) | **DERIVED / UPPER_BOUND** | `CIFAR_EFFICIENCY_ACCOUNTING.md` §2.2 (826 − 723.5). Not a directly timed construction. |
| — ridge solve (per solve) | ~1.0 s | MEASURED (probe) | `results/conv_construction_rank/*/timings.json` |
| — Gram accumulation | 0.089 s (1024) / 0.46 s (10k) | MEASURED (probe) | same |
| — eigendecomposition (2305²) | 0.835 s | MEASURED (probe) | same |
| — activation extract / perturb gen / member build | — | **MISSING** | not separately timed for the anchor |
| SWAG post-train construction | 55.80 ± 5.06 s (n=3) | MEASURED | `swag_construction_time.json` |
| SWAG full training | ~11,177 s Δ (≈9,570 s corrected) | DERIVED/UPPER_BOUND (ckpt mtimes) | `CIFAR_EFFICIENCY_ACCOUNTING.md` §2.6 |
| Laplace (LLLA) construction | 15.56 ± 0.30 s (n=3) | MEASURED | `results/cifar10/baseline_llla_*_seed*.json` `train_time` |
| — LLLA GGN vs inverse vs Cholesky split | — | **MISSING** | timer spans all three |
| SCOD build (seed0) | 349.1 s (sketch 167.3 s) | MEASURED | `../scod/timing/profile.json` |
| Epinet training | ~1,220 s Δ / ~20 min | DERIVED/ESTIMATED (mtime) | §2.6 |
| Deep Ensemble n=5 / n=50 training | 40,050 s / 400,500 s | DERIVED (5×/50× base) | §2.4 (no standalone DE-training timer) |

**Environment:** NVIDIA RTX 5060 (8,151 MiB), WSL2, JAX 0.9.1 + Flax NNX 0.12.3. Timing runs: 1 repetition, cold/JIT batch excluded.

**Key caveat:** "≤100 s P&C construction" is a DERIVED upper bound (residual), never a directly-timed construction (~65 s once the untimed ID-val pass is netted out). No per-operation breakdown of the anchor construction exists (only probe-harness Gram/eig/ridge component times).
