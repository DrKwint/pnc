# Phase 7 — CIFAR-10 Inference Efficiency (verified)

Methodology (from `scripts/benchmark_inference_cost.py`, seed 0): warm per-sample latency over N=5000 test images, batch_size=256, GPU-synced with `block_until_ready`, first (cold/JIT) batch excluded. Member counts VERIFIED against the submitted methods: P&C M=50, MC Dropout n=32, Deep Ensemble n=5, SWAG/LLLA/Epinet n=50.

| Method | ms/sample | throughput (samp/s) | fwd passes | train cost |
|---|---|---|---|---|
| PreAct ResNet-18 / MSP / Energy | 0.617 | 1622 | 1 | 1× |
| ReAct+Energy | 0.614 | 1629 | 1 | 1× |
| Mahalanobis | 0.735 | 1361 | 1 | 1× |
| Epinet n=50 | 1.617 | 618 | 50 | 1.05× |
| MC Dropout n=32 | 5.177 | 193 | 32 | 1× |
| SWAG n=50 | 7.325 | 136 | 50 | 1× |
| LLLA n=50 | 7.680 | 130 | 50 | 1× |
| PnC single-block scale=25 | 7.419 | 135 | 50 | 1× |
| PnC multi-block scale=7 | 9.688 | 103 | 50 | 1× |
| Deep Ensemble n=5 | 1.338 | 747 | 5 | 5× |
| Deep Ensemble n=50 (matched-M) | 7.261 | 138 | 50 | 50× |

## Two Deep-Ensemble comparisons (kept distinct, per task)

1. **Submitted-cost:** Deep Ensemble **n=5** — 1.338 ms/sample, 5 fwd, **5× training**.

2. **Matched-M:** Deep Ensemble **n=50** — 7.261 ms/sample, 50 fwd, **50× training**.

## Honest reading (P&C is NOT cheaper at inference)

- At **matched member count**, P&C single (7.42 ms, 50 fwd) and a 50-member Deep Ensemble (7.26 ms, 50 fwd) have **essentially identical inference latency** — as expected, since both do 50 forward passes. P&C buys nothing at inference over an equal-member ensemble.

- Versus the **submitted** Deep Ensemble (n=5, 1.34 ms), P&C is **~5.5× SLOWER** at inference (50 vs 5 forward passes), while being **5× cheaper to train** (1× vs 5×). Against the matched-M ensemble the training gap is 50× (1× vs 50×).

- P&C's genuine efficiency advantages are **training cost (1×)** and **model storage** (one base network plus 50 small conv2 corrections, vs 50 full network copies for the matched-M ensemble) — NOT inference latency. Single forward pass reference: 0.617 ms.

Source rows: `efficiency_cifar_table.csv`; matched-M measurement: `inference_cost_deep_ensemble_n50_matched.json`; cached submitted timings: `results/cifar10/inference_cost/*.json`. Peak GPU memory was not captured in the cached run (methodology times latency only); the storage argument above is structural, not a measured peak.