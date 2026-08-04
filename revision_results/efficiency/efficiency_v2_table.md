# Efficiency accounting (measured, matched protocol)

All methods share one base-training budget and one machine. P&C's construction time includes the least-squares correction solve. Latency is median over timed reps after a same-shape warm-up.

## Ant-v5  (steps=5000, 1 seed(s), NVIDIA TITAN X (Pascal))

| method | nets trained | construction (s) | of which build (s) | storage (MB) | batch10000 lat (ms) | batch1000 lat (ms) | batch1 lat (ms) |
|---|---|---|---|---|---|---|---|
| Deep Ensemble | 50 | 1131.9 | 0.00 | 37.1 | 150.94 | 164.41 | 153.32 |
| P&C | 1 | 18.4  (62× cheaper) | 1.75 | 29.3 | 200.64 | 195.67 | 196.45 |
| SWAG | 1 | 41.6  (27× cheaper) | 0.00 | 2.2 | 665.35 | 649.94 | 621.64 |
| Laplace | 1 | 14.8  (77× cheaper) | 0.85 | 3.9 | 1755.92 | 1717.79 | 1743.19 |
| MC Dropout | 1 | 22.4  (50× cheaper) | 0.00 | 0.7 | 710.51 | 708.41 | 688.66 |
| Single net (reference) | 1 | 21.3  (53× cheaper) | 0.00 | 0.7 | 3.89 | 3.56 | 3.53 |

P&C storage: 29.3 MB resident for inference (shared base + per-member corrected blocks + per-member dW), vs 37.1 MB for the Deep Ensemble. Minimal checkpoint is 0.76 MB (base + latent coefficients; directions regenerate from the seed), but that requires re-solving the corrections at load time -- not what the code currently does.

## HalfCheetah-v5  (steps=5000, 1 seed(s), NVIDIA TITAN X (Pascal))

| method | nets trained | construction (s) | of which build (s) | storage (MB) | batch10000 lat (ms) | batch1000 lat (ms) | batch1 lat (ms) |
|---|---|---|---|---|---|---|---|
| Deep Ensemble | 50 | 1321.9 | 0.00 | 26.4 | 158.63 | 161.80 | 146.13 |
| P&C | 1 | 27.0  (49× cheaper) | 1.52 | 25.5 | 196.15 | 188.03 | 189.56 |
| SWAG | 1 | 61.9  (21× cheaper) | 0.00 | 1.6 | 579.96 | 559.01 | 567.91 |
| Laplace | 1 | 26.6  (50× cheaper) | 0.82 | 3.4 | 1689.26 | 1666.80 | 1651.74 |
| MC Dropout | 1 | 29.9  (44× cheaper) | 0.00 | 0.5 | 722.91 | 696.88 | 680.81 |
| Single net (reference) | 1 | 25.9  (51× cheaper) | 0.00 | 0.5 | 3.77 | 3.61 | 3.54 |

P&C storage: 25.5 MB resident for inference (shared base + per-member corrected blocks + per-member dW), vs 26.4 MB for the Deep Ensemble. Minimal checkpoint is 0.55 MB (base + latent coefficients; directions regenerate from the seed), but that requires re-solving the corrections at load time -- not what the code currently does.

## Hopper-v5  (steps=5000, 1 seed(s), NVIDIA TITAN X (Pascal))

| method | nets trained | construction (s) | of which build (s) | storage (MB) | batch10000 lat (ms) | batch1000 lat (ms) | batch1 lat (ms) |
|---|---|---|---|---|---|---|---|
| Deep Ensemble | 50 | 1279.3 | 0.00 | 25.6 | 157.63 | 146.13 | 139.15 |
| P&C | 1 | 27.3  (47× cheaper) | 1.73 | 25.2 | 197.22 | 190.57 | 194.97 |
| SWAG | 1 | 60.3  (21× cheaper) | 0.00 | 1.5 | 566.75 | 561.66 | 562.14 |
| Laplace | 1 | 26.1  (49× cheaper) | 0.84 | 3.4 | 1987.28 | 1655.80 | 1666.84 |
| MC Dropout | 1 | 29.3  (44× cheaper) | 0.00 | 0.5 | 694.94 | 636.69 | 680.62 |
| Single net (reference) | 1 | 25.2  (51× cheaper) | 0.00 | 0.5 | 3.70 | 3.59 | 3.50 |

P&C storage: 25.2 MB resident for inference (shared base + per-member corrected blocks + per-member dW), vs 25.6 MB for the Deep Ensemble. Minimal checkpoint is 0.53 MB (base + latent coefficients; directions regenerate from the seed), but that requires re-solving the corrections at load time -- not what the code currently does.
