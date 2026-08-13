| Method | Extra opt? | Cal N | ID Acc | ID NLL | ID ECE | Near AUROC | Near FPR95 | Far AUROC | Far FPR95 | Fit | Storage | Evals/img |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| MSP | no | — | 81.068 | 0.8482 | 0.0913 | 73.52 | 81.84 | 86.04 | 51.74 | 0.0s | 0.0 MiB | 1 |
| Energy | no | — | 81.068 | 0.8482 | 0.0913 | 62.39 | 93.16 | 78.96 | 85.29 | 0.0s | 0.0 MiB | 1 |
| ReAct + Energy | no | 8192 | 81.068 | 0.8482 | 0.0913 | 69.21 | 84.23 | 85.61 | 53.90 | 0.1s | 0.0 MiB | 1 |
| Mahalanobis | no | 32768 | 81.068 | 0.8482 | 0.0913 | 78.82 | 66.36 | 92.55 | 30.23 | 0.3s | 5.2 MiB | 1 |
| Laplace (KFAC) | no | 32768 | 81.014 | 0.8446 | 0.0901 | 74.63 | 75.03 | 86.56 | 48.79 | 1.1s | 6.1 MiB | 20 |
| Uncorrected perturb. | no | 32768 | 80.619 | 0.9048 | 0.0466 | 72.53 ± 0.30 | 71.66 | 84.40 ± 0.27 | 46.51 | 12.0s | 0.0 MiB | 20 |
| P&C (r=2, primary) | no | 32768 | 80.830 | 0.8045 | 0.0542 | 76.49 ± 0.09 | 67.65 | 87.89 ± 0.12 | 44.97 | 12.0s | 167.0 MiB | 20 |

| SCOD | — | — | \multicolumn{9}{l}{**SCOD_NOT_TRACTABLE_AT_VIT_SCALE** — see `metrics/scod_preflight.json`} |
| LLLA (dense) | — | — | \multicolumn{9}{l}{**MEMORY_INFEASIBLE** — dense covariance is 769,000² = 2.37 TB} |
