| Method / score | ID error AUROC | Near AUROC | Far AUROC | Near FPR95 | Far FPR95 |
|---|---|---|---|---|---|
| MSP | 85.61 | 73.52 | 86.04 | 81.84 | 51.74 |
| Base entropy (raw) | 82.40 | 72.95 | 87.42 | 85.93 | 55.66 |
| Base entropy (T=0.7) | 86.81 | 74.53 | 86.42 | 75.74 | 49.23 |
| Mahalanobis | 78.00 | 78.82 | 92.55 | 66.36 | 30.23 |
| Mahalanobis (unconditional) | 71.22 | 75.54 | 91.13 | 72.68 | 36.39 |
| P&C expected member entropy | 85.78 | 76.55 | 87.85 | 67.60 | 45.74 |
| P&C predictive entropy | 85.58 | 76.43 | 87.74 | 67.63 | 45.73 |
| P&C mutual information | 80.92 | 73.97 | 84.30 | 68.30 | 42.51 |
| P&C K=5 | 85.49 | 76.53 | 87.82 | 68.34 | 45.58 |
| P&C K=20 | 85.37 | 76.49 | 87.93 | 68.06 | 44.99 |
| P&C K=40 | 85.35 | 76.51 | 87.93 | 67.95 | 44.77 |
| P&C K=80 | 85.45 | 76.41 | 87.93 | 68.20 | 44.72 |
| LLLA-Kron | 82.40 | 72.95 | 87.42 | 85.93 | 55.66 |
| LLLA-Kron+Temp | 86.78 | 74.54 | 86.49 | 76.08 | 49.17 |
| SCOD-linear | 85.61 | 75.15 | 88.48 | 73.18 | 43.37 |
| SCOD-ffn | 82.57 | 72.02 | 84.63 | 75.41 | 46.38 |
| SCOD-last-block | NA | NA | NA | NA | NA |

All values are percentages. `NA` marks a method that was not run in this round; see the report for why. ID error AUROC treats a wrong base-model top-1 on the 50k validation set as the positive class.

| Method | configuration |
|---|---|
| P&C K=5 | r=2.0, lam=1000, realized r=2.000, dAcc=-0.235 pp |
| P&C K=20 | r=2.0, lam=1000, realized r=2.000, dAcc=-0.255 pp |
| P&C K=40 | r=2.0, lam=1000, realized r=2.000, dAcc=-0.261 pp |
| P&C K=80 | r=2.0, lam=1000, realized r=2.000, dAcc=-0.260 pp |
| LLLA-Kron | prior=1e+09, T=1.0000 |
| LLLA-Kron+Temp | prior=1e+09, T=0.7063 |
| SCOD-linear | k=30, T=184, q=None, N=32768 |
| SCOD-ffn | k=30, T=184, q=None, N=32768 |
