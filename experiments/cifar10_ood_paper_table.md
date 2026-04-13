# CIFAR-10 OOD Detection — Paper Table

All methods use frozen ID hyperparameters; no OOD data was used for tuning, model selection, or threshold choice. Mean ± std reported across N seeds.

*Note: the Mahalanobis baseline reported here uses a paper-clean variant — single penultimate layer, per-class means + shared covariance fitted on ID train, no OOD-trained logistic combiner. The original Lee et al. (2018) numbers use OOD validation data and are not directly comparable.*

| Method | Train cost | N | Acc% ↑ | NLL ↓ | ECE ↓ | Near AUROC ↑ | Near FPR95 ↓ | Far AUROC ↑ | Far FPR95 ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Post-hoc OOD baselines (frozen single model)** |  |  |  |  |  |  |  |  |  |
| PreAct ResNet-18 | 1× | 3 | 95.74 ± 0.18 | 0.1439 ± 0.0053 | 0.0102 ± 0.0013 | 87.85 ± 0.08 | 66.30 ± 1.07 | 91.86 ± 1.20 | 38.30 ± 6.66 |
| MSP | 1× | 3 | 95.74 ± 0.18 | 0.1439 ± 0.0053 | 0.0102 ± 0.0013 | 87.66 ± 0.06 | 65.32 ± 1.13 | 91.53 ± 1.11 | 37.47 ± 6.21 |
| Energy | 1× | 3 | 95.74 ± 0.18 | 0.1439 ± 0.0053 | 0.0102 ± 0.0013 | 87.00 ± 0.12 | 73.20 ± 1.17 | 91.38 ± 1.51 | 44.99 ± 8.78 |
| Mahalanobis | 1× | 3 | 95.74 ± 0.18 | 0.1439 ± 0.0053 | 0.0102 ± 0.0013 | 87.98 ± 0.13 | 42.63 ± 1.11 | 93.25 ± 0.32 | 21.00 ± 0.42 |
| ReAct+Energy | 1× | 3 | 95.77 ± 0.20 | 0.1480 ± 0.0051 | 0.0116 ± 0.0012 | 88.63 ± 0.18 | 59.49 ± 1.69 | 92.50 ± 1.47 | 33.55 ± 6.58 |
| **Post-hoc UQ methods (single model)** |  |  |  |  |  |  |  |  |  |
| LLLA | 1× | 3 | 95.77 ± 0.25 | 0.1403 ± 0.0063 | 0.0083 ± 0.0016 | 88.97 ± 0.84 | 54.12 ± 6.98 | 93.04 ± 1.17 | 28.44 ± 8.23 |
| Epinet | 1× | 3 | 95.76 ± 0.22 | 0.1481 ± 0.0056 | 0.0103 ± 0.0027 | 88.07 ± 0.05 | 63.37 ± 1.72 | 92.16 ± 1.25 | 35.10 ± 6.82 |
| PnC scale=25.0 | 1× | 3 | 95.70 ± 0.19 | 0.1408 ± 0.0052 | 0.0077 ± 0.0012 | 91.11 ± 0.13 | 34.99 ± 0.64 | 94.04 ± 0.86 | 21.10 ± 1.78 |
| PnC scale=50.0 | 1× | 3 | 95.59 ± 0.12 | 0.1440 ± 0.0059 | 0.0079 ± 0.0011 | 91.51 ± 0.21 | 32.17 ± 1.23 | 94.10 ± 1.00 | 20.54 ± 2.27 |
| Multi-block PnC scale=7.0 | 1× | 3 | 95.09 ± 0.18 | 0.1522 ± 0.0042 | 0.0077 ± 0.0017 | 89.96 ± 0.09 | 38.82 ± 0.99 | 92.86 ± 1.14 | 24.41 ± 1.70 |
| **Train-time UQ methods (single model)** |  |  |  |  |  |  |  |  |  |
| MC Dropout | 1× | 3 | 95.76 ± 0.14 | 0.1475 ± 0.0059 | 0.0099 ± 0.0037 | 87.25 ± 0.18 | 71.04 ± 0.85 | 91.34 ± 0.27 | 42.53 ± 1.42 |
| SWAG | 1× | 3 | 95.37 ± 0.07 | 0.1462 ± 0.0035 | 0.0075 ± 0.0012 | 90.03 ± 0.19 | 44.71 ± 0.97 | 94.19 ± 1.26 | 22.09 ± 6.01 |
| **Reference: 5× training cost** |  |  |  |  |  |  |  |  |  |
| Standard Ensemble | 5× | 3 | 96.56 ± 0.06 | 0.1086 ± 0.0003 | 0.0046 ± 0.0004 | 91.10 ± 0.04 | 40.39 ± 0.47 | 94.63 ± 0.17 | 19.49 ± 0.32 |

*Train cost is the multiplier of base-model training compute.*
