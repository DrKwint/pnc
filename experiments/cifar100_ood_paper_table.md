# CIFAR-100 OOD Detection — Paper Table

All methods use frozen ID hyperparameters; no OOD data was used for tuning, model selection, or threshold choice. Mean ± std reported across N seeds.

*Note: the Mahalanobis baseline reported here uses a paper-clean variant — single penultimate layer, per-class means + shared covariance fitted on ID train, no OOD-trained logistic combiner. The original Lee et al. (2018) numbers use OOD validation data and are not directly comparable.*

| Method | Train cost | N | Acc% ↑ | NLL ↓ | ECE ↓ | Near AUROC ↑ | Near FPR95 ↓ | Far AUROC ↑ | Far FPR95 ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Post-hoc OOD baselines (frozen single model)** |  |  |  |  |  |  |  |  |  |
| PreAct ResNet-18 | 1× | 3 | 78.31 ± 0.13 | 0.8647 ± 0.0065 | 0.0423 ± 0.0016 | 80.61 ± 0.20 | 57.89 ± 0.43 | 78.64 ± 0.68 | 57.11 ± 1.19 |
| MSP | 1× | 3 | 78.30 ± 0.12 | 0.8647 ± 0.0065 | 0.0423 ± 0.0016 | 79.99 ± 0.18 | 57.78 ± 0.52 | 77.69 ± 0.73 | 57.37 ± 1.15 |
| Energy | 1× | 3 | 78.31 ± 0.13 | 0.8647 ± 0.0065 | 0.0423 ± 0.0017 | 80.44 ± 0.24 | 58.27 ± 0.34 | 78.91 ± 0.71 | 56.32 ± 1.60 |
| Mahalanobis | 1× | 3 | 78.30 ± 0.12 | 0.8647 ± 0.0065 | 0.0423 ± 0.0016 | 73.60 ± 0.58 | 72.85 ± 0.37 | 82.41 ± 0.62 | 52.18 ± 0.78 |
| ReAct+Energy | 1× | 3 | 75.56 ± 0.14 | 0.9493 ± 0.0133 | 0.0113 ± 0.0008 | 78.30 ± 0.49 | 61.73 ± 0.48 | 73.86 ± 0.85 | 68.67 ± 3.29 |
| **Post-hoc UQ methods (single model)** |  |  |  |  |  |  |  |  |  |
| LLLA | 1× | 0 | --- | --- | --- | --- | --- | --- | --- |
| Epinet | 1× | 3 | 78.31 ± 0.24 | 0.8795 ± 0.0045 | 0.0400 ± 0.0041 | 80.49 ± 0.26 | 58.26 ± 0.57 | 79.23 ± 0.85 | 55.41 ± 1.30 |
| PnC scale=25.0 | 1× | 0 | --- | --- | --- | --- | --- | --- | --- |
| PnC scale=50.0 | 1× | 3 | 75.47 ± 0.31 | 0.8975 ± 0.0023 | 0.0280 ± 0.0007 | 79.51 ± 0.30 | 61.34 ± 0.93 | 75.42 ± 0.71 | 63.17 ± 3.03 |
| Multi-block PnC scale=7.0 | 1× | 0 | --- | --- | --- | --- | --- | --- | --- |
| **Train-time UQ methods (single model)** |  |  |  |  |  |  |  |  |  |
| MC Dropout | 1× | 3 | 78.36 ± 0.22 | 0.8675 ± 0.0046 | 0.0421 ± 0.0027 | 80.64 ± 0.31 | 58.48 ± 1.01 | 78.63 ± 1.58 | 56.14 ± 3.39 |
| SWAG | 1× | 3 | 77.23 ± 0.28 | 0.8736 ± 0.0138 | 0.0311 ± 0.0035 | 80.32 ± 0.21 | 58.88 ± 0.21 | 81.28 ± 1.12 | 53.33 ± 2.36 |
| **Reference: 5× training cost** |  |  |  |  |  |  |  |  |  |
| Standard Ensemble | 5× | 3 | 81.25 ± 0.13 | 0.7056 ± 0.0030 | 0.0211 ± 0.0012 | 82.62 ± 0.03 | 54.98 ± 0.11 | 80.61 ± 0.20 | 53.67 ± 0.32 |

*Train cost is the multiplier of base-model training compute.*
