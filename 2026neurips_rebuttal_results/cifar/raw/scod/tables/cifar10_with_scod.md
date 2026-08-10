# CIFAR-10 OpenOOD v1.5 -- SCOD-1024 vs submitted methods

Primary SCOD variant: **SCOD-1024**, k=10, Meps=5000, T=6k_max+4=124, tempered posterior_pred score, base-classifier temperature on ID-val. Mean ± sample-std over 3 checkpoint seeds. SCOD Acc/NLL are the unchanged base classifier's. Comparator rows are the reproduced submitted 3-seed table.

| Method | Acc % ↑ | Near AUROC ↑ | Near FPR95 ↓ | Far AUROC ↑ | Far FPR95 ↓ |
|---|---:|---:|---:|---:|---:|
| PreActResNet-18 / MSP | 95.74 | 87.70 | 66.30 | 91.50 | 38.30 |
| Mahalanobis | 95.74 | 87.98 | 66.30 | 93.25 | 38.30 |
| MC Dropout (n=32) | 95.76 | 87.25 | 71.00 | 91.34 | 42.50 |
| LLLA (n=50) | 95.77 | 88.97 | 54.10 | 93.04 | 28.40 |
| SWAG (n=50) | 95.37 | 90.03 | 44.70 | 94.19 | 22.10 |
| **SCOD-1024 (k=10)** | 95.74±0.18 | 89.69±0.11 | 39.40±1.17 | 92.56±0.45 | 21.41±1.06 |
| P&C s3b0 (M=50) | 95.59 | 91.55 | 33.08 | 95.09 | 18.15 |
| Deep Ensemble (n=5) | 96.56 | 91.10 | 40.40 | 94.63 | 19.50 |

SCOD-1024 ID: Acc 95.74±0.18, NLL 0.144±0.005. Score orientation: higher = more OOD.
