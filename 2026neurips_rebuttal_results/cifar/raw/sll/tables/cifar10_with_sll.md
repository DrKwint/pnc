# CIFAR-10 OpenOOD v1.5 -- SLL-Backbone vs post-hoc / ensemble methods

SLL-Backbone: full-covariance linearized Laplace over S=2048 backbone weights selected by predictive-variance contribution (ID-only). Probit predictive entropy. Mean ± sample-std, 3 seeds.

| Method | Acc | Near AUROC | Near FPR95 | Far AUROC | Far FPR95 |
|---|---:|---:|---:|---:|---:|
| LLLA (n=50) | 95.77 | 88.97 | 54.10 | 93.04 | 28.40 |
| SCOD-1024 | 95.74 | 89.69 | 39.40 | 92.56 | 21.41 |
| **SLL-Backbone (S=2048)** | 95.46±0.09 | 79.05±2.80 | 80.70±5.21 | 86.44±3.04 | 56.08±8.79 |
| P&C s3b0 (M=50) | 95.59 | 91.55 | 33.08 | 95.09 | 18.15 |
| Standard Ensemble (n=5) | 96.56 | 91.10 | 40.40 | 94.63 | 19.50 |
