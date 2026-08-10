# CIFAR-10 One-Factor Sweeps: Perturbation Scale & Bootstrap Fraction

**Anchor (original-paper config, held fixed for all non-varying factors):** single-block P&C on
`s3b0` (stage_idx=3, block_idx=0), K=20 directions, M=50 members, ps=25, lambda=1e-3, subset_size=1024,
random directions, bootstrap_frac=0.05, temperature fit on the ID-val split. Score = predictive-entropy;
full OpenOOD v1.5 sets (Near = cifar100, tiny_imagenet; Far = mnist, svhn, textures, places365).
3 seeds (0,1,2), mean±std. The center cell (ps=25, bf=0.05) is the shared anchor, reused in both grids.

### Perturbation-scale multiplier (bf fixed = 0.05)

| setting | Acc | Near-AUROC | Near-FPR95 | Far-AUROC | Far-FPR95 | seeds |
|---|---|---|---|---|---|---|
| 0.50x (ps=12.5) | 95.72±0.15 | 89.82±0.03 | 46.46±0.45 | 93.58±0.59 | 24.78±1.55 | 3/3 |
| 0.75x (ps=18.75) | 95.64±0.17 | 90.89±0.09 | 36.71±0.69 | 94.43±0.49 | 20.25±1.00 | 3/3 |
| 1.00x (ps=25) | 95.59±0.20 | 91.55±0.13 | 33.06±0.66 | 95.10±0.50 | 18.12±1.05 | 3/3 |
| 1.25x (ps=31.25) | 94.99±0.23 | 90.86±0.23 | 37.82±1.70 | 94.57±0.75 | 18.91±1.71 | 3/3 |
| 1.50x (ps=37.5) | 9.99±0.02 | 64.08±1.15 | 83.57±2.84 | 55.33±4.76 | 88.37±2.67 | 3/3 |

### Bootstrap fraction (ps fixed = 25)

| setting | Acc | Near-AUROC | Near-FPR95 | Far-AUROC | Far-FPR95 | seeds |
|---|---|---|---|---|---|---|
| bf=0.01 | 95.58±0.20 | 91.54±0.14 | 33.17±0.40 | 95.28±0.39 | 17.71±0.92 | 3/3 |
| bf=0.03 | 95.63±0.18 | 91.55±0.13 | 33.32±0.32 | 95.22±0.39 | 18.00±0.94 | 3/3 |
| bf=0.05 (anchor) | 95.59±0.20 | 91.55±0.13 | 33.06±0.66 | 95.10±0.50 | 18.12±1.05 | 3/3 |
| bf=0.07 | 95.37±0.11 | 90.73±0.26 | 39.75±1.42 | 94.38±0.41 | 19.74±0.98 | 3/3 |
| bf=0.1 | 69.84±15.50 | 69.25±8.93 | 86.70±12.66 | 70.27±8.71 | 81.28±16.37 | 3/3 |

*Metrics are macro-mean over the OOD datasets, matching the submitted protocol. AUROC/FPR95 in %.*
