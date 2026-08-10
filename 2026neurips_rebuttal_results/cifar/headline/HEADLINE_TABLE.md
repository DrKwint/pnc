# CIFAR-10 OpenOOD headline table — recovered & verified

**Verified:** the full manuscript CIFAR-10 table (`experiments/cifar_tables_paper.tex` Table `tab:ood_cifar10`, lines 31–43) reproduces **exactly** (means to the printed digit) by recomputing from the per-seed OpenOOD JSONs in `results/cifar10/`. Every method has **3 seeds {0,1,2}**. The only manuscript-vs-recompute difference is the **std convention** (manuscript = population std ddof=0; table below = sample std ddof=1, ~1.22× larger — no value discrepancy).

## Aggregate (3-seed mean ± sample-std, ddof=1)
| Method | Acc% | NLL | ECE | Near AUROC | Near FPR95↓ | Far AUROC | Far FPR95↓ |
|---|---|---|---|---|---|---|---|
| PreAct ResNet-18 | 95.74±0.18 | 0.144±0.005 | 0.0102 | 87.85±0.08 | 66.30 | 91.86±1.20 | 38.30 |
| MSP | 95.74±0.18 | 0.144 | 0.0102 | 87.66±0.06 | 66.30 | 91.53±1.11 | 38.30 |
| Energy | 95.74±0.18 | 0.144 | 0.0102 | 87.00±0.12 | 66.31 | 91.38±1.51 | 38.30 |
| Mahalanobis | 95.74±0.18 | 0.144 | 0.0102 | 87.98±0.13 | 66.30 | 93.25±0.32 | 38.30 |
| ReAct+Energy | 95.77±0.20 | 0.148 | 0.0116 | 88.63±0.18 | 54.39 | 92.50±1.47 | 29.78 |
| LLLA | 95.77±0.25 | 0.140 | 0.0083 | 88.97±0.84 | 54.12 | 93.04±1.17 | 28.44 |
| Epinet | 95.76±0.22 | 0.148 | 0.0103 | 88.07±0.05 | 63.37 | 92.16±1.25 | 35.10 |
| MC Dropout | 95.76±0.14 | 0.148 | 0.0099 | 87.25±0.18 | 71.04 | 91.34±0.27 | 42.53 |
| SWAG | 95.37±0.07 | 0.146 | 0.0075 | 90.03±0.19 | 44.71 | 94.19±1.26 | 22.09 |
| **P&C (s3b0, manuscript)** | **95.59±0.25** | **0.138** | **0.0050** | **91.55±0.16** | **33.08** | **95.09±0.62** | **18.15** |
| P&C (s3b1, alternative) | 95.69±0.13 | 0.140 | 0.0069 | 90.99±0.10 | 37.39 | 94.83±0.63 | 19.52 |
| Standard (Deep) Ensemble n=5 | 96.56±0.06 | 0.109 | 0.0046 | 91.10±0.04 | 40.39 | 94.63±0.17 | 19.49 |
| **SCOD-1024** (rebuttal) | 95.74±0.18 | 0.144 | — | 89.69±0.11 | 39.40 | 92.56±0.45 | 21.41 |

SCOD row from `../scod/tables/scod_aggregate.json` (rebuttal-era, same 3 checkpoints; SCOD Acc/NLL = unchanged base classifier).

## Provenance
- **Checkpoints:** 3 independently trained PreActResNet18 (seeds 0,1,2). (7 checkpoints seeds 0–6 exist on disk; the paper uses 0,1,2.) P&C construction seed = checkpoint seed (nested; not independent construction seeds). Aggregation = mean over the 3 checkpoint×construction seeds, macro over OOD datasets within Near/Far.
- **Metric mapping:** Acc=`id_metrics.accuracy*100`, NLL=`id_metrics.nll`, ECE=`id_metrics.ece`, Near/Far AUROC=`{near,far}_ood_auroc*100`, Near/Far FPR95=`{near,far}_ood.aggregate.predictive_entropy.mean_fpr95*100`. Per-method `primary_score` varies (MSP→max_softmax, Energy→energy, Maha→mahalanobis, rest→predictive_entropy) — hence MSP/Energy/Maha AUROC differ from base while FPR95 (shared predictive_entropy aggregate) coincide at 66.30.
- **Split:** Near={cifar100,tiny_imagenet}, Far={mnist,svhn,textures,places365}, id=cifar10, `uses_ood_validation=false` — confirmed in all 12 methods × 3 seeds.
- **P&C row = `s3b0`** (`..._pnc_single_block_vcal_s3b0_k20_n50_ps25.0_lr0.001_bf0.05_..._chunksize1024_seed{0,1,2}_random.json`, mtime 04-28). s3b1 files exist (mtime 04-19) and reproduce 90.99/94.83 but are NOT the table.

## Per-seed (P&C s3b0 anchor)
| seed | Acc | NLL | ECE | nAUROC | nFPR95 | fAUROC | fFPR95 |
|---|---|---|---|---|---|---|---|
| 0 | 95.85 | 0.134 | 0.0049 | 91.41 | 33.98 | 95.69 | 17.16 |
| 1 | 95.57 | 0.139 | 0.0040 | 91.73 | 32.67 | 95.13 | 17.64 |
| 2 | 95.36 | 0.142 | 0.0061 | 91.52 | 32.61 | 94.46 | 19.64 |

Full per-seed for SWAG/LLLA/Deep-Ensemble/s3b1 in the RESULTS_AUDIT reproduction section. Raw JSONs: `results/cifar10/openood_v1p5_*_seed{0,1,2}.json`.
