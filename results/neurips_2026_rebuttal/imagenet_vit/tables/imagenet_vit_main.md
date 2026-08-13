| Method | ID Acc | ID NLL | ID ECE | Near AUROC | Near FPR95 | Far AUROC | Far FPR95 |
|---|---|---|---|---|---|---|---|
| Base / MSP | 81.07 | 0.8482 | 0.0913 | 73.52 | 81.84 | 86.04 | 51.74 |
| Energy | 81.07 | 0.8482 | 0.0913 | 62.39 | 93.16 | 78.96 | 85.29 |
| ReAct + Energy | 81.07 | 0.8482 | 0.0913 | 69.21 | 84.23 | 85.61 | 53.90 |
| Uncorrected perturb. | 81.05 ± 0.02 | 0.8440 ± 0.0005 | 0.0896 ± 0.0004 | 74.60 ± 0.02 | 74.63 ± 0.35 | 86.52 ± 0.02 | 48.70 ± 0.15 |
| P&C | 81.04 ± 0.01 | 0.8431 ± 0.0004 | 0.0901 ± 0.0002 | 74.74 ± 0.02 | 74.37 ± 0.14 | 86.52 ± 0.01 | 48.75 ± 0.08 |

ImageNet-1k, ViT-B/16 (ViT_B_16_Weights.IMAGENET1K_V1). Accuracy/AUROC/FPR95 in %. P&C and the uncorrected ablation are mean ± std over 5 construction seeds at M=20; MSP, Energy and ReAct+Energy are deterministic post-hoc scores on the same base model. Near = SSB-hard, NINCO. Far = iNaturalist, Textures, OpenImage-O (macro mean over datasets).
