# MANIFEST — matched post-hoc uncertainty baselines, ImageNet ViT-B/16

## Code
- git commit: `ccab71b1181da4fcbfbedf7929828e1cb5bc1e7b`
- branch: `agent/vit-imagenet-baselines`

## Hardware / environment
- GPU: NVIDIA TITAN X (Pascal) (12.00 GiB, sm_61); driver 581.80
- CPU: 12th Gen Intel(R) Core(TM) i7-12700KF (20 threads), RAM 24607916 kB
- torch 2.7.1+cu126 · torchvision 0.22.1+cu126 · CUDA 12.6
- numpy 2.4.4 · scipy 1.18.0 · scikit-learn 1.9.0
- python 3.12.3 (/home/elean/pnc/.venv_vit/bin/python) · Linux-6.6.87.2-microsoft-standard-WSL2-x86_64-with-glibc2.39

## Checkpoint (shared by every matched method)
- enum `ViT_B_16_Weights.IMAGENET1K_V1` · SHA-256 `c867db91d3e12c6cbadabb610d73c24a546bf82d8c03a9fea34f43a712ddb0e9` · 86,567,656 parameters
- preprocessing `ImageClassification(     crop_size=[224]     resize_size=[256]     mean=[0.485, 0.456, 0.406]     std=[0.229, 0.224, 0.225]     interpolation=InterpolationMode.BILINEAR )`

## Data (reused byte-for-byte from ../imagenet_vit/)
- ImageNet source `ILSVRC/imagenet-1k` (gated originals); split seed 20260813
- OOD image lists: OpenOOD canonical imglists via `torch-uncertainty/ood-datasets-splits`
- OOD images: `torch-uncertainty` mirrors — SSB-hard 49,000 · NINCO 5,879 · iNaturalist 10,000 · Textures 5,160 · OpenImage-O 15,869

| pool | n | SHA-256 |
|---|---|---|
| correction | 32768 | `08affc7e865e58cee7b3aa110e83072e861760b97894ee7d087c6c02d160b64d` |
| selection | 8192 | `71edce32d39bd6a17a2a56d0993ba73d05fb05278764710428d38de8529078dd` |
| temperature | 8192 | `3271a90afd4a7c3aebd6936e4083ef92dd69876e63416eb9bafb7e95e3e23ee2` |

## Method sources (ported, not reinvented)
| method | source | calibration N | selection grid | selected | samples/img | temperature |
|---|---|---|---|---|---|---|
| MSP | `experiments/imagenet_vit_pnc/full_baselines.py` (reused scores) | 0 | — | — | 1 | none (raw logits) |
| Energy | same | 0 | — | — | 1 | none (raw logits) |
| ReAct+Energy | same | 8,192 | fixed p90 (canonical) | c = 0.6918 | 1 | none |
| Mahalanobis | `pnc_core/openood_eval.py::_fit_mahalanobis` | 32,768 | none (fixed 1e-6 ridge) | — | 1 | none |
| Laplace (KFAC) | `pnc_core/ensembles.py::LaplaceEnsemble` + `pnc_core/laplace.py::compute_kfac_factors(is_classification=True)` | 32,768 | [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0, 30000.0, 100000.0, 300000.0, 1000000.0, 10000000.0, 100000000.0] | λ = 10000 | 20 | T = 0.7 (reused ID fit) |
| P&C (frozen) | `../imagenet_vit_preservation_frontier/` | 32,768 | frozen, not retuned | r=2.0, λ=1000 | 20 | T = 0.700 |
| SCOD | `experiments/posthoc/scod_*.py` | — | — | not run | — | — |

- Laplace selection criterion: **ID selection-pool NLL**; `OOD data accessed before selection: NO`
- construction seeds (P&C): 0, 10, 42, 123, 2026

## SCOD feasibility (measured, §13)
- P = 86,567,656; repo config `num_eigs_max=100`, `num_samples=604` (`configs/mujoco_posthoc.yaml`)
- one Fisher matvec: 50.9 ms/image at N=1,024, peak 5.08 GiB
- extrapolated: 27.8 min per matvec, 280 h for the full sketch
- sketch storage 390 GiB (rank-10 fallback still 41 GiB)
- verdict **SCOD_NOT_TRACTABLE_AT_VIT_SCALE**

## Commands
```bash
.venv_vit/bin/python -m experiments.imagenet_vit_pnc.baselines_run --stage parity
.venv_vit/bin/python -m experiments.imagenet_vit_pnc.baselines_run --stage fit
.venv_vit/bin/python -m experiments.imagenet_vit_pnc.baselines_run --stage scod
.venv_vit/bin/python -m experiments.imagenet_vit_pnc.baselines_tables
```

## Parity
The common harness reproduces the completed ViT experiment on 9/9 checks to <1e-6
(base top-1/top-5/NLL/ECE, ReAct threshold, MSP and Energy Near/Far AUROC) —
`provenance/parity_gate.json`.
