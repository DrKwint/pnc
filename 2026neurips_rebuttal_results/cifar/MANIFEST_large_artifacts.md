# CIFAR results MANIFEST — large-artifact index

Large artifacts (checkpoints, sketches, prediction caches, OOD input datasets) are **NOT committed**. Recorded here by exact path, size, SHA-256 (where computed), and config id. All paths under `/home/elean/pnc/` (repo DrKwint/pnc, branch neurips-2026-rebuttal, commit 854f4d8; artifacts git-ignored/untracked).

## Base checkpoints (PreActResNet18, e300) — 3 used (seeds 0,1,2); seeds 0–6 exist
| path | size | SHA-256 | config id |
|---|---|---|---|
| `results/cifar10/preact_resnet18_train_e300_optsgd_lr1e-01_wd5e-04_bs128_wu5_mom0p9_n1_augfcco8_ls0_seed0.pkl` | 44729016 B (42.7 MiB) | `b4c45628adaa9e5321e15ae83679841e2474e23c2126968fe6cdffcb2143a7e5` | base_seed0 |
| `results/cifar10/preact_resnet18_train_e300_optsgd_lr1e-01_wd5e-04_bs128_wu5_mom0p9_n1_augfcco8_ls0_seed1.pkl` | 44729016 B (42.7 MiB) | `2e942d691915025dac8e15d56aeaf289a4bebaa065184338f6a1ef18105674b4` | base_seed1 |
| `results/cifar10/preact_resnet18_train_e300_optsgd_lr1e-01_wd5e-04_bs128_wu5_mom0p9_n1_augfcco8_ls0_seed2.pkl` | 44729016 B (42.7 MiB) | `9b5d9de691ba2d51f63bdb8cf9445ee05e7cdc292e6c57c68847ff77e8793035` | base_seed2 |

## SCOD sketches (large; size only)
| path | size | config id |
|---|---|---|
| `results/scod_cifar/sketches/scod1024_seed0_sketchseed100000_tempered.npz` | 852 MiB | scod1024_seed0_sketchseed100000_tempered |
| `results/scod_cifar/sketches/scod1024_seed0_sketchseed100000_untempered.npz` | 852 MiB | scod1024_seed0_sketchseed100000_untempered |
| `results/scod_cifar/sketches/scod1024_seed1_sketchseed100001_tempered.npz` | 852 MiB | scod1024_seed1_sketchseed100001_tempered |
| `results/scod_cifar/sketches/scod1024_seed1_sketchseed100001_untempered.npz` | 852 MiB | scod1024_seed1_sketchseed100001_untempered |
| `results/scod_cifar/sketches/scod1024_seed2_sketchseed100002_tempered.npz` | 852 MiB | scod1024_seed2_sketchseed100002_tempered |
| `results/scod_cifar/sketches/scod1024_seed2_sketchseed100002_untempered.npz` | 852 MiB | scod1024_seed2_sketchseed100002_untempered |
