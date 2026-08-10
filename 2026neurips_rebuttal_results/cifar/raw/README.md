# raw/ — pointers to large uncopied raw trees

Large raw artifacts are NOT copied here (kept out of git). Recover from these on-machine paths
(repo DrKwint/pnc, machine /home/elean/pnc); checksums for checkpoints/sketches in ../MANIFEST_large_artifacts.md.

- Submitted per-method OpenOOD JSONs (all methods, 3 seeds): `results/cifar10/openood_v1p5_*_seed{0,1,2}.json`
- Original P&C sensitivity JSONs (ps/bf/λ/K sweeps): `results/cifar10/pnc_single_block_*.json`, `results/cifar10/pnc_multi_block_*.json`
- Base checkpoints: `results/cifar10/preact_resnet18_train_e300_..._seed{0..6}.pkl`
- SCOD full tree (sketches, predictions, plots): `results/scod_cifar/`
- Joint grid full tree (162 candidates + val_predictions.npz + per-seed OpenOOD): `results/neurips_2026_rebuttal/cifar/pnc_full_grid/`
- Sensitivity sweep shards + per-cell JSONs: `results/neurips_2026_rebuttal/cifar/sweeps/`
- Peak-memory per-method JSONs: `results/neurips_2026_rebuttal/cifar/peak_memory/` (also copied to ../storage_memory/peak_memory/)
- Off-machine report copies: /mnt/c/Users/CARVA/OneDrive/Desktop/neurips_2026_rebuttal/ (reports only, no raw)
