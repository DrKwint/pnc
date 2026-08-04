# PROVENANCE — NeurIPS 2026 rebuttal experiments

Global reproducibility record. Per-experiment provenance (command, config, seed, runtime,
checkpoint) is recorded alongside each result CSV/JSON under this directory.

## Git
- Branch: `neurips-2026-rebuttal`
- HEAD commit: `70fb480fadcc4ee8e9d8ae0b8a060de431007aa0`
- Working tree has uncommitted edits from the prior "self-tuning P&C" investigation
  (see INVENTORY.md §4.3). P&C reproduction pins `hidden_dims=[200,200,200,200]`.

## Hardware
- GPU (JAX device): cuda:0
- GPU (nvidia-smi): NVIDIA TITAN X (Pascal), 12288 MiB
- CPU: 12th Gen Intel(R) Core(TM) i7-12700KF (20 threads)
- RAM: 15Gi
- Platform: linux (WSL2)

## Software
- Python: Python 3.12.3
- JAX: 0.9.1
- Torch: 2.11.0+cu130
- Interpreter: /home/elean/pnc/.venv/bin/python (ALWAYS use this)

## Policy
- One GPU job at a time (strictly sequential).
- No OOD data used for hyperparameter/model selection (ID-only protocol preserved).
- Do not overwrite existing outputs/checkpoints; all new outputs under results/neurips_2026_rebuttal/.
