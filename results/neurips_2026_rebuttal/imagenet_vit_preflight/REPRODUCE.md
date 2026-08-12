# Reproducing the ViT-B/16 P&C TITAN X preflight

All code lives in `experiments/imagenet_vit_pnc/`. Run from the repo root.

## 0. Untracked dependency (read first)

This preflight imports two things that are **not committed to the repository** — they exist
only in the working tree (`git status` reports both as untracked):

```text
experiments/scripts/pnc_theory/     # ridge_solve, the reused P&C solver
experiments/banking77_pnc/          # the DistilBERT P&C conventions gated against
```

`validate.py` will fail on a fresh clone until those are committed or restored. Nothing
here modifies them; flagging it because the reuse gates depend on their exact contents.

## 1. Environment

The repo's main `.venv` carries torch 2.11.0+cu130, whose kernels start at `sm_75`. The
TITAN X Pascal is `sm_61`, so every CUDA call there fails with
`CUDA error: no kernel image is available for execution on the device`. This preflight
uses a sibling venv (same convention as `.venv_bank`, covered by the `.venv_*/` gitignore
rule); the main `.venv` is left untouched.

```bash
/usr/bin/python3 -m venv .venv_vit
.venv_vit/bin/python -m pip install --index-url https://download.pytorch.org/whl/cu126 \
    torch==2.7.1 torchvision==0.22.1
.venv_vit/bin/python -m pip install numpy scipy pillow huggingface_hub pyarrow
```

torch 2.7.1+cu126 ships `sm_60` cubins, which are binary-compatible with sm_61 (same major
compute-capability version). Verify with:

```bash
.venv_vit/bin/python -c "import torch; print(torch.cuda.get_arch_list()); \
    print((torch.randn(64,64,device='cuda')@torch.randn(64,64,device='cuda')).sum())"
```

## 2. Correctness gates (run these first)

```bash
.venv_vit/bin/python -m experiments.imagenet_vit_pnc.validate \
    --out results/neurips_2026_rebuttal/imagenet_vit_preflight/raw/validate_gates.json

# basis/coefficient/scale parity against Banking77 needs jax, so use the main venv:
JAX_PLATFORMS=cpu .venv/bin/python -m experiments.imagenet_vit_pnc.validate --only basis_parity
```

Nine gates must pass, including bitwise equality between the streamed ridge solve and
`pnc_theory.linalg.ridge_solve`, and bitwise equality of the perturbation conventions with
`experiments/banking77_pnc/construct.py`.

## 3. Environment inventory

```bash
.venv_vit/bin/python -m experiments.imagenet_vit_pnc.environment
```

## 4. Measurement stages

Run sequentially — never two GPU jobs at once on this card.

```bash
# sections 6-9: batch sweep, sustained throughput, activations, basis, perturbed member
.venv_vit/bin/python -m experiments.imagenet_vit_pnc.run_preflight --stage memory --sustain-s 90

# sections 10-14: token modes, row budgets, ridge benchmark, corrected member
.venv_vit/bin/python -m experiments.imagenet_vit_pnc.run_preflight \
    --stage correction --n-calib 40000 --n-heldout 8192

# ID-preservation distribution vs row budget and ridge (the decisive quality measurement)
.venv_vit/bin/python -m experiments.imagenet_vit_pnc.run_preflight \
    --stage quality --token-mode cls --n-calib 40000 --n-heldout 8192

# section 14: one complete corrected member at the recommended configuration
.venv_vit/bin/python -m experiments.imagenet_vit_pnc.run_preflight \
    --stage member --token-mode cls --row-budget 32768 --n-calib 40000 --n-heldout 8192

# sections 15-17: cached-(h,z) construction, shared-prefix M sweep, storage, fp16
.venv_vit/bin/python -m experiments.imagenet_vit_pnc.run_preflight \
    --stage ensemble --token-mode cls --row-budget 32768 --naive-members 1 \
    --n-calib 40000 --n-heldout 8192

# section 18: runtime projection from the measured primitives
# (needs raw/block_tail_costs.json -- per-block tail costs for blocks 11/10/9)
.venv_vit/bin/python -m experiments.imagenet_vit_pnc.project_runtime
```

Stages communicate through `raw/state.json` (the safe batch size chosen in `memory` is
reused downstream), so run `memory` before the others.

## Artifacts not committed

`raw/*.npz` (the shared basis and the 20 corrected W2 matrices, 335 MiB) are gitignored:
they are regenerable from the fixed seeds, and the figure that matters — their storage
footprint — is recorded in `raw/stage_ensemble.json` and the report. The largest file
otherwise tracked in this repo is 584 KiB.

## Data

ImageNet-1k validation is pulled from the public HuggingFace mirror
`evanarlian/imagenet_1k_resized_256` (both val shards, 50,000 images, ~0.85 GiB) and cached
under `~/.cache/huggingface`. No credentials required. The end-to-end check is
`data.accuracy_gate`, which reproduces the published 81.07% top-1 / 95.32% top-5 on a
held-out sample.

**No OOD data is downloaded or evaluated anywhere in this preflight** (spec section 21).
