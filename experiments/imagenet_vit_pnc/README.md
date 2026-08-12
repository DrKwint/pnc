# ImageNet ViT-B/16 P&C — TITAN X feasibility preflight

Staged memory, numerical-correctness and throughput preflight for running P&C on a
pretrained torchvision ViT-B/16 on the 12 GB TITAN X (Pascal). **This is a preflight only —
it does not run the ImageNet/OpenOOD experiment, and it never loads OOD data.**

Results, report and the reuse map live in
`results/neurips_2026_rebuttal/imagenet_vit_preflight/`.

| File | What it is |
|---|---|
| `vit_adapter.py` | Pinned ViT-B/16 checkpoint, prefix/tail split around the target FFN, in-place mutation + exact restore |
| `pnc_core.py` | Perturbation basis/coefficient/scale conventions, streamed `G`/`C` sufficient statistics, ridge solve from statistics, shared Cholesky, deterministic token sampling |
| `memprobe.py` | CUDA/CPU memory + timing probes, OOM-tolerant incremental CSV recording |
| `data.py` | ImageNet-1k validation (both val shards, 50k images) with the pinned transform. ID only — no OOD code path |
| `validate.py` | The nine correctness gates. Run these before believing any measurement |
| `stages_memory.py` | Spec §6–9, §15–16: batch sweep, sustained throughput, activations, basis, perturbed member, ensemble, storage |
| `stages_correction.py` | Spec §10–14, §17: token modes, row budgets, ridge benchmark, corrected member, cached-(h,z) construction, fp16 |
| `run_preflight.py` | Stage driver; stages share `raw/state.json` |
| `project_runtime.py` | Spec §18: full-experiment runtime projection from measured primitives |

## Reuse

The ridge solver is the existing `experiments/scripts/pnc_theory/linalg.py::ridge_solve`,
and the perturbation conventions come from `experiments/banking77_pnc/construct.py` — that
experiment targets an FFN with the identical `768 → 3072 → 768` geometry. Where an exact
import was impossible, the reformulation is gated against the original in `validate.py`
rather than assumed equivalent. See
[`REUSE_MAP.md`](../../results/neurips_2026_rebuttal/imagenet_vit_preflight/REUSE_MAP.md).

## Running

Use `.venv_vit`, not `.venv` — the main venv's torch 2.11+cu130 has no `sm_61` kernels and
fails on this card. See
[`REPRODUCE.md`](../../results/neurips_2026_rebuttal/imagenet_vit_preflight/REPRODUCE.md)
for the full sequence. Run GPU stages one at a time.
