# MANIFEST — ImageNet ViT-B/16 final-block P&C
## Code
- git commit: `ca851208ca1d3d2fca41973778b2ee323e40566f`
- branch: `agent/vit-imagenet-full`
- working tree clean: no (see diff)
## Environment
- GPU: NVIDIA TITAN X (Pascal) (12.00 GiB, sm_61)
- driver: 581.80
- CUDA (torch runtime): 12.6
- torch: 2.7.1+cu126
- torchvision: 0.22.1+cu126
- python: 3.12.3  (/home/elean/pnc/.venv_vit/bin/python)
- CPU: 12th Gen Intel(R) Core(TM) i7-12700KF (20 threads), RAM 24607916 kB
- platform: Linux-6.6.87.2-microsoft-standard-WSL2-x86_64-with-glibc2.39
## Checkpoint
- weight enum: `ViT_B_16_Weights.IMAGENET1K_V1`
- url: https://download.pytorch.org/models/vit_b_16-c867db91.pth
- SHA-256: `c867db91d3e12c6cbadabb610d73c24a546bf82d8c03a9fea34f43a712ddb0e9`
- parameters: 86,567,656
- preprocessing: `ImageClassification(     crop_size=[224]     resize_size=[256]     mean=[0.485, 0.456, 0.406]     std=[0.229, 0.224, 0.225]     interpolation=InterpolationMode.BILINEAR )`
## Data
- ImageNet source: `ILSVRC/imagenet-1k` (gated ILSVRC-2012 originals), 28 train shards + all validation shards
- training rows indexed: 122,024
- ID evaluation: the official 50,000-image validation split, untouched by construction, selection and calibration
- OOD splits: OpenOOD canonical image lists from `torch-uncertainty/ood-datasets-splits`; images from the `torch-uncertainty` mirrors
| pool | n | classes | per-class | SHA-256 |
|---|---|---|---|---|
| correction | 32768 | 1000 | 32-33 | `08affc7e865e58cee7b3aa110e83072e861760b97894ee7d087c6c02d160b64d` |
| selection | 8192 | 1000 | 8-9 | `71edce32d39bd6a17a2a56d0993ba73d05fb05278764710428d38de8529078dd` |
| temperature | 8192 | 1000 | 8-9 | `3271a90afd4a7c3aebd6936e4083ef92dd69876e63416eb9bafb7e95e3e23ee2` |
- split seed: 20260813
| OOD dataset | n | group |
|---|---|---|
| ssb_hard | 49000 | near |
| ninco | 5879 | near |
| inaturalist | 10000 | far |
| textures | 5160 | far |
| openimage_o | 15869 | far |
## P&C configuration
- target block: `encoder.layers.encoder_layer_11`
- perturbed: `.mlp.0` (W1, 768->3072); corrected: `.mlp.3` (W2, 3072->768)
- correction observation: cls token only, 1 row per calibration image
- K: 20   M (final): 20
- construction seeds: [0, 10, 42, 123, 2026]
### Searched (ID data only)
- scales r (Stage A): [0.125, 0.25, 0.5, 1.0, 2.0]
- scales r (Stage B): [0.25, 0.375, 0.5]
- Stage A r_boundary: 0.5
- ridge lambda: [0.001, 1.0, 100.0]
- correction sizes: [16384, 32768]
- ID-stability gate: {"max_top1_drop_pp": 0.25, "min_base_agreement": 0.99, "require_finite": true, "require_corrected_median_lt_uncorrected": true, "require_corrected_p99_le_uncorrected": true}
### Selected
```json
{
  "r_target": 0.375,
  "r_realized_median": 0.37499968707561493,
  "scale": 4.923062801361084,
  "n_cal": 16384,
  "lambda": 1.0,
  "K": 20,
  "M_final": 20,
  "token_mode": "cls",
  "target_block": 11,
  "selection_nll": 0.3260294048216372,
  "selection_top1": 0.93798828125,
  "base_top1_selection_pool": 0.938232421875,
  "n_survivors": 15,
  "n_tied": 8,
  "ood_data_accessed_before_freeze": false
}
```
## Calibration and scoring
- temperature: 0.7 — shared scalar T, grid search on base model logits over the ID temperature pool; applied to every member's logits before softmax (matches banking77_pnc/evaluate.py)
- P&C OOD score: predictive entropy of the temperature-scaled mean member softmax (fixed before OOD evaluation)
- MSP: -max softmax. Energy: -logsumexp(logits). ReAct+Energy: Energy after clipping penultimate features at the ID p90
- metrics/aggregation: `full_oodmetrics`, gated against `pnc_core/openood_eval.py`; Near/Far are macro means over their datasets
## Commands
```bash
.venv_vit/bin/python -m experiments.imagenet_vit_pnc.full_splits
.venv_vit/bin/python -m experiments.imagenet_vit_pnc.full_run --stage parity
.venv_vit/bin/python -m experiments.imagenet_vit_pnc.full_run --stage cache
.venv_vit/bin/python -m experiments.imagenet_vit_pnc.full_run --stage stage_a
.venv_vit/bin/python -m experiments.imagenet_vit_pnc.full_run --stage stage_b
.venv_vit/bin/python -m experiments.imagenet_vit_pnc.full_run --stage robustness
.venv_vit/bin/python -m experiments.imagenet_vit_pnc.full_run --stage final --n-seeds 5
.venv_vit/bin/python -m experiments.imagenet_vit_pnc.full_run --stage ood
.venv_vit/bin/python -m experiments.imagenet_vit_pnc.full_timing
.venv_vit/bin/python -m experiments.imagenet_vit_pnc.full_tables
.venv_vit/bin/python -m experiments.imagenet_vit_pnc.full_validate
```
## Reproducibility notes
- every table is generated from `metrics/*.json` and `selection/*.csv` by `full_tables.py`; no number is transcribed by hand
- images, checkpoints and activation caches are **not** committed; their sources and checksums are recorded above
- `raw/cache_*.npz` and `raw/base_val_logits.npy` are regenerable from the commands above
