# MANIFEST — ImageNet ViT-B/16 preservation-frontier follow-up

## Code

- git commit: `74f6d56c620acebd22237ad61a93278316c95c5f`
- branch: `agent/vit-preservation-frontier`
- selection rule committed at: `e25c8c9` (before any follow-up OOD)
- frozen configs committed at: `74f6d56` (before any follow-up OOD)

## Environment

- GPU: NVIDIA TITAN X (Pascal) (12.00 GiB, sm_61)
- driver: 581.80
- torch 2.7.1+cu126 / torchvision 0.22.1+cu126 / CUDA 12.6
- python 3.12.3 (/home/elean/pnc/.venv_vit/bin/python)
- CPU: 12th Gen Intel(R) Core(TM) i7-12700KF (20 threads)
- platform: Linux-6.6.87.2-microsoft-standard-WSL2-x86_64-with-glibc2.39

## Checkpoint (unchanged from the original experiment)

- weight enum: `ViT_B_16_Weights.IMAGENET1K_V1`
- SHA-256: `c867db91d3e12c6cbadabb610d73c24a546bf82d8c03a9fea34f43a712ddb0e9`
- parameters: 86,567,656
- preprocessing: `ImageClassification(     crop_size=[224]     resize_size=[256]     mean=[0.485, 0.456, 0.406]     std=[0.229, 0.224, 0.225]     interpolation=InterpolationMode.BILINEAR )`

## Reused byte-for-byte from ../imagenet_vit/

- split manifests (correction 32,768 / selection 8,192 / temperature 8,192), seed 20260813
- activation caches: `cache_correction`, `cache_selection`, `cache_val50k`, `cache_ood_*`
- base validation logits
- shared temperature T = 0.7 (NOT refitted per configuration)
- deterministic MSP / Energy / ReAct+Energy OOD scores
- ImageNet source: `ILSVRC/imagenet-1k` (gated originals); OOD lists: OpenOOD canonical imglists

| pool | n | classes | SHA-256 |
|---|---|---|---|
| correction | 32768 | 1000 | `08affc7e865e58cee7b3aa110e83072e861760b97894ee7d087c6c02d160b64d` |
| selection | 8192 | 1000 | `71edce32d39bd6a17a2a56d0993ba73d05fb05278764710428d38de8529078dd` |
| temperature | 8192 | 1000 | `3271a90afd4a7c3aebd6936e4083ef92dd69876e63416eb9bafb7e95e3e23ee2` |

## P&C configuration (fixed, not searched)

- target: `encoder.layers.encoder_layer_11.mlp.0` (perturb W1) -> GELU -> `.mlp.3` (correct W2)
- correction observation: CLS token only, 1 row per calibration image
- K = 20, n_cal = 32,768, dtype float32, ridge solve float64 CPU, original-centred
- OOD score: predictive entropy of the temperature-scaled mean member softmax

## Searched (ID data only)

- scales r: [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.125, 2.25, 2.5, 3.0, 4.0]
- ridge lambda: [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]
- search: M = 10, seeds [0, 10, 42]; final: M = 20, seeds [0, 10, 42, 123, 2026]
- budgets (pp): {'strict': 0.25, 'primary': 0.5, 'relaxed': 1.0}
- preservation test: paired bootstrap, 10,000 replicates, seed 20260814, one-sided 95% LCB
- OOD accessed before freeze: **False**

## Selected

```json
{
  "strict": {
    "budget": "strict",
    "max_top1_loss_pp": 0.25,
    "r_target": 1.25,
    "realized_r_median": 1.2500061551334198,
    "lambda": 1000.0,
    "n_cal": 32768,
    "selection_delta_top1_pp": -0.09358723958333703,
    "selection_lcb_pp": -0.2034505208333334,
    "boundary": {
      "largest_passing_r": 1.25,
      "smallest_failing_r_above": 1.5,
      "tested": {
        "0.5": true,
        "0.75": true,
        "1.0": true,
        "1.25": true,
        "1.5": false,
        "2.0": false,
        "2.125": false,
        "2.25": false,
        "2.5": false,
        "3.0": false,
        "4.0": false
      }
    }
  },
  "primary": {
    "budget": "primary",
    "max_top1_loss_pp": 0.5,
    "r_target": 2.0,
    "realized_r_median": 2.0000098693897583,
    "lambda": 1000.0,
    "n_cal": 32768,
    "selection_delta_top1_pp": -0.30517578125,
    "selection_lcb_pp": -0.45979817708333337,
    "boundary": {
      "largest_passing_r": 2.0,
      "smallest_failing_r_above": 2.125,
      "tested": {
        "0.5": true,
        "0.75": true,
        "1.0": true,
        "1.25": true,
        "1.5": true,
        "2.0": true,
        "2.125": false,
        "2.25": false,
        "2.5": false,
        "3.0": false,
        "4.0": false
      }
    }
  },
  "relaxed": {
    "budget": "relaxed",
    "max_top1_loss_pp": 1.0,
    "r_target": 3.0,
    "realized_r_median": 3.000014751135506,
    "lambda": 1000.0,
    "n_cal": 32768,
    "selection_delta_top1_pp": -0.655110677083337,
    "selection_lcb_pp": -0.8544921875000002,
    "boundary": {
      "largest_passing_r": 3.0,
      "smallest_failing_r_above": 4.0,
      "tested": {
        "0.5": true,
        "0.75": true,
        "1.0": true,
        "1.25": true,
        "1.5": true,
        "2.0": true,
        "2.125": true,
        "2.25": true,
        "2.5": true,
        "3.0": true,
        "4.0": false
      }
    }
  }
}
```

## Budget re-validation on the untouched 50k validation set (M = 20, 5 seeds)

| config | Δ top-1 pp | 95% LCB pp | budget pp | result |
|---|---|---|---|---|
| strict | -0.100 | -0.153 | -0.25 | PASS |
| primary | -0.238 | -0.315 | -0.50 | PASS |
| relaxed | -0.470 | -0.571 | -1.00 | PASS |

## Commands

```bash
.venv_vit/bin/python -m experiments.imagenet_vit_pnc.frontier_run --stage search
.venv_vit/bin/python -m experiments.imagenet_vit_pnc.frontier_run --stage final --n-seeds 5
.venv_vit/bin/python -m experiments.imagenet_vit_pnc.frontier_run --stage ood --n-seeds 5
.venv_vit/bin/python -m experiments.imagenet_vit_pnc.frontier_tables
```

## Provenance caveat

This protocol is a follow-up motivated by the earlier experiment, which had already
inspected OOD performance at several scales including r = 1 and r = 2. It is **not** a
pristine preregistered OOD experiment. The narrower claim that does hold: once the rule was
fixed and committed, the configurations it selected were determined by ID data alone, and
the predeclared budgets were not revised after OOD results were seen. The original
cleanly-selected result is preserved unchanged in `../imagenet_vit/`.

## Not committed

`raw/members_*.npz` (member weight matrices) are gitignored and regenerable from the
commands above. No images, checkpoints or activation caches are stored here.

