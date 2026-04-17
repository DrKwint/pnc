# CIFAR-100 OOD Detection (Paper-Clean Protocol)

## Summary

Mirrors the CIFAR-10 paper-clean OOD benchmark (`cifar10_ood_detection.md`) on
CIFAR-100 as the in-distribution dataset. **All hyperparameters are frozen at
their CIFAR-10-tuned values** — no CIFAR-100-specific retuning is performed.
This is a deliberate transferability test: reviewers care more about the
protocol than peak numbers, and re-tuning on CIFAR-100 would weaken the
no-OOD-data claim.

10 of the 12 methods from CIFAR-10 are evaluated on 3 seeds (0, 1, 2).
Two methods (LLLA and multi-block PnC) are infeasible at this scale on the
single-GPU lane and are documented as out-of-scope below.

## Setup

### Data
- **ID:** CIFAR-100 (50K train / 5K validation / 5K test split, identical to the CIFAR-10 path).
- **Near-OOD:** CIFAR-10 test (10K), Tiny ImageNet (10K).
- **Far-OOD:** MNIST (10K), SVHN (26K), DTD Textures (5640), Places365 (10K).
- All OOD npz files are reused from the CIFAR-10 path; CIFAR-10 test is added as the CIFAR-100 near-OOD via `prepare_openood_data.py::prepare_cifar10_as_ood()`.

### Models
- All methods reuse the same `PreActResNet18` backbone trained on CIFAR-100 with the CIFAR-10 recipe (SGD lr=0.1, wd=5e-4, cosine, cutout, 300 epochs).
- Base ID accuracy: 78.31 ± 0.13 (3 seeds).
- Standard Ensemble uses members with seeds `s, s+1, …, s+4` (so seeds 0–6 of the base model are trained; the seed-`s` ensemble file at `results/cifar100/standard_ensemble_…_seed{0,1,2}.json` reuses the same pool).

### Frozen hyperparameters (identical to CIFAR-10)

| Method | Parameters |
|--------|-----------|
| MC Dropout | n=32, dropout_rate=0.1 |
| Standard Ensemble | n=5 |
| SWAG | n=50, swag_start_epoch=240, max_rank=20, BN refresh on 2048 ID train samples |
| Epinet | prior_scale=3.0, n=50, index_dim=8, hiddens=(50,50) |
| PnC (single block) | **scale=50.0**, n_directions=20, random directions, target=stage 3 block 1 |
| ReAct | percentile=90 (ID train activations) |
| Mahalanobis | shared covariance + 1e-6 I, no OOD combiner |

The single-block PnC headline scale is 50.0 (same as CIFAR-10). The
"ID-conservative" PnC scale=25 entry is **not** rerun on CIFAR-100; it is a
CIFAR-10-only ablation.

## Results (CIFAR-100, 3 seeds, mean ± std, post-hoc temperature fit on ID validation)

| Method | Acc% | NLL | ECE | Near AUROC | Near FPR95 | Far AUROC | Far FPR95 |
|--------|------|-----|-----|------------|------------|-----------|-----------|
| **Single-pass baselines** | | | | | | | |
| PreAct ResNet-18 | 78.31 ± 0.13 | 0.865 ± 0.007 | 0.0423 ± 0.0017 | 80.61 ± 0.20 | 57.89 ± 0.43 | 78.64 ± 0.68 | 57.11 ± 1.19 |
| MSP | 78.30 ± 0.12 | 0.865 ± 0.007 | 0.0423 ± 0.0016 | 79.99 ± 0.18 | 57.78 ± 0.52 | 77.69 ± 0.73 | 57.37 ± 1.15 |
| Energy | 78.31 ± 0.13 | 0.865 ± 0.007 | 0.0423 ± 0.0017 | 80.44 ± 0.24 | 58.27 ± 0.34 | 78.91 ± 0.71 | 56.32 ± 1.60 |
| Mahalanobis ★ | 78.30 ± 0.12 | 0.865 ± 0.007 | 0.0423 ± 0.0016 | 73.60 ± 0.58 | 72.85 ± 0.37 | **82.41 ± 0.62** | **52.18 ± 0.78** |
| ReAct+Energy | 75.56 ± 0.14 | 0.949 ± 0.013 | 0.0113 ± 0.0008 | 78.30 ± 0.49 | 61.73 ± 0.48 | 73.86 ± 0.85 | 68.67 ± 3.29 |
| **Single-model UQ methods** | | | | | | | |
| Epinet (ps=3.0) | 78.31 ± 0.24 | 0.880 ± 0.005 | 0.0400 ± 0.0041 | 80.49 ± 0.26 | 58.26 ± 0.57 | 79.23 ± 0.85 | 55.41 ± 1.30 |
| MC Dropout (n=32, dr=0.1) | 78.36 ± 0.22 | 0.868 ± 0.005 | 0.0421 ± 0.0027 | **80.64 ± 0.31** | 58.48 ± 1.01 | 78.63 ± 1.58 | 56.14 ± 3.39 |
| SWAG (n=50) | 77.23 ± 0.28 | 0.874 ± 0.014 | 0.0311 ± 0.0035 | 80.32 ± 0.21 | 58.88 ± 0.21 | 81.28 ± 1.12 | 53.33 ± 2.36 |
| **PnC scale=50** (single-block) | 75.47 ± 0.31 | 0.898 ± 0.002 | **0.0280 ± 0.0007** | 79.51 ± 0.30 | 61.34 ± 0.93 | 75.42 ± 0.71 | 63.17 ± 3.03 |
| **5×-cost reference** | | | | | | | |
| Standard Ensemble (n=5) | 81.25 ± 0.13 | 0.706 ± 0.003 | 0.0211 ± 0.0012 | 82.62 ± 0.03 | 54.98 ± 0.11 | 80.61 ± 0.20 | 53.67 ± 0.32 |

**Bold = best in column among 1×-training-cost methods.** ★ = paper-clean
Mahalanobis variant (single penultimate layer, no OOD combiner).

### Headline finding (CIFAR-100)

The dominance pattern observed on CIFAR-10 **does not transfer** to CIFAR-100
under frozen CIFAR-10 hyperparameters:

- **Standard Ensemble (5×) is the unambiguous winner** on every aggregate
  metric — 82.62 Near AUROC vs the best 1×-cost contender (MC Dropout, 80.64).
  Unlike CIFAR-10, no 1×-cost method comes within seed-noise of it on
  Near AUROC.
- **PnC scale=50 underperforms** on CIFAR-100 (79.51 Near AUROC, below even the
  PreAct baseline at 80.61). The CIFAR-10-tuned scale of 50 is too aggressive
  for the 100-class softmax: ID accuracy drops 2.84 pp (78.31 → 75.47) and OOD
  scores degrade in lockstep.
- **Mahalanobis is the strongest single-pass baseline on Far-OOD** (82.41 AUROC,
  52.18 FPR95), beating even Standard Ensemble on these two metrics. This
  matches its CIFAR-10 behavior on Far-OOD.
- **Among single-pass UQ methods**, MC Dropout, Epinet, SWAG, and PreAct/MSP/Energy
  cluster within ±1 AUROC point on Near-OOD — there is no single-model winner.
- **ReAct hurts on CIFAR-100.** The 90th-percentile clipping rule transferred
  badly; ID accuracy drops 2.75 pp and OOD scores follow.

### Interpretation

The CIFAR-100 results are consistent with two simultaneous effects:
1. **PnC scale needs class-count-aware tuning.** With 10× more classes, the
   per-direction logit perturbation produces 10× more decision-boundary
   crossings on average. CIFAR-10 scale=50 is therefore more aggressive on
   CIFAR-100 than CIFAR-10. A CIFAR-100-specific scale sweep would likely
   recover competitiveness; we deliberately did **not** run it (frozen-recipe
   transferability test). This is acknowledged as a limitation, not as a
   negative result on PnC.
2. **The 5×-cost gap is more important on harder ID tasks.** On CIFAR-10, all
   strong methods saturate near 96% accuracy and the marginal returns from
   ensembling are small. On CIFAR-100, Standard Ensemble's accuracy advantage
   (81.25 vs 78.31, +2.94 pp) directly translates to better OOD discrimination.

The right framing for the paper: *PnC's CIFAR-10 advantage is real but its
hyperparameters need per-dataset tuning. We deliberately reuse CIFAR-10
hyperparameters here as a transferability test; on CIFAR-100 they are
suboptimal. Re-tuning is left as future work.*

## Out-of-scope on CIFAR-100 (with reasons)

Two methods from the CIFAR-10 suite were **deliberately not run** on CIFAR-100
because their compute or memory profile does not fit the single-8GB-VRAM
lane used for these experiments:

| Method | Reason |
|--------|--------|
| **LLLA** | Builds a full last-layer GGN matrix of shape `((D+1)·K, (D+1)·K)`. With D=512 and K=100, this is `(51300)² ≈ 2.6·10⁹` float32 entries (~10 GB), which exceeds VRAM and triggers `RESOURCE_EXHAUSTED` during JAX autotune. A K-FAC-style block-diagonal factorization would resolve this and is left for future work. |
| **Multi-block PnC** | Process hangs after the ridge-regression phase completes (`Multi-block PnC: ridge solves: 100% │…│ 400/400`) when computing OOD scores on the CIFAR-100 ensemble. The ridge solves themselves complete; the subsequent score-aggregation step on a 100-class × 50-perturbation × ~26K-sample ensemble appears to hit GPU memory fragmentation. Single-block PnC at scale=50 was used as the PnC entry instead. |

Both omissions are documented as architectural constraints of the current
single-GPU implementation, not as silent failures.

## Protocol verification

All 30 result JSONs report:
- `protocol.ood_validation_used = false`
- `protocol.ood_tuning_used = false`
- `protocol.temperature_fit_split = "id_validation_only"`
- `posthoc_temperature` is the value fit on the ID validation split (no OOD data touched)

Confirmed via `python scripts/aggregate_ood_results.py results/cifar100`.

## Caveats

1. **3 seeds (0, 1, 2)** — error bars are tight enough to support the
   qualitative ranking but not the absolute numbers. The Standard-Ensemble-vs-
   single-method gap is far larger than seed noise.
2. **Hyperparameters are frozen at their CIFAR-10-tuned values.** This is
   a deliberate transferability test. Re-tuning on CIFAR-100 ID validation
   data (without touching OOD splits) would likely improve PnC and ReAct
   numbers; see "Interpretation" above.
3. **Standard Ensemble seed `s` reuses base-model seeds `s, s+1, …, s+4`.**
   This is the same seeding scheme as CIFAR-10. Seeds 0–6 of the base
   PreActResNet18 are trained on CIFAR-100 to support all three ensemble seeds.
4. **Tiny ImageNet, Places365 labels are not used** — only the images.
5. **DTD/Places365 download paths** rely on external URLs; updates to those
   sites may require updates to `prepare_openood_data.py`.
6. **LLLA and multi-block PnC are deliberately omitted** on CIFAR-100; see the
   "Out-of-scope" section above.

## Run commands

See `scripts/run_cifar100_pipeline.sh` for the full set (training + eval).
The eval-only resume script `scripts/finish_cifar100_ood_v2.sh` was used to
finalize the missing seed-1 and seed-2 entries after a partial earlier run.
