# SLL-Backbone — Implementation Manifest

**What:** Subnetwork Linearized Laplace (full-covariance GGN over selected internal backbone
weights; linearized probit prediction) as a frozen-checkpoint post-hoc CIFAR-10 comparator.
**Where:** code in `experiments/sll_cifar/`, artifacts in `results/neurips_2026_rebuttal/cifar/sll/`.
**Env:** Python 3.12, JAX 0.9.1, Flax 0.12.3, RTX 5060 8 GB. Branch `neurips-2026-rebuttal`.
`XLA_FLAGS=--xla_gpu_enable_triton_gemm=false` (cuBLAS for tall matmuls). Filesystem GPU lock
(`flock` on `sll/.gpu.lock`) — waits rather than competing with an active CIFAR job.

## Code (reuses the validated SCOD infrastructure)
- `subnetwork.py` — `SubnetworkSpec`, `logits_from_subnetwork` (scatter S values into a copy of the
  frozen flat vector), per-example selected Jacobian `[C,S]` via `jacrev(argnums=0)` (never a `[C,P]`
  Jacobian), Fisher-weighted selected Jacobian for the GGN.
- `selection.py` — full-P diagonal GGN + raw-Jacobian energy from one raw-logit Jacobian pass;
  selection by predictive-variance contribution `R_j/(G_diag+λ0)` (faithful variance rule
  degenerates; see report). Backbone = flat idx ≥ 5130 (excludes fc).
- `posterior.py` — dense `G_S=F^TF` (per-example `F_n=dz̃/dw_sub`, not vmapped — vmapping jacrev
  OOMs), eigensystem, data-adaptive prior grid, eigensystem sampler.
- `predict.py` — probit rule (primary) + MC posterior predictive (secondary), temperature fit,
  ID metrics (acc/NLL/ECE/Brier).
- `run_sll.py` — modes `smoke|pilot|run`; GPU lock; selection→GGN→prior/temperature→eval.
- `run_eval.py` — ID + 6 OOD scoring, per-example Parquet + logit npz, macro Near/Far.
- `tables_report.py` — 3-seed aggregation and comparison tables.
- Reused from SCOD: `scod_cifar.parameter_layout` (model load, param split, flat layout),
  `scod_cifar.protocol` (splits, 1024 subset, temperature, benchmark, hashes),
  `scod_cifar.categorical_fisher` (Fisher factor + gate 2).

## Commands
```
python -m experiments.sll_cifar.run_sll --mode smoke
python -m experiments.sll_cifar.run_sll --mode pilot --M 50           # S in {512,1024,2048}
python -m experiments.sll_cifar.run_sll --mode run --S 2048 --M 50 --seeds 0 1 2
python -m experiments.sll_cifar.tables_report
```

## Config (frozen for the reported run)
S=2048 (chosen by ID-val NLL), M=50, selection=predictive_variance_contribution, prediction=probit
predictive entropy, sample seed = 100000+checkpoint_seed. Prior λ_S and temperature: ID-val only.

## Completion status
DONE: 3 seeds × (ID + 6 OOD); aggregate reproduces from per-example (exact); no NaN/inf; checkpoint
hashes unchanged and == SCOD manifest; gates 1–7, 9 PASS; MANIFEST + validation + tables + report.
NOT written: `SLL_CIFAR_COMPLETE` — gate 8 (last-layer sanity vs existing LLLA under matched
conventions) is NOT_RUN (deferred: LLLA uses a different last-layer approximation). Optional `SLL-All`
variant not run. The primary SLL-Backbone 3-seed result is complete and validated.
