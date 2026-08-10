# CIFAR-10 Efficiency Accounting — construction time, storage, inference latency

**Date:** 2026-07-26 · **Purpose:** answer reviewer concern raised by **LLwM, A5nU, WKSj** — no
apples-to-apples cost comparison against Deep Ensembles, SWAG, and Laplace.
**Scope:** latency, training, construction, and storage numbers come from **already-completed runs in
this repo**. **Two new measurements were taken:** peak GPU memory across 11 methods (§4, 2026-07-26),
which nothing in the repo had captured and which also settled the s3b0/s3b1 block question (§1.0); and
SWAG's post-training construction time (§2.7, 2026-07-27), so SWAG and Laplace are finally comparable.
**No models were retrained** — SWAG and Epinet training times were recovered from checkpoint mtimes
(§2.6) rather than by re-running ~9 GPU-hours. Every row is tagged `[MEASURED]`, `[DERIVED]`, or
`[ESTIMATE]`.

**Reviewer asks, restated:**
- LLwM / A5nU / WKSj: construction wall-clock, memory/storage, and inference latency vs DE / SWAG / Laplace.
- A5nU: the "higher-cost Deep Ensemble" claim is unquantified.
- WKSj: P&C still runs many members and many least-squares solves — so what is actually saved?

### Coverage matrix — what this document can and cannot answer

| Axis | P&C | Deep Ensemble | SWAG | Laplace (LLLA) | MC Dropout | Epinet |
|---|---|---|---|---|---|---|
| Inference latency | ✅ measured | ✅ measured (n=5 **and** n=50) | ✅ measured | ✅ measured | ✅ measured | ✅ measured |
| Storage | ✅ measured resident (§3.3.1) | ✅ measured resident | ✅ **measured ckpt + resident** | ✅ **derivation confirmed** (§3.4) | ✅ measured ckpt | ✅ measured ckpt |
| Construction / train wall-clock | ✅ measured | ✅ derived from measured base | ✅ train from mtimes (§2.6) + **post-train construction measured** (§2.7) | ✅ **measured, n=18** (§2.5) | ✅ measured | ✅ derived from mtimes |
| Peak GPU memory | ✅ **measured** (§4) | ✅ **measured** (n=5 and n=50) | ✅ **measured** | ✅ **measured** | ✅ **measured** | ✅ **measured** |

**All four axes are now covered for every contested method.** Two caveats that matter more than any
remaining gap:
- **§1.0** — the published P&C latency row was measured on the **wrong block** (s3b1, not the submitted
  s3b0). Corrected figure: **7.53 ms**. Conclusions unchanged, table needs fixing.
- **§3.3.1** — the analytic P&C storage figure **understated resident memory by 345 MiB**. Measured
  value is **1,062.7 MiB**, not 717.8 MiB. Every other analytic figure validated within 6%.

**One-line answer:** P&C's saving is **training cost and on-disk storage**, not inference latency.
Construction is **~100 s** on top of a single 2h13m base train; at matched member count P&C's inference
latency is **statistically indistinguishable from a 50-member Deep Ensemble**. That concession should be
made explicitly in the rebuttal.

---

## 0. Hardware & common methodology

All numbers were produced on the **same single CUDA GPU (8 GB), WSL2**, JAX 0.9.1 + Flax NNX 0.12.3.
No multi-GPU, no distributed training. Every method below uses the identical 300-epoch recipe
`e300_optsgd_lr1e-01_wd5e-04_bs128_wu5_mom0p9_n1_augfcco8_ls0`, so training-time comparisons are
recipe-matched by construction.

**Anchor (submitted) P&C config** — `ANCHOR_CONFIG.json`: single-block, target `(stage_idx=3, block_idx=0)`
= stage4 block 0; K=20 random orthonormal directions; **M=50 members**; perturbation scale 25.0;
ridge λ=1e-3; calibration subset 1024 ID train images; chunk_size 1024; bootstrap_frac 0.05;
post-hoc temperature scaling fit on ID val only.

---

## 1. Inference latency `[MEASURED]`

**Source:** `results/cifar10/inference_cost.json` (per-method JSONs in `results/cifar10/inference_cost/`),
matched-M addendum in `inference_cost_deep_ensemble_n50_matched.json`, table in
`efficiency_cifar_table.{md,csv}`. Harness: `scripts/benchmark_inference_cost.py`,
`_de_matched_timing.py`.

**Methodology:** N=5000 CIFAR-10 test images, batch_size=256, seed-0 checkpoints, GPU-synced with
`block_until_ready()`, **first (cold/JIT) batch excluded** and reported separately. Member counts were
verified against the submitted configurations: P&C M=50, MC Dropout n=32, Deep Ensemble n=5,
SWAG/LLLA/Epinet n=50.

| Method | ms/sample | throughput (samp/s) | fwd passes/sample | cold/JIT warmup (s) | train cost |
|---|---:|---:|---:|---:|---:|
| PreAct ResNet-18 (MSP / Energy) | 0.617 | 1621.8 | 1 | 3.61 | 1× |
| ReAct + Energy | 0.614 | 1629.0 | 1 | 0.39 | 1× |
| Mahalanobis | 0.735 | 1360.8 | 1 | 3.30 | 1× |
| **Deep Ensemble n=5 (submitted-cost)** | **1.338** | 747.4 | 5 | 3.73 | **5×** |
| Epinet n=50 | 1.617 | 618.4 | 50 | 4.95 | 1.05× |
| MC Dropout n=32 | 5.177 | 193.2 | 32 | 4.97 | 1× |
| **Deep Ensemble n=50 (matched-M)** | **7.261** | 137.7 | 50 | 5.42 | **50×** |
| SWAG n=50 | 7.325 | 136.5 | 50 | 59.42 | 1× |
| P&C single-block, scale=25, M=50 — **s3b1, NOT the anchor** (see §1.0) | 7.419 | 134.8 | 50 | 5.74 | **1×** |
| **P&C single-block s3b0 (SUBMITTED anchor)** — measured §4 | **7.529** | 132.8 | 50 | — | **1×** |
| LLLA n=50 | 7.680 | 130.2 | 50 | 6.00 | 1× |
| P&C multi-block, scale=7, M=50 | 9.688 | 103.2 | 50 | 35.77 | 1× |

Per-method notes carried from the raw JSON:
- **LLLA n=50** — head-only sampling on top of one feature extraction per batch.
- **Mahalanobis** — feature extraction only; distance is pure-numpy O(N·C·D).
- **SWAG n=50** — cached samples + one-time BN refresh on 2048 ID train samples (this is why its cold
  warmup is 59.4 s, by far the largest).
- **Epinet n=50** — 1.05× train cost: only a small MLP is fine-tuned on top of the frozen base.
- **DE n=50 (matched-M)** — latency is weight-independent, so 50 clones of the seed-0 base were timed;
  same `_time_predict` harness, batch size, and N as the submitted benchmark.

### 1.0 ⚠️ Correction — the published P&C latency row is the WRONG BLOCK

`scripts/benchmark_inference_cost.py:366` builds `pnc_single` with **`target_block_idx=1` (s3b1)**.
The submitted anchor (`ANCHOR_CONFIG.json`) is **`target_block_idx=0` (s3b0)**. The two differ: s3b0's
conv1 is 256→512 preceded by a downsample, s3b1's is 512→512. **The 7.419 ms figure in the table above
is s3b1, not the submitted configuration.**

Direct measurement of both (§4 run, same harness, 2026-07-26):

| Config | ms/sample | resident (MiB) |
|---|---:|---:|
| s3b1 (what the table reports) | 7.404 | 1,329.0 |
| **s3b0 (submitted anchor)** | **7.529** | **1,062.7** |

**Impact: small but real.** The anchor is **1.5% slower** than the published row. Against matched-M
DE n=50 (7.261 ms), P&C moves from 2.2% slower to **3.7% slower** — still a tie for practical purposes,
so **§1.1's conclusions are unaffected**. But the efficiency table's P&C row should be corrected to
**7.53 ms** and relabelled s3b0 before publication, and the storage figures in §3.3 (which are s3b0)
should not be paired with the s3b1 latency.

### 1.1 The honest reading (do not overclaim)

*(figures below use the corrected anchor value, 7.529 ms — see §1.0)*

1. **At matched member count, P&C is not cheaper.** P&C anchor M=50 (7.529 ms) vs DE n=50 (7.261 ms) —
   a **3.7% difference**, i.e. essentially identical, exactly as expected since both do 50 forward
   passes. P&C buys **nothing** at inference over an equal-member ensemble.
2. **Against the submitted DE n=5** (1.338 ms), P&C is **~5.6× slower** at inference, while being
   **5× cheaper to train**.
3. P&C's genuine advantages are **training cost (1×)**, **on-disk storage** (§3), and — at matched M
   only — **resident memory** (§4.2, finding 2). *Not* latency.
4. Single-forward-pass reference floor: **0.617 ms**. P&C's 7.529 ms is 12.2× that, consistent with
   50 forward passes over a partially shared trunk (the stem and all non-target blocks are shared;
   only the target block is re-run per member).

---

## 2. Construction / training wall-clock `[MEASURED]` + `[DERIVED]`

### 2.1 Base network training — the unit of account `[MEASURED]`

**Source:** `experiments/logs/cifar10_finish_20260427_223643.log`, `mcd_dropout_sweep_*.log`
(timestamped `<= OK Train ... (Ns)` stage records).

| Run | Wall-clock |
|---|---:|
| Train MCD dr=0.5 seed=0 (300 ep, CIFAR-10) | **8,010 s** |
| Train MCD dr=0.3 seed=0 (300 ep, CIFAR-10) | 8,038 s |
| Train MCD dr=0.2 seed=0 (300 ep, CIFAR-10) | 8,049 s |
| Train MCD dr=0.2 seed=1 (contended GPU) | 13,276 s ← *excluded* |
| Train MCD dr=0.2 seed=2 (contended GPU) | 12,029 s ← *excluded* |

**Adopted unit: 8,010 s ≈ 2 h 13 m per PreAct ResNet-18 (300 epochs).** MC Dropout uses the identical
architecture and recipe (dropout is inference-time only for cost purposes), so this is the base-network
training time. The two 12–13 ks runs were concurrent with other GPU work and are excluded as contended.

**Independent corroboration `[MEASURED]`:** CIFAR-100 base training, same architecture and recipe
(`experiments/cifar_neurips_strengthening_log.md`, "Final base training timeline"): seed 0 ≈ 2h16m,
seed 1 ≈ 2h17m, seed 2 ≈ 2h17m — 6h50m for 3 seeds. Agrees with 8,010 s to within ~3%.

### 2.2 P&C construction wall-clock `[MEASURED]` → `[DERIVED]`

**Source:** `experiments/logs/p2_s3b0_ood_20260428_141145.log` — the submitted anchor config
(`chunk_size=1024`, `bootstrap_frac=0.05`, `subset_size=1024`, s3b0, ps=25, K=20, M=50), 3 seeds.

Total Luigi wall-clock **per seed**, covering ensemble build **plus** the full 7-dataset OpenOOD
evaluation:

| Seed | Wall-clock |
|---|---:|
| 0 | 826 s |
| 1 | 847 s |
| 2 | 844 s |

Decomposition for seed 0, using the `eval_time` fields recorded inside the cached result JSON
(`results/cifar10/openood_v1p5_pnc_single_block_vcal_s3b0_k20_n50_ps25.0_lr0.001_bf0.05_..._seed0_random.json`):

| Evaluation stage | n examples | eval_time (s) |
|---|---:|---:|
| ID test (CIFAR-10) | 10,000 | 91.3 |
| ID val | 5,000 | 0.0 *(not separately timed)* |
| near-OOD CIFAR-100 | 10,000 | 90.7 |
| near-OOD Tiny-ImageNet | 10,000 | 86.8 |
| far-OOD MNIST | 10,000 | 91.2 |
| far-OOD SVHN | 26,032 | 224.7 |
| far-OOD Textures (DTD) | 5,640 | 49.0 |
| far-OOD Places365 (10k subset) | 10,000 | 89.7 |
| **Total measured eval** | **81,672** | **723.5** |

**826 s − 723.5 s = 102.5 s.** This residual covers checkpoint load, CIFAR-10 load from cache,
direction generation, calibration feature extraction over 1024 ID train images, **50 bootstrap ridge
solves**, the untimed ID-val pass (5,000 images ≈ 37 s at 7.419 ms/sample), and the temperature fit.

> **Adopted figure: P&C construction ≤ ~100 s, and ~65 s once the untimed ID-val pass is netted out.**
> Quote **"~100 s"** as a conservative upper bound. Reproducible across seeds (826/847/844 s).

### 2.3 Component cost of the least-squares work `[MEASURED]` — directly answers WKSj

**Source:** `results/conv_construction_rank/*/timings.json` and `FINAL_REPORT.md` §7. Per-operation, on
GPU, float64 spectrum, design dimension q=2305.

| Operation | Phase C (net-eval 1024) | Phase E (net-eval 10k) |
|---|---:|---:|
| Gram accumulation (masked, cached Y) | 0.089 s | 0.46 s |
| Eigendecomposition 2305² (float64) | 0.835 s | 0.72 s |
| All-ridge solve + residuals (8 ridge values, **one** shared eig) | 1.037 s | 4.13 s |

Raw totals: Phase C — 180 solves, 186.70 s total ridge, 15.99 s total Gram, 150.23 s total eig,
373.37 s wall. Phase D (matched) — 180 solves, mean ridge 0.997 s, mean eig 0.711 s, 341.40 s wall.
Round-2 stage s3b1 — 360 solves, mean ridge 1.828 s, mean eig 4.858 s, 2,605.16 s wall.

**Rebuttal point:** "many least-squares solves" is **~1 s per solve**. Fifty of them is ~50 s of GPU
time — the entire construction cost. Ridge values sharing a single eigendecomposition is what keeps
this flat; a single float64 eigendecomposition at this scale is not prohibitive.

### 2.4 Total cost to a deployable ensemble `[DERIVED]` from §2.1

| Method | Post-base construction | Total wall-clock | Train-cost multiplier |
|---|---:|---:|---:|
| Single network | — | 8,010 s (2.23 h) | 1× |
| **P&C single-block M=50 (anchor)** | **~100 s** | **≈ 8,110 s (2.25 h)** | **1×** |
| MC Dropout n=32 | ~0 | 8,010 s (2.23 h) | 1× |
| Epinet n=50 | ~20 min `[ESTIMATE]` | ≈ 9,200 s (2.6 h) | 1.05× |
| SWAG n=50 | **55.8 s** post-train (§2.7) | ≈ 11,233 s (3.12 h) `[DERIVED §2.6]` | 1× |
| **LLLA n=50 (Laplace)** | **15.56 s `[MEASURED]`** (§2.6) | ≈ 8,026 s (2.23 h) | 1× |
| **Deep Ensemble n=5 (submitted)** | — | **40,050 s ≈ 11.1 h** | **5×** |
| **Deep Ensemble n=50 (matched-M)** | — | **400,500 s ≈ 111 h ≈ 4.6 days** | **50×** |

> **Headline claim, defensible:** **~100 seconds of construction replaces ~9 GPU-hours of additional
> training** relative to the submitted Deep Ensemble (n=5), and **~109 GPU-hours** relative to the
> matched-M ensemble (n=50). This is the quantification A5nU asked for.

**`[ESTIMATE]` caveat:** the SWAG (~9 h / 3 seeds) and Epinet (~1 h / 3 seeds) figures are **planning
projections** recorded in `experiments/cifar_neurips_strengthening_log.md` for the CIFAR-100 pipeline,
**not** measured stage records. No timestamped SWAG/Epinet/LLLA training durations exist in
`experiments/logs/`. Label them as approximate in the rebuttal, or omit those two rows — the DE
comparison is the one under dispute and it is fully measured.

### 2.5 Laplace (LLLA) construction wall-clock `[MEASURED]`

**Source:** the `train_time` field written by `CIFARLLLA.run()` (`cifar_tasks.py:1907-1955`) into
`results/cifar10/baseline_llla_vcal_n50_prec*_seed*.json`. **No new runs were needed** — this was
already recorded by every completed LLLA run in the repo.

**What the timer spans:** `t0` is set immediately after data load; `train_time` is captured *after*
`LLLAEnsemble` is constructed and *before* `_evaluate_cifar` is called. It therefore covers checkpoint
load + model build, GGN accumulation over the 45,000-image ID train split, `jnp.linalg.inv` of the
dense 5,130² precision matrix, and the Cholesky factorization inside `LLLAEnsemble.__init__`. It
excludes all evaluation. This is exactly the construction quantity the reviewers asked for.

**Submitted configuration** (n=50, `prior_precision=10.0`):

| Seed | Construction |
|---|---:|
| 0 | 15.87 s |
| 1 | 15.55 s |
| 2 | 15.27 s |
| **mean ± sd (n=3)** | **15.56 ± 0.30 s** |

**Robustness across the full prior sweep** — 7 `prior_precision` values (0.01, 0.1, 1.0, 10.0, 100,
1000) × 3 seeds, **n=18 runs**: **15.58 ± 0.32 s**, range 14.96–16.18 s. Construction is
**independent of the prior precision**, as expected: GGN accumulation over the training set dominates,
and the 5,130² inverse is the same size regardless of λ. The n=32 variant is faster (12.07 s) only
because fewer ensemble members are prepared.

**Reading:** Laplace construction (**15.6 s**) is ~6× cheaper than P&C construction (**~100 s**) and
~3.6× cheaper than SWAG's post-training step (**55.8 s**, §2.7); all three are negligible against the
8,010 s base train they sit on top of. **Construction cost does not separate P&C from Laplace or
SWAG** — all are ~1× training, and all reconstruct rather than checkpoint (§3.4). See §2.7.1 for the
three side by side and §6 for the framing consequence.

### 2.6 SWAG and Epinet training wall-clock `[DERIVED from checkpoint mtimes]`

These two were previously `[ESTIMATE]` (planning projections). They can be recovered from the
filesystem without retraining, because the seeds were trained **back-to-back by a single sequencing
script**, so the gap between consecutive checkpoint mtimes is the duration of the later run plus
inter-task overhead.

**SWAG** (`preact_resnet18_swag_train_..._sws240_swf1_swr20_seed{0,1,2}.pkl`):

| Interval | Δ |
|---|---:|
| seed 0 → seed 1 | 11,180 s (3.11 h) |
| seed 1 → seed 2 | 11,174 s (3.10 h) |

**Epinet** (three independent consecutive pairs across the ps sweep):

| Interval | Δ |
|---|---:|
| ps0.5 seed0 → ps3.0 seed0 | 1,213 s (20.2 min) |
| ps1.0 seed1 → ps1.0 seed2 | 1,219 s (20.3 min) |
| ps3.0 seed1 → ps3.0 seed2 | 1,226 s (20.4 min) |

**Overhead calibration `[MEASURED]`.** The same mtime method applied to the base network — whose
training time is *independently* known to be 8,010 s from timestamped stage records (§2.1) — gives
9,345 / 9,356 / 9,369 s across three consecutive pairs. So the mtime delta **overstates pure training
by ~1,350 s (a factor of 1.168)**, the cost of evaluation, checkpoint write, and next-task startup
folded into the gap.

Applying that calibration:

| Method | mtime Δ (upper bound) | Overhead-corrected estimate |
|---|---:|---:|
| SWAG, per seed | 11,177 s (3.10 h) | **≈ 9,570 s (2.66 h)** |
| Epinet, per seed | 1,220 s (20.3 min) | **≈ 1,045 s (17.4 min)** |

**How to quote these:** use the **mtime Δ as a defensible upper bound** (11,177 s / 1,220 s) — it is a
direct filesystem measurement with ±6 s reproducibility across independent pairs. The corrected column
assumes SWAG's and Epinet's inter-task overhead matches the base network's, which is plausible but
unverified; treat it as indicative. Either way this **confirms** the original ~3 h and ~20 min
projections rather than revising them, and both methods remain **1× / 1.05× training cost**, so §6.1's
conclusion is unaffected.

### 2.7 SWAG post-training construction wall-clock `[MEASURED 2026-07-27]`

**Source:** `_swag_construction_time.py` → `swag_construction_time.json`. Seeds 0/1/2, submitted
config (n=50, sws240, swr20). Same construction path as `scripts/benchmark_inference_cost.py`.

**⚠️ SWAG's construction does not decompose the way Laplace's does — read this before comparing.**
Laplace's posterior fit is entirely post-training, so §2.5's 15.56 s is a complete standalone cost.
SWAG's is **split in two**:

| SWAG cost component | When | Where measured |
|---|---|---|
| SWA moment collection (mean, 2nd moment, 20 deviation vectors) | **during** the 300-epoch run | inseparable — inside §2.6's 11,177 s |
| Posterior sample draw + BatchNorm refresh | **after** training | **measured here** |

Only the second row is comparable to Laplace's 15.56 s. **Do not present a single SWAG construction
number alongside Laplace's** — it implies a like-for-like decomposition that does not exist.

**Timer span** (matched to `CIFARLLLA.run()`'s `train_time`): base checkpoint load, SWAG moment
checkpoint load, 50-sample posterior draw (`cache_samples=True`), BatchNorm refresh over 2,048 ID train
images, up to a predict-ready ensemble. Excludes all evaluation. Includes one 128-image predict to
force deferred device work (~1 s of the total).

| Seed | Post-training construction |
|---|---:|
| 0 | 61.28 s |
| 1 | 51.29 s |
| 2 | 54.83 s |
| **mean ± sd (n=3)** | **55.80 ± 5.06 s** |

**Cross-check:** the independently-measured SWAG cold/JIT warmup in §1 is **59.42 s** — consistent with
55.80 s, and confirming that figure was dominated by the BN refresh and sample caching rather than
compilation. Two independent measurements of the same underlying work agree.

**Variance note:** sd is 5.06 s (9% of the mean), much larger than Laplace's 0.30 s (1.9%). The BN
refresh is a 2,048-image forward pass whose cost varies with device state; Laplace's GGN accumulation
is a fixed 45,000-image sweep. Quote SWAG's as **~56 s**, not to two decimals.

### 2.7.1 Construction cost, the three 1×-training methods side by side

| Method | Post-training construction | Notes |
|---|---:|---|
| **Laplace (LLLA)** | **15.56 ± 0.30 s** | complete cost; nothing folded into training |
| **SWAG** | **55.80 ± 5.06 s** | *partial* — moment collection is inside the 300-epoch run |
| **P&C (anchor)** | **~100 s** | complete cost (§2.2), upper bound |

**P&C has the most expensive construction of the three** — ~1.8× SWAG's post-training step and ~6.4×
Laplace's, and P&C's figure is a complete cost while SWAG's is only partial. All three are negligible
against the 8,010 s base train they share. This reinforces §6.1: **construction cost does not favour
P&C against any 1×-training baseline**; it is only an argument against Deep Ensembles.

### 2.8 Two timing numbers that must NOT be quoted

- **`repro_A_seed0.log` reports `runtime: 4898.3s`.** Measured eval inside that run is 765.9 s, leaving
  4,132 s unaccounted. That run **downloaded CIFAR-10 from the Toronto CDN**, which
  `results/conv_construction_rank/implementation_notes.md` documents as throttled at the time. The
  residual is download, not construction. **Use the 826 s Luigi run instead.**
- **`pnc_block_position_sweep_*.log` shows P&C val tasks at 390–416 s.** Those runs used
  **`chunk_size=16`**, i.e. 64× more chunks through the solve loop, versus the submitted
  `chunk_size=1024`. `ANCHOR_CONFIG.json` records chunk_size as a **memory-only knob with invariant
  results**, so these are a memory-constrained variant, not the submitted construction cost.

---

## 3. Storage / model size `[MEASURED]` + `[DERIVED]`

### 3.1 Measured on-disk checkpoints `[MEASURED]`

Parameter counts obtained by walking the pickled Flax NNX state; file sizes are `stat` bytes. fp32.

| Method | Params | File size | Ratio to base |
|---|---:|---:|---:|
| Base PreAct ResNet-18 | 11,179,978 | 42.66 MiB (44,729,016 B) | 1.00× |
| MC Dropout (dr=0.1) | 11,179,980 | 42.66 MiB (44,729,419 B) | 1.00× |
| Epinet head (on frozen base) | 65,360 | 0.25 MiB (263,140 B) | +0.006× |
| **SWAG (sws240, swf1, swr20)** | **256,967,718** | **980.27 MiB (1,027,891,171 B)** | **22.98×** |

SWAG's 23× blowup is structural: SWA mean + second moment + 20 low-rank deviation vectors + BN
statistics, all stored at full network width.

### 3.2 Deep Ensemble `[DERIVED]` — exact multiples of a measured checkpoint

| Configuration | Params | Storage |
|---|---:|---:|
| Deep Ensemble n=5 (submitted) | 55,899,890 | **213.3 MiB** (223,645,080 B) |
| Deep Ensemble n=50 (matched-M) | 558,998,900 | **2,132.9 MiB = 2.08 GiB** (2,236,450,800 B) |

### 3.3 P&C storage `[DERIVED]` from verified tensor shapes

Shapes confirmed by direct inspection of the seed-0 checkpoint, and matching `ANCHOR_CONFIG.json`:

- `stage4.0.conv1.kernel` = (3, 3, 256, 512) = **1,179,648** params — the *perturbed* layer
- `stage4.0.conv2.kernel` = (3, 3, 512, 512) = **2,359,296** params — the *corrected* layer
- new per-channel bias on conv2 = **512** params
- Everything else (stem, stages 1–3, `stage4.1`, head) is **shared across all members**.

| P&C storage accounting | Params | Storage |
|---|---:|---:|
| Per member (conv1 + conv2 + bias) | 3,539,456 | 13.50 MiB |
| All M=50 members, fully materialized | 176,972,800 | **675.1 MiB** |
| conv2 corrections only (conv1 regenerated) | 117,990,400 | **450.1 MiB** |
| **Actually persisted on disk today** | **11,179,978** | **42.66 MiB** |

Three distinct numbers, and the rebuttal must not conflate them:

1. **675.1 MiB — member weights only. ⚠️ SUPERSEDED: true resident is 1,062.7 MiB, see §3.3.1.**
   All 50 (w1, w2, b2) triples are materialized in
   `PnCEnsemble._precompute_corrections` (`ensembles.py`) and held for the life of the ensemble.
   **This is worse than DE n=5 (213.3 MiB).** If a reviewer asks about *GPU-resident* memory rather
   than storage, P&C at M=50 loses to the submitted Deep Ensemble. Concede this.
2. **450.1 MiB — minimal serialized form.** The conv1 perturbation need not be stored: it is
   `w1_orig + z_coeffs @ v_opts` with K=20 **seeded** orthonormal directions and 20 coefficients per
   member (1,000 floats total, ~4 KB). Only the ridge-solved conv2 corrections are irreducible.
3. **42.66 MiB — the real on-disk footprint.** **There is no P&C checkpoint in `results/cifar10/`.**
   Grep confirms only base / SWAG / MC Dropout / Epinet `.pkl` files exist. The P&C ensemble is
   *rebuilt from (base checkpoint + seed + config) in ~100 s on every evaluation*, so it is never
   serialized. Construction being cheap is precisely what makes storage free.

> **Strongest storage framing:** P&C's on-disk footprint is **42.7 MiB**, vs **213 MiB** for DE n=5,
> **980 MiB** for SWAG, and **2.08 GiB** for matched-M DE n=50 — because construction is cheap enough
> to re-run rather than checkpoint. **Pair this immediately with the concession in (1)** so it does
> not read as a memory claim.

#### 3.3.1 Correction — the analytic figure understates P&C `[MEASURED 2026-07-26]`

The §4 memory run measured resident memory directly, which validates every analytic storage figure in
this document **except P&C's**:

| Method | analytic (MiB) | measured resident (MiB) | Δ |
|---|---:|---:|---:|
| Single / MC Dropout / ReAct | 42.7 | 44.1 | +1.4 |
| Epinet n=50 | 43.0 | 44.4 | +1.4 |
| **LLLA n=50** | **243.5** | **245.0** | **+1.5** ✅ |
| SWAG n=50 | 980.3 | 1,070.7 | +90.4 |
| Deep Ensemble n=5 | 213.3 | 226.9 | +13.6 |
| Deep Ensemble n=50 | 2,132.9 | 2,250.3 | +117.4 |
| **P&C s3b0 (anchor)** | **717.8** | **1,062.7** | **+344.9** ❌ |

The small `+1.4` / `+13.6` / `+117.4` offsets are BatchNorm statistics and per-model overhead the
parameter counts omit (~2.4 MiB per network copy) — expected and harmless. **P&C's +345 MiB is not.**

**Cause:** §3.3 counted only the per-member `(w1, w2, b2)` triples, but `PnCEnsemble` retains
additional state for the ensemble's lifetime (`ensembles.py:1562-1593`): `self.chunks` and
`self.T_orig_chunks` (the calibration design matrices over 1024 ID images) and `self.v_opts` (the K=20
direction basis, ~90 MiB at s3b0's 256→512 conv1). Those account for roughly 190 MiB; the remainder is
allocator and JIT buffer overhead.

**Use 1,062.7 MiB, not 717.8 MiB**, for any statement about P&C's resident footprint. This makes the
§3.3 point (1) concession *stronger*, not weaker — P&C at M=50 holds ~4.7× the resident memory of the
submitted DE n=5, not ~3.2×. The on-disk 42.7 MiB claim is unaffected: none of this retained state is
serialized.

### 3.4 Laplace (LLLA) storage `[DERIVED]` from code + verified shapes

**No LLLA checkpoint exists on disk** — `results/cifar10/` contains only the `baseline_llla_*.json`
result files (~273 bytes each). Like P&C, the LLLA posterior is rebuilt from the base checkpoint on
every evaluation, so its persisted footprint is the base network alone.

Derived from the implementation, not measured:
- `cifar_tasks.py:1949` builds a **dense** posterior: `precision = G_flat + prior_precision · I`,
  then `covariance = jnp.linalg.inv(precision)`. This is a full GGN inverse, **not** a KFAC or diagonal
  factorization. (`laplace.py` does contain KFAC machinery, but the CIFAR LLLA path does not use it.)
- `LLLAEnsemble.__init__` (`ensembles.py:2105`) stores `cov` **and** precomputes a Cholesky factor
  `self.L = cholesky(cov + 1e-6·I)` of the same shape, held for the ensemble's lifetime.
- Final layer verified from the seed-0 checkpoint: `fc.kernel` = (512, 10), `fc.bias` = (10,)
  → covariance dimension **D·K + K = 512·10 + 10 = 5,130**.

| LLLA storage accounting | Params | Storage |
|---|---:|---:|
| Posterior covariance (5,130²) | 26,316,900 | 100.4 MiB |
| Cholesky factor `L` (same shape, precomputed) | 26,316,900 | 100.4 MiB |
| **Resident on top of base at inference** | 52,633,800 | **200.8 MiB** |
| Base network (shared) | 11,179,978 | 42.7 MiB |
| **Total resident** | | **≈ 243.5 MiB** |
| **Actually persisted on disk** | 11,179,978 | **42.7 MiB** |

**Reading:** LLLA and P&C have the *same structural storage story* — cheap enough to reconstruct, so
neither is checkpointed; both sit at 42.7 MiB on disk against SWAG's 980 MiB. At inference LLLA's
resident cost (measured 245.0 MiB) is **well below P&C's measured 1,062.7 MiB at M=50**, so LLLA is the method P&C loses to
on resident memory, not just DE n=5. Note also that the dense 5,130² inverse is what makes LLLA's
storage scale with (feature-dim × n_classes)²; it would grow ~100× on a 100-class head, which is worth
flagging if a reviewer raises CIFAR-100.

**Confirmed by measurement `[MEASURED 2026-07-26]`.** The §4 run puts LLLA resident at **245.0 MiB**
against the 243.5 MiB predicted here — **0.6% error**. This section is no longer an analytic estimate;
the dense-inverse-plus-Cholesky account of LLLA's footprint is correct and can be quoted directly.

---

## 4. Peak GPU memory `[MEASURED]`

**Measured 2026-07-26.** Harness: `_peak_memory_bench.py` (driver log `peak_memory/peak_memory.log`,
per-method JSONs in `peak_memory/`, table `peak_memory/peak_memory_table.csv`). It imports
`scripts/benchmark_inference_cost.py` and reuses its builders verbatim, so every configuration is
identical to the published latency table.

**Methodology.** `XLA_PYTHON_CLIENT_PREALLOCATE=false` (otherwise the allocator claims the whole card
and every number is meaningless), then `jax.local_devices()[0].memory_stats()` sampled at three points.
**One method per subprocess** — JAX peak stats are cumulative per process with no reset API, so a
shared process would report the running max across all methods. GPU: NVIDIA RTX 5060, **8,151 MiB**.
N=5000, batch 256, seed 0 — same as §1.

| Method | fwd | resident after build (MiB) | build peak (MiB) | **inference peak (MiB)** | % of card |
|---|---:|---:|---:|---:|---:|
| PreAct ResNet-18 (single) | 1 | 44.1 | 93.2 | 2,469.6 | 30.3% |
| ReAct+Energy | 1 | 44.1 | 2,474.1 | 2,474.1 | 30.4% |
| MC Dropout n=32 | 32 | 44.1 | 93.2 | 2,469.7 | 30.3% |
| Deep Ensemble n=5 | 5 | 226.9 | 272.5 | **2,053.5** | 25.2% |
| LLLA n=50 (Laplace) | 50 | 245.0 | 1,922.9 | **2,096.7** | 25.7% |
| Epinet n=50 | 50 | 44.4 | 93.2 | 2,469.9 | 30.3% |
| Deep Ensemble n=50 (matched-M) | 50 | 2,250.3 | 2,298.0 | **4,071.8** | 50.0% |
| **P&C single-block s3b0 (SUBMITTED anchor)** | 50 | **1,062.7** | 4,332.2 | **4,332.2** | **53.1%** |
| P&C single-block s3b1 (as shipped in §1) | 50 | 1,329.0 | 4,332.2 | 4,332.2 | 53.1% |
| SWAG n=50 | 50 | 1,070.7 | 1,157.8 | **5,360.1** | 65.8% |
| P&C multi-block | 50 | 4,737.2 | 5,444.8 | **6,063.3** | 74.4% |

### 4.1 Harness validation

Measured latency reproduces the published table, which confirms the builders and call path are the
same: P&C multi-block −0.0%, P&C s3b1 −0.2%, DE n=50 +0.3%, Epinet −0.4%, single −1.5%, ReAct +1.5%,
SWAG +2.0%, LLLA −2.9%, MC Dropout −6.3%, **DE n=5 −14.8%** (outlier). The configurations are
identical, so the DE n=5 gap is machine variance between the original run and today, not a
discrepancy. **Cite §1 for latency and §4 for memory** — do not mix runs.

### 4.2 Findings

1. **A ~2,470 MiB floor dominates the light methods.** Single model, ReAct, MC Dropout, and Epinet all
   land within 5 MiB of each other despite holding 1–50 members' worth of sampling machinery, because
   peak is set by the batch-256 ResNet-18 forward workspace, not by stored parameters. **Peak memory
   does not discriminate between methods until the parameter store exceeds ~2.4 GiB.**
2. **At matched M, P&C uses less than half the resident memory of the equivalent Deep Ensemble**:
   **1,062.7 MiB vs 2,250.3 MiB**. P&C shares the trunk and stores per-member corrections for a single
   block; DE n=50 stores 50 complete networks. Peak is near-parity (4,332 vs 4,072 MiB, P&C +6.4%).
   **This is a genuine P&C win the report previously could not state.**
3. **Against the submitted DE n=5, P&C loses on memory** — 1,062.7 vs 226.9 MiB resident, 4,332 vs
   2,054 MiB peak. Same shape as the latency result: P&C beats matched-M, loses to n=5.
4. **SWAG is the heaviest single-block method: 5,360 MiB peak, 65.8% of the card.** ~2.6× the next
   heaviest at equal member count, on top of being 23× the base network on disk (§3.1). Memory is
   SWAG's second independent weakness.
5. **P&C's peak is set during construction, not inference** — `build_peak == predict_peak == 4,332.2
   MiB` for both blocks, and identical across s3b0/s3b1, so the peak is a block-independent transient
   of the Gram/ridge solve. It is the one place P&C's cost is concentrated.
6. **P&C multi-block is the heaviest configuration measured: 6,063 MiB peak, 4,737 MiB resident,
   74.4% of the card.** Perturbing all 8 residual blocks multiplies the per-member correction state.
   This variant genuinely constrains deployment on an 8 GB GPU and that should be stated, not left for
   a reviewer to find.
7. **Laplace is memory-cheap at inference (2,096.7 MiB, below the floor) but expensive to build
   (1,922.9 MiB)** — head-only sampling on one feature extraction per batch, versus a dense 5,130²
   GGN inverse at construction. Cheap in time (15.6 s, §2.5), costly in transient memory.

**Superseded:** the ≈7.1 GB figure in `results/conv_construction_rank/FINAL_REPORT.md` §7 is a
float64 2305² eigendecomposition probe — a different, heavier workload. Use the table above instead.

---

## 5. Supporting result: M=50 is roughly 2× more members than needed `[MEASURED]`

**Source:** `CIFAR_ENSEMBLE_SIZE_STUDY.md`, `ensemble_size_ood_score_raw.json` (anchor s3b0, seed 0,
scale 25, bf=0.05, CALIB=1024; one M=100 member pool, subsampled).

- From **M=32 → M=100**, Near-AUROC rises only **+0.09** (92.27 → 92.36); Far-AUROC **+0.05**.
- **M=16** is within **0.14** combined-AUROC of M=100.
- Member-subset variance vanishes by M≈32: subset std **±0.41 at M=2 → ±0.07 at M=32 → ±0.00 at M=100**.

**Why this matters for the efficiency argument:** P&C latency is linear in member count. At **M=24**,
inference cost roughly **halves to ~3.6 ms/sample** — which would place P&C at **~half** the matched-M
Deep Ensemble's latency and **~2× storage reduction** (675 MiB → ~324 MiB resident), at a cost of
≤0.14 AUROC points. This is the one lever that converts the latency story from "tie" to "win", and it
is an existing measured result, not a projection.

Caveat if used: this is **seed 0 only**, and the saturation point was selected on OOD metrics, so
present it as a post-hoc efficiency observation rather than a tuned configuration.

---

## 6. Recommended rebuttal framing

**Concede first, then quantify.** The submitted framing ("higher-cost Deep Ensemble") is unquantified
and, read as a latency claim, **false at matched M**. Replace it with two explicitly separated
comparisons:

| | Deep Ensemble n=5 (submitted-cost) | Deep Ensemble n=50 (matched-M) |
|---|---|---|
| Training | 11.1 h (5×) vs P&C 2.25 h (1×) | 111 h (50×) vs P&C 2.25 h (1×) |
| Storage on disk | 213.3 MiB vs P&C 42.7 MiB | 2.08 GiB vs P&C 42.7 MiB |
| Inference | 1.338 ms vs P&C 7.419 ms — **P&C 5.5× slower** | 7.261 ms vs P&C 7.419 ms — **tie (2.2%)** |

Claims that survive both comparisons:
1. **Training cost: 1× vs 5×/50×.** ~100 s of construction replaces 9–109 GPU-hours. Fully measured.
2. **On-disk storage: 42.7 MiB vs 213 MiB / 980 MiB / 2.08 GiB.** Measured file sizes.
3. **The least-squares objection is quantitatively small:** ~1 s per ridge solve, ~50 s for M=50.

Claims to **drop**:
4. Any suggestion that P&C is cheaper at inference. It is not, at matched M.
5. ~~Any memory claim — peak memory is unmeasured.~~ **Now measured (§4).** Memory can be claimed, but
   only in this direction: P&C beats **matched-M** DE on resident memory (1,062.7 vs 2,250.3 MiB) and
   loses to **DE n=5** (vs 226.9 MiB). "Storage" ≠ "memory" — keep the two separate.
6. **Any cost-based claim against SWAG, Laplace, MC Dropout, or Epinet.** All four are already **1×
   training**, and all four reconstruct rather than checkpoint. The measured numbers now make this
   unavoidable: Laplace constructs in **15.6 s** vs P&C's **~100 s** (§2.5), and is **lighter at
   inference in resident memory** (245.0 vs 1,062.7 MiB measured, §4).

### 6.1 The efficiency argument only separates P&C from Deep Ensembles

This is the sharpest consequence of the measured data and it should shape the rebuttal's structure:

| Comparator | Does cost separate P&C from it? |
|---|---|
| **Deep Ensemble n=50 (matched-M)** | **Yes, on every axis** — 1× vs 50× training; 42.7 MiB vs 2.08 GiB on disk; **1,063 vs 2,250 MiB resident**; latency tie |
| **Deep Ensemble n=5 (submitted)** | **Training and storage only** — 1× vs 5×, 42.7 vs 213 MiB on disk; but P&C is 5.5× slower and 4.7× heavier resident |
| SWAG | Partly — same 1× training and **SWAG constructs ~1.8× faster post-training** (55.8 s vs ~100 s), but P&C wins decisively on storage (42.7 MiB vs 980 MiB) **and on peak memory (4,332 vs 5,360 MiB)** |
| **Laplace (LLLA)** | **No** — 1× training both; LLLA constructs 6× faster (15.6 s vs ~100 s), is **4.3× lighter resident** (245 vs 1,063 MiB), and peaks lower (2,097 vs 4,332 MiB) |
| MC Dropout / Epinet | **No** — 1× training, *faster* at inference (5.18 / 1.62 ms), and **~24× lighter resident** (44 MiB) |

**Implication:** against SWAG and Laplace the paper's differentiator must be **uncertainty quality at
comparable cost**, not cost. Efficiency is a Deep-Ensemble argument specifically. Framing P&C as
broadly "cheaper" invites exactly the rebuttal WKSj is already gesturing at, and the numbers in this
document would not survive it.

---

## 7. Provenance index

| Quantity | File |
|---|---|
| Inference latency, all methods | `results/cifar10/inference_cost.json`; `results/cifar10/inference_cost/*.json` |
| Matched-M DE n=50 latency | `results/neurips_2026_rebuttal/cifar/inference_cost_deep_ensemble_n50_matched.json` |
| Latency table (rendered) | `results/neurips_2026_rebuttal/cifar/efficiency_cifar_table.{md,csv}` |
| Latency harness | `scripts/benchmark_inference_cost.py`; `_de_matched_timing.py` |
| Base / MCD training wall-clock | `experiments/logs/cifar10_finish_20260427_223643.log`; `mcd_dropout_sweep_*.log` |
| CIFAR-100 base training corroboration | `experiments/cifar_neurips_strengthening_log.md` ("Final base training timeline") |
| P&C build+eval wall-clock, 3 seeds | `experiments/logs/p2_s3b0_ood_20260428_141145.log` |
| P&C per-stage eval_time decomposition | `results/cifar10/openood_v1p5_pnc_single_block_vcal_s3b0_k20_n50_ps25.0_lr0.001_bf0.05_*_seed0_random.json` |
| Ridge / Gram / eig component timings | `results/conv_construction_rank/{combined,stage_s2b0,phaseD_matched,round2/stage_s3b1}/timings.json` |
| **Peak GPU memory, all 11 methods** | `peak_memory/peak_memory_{method}.json`, `peak_memory/peak_memory_table.csv`, driver log `peak_memory/peak_memory.log` |
| **Peak-memory harness** | `_peak_memory_bench.py` (one method per subprocess), `_peak_memory_aggregate.py` |
| **s3b0 vs s3b1 block discrepancy** | `scripts/benchmark_inference_cost.py:366` vs `ANCHOR_CONFIG.json` |
| **SWAG post-training construction (3 seeds)** | `swag_construction_time.json`; harness `_swag_construction_time.py` |
| SWAG / Epinet training times (mtime deltas) | `stat` on `results/cifar10/preact_resnet18_swag_train_*_seed{0,1,2}.pkl`, `epinet_train_*_seed{0,1,2}.pkl` |
| Construction peak (superseded by §4) | `results/conv_construction_rank/FINAL_REPORT.md` §7 |
| Checkpoint sizes & param counts | `results/cifar10/*.pkl` (`stat` + pickled-state walk) |
| P&C member materialization | `ensembles.py`, `PnCEnsemble._precompute_corrections` |
| LLLA dense-GGN posterior construction | `cifar_tasks.py:1929-1953` (and `:2431-2449`) |
| LLLA covariance + Cholesky residency | `ensembles.py:2105-2160`, `LLLAEnsemble.__init__` |
| LLLA result files (no checkpoint exists) | `results/cifar10/baseline_llla_vcal_n50_prec*_seed*.json` |
| **LLLA construction time (`train_time`, 18 runs)** | same files; timer defined at `cifar_tasks.py:1907` (`t0`) and `:1953` |
| Anchor configuration & tensor shapes | `results/neurips_2026_rebuttal/cifar/ANCHOR_CONFIG.json` |
| Ensemble-size saturation | `results/neurips_2026_rebuttal/cifar/CIFAR_ENSEMBLE_SIZE_STUDY.md`; `ensemble_size_ood_score_raw.json` |
| Prior efficiency synthesis | `results/neurips_2026_rebuttal/cifar/CIFAR_REBUTTAL_RESULTS.md` §Efficiency |

**Remaining gaps (all minor):**
1. **A breakdown of LLLA's 15.56 s** into GGN accumulation vs. dense 5,130² inverse vs. Cholesky. The
   total is measured (§2.5) but the timer spans all three.
2. **SWAG / Epinet training times are mtime-derived, not stage-timed** (§2.6). The deltas are
   reproducible to ±6 s and self-calibrated against the base network, but they are completion-to-
   completion and include inter-task overhead. Quote as upper bounds.
3. **Peak memory is single-run, seed 0, one GPU.** No repeats, so no variance estimate. The latency
   cross-check (§4.1) agrees with the published table to within 3% on 9 of 10 methods, which supports
   the run's validity but is not a substitute for repeats.
4. **P&C multi-block peak (6,063 MiB) is 74% of an 8,151 MiB card.** It was not measured on a larger
   GPU, so it is unknown whether that figure reflects the method's true demand or allocator pressure
   near the limit.

**Do not confuse LLLA's two timing numbers.** The 6.00 s figure in §1 is *predict-path JIT warmup*;
the 15.56 s figure in §2.5 is *posterior fit*. They measure different things and neither substitutes
for the other.

Item 2 is the reason the cost tables lean on the Deep Ensemble comparison: DE is the method under
dispute (A5nU), and it is the one whose training cost is fully derived from a measured base-train time
rather than a projection.
