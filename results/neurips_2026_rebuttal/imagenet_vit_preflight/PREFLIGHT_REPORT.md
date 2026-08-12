# ViT-B/16 P&C on the TITAN X — feasibility preflight

**Verdict: `GO_WITH_TOKEN_SUBSAMPLING`** — every memory, numerical and runtime criterion is
met with wide margin, but the correction must be fitted on **CLS-token rows only**. Fitting
on patch tokens produces a correction that is *worse than applying no correction at all*,
even at 43× overdetermination. See §9 and §12.

Machine: NVIDIA TITAN X (Pascal), 12 GiB, sm_61 · i7-12700KF (20 threads), 24.6 GiB RAM ·
driver 581.80 / CUDA 13.0. Model: torchvision `vit_b_16`,
`ViT_B_16_Weights.IMAGENET1K_V1` (pinned, SHA-256 `c867db91…`), 86,567,656 parameters,
float32, `eval()`, every forward under `torch.inference_mode()`, no gradients anywhere.

Data: ImageNet-1k validation, both shards (50,000 real images) from the public mirror
`evanarlian/imagenet_1k_resized_256`, with the checkpoint's own transform. Splits are a
fixed permutation: 40,000 calibration / 8,192 held-out ID / 512 benchmark, disjoint.
**No OOD data was downloaded or evaluated** (spec §21); `data.py` has no code path that can.

Raw CSV/JSON are alongside this file; `REPRODUCE.md` regenerates everything.

---

## 0. A blocker found before any measurement

The repo's main `.venv` carries **torch 2.11.0+cu130, which ships no `sm_61` kernels** —
its arch list starts at `sm_75`. Every CUDA call on this card fails with
`CUDA error: no kernel image is available for execution on the device`. This would have
stopped the ImageNet experiment outright, and it is invisible until a kernel actually
launches: `torch.cuda.is_available()` returns `True` and the model loads fine.

Fix applied: a sibling venv `.venv_vit` with **torch 2.7.1+cu126**, whose `sm_60` cubins are
binary-compatible with sm_61 (same major compute-capability version). Verified end-to-end —
ViT-B/16 reproduces **81.64 % top-1 / 96.29 % top-5** on 512 held-out validation images
against the published 81.07 / 95.32, so checkpoint and preprocessing are faithful. The main
`.venv` is untouched; the naming matches the repo's existing `.venv_bank` convention and the
`.venv_*/` gitignore rule.

## 1. Correctness gates

All nine pass (`raw/validate_gates.json`). The reuse-critical two are exact:

| Gate | Result |
|---|---|
| `solver_parity` — streamed `G`/`C` solve vs `pnc_theory.linalg.ridge_solve` | **bitwise identical** |
| `basis_parity` — basis / coefficients / scale vs `banking77_pnc.construct` | **bitwise identical** |
| `cached_prefix_parity` — prefix+tail vs full model | 9.5e-07 |
| `cls_only_parity` — CLS-only tail vs full-token tail | 9.5e-07 |
| `mutation_restore` — perturb W1 in place, then restore | logits **bitwise** equal to base |
| `cho_parity` / `residual_from_stats` / `stats_accumulation` | 3.9e-16 / 1.1e-16 / exact |
| `token_sampling` | deterministic, seed-sensitive, CLS always present |

`cls_only_parity` is structural and drives much of what follows: after the **final** block's
FFN everything left (`encoder.ln`, CLS slice, `heads`) is token-wise, so a member's tail
only ever needs the CLS row. It does **not** hold for earlier blocks, where later attention
layers mix tokens — quantified in §11.

---

## 2. Does ViT-B/16 fit in float32? Yes, with ~10.6 GiB spare

| | GiB |
|---|---|
| fp32 weights resident | 0.340 |
| peak allocated / reserved, batch 16 | 0.474 / **0.555** |
| peak reserved, batch 256 | 2.852 |
| other processes on the card | 0.875 |
| **free at batch 16** | **≈ 10.6** |

Memory never became the binding constraint. The spec's stop rule (first OOM, or >11 GiB
reserved) **never fired**; the sweep was extended to batch 256 and still only reached
2.85 GiB. There is no OOM row in `memory_preflight.csv` because nothing came near one.

## 3. Safe batch size: **16**

| batch | 1 | 2 | 4 | 8 | **16** | 32 | 64 | 128 | 256 |
|---|---|---|---|---|---|---|---|---|---|
| img/s (cold) | 90.1 | 113.0 | 148.8 | 149.1 | **148.3** | 123.4 | 94.2 | 130.2 | 97.1 |
| peak reserved GiB | 0.40 | 0.41 | 0.43 | 0.49 | **0.56** | 0.71 | 1.02 | 1.62 | 2.85 |

Throughput plateaus at batch 4–16 and **degrades** beyond it, reproducibly across randomised
re-measurement (batch 32 ≈ 110, batch 64 ≈ 85 img/s on a second pass). Because every
candidate clears the 1.5 GiB headroom requirement by ~9 GiB, the headroom criterion is
non-binding and the choice is made on throughput. Batch 16 sits on the plateau, leaves
~10.6 GiB free, and amortises the per-member loop better than batch 4.

**Sustained throughput is 129.6 img/s, not 148.** Under continuous load the card reaches
87–90 °C, the SM clock falls 1683 → 1417 MHz and `SwThermalSlowdown` engages. All runtime
projections use the settled figure; the cold number would understate a multi-hour run by
~13 %.

## 4. FFN activations (batch 16, T = 197 = 1 CLS + 14×14 patches)

| tensor | shape | MiB | MiB/image |
|---|---|---|---|
| `h` = `ln_2(x)` — input to W1 | (16, 197, 768) | 9.23 | 0.577 |
| `y` = post-GELU — input to W2 | (16, 197, 3072) | 36.94 | 2.309 |
| `z` = original FFN output | (16, 197, 768) | 9.23 | 0.577 |

Retaining all three costs 55.4 MiB at batch 16. Activations are released every batch;
nothing is accumulated across the dataset on GPU.

## 5. Perturbation machinery and one perturbed member

| | |
|---|---|
| basis `U` (K=20, D_flat = 768×3072 = 2,359,296) | 180.0 MiB **CPU**, 0 GPU, 0.87 s |
| one member's `dW1`, on demand | 9.0 MiB, 7.6 ms |
| median ‖dW1‖<sub>F</sub>/‖W1‖<sub>F</sub> | 0.500 (‖W1‖<sub>F</sub> = 52.66, scale 5.726) |
| compact state, M=20 | **360 MiB** vs **6.45 GiB** for 20 full models |
| **peak GPU, perturbed member (batch 16)** | **0.483 GiB** |

The perturbed forward is finite, moves the logits (max |Δlogit| 0.298), and functional vs
in-place mutation agree to 2e-06. After `restore()` the weights are bitwise pristine and the
logits **bitwise equal** to base.

A random 50 % Frobenius perturbation of W1 changes the pre-activation by only ~1.8 %
(≈ 0.5/√768), because the direction is random in 2.36M dimensions — hence top-1 agreement
against base is still ~0.98. Member diversity is set by the scale sweep, which this
preflight deliberately does not tune.

---

## 6. All-token X<sup>T</sup>X accumulation is cheap — rows are not the bottleneck

512 calibration images, batch 16:

| mode | rows/image | rows/s | img/s | X<sup>T</sup>X per batch | 1,000 images | rows / 3073 |
|---|---|---|---|---|---|---|
| cls | 1 | 117 | 116.8 | 1.6 ms | 9 s | 0.17 |
| cls+4 | 5 | 595 | 119.0 | 1.0 ms | 8 s | 0.83 |
| cls+16 | 17 | 2,009 | 118.2 | 1.6 ms | 8 s | 2.83 |
| **all** | **197** | **20,434** | **103.7** | **11.0 ms** | **10 s** | **32.8** |

Accumulation is ~7 % of batch cost even at 197 rows/image; all-token runs at 104 img/s
against CLS-only's 117. `G` (3073²) and `C` (3073×768) are 37.8 + 9.4 MiB on GPU regardless
of dataset size — the design matrix is never materialised.

**So the expensive resource is images, not tokens.** That inverts the question the spec
anticipated: the issue is not whether we can afford all tokens, but whether they help.

## 7. How expensive is the 3072-dimensional solve? Negligible — if the factorisation is shared

Per member, 3073×3073 system with 768 right-hand sides, float64 on CPU:

| | |
|---|---|
| Cholesky factorisation | **0.05 s** |
| solve, all 768 columns from that one factor | **0.16 s** |
| **total** | **0.28 s** |
| 768 independent solves (timed on 8, scaled) | **≈ 120 s** |
| **saving from sharing the factorisation** | **≈ 400×** |
| M = 20 members | 5.6 s |
| peak CPU RSS | 2.1–2.7 GiB of 24.6 GiB |

Stable across every row budget and token mode (`ridge_benchmark.csv`, 16 fits). The solve
stays on CPU in float64 as the Banking77 experiment does; there is no reason to move a
0.28 s factorisation onto a Pascal card.

## 8. Row budget: 4,096 rows actively *harms* the model

Common yardstick — every fit scored on the same held-out CLS residual and CLS logit MSE, so
modes are comparable (each mode's own-token residual is not).

Uncorrected baseline: CLS residual **0.2270**, logit MSE **6.68e-03**.

| rows | rows/3073 | **cls** CLS-res / logit MSE | **cls+16** | **all** |
|---|---|---|---|---|
| 4,096 | 1.33 | 0.365 / 1.80e-01 | 0.674 / 8.91e-02 | 0.770 / 9.80e-02 |
| 8,192 | 2.67 | 0.160 / **4.50e-03** | 0.405 / 3.69e-02 | 0.480 / 4.35e-02 |
| 16,384 | 5.33 | 0.135 / **2.87e-03** | 0.326 / 2.66e-02 | 0.392 / 3.05e-02 |
| 32,768 | 10.7 | **0.126 / 2.62e-03** | 0.257 / 2.29e-02 | 0.362 / 2.60e-02 |
| 65,536 | 21.3 | — | 0.201 / 2.10e-02 | 0.338 / 2.29e-02 |
| 131,072 | 42.7 | — | 0.155 / 4.34e-03 | 0.314 / 2.13e-02 |

At 4,096 rows every mode overfits hard: calibration residual is tiny (0.011–0.030) while
held-out behaviour is far worse than doing nothing. **A barely-overdetermined ridge is worse
than no correction**, and the calibration residual gives no warning — it looks best exactly
where the fit is worst.

## 9. Token subsampling: it is the *number of images* that matters, not tokens per image

Read the table two ways:

* **At a fixed row budget**, more images wins by a wide margin. At 32,768 rows: CLS-only
  (32,768 images) gives CLS residual 0.126; cls+16 (1,936 images) 0.257; all-token
  (176 images) 0.362 — **2.9× worse** from the same number of regression rows.
* **At a matched image budget**, extra tokens per image add almost nothing. cls at 8,192
  images → 0.160; cls+16 at 7,711 images (131,072 rows, 16× more rows) → 0.155.
  Statistically indistinguishable.

Tokens from the same image are highly redundant; they inflate the row count without adding
information. And for a **final-block** target only the CLS row influences the logits at all,
so patch rows actively pull the least-squares fit away from the one direction that matters.

This conclusion rests on the CLS residual, which aggregates 1,024 held-out CLS rows and is
not outlier-sensitive; the logit MSE column agrees. The finer claim (cls ≈ cls+16 at matched
images) is measured on a 512-image logit probe and should be reconfirmed with the
distributional metric of §10 before it is relied on.

## 10. ID preservation is heavy-tailed — a mean over a few hundred images is not a yardstick

The same fitted correction scored 2.87e-03 on one 512-image held-out set and 4.49e-02 on
another. Per-image logit MSE over 2,048 held-out ID images explains why: the correction
improves the typical image several-fold but has a long tail that dominates the mean.

`id_preservation.csv`, CLS-only, 2,048 held-out ID images.
Uncorrected: mean 6.13e-03, median 4.51e-03, p99 2.42e-02, top-1 agreement 0.9883.

| rows | λ | median (×better) | mean | p99 | % images improved | top-1 agree |
|---|---|---|---|---|---|---|
| 8,192 | 1e-3 | 1.54e-03 (2.9×) | 7.02e-03 | 7.86e-02 | 81.2 % | 0.9858 |
| 8,192 | 1e0 | 1.14e-03 (4.0×) | 2.99e-03 | 2.25e-02 | 91.6 % | 0.9902 |
| 16,384 | 1e-3 | 9.67e-04 (4.7×) | 2.71e-03 | 3.08e-02 | 97.2 % | 0.9893 |
| 16,384 | 1e0 | 9.10e-04 (5.0×) | **2.16e-03** | 1.73e-02 | 98.6 % | 0.9902 |
| **32,768** | **1e-3** | **1.22e-03 (3.7×)** | **2.40e-03** | **1.42e-02** | **99.0 %** | **0.9912** |
| 32,768 | 1e0 | 1.21e-03 (3.7×) | 2.39e-03 | 1.38e-02 | 99.2 % | 0.9907 |
| 32,768 | 1e4 | 3.66e-03 (1.2×) | 5.25e-03 | 2.27e-02 | 97.5 % | 0.9883 |

**At 32,768 rows the spec's λ = 1e-3 works cleanly**: mean, median *and* p99 all beat
uncorrected, and 99.0 % of individual images improve. Below that, λ = 1e-3 leaves a tail
worse than the uncorrected model (p99 3.08e-02 vs 2.42e-02 at 16,384 rows) and needs
λ ≈ 1 to control it. λ = 1e-3 in the **summed** objective is λ/n ≈ 3e-8 per row — effectively
unregularised, so at these row counts the regularisation that matters comes from row count.

Two consequences for the real experiment: report **median and p99, not just mean**, and use
**≥ 2,000 evaluation images** for any ID-preservation gate.

## 11. One complete corrected member, and the cost of each target block

End-to-end member (CLS-only, 32,768 rows, λ=1e-3), evaluated on 512 held-out ID images:

| | |
|---|---|
| construction | **282.8 s** (accumulate 282.4 s over 32,768 images + solve 0.38 s) |
| **peak GPU** | **0.527 GiB** | 
| peak CPU RSS | 1.54 GiB |
| top-1: base / uncorrected / corrected | 0.8047 / 0.8066 / 0.8066 |
| agreement with base: uncorrected → corrected | 0.9805 → **0.9824** |
| logit MSE vs base: uncorrected → corrected | 8.82e-03 → **5.47e-03 (1.61×)** |
| all finite | yes |

**Construction cost collapses once `(h, z)` is cached.** Neither the member nor the scale
changes the FFN input, so the ViT prefix is paid **once per target block**, not once per
member:

| | |
|---|---|
| `(h, z)` cache, 32,768 rows | 277.4 s, **192 MiB** GPU |
| per member thereafter | **0.45 s** (0.17 s GPU accumulation + 0.27 s CPU solve) |
| naive, re-running the prefix per member | 281.6 s |
| **speedup** | **632×** |

Measured tail cost per target block at M=20 (batch 16):

| target block | tail | prefix | per member | img/s at M=20 | peak GiB |
|---|---|---|---|---|---|
| **11 (final)** | CLS-only | 6.49 ms | **0.015 ms** | **147.0** | 0.817 |
| 10 | all 197 tokens | 5.94 ms | 0.850 ms | 43.6 | 0.863 |
| 9 | all 197 tokens | 5.42 ms | 1.472 ms | 28.7 | 0.863 |

Members on the final block are **57× cheaper** than on block 10. Earlier blocks need the
full-token tail because subsequent attention mixes tokens — so the CLS-only shortcut, and
the CLS-only *correction design* of §9, both need re-deriving per block.

## 12. M = 20 shared-prefix inference: memory is flat in M

Prefix computed once per batch, members evaluated sequentially, temporaries freed each
member. Tail-only throughput at batch 16:

| M | resident, `dW1` on demand | **resident, W1 precomputed** | streaming from CPU |
|---|---|---|---|
| 1 | 1,867 img/s · 0.632 GiB | 36,502 img/s · 0.808 GiB | 1,795 img/s · 0.632 GiB |
| 5 | 499 img/s · 0.632 GiB | 10,222 img/s · 0.808 GiB | 403 img/s · 0.632 GiB |
| **20** | **125 img/s · 0.632 GiB** | **3,323 img/s · 0.808 GiB** | **100 img/s · 0.632 GiB** |

**Peak VRAM is identical at M = 1, 5 and 20** — 0.632 GiB throughout — so no member tensors
are being retained. Precomputing the 20 perturbed W1 once (+176 MiB, total 0.808 GiB, still
flat in M) is **26× faster** at M=20: re-expanding `coeff @ U` over the 2.36M-dim basis costs
~150 ms of CPU per batch otherwise, which would dominate a 50k-image pass. Use it.

Streaming corrections from CPU is 20 % *slower* at M=20 with no memory benefit (same
0.632 GiB), so **keep corrections resident** — they are only 180 MiB.

### Storage (M = 20)

| | |
|---|---|
| shared basis | 180.0 MiB |
| 20 corrected W2 + b2 | 180.1 MiB |
| coefficients | 1.6 KiB |
| **CPU total / on disk** | **360.1 MiB / 334.3 MiB** |
| 20 full model copies (avoided) | 6.45 GiB |

### fp16 (§17) — do not use it

72.7 img/s versus ~130 for float32: fp16 is **44 % slower** on Pascal, which has no
tensor cores and quarter-rate fp16 ALUs. Logits agree with fp32 to 5.1e-03 with 1.000 top-1
agreement, so it is *correct* — just pointless here. float32 remains the only path; `G`/`C`
and the ridge solve were never fp16 regardless.

---

## 13. Projected runtime for the full experiment

From measured primitives, derated by the 1.145× cold→sustained thermal factor
(`raw/runtime_projection.json`).

**Construction** — 3 blocks × 3 scales × M=10 selection members, plus M=20 final = 110 members:

| | |
|---|---|
| `(h, z)` cache, 3 blocks × 277 s | 13.9 min |
| 110 members × 0.45 s | 49 s |
| **total** | **14.7 min** |
| naive (prefix per member) | 8.6 h → **35× saving** |

**Evaluation** (M = 20):

| target block | ID 50k | OpenOOD suite (86,388) | per 10k OOD |
|---|---|---|---|
| 11 (final) | 6.5 min | 11.2 min | 1.3 min |
| 10 | 21.9 min | 37.8 min | 4.4 min |
| 9 | 33.3 min | 57.5 min | 6.7 min |

**Totals**

| scenario | construction | selection | ID | OOD | **total** | with 20 % margin |
|---|---|---|---|---|---|---|
| final block only | 5.4 min | 1.9 min | 6.5 min | 11.2 min | **0.42 GPU-h** | 0.50 h |
| all three blocks | 14.7 min | 11.8 min | 61.6 min | 106.5 min | **3.24 GPU-h** | **3.89 h** |

Disk I/O is 334 MiB written once — negligible. CPU ridge solves total 110 × 0.27 s = 30 s.
Wall-clock ≈ GPU-hours: the pipeline is GPU-bound, with JPEG decode (448 img/s across 8
threads) comfortably ahead of the 130 img/s the card sustains.

---

## Answers to the twelve questions

1. **Does ViT-B/16 fit in float32?** Yes — 0.340 GiB resident, 0.555 GiB peak reserved at
   batch 16, ~10.6 GiB free. Memory was never close to binding.
2. **Safest useful batch size?** **16** — on the throughput plateau, ~10.6 GiB headroom.
   Throughput *falls* above 32, so bigger is worse, not just riskier.
3. **Peak memory, ordinary inference?** 0.474 GiB allocated / 0.555 GiB reserved (batch 16).
4. **Peak memory, perturbed member?** **0.483 GiB**.
5. **Peak memory, corrected member?** **0.527 GiB** (1.40 GiB during the cached-(h,z)
   construction, the largest figure anywhere — still 7× under budget).
6. **Does M=20 shared-prefix sequential inference fit?** Yes, and peak VRAM is **flat in M**
   (0.632 GiB at M=1, 5 and 20; 0.808 GiB with W1 precomputed).
7. **Cost of the 3072-dim solve?** 0.28 s per member (0.05 s Cholesky + 0.16 s for all 768
   columns), 2.1–2.7 GiB CPU RSS. Sharing the factorisation saves ~400×.
8. **Cost of all-token X<sup>T</sup>X accumulation?** 11 ms per batch of 16 — about 7 % of
   the batch. All-token runs at 104 img/s vs 117 for CLS-only. Not a bottleneck.
9. **Is token subsampling sufficient on held-out ID data?** More than sufficient — it is
   *required*. CLS-only is the best design at every budget; patch-token rows make the
   correction worse than no correction. What matters is the number of distinct calibration
   images, not rows per image.
10. **Recommended row budget?** **32,768 CLS rows = 32,768 calibration images**, λ=1e-3.
    That is where mean, median and p99 logit error all beat the uncorrected model and 99.0 %
    of images improve. 16,384 rows with λ≈1 is an equally good, half-cost alternative worth
    confirming.
11. **Projected runtime?** **0.42 GPU-h** for the final block alone; **3.24 GPU-h**
    (≈3.9 h with margin) to construct and evaluate all three target blocks.
12. **Verdict?** `GO_WITH_TOKEN_SUBSAMPLING`.

## Recommended configuration

```text
Model:                    ViT-B/16, torchvision, ViT_B_16_Weights.IMAGENET1K_V1 (pinned)
dtype:                    float32          (fp16 is 44% slower on Pascal — do not use)
Target:                   final FFN (encoder_layer_11.mlp.0 → GELU → mlp.3) first;
                          blocks 10 and 9 are affordable but need per-block re-derivation
                          of both the tail and the token design
safe batch size:          16
correction rows:          32,768
token sampling:           CLS only (1 row/image) — NOT all-token, NOT random patch tokens
calibration images:       32,768 (rows == images at CLS-only)
K:                        20
M selection:              10        M final: 20
ridge:                    1e-3 (summed objective, centred on the original W2)
solve:                    float64 CPU, one Cholesky shared across all 768 outputs
expected peak VRAM:       0.81 GiB inference (M=20) / 1.40 GiB construction
construction/member:      0.45 s after a one-time 277 s (h, z) cache per target block
full evaluation:          6.5 min ID (50k) + 11.2 min OpenOOD, at M=20, final block
```

Implementation notes that materially change cost or correctness:

* Cache `(h, z)` once per target block — 632× cheaper per member than re-running the prefix.
* Precompute the M perturbed W1 once per pass (+176 MiB) — 26× faster than expanding
  `coeff @ U` per batch.
* Keep corrections resident on GPU (180 MiB); streaming is slower and saves nothing.
* Gate ID preservation on **median and p99** over ≥2,000 images, never a mean over a few
  hundred.
* Watch the calibration residual as a *warning sign*, not a target: it is lowest exactly
  where the correction is most overfitted.

## Why `GO_WITH_TOKEN_SUBSAMPLING` and not plain `GO`

All seven of the spec's preferred GO criteria hold: float32 inference fits with ~10.6 GiB
spare (≫ 1.5 GiB), single-member construction fits, M=20 shared-prefix inference fits with
flat peak VRAM, the solve uses ~2.5 GiB of 24.6 GiB RAM, a stable correction is obtained at
a tractable 32,768-row budget, nothing was non-finite anywhere, and 3.24 GPU-h is a
comfortable runtime.

The label is qualified because the experiment **cannot** be run with the all-token
correction design and still preserve ID behaviour. The spec anticipated reaching this label
because all-token would be too slow; the measured reason is the opposite — all-token is
cheap (7 % of batch cost) but produces a correction 3–8× worse than doing nothing. The
operative instruction is the same either way: subsample tokens, use 8k–32k rows.

## Caveats

* Single perturbation scale (`target_rel = 0.5`, multiplier 1.0) and a single member for the
  budget sweeps. Scale selection is explicitly out of scope here.
* Only the final block was characterised for correction *quality*; blocks 10 and 9 were
  measured for cost only. Their token design must be re-derived, since patch tokens do reach
  the CLS output through later attention there.
* The cross-mode comparison at matched image count rests on a 512-image logit probe (§9);
  the CLS-residual evidence for the same conclusion is more robust.
* Thermals: the card runs at 87–90 °C under sustained load with `SwThermalSlowdown` active.
  Projections use settled throughput, but a warmer ambient would degrade them further.
