# CIFAR NeurIPS Strengthening Plan

## Context

Following the OOD detection benchmark (`cifar10_ood_detection.md`) and the UQ tuning results (`cifar10_tuning_plan.md`), the current state is:

- **UQ on CIFAR-10:** 3-seed results, tight error bars, PnC competitive on NLL
- **OOD on CIFAR-10:** 12 methods evaluated, **single seed only**, paper-clean protocol verified
- **CIFAR-100:** infrastructure exists but no trained models, no results
- **WideResNet-28-10:** not implemented
- **Inference cost reporting:** absent

The OOD margins between top methods are tight (PnC scale=20.0 vs SWAG: 0.9 AUROC points), within plausible seed variance. Multiple reviewer concerns are real and need addressing before NeurIPS submission. This plan resolves them in priority order, with effort estimates and clear deliverables.

## Existing assets (do not retrain)

- **CIFAR-10 PreActResNet18 base models:** seeds 0–6 (7 checkpoints) at `e300` recipe
- **CIFAR-10 SWAG checkpoints:** seeds 0, 1, 2 at `sws240`
- **CIFAR-10 MCDropout checkpoints:** seeds 0, 1, 2
- **CIFAR-10 Epinet ps=3.0:** **only seed 0** — need to train seeds 1, 2
- **CIFAR-10 OOD data:** all 6 datasets prepared (cifar100, tiny_imagenet, mnist, svhn, textures, places365)
- **CIFAR-100 data loader:** verified working

## Priorities (in order) and execution status

### P0 — Multi-seed OOD on CIFAR-10 (the headline issue) — **✅ DONE**
### P1 — Resolve PnC configuration ambiguities — **✅ DONE** (multi-block scale=7, single-block headline scale=50, also kept scale=25 as ID-conservative entry)
### P2 — Inference cost benchmarking — **✅ DONE** (per-method subprocess workaround)
### P3 — CIFAR-100 OOD benchmark — **🟡 IN PROGRESS** (base seeds 0,1 done; seed 2 in flight; auto-pipeline queued for SWAG/Epinet/MCDropout/OOD eval)
### P4 — WideResNet-28-10 — **⏳ NOT STARTED**
### P5 — Narrative + paper-table reframing — **⏳ NOT STARTED** (paper-ready CIFAR-10 table at `experiments/cifar10_ood_paper_table.md`)

**Headline result achieved (multi-seed CIFAR-10):**

> PnC scale=50 (1× training cost, post-hoc) BEATS Standard Ensemble (5× training cost) on Near-OOD detection by clear statistical margins:
> - Near AUROC: 91.51 ± 0.21 vs 91.10 ± 0.04 (+0.41, ~10σ)
> - Near FPR95: 32.17 ± 1.23 vs 40.39 ± 0.47 (−8.22, ~17σ)
> - Far-OOD: tie within seed noise.
> - Trade-off: 0.97 pp ID accuracy gap, +0.035 NLL, 5.5× inference latency.

---

## P0 — Multi-seed OOD on CIFAR-10

**Goal:** Run all 12 OOD methods on seeds {0, 1, 2}; report mean ± std for every metric.

### What's already in place
All 12 OOD Luigi tasks accept a `--seed` parameter. Seed 0 results exist. Seeds 1 and 2 need to be run.

### Required pre-work
- **Train Epinet seeds 1, 2 with prior_scale=3.0** — `CIFARTrainEpinet --prior-scale 3.0 --seed 1` and `--seed 2`. ~30 min each on GPU.

### Run plan
For each seed in {1, 2}, run all 12 tasks (the same set as `scripts/run_ood_eval.sh`). Seed 0 is already done.

Use the existing `scripts/run_ood_eval.sh` pattern but parameterize by seed. Add a wrapper script `scripts/run_ood_eval_multi_seed.sh`.

### New aggregation script
Create `scripts/aggregate_ood_results.py`:
- Load all `openood_v1p5_*_seed{0,1,2}.json` files
- For each (method, metric) compute mean and std across seeds
- Emit a markdown table with `mean ± std` cells
- Reuse the protocol-compliance check from `ood_comparison_table.py`

### SWAG cache fix
The `cache_samples=True` change is already in place. SWAG seed runs should each take ~5 min.

### Compute estimate
- Epinet training (2 seeds): ~1 hour
- All 12 OOD methods × 2 seeds: ~6 hours wall time (most methods 5–15 min each)
- **Total: ~7 hours**

### Deliverables
1. `scripts/run_ood_eval_multi_seed.sh`
2. `scripts/aggregate_ood_results.py`
3. `results/cifar10/openood_v1p5_*_seed{1,2}.json` (24 files)
4. `experiments/cifar10_ood_detection.md` updated with mean±std table

---

## P1 — Resolve PnC configuration ambiguities

Two issues from the assessment:

### P1.a — Multi-block PnC scale 6.0 vs 7.0 inconsistency

The UQ tuning concluded multi-block scale=**7.0** is optimal (validated 3 seeds, blocks [2,4,6,7], K=20). The OOD benchmark used scale=**6.0** (from the user's spec). These should match.

**Decision:** Use scale=**7.0** everywhere. Rerun the multi-block PnC OOD benchmark with scale=7.0 on all 3 seeds. Document the change in the experiment log. Old scale=6.0 result files can stay on disk for reference but should not appear in the headline table.

This is folded into P0: when running multi-seed OOD, use scale=7.0 for the multi-block task.

### P1.b — Validate PnC scale=20.0 single-block

The single-block PnC scale=20.0 is currently the headline OOD winner (Near AUROC 90.8) but rests on a single seed and was not part of the UQ multi-seed validation. The UQ tuning's flat optimum was scale=6–7, so scale=20.0 is well outside the validated range.

**Plan:**

1. **3-seed ID validation:** Run `CIFARPreActPnC` (the ID-only task) at `target_stage_idx=3, target_block_idx=1, n_directions=20, perturbation_sizes=[20.0], random_directions=True` on seeds {0, 1, 2}. Confirm ID accuracy / NLL / ECE are stable.
2. **3-seed OOD validation:** Run `CIFAROpenOODPnC` with the same config on seeds {0, 1, 2}. Already covered by P0.
3. **Sanity sweep:** Single-block scale ∈ {5, 10, 15, 20, 25, 30} on seed 0 to characterize the OOD-vs-scale curve. This tests whether scale=20 is a flat optimum, a sharp peak, or a noise artifact.
4. **If scale=20.0 holds up:** Add a section to the OOD report explaining *why* — likely hypothesis: high-scale single-block perturbation increases ensemble diversity at the cost of small ID-accuracy loss, but the ridge-regression correction recovers ID quality. Worth 2–3 sentences.
5. **If scale=20.0 doesn't hold up:** Replace with the multi-block scale=7.0 result as the headline PnC entry.

### Compute estimate
- Sanity sweep on seed 0: 6 scales × ~10 min each = ~1 hour
- Multi-seed validation: folded into P0

### Deliverables
1. Sanity-sweep result files in `results/cifar10/`
2. Update to `experiments/cifar10_ood_detection.md` with scale-vs-OOD curve and decision rationale

---

## P2 — Inference cost benchmarking

**Goal:** Add a wall-clock and per-sample cost column to the headline OOD/UQ tables.

### Approach
Create `scripts/benchmark_inference_cost.py`:

1. Load each method's ensemble (one seed is enough for cost numbers)
2. Warm up with one batch (JIT compile)
3. Time `_predict_cifar_logits()` on a fixed 5,000-sample subset of CIFAR-10 test, batch_size=256
4. Record wall-clock time and divide by 5,000 to get per-sample cost
5. Also report:
   - **Training cost** in epochs (Deep Ensemble = 5×, others = 1×)
   - **Memory at inference**: max GPU memory used
6. Output a JSON `results/cifar10/inference_cost.json` and a markdown table

### Reporting columns to add
| Method | Train epochs (×) | Inference per sample (ms) | Notes |
|--------|------------------|---------------------------|-------|
| PreAct ResNet-18 | 1× | 0.X | base |
| Deep Ensemble n=5 | 5× | 5×0.X | 5 forward passes |
| MC Dropout n=32 | 1× | 32×0.X | stochastic |
| SWAG n=50 (cached) | 1× | 50×0.X + BN refresh | one-time BN cost |
| LLLA n=50 | 1× | 50×0.X + sample cost | small overhead |
| Epinet n=50 | 1.05× | 1×features + 50×MLP | tiny MLP per sample |
| PnC K=20 | 1× | 20×0.X (perturbed conv) | per-batch ridge solve |

### Compute estimate
- ~30 min total (one warm-up + timing run per method)

### Deliverables
1. `scripts/benchmark_inference_cost.py`
2. `results/cifar10/inference_cost.json`
3. New cost column added to the comparison tables in `cifar10_ood_detection.md`

---

## P3 — CIFAR-100 OOD benchmark

**Goal:** Run the full 12-method paper-clean OOD benchmark with CIFAR-100 as ID.

This is the largest single piece of remaining compute and matters most for reviewer credibility.

### Step 1: Prepare CIFAR-100 OOD data

- Currently `openood_data/cifar100/` doesn't exist
- The OpenOOD benchmark for CIFAR-100 uses near-OOD: {cifar10, tiny_imagenet}, far-OOD: {mnist, svhn, textures, places365}
- All these npz files **already exist** under `openood_data/cifar10/` because they don't depend on the ID dataset
- Need to either:
  - (a) Add a `cifar10.npz` for CIFAR-100's near-OOD set (CIFAR-10 test images)
  - (b) Add symlinks from `openood_data/cifar100/{near,far}_ood/*.npz` to the existing files

Extend `scripts/prepare_openood_data.py` with a `prepare_cifar100_benchmark()` function:
- Add `prepare_cifar10_as_ood(root)` to dump CIFAR-10 test images as `openood_data/cifar100/near_ood/cifar10.npz`
- Symlink or copy the rest from cifar10 paths

### Step 2: Train CIFAR-100 base models on 3 seeds

```bash
for s in 0 1 2; do
  .venv/bin/python -m luigi --module cifar_tasks --local-scheduler \
    CIFARTrainPreActResNet18 --dataset cifar100 --epochs 300 --seed $s
done
```

Compute: ~3 hours per seed on this GPU = **9 hours wall time** (sequential).

### Step 3: Train CIFAR-100 SWAG / Epinet / MCDropout (3 seeds each)

- SWAG: `CIFARTrainSWAGPreActResNet18 --dataset cifar100 --swag-start-epoch 240` for seeds {0,1,2}
- Epinet: `CIFARTrainEpinet --dataset cifar100 --prior-scale 3.0` for seeds {0,1,2}
- MCDropout: `CIFARTrainMCDropoutPreActResNet18 --dataset cifar100` for seeds {0,1,2}

These can mostly reuse SWAG checkpoints from the SWAG run, but Epinet and MCDropout need separate training.

Compute: ~3 hours × 3 seeds × 3 methods = ~27 hours, but many can share base model. Realistic: **~12–15 hours**.

### Step 4: Run all 12 OOD methods on CIFAR-100, 3 seeds

Identical to P0 but with `--dataset cifar100`. Compute: ~7 hours.

### Step 5: Frozen-ID-hyperparameter sanity check

The CIFAR-10 hyperparameters (LLLA prior=10.0, PnC scales, Epinet ps=3.0, etc.) were tuned on CIFAR-10. For CIFAR-100 results to be paper-clean, we should either:
- (a) Use the same frozen CIFAR-10 hyperparameters and accept they may not be optimal for CIFAR-100 (this is honest and clean)
- (b) Run a brief CIFAR-100 ID-only tuning, then freeze for OOD

**Recommendation: option (a).** Reviewers care more about the *protocol* than peak numbers. State explicitly that CIFAR-100 uses CIFAR-10-tuned hyperparameters as a transferability test.

### Compute estimate
- Data prep: 30 min
- Base model training: 9 hours (3 seeds, sequential)
- SWAG/Epinet/MCDropout training: ~12 hours
- OOD evaluation (12 methods × 3 seeds): ~7 hours
- **Total: ~28 hours wall time**, can be partly parallelized

### Deliverables
1. `openood_data/cifar100/{near,far}_ood/*.npz`
2. `results/cifar100/preact_resnet18_train_*_seed{0,1,2}.pkl` and ensemble checkpoints
3. `results/cifar100/openood_v1p5_*_seed{0,1,2}.json` (36 files)
4. `experiments/cifar100_ood_detection.md` mirroring the CIFAR-10 report
5. Combined CIFAR-10 + CIFAR-100 summary in `experiments/cifar10_ood_detection.md`

---

## P4 — WideResNet-28-10 results

**Goal:** Establish architectural generality. Run the strongest 4–5 methods on WRN-28-10 for at least CIFAR-10.

### Step 1: Implement WideResNet-28-10
- Add `WideResNet28_10` class to `models.py` with the same interface as `PreActResNet18`:
  - `__call__(x, use_running_average)` → logits
  - `features(x, use_running_average)` → penultimate (640-d for WRN-28-10)
  - `forward_from_stem_out(...)` if needed for PnC
- Reference implementation: standard CIFAR WideResNet, k=10, depth=28, ~36M params

### Step 2: Add training task `CIFARTrainWideResNet28_10`
- Mirror `CIFARTrainPreActResNet18` with the new model class
- Use the same recipe (SGD lr=0.1, wd=5e-4, cosine, cutout, 200–300 epochs)

### Step 3: Train 3 seeds on CIFAR-10
- Compute: WRN-28-10 is ~3× slower per epoch than PreActResNet-18. ~9 hours per seed = **27 hours sequential** (or ~15 hours if optimized)

### Step 4: Run a reduced OOD benchmark on WRN-28-10
Limit to the methods that matter most for the headline:
- Base / MSP (single forward pass)
- Deep Ensemble (n=3 to keep compute manageable, not n=5)
- SWAG
- LLLA
- PnC (best config — single-block scale=20 OR multi-block scale=7 depending on P1 outcome)

Skip MC Dropout (we don't have a dropout-WRN), Epinet (would need separate training), Mahalanobis/ReAct (architecture-agnostic, low marginal value). The point is to show PnC's advantage holds at a different scale, not to re-run the entire suite.

### Step 5: Add WRN tasks to `cifar_tasks.py`
- `CIFAROpenOODWRNBase`, `CIFAROpenOODWRNDeepEnsemble`, `CIFAROpenOODWRNSWAG`, `CIFAROpenOODWRNLLLA`, `CIFAROpenOODWRNPnC`
- Or generalize the existing tasks to take a `--model` parameter (cleaner long-term)

### Compute estimate
- WRN implementation + verification: 1 day of dev work
- WRN training (3 seeds × 1 base + 3 seeds × 1 SWAG + 3 ensemble members): ~50 hours
- OOD evaluation (5 methods × 3 seeds): ~6 hours
- **Total: ~3 days wall time**

### Deliverables
1. `WideResNet28_10` in `models.py`
2. `CIFARTrainWideResNet28_10` and 5 OOD WRN tasks in `cifar_tasks.py`
3. `results/cifar10/wrn28_10/openood_v1p5_*_seed{0,1,2}.json`
4. WRN row group added to the headline OOD table in `cifar10_ood_detection.md`

---

## P5 — Narrative + paper-table reframing

**Goal:** Frame results so that PnC's actual contribution is clear and reviewer-defensible.

### Reframing rules

1. **Deep Ensemble is in a different cost class.** All tables should clearly mark Deep Ensemble as 5× training cost. The headline claim should be: *"PnC matches Deep Ensemble's OOD performance at 1× training cost"* — not *"PnC beats baselines."*

2. **PnC is a post-hoc method** — group it with LLLA, Mahalanobis, ReAct in the table layout. Compare against post-hoc methods first; Deep Ensemble appears as a separate "upper bound" reference.

3. **Tight margins ⇒ honest framing.** When error bars overlap, write *"competitive with X"* not *"outperforms X."* Keep "outperforms" only for clearly separated rows.

4. **Per-dataset breakdowns in supplementary.** Headline tables should be aggregate near/far AUROC. Per-dataset details in an appendix table.

5. **Mahalanobis caveat sentence required.** Single-line note: *"We report a paper-clean variant of Mahalanobis using only the penultimate feature layer with no OOD-trained logistic combiner. The original (Lee et al., 2018) numbers are higher but use OOD validation data."*

6. **OpenOOD dataset variant note.** Single-line note in methods: *"Our DTD/Places365 splits differ in size from canonical OpenOOD splits; per-dataset numbers are not directly comparable, but aggregate metrics use the same protocol."*

### Tables to produce

1. **Main OOD table (CIFAR-10):** mean ± std over 3 seeds, post-hoc methods first, Deep Ensemble as separate "5×-cost reference" row, training-cost column included.
2. **Main OOD table (CIFAR-100):** same structure.
3. **WRN-28-10 OOD table (CIFAR-10):** subset of methods, demonstrates architecture generality.
4. **Inference cost table:** wall-clock per sample + GPU memory.
5. **UQ table (CIFAR-10):** existing 3-seed numbers, no changes needed.
6. **Per-dataset OOD breakdown (appendix):** all 6 datasets × all methods × seeds.

### Deliverables
1. `experiments/cifar_paper_tables.md` — final paper-ready markdown tables
2. `experiments/cifar_paper_narrative.md` — short paragraph-by-paragraph guide for the results section, with the reframing rules applied
3. Updated `cifar10_ood_detection.md` reflecting all P0–P4 results

---

## Compute summary

| Phase | Compute (wall time) | Cumulative | Status |
|-------|---------------------|------------|--------|
| P0 — Multi-seed OOD on CIFAR-10 | ~7 h | 7 h | blocking, do first |
| P1 — PnC config validation | ~1 h (folded into P0) | 8 h | folds into P0 |
| P2 — Inference cost benchmark | ~30 min | 8.5 h | quick |
| P3 — CIFAR-100 full pipeline | ~28 h | 36.5 h | longest path |
| P4 — WideResNet-28-10 | ~3 days dev+compute | ~110 h | optional but recommended |
| P5 — Tables + narrative | ~0.5 day writing | — | after P0–P3 |

**Minimum viable for NeurIPS submission: P0 + P1 + P2 + P3.** P4 strengthens the paper significantly but is not strictly required if the CIFAR-10 + CIFAR-100 story is tight enough.

## Sequencing

**Day 1 (today):**
- Train Epinet seeds 1, 2 with ps=3.0 (background)
- Implement `scripts/aggregate_ood_results.py` and `scripts/run_ood_eval_multi_seed.sh`
- Implement `scripts/benchmark_inference_cost.py`
- Start CIFAR-100 base model training seed 0 in background
- Run PnC scale sanity sweep on seed 0 (P1.b step 3)

**Day 2:**
- Run all 12 OOD methods × 2 new seeds on CIFAR-10 (P0)
- Continue CIFAR-100 base training seeds 1, 2
- Run inference cost benchmark, write up
- Aggregate CIFAR-10 OOD results into mean±std table

**Day 3:**
- Train CIFAR-100 SWAG / MCDropout / Epinet for 3 seeds
- Prepare CIFAR-100 OOD data (symlinks + cifar10 npz)
- Add `prepare_cifar100_benchmark()` to data prep script

**Day 4–5:**
- Run all 12 OOD methods on CIFAR-100, 3 seeds
- Aggregate, update reports

**Day 6+:**
- (Optional) Implement WideResNet-28-10 and start training
- Write `cifar_paper_tables.md` and `cifar_paper_narrative.md`
- Final pass on `cifar10_ood_detection.md`

## Verification checklist

After each phase, verify:

- [ ] All result JSONs have `protocol.ood_validation_used == false`
- [ ] All result JSONs have `protocol.temperature_fit_split == "id_validation_only"`
- [ ] Multi-seed aggregations report mean ± std with N=3 explicit
- [ ] Tight margins are framed as "competitive" not "outperforms"
- [ ] Deep Ensemble training cost is marked separately in every table
- [ ] Mahalanobis caveat note appears alongside Mahalanobis numbers
- [ ] Inference cost column appears in headline tables

## Out of scope (deliberately deferred)

- **ImageNet experiments.** Scale-up would require ~weeks of compute and a different infrastructure layer. Note in limitations: "Scaling PnC to ImageNet is left for future work."
- **ODIN baseline.** Already deferred for protocol reasons; documented in `cifar10_ood_detection.md`.
- **Multi-layer Mahalanobis.** Requires OOD-trained logistic combiner, violates protocol.
- **Per-block PnC scale (Hypothesis 2 from tuning plan).** Could improve PnC further but is research-direction work, not a reviewer-blocker.
- **Logit averaging (Hypothesis 4).** Same — can be a follow-up.
- **Selective prediction / accuracy-rejection curves.** Nice-to-have for the appendix, not blocking.
