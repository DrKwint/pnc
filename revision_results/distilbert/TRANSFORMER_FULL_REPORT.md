# Post-hoc P&C on a Pretrained Banking77 DistilBERT — Full Report

_Transformer architecture-scope experiment. 5 P&C construction seeds (0, 10, 123, 2026, 42); no model fine-tuning. All artifacts under `results/banking77_distilbert_pnc/`._

## 1. Summary

We apply Perturb-and-Correct post-hoc to an already fine-tuned DistilBERT Banking77 intent classifier, perturbing the first affine map of the final transformer feed-forward block and correcting the second on Banking77 ID activations — no weights retrained. This is a third, qualitatively different model family alongside the submitted MLP dynamics and ResNet-18 experiments (tokenized inputs, self-attention, residual connections, layer normalization, a pretrained backbone, 77-way categorical prediction).

**Headline results:** (1) P&C fully preserves ID accuracy **and** calibration; (2) the affine correction is *essential* — removing it keeps accuracy but destroys calibration (ECE 0.52 vs 0.03); (3) OOD detection is competitive with strong baselines and **invariant to perturbation scale** but **controllable via correction strength**, which exposes a Pareto regime where a moderately weakened correction improves OOD at no ID-calibration cost.

## 2. Model, data, and validation

- **Checkpoint:** `optimum/distilbert-base-uncased-finetuned-banking77` @ `89565753ee34239b4ab70c9c90317085cc0f4f8e` — 67,012,685 float32 params, 77 labels. Converted PyTorch→Flax once (msgpack).
- **PT/Flax parity gate:** max logit diff **7.2e-06**, top-1 agreement 100%, Flax test accuracy **0.925** (= reported 0.925).
- **Banking77:** 10003 train → 9000 calibration / 1003 ID-val (stratified, seed 20260726); 3080 test. Labels matched to the checkpoint's id2label by string (validated). max_length 64 (0.2% truncated).
- **OOD:** CLINC-OOS — Near (banking + credit_cards), Cross-domain (non-financial in-scope), Far (official oos). Taxonomy-based domain mapping validated against dataset intents.
- **Cached-tail parity:** the layer-5 attention + FFN + residual + LayerNorm + head are replicated from the converted weights and reproduce full-model logits to **< 2.4e-6**; each P&C member re-runs only this tail on a once-cached prefix.

## 3. Method

Target block = final DistilBERT FFN (layer 5): `lin1.kernel [768,3072]`, `lin2.kernel [3072,768]`. Perturb `lin1` with a random rank-20 basis, recompute the post-GELU activations, and correct `(lin2.kernel, lin2.bias)` by the **existing shared ridge solver** (`pnc_theory.linalg.ridge_solve`, ridge 1e-3 toward original, full bootstrap) on Banking77 ID activations — reused unchanged (see `REUSE_MAP.md`). M=20 members, K=20 directions. Scale selected by **ID-validation NLL only** (constraints: ID acc drop ≤0.25pp, base agreement ≥99%); OOD never used for tuning. One base temperature fit on ID-val, shared across methods.

## 4. Main comparison (all methods, 5 seeds)

| Method | ID Acc | ID NLL | ID ECE | Near AUROC | Cross AUROC | Far AUROC | Far FPR95 |
|---|---|---|---|---|---|---|---|
| MSP | 0.925 | 0.303 | 0.032 | 0.892 | 0.954 | 0.968 | 0.140 |
| Energy | 0.925 | 0.303 | 0.032 | 0.929 | 0.980 | 0.990 | 0.044 |
| Entropy | 0.925 | 0.303 | 0.032 | 0.903 | 0.964 | 0.978 | 0.117 |
| MC Dropout | 0.925 | 0.283 | 0.013 | 0.917 | 0.969 | 0.982 | 0.104 |
| Uncorrected perturb | 0.921 | 1.079 | 0.525 | 0.885 | 0.972 | 0.989 | 0.033 |
| Head P&C | 0.925 | 0.304 | 0.032 | 0.902 | 0.964 | 0.978 | 0.118 |
| Internal FFN P&C | 0.925 | 0.301 | 0.031 | 0.907 | 0.967 | 0.981 | 0.106 |

ID metrics use temperature T≈0.80 (fit on ID-val). Seed SE ≤0.001 on all cells.

## 5. Findings

**1. ID preservation (accuracy + calibration).** Internal-FFN P&C ID accuracy 0.925 (base 0.925), NLL 0.301, ECE 0.031 — indistinguishable from the base checkpoint. The correction reproduces base predictions even at the selected 4× scale, where the perturbation Frobenius norm is 2× the weight matrix's own norm.

**2. The correction is essential — for calibration, not accuracy.** Uncorrected perturbation keeps accuracy (~0.921, because 20-member averaging masks per-member drift) but destroys calibration: **ECE 0.525 vs 0.031** and **NLL 1.079 vs 0.301**. The affine correction is precisely what turns a wildly miscalibrated perturbed ensemble into a usable probabilistic model.

**3. OOD detection is competitive; Energy leads.** Internal-FFN P&C Far AUROC 0.981 beats MSP/Entropy/Head-P&C, ≈ MC Dropout, below Energy (0.990). On a well-calibrated 92.5%-accurate classifier the simple Energy score is strong. Uncorrected shows high Far AUROC (0.989) but that number is attached to a broken predictive distribution (ECE 0.52) — a good OOD *score* on an unusable model.

## 6. Perturbation-scale sweep (diagnostic)

P&C-only, 3 seeds, fixed ridge 1e-3, powers of 2. `rel` = ‖ΔW₁‖/‖W₁‖.

| scale | rel | ID acc | agree | cal_res | Near AUROC | Far AUROC | ID-ok |
|---|---:|---:|---:|---:|---:|---:|:--:|
| 1× | 0.5 | 0.925 | 1.000 | 0.039 | 0.913 | 0.985 | yes |
| 2× | 1.0 | 0.925 | 0.999 | 0.056 | 0.914 | 0.985 | yes |
| 4× | 2.0 | 0.925 | 0.998 | 0.065 | 0.915 | 0.986 | yes |
| 8× | 4.0 | 0.926 | 0.998 | 0.068 | 0.915 | 0.986 | yes |
| 16× | 8.0 | 0.926 | 0.998 | 0.069 | 0.915 | 0.986 | yes |
| 32× | 16.0 | 0.926 | 0.997 | 0.069 | 0.915 | 0.986 | yes |

**OOD detection is scale-invariant.** Over a 64× range (rel 0.5→16), Near moves 0.913→0.915 and Far 0.985→0.986 (±0.000 across seeds). The correction returns members toward the base on ID/near-ID inputs, so extra perturbation adds no discriminative diversity. The correction absorbs perturbations up to 16× the weight norm with no ID-constraint failure — explaining the main run's boundary scale selection (flat ID-val NLL → the diversity tie-break takes the max).

## 7. Correction-strength (ridge) sweep — the actual lever

P&C-only, 3 seeds, fixed 4× scale, vary λ. λ→0 = full correction; λ→∞ = uncorrected (corrected lin2 → original).

| λ | cal_res | ID acc | ID ECE | agree | Near AUROC | Cross AUROC | Far AUROC | Far FPR95 | ID-ok |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|:--:|
| 1e-03 | 0.065 | 0.925 | 0.019 | 0.998 | 0.915 | 0.973 | 0.986 | 0.070 | yes |
| 1e-01 | 0.065 | 0.925 | 0.019 | 0.998 | 0.915 | 0.973 | 0.986 | 0.070 | yes |
| 1e+01 | 0.071 | 0.926 | 0.019 | 0.999 | 0.914 | 0.972 | 0.985 | 0.073 | yes |
| 1e+03 | 0.299 | 0.926 | 0.020 | 0.998 | 0.924 | 0.978 | 0.989 | 0.046 | yes |
| 1e+05 | 0.907 | 0.924 | 0.099 | 0.997 | 0.925 | 0.979 | 0.991 | 0.035 | yes |
| 1e+07 | 2.104 | 0.922 | 0.566 | 0.983 | 0.899 | 0.978 | 0.991 | 0.026 | no |

**Correction strength — not perturbation scale — controls the ID-calibration ↔ OOD tradeoff.** Weakening the correction lifts Far AUROC 0.986→**0.991**, Near 0.915→**0.925**, Far FPR95 0.070→**0.035**. There is a **Pareto sweet spot at λ≈1e3**: OOD improves (Far 0.989, FPR95 0.046) while calibration is *fully preserved* (ECE 0.020) and ID constraints still pass. At λ=1e5 OOD is best (Far 0.991) at a mild calibration cost (ECE 0.099). Past that (λ=1e7 ≈ uncorrected) it breaks: ECE 0.57, the ID agreement constraint fails, and Near even regresses (0.899). cal_residual climbs monotonically (0.065→2.10) — more member divergence → better OOD ranking, until the divergence also corrupts the ID predictive distribution.

**Protocol caveat.** This sweet spot is a *diagnostic* finding, not a selectable operating point: it was identified using OOD, which the protocol forbids for selection, and it is invisible to ID-only selection (ID-val NLL/ECE are flat from λ=1e-3 to 1e3). Reported honestly: *a moderately weakened correction would improve OOD at no ID-calibration cost, but choosing it requires OOD signal or an a-priori rationale, not the ID-only criterion.*

## 8. Cost & memory

Ran on one TITAN X Pascal (12 GB). Peak GPU ≈3.3 GB (main run) — the 3073×3073 ridge solves are on CPU; per-member evaluation reuses a cached transformer prefix. Members stored compactly (shared basis + coefficients + corrected lin2), not 20 model copies. Full run enqueued behind any running GPU jobs via a CPU-only GPU-idle waiter.

## 9. Limitations

- P&C construction seeds are not independent model-training seeds (one public checkpoint).
- ID-only scale selection hits the grid maximum because ID behavior is scale-invariant; the operating scale is therefore weakly identified (any large scale is admissible).
- The OOD-improving weak-correction regime cannot be selected under the ID-only protocol.
- Energy is the strongest OOD baseline here; P&C's advantage is calibrated predictive uncertainty, not raw OOD ranking.

## 10. Reviewer-facing summary (≤180 words)

We add a third, qualitatively different evaluation using an already fine-tuned DistilBERT Banking77 classifier, with no model retraining. P&C is applied inside the final transformer feed-forward block by perturbing its first affine map and correcting the second using Banking77 ID activations, tested under tokenized inputs, self-attention, residual connections, layer normalization, a pretrained backbone, and 77-way prediction. The corrected ensemble preserves in-distribution accuracy (92.5%) and calibration (ECE 0.03), whereas uncorrected perturbations keep accuracy but destroy calibration (ECE 0.52) — confirming affine correction remains necessary in the transformer setting. On CLINC financial Near-OOD and official OOS Far-OOD, internal-FFN P&C detects OOD competitively with MSP, Energy, and MC Dropout. OOD quality is invariant to perturbation scale but controllable by correction strength, revealing a regime where a weaker correction improves OOD at no calibration cost. Together with the MLP dynamics and ResNet-18 experiments, this shows applicability across three substantially different model families, while we narrow the paper's language to avoid claiming exhaustive architectural coverage.

## 11. Artifacts

- Main: `metrics/raw.csv` (175 rows), `tables/banking77_pnc.{md,tex,csv}`, `predictions/<method>/<seed>/<split>.parquet`, `members/seed_<s>/`.
- Scale sweep: `scale_sweep/scale_sweep_raw.csv` (18 rows), `SCALE_SWEEP_FINDINGS.md`.
- Ridge sweep: `ridge_sweep/ridge_sweep_raw.csv` (18 rows).
- Checkpoint/parity: `checkpoint/{checkpoint_manifest,parity}.json`. Reuse map: `REUSE_MAP.md`.
- Code: `experiments/banking77_pnc/`. Preflight gate: `validate.py --stage preflight` (passed).
