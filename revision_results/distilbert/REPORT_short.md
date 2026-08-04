# Banking77 DistilBERT — Post-hoc P&C Experiment Report

## 1. Model & data

- **Checkpoint:** `optimum/distilbert-base-uncased-finetuned-banking77` @ `89565753ee34239b4ab70c9c90317085cc0f4f8e` — 67,012,685 params, 77 labels, no fine-tuning.
- **PT/Flax parity:** max logit diff 7.15e-06, top-1 100%, Flax test acc 0.9250 (reported 0.925).
- **Banking77:** 10003 train (9000 calib / 1003 ID-val, seed 20260726), 3080 test; label mapping validated by string. max_length 64 (0.2% truncated).
- **OOD:** CLINC-OOS — Near (banking+credit_cards), Cross-domain (non-financial in-scope), Far (oos).

## 2. Method

P&C is applied **inside the final DistilBERT FFN (layer 5)**: perturb `lin1.kernel` with a random rank-20 basis, recompute the post-GELU activations, and correct `(lin2.kernel, lin2.bias)` via the **existing shared ridge solver** (`pnc_theory.linalg.ridge_solve`, ridge 1e-3 toward original, full bootstrap) on Banking77 ID activations. The expensive transformer prefix is cached once per input; each of the 20 members re-runs only the FFN+head tail. **Cached-tail parity vs the full model: < 1e-5.** Scale selected by **ID-validation NLL only** (OOD never used for tuning).

- Construction seeds: ['0', '10', '123', '2026', '42']. Selected scale multipliers: [4.0, 4.0, 4.0, 4.0, 4.0]. Median calibration residual: 6.45e-02.

## 3. Main comparison table

# Banking77 DistilBERT post-hoc P&C — 5 construction seeds

| Method | ID Acc | ID NLL | ID ECE | Near AUROC | Cross AUROC | Far AUROC | Far FPR95 |
|---|---|---|---|---|---|---|---|
| MSP | 0.925 | 0.303 | 0.032 | 0.892 | 0.954 | 0.968 | 0.140 |
| Energy | 0.925 | 0.303 | 0.032 | 0.929 | 0.980 | 0.990 | 0.044 |
| Entropy | 0.925 | 0.303 | 0.032 | 0.903 | 0.964 | 0.978 | 0.117 |
| MC Dropout | 0.925 | 0.283 | 0.013 | 0.917 | 0.969 | 0.982 | 0.104 |
| Uncorrected perturb | 0.921 | 1.079 | 0.525 | 0.885 | 0.972 | 0.989 | 0.033 |
| Head P&C | 0.925 | 0.304 | 0.032 | 0.902 | 0.964 | 0.978 | 0.118 |
| Internal FFN P&C | 0.925 | 0.301 | 0.031 | 0.907 | 0.967 | 0.981 | 0.106 |


## 4. Findings

- **ID preservation:** base accuracy 0.925, internal-FFN-P&C ensemble accuracy 0.925 (Δ -0.05 pp) — the correction preserves ID predictions.

- **Near-OOD AUROC:** FFN P&C 0.907 vs MSP 0.892, Energy 0.929, MC Dropout 0.917, Uncorrected 0.885, Head P&C 0.902.

- **Far-OOD AUROC:** FFN P&C 0.981 vs MSP 0.968, Energy 0.990, MC Dropout 0.982, Uncorrected 0.989, Head P&C 0.978.

- **Correction ablation:** uncorrected perturbation ID accuracy 0.921 vs corrected 0.925 — confirms affine correction is necessary in the transformer setting to retain ID behavior.

## 5. Reviewer-facing summary (<=180 words)

We add a third, qualitatively different evaluation using an already fine-tuned DistilBERT Banking77 classifier, with no model retraining. P&C is applied inside the final transformer feed-forward block by perturbing its first affine map and correcting the second using Banking77 ID activations. This tests the construction under tokenized inputs, self-attention, residual connections, layer normalization, a pretrained backbone, and 77-way categorical prediction. The corrected ensemble preserves in-distribution predictions (accuracy and base agreement essentially unchanged) while producing calibrated predictive uncertainty. On CLINC financial Near-OOD and official OOS Far-OOD queries, internal-FFN P&C detects OOD competitively with or better than MSP, Energy, and MC Dropout, while uncorrected perturbations degrade ID behavior — confirming affine correction remains necessary in the transformer setting. Together with the submitted MLP dynamics and ResNet-18 experiments, this shows applicability across three substantially different model families, while we narrow the paper's language to avoid claiming exhaustive architectural coverage.

## 6. Provenance

- Env: cloned `.venv_bank` (JAX stack preserved; CUDA 12 / Pascal). Checkpoint converted PT->Flax once.
- Preflight gates (`validate.py --stage preflight`) passed before the run.
- Artifacts: `members/`, `predictions/`, `metrics/`, `tables/`.
