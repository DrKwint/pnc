# CIFAR-10 inference & temperature-scaling protocol (verified from code)

Source: `openood_eval.py`, `util.py` (repo DrKwint/pnc, commit 854f4d8).

## Temperature scaling
- **One scalar temperature per model/configuration**, fit on the **ID-validation split only** (the 5,000-example split; `calibration_data=(x_va,y_va)`). `openood_eval.py:115–121` → `temperature = _fit_posthoc_temperature(cal_logits, cal_targets)`; when `posthoc_calibrate=False`, `temperature = 1.0`.
- **Fitting objective** (`util._fit_posthoc_temperature`, `util.py:142–190`): golden-section search over log-temperature ∈ [log 1e-2, log 1e2], 40 iters, minimizing mean ensemble NLL `−mean log( mean_s softmax(logits/T)[y] )`. Returns 1.0 if non-finite. Recorded as `id_metrics.posthoc_temperature`.
- **The same scalar T is applied to every ensemble member's logits** before softmax: `scaled = logits_np / T` (`openood_eval.py:21`), for the full `(S, N, C)` member-logit tensor.

## Predictive distribution & scores
- **Member probabilities are averaged (mixture):** `mean_probs = softmax(logits/T, axis=-1).mean(axis=0)` over the S members (`openood_eval.py` `_uncertainty_scores_from_logits`). Temperature is applied to logits **before** softmax and **before** averaging.
- **Predictive entropy** (primary OOD score): `predictive_entropy = −Σ_c mean_probs_c · log(mean_probs_c + 1e-8)` (`openood_eval.py:29`).
- Other scores: `max_softmax_uncertainty = 1 − max mean_probs` (:30); `energy = mean_s (−T · logsumexp(logits_s/T))` (:32–39); `mutual_information = predictive_entropy − mean_s entropy(softmax(logits_s/T))` (:55, ensemble only).
- **OOD score sign convention:** higher score = more OOD; ID is the negative class (label 0), OOD positive (label 1). `openood_eval.py` `_binary_ood_metrics:62–72`.
- **AUROC/AUPR/FPR95:** sklearn `roc_auc_score`, `average_precision_score`, `roc_curve`; **FPR95 = FPR at the first threshold where TPR ≥ 0.95** (`_binary_ood_metrics:67–71`).
- **NLL** (ID): `−mean log( mean_probs[arange, y] )`.
- **Near/Far macro aggregate:** per-dataset metric then arithmetic mean over datasets (`_aggregate_family_metrics:75–95`, `mean_auroc`/`mean_fpr95`). Top-level `near_ood_auroc`/`far_ood_auroc` = the macro mean under each method's `primary_score` (predictive_entropy for base/P&C/SCOD-adjacent methods).

## Per-method primary score (verified from JSON `protocol.primary_score`)
MSP → `max_softmax_uncertainty`; Energy → `energy_score`; Mahalanobis → `mahalanobis`; PreAct/ReAct/LLLA/Epinet/MC-Dropout/SWAG/P&C/Deep-Ensemble → `predictive_entropy`. (This is why MSP/Energy/Maha AUROCs differ from the base row while FPR95 — taken from the shared predictive_entropy aggregate — coincide.)

## OpenOOD tiers (verified from every method's `per_dataset` keys, 12 methods × 3 seeds)
- **Near = {cifar100, tiny_imagenet}**; **Far = {mnist, svhn, textures, places365}**; `id_dataset = cifar10`; `uses_ood_validation = false`. Exactly as specified.
