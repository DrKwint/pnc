"""OOD metrics and Near/Far aggregation, matching the repo's existing OpenOOD evaluator.

``pnc_core/openood_eval.py`` is the convention of record for this project's OpenOOD
results, but it imports jax at module level and therefore cannot be loaded in the
torch-only environment this experiment runs in. The two relevant functions are reproduced
here verbatim in pure numpy/sklearn, and ``full_validate.py`` gates them against the
originals on random inputs (run under the main ``.venv``, which has jax).

Conventions preserved exactly:

  * higher score = more OOD
  * AUPR with **OOD as the positive class** (``aupr``), plus the ID-positive direction
    (``aupr_in``) which the spec's table asks for
  * FPR95 read off ``roc_curve`` at the first point with TPR >= 0.95 (not a quantile)
  * a family (Near / Far) aggregates as the **macro mean over its datasets**, with the
    pooled ``concat_auroc`` also reported
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve


def binary_ood_metrics(id_scores: np.ndarray, ood_scores: np.ndarray) -> dict[str, float]:
    """Verbatim ``pnc_core.openood_eval._binary_ood_metrics``, plus AUPR-IN and counts."""
    id_scores = np.asarray(id_scores, dtype=np.float64)
    ood_scores = np.asarray(ood_scores, dtype=np.float64)
    labels = np.concatenate([np.zeros(len(id_scores)), np.ones(len(ood_scores))])
    scores = np.concatenate([id_scores, ood_scores])
    auroc = float(roc_auc_score(labels, scores))
    aupr = float(average_precision_score(labels, scores))
    fpr, tpr, _ = roc_curve(labels, scores)
    fpr95 = 1.0
    meets = np.where(tpr >= 0.95)[0]
    if len(meets) > 0:
        fpr95 = float(fpr[meets[0]])
    return {"auroc": auroc, "aupr": aupr, "fpr95": fpr95,
            "aupr_in": float(average_precision_score(1 - labels, -scores)),
            "aupr_out": aupr,
            "n_id": int(len(id_scores)), "n_ood": int(len(ood_scores))}


def aggregate_family_metrics(id_scores: np.ndarray,
                             family_scores: dict[str, np.ndarray]) -> dict[str, float]:
    """Verbatim ``pnc_core.openood_eval._aggregate_family_metrics`` for a single score."""
    per_dataset, concat_id, concat_ood = [], [], []
    for _, ood in family_scores.items():
        per_dataset.append(binary_ood_metrics(id_scores, ood))
        concat_id.append(id_scores)
        concat_ood.append(ood)
    return {
        "mean_auroc": float(np.mean([m["auroc"] for m in per_dataset])),
        "mean_aupr": float(np.mean([m["aupr"] for m in per_dataset])),
        "mean_fpr95": float(np.mean([m["fpr95"] for m in per_dataset])),
        "mean_aupr_in": float(np.mean([m["aupr_in"] for m in per_dataset])),
        "concat_auroc": float(binary_ood_metrics(np.concatenate(concat_id),
                                                 np.concatenate(concat_ood))["auroc"]),
        "n_datasets": len(per_dataset),
    }
