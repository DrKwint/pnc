import time
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

from util import _evaluate_cifar_logits, _fit_posthoc_temperature, _predict_cifar_logits


def _uncertainty_scores_from_logits(logits: jax.Array, temperature: float) -> dict[str, np.ndarray]:
    probs = jax.nn.softmax(logits / temperature, axis=-1)  # (S, N, C)
    mean_probs = jnp.mean(probs, axis=0)
    predictive_entropy = -jnp.sum(mean_probs * jnp.log(mean_probs + 1e-8), axis=-1)
    max_softmax_uncertainty = 1.0 - jnp.max(mean_probs, axis=-1)

    # Energy score: mean per-member negative logsumexp (larger = more OOD)
    energy_per_member = -temperature * jax.scipy.special.logsumexp(
        logits / temperature, axis=-1
    )  # (S, N)
    energy_score = jnp.mean(energy_per_member, axis=0)  # (N,)

    # Margin uncertainty: 1 - (top1_prob - top2_prob) on mean probs
    top2 = jax.lax.top_k(mean_probs, 2)[0]  # (N, 2)
    margin_uncertainty = 1.0 - (top2[:, 0] - top2[:, 1])

    scores = {
        "predictive_entropy": np.asarray(predictive_entropy),
        "max_softmax_uncertainty": np.asarray(max_softmax_uncertainty),
        "energy_score": np.asarray(energy_score),
        "margin_uncertainty": np.asarray(margin_uncertainty),
    }
    if probs.shape[0] > 1:
        sample_entropy = -jnp.sum(probs * jnp.log(probs + 1e-8), axis=-1)
        mutual_information = predictive_entropy - jnp.mean(sample_entropy, axis=0)
        variation_ratio = 1.0 - jnp.max(mean_probs, axis=-1)
        scores["mutual_information"] = np.asarray(mutual_information)
        scores["variation_ratio"] = np.asarray(variation_ratio)
    return scores


def _binary_ood_metrics(id_scores: np.ndarray, ood_scores: np.ndarray) -> dict[str, float]:
    labels = np.concatenate([np.zeros(len(id_scores)), np.ones(len(ood_scores))])
    scores = np.concatenate([id_scores, ood_scores])
    auroc = float(roc_auc_score(labels, scores))
    aupr = float(average_precision_score(labels, scores))
    fpr, tpr, _ = roc_curve(labels, scores)
    fpr95 = 1.0
    meets = np.where(tpr >= 0.95)[0]
    if len(meets) > 0:
        fpr95 = float(fpr[meets[0]])
    return {"auroc": auroc, "aupr": aupr, "fpr95": fpr95}


def _aggregate_family_metrics(
    id_scores_by_name: dict[str, np.ndarray],
    family_scores: dict[str, dict[str, np.ndarray]],
) -> dict[str, dict[str, float]]:
    aggregated = {}
    for score_name in next(iter(family_scores.values())).keys():
        per_dataset = []
        concat_id = []
        concat_ood = []
        for dataset_name, score_map in family_scores.items():
            metrics = _binary_ood_metrics(id_scores_by_name[score_name], score_map[score_name])
            per_dataset.append(metrics)
            concat_id.append(id_scores_by_name[score_name])
            concat_ood.append(score_map[score_name])
        aggregated[score_name] = {
            "mean_auroc": float(np.mean([m["auroc"] for m in per_dataset])),
            "mean_aupr": float(np.mean([m["aupr"] for m in per_dataset])),
            "mean_fpr95": float(np.mean([m["fpr95"] for m in per_dataset])),
            "concat_auroc": float(_binary_ood_metrics(np.concatenate(concat_id), np.concatenate(concat_ood))["auroc"]),
        }
    return aggregated


def evaluate_openood_cifar(
    ensemble_name: str,
    ensemble: Any,
    benchmark: dict[str, object],
    *,
    calibration_data: tuple[np.ndarray, np.ndarray] | None = None,
    posthoc_calibrate: bool = False,
    batch_size: int = 256,
    primary_score: str = "predictive_entropy",
) -> dict[str, object]:
    """Evaluate a CIFAR ensemble on OpenOOD v1.5 without using any OOD validation split."""
    id_test = benchmark["id_test"]
    x_id = id_test["inputs"]
    y_id = id_test["targets"]

    _ = ensemble.predict(x_id[:1])

    temperature = 1.0
    if posthoc_calibrate and calibration_data is not None:
        cal_inputs, cal_targets = calibration_data
        cal_logits = _predict_cifar_logits(ensemble, cal_inputs, batch_size=batch_size)
        jax.block_until_ready(cal_logits)
        temperature = _fit_posthoc_temperature(cal_logits, cal_targets)

    t0 = time.time()
    id_logits = _predict_cifar_logits(ensemble, x_id, batch_size=batch_size)
    jax.block_until_ready(id_logits)
    id_eval_time = time.time() - t0

    id_metrics = _evaluate_cifar_logits(
        f"{ensemble_name} ID Test",
        id_logits,
        y_id,
        n_classes=int(id_logits.shape[-1]),
        temperature=temperature,
        eval_time=id_eval_time,
    )
    id_metrics["posthoc_temperature"] = id_metrics.pop("temperature")

    id_scores = _uncertainty_scores_from_logits(id_logits, temperature=temperature)
    near_results = {}
    far_results = {}
    near_family_scores = {}
    far_family_scores = {}

    for family_name, container, score_cache in (
        ("near_ood", near_results, near_family_scores),
        ("far_ood", far_results, far_family_scores),
    ):
        family = benchmark[family_name]
        for dataset_key, dataset in family.items():
            x_ood = dataset["inputs"]
            t1 = time.time()
            ood_logits = _predict_cifar_logits(ensemble, x_ood, batch_size=batch_size)
            jax.block_until_ready(ood_logits)
            eval_time = time.time() - t1
            ood_scores = _uncertainty_scores_from_logits(ood_logits, temperature=temperature)
            score_cache[dataset_key] = ood_scores
            score_metrics = {
                score_name: _binary_ood_metrics(id_scores[score_name], ood_scores[score_name])
                for score_name in id_scores.keys()
                if score_name in ood_scores
            }
            container[dataset_key] = {
                "name": dataset["name"],
                "n_examples": int(len(x_ood)),
                "eval_time": float(eval_time),
                "scores": score_metrics,
            }

    near_agg = _aggregate_family_metrics(id_scores, near_family_scores) if near_results else {}
    far_agg = _aggregate_family_metrics(id_scores, far_family_scores) if far_results else {}

    return {
        "benchmark": "openood_v1_5",
        "ensemble_name": ensemble_name,
        "benchmark_metadata": benchmark.get("metadata", {}),
        "posthoc_temperature": float(temperature),
        "posthoc_calibrate": bool(posthoc_calibrate),
        "protocol": {
            "ood_validation_used": False,
            "ood_tuning_used": False,
            "temperature_fit_split": "id_validation_only" if posthoc_calibrate else "disabled",
            "primary_score": primary_score,
        },
        "id_metrics": id_metrics,
        "near_ood": {
            "per_dataset": near_results,
            "aggregate": near_agg,
        },
        "far_ood": {
            "per_dataset": far_results,
            "aggregate": far_agg,
        },
        "near_ood_auroc": float(near_agg.get(primary_score, {}).get("mean_auroc", float("nan"))),
        "far_ood_auroc": float(far_agg.get(primary_score, {}).get("mean_auroc", float("nan"))),
    }


# ---------------------------------------------------------------------------
# Feature-level helpers (Mahalanobis, ReAct)
# ---------------------------------------------------------------------------


def _extract_features_batched(
    model: Any, inputs: np.ndarray, batch_size: int = 256
) -> np.ndarray:
    """Extract penultimate (512-d) features in mini-batches. Returns (N, D) numpy array."""
    feats = []
    for start in range(0, len(inputs), batch_size):
        x_batch = jnp.asarray(inputs[start : start + batch_size])
        f = model.features(x_batch, use_running_average=True)
        feats.append(np.asarray(f))
    return np.concatenate(feats, axis=0)


def _fit_mahalanobis(
    features: np.ndarray, targets: np.ndarray, n_classes: int
) -> tuple[np.ndarray, np.ndarray]:
    """Fit per-class means and shared precision on ID train features.

    Returns (class_means (C, D), precision (D, D)).
    """
    D = features.shape[1]
    class_means = np.zeros((n_classes, D), dtype=np.float64)
    for c in range(n_classes):
        mask = targets == c
        if mask.sum() > 0:
            class_means[c] = features[mask].astype(np.float64).mean(axis=0)
    centered = features.astype(np.float64) - class_means[targets]
    cov = (centered.T @ centered) / len(features) + 1e-6 * np.eye(D)
    precision = np.linalg.inv(cov)
    return class_means, precision


def _mahalanobis_scores(
    features: np.ndarray,
    class_means: np.ndarray,
    precision: np.ndarray,
) -> np.ndarray:
    """Mahalanobis OOD score. Larger = more OOD.

    Returns min squared Mahalanobis distance across classes for each sample.
    """
    feats64 = features.astype(np.float64)
    diffs = feats64[:, None, :] - class_means[None, :, :]  # (N, C, D)
    mahal_sq = np.einsum("ncd,de,nce->nc", diffs, precision, diffs)  # (N, C)
    return np.min(mahal_sq, axis=1)  # (N,)


def evaluate_openood_cifar_mahalanobis(
    method_name: str,
    model: Any,
    benchmark: dict[str, object],
    *,
    calibration_data: tuple[np.ndarray, np.ndarray] | None = None,
    posthoc_calibrate: bool = False,
    batch_size: int = 256,
) -> dict[str, object]:
    """Evaluate Mahalanobis OOD detection on top of logit-based evaluation.

    Fits Mahalanobis detector on ID train features only.  Injects 'mahalanobis'
    scores into the standard result dict alongside logit-based scores.
    """

    # 1. Standard logit-based evaluation via existing path
    class _SingleEns:
        def __init__(self, m):
            self.m = m

        def predict(self, x):
            return jnp.expand_dims(self.m(x, use_running_average=True), axis=0)

    base_results = evaluate_openood_cifar(
        method_name,
        _SingleEns(model),
        benchmark,
        calibration_data=calibration_data,
        posthoc_calibrate=posthoc_calibrate,
        batch_size=batch_size,
        primary_score="predictive_entropy",  # placeholder
    )

    # 2. Fit Mahalanobis on ID train
    id_train = benchmark["id_train"]
    n_classes = int(model.fc.kernel.shape[-1])
    train_feats = _extract_features_batched(model, id_train["inputs"], batch_size)
    class_means, precision = _fit_mahalanobis(
        train_feats, np.asarray(id_train["targets"]), n_classes
    )

    # 3. Score ID test
    id_test_feats = _extract_features_batched(
        model, benchmark["id_test"]["inputs"], batch_size
    )
    id_mahal = _mahalanobis_scores(id_test_feats, class_means, precision)

    # 4. Score each OOD dataset and inject into results
    for family_name in ("near_ood", "far_ood"):
        family = benchmark[family_name]
        family_ood_scores = {}
        for dataset_key, dataset in family.items():
            ood_feats = _extract_features_batched(model, dataset["inputs"], batch_size)
            ood_mahal = _mahalanobis_scores(ood_feats, class_means, precision)
            family_ood_scores[dataset_key] = ood_mahal
            metrics = _binary_ood_metrics(id_mahal, ood_mahal)
            base_results[family_name]["per_dataset"][dataset_key]["scores"][
                "mahalanobis"
            ] = metrics

        # 5. Aggregate Mahalanobis family metrics
        per_dataset_metrics = []
        concat_id_all = []
        concat_ood_all = []
        for dataset_key in family.keys():
            m = _binary_ood_metrics(id_mahal, family_ood_scores[dataset_key])
            per_dataset_metrics.append(m)
            concat_id_all.append(id_mahal)
            concat_ood_all.append(family_ood_scores[dataset_key])
        agg = {
            "mean_auroc": float(np.mean([m["auroc"] for m in per_dataset_metrics])),
            "mean_aupr": float(np.mean([m["aupr"] for m in per_dataset_metrics])),
            "mean_fpr95": float(np.mean([m["fpr95"] for m in per_dataset_metrics])),
            "concat_auroc": float(
                _binary_ood_metrics(
                    np.concatenate(concat_id_all), np.concatenate(concat_ood_all)
                )["auroc"]
            ),
        }
        base_results[family_name]["aggregate"]["mahalanobis"] = agg

    # 6. Override primary score and top-level AUROC keys
    base_results["protocol"]["primary_score"] = "mahalanobis"
    near_mahal = base_results["near_ood"]["aggregate"].get("mahalanobis", {})
    far_mahal = base_results["far_ood"]["aggregate"].get("mahalanobis", {})
    base_results["near_ood_auroc"] = float(near_mahal.get("mean_auroc", float("nan")))
    base_results["far_ood_auroc"] = float(far_mahal.get("mean_auroc", float("nan")))

    return base_results
