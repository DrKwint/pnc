import numpy as np

from cifar_tasks import CIFARSelectPnCPaperProtocol, _select_best_pnc_candidate
from util import _make_cifar_protocol_splits


class _ExplodingMapping(dict):
    def __getitem__(self, key):
        raise AssertionError(f"Selection helper should not read test-side key {key!r}")


def test_make_cifar_protocol_splits_is_deterministic_and_disjoint():
    x_train = np.arange(20)
    y_train = np.arange(20)
    x_test = np.arange(100, 106)
    y_test = np.arange(6)

    splits_a = _make_cifar_protocol_splits(
        x_train,
        y_train,
        x_test,
        y_test,
        val_model_select_split=0.2,
        train_bn_refresh_split=0.25,
        seed=7,
    )
    splits_b = _make_cifar_protocol_splits(
        x_train,
        y_train,
        x_test,
        y_test,
        val_model_select_split=0.2,
        train_bn_refresh_split=0.25,
        seed=7,
    )

    fit_a = set(np.asarray(splits_a["train_fit"][0]).tolist())
    bn_a = set(np.asarray(splits_a["train_bn_refresh"][0]).tolist())
    val_a = set(np.asarray(splits_a["val_model_select"][0]).tolist())

    assert np.array_equal(splits_a["train_fit"][0], splits_b["train_fit"][0])
    assert np.array_equal(splits_a["train_bn_refresh"][0], splits_b["train_bn_refresh"][0])
    assert np.array_equal(splits_a["val_model_select"][0], splits_b["val_model_select"][0])
    assert fit_a.isdisjoint(bn_a)
    assert fit_a.isdisjoint(val_a)
    assert bn_a.isdisjoint(val_a)
    assert splits_a["metadata"]["n_train_fit"] + splits_a["metadata"]["n_train_bn_refresh"] + splits_a["metadata"]["n_val_model_select"] == 20


def test_pnc_selection_uses_validation_metrics_only():
    candidate_results = [
        {
            "candidate_id": 0,
            "config": {"name": "fallback"},
            "val_metrics": {"accuracy": 0.80, "nll": 0.30, "ece": 0.05, "brier": 0.12},
            "test_metrics": _ExplodingMapping(),
        },
        {
            "candidate_id": 1,
            "config": {"name": "selected"},
            "val_metrics": {"accuracy": 0.89, "nll": 0.20, "ece": 0.03, "brier": 0.10},
            "test_metrics": _ExplodingMapping(),
        },
    ]

    selected = _select_best_pnc_candidate(
        candidate_results,
        selection_metric="nll_within_acc_drop",
        reference_accuracy=0.90,
        max_validation_accuracy_drop=0.02,
    )

    assert selected["selected_candidate"]["candidate_id"] == 1
    assert selected["selection_summary"]["selection_criterion"] == "best_validation_nll_subject_to_accuracy_drop"


def test_pnc_protocol_task_output_path_includes_selection_metadata():
    task = CIFARSelectPnCPaperProtocol(
        selection_metric="ece",
        val_model_select_split=0.15,
        train_bn_refresh_split=0.2,
        candidate_block_modes=["single"],
        candidate_direction_methods=["lanczos"],
        candidate_n_directions=[8],
        candidate_n_perturbations=[16],
        candidate_perturbation_sizes=[25.0],
        candidate_lambda_regs=[1e-3],
    )

    assert "_selece_" in task.output().path
    assert "_vsplit0p15_" in task.output().path
    assert "_bnsplit0p2_" in task.output().path
