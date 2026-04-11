from pathlib import Path

import jax.numpy as jnp
import numpy as np

import data as data_module
from data import load_openood_cifar_benchmark
from openood_eval import evaluate_openood_cifar


class _DummyEnsemble:
    def predict(self, x):
        x_np = np.asarray(x, dtype=np.float32)
        mean_signal = x_np.reshape(len(x_np), -1).mean(axis=1)
        logits = np.stack([mean_signal, -mean_signal], axis=-1).astype(np.float32)
        return jnp.expand_dims(jnp.asarray(logits), axis=0)


class _ExplodingMapping(dict):
    def __getitem__(self, key):
        raise AssertionError(f"Should not read OOD validation key {key!r}")


def _tiny_images(n: int) -> np.ndarray:
    base = np.linspace(0, 255, num=n * 32 * 32 * 3, dtype=np.float32)
    return base.reshape(n, 32, 32, 3)


def test_load_openood_cifar_benchmark_from_npz_layout(tmp_path: Path):
    root = tmp_path / "openood"
    for family, datasets in {
        "near_ood": ["cifar100", "tiny_imagenet"],
        "far_ood": ["mnist", "svhn", "textures", "places365"],
    }.items():
        family_dir = root / "cifar10" / family
        family_dir.mkdir(parents=True, exist_ok=True)
        for idx, name in enumerate(datasets):
            np.savez(
                family_dir / f"{name}.npz",
                images=_tiny_images(3) + idx,
                labels=np.full((3,), idx, dtype=np.int32),
            )

    x_train = _tiny_images(6)
    y_train = np.arange(6, dtype=np.int32) % 10
    x_test = _tiny_images(4)
    y_test = np.arange(4, dtype=np.int32) % 10

    original_loader = data_module.load_cifar10
    data_module.load_cifar10 = lambda normalize=True: (x_train, y_train, x_test, y_test)
    try:
        benchmark = load_openood_cifar_benchmark("cifar10", root_dir=str(root), max_examples_per_dataset=2)
    finally:
        data_module.load_cifar10 = original_loader

    assert benchmark["metadata"]["uses_ood_validation"] is False
    assert set(benchmark["near_ood"].keys()) == {"cifar100", "tiny_imagenet"}
    assert set(benchmark["far_ood"].keys()) == {"mnist", "svhn", "textures", "places365"}
    assert benchmark["id_test"]["inputs"].shape[0] == 2
    assert benchmark["near_ood"]["cifar100"]["inputs"].shape[0] == 2


def test_evaluate_openood_cifar_uses_id_validation_only_and_reports_family_aurocs():
    ensemble = _DummyEnsemble()
    benchmark = {
        "id_test": {
            "name": "cifar10",
            "inputs": _tiny_images(4) / 255.0,
            "targets": np.array([0, 1, 0, 1], dtype=np.int32),
        },
        "near_ood": {
            "cifar100": {"name": "CIFAR-100", "inputs": _tiny_images(3) / 255.0, "targets": np.full((3,), -1, dtype=np.int32)},
        },
        "far_ood": {
            "mnist": {"name": "MNIST", "inputs": np.flip(_tiny_images(3), axis=-1) / 255.0, "targets": np.full((3,), -1, dtype=np.int32)},
        },
        "ood_validation": _ExplodingMapping(),
    }
    x_val = _tiny_images(3) / 255.0
    y_val = np.array([0, 1, 0], dtype=np.int32)

    results = evaluate_openood_cifar(
        "dummy",
        ensemble,
        benchmark,
        calibration_data=(x_val, y_val),
        posthoc_calibrate=True,
        batch_size=2,
    )

    assert "near_ood_auroc" in results
    assert "far_ood_auroc" in results
    assert "nll" in results["id_metrics"]
    assert "ece" in results["id_metrics"]
    assert results["protocol"]["ood_validation_used"] is False
    assert results["protocol"]["temperature_fit_split"] == "id_validation_only"
    assert results["posthoc_temperature"] == results["id_metrics"]["posthoc_temperature"]
    assert "cifar100" in results["near_ood"]["per_dataset"]
    assert "mnist" in results["far_ood"]["per_dataset"]
