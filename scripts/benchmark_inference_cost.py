#!/usr/bin/env python3
"""Benchmark wall-clock inference cost for each CIFAR-10 OOD method.

For every method we report:
  - cold_warmup_s     : time of first batch (includes JIT)
  - warm_per_sample_ms: time per sample after warm-up, averaged over 5000 samples
  - n_forward_passes  : effective number of forward passes per sample
  - train_cost_factor : training compute multiplier vs single model

Saves results to results/cifar10/inference_cost.json and prints a markdown table.
Uses seed 0 checkpoints. Run once; results don't depend on seed.
"""

from __future__ import annotations

import json
import os
import pickle
import sys
import time
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["XLA_FLAGS"] = "--xla_gpu_strict_conv_algorithm_picker=false"

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

# Make repo root importable when run from anywhere
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cifar_tasks import (  # noqa: E402
    _build_multi_block_pnc_ensemble,
    _build_single_block_pnc_ensemble,
)
from data import load_cifar10  # noqa: E402
from ensembles import (  # noqa: E402
    EpinetEnsemble,
    EpinetWithPrior,
    LLLAEnsemble,
    SWAGEnsemble,
)
from models import MCDropoutPreActResNet18, PreActResNet18  # noqa: E402
from openood_eval import _extract_features_batched  # noqa: E402
from util import _predict_cifar_logits  # noqa: E402

# ----------- Configuration -----------
RECIPE = "_e300_optsgd_lr1e-01_wd5e-04_bs128_wu5_mom0p9_n1_augfcco8_ls0"
CKPT_BASE = f"results/cifar10/preact_resnet18_train{RECIPE}_seed0.pkl"
CKPT_MCDROPOUT = f"results/cifar10/preact_resnet18_mcdropout_train_dr0.1{RECIPE}_seed0.pkl"
CKPT_SWAG = f"results/cifar10/preact_resnet18_swag_train{RECIPE}_sws240_swf1_swr20_seed0.pkl"
CKPT_EPINET_PS3 = f"results/cifar10/epinet_train_epi100_idim8_h50,50_ps3.0{RECIPE}_seed0.pkl"

N_BENCH_SAMPLES = 5000
BATCH_SIZE = 256
SEED = 0
N_CLASSES = 10


def _load_state(model, path: str) -> None:
    with open(path, "rb") as f:
        ckpt = pickle.load(f)
    nnx.update(model, ckpt["state"])


def _time_predict(ensemble, x: np.ndarray, batch_size: int) -> tuple[float, float]:
    """Returns (cold_warmup_s, warm_total_s) where warm_total covers all samples."""
    # Cold warmup with one batch
    t0 = time.time()
    out = ensemble.predict(x[:batch_size])
    if hasattr(out, "block_until_ready"):
        out.block_until_ready()
    cold = time.time() - t0

    # Warm timing over the full benchmark set
    t0 = time.time()
    logits = _predict_cifar_logits(ensemble, x, batch_size=batch_size)
    if hasattr(logits, "block_until_ready"):
        logits.block_until_ready()
    warm = time.time() - t0
    return cold, warm


def _make_single_ens(model):
    class _Single:
        def __init__(self, m):
            self.m = m

        def predict(self, x):
            return jnp.expand_dims(self.m(x, use_running_average=True), axis=0)

    return _Single(model)


def _make_react_ens(model, threshold):
    fc = model.fc

    class _ReAct:
        def __init__(self, m, t, fc_):
            self.m = m
            self.t = t
            self.fc = fc_

        def predict(self, x):
            feats = self.m.features(x, use_running_average=True)
            feats = jnp.clip(feats, a_max=self.t)
            logits = self.fc(feats)
            return jnp.expand_dims(logits, axis=0)

    return _ReAct(model, threshold, fc)


def _make_mc_dropout_ens(model, n):
    class _MCEns:
        def __init__(self, m, n_):
            self.m = m
            self.n_models = n_

        def predict(self, x):
            outs = [
                self.m(x, use_running_average=True, deterministic=False)
                for _ in range(self.n_models)
            ]
            return jnp.stack(outs, axis=0)

    return _MCEns(model, n)


def _make_deep_ensemble(models):
    class _DE:
        def __init__(self, ms):
            self.models = ms

        def predict(self, x):
            outs = [m(x, use_running_average=True) for m in self.models]
            return jnp.stack(outs, axis=0)

    return _DE(models)


def _build_llla(model, x_train, n_perturbations, prior_precision):
    @jax.jit
    def get_features(x):
        h = model.stem(x)
        h = model._run_stages(h, use_running_average=True)
        h = model.final_bn(h, use_running_average=True)
        h = jax.nn.relu(h)
        h = jnp.mean(h, axis=(1, 2))
        return h

    @jax.jit
    def compute_batch_ggn(x_hat, probs):
        H = jax.vmap(lambda p: jnp.diag(p) - jnp.outer(p, p))(probs)
        return jnp.einsum("si,sj,skm->ikjm", x_hat, x_hat, H)

    D, K = 512, N_CLASSES
    G = jnp.zeros((D + 1, K, D + 1, K))
    for i in range(0, len(x_train), 128):
        x_batch = x_train[i : i + 128]
        feats = get_features(x_batch)
        logits = model.fc(feats)
        probs = jax.nn.softmax(logits)
        x_hat = jnp.concatenate([feats, jnp.ones((feats.shape[0], 1))], axis=-1)
        G += compute_batch_ggn(x_hat, probs)
    G_flat = G.reshape((D + 1) * K, (D + 1) * K)
    precision = G_flat + prior_precision * jnp.eye(G_flat.shape[0])
    covariance = jnp.linalg.inv(precision)
    fc_state = nnx.state(model.fc)
    return LLLAEnsemble(model, fc_state, covariance, n_perturbations, SEED)


METHOD_KEYS = [
    "single_model",
    "mahalanobis",
    "react",
    "mc_dropout",
    "deep_ensemble",
    "swag",
    "llla",
    "epinet",
    "pnc_single",
    "pnc_multi",
]


def _save_one(name: str, payload: dict, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"inference_cost_{name}.json", "w") as f:
        json.dump(payload, f, indent=2)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method",
        choices=METHOD_KEYS + ["all"],
        default="all",
        help="Which method to benchmark. Use this to run a single method per subprocess "
             "(prevents GPU memory accumulation).",
    )
    parser.add_argument("--out-dir", default="results/cifar10/inference_cost", help="Per-method output directory")
    args = parser.parse_args()
    out_dir = Path(args.out_dir)

    print("Loading CIFAR-10...")
    x_train, _, x_test, _ = load_cifar10()
    x_bench = x_test[:N_BENCH_SAMPLES]
    print(f"Benchmark set: {x_bench.shape}")

    sel = args.method
    results = {}

    if sel in ("all", "single_model"):
        print("\n[1/10] PreAct ResNet-18 base / MSP / Energy")
        model = PreActResNet18(n_classes=N_CLASSES, rngs=nnx.Rngs(SEED))
        _load_state(model, CKPT_BASE)
        ens = _make_single_ens(model)
        cold, warm = _time_predict(ens, x_bench, BATCH_SIZE)
        results["PreAct ResNet-18 / MSP / Energy"] = {
            "cold_warmup_s": cold,
            "warm_total_s": warm,
            "warm_per_sample_ms": 1000.0 * warm / N_BENCH_SAMPLES,
            "n_forward_passes": 1,
            "train_cost_factor": 1.0,
        }

    if sel in ("all", "mahalanobis"):
        print("[2/10] Mahalanobis (feature extraction only)")
        m_maha = PreActResNet18(n_classes=N_CLASSES, rngs=nnx.Rngs(SEED))
        _load_state(m_maha, CKPT_BASE)
        t0 = time.time()
        _ = _extract_features_batched(m_maha, x_bench[:BATCH_SIZE], BATCH_SIZE)
        cold = time.time() - t0
        t0 = time.time()
        _ = _extract_features_batched(m_maha, x_bench, BATCH_SIZE)
        warm = time.time() - t0
        results["Mahalanobis"] = {
            "cold_warmup_s": cold,
            "warm_total_s": warm,
            "warm_per_sample_ms": 1000.0 * warm / N_BENCH_SAMPLES,
            "n_forward_passes": 1,
            "train_cost_factor": 1.0,
            "note": "feature extraction only; Mahalanobis distance is pure-numpy O(N*C*D)",
        }

    if sel in ("all", "react"):
        print("[3/10] ReAct + Energy")
        m_react = PreActResNet18(n_classes=N_CLASSES, rngs=nnx.Rngs(SEED))
        _load_state(m_react, CKPT_BASE)
        feats = _extract_features_batched(m_react, x_train[:1024], BATCH_SIZE)
        threshold = float(np.percentile(feats, 90))
        ens = _make_react_ens(m_react, threshold)
        cold, warm = _time_predict(ens, x_bench, BATCH_SIZE)
        results["ReAct+Energy"] = {
            "cold_warmup_s": cold,
            "warm_total_s": warm,
            "warm_per_sample_ms": 1000.0 * warm / N_BENCH_SAMPLES,
            "n_forward_passes": 1,
            "train_cost_factor": 1.0,
        }

    if sel in ("all", "mc_dropout"):
        print("[4/10] MC Dropout n=32")
        mc_model = MCDropoutPreActResNet18(n_classes=N_CLASSES, dropout_rate=0.1, rngs=nnx.Rngs(SEED))
        _load_state(mc_model, CKPT_MCDROPOUT)
        ens = _make_mc_dropout_ens(mc_model, 32)
        cold, warm = _time_predict(ens, x_bench, BATCH_SIZE)
        results["MC Dropout n=32"] = {
            "cold_warmup_s": cold,
            "warm_total_s": warm,
            "warm_per_sample_ms": 1000.0 * warm / N_BENCH_SAMPLES,
            "n_forward_passes": 32,
            "train_cost_factor": 1.0,
        }

    if sel in ("all", "deep_ensemble"):
        print("[5/10] Deep Ensemble n=5")
        ens_models = []
        for s in range(5):
            m = PreActResNet18(n_classes=N_CLASSES, rngs=nnx.Rngs(s))
            _load_state(m, f"results/cifar10/preact_resnet18_train{RECIPE}_seed{s}.pkl")
            ens_models.append(m)
        ens = _make_deep_ensemble(ens_models)
        cold, warm = _time_predict(ens, x_bench, BATCH_SIZE)
        results["Deep Ensemble n=5"] = {
            "cold_warmup_s": cold,
            "warm_total_s": warm,
            "warm_per_sample_ms": 1000.0 * warm / N_BENCH_SAMPLES,
            "n_forward_passes": 5,
            "train_cost_factor": 5.0,
        }

    if sel in ("all", "swag"):
        print("[6/10] SWAG n=50 (cached samples)")
        swag_model = PreActResNet18(n_classes=N_CLASSES, rngs=nnx.Rngs(SEED))
        _load_state(swag_model, CKPT_BASE)
        with open(CKPT_SWAG, "rb") as f:
            sckpt = pickle.load(f)
        rng_sub = np.random.RandomState(SEED)
        bn_idx = rng_sub.choice(len(x_train), size=2048, replace=False)
        swag_ens = SWAGEnsemble(
            swag_model,
            sckpt["swag_mean"],
            sckpt["swag_var"],
            50,
            swag_cov_mat_sqrt=sckpt.get("swag_cov_mat_sqrt"),
            bn_refresh_inputs=x_train[bn_idx],
            bn_refresh_batch_size=128,
            use_bn_refresh=True,
            seed=SEED,
            cache_samples=True,
        )
        cold, warm = _time_predict(swag_ens, x_bench, BATCH_SIZE)
        results["SWAG n=50"] = {
            "cold_warmup_s": cold,
            "warm_total_s": warm,
            "warm_per_sample_ms": 1000.0 * warm / N_BENCH_SAMPLES,
            "n_forward_passes": 50,
            "train_cost_factor": 1.0,
            "note": "cached samples + BN refresh on 2048 ID train samples (one-time)",
        }

    if sel in ("all", "llla"):
        print("[7/10] LLLA n=50 (prior=10.0)")
        llla_model = PreActResNet18(n_classes=N_CLASSES, rngs=nnx.Rngs(SEED))
        _load_state(llla_model, CKPT_BASE)
        llla_ens = _build_llla(llla_model, x_train, n_perturbations=50, prior_precision=10.0)
        cold, warm = _time_predict(llla_ens, x_bench, BATCH_SIZE)
        results["LLLA n=50"] = {
            "cold_warmup_s": cold,
            "warm_total_s": warm,
            "warm_per_sample_ms": 1000.0 * warm / N_BENCH_SAMPLES,
            "n_forward_passes": 50,
            "train_cost_factor": 1.0,
            "note": "head-only sampling on top of one feature extraction per batch",
        }

    if sel in ("all", "epinet"):
        print("[8/10] Epinet n=50 (ps=3.0)")
        base_for_epi = PreActResNet18(n_classes=N_CLASSES, rngs=nnx.Rngs(SEED))
        _load_state(base_for_epi, CKPT_BASE)
        with open(CKPT_EPINET_PS3, "rb") as f:
            epi_ckpt = pickle.load(f)
        hiddens = epi_ckpt["hiddens"]
        epinet = EpinetWithPrior(
            feature_dim=512, n_classes=N_CLASSES,
            index_dim=epi_ckpt["index_dim"], hiddens=hiddens,
            prior_scale=epi_ckpt["prior_scale"], rngs=nnx.Rngs(SEED + 1000),
        )
        nnx.update(epinet, epi_ckpt["epinet_state"])
        epi_ens = EpinetEnsemble(base_for_epi, epinet, n_models=50, index_dim=epi_ckpt["index_dim"], seed=SEED)
        cold, warm = _time_predict(epi_ens, x_bench, BATCH_SIZE)
        results["Epinet n=50"] = {
            "cold_warmup_s": cold,
            "warm_total_s": warm,
            "warm_per_sample_ms": 1000.0 * warm / N_BENCH_SAMPLES,
            "n_forward_passes": 50,
            "train_cost_factor": 1.05,
            "note": "1.05x train cost: small MLP fine-tuned on top of frozen base",
        }

    if sel in ("all", "pnc_single"):
        print("[9/10] PnC single-block (s3b1, scale=25.0, K=20)")
        pnc_model = PreActResNet18(n_classes=N_CLASSES, rngs=nnx.Rngs(SEED))
        _load_state(pnc_model, CKPT_BASE)
        pnc_ens, _ = _build_single_block_pnc_ensemble(
            pnc_model, x_train,
            target_stage_idx=3, target_block_idx=1,
            n_directions=20, n_perturbations=50,
            perturbation_scale=25.0, subset_size=1024,
            chunk_size=1024, lambda_reg=1e-3,
            random_directions=True, seed=SEED,
        )
        cold, warm = _time_predict(pnc_ens, x_bench, BATCH_SIZE)
        results["PnC single-block scale=25"] = {
            "cold_warmup_s": cold,
            "warm_total_s": warm,
            "warm_per_sample_ms": 1000.0 * warm / N_BENCH_SAMPLES,
            "n_forward_passes": 50,
            "train_cost_factor": 1.0,
            "note": "ridge-regression correction per perturbed Conv1",
        }

    if sel in ("all", "pnc_multi"):
        print("[10/10] PnC multi-block (scale=7.0, K=20)")
        pnc_mb_model = PreActResNet18(n_classes=N_CLASSES, rngs=nnx.Rngs(SEED))
        _load_state(pnc_mb_model, CKPT_BASE)
        pnc_mb_ens, _ = _build_multi_block_pnc_ensemble(
            pnc_mb_model, x_train,
            n_directions=20, n_perturbations=50,
            perturbation_scale=7.0, subset_size=1024,
            chunk_size=64, lambda_reg=1e-3,
            sigma_sq_weights=False, random_directions=True, seed=SEED,
        )
        cold, warm = _time_predict(pnc_mb_ens, x_bench, BATCH_SIZE)
        results["PnC multi-block scale=7"] = {
            "cold_warmup_s": cold,
            "warm_total_s": warm,
            "warm_per_sample_ms": 1000.0 * warm / N_BENCH_SAMPLES,
            "n_forward_passes": 50,
            "train_cost_factor": 1.0,
            "note": "perturbation across all 8 residual blocks",
        }

    # ---- Save ----
    payload = {
        "n_bench_samples": N_BENCH_SAMPLES,
        "batch_size": BATCH_SIZE,
        "seed": SEED,
        "method_selected": sel,
        "methods": results,
    }
    if sel == "all":
        out_path = Path("results/cifar10/inference_cost.json")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"\nSaved {out_path}")
    else:
        _save_one(sel, payload, out_dir)
        print(f"\nSaved per-method {out_dir / f'inference_cost_{sel}.json'}")

    # Markdown table
    print("\n| Method | Train cost (×) | Forward passes/sample | Warm ms/sample | Cold warmup (s) |")
    print("|---|---:|---:|---:|---:|")
    for name, r in results.items():
        print(
            f"| {name} | {r['train_cost_factor']:.2f}× | {r['n_forward_passes']} "
            f"| {r['warm_per_sample_ms']:.2f} | {r['cold_warmup_s']:.2f} |"
        )


if __name__ == "__main__":
    main()
