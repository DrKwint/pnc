#!/usr/bin/env python3
"""Peak GPU memory per CIFAR-10 OOD method.

Closes the one gap in CIFAR_EFFICIENCY_ACCOUNTING.md section 4: the shipped
benchmark times latency only and never captured memory.

Reuses scripts/benchmark_inference_cost.py verbatim for model construction so the
configurations are identical to the published latency table. Adds three memory
probes per method, using JAX's allocator stats:

  build_peak_mib    : peak during ensemble construction (includes transient solve/inverse buffers)
  resident_mib      : bytes_in_use after construction, before inference  <- validates the DERIVED
                      storage numbers in sections 3.3 / 3.4
  predict_peak_mib  : peak across construction + warm inference over N samples

Must be run ONE METHOD PER SUBPROCESS (see the driver .sh): peak stats are
cumulative per process and JAX exposes no reset, so a shared process would report
the running max across all methods.

Writes results/neurips_2026_rebuttal/cifar/peak_memory/peak_memory_<method>.json
"""
from __future__ import annotations

import os
# Must be set before JAX initialises, or the allocator grabs the whole card and
# every memory number becomes meaningless.
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import argparse
import gc
import importlib.util
import json
import pickle
import sys
from pathlib import Path

REPO = Path("/home/elean/pnc")
os.chdir(REPO)
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

# Import the published benchmark module so construction is bit-identical.
_spec = importlib.util.spec_from_file_location(
    "bic", str(REPO / "scripts" / "benchmark_inference_cost.py")
)
bic = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bic)

from models import PreActResNet18, MCDropoutPreActResNet18
from ensembles import SWAGEnsemble, EpinetEnsemble, EpinetWithPrior
from openood_eval import _extract_features_batched
from cifar_tasks import (
    _build_single_block_pnc_ensemble,
    _build_multi_block_pnc_ensemble,
)

OUT_DIR = REPO / "results" / "neurips_2026_rebuttal" / "cifar" / "peak_memory"
MIB = 1024.0 ** 2


def _dev():
    return jax.local_devices()[0]


def _stats() -> dict:
    try:
        return _dev().memory_stats() or {}
    except Exception:
        return {}


def _peak_mib() -> float:
    s = _stats()
    v = s.get("peak_bytes_in_use")
    return float(v) / MIB if v is not None else float("nan")


def _in_use_mib() -> float:
    s = _stats()
    v = s.get("bytes_in_use")
    return float(v) / MIB if v is not None else float("nan")


# --------------------------------------------------------------------------
# Builders. Each returns (ensemble, n_forward_passes, note).
# Configurations mirror scripts/benchmark_inference_cost.py exactly.
# --------------------------------------------------------------------------

def build(method: str, x_train, x_bench):
    N = bic.N_CLASSES
    S = bic.SEED

    if method == "single_model":
        m = PreActResNet18(n_classes=N, rngs=nnx.Rngs(S))
        bic._load_state(m, bic.CKPT_BASE)
        return bic._make_single_ens(m), 1, "single forward pass"

    if method == "react":
        m = PreActResNet18(n_classes=N, rngs=nnx.Rngs(S))
        bic._load_state(m, bic.CKPT_BASE)
        feats = _extract_features_batched(m, x_train[:1024], bic.BATCH_SIZE)
        thr = float(np.percentile(feats, 90))
        return bic._make_react_ens(m, thr), 1, "ReAct clipping at 90th pct"

    if method == "mc_dropout":
        m = MCDropoutPreActResNet18(n_classes=N, dropout_rate=0.1, rngs=nnx.Rngs(S))
        bic._load_state(m, bic.CKPT_MCDROPOUT)
        return bic._make_mc_dropout_ens(m, 32), 32, "n=32"

    if method in ("deep_ensemble", "deep_ensemble_50"):
        n = 5 if method == "deep_ensemble" else 50
        ms = []
        for i in range(n):
            # DE n=5 uses genuinely distinct seeds; n=50 uses clones of seed 0
            # (same convention as _de_matched_timing.py -- memory is
            # weight-independent, only the count matters).
            src = i if n == 5 else 0
            m = PreActResNet18(n_classes=N, rngs=nnx.Rngs(i))
            bic._load_state(m, f"results/cifar10/preact_resnet18_train{bic.RECIPE}_seed{src}.pkl")
            ms.append(m)
        note = "distinct seeds 0-4" if n == 5 else "50 clones of seed 0 (matched-M)"
        return bic._make_deep_ensemble(ms), n, note

    if method == "swag":
        m = PreActResNet18(n_classes=N, rngs=nnx.Rngs(S))
        bic._load_state(m, bic.CKPT_BASE)
        with open(bic.CKPT_SWAG, "rb") as f:
            sck = pickle.load(f)
        rng = np.random.RandomState(S)
        idx = rng.choice(len(x_train), size=2048, replace=False)
        ens = SWAGEnsemble(
            m, sck["swag_mean"], sck["swag_var"], 50,
            swag_cov_mat_sqrt=sck.get("swag_cov_mat_sqrt"),
            bn_refresh_inputs=x_train[idx], bn_refresh_batch_size=128,
            use_bn_refresh=True, seed=S, cache_samples=True,
        )
        return ens, 50, "cached samples + BN refresh on 2048 train imgs"

    if method == "llla":
        m = PreActResNet18(n_classes=N, rngs=nnx.Rngs(S))
        bic._load_state(m, bic.CKPT_BASE)
        ens = bic._build_llla(m, x_train, n_perturbations=50, prior_precision=10.0)
        return ens, 50, "dense 5130^2 GGN inverse + Cholesky, prior=10.0"

    if method == "epinet":
        m = PreActResNet18(n_classes=N, rngs=nnx.Rngs(S))
        bic._load_state(m, bic.CKPT_BASE)
        with open(bic.CKPT_EPINET_PS3, "rb") as f:
            ck = pickle.load(f)
        epi = EpinetWithPrior(
            feature_dim=512, n_classes=N, index_dim=ck["index_dim"],
            hiddens=ck["hiddens"], prior_scale=ck["prior_scale"],
            rngs=nnx.Rngs(S + 1000),
        )
        nnx.update(epi, ck["epinet_state"])
        ens = EpinetEnsemble(m, epi, n_models=50, index_dim=ck["index_dim"], seed=S)
        return ens, 50, "ps=3.0, n=50"

    if method in ("pnc_single_s3b1", "pnc_anchor_s3b0"):
        # s3b1 is what the SHIPPED latency benchmark measured.
        # s3b0 is the SUBMITTED anchor (ANCHOR_CONFIG.json). Different conv1
        # shapes -> different per-member footprint, so both are reported.
        blk = 1 if method == "pnc_single_s3b1" else 0
        m = PreActResNet18(n_classes=N, rngs=nnx.Rngs(S))
        bic._load_state(m, bic.CKPT_BASE)
        kwargs = dict(
            target_stage_idx=3, target_block_idx=blk,
            n_directions=20, n_perturbations=50,
            perturbation_scale=25.0, subset_size=1024,
            chunk_size=1024, lambda_reg=1e-3,
            random_directions=True, seed=S,
        )
        if blk == 0:
            # anchor uses per-member bootstrap of the ridge inputs
            kwargs["bootstrap_frac"] = 0.05
        ens, _ = _build_single_block_pnc_ensemble(m, x_train, **kwargs)
        note = "s3b1 (as shipped in latency table)" if blk == 1 else \
               "s3b0 + bootstrap_frac=0.05 (SUBMITTED anchor)"
        return ens, 50, note

    if method == "pnc_multi":
        m = PreActResNet18(n_classes=N, rngs=nnx.Rngs(S))
        bic._load_state(m, bic.CKPT_BASE)
        ens, _ = _build_multi_block_pnc_ensemble(
            m, x_train, n_directions=20, n_perturbations=50,
            perturbation_scale=7.0, subset_size=1024, chunk_size=64,
            lambda_reg=1e-3, sigma_sq_weights=False,
            random_directions=True, seed=S,
        )
        return ens, 50, "all 8 residual blocks, scale=7.0"

    raise ValueError(f"unknown method {method}")


METHODS = [
    "single_model", "react", "mc_dropout", "deep_ensemble", "deep_ensemble_50",
    "swag", "llla", "epinet", "pnc_single_s3b1", "pnc_anchor_s3b0", "pnc_multi",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=METHODS)
    args = ap.parse_args()
    method = args.method

    if not _stats():
        print("FATAL: allocator stats unavailable on this backend", file=sys.stderr)
        sys.exit(2)

    x_train, _, x_test, _ = bic.load_cifar10()
    x_bench = x_test[: bic.N_BENCH_SAMPLES]

    # Baseline after data is on host but before any model touches the device.
    gc.collect()
    base_peak = _peak_mib()
    base_use = _in_use_mib()

    print(f"[{method}] building...", flush=True)
    ens, n_fwd, note = build(method, x_train, x_bench)
    jax.block_until_ready(jax.device_put(0.0))
    gc.collect()
    build_peak = _peak_mib()
    resident = _in_use_mib()

    print(f"[{method}] warm inference over {bic.N_BENCH_SAMPLES} samples...", flush=True)
    # Identical call path to the latency benchmark.
    cold, warm = bic._time_predict(ens, x_bench, bic.BATCH_SIZE)
    gc.collect()
    predict_peak = _peak_mib()
    predict_use = _in_use_mib()

    s = _stats()
    payload = {
        "method": method,
        "note": note,
        "n_forward_passes": n_fwd,
        "n_bench_samples": bic.N_BENCH_SAMPLES,
        "batch_size": bic.BATCH_SIZE,
        "seed": bic.SEED,
        "gpu_total_mib": float(s.get("bytes_limit", 0)) / MIB or None,
        "baseline_peak_mib": base_peak,
        "baseline_in_use_mib": base_use,
        "build_peak_mib": build_peak,
        "resident_after_build_mib": resident,
        "predict_peak_mib": predict_peak,
        "in_use_after_predict_mib": predict_use,
        # cross-check against the published latency table
        "cold_warmup_s": cold,
        "warm_total_s": warm,
        "warm_per_sample_ms": 1000.0 * warm / bic.N_BENCH_SAMPLES,
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"peak_memory_{method}.json"
    with open(out, "w") as f:
        json.dump(payload, f, indent=2)

    print(
        f"[{method}] build_peak={build_peak:.1f} MiB  resident={resident:.1f} MiB  "
        f"predict_peak={predict_peak:.1f} MiB  ({1000.0*warm/bic.N_BENCH_SAMPLES:.3f} ms/sample)",
        flush=True,
    )
    print(f"[{method}] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
