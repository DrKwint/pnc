#!/usr/bin/env python3
"""SWAG post-training construction wall-clock, seeds 0/1/2.

Gives SWAG the same construction number Laplace already has (report section 2.5), so the
two are comparable. Timer semantics are matched to CIFARLLLA.run()'s `train_time`:

  t0 -> after data load, before any checkpoint is touched
  t1 -> after the ensemble object is ready to predict, before any evaluation

For SWAG that span is: base checkpoint load, SWAG moment checkpoint load, posterior
sample draw (cache_samples=True), and the BatchNorm refresh over 2048 ID train images.
It excludes the 300-epoch training run that collected the moments (report section 2.6).

Uses the same construction path as scripts/benchmark_inference_cost.py so the numbers
line up with the latency and peak-memory tables.
"""
from __future__ import annotations

import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import importlib.util
import json
import pickle
import statistics as stats
import sys
import time
from pathlib import Path

REPO = Path("/home/elean/pnc")
os.chdir(REPO)
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import jax
import numpy as np
from flax import nnx

_spec = importlib.util.spec_from_file_location(
    "bic", str(REPO / "scripts" / "benchmark_inference_cost.py")
)
bic = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bic)

from models import PreActResNet18
from ensembles import SWAGEnsemble

RECIPE = bic.RECIPE
SEEDS = [0, 1, 2]
OUT = REPO / "results" / "neurips_2026_rebuttal" / "cifar" / "swag_construction_time.json"


def swag_ckpt(seed: int) -> str:
    return (f"results/cifar10/preact_resnet18_swag_train{RECIPE}"
            f"_sws240_swf1_swr20_seed{seed}.pkl")


def base_ckpt(seed: int) -> str:
    return f"results/cifar10/preact_resnet18_train{RECIPE}_seed{seed}.pkl"


def main():
    x_train, _, x_test, _ = bic.load_cifar10()

    rows = []
    for seed in SEEDS:
        # ---- timed span begins (matches CIFARLLLA.run() t0 placement) ----
        t0 = time.time()

        model = PreActResNet18(n_classes=bic.N_CLASSES, rngs=nnx.Rngs(seed))
        bic._load_state(model, base_ckpt(seed))

        with open(swag_ckpt(seed), "rb") as f:
            sck = pickle.load(f)

        rng = np.random.RandomState(seed)
        bn_idx = rng.choice(len(x_train), size=2048, replace=False)

        ens = SWAGEnsemble(
            model, sck["swag_mean"], sck["swag_var"], 50,
            swag_cov_mat_sqrt=sck.get("swag_cov_mat_sqrt"),
            bn_refresh_inputs=x_train[bn_idx], bn_refresh_batch_size=128,
            use_bn_refresh=True, seed=seed, cache_samples=True,
        )
        # force any lazily-deferred device work to complete before stopping the clock
        out = ens.predict(x_test[:128])
        if hasattr(out, "block_until_ready"):
            out.block_until_ready()
        jax.block_until_ready(out)

        t = time.time() - t0
        # ---- timed span ends ----

        rows.append({"seed": seed, "construction_s": t})
        print(f"[swag] seed {seed}: {t:.2f} s", flush=True)

    vals = [r["construction_s"] for r in rows]
    payload = {
        "method": "SWAG n=50 (sws240, swr20)",
        "what_is_timed": (
            "base ckpt load + SWAG moment ckpt load + 50-sample posterior draw "
            "(cache_samples=True) + BatchNorm refresh over 2048 ID train images, "
            "up to a predict-ready ensemble. Excludes the 300-epoch training run "
            "that collected the moments (report section 2.6). Includes one 128-image "
            "predict to force deferred device work; that is ~1 s of the total."
        ),
        "timer_semantics_matched_to": "CIFARLLLA.run() train_time (report section 2.5)",
        "n_members": 50,
        "bn_refresh_images": 2048,
        "seeds": rows,
        "mean_s": stats.mean(vals),
        "sd_s": stats.stdev(vals) if len(vals) > 1 else 0.0,
    }
    OUT.write_text(json.dumps(payload, indent=2))
    print(f"\n[swag] mean {payload['mean_s']:.2f} s  sd {payload['sd_s']:.2f} s  (n={len(vals)})")
    print(f"[swag] wrote {OUT}")


if __name__ == "__main__":
    main()
