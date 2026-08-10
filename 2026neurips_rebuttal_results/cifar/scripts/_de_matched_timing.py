#!/usr/bin/env python3
"""Phase 7 addendum: matched-M Deep Ensemble (M=50) inference timing.

The submitted efficiency table has Deep Ensemble at its SUBMITTED member count (n=5).
The task also requires a MATCHED-M comparison (same 50 forward passes as P&C anchor),
to make the honest point that P&C is NOT cheaper than an equal-member ensemble at
inference -- its advantage is 1x TRAINING cost, not inference latency.

Latency is weight-independent, so 50 clones of the seed-0 base model suffice for timing.
Reuses the submitted benchmark's own _time_predict / _make_deep_ensemble (same GPU-sync
+ cold/warm methodology, batch_size, N).
"""
from __future__ import annotations
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import sys, json, importlib.util
from pathlib import Path
from flax import nnx

REPO = Path(__file__).resolve().parents[3]
os.chdir(REPO); sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))
import pickle
from models import PreActResNet18
from data import load_cifar10

# import the submitted benchmark module for identical methodology/constants
spec = importlib.util.spec_from_file_location("bic", str(REPO / "scripts" / "benchmark_inference_cost.py"))
bic = importlib.util.module_from_spec(spec); spec.loader.exec_module(bic)

OUT = REPO / "results" / "neurips_2026_rebuttal" / "cifar"
M = 50
CKPT = f"results/cifar10/preact_resnet18_train{bic.RECIPE}_seed0.pkl"

x_train, _, x_test, _ = load_cifar10()
x_bench = x_test[: bic.N_BENCH_SAMPLES]

with open(CKPT, "rb") as f:
    state = pickle.load(f)["state"]
base = PreActResNet18(n_classes=10, rngs=nnx.Rngs(0))
nnx.update(base, state)
models = [nnx.clone(base) for _ in range(M)]   # weight-independent latency
ens = bic._make_deep_ensemble(models)

cold, warm = bic._time_predict(ens, x_bench, bic.BATCH_SIZE)
payload = {
    "n_bench_samples": bic.N_BENCH_SAMPLES, "batch_size": bic.BATCH_SIZE, "seed": 0,
    "method_selected": "deep_ensemble_matched_m50",
    "methods": {
        "Deep Ensemble n=50 (matched-M)": {
            "cold_warmup_s": cold, "warm_total_s": warm,
            "warm_per_sample_ms": 1000.0 * warm / bic.N_BENCH_SAMPLES,
            "n_forward_passes": M, "train_cost_factor": float(M),
            "note": "matched-M latency comparison to P&C anchor (both 50 fwd passes); "
                    "weights are 50 clones of seed-0 base (latency is weight-independent).",
        }
    },
}
outp = OUT / "inference_cost_deep_ensemble_n50_matched.json"
json.dump(payload, open(outp, "w"), indent=2)
m = payload["methods"]["Deep Ensemble n=50 (matched-M)"]
print(f"DE n=50 matched-M: {m['warm_per_sample_ms']:.3f} ms/sample, {M} fwd passes -> {outp}")
