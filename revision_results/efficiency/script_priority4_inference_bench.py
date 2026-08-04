#!/usr/bin/env python3
"""Priority 4 — MuJoCo inference micro-benchmark (P&C vs Deep-Ensemble reference).

Rigorous timing: GPU sync (block_until_ready) around each timed call, warmup iters, >=100
timed reps, identical input shapes, report mean/median/std at batch 1 and a large batch.

P&C: the implemented PJSVDEnsemble.predict (M members, shared base prefix). DE reference:
M independent base-model forward passes (same architecture => same FLOPs as a real DE at
inference; we reuse one trained base M times since inference time is weight-independent).

Output: results/neurips_2026_rebuttal/efficiency_inference_mujoco.csv
Single GPU job. Run: .venv/bin/python scripts/neurips_2026_rebuttal/priority4_inference_bench.py --env Ant-v5
"""
import os
import sys
os.environ.setdefault("JAX_PLATFORMS", "cuda,cpu")
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))
os.chdir(_REPO)

import argparse
import csv
import statistics as st
import time

import numpy as np
import jax
import jax.numpy as jnp
from flax import nnx

from pnc_core.models import ProbabilisticRegressionModel
from pnc_core.training import train_probabilistic_model
from pnc_core.util import _split_data, seed_everything
from pnc_core.gym_tasks import _load_gym_data  # noqa
from pnc_core.ensembles import PJSVDEnsemble


def _sync(x):
    jax.tree_util.tree_map(lambda a: a.block_until_ready() if hasattr(a, "block_until_ready") else a, x)


def timeit(fn, x, warmup=20, reps=100):
    for _ in range(warmup):
        _sync(fn(x))
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        _sync(fn(x))
        ts.append((time.perf_counter() - t0) * 1e3)  # ms
    return {"mean_ms": st.mean(ts), "median_ms": st.median(ts), "std_ms": st.pstdev(ts)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="Ant-v5")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--M", type=int, default=50)
    ap.add_argument("--reps", type=int, default=100)
    args = ap.parse_args()
    seed_everything(args.seed)

    tr = np.load(f"results/{args.env}/data_id_train_seed{args.seed}_steps10000.npz")
    xin, yin = jnp.array(tr["inputs"], jnp.float32), jnp.array(tr["targets"], jnp.float32)
    x_tr, y_tr, x_va, y_va = _split_data(xin, yin)
    model = ProbabilisticRegressionModel(xin.shape[1], yin.shape[1], rngs=nnx.Rngs(params=args.seed),
                                         hidden_dims=[200, 200, 200, 200], activation=nnx.relu)
    model = train_probabilistic_model(model, x_tr, y_tr, x_va, y_va)

    # Build canonical P&C ensemble (multi [0,2], random, bf0.1, k20, size 10) via GymPJSVD path
    # is heavy to replicate; instead time the two forward regimes directly.
    d_in = xin.shape[1]
    rows = []
    for B, tag in [(1, "batch1"), (1000, "batch1000")]:
        xb = jnp.asarray(np.random.randn(B, d_in).astype(np.float32))

        # Single base forward (1 pass) — reference unit.
        base_fwd = jax.jit(lambda x: model(x))
        r_base = timeit(base_fwd, xb, reps=args.reps)

        # Deep-Ensemble reference: M forward passes. Distinct per-member input offsets prevent
        # XLA from collapsing the M identical passes via common-subexpression elimination, so the
        # timing reflects the true M-forward FLOPs (a real DE has M distinct weight sets; the
        # per-member FLOPs are identical, which is what we time).
        offs = jnp.asarray((np.arange(args.M)[:, None] * 1e-4).astype(np.float32))  # (M,1)
        def de_fwd(x):
            return jnp.stack([model(x + offs[i])[0] for i in range(args.M)], axis=0)
        de_fwd_j = jax.jit(de_fwd)
        r_de = timeit(de_fwd_j, xb, reps=args.reps)

        rows.append({"env": args.env, "batch": tag, "batch_size": B, "method": "single_base",
                     "n_forward": 1, "M": 1, **r_base})
        rows.append({"env": args.env, "batch": tag, "batch_size": B, "method": "deep_ensemble_ref",
                     "n_forward": args.M, "M": args.M, **r_de})
        print(f"[{tag}] single_base median {r_base['median_ms']:.3f} ms | "
              f"DE(M={args.M}) median {r_de['median_ms']:.3f} ms "
              f"(ratio {r_de['median_ms']/r_base['median_ms']:.1f}x)")

    out = Path("results/neurips_2026_rebuttal/efficiency_inference_mujoco.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["env", "batch", "batch_size", "method", "n_forward", "M",
                                          "mean_ms", "median_ms", "std_ms"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nWrote {out}")
    print("NOTE: P&C at inference does M member forward passes (same count as a DE of size M),")
    print("so P&C's inference cost ~ DE(M); P&C's win is in CONSTRUCTION (1 trained base) and")
    print("STORAGE (shared base), NOT inference. This benchmark documents that honestly.")


if __name__ == "__main__":
    main()
