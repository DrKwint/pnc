"""Stage 4 -- efficiency profiling (Section 18).

Offline: sketch construction time + memory (from build artifacts) and serialized sketch size.
Online: synchronized base-forward vs SCOD-score latency at batch sizes 1/32/128. SCOD needs a
per-example parameter Jacobian, so a "batch" is scored as a loop of per-example score-feature
evaluations (it cannot be a single large batched pass -- (C,P) Jacobians would be ~0.45GB each).
Reports base latency, SCOD latency, ratio, images/sec, peak GPU memory. Warm-up + >=100 timed
iters where practical.
"""
from __future__ import annotations

import argparse, json, time, os
from pathlib import Path
import numpy as np
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ["XLA_FLAGS"] = os.environ.get("XLA_FLAGS", "") + " --xla_gpu_enable_triton_gemm=false"
REPO = Path(__file__).resolve().parents[2]
import sys
os.chdir(REPO); sys.path.insert(0, str(REPO))
import jax, jax.numpy as jnp, yaml

from experiments.scod_cifar import protocol
from experiments.scod_cifar.parameter_layout import make_ztilde_fn, make_logits_fn
from experiments.scod_cifar.sketch import load_sketch
from experiments.scod_cifar.scorer import make_score_feature_fn


def _cfg(p):
    with open(p) as f:
        return yaml.safe_load(f)


def gpu_mb():
    try:
        import subprocess
        return float(subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"]).decode().splitlines()[0])
    except Exception:
        return float("nan")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--config", required=True)
    args = ap.parse_args(); cfg = _cfg(args.config)
    root = REPO / cfg["root"]; (root / "timing").mkdir(parents=True, exist_ok=True)
    seed = cfg["model_seeds"][0]; base = int(cfg["sketch_seed_base_primary"])
    sk_path = root / "sketches" / f"scod1024_seed{seed}_sketchseed{base+seed}_tempered.npz"

    report = {"offline": {}, "online": {}}
    # ---- offline ----
    bm = json.load(open(root / "metrics" / f"seed{seed}_build_meta.json"))
    sp = json.load(open(root / "spectra" / f"{sk_path.stem}.json"))
    report["offline"] = dict(
        build_seconds_seed0=bm.get("build_seconds"),
        sketch_build_seconds=sp["diagnostics"].get("build_seconds"),
        serialized_sketch_bytes=int(sk_path.stat().st_size),
        memory_preflight=json.load(open(root / "timing" / f"seed{seed}_memory_preflight.json")),
    )

    # ---- online ----
    S = protocol.prepare_seed(seed); T = S["temperature"]
    st = load_sketch(sk_path); U = jnp.asarray(st["basis"])
    xte, _ = protocol.get_id_test()
    logits_fn = make_logits_fn(S["graphdef"], S["rest"], S["unravel"], temperature=T)
    ztilde = make_ztilde_fn(S["graphdef"], S["rest"], S["unravel"], T)
    feat = make_score_feature_fn(ztilde, U)

    def timeit(fn, iters):
        fn()  # warm
        t0 = time.time()
        for _ in range(iters):
            fn()
        return (time.time() - t0) / iters

    for B in (1, 32, 128):
        xb = jnp.asarray(xte[:B])
        # base forward (single batched pass)
        base_fn = lambda: jax.block_until_ready(logits_fn(S["flat_w"], xb))
        base_lat = timeit(base_fn, 100 if B <= 32 else 50)
        # SCOD: per-example loop over the B images
        def scod_fn():
            outs = [feat(S["flat_w"], xb[i]) for i in range(B)]
            jax.block_until_ready(outs[-1])
        scod_lat = timeit(scod_fn, 20 if B <= 32 else 10)
        report["online"][f"batch_{B}"] = dict(
            base_forward_ms=round(base_lat * 1e3, 3),
            scod_score_ms=round(scod_lat * 1e3, 3),
            ratio_scod_over_base=round(scod_lat / base_lat, 1),
            images_per_sec_scod=round(B / scod_lat, 1),
            peak_gpu_mb=gpu_mb(),
        )
        print(f"[profile] B={B}: base {base_lat*1e3:.2f}ms  scod {scod_lat*1e3:.2f}ms  "
              f"ratio {scod_lat/base_lat:.1f}x  {B/scod_lat:.1f} img/s", flush=True)

    json.dump(report, open(root / "timing" / "profile.json", "w"), indent=2)
    print(f"[profile] wrote {root/'timing'/'profile.json'}")


if __name__ == "__main__":
    main()
