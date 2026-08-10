#!/usr/bin/env python3
"""Phase 2 / Section 2.1 + 2.5: per-block correction geometry & conditioning.

For every PreActResNet-18 residual block, accumulate the augmented conv2 design Gram
G_v = X_v^T X_v (p x p, memory-safe streaming) at the operating perturbation (scale 25,
one unit-norm direction) on 1024 ID calibration images, and report:
  p, patch-rows, numerical rank, stable rank, cond(G_v), cond(G_v+lambda I), smallest
  supported singular value, nominal rows/p, numerical-rank/p; and the rank-vs-#images
  curve (does numerical rank grow with images or saturate = spatial redundancy).

Writes block_geometry_raw.json.
"""
from __future__ import annotations
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import sys, json, time
from pathlib import Path
import numpy as np
import jax, jax.numpy as jnp
from flax import nnx
import pickle

REPO = Path(__file__).resolve().parents[3]
os.chdir(REPO); sys.path.insert(0, str(REPO))
from models import PreActResNet18
from cifar_tasks import _load_cifar_openood_context, compute_cifar_block_preacts, CIFAROpenOODPnC
from pnc import extract_patches, find_random_directions

OUT = REPO / "results" / "neurips_2026_rebuttal" / "cifar"
SEED = 0; CALIB = 1024; IMG_CHUNK = 64; LAM = 1e-3; SCALE = 25.0
CHECKPOINTS = [64, 128, 256, 512, 1024]
BLOCKS = [(s, b) for s in range(4) for b in range(2)]


def main():
    t0 = time.time()
    task = CIFAROpenOODPnC(dataset="cifar10", epochs=300, perturbation_sizes=[SCALE], subset_size=CALIB,
                           chunk_size=256, target_stage_idx=3, target_block_idx=0, random_directions=True,
                           seed=SEED, lambda_reg=LAM, posthoc_calibrate=True, bootstrap_frac=0.0)
    benchmark, x_tr, _, x_va, y_va, n_cls = _load_cifar_openood_context(task)
    model = PreActResNet18(n_classes=n_cls, rngs=nnx.Rngs(SEED))
    with open(task.input().path, "rb") as f:
        nnx.update(model, pickle.load(f)["state"])
    stages = [model.stage1, model.stage2, model.stage3, model.stage4]
    Xcal = np.asarray(x_tr[:CALIB], np.float32)

    results = {}
    for (si, bi) in BLOCKS:
        name = f"s{si}b{bi}"
        blk = stages[si][bi]
        w1 = np.asarray(blk.conv1.kernel[...], np.float32)
        Cin2 = blk.conv2.kernel.shape[2]
        p = 9 * Cin2 + 1
        # unit-norm perturbed conv1 at scale 25
        V, _ = find_random_directions(w1.size, 20, seed=SEED)
        v = np.array(V[0], np.float32).reshape(w1.shape); v = v / (np.linalg.norm(v) + 1e-30)
        w1p = (w1 + SCALE * v).astype(np.float32)

        def U(w1w, h):
            out = jax.nn.relu(blk.bn1(jnp.asarray(h, jnp.float32), use_running_average=True))
            y = jax.lax.conv_general_dilated(lhs=out, rhs=jnp.asarray(w1w).transpose(3, 2, 0, 1),
                window_strides=tuple(blk.conv1.strides), padding="SAME", dimension_numbers=("NHWC", "OIHW", "NHWC"))
            return jax.nn.relu(blk.bn2(y, use_running_average=True))

        G = np.zeros((p, p), np.float64)     # augmented Gram of X_v
        rows_done = 0; imgs_done = 0
        rank_curve = []
        n_ch = int(np.ceil(CALIB / IMG_CHUNK))
        ckpt_set = set(CHECKPOINTS)
        for i in range(n_ch):
            xb = Xcal[i*IMG_CHUNK:(i+1)*IMG_CHUNK]
            h = compute_cifar_block_preacts(model, jnp.asarray(xb), IMG_CHUNK, si, bi, jnp.asarray(w1))[0][0]
            phiv = np.asarray(extract_patches(U(w1p, h), k=3, strides=1), np.float64)
            M = np.concatenate([np.ones((phiv.shape[0], 1)), phiv], axis=1)
            G += M.T @ M
            rows_done += M.shape[0]; imgs_done += len(xb)
            if imgs_done in ckpt_set:
                ev = np.linalg.eigvalsh(G)                     # eigenvalues of Gram = singular values^2
                ev = np.clip(ev, 0, None)
                sv = np.sqrt(ev[::-1])                         # descending singular values
                tol = sv[0] * max(G.shape) * np.finfo(np.float64).eps
                nrank = int(np.sum(sv > tol))
                rank_curve.append({"images": imgs_done, "rows": rows_done, "num_rank": nrank})
        # final diagnostics from full G
        ev = np.clip(np.linalg.eigvalsh(G), 0, None)
        sv = np.sqrt(ev[::-1])
        smax = float(sv[0]); tol = smax * p * np.finfo(np.float64).eps
        nrank = int(np.sum(sv > tol))
        stable_rank = float((sv**2).sum() / (smax**2 + 1e-300))
        smin_supported = float(sv[nrank-1]) if nrank > 0 else 0.0
        lam_abs = LAM
        evG = ev[::-1]  # descending eigenvalues of G (= sv^2)
        cond_G = float(evG[0] / (evG[nrank-1] + 1e-300)) if nrank > 0 else float("inf")
        cond_reg = float((evG[0] + lam_abs) / (evG[-1] + lam_abs))
        results[name] = {
            "stage": si, "block": bi, "p": p, "Cin_conv2": int(Cin2),
            "patch_rows": rows_done, "nominal_rows_over_p": rows_done / p,
            "numerical_rank": nrank, "num_rank_over_p": nrank / p,
            "stable_rank": stable_rank, "cond_G": cond_G, "cond_G_plus_lambda": cond_reg,
            "smallest_supported_sv": smin_supported, "largest_sv": smax,
            "lambda_rel": lam_abs / (float(np.trace(G)) / p),
            "rank_curve": rank_curve,
        }
        rr = results[name]
        print(f"[geo] {name}: p={p} rows={rows_done} rows/p={rr['nominal_rows_over_p']:.1f} "
              f"numrank={nrank} rank/p={rr['num_rank_over_p']:.2f} stable_rank={stable_rank:.1f} "
              f"cond(G)={cond_G:.1e} cond(G+λ)={cond_reg:.1e} λ_rel={rr['lambda_rel']:.1e}", flush=True)

    json.dump({"meta": {"seed": SEED, "calib": CALIB, "scale": SCALE, "lambda": LAM,
                        "runtime_sec": time.time()-t0}, "results": results},
              open(OUT / "block_geometry_raw.json", "w"), indent=2)
    print(f"[geo] wrote block_geometry_raw.json in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
