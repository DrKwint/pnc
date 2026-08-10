#!/usr/bin/env python3
"""Phase 1 / Section 3: first-order validity range of the CORRECTED residual.

For a fixed unit-norm direction v in conv1 parameter space, sweep scale t on a log grid.
At each t: w1(t)=w1_orig+t*v, refit the toward-original ridge correction (exact, code solver),
and compute the corrected conv2 residual R_{tv}(x) via the (validated) patch formula.

First-order prediction: A_S(x)v estimated by central-ish finite difference at the smallest scale,
A ~= R_{t0 v}(x)/t0, cross-checked at two tiny scales. Compare R_{tv} vs t*A across the grid.

Reports per scale (and per regime ID/near/far): relerr(R_t, t*A) normalized by linear & by actual,
cosine, remainder norm, and the log-log remainder slope at small scales. Finds the largest scale
within 10% of the linearization; compares to submitted scale 25.
Writes firstorder_sweep_raw.json.
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
from cifar_tasks import (_load_cifar_openood_context, compute_cifar_block_preacts,
                         make_cifar_block_get_Y_fn, CIFAROpenOODPnC)
from pnc import extract_patches, flatten_conv_kernel_to_patches, unflatten_conv_kernel_from_patches, find_random_directions, solve_chunked_conv2_correction

OUT = REPO / "results" / "neurips_2026_rebuttal" / "cifar"
SEED = 0
CALIB = 1024
CHUNK = 256
PHI_BATCH = 128
N_EVAL = 64
LAM = 1e-3
K = 20
BLOCK = (3, 0)          # submitted anchor
SCALES = [0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 25.0, 50.0, 100.0]
SUBMITTED = 25.0


def relerr(a, b):
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-30))


def main():
    t0 = time.time()
    si, bi = BLOCK
    task = CIFAROpenOODPnC(dataset="cifar10", epochs=300, perturbation_sizes=[25.0], n_directions=K,
                           n_perturbations=50, subset_size=CALIB, chunk_size=CHUNK, target_stage_idx=si,
                           target_block_idx=bi, random_directions=True, seed=SEED, lambda_reg=LAM,
                           posthoc_calibrate=True, bootstrap_frac=0.0)
    benchmark, x_tr, _, x_va, y_va, n_cls = _load_cifar_openood_context(task)
    model = PreActResNet18(n_classes=n_cls, rngs=nnx.Rngs(SEED))
    with open(task.input().path, "rb") as f:
        nnx.update(model, pickle.load(f)["state"])
    stages = [model.stage1, model.stage2, model.stage3, model.stage4]
    blk = stages[si][bi]
    w1_orig = np.asarray(blk.conv1.kernel[...], np.float32)
    w2_orig = np.asarray(blk.conv2.kernel[...], np.float32)
    w1norm = float(np.linalg.norm(w1_orig))
    Theta = np.asarray(flatten_kernel := flatten_conv_kernel_to_patches(jnp.asarray(w2_orig)), np.float64)
    cout = w2_orig.shape[-1]
    get_Y = make_cifar_block_get_Y_fn(blk)

    # unit-norm direction (row 0 of orthonormal basis)
    V, _ = find_random_directions(w1_orig.size, K, seed=SEED)
    v = np.asarray(V[0], np.float32).reshape(w1_orig.shape)
    v = v / (np.linalg.norm(v) + 1e-30)

    # calibration chunks (block inputs h and T_orig for the ORIGINAL conv1)
    pa_chunks, t_chunks = compute_cifar_block_preacts(model, jnp.asarray(x_tr[:CALIB]), CHUNK, si, bi, jnp.asarray(w1_orig))

    # eval sets
    rng = np.random.RandomState(7)
    def take(x):
        return np.asarray(x)[rng.choice(len(x), min(N_EVAL, len(x)), replace=False)]
    evalsets = {"id_test": ("id", take(benchmark["id_test"]["inputs"])),
                "cifar100": ("near", take(benchmark["near_ood"]["cifar100"]["inputs"])),
                "svhn": ("far", take(benchmark["far_ood"]["svhn"]["inputs"]))}

    def U_f32(w1, h):
        out = jax.nn.relu(blk.bn1(jnp.asarray(h, jnp.float32), use_running_average=True))
        y = jax.lax.conv_general_dilated(lhs=out, rhs=jnp.asarray(w1, jnp.float32).transpose(3, 2, 0, 1),
            window_strides=tuple(blk.conv1.strides), padding="SAME", dimension_numbers=("NHWC", "OIHW", "NHWC"))
        return jax.nn.relu(blk.bn2(y, use_running_average=True))
    def Phi(w1, h):
        outs = []
        for i in range(0, len(h), PHI_BATCH):
            outs.append(np.asarray(extract_patches(U_f32(w1, h[i:i+PHI_BATCH]), k=3, strides=1), np.float64))
        return np.concatenate(outs, 0)

    # precompute original eval patches Phi(x), eval block inputs, and calibration design (FLOAT64)
    eval_h = {name: compute_cifar_block_preacts(model, jnp.asarray(X), CHUNK, si, bi, jnp.asarray(w1_orig))[0][0]
              for name, (_, X) in evalsets.items()}
    eval_phi0 = {name: Phi(w1_orig, eval_h[name]) for name in evalsets}
    eval_shapes = {name: (len(X),) for name, (_, X) in evalsets.items()}
    cal_h_all = np.concatenate([np.asarray(c) for c in pa_chunks], axis=0)
    X_cal = Phi(w1_orig, cal_h_all)                         # (rows, p-1) f64
    ones_cal = np.ones((X_cal.shape[0], 1))
    Xaug_cal = np.concatenate([ones_cal, X_cal], axis=1)
    T_cal = X_cal @ Theta                                   # recomputed original conv2 output (f64, self-consistent)
    Theta_aug = np.concatenate([np.zeros((1, cout)), Theta], axis=0)

    def corrected_residual(tt):
        """R_{tv}(x) per eval set via an independent FLOAT64 toward-original ridge solve + patch formula."""
        w1p = (w1_orig + tt * v).astype(np.float32)
        Xv_cal = Phi(w1p, cal_h_all)
        Xvaug = np.concatenate([ones_cal, Xv_cal], axis=1)
        Gv = Xvaug.T @ Xvaug + LAM * np.eye(Xvaug.shape[1])
        R_target = T_cal - Xv_cal @ Theta
        Theta_delta = np.linalg.solve(Gv, Xvaug.T @ R_target)
        Theta_v_aug = Theta_aug + Theta_delta
        out, sh_disp = {}, {}
        for name in evalsets:
            phi0 = eval_phi0[name]; phiv = Phi(w1p, eval_h[name])
            onse = np.ones((phiv.shape[0], 1))
            out[name] = np.concatenate([onse, phiv], 1) @ Theta_v_aug - phi0 @ Theta
            sh_disp[name] = float(np.mean(np.linalg.norm(phiv - phi0, axis=1)))
        return out, sh_disp

    # linear coefficient A = R_{t0 v}/t0 at smallest scale; cross-check at 2nd smallest
    t0s = SCALES[0]; t1s = SCALES[1]
    R_t0, _ = corrected_residual(t0s)
    R_t1, _ = corrected_residual(t1s)
    A = {name: R_t0[name] / t0s for name in evalsets}
    lin_crosscheck = {name: relerr(R_t1[name] / t1s, A[name]) for name in evalsets}

    per_scale = []
    for tt in SCALES:
        R, disp = corrected_residual(tt)
        row = {"scale": tt, "s_rel": tt / w1norm}
        for name, (regime, _) in evalsets.items():
            Rt = R[name]; lin = tt * A[name]
            rem = Rt - lin
            # per-image norms
            n = eval_shapes[name][0]
            Rt_img = Rt.reshape(n, -1); lin_img = lin.reshape(n, -1); rem_img = rem.reshape(n, -1)
            cos = float(np.mean([np.dot(Rt_img[j], lin_img[j]) /
                                 (np.linalg.norm(Rt_img[j]) * np.linalg.norm(lin_img[j]) + 1e-30) for j in range(n)]))
            row[name] = {
                "regime": regime,
                "relerr_norm_by_linear": relerr(Rt, lin),
                "relerr_norm_by_actual": float(np.linalg.norm(rem) / (np.linalg.norm(Rt) + 1e-30)),
                "cosine": cos,
                "remainder_norm": float(np.linalg.norm(rem)),
                "actual_norm": float(np.linalg.norm(Rt)),
                "hidden_feat_disp": disp[name],
            }
        per_scale.append(row)
        print(f"[fo] t={tt:7.3f} s_rel={tt/w1norm:.3f} | " +
              " ".join(f"{name}:relL={row[name]['relerr_norm_by_linear']:.2e},cos={row[name]['cosine']:.3f}"
                       for name in evalsets), flush=True)

    # largest scale within 10% (by relerr_norm_by_linear) per regime
    within10 = {}
    for name in evalsets:
        ok = [r["scale"] for r in per_scale if r[name]["relerr_norm_by_linear"] <= 0.10]
        within10[name] = max(ok) if ok else None
    # log-log remainder slope at small scales (first 5 points)
    slopes = {}
    for name in evalsets:
        xs = np.log10([r["scale"] for r in per_scale[:5]])
        ys = np.log10([r[name]["remainder_norm"] + 1e-30 for r in per_scale[:5]])
        slopes[name] = float(np.polyfit(xs, ys, 1)[0])

    meta = {"seed": SEED, "block": f"s{si}b{bi}", "calib": CALIB, "n_eval": N_EVAL, "lambda": LAM,
            "w1_fro_norm": w1norm, "submitted_scale": SUBMITTED, "lin_crosscheck_relerr": lin_crosscheck,
            "largest_scale_within_10pct": within10, "loglog_remainder_slope_smallscale": slopes,
            "runtime_sec": time.time() - t0}
    json.dump({"meta": meta, "per_scale": per_scale}, open(OUT / "firstorder_sweep_raw.json", "w"), indent=2)
    print(f"[fo] within-10% scale: {within10} ; submitted={SUBMITTED} ; remainder slope: "
          f"{ {k: round(v,2) for k,v in slopes.items()} } ; xcheck={ {k: f'{v:.1e}' for k,v in lin_crosscheck.items()} }", flush=True)
    print(f"[fo] wrote firstorder_sweep_raw.json in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
