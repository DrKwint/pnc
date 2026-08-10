#!/usr/bin/env python3
"""Phase 3 / Section 12.4: sequential re-repair in the conv net (self-contained 2-block).

Blocks a=s2b1 (stage3 blk1) -> b=s3b0 (stage4 blk0), adjacent so a's output == b's input.
Perturb+correct block a (direction v_a, scale 25; my validated float64 toward-original ridge),
propagate to block b's input. Then RE-CORRECT block b's conv2 (v_b=0) toward the BASE block-b conv2
output, given the perturbed upstream. Measure the survival ratio of block a's injected residual:
    S_{b<-a}(x) = || r_after(x) || / || r_before(x) ||
  r_before = conv2_b^orig(perturbed upstream) - conv2_b^orig(base upstream)   [uncorrected]
  r_after  = conv2_b^corrected(perturbed upstream) - conv2_b^orig(base upstream) [re-corrected]
Test whether S_ID < S_OOD (asymmetric re-repair: block b suppresses upstream residual more on ID).
Writes multiblock_rerepair_raw.json.
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
from pnc import extract_patches, flatten_conv_kernel_to_patches, find_random_directions

OUT = REPO / "results" / "neurips_2026_rebuttal" / "cifar"
SEED = 0; CALIB = 512; CHUNK = 256; PHI_BATCH = 128; N_EVAL = 128; SCALE = 25.0; LAM = 1e-3; K = 20
A = (2, 1); B = (3, 0)   # a before b (adjacent)


def flatten_np(w): return np.asarray(w, np.float64).transpose(2, 0, 1, 3).reshape(-1, w.shape[-1])
def unflat_np(wf, shp):
    kh, kw, ci, co = shp; return wf.reshape(ci, kh, kw, co).transpose(1, 2, 0, 3)


def main():
    t0 = time.time()
    task = CIFAROpenOODPnC(dataset="cifar10", epochs=300, perturbation_sizes=[SCALE], subset_size=CALIB,
                           chunk_size=CHUNK, target_stage_idx=3, target_block_idx=0, random_directions=True,
                           seed=SEED, lambda_reg=LAM, posthoc_calibrate=True, bootstrap_frac=0.0)
    benchmark, x_tr, _, x_va, y_va, n_cls = _load_cifar_openood_context(task)
    model = PreActResNet18(n_classes=n_cls, rngs=nnx.Rngs(SEED))
    with open(task.input().path, "rb") as f:
        nnx.update(model, pickle.load(f)["state"])
    stages = [model.stage1, model.stage2, model.stage3, model.stage4]
    blkA = stages[A[0]][A[1]]; blkB = stages[B[0]][B[1]]
    w1A = np.asarray(blkA.conv1.kernel[...], np.float32); w2A = np.asarray(blkA.conv2.kernel[...], np.float32)
    w1B = np.asarray(blkB.conv1.kernel[...], np.float32); w2B = np.asarray(blkB.conv2.kernel[...], np.float32)
    ThetaA = flatten_np(w2A); ThetaB = flatten_np(w2B); coutA = w2A.shape[-1]; coutB = w2B.shape[-1]

    def U(blk, w1, h):  # post-bn2-relu of conv1(w1) on block input h
        out = jax.nn.relu(blk.bn1(jnp.asarray(h, jnp.float32), use_running_average=True))
        y = jax.lax.conv_general_dilated(lhs=out, rhs=jnp.asarray(w1, jnp.float32).transpose(3, 2, 0, 1),
            window_strides=tuple(blk.conv1.strides), padding="SAME", dimension_numbers=("NHWC", "OIHW", "NHWC"))
        return jax.nn.relu(blk.bn2(y, use_running_average=True))
    def Phi(blk, w1, h):
        outs = []
        for i in range(0, len(h), PHI_BATCH):
            outs.append(np.asarray(extract_patches(U(blk, w1, h[i:i+PHI_BATCH]), k=3, strides=1), np.float64))
        return np.concatenate(outs, 0)
    def block_out(blk, h, w1, w2f_or_none, w2, b2):
        """Full block output (conv2 branch + shortcut). w2 HWIO f32/f64, b2 (cout,) or None."""
        hh = jnp.asarray(h, jnp.float32)
        out_relu1 = jax.nn.relu(blk.bn1(hh, use_running_average=True))
        y = jax.lax.conv_general_dilated(lhs=out_relu1, rhs=jnp.asarray(w1, jnp.float32).transpose(3, 2, 0, 1),
            window_strides=tuple(blk.conv1.strides), padding="SAME", dimension_numbers=("NHWC", "OIHW", "NHWC"))
        y = jax.nn.relu(blk.bn2(y, use_running_average=True))
        t = jax.lax.conv_general_dilated(lhs=y, rhs=jnp.asarray(w2, jnp.float32).transpose(3, 2, 0, 1),
            window_strides=(1, 1), padding="SAME", dimension_numbers=("NHWC", "OIHW", "NHWC"))
        t = np.asarray(t, np.float64)
        if b2 is not None: t = t + np.asarray(b2).reshape(1, 1, 1, -1)
        ident = np.asarray(blk.downsample(out_relu1), np.float64) if blk.downsample is not None else np.asarray(hh, np.float64)
        return t + ident

    # direction for block a (unit norm), scale 25
    VA, _ = find_random_directions(w1A.size, K, seed=SEED)
    vA = np.array(VA[0], np.float32).reshape(w1A.shape); vA = vA / (np.linalg.norm(vA)+1e-30)
    w1A_p = (w1A + SCALE * vA).astype(np.float32)

    # calibration block-a input
    hA_cal = np.concatenate([np.asarray(c) for c in compute_cifar_block_preacts(
        model, jnp.asarray(x_tr[:CALIB]), CHUNK, A[0], A[1], jnp.asarray(w1A))[0]], 0)
    # correct block a (toward-original ridge, float64)
    Xa = Phi(blkA, w1A, hA_cal); Xav = Phi(blkA, w1A_p, hA_cal)
    onesa = np.ones((Xa.shape[0], 1)); Xav_aug = np.concatenate([onesa, Xav], 1)
    Ta = Xa @ ThetaA
    GA = Xav_aug.T @ Xav_aug + LAM*np.eye(Xav_aug.shape[1])
    dA = np.linalg.solve(GA, Xav_aug.T @ (Ta - Xav @ ThetaA))
    w2A_new = np.asarray(w2A, np.float64) + unflat_np(dA[1:], w2A.shape); b2A_new = dA[0:1].reshape(-1)

    # calibration: propagate to block b input (base and a-corrected), fit block b re-correction
    hB_base_cal = block_out(blkA, hA_cal, w1A, None, w2A, None)                    # base block a out = b input
    hB_pert_cal = block_out(blkA, hA_cal, w1A_p, None, w2A_new, b2A_new)           # a-corrected b input
    XbBase = Phi(blkB, w1B, hB_base_cal); TB_base = XbBase @ ThetaB                # base block-b conv2 target
    XbPert = Phi(blkB, w1B, hB_pert_cal); onesb = np.ones((XbPert.shape[0], 1)); XbPert_aug = np.concatenate([onesb, XbPert], 1)
    GB = XbPert_aug.T @ XbPert_aug + LAM*np.eye(XbPert_aug.shape[1])
    dB = np.linalg.solve(GB, XbPert_aug.T @ (TB_base - XbPert @ ThetaB))
    ThetaB_new = np.concatenate([np.zeros((1, coutB)), ThetaB], 0) + dB            # augmented corrected block-b conv2
    print(f"[rr] built corrections ({time.time()-t0:.0f}s)", flush=True)

    rng = np.random.RandomState(7)
    def take(x): return np.asarray(x)[rng.choice(len(x), min(N_EVAL, len(x)), replace=False)]
    evalsets = {"id_test": ("id", take(benchmark["id_test"]["inputs"])),
                "cifar100": ("near", take(benchmark["near_ood"]["cifar100"]["inputs"])),
                "svhn": ("far", take(benchmark["far_ood"]["svhn"]["inputs"]))}
    results = {}
    for name, (regime, X) in evalsets.items():
        hA = np.concatenate([np.asarray(c) for c in compute_cifar_block_preacts(
            model, jnp.asarray(X), CHUNK, A[0], A[1], jnp.asarray(w1A))[0]], 0)
        hB_base = block_out(blkA, hA, w1A, None, w2A, None)
        hB_pert = block_out(blkA, hA, w1A_p, None, w2A_new, b2A_new)
        XbB = Phi(blkB, w1B, hB_base); base_conv2B = XbB @ ThetaB
        XbP = Phi(blkB, w1B, hB_pert); onse = np.ones((XbP.shape[0], 1)); XbP_aug = np.concatenate([onse, XbP], 1)
        r_before = XbP @ ThetaB - base_conv2B                          # uncorrected upstream residual
        r_after = XbP_aug @ ThetaB_new - base_conv2B                   # after block-b re-correction
        N = len(X); npatch = XbP.shape[0] // N
        rb = np.linalg.norm(r_before.reshape(N, -1), axis=1); ra = np.linalg.norm(r_after.reshape(N, -1), axis=1)
        S = ra / (rb + 1e-30)
        results[name] = {"regime": regime, "n": N, "S_median": float(np.median(S)), "S_mean": float(np.mean(S)),
                         "r_before_med": float(np.median(rb)), "r_after_med": float(np.median(ra)),
                         "suppression_pct_med": float(np.median(1 - S))*100}
        print(f"[rr] {name:9} ({regime}) S_median={np.median(S):.3f} suppression={np.median(1-S)*100:.1f}% "
              f"r_before={np.median(rb):.3f} r_after={np.median(ra):.3f}", flush=True)

    sid = results["id_test"]["S_median"]; snear = results["cifar100"]["S_median"]; sfar = results["svhn"]["S_median"]
    meta = {"seed": SEED, "block_a": f"s{A[0]}b{A[1]}", "block_b": f"s{B[0]}b{B[1]}", "scale": SCALE, "calib": CALIB,
            "n_eval": N_EVAL, "S_id_lt_S_ood": bool(sid < snear and sid < sfar), "runtime_sec": time.time()-t0}
    json.dump({"meta": meta, "results": results}, open(OUT / "multiblock_rerepair_raw.json", "w"), indent=2)
    print(f"[rr] S_ID={sid:.3f} < S_near={snear:.3f}, S_far={sfar:.3f} ? asymmetric_rerepair={meta['S_id_lt_S_ood']} "
          f"({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
