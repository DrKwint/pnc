#!/usr/bin/env python3
"""Phase 4 / Section 11: is frozen BN a convenience or part of the mechanism?

Anchor s3b0, M=16 members, scale 25, CALIB=1024. Compare:
  (1) FROZEN bn2 (headline): correction fit + forward with base running bn2 stats.
  (2) per-member bn2 REFRESH: recompute bn2 running stats from the member's perturbed conv1 output on the
      calibration subset, refit the conv2 correction under the refreshed bn2, forward with refreshed bn2.
Both correct toward the SAME base conv2 target (no OOD data used). Report ID acc/NLL, ID MI, Near/Far AUROC.
Writes bn_study_raw.json.
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
SEED = 0; CALIB = 1024; CHUNK = 256; PHI_BATCH = 128; N_EVAL = 400; SCALE = 25.0; LAM = 1e-3; K = 20; M = 16
SI, BI = 3, 0
NEAR = ["cifar100", "tiny_imagenet"]; FAR = ["mnist", "svhn", "textures", "places365"]


def flatten_np(w): return np.asarray(w, np.float64).transpose(2, 0, 1, 3).reshape(-1, w.shape[-1])
def unflat_np(wf, shp):
    kh, kw, ci, co = shp; return wf.reshape(ci, kh, kw, co).transpose(1, 2, 0, 3)
def softmax(z):
    z = z - z.max(-1, keepdims=True); e = np.exp(z); return e / e.sum(-1, keepdims=True)
def auroc(a, b):
    s = np.concatenate([a, b]); o = s.argsort(); r = np.empty(len(s)); r[o] = np.arange(1, len(s)+1)
    n1, n2 = len(b), len(a); return (r[len(a):].sum()-n1*(n1+1)/2)/(n1*n2)


def main():
    t0 = time.time()
    task = CIFAROpenOODPnC(dataset="cifar10", epochs=300, perturbation_sizes=[SCALE], subset_size=CALIB,
                           chunk_size=CHUNK, target_stage_idx=SI, target_block_idx=BI, random_directions=True,
                           seed=SEED, lambda_reg=LAM, posthoc_calibrate=True, bootstrap_frac=0.0)
    benchmark, x_tr, _, x_va, y_va, n_cls = _load_cifar_openood_context(task)
    model = PreActResNet18(n_classes=n_cls, rngs=nnx.Rngs(SEED))
    with open(task.input().path, "rb") as f:
        nnx.update(model, pickle.load(f)["state"])
    stages = [model.stage1, model.stage2, model.stage3, model.stage4]
    blk = stages[SI][BI]
    w1o = np.asarray(blk.conv1.kernel[...], np.float32); w2o = np.asarray(blk.conv2.kernel[...], np.float32)
    Theta = flatten_np(w2o); cout = w2o.shape[-1]
    # bn2 params + base running stats
    bn2 = blk.bn2
    GAMMA = np.asarray(bn2.scale[...], np.float64); BETA = np.asarray(bn2.bias[...], np.float64)
    rm0 = np.asarray(bn2.mean[...], np.float64); rv0 = np.asarray(bn2.var[...], np.float64)
    EPS = 1e-5

    def conv1_raw(w1, h):  # pre-bn2 conv1 output (NHWC)
        out = jax.nn.relu(blk.bn1(jnp.asarray(h, jnp.float32), use_running_average=True))
        return np.asarray(jax.lax.conv_general_dilated(lhs=out, rhs=jnp.asarray(w1, jnp.float32).transpose(3, 2, 0, 1),
            window_strides=tuple(blk.conv1.strides), padding="SAME", dimension_numbers=("NHWC", "OIHW", "NHWC")), np.float64)
    def relu_bn2(yraw, rm, rv):
        return np.maximum((yraw - rm)/np.sqrt(rv + EPS)*GAMMA + BETA, 0.0)
    def patches(u):
        return np.asarray(extract_patches(jnp.asarray(u, jnp.float32), k=3, strides=1), np.float64)
    def Phi_from_yraw(yraw_chunks_or_arr, rm, rv):
        u = relu_bn2(yraw_chunks_or_arr, rm, rv); return patches(u)

    # downstream: block output -> logits
    def downstream(hb):
        h = jnp.asarray(hb, jnp.float32)
        for j in range(BI+1, len(stages[SI])): h = stages[SI][j](h, use_running_average=True)
        for s in range(SI+1, 4):
            for b2 in range(len(stages[s])): h = stages[s][b2](h, use_running_average=True)
        h = jax.nn.relu(model.final_bn(h, use_running_average=True))
        return np.asarray(model.fc(jnp.mean(h, axis=(1, 2))), np.float64)
    def shortcut(h, w1):
        out = jax.nn.relu(blk.bn1(jnp.asarray(h, jnp.float32), use_running_average=True))
        return np.asarray(blk.downsample(out), np.float64) if blk.downsample is not None else np.asarray(h, np.float64)

    # calibration: block input h, base conv1 raw, base target T (base conv2 out)
    hcal = np.concatenate([np.asarray(c) for c in compute_cifar_block_preacts(model, jnp.asarray(x_tr[:CALIB]), CHUNK, SI, BI, jnp.asarray(w1o))[0]], 0)
    yraw_base = np.concatenate([conv1_raw(w1o, hcal[i:i+PHI_BATCH]) for i in range(0, len(hcal), PHI_BATCH)], 0)
    Xbase = patches(relu_bn2(yraw_base, rm0, rv0)); Tcal = Xbase @ Theta   # base conv2 target

    # members
    V, _ = find_random_directions(w1o.size, K, seed=SEED)
    zc = np.random.RandomState(SEED+17).normal(0, 1, size=(M, K))
    zc = zc / (np.linalg.norm(zc, axis=1, keepdims=True)+1e-12) * SCALE
    members = [(w1o + (zc[m] @ np.asarray(V, np.float64)).reshape(w1o.shape).astype(np.float32)) for m in range(M)]

    rng = np.random.RandomState(7)
    def take(x): return np.asarray(x)[rng.choice(len(x), min(N_EVAL, len(x)), replace=False)]
    groups = {"id": take(benchmark["id_test"]["inputs"])}
    idy = None
    rng2 = np.random.RandomState(7); sel = rng2.choice(len(benchmark["id_test"]["inputs"]), min(N_EVAL, len(benchmark["id_test"]["inputs"])), replace=False)
    groups["id"] = benchmark["id_test"]["inputs"][sel]; idy = np.asarray(benchmark["id_test"]["targets"])[sel].astype(int)
    for d in NEAR+FAR:
        fam = "near_ood" if d in NEAR else "far_ood"; groups[d] = take(benchmark[fam][d]["inputs"])
    # eval block inputs + base conv1 raw cache
    geval = {gname: np.concatenate([np.asarray(c) for c in compute_cifar_block_preacts(model, jnp.asarray(X), CHUNK, SI, BI, jnp.asarray(w1o))[0]], 0) for gname, X in groups.items()}

    def member_logits(w1p, refresh):
        yraw_cal = np.concatenate([conv1_raw(w1p, hcal[i:i+PHI_BATCH]) for i in range(0, len(hcal), PHI_BATCH)], 0)
        if refresh:
            rm = yraw_cal.reshape(-1, cout).mean(0); rv = yraw_cal.reshape(-1, cout).var(0)
        else:
            rm, rv = rm0, rv0
        Xv = Phi_from_yraw(yraw_cal, rm, rv); onse = np.ones((Xv.shape[0], 1)); Xva = np.concatenate([onse, Xv], 1)
        G = Xva.T @ Xva + LAM*np.eye(Xva.shape[1])
        d = np.linalg.solve(G, Xva.T @ (Tcal - Xv @ Theta))
        w2n = np.asarray(w2o, np.float64) + unflat_np(d[1:], w2o.shape); b2n = d[0:1].reshape(-1)
        # eval forward
        logits = {}
        for gname, hE in geval.items():
            yraw = np.concatenate([conv1_raw(w1p, hE[i:i+PHI_BATCH]) for i in range(0, len(hE), PHI_BATCH)], 0)
            u = relu_bn2(yraw, rm, rv)
            t = np.asarray(jax.lax.conv_general_dilated(lhs=jnp.asarray(u, jnp.float32), rhs=jnp.asarray(w2n.transpose(3, 2, 0, 1), jnp.float32),
                window_strides=(1, 1), padding="SAME", dimension_numbers=("NHWC", "OIHW", "NHWC")), np.float64) + b2n.reshape(1, 1, 1, -1)
            hb = t + shortcut(hE, w1p)
            logits[gname] = downstream(hb)
        return logits

    out = {}
    for variant, refresh in [("frozen", False), ("per_member_refresh", True)]:
        Ls = {g: [] for g in groups}
        for m in range(M):
            ml = member_logits(members[m], refresh)
            for g in groups: Ls[g].append(ml[g])
        L = {g: np.stack(Ls[g], 0) for g in groups}   # (M,N,C)
        def pe(Ld): mp = softmax(Ld).mean(0); return -(mp*np.log(mp+1e-12)).sum(-1)
        mp = softmax(L["id"]).mean(0); eps = 1e-12
        acc = float((mp.argmax(-1) == idy).mean())*100; nll = float(-np.log(mp[np.arange(len(idy)), idy]+eps).mean())
        mi = float(np.median((-(mp*np.log(mp+eps)).sum(-1)) - (-(softmax(L["id"])*np.log(softmax(L["id"])+eps)).sum(-1).mean(0))))
        na = float(np.mean([auroc(pe(L["id"]), pe(L[d]))*100 for d in NEAR]))
        fa = float(np.mean([auroc(pe(L["id"]), pe(L[d]))*100 for d in FAR]))
        out[variant] = {"id_acc": acc, "id_nll": nll, "id_MI": mi, "near_auroc": na, "far_auroc": fa}
        print(f"[bn] {variant:20} id_acc={acc:.2f} id_nll={nll:.3f} id_MI={mi:.4f} nearAUROC={na:.2f} farAUROC={fa:.2f}", flush=True)

    json.dump({"meta": {"seed": SEED, "M": M, "n_eval": N_EVAL, "scale": SCALE, "runtime_sec": time.time()-t0}, "results": out},
              open(OUT / "bn_study_raw.json", "w"), indent=2)
    print(f"[bn] wrote bn_study_raw.json in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
