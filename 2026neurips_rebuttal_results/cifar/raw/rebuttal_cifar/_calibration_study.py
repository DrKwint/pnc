#!/usr/bin/env python3
"""Phase 2 / Section 9: calibration-set construction for s3b0.

Given the geometry (s3b0 rank grows with images to ~512; stable rank ~5), test whether smarter image
selection beats random at SMALL calibration sizes. Compare {random, class-balanced, k-center on block-input
features} at calib in {128, 256, 512}, all else at anchor (s3b0, ps25, bf=0, M=50). Report ID acc/NLL,
ID MI, Near/Far AUROC (predictive_entropy). bf=0 isolates the calibration-set effect from bootstrap.
Writes calibration_study_raw.json.
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
from cifar_tasks import _load_cifar_openood_context, _build_single_block_pnc_ensemble, CIFAROpenOODPnC

OUT = REPO / "results" / "neurips_2026_rebuttal" / "cifar"
SEED = 0; CHUNK = 256; SCALE = 25.0; LAM = 1e-3; K = 20; SI, BI = 3, 0; M = 50
N_EVAL = 800; BATCH = 200; SIZES = [128, 256, 512]; METHODS = ["random", "class_balanced", "kcenter"]
NEAR = ["cifar100", "tiny_imagenet"]; FAR = ["mnist", "svhn", "textures", "places365"]


def softmax(z):
    z = z - z.max(-1, keepdims=True); e = np.exp(z); return e / e.sum(-1, keepdims=True)
def mlogits(ens, x):
    return np.concatenate([np.asarray(ens.predict(jnp.asarray(x[i:i+BATCH]))) for i in range(0, len(x), BATCH)], axis=1)
def auroc(a, b):
    s = np.concatenate([a, b]); o = s.argsort(); r = np.empty(len(s)); r[o] = np.arange(1, len(s)+1)
    n1, n2 = len(b), len(a); return (r[len(a):].sum()-n1*(n1+1)/2)/(n1*n2)


def main():
    t0 = time.time()
    task = CIFAROpenOODPnC(dataset="cifar10", epochs=300, perturbation_sizes=[SCALE], n_directions=K,
                           n_perturbations=M, subset_size=1024, chunk_size=CHUNK, target_stage_idx=SI,
                           target_block_idx=BI, random_directions=True, seed=SEED, lambda_reg=LAM,
                           posthoc_calibrate=True, bootstrap_frac=0.0)
    benchmark, x_tr, y_tr, x_va, y_va, n_cls = _load_cifar_openood_context(task)
    y_tr = np.asarray(y_tr).astype(int)
    model = PreActResNet18(n_classes=n_cls, rngs=nnx.Rngs(SEED))
    with open(task.input().path, "rb") as f:
        nnx.update(model, pickle.load(f)["state"])
    stages = [model.stage1, model.stage2, model.stage3, model.stage4]

    def block_input_rep(x):
        h = model.stem(jnp.asarray(x, jnp.float32))
        for s in range(SI):
            for b in range(len(stages[s])):
                h = stages[s][b](h, use_running_average=True)
        return np.asarray(jnp.mean(h, axis=(1, 2)), np.float64)

    # precompute pool features for k-center (subset of pool for tractability)
    POOL = 4096
    pool_idx = np.random.RandomState(0).choice(len(x_tr), POOL, replace=False)
    pool_x = np.asarray(x_tr)[pool_idx]; pool_y = y_tr[pool_idx]
    pool_feat = np.concatenate([block_input_rep(pool_x[i:i+256]) for i in range(0, POOL, 256)], 0)

    def select(method, sz):
        if method == "random":
            return pool_x[np.random.RandomState(1).choice(POOL, sz, replace=False)]
        if method == "class_balanced":
            per = sz // n_cls; idx = []
            rs = np.random.RandomState(2)
            for c in range(n_cls):
                ci = np.where(pool_y == c)[0]; idx.extend(rs.choice(ci, min(per, len(ci)), replace=False))
            idx = np.array(idx)
            if len(idx) < sz:
                extra = rs.choice(np.setdiff1d(np.arange(POOL), idx), sz-len(idx), replace=False); idx = np.concatenate([idx, extra])
            return pool_x[idx[:sz]]
        if method == "kcenter":  # greedy k-center on pool_feat
            rs = np.random.RandomState(3); first = rs.randint(POOL)
            chosen = [first]; d = np.linalg.norm(pool_feat - pool_feat[first], axis=1)
            for _ in range(sz-1):
                nxt = int(np.argmax(d)); chosen.append(nxt)
                d = np.minimum(d, np.linalg.norm(pool_feat - pool_feat[nxt], axis=1))
            return pool_x[np.array(chosen)]

    rng = np.random.RandomState(7)
    def take(x): return np.asarray(x)[rng.choice(len(x), min(N_EVAL, len(x)), replace=False)]
    rng2 = np.random.RandomState(7); sel = rng2.choice(len(benchmark["id_test"]["inputs"]), min(N_EVAL, len(benchmark["id_test"]["inputs"])), replace=False)
    idx = benchmark["id_test"]["inputs"][sel]; idy = np.asarray(benchmark["id_test"]["targets"])[sel].astype(int)
    near = {d: take(benchmark["near_ood"][d]["inputs"]) for d in NEAR}
    far = {d: take(benchmark["far_ood"][d]["inputs"]) for d in FAR}

    results = []
    for sz in SIZES:
        for method in METHODS:
            xcal = select(method, sz)
            ens, _ = _build_single_block_pnc_ensemble(nnx.clone(model), xcal, target_stage_idx=SI, target_block_idx=BI,
                n_directions=K, n_perturbations=M, perturbation_scale=SCALE, subset_size=sz, chunk_size=CHUNK,
                lambda_reg=LAM, random_directions=True, seed=SEED, bootstrap_frac=0.0, bootstrap_seed=SEED)
            Lid = mlogits(ens, idx); p = softmax(Lid); mp = p.mean(0); eps = 1e-12
            pe = -(mp*np.log(mp+eps)).sum(-1); mi = pe - (-(p*np.log(p+eps)).sum(-1).mean(0))
            acc = float((mp.argmax(-1) == idy).mean())*100; nll = float(-np.log(mp[np.arange(len(idy)), idy]+eps).mean())
            def macro(group):
                a = [auroc(pe, (lambda L: (lambda m: -(m*np.log(m+eps)).sum(-1))(softmax(L).mean(0)))(mlogits(ens, group[d])))*100 for d in group]
                return float(np.mean(a))
            na = macro(near); fa = macro(far)
            results.append({"size": sz, "method": method, "id_acc": acc, "id_nll": nll, "id_MI": float(mi.mean()),
                            "near_auroc": na, "far_auroc": fa})
            print(f"[cal] sz={sz:4d} {method:15} id_acc={acc:.2f} id_nll={nll:.3f} id_MI={mi.mean():.4f} nearAUROC={na:.2f} farAUROC={fa:.2f}", flush=True)

    json.dump({"meta": {"seed": SEED, "block": f"s{SI}b{BI}", "M": M, "sizes": SIZES, "methods": METHODS,
                        "runtime_sec": time.time()-t0}, "results": results},
              open(OUT / "calibration_study_raw.json", "w"), indent=2)
    print(f"[cal] wrote calibration_study_raw.json in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
