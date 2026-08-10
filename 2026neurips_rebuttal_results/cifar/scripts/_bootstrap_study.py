#!/usr/bin/env python3
"""Phase 4 / Section 10: does bootstrap add useful diversity or estimation noise?

Build the anchor s3b0 ensemble (M=50, seed 0) at bootstrap_frac in {0.0, 0.05(submitted), 0.5}, all else fixed,
and compare on ID test + Near + Far: ID acc/NLL, ID predictive entropy & MI (calibration cost), logit-cov trace,
Near/Far AUROC & FPR95 (predictive_entropy + logit_cov_tr). Tests whether subset-induced variation helps OOD
(good) without inflating ID uncertainty (noise). Writes bootstrap_study_raw.json.
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
SEED = 0; CALIB = 1024; CHUNK = 256; SCALE = 25.0; LAM = 1e-3; K = 20; SI, BI = 3, 0; M = 50
N_EVAL = 800; BATCH = 200; BFS = [0.0, 0.05, 0.5]
NEAR = ["cifar100", "tiny_imagenet"]; FAR = ["mnist", "svhn", "textures", "places365"]


def softmax(z):
    z = z - z.max(-1, keepdims=True); e = np.exp(z); return e / e.sum(-1, keepdims=True)
def mlogits(ens, x):
    return np.concatenate([np.asarray(ens.predict(jnp.asarray(x[i:i+BATCH]))) for i in range(0, len(x), BATCH)], axis=1)
def auroc(a, b):
    s = np.concatenate([a, b]); o = s.argsort(); r = np.empty(len(s)); r[o] = np.arange(1, len(s)+1)
    n1, n2 = len(b), len(a); return (r[len(a):].sum()-n1*(n1+1)/2)/(n1*n2)
def fpr95(a, b):
    return float((a >= np.quantile(b, 0.05)).mean())


def main():
    t0 = time.time()
    task = CIFAROpenOODPnC(dataset="cifar10", epochs=300, perturbation_sizes=[SCALE], n_directions=K,
                           n_perturbations=M, subset_size=CALIB, chunk_size=CHUNK, target_stage_idx=SI,
                           target_block_idx=BI, random_directions=True, seed=SEED, lambda_reg=LAM,
                           posthoc_calibrate=True, bootstrap_frac=0.05)
    benchmark, x_tr, _, x_va, y_va, n_cls = _load_cifar_openood_context(task)
    model = PreActResNet18(n_classes=n_cls, rngs=nnx.Rngs(SEED))
    with open(task.input().path, "rb") as f:
        nnx.update(model, pickle.load(f)["state"])
    rng = np.random.RandomState(7)
    def take(x): return np.asarray(x)[rng.choice(len(x), min(N_EVAL, len(x)), replace=False)]
    idx = take(benchmark["id_test"]["inputs"]); idy = None
    rng2 = np.random.RandomState(7); sel = rng2.choice(len(benchmark["id_test"]["inputs"]), min(N_EVAL, len(benchmark["id_test"]["inputs"])), replace=False)
    idx = benchmark["id_test"]["inputs"][sel]; idy = np.asarray(benchmark["id_test"]["targets"])[sel].astype(int)
    near = {d: take(benchmark["near_ood"][d]["inputs"]) for d in NEAR}
    far = {d: take(benchmark["far_ood"][d]["inputs"]) for d in FAR}

    results = {}
    for bf in BFS:
        ens, _ = _build_single_block_pnc_ensemble(nnx.clone(model), x_tr, target_stage_idx=SI, target_block_idx=BI,
            n_directions=K, n_perturbations=M, perturbation_scale=SCALE, subset_size=CALIB, chunk_size=CHUNK,
            lambda_reg=LAM, random_directions=True, seed=SEED, bootstrap_frac=bf, bootstrap_seed=SEED)
        Lid = mlogits(ens, idx); p = softmax(Lid); mp = p.mean(0); eps = 1e-12
        pe = -(mp*np.log(mp+eps)).sum(-1); mi = pe - (-(p*np.log(p+eps)).sum(-1).mean(0))
        lct = np.array([np.trace(np.cov(Lid[:, j, :].T)) for j in range(Lid.shape[1])])
        acc = float((mp.argmax(-1) == idy).mean()); nll = float(-np.log(mp[np.arange(len(idy)), idy]+eps).mean())
        def dscore(Ld, which):
            pp = softmax(Ld); m = pp.mean(0)
            if which == "pe": return -(m*np.log(m+eps)).sum(-1)
            return np.array([np.trace(np.cov(Ld[:, j, :].T)) for j in range(Ld.shape[1])])
        def macro(group, which):
            si_ = dscore(Lid, which)
            a = [auroc(si_, dscore(mlogits(ens, group[d]), which))*100 for d in group]
            fr = [fpr95(si_, dscore(mlogits(ens, group[d]), which))*100 for d in group]
            return float(np.mean(a)), float(np.mean(fr))
        na_pe, nf_pe = macro(near, "pe"); fa_pe, ff_pe = macro(far, "pe")
        na_lc, nf_lc = macro(near, "lc"); fa_lc, ff_lc = macro(far, "lc")
        results[f"bf{bf}"] = {"bf": bf, "id_acc": acc*100, "id_nll": nll,
                              "id_pred_ent": float(pe.mean()), "id_MI": float(mi.mean()), "id_logit_cov_tr": float(lct.mean()),
                              "near_auroc_pe": na_pe, "near_fpr95_pe": nf_pe, "far_auroc_pe": fa_pe, "far_fpr95_pe": ff_pe,
                              "near_auroc_lc": na_lc, "far_auroc_lc": fa_lc}
        print(f"[bs] bf={bf}: id_acc={acc*100:.2f} id_nll={nll:.3f} id_MI={mi.mean():.4f} "
              f"near_AUROC(pe)={na_pe:.2f} far_AUROC(pe)={fa_pe:.2f} near_AUROC(lc)={na_lc:.2f}", flush=True)

    json.dump({"meta": {"seed": SEED, "block": f"s{SI}b{BI}", "M": M, "n_eval": N_EVAL, "bfs": BFS,
                        "runtime_sec": time.time()-t0}, "results": results},
              open(OUT / "bootstrap_study_raw.json", "w"), indent=2)
    print(f"[bs] wrote bootstrap_study_raw.json in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
