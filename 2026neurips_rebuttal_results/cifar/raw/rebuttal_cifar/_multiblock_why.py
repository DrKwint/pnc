#!/usr/bin/env python3
"""Why doesn't multi-block beat single on CIFAR-10? Test redundancy vs dilution.

Build single-block P&C ensembles at s2b1 (mid) and s3b0 (late/anchor), M=50, scale 25, bf=0.
Compare, on ID + Near + Far:
  * per-block Near/Far AUROC (predictive_entropy + logit_cov) and ID MI  -> dilution (is s2b1 weaker?),
  * per-image PRINCIPAL ANGLES between the two blocks' leading logit-covariance eigenspaces
    -> redundancy (do they disagree in the SAME output directions?),
  * POOLED 100-member ensemble (both blocks' members) AUROC/MI -> does combining help or dilute?
Writes multiblock_why_raw.json.
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
from scipy.linalg import subspace_angles
import pickle

REPO = Path(__file__).resolve().parents[3]
os.chdir(REPO); sys.path.insert(0, str(REPO))
from models import PreActResNet18
from cifar_tasks import _load_cifar_openood_context, _build_single_block_pnc_ensemble, CIFAROpenOODPnC

OUT = REPO / "results" / "neurips_2026_rebuttal" / "cifar"
SEED = 0; CALIB = 1024; CHUNK = 256; SCALE = 25.0; LAM = 1e-3; K = 20; M = 50
N_EVAL = 500; BATCH = 200
BLOCKS = {"s2b1": (2, 1), "s3b0": (3, 0)}
NEAR = ["cifar100", "tiny_imagenet"]; FAR = ["mnist", "svhn", "textures", "places365"]


def softmax(z):
    z = z - z.max(-1, keepdims=True); e = np.exp(z); return e / e.sum(-1, keepdims=True)
def mlog(ens, x):
    return np.concatenate([np.asarray(ens.predict(jnp.asarray(x[i:i+BATCH]))) for i in range(0, len(x), BATCH)], axis=1)
def auroc(a, b):
    s = np.concatenate([a, b]); o = s.argsort(); r = np.empty(len(s)); r[o] = np.arange(1, len(s)+1)
    n1, n2 = len(b), len(a); return (r[len(a):].sum()-n1*(n1+1)/2)/(n1*n2)
def pe_score(L):
    m = softmax(L).mean(0); eps = 1e-12; return -(m*np.log(m+eps)).sum(-1)
def lc_score(L):
    return np.array([np.trace(np.cov(L[:, j, :].T)) for j in range(L.shape[1])])
def mi_arr(L):
    p = softmax(L); mp = p.mean(0); eps = 1e-12
    return (-(mp*np.log(mp+eps)).sum(-1)) - (-(p*np.log(p+eps)).sum(-1).mean(0))


def main():
    t0 = time.time()
    task = CIFAROpenOODPnC(dataset="cifar10", epochs=300, perturbation_sizes=[SCALE], n_directions=K,
                           n_perturbations=M, subset_size=CALIB, chunk_size=CHUNK, target_stage_idx=3,
                           target_block_idx=0, random_directions=True, seed=SEED, lambda_reg=LAM,
                           posthoc_calibrate=True, bootstrap_frac=0.0)
    benchmark, x_tr, _, x_va, y_va, n_cls = _load_cifar_openood_context(task)
    model = PreActResNet18(n_classes=n_cls, rngs=nnx.Rngs(SEED))
    with open(task.input().path, "rb") as f:
        nnx.update(model, pickle.load(f)["state"])
    ens = {}
    for name, (si, bi) in BLOCKS.items():
        ens[name], _ = _build_single_block_pnc_ensemble(nnx.clone(model), x_tr, target_stage_idx=si, target_block_idx=bi,
            n_directions=K, n_perturbations=M, perturbation_scale=SCALE, subset_size=CALIB, chunk_size=CHUNK,
            lambda_reg=LAM, random_directions=True, seed=SEED, bootstrap_frac=0.0, bootstrap_seed=SEED)
    print(f"[why] built s2b1 + s3b0 ensembles ({time.time()-t0:.0f}s)", flush=True)

    rng = np.random.RandomState(7)
    def take(x): return np.asarray(x)[rng.choice(len(x), min(N_EVAL, len(x)), replace=False)]
    id_x = take(benchmark["id_test"]["inputs"])
    groups = {"id": id_x}
    for d in NEAR: groups[d] = take(benchmark["near_ood"][d]["inputs"])
    for d in FAR: groups[d] = take(benchmark["far_ood"][d]["inputs"])

    # member logits per block per dataset
    L = {name: {g: mlog(ens[name], groups[g]) for g in groups} for name in BLOCKS}
    Lpool = {g: np.concatenate([L["s2b1"][g], L["s3b0"][g]], axis=0) for g in groups}  # 100 members

    def macro(Ld, group_names, score):
        sid = score(Ld["id"]); return float(np.mean([auroc(sid, score(Ld[g]))*100 for g in group_names]))

    out = {"per_block": {}, "pooled": {}, "overlap": {}, "dilution": {}}
    for name in BLOCKS:
        Ld = L[name]
        out["per_block"][name] = {
            "near_auroc_pe": macro(Ld, NEAR, pe_score), "far_auroc_pe": macro(Ld, FAR, pe_score),
            "near_auroc_lc": macro(Ld, NEAR, lc_score), "far_auroc_lc": macro(Ld, FAR, lc_score),
            "id_MI": float(np.median(mi_arr(Ld["id"]))),
            "near_MI": float(np.median(np.concatenate([mi_arr(Ld[d]) for d in NEAR]))),
        }
    out["pooled"] = {"near_auroc_pe": macro(Lpool, NEAR, pe_score), "far_auroc_pe": macro(Lpool, FAR, pe_score),
                     "near_auroc_lc": macro(Lpool, NEAR, lc_score), "far_auroc_lc": macro(Lpool, FAR, lc_score),
                     "id_MI": float(np.median(mi_arr(Lpool["id"])))}

    # redundancy: principal-angle cosine between s2b1 & s3b0 leading logit-cov eigenspaces, per regime
    def lead(cov, k=3):
        w, V = np.linalg.eigh(cov); return V[:, ::-1][:, :k]
    for regime, gs in [("id", ["id"]), ("near", NEAR), ("far", FAR)]:
        cosines = []
        for g in gs:
            La = L["s2b1"][g]; Lb = L["s3b0"][g]
            for j in range(La.shape[1]):
                Va = lead(np.cov(La[:, j, :].T)); Vb = lead(np.cov(Lb[:, j, :].T))
                try: cosines.append(np.mean(np.cos(subspace_angles(Va, Vb))))
                except Exception: pass
        out["overlap"][regime] = {"princ_cos_top3_med": float(np.median(cosines)),
                                  "random_baseline_approx": 0.55}

    # dilution summary
    a = out["per_block"]["s2b1"]; b = out["per_block"]["s3b0"]; p = out["pooled"]
    out["dilution"] = {
        "weak_block_near_auroc_pe": a["near_auroc_pe"], "strong_block_near_auroc_pe": b["near_auroc_pe"],
        "pooled_near_auroc_pe": p["near_auroc_pe"],
        "pooled_vs_best_single_pe": p["near_auroc_pe"] - max(a["near_auroc_pe"], b["near_auroc_pe"]),
        "pooled_near_between_weak_and_strong": bool(a["near_auroc_pe"] <= p["near_auroc_pe"] <= b["near_auroc_pe"]),
    }
    print(f"[why] s2b1 near_AUROC(pe)={a['near_auroc_pe']:.2f} s3b0={b['near_auroc_pe']:.2f} "
          f"pooled={p['near_auroc_pe']:.2f} (Δvs best single={out['dilution']['pooled_vs_best_single_pe']:+.2f})", flush=True)
    print(f"[why] overlap princ_cos top3: id={out['overlap']['id']['princ_cos_top3_med']:.3f} "
          f"near={out['overlap']['near']['princ_cos_top3_med']:.3f} far={out['overlap']['far']['princ_cos_top3_med']:.3f} "
          f"(random~0.55)", flush=True)

    json.dump({"meta": {"seed": SEED, "M": M, "n_eval": N_EVAL, "scale": SCALE, "runtime_sec": time.time()-t0}, **out},
              open(OUT / "multiblock_why_raw.json", "w"), indent=2)
    print(f"[why] wrote multiblock_why_raw.json in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
