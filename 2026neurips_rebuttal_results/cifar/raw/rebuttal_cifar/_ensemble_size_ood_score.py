#!/usr/bin/env python3
"""Phase 4 / Sections 14 + 15: ensemble-size convergence and OOD-score analysis.

Build the anchor s3b0 P&C ensemble with a M=100 member pool (ps25, bf0.05, seed 0), get all
100 member logits on ID test + 2 Near + 4 Far datasets (subsampled), then:
  Section 14: for M in {2,4,8,16,32,50,64,100} (multiple random member-subsets), track convergence
              of predictive entropy, MI, Near/Far AUROC & FPR95, logit-covariance trace.
  Section 15: at M=50 (submitted) compute ALL scalar OOD scores' per-dataset AUROC/FPR95 under two
              temperature conditions (T=1 and the ID-fit anchor temperature).
Writes ensemble_size_ood_score_raw.json.
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
from util import _fit_posthoc_temperature

OUT = REPO / "results" / "neurips_2026_rebuttal" / "cifar"
SEED = 0; CALIB = 1024; CHUNK = 256; SCALE = 25.0; LAM = 1e-3; K = 20; SI, BI = 3, 0
M_POOL = 100; N_EVAL = 800; BATCH = 200
M_GRID = [2, 4, 8, 16, 32, 50, 64, 100]; N_SUBSETS = 5
NEAR = ["cifar100", "tiny_imagenet"]; FAR = ["mnist", "svhn", "textures", "places365"]


def softmax(z, T=1.0):
    z = z / T; z = z - z.max(-1, keepdims=True); e = np.exp(z); return e / e.sum(-1, keepdims=True)


def member_logits(ens, x, bs=BATCH):
    return np.concatenate([np.asarray(ens.predict(jnp.asarray(x[i:i+bs]))) for i in range(0, len(x), bs)], axis=1)


def scores_from_logits(L, T=1.0):
    """L (M,N,C) -> dict of per-example OOD scores (higher = more OOD)."""
    p = softmax(L, T)                       # (M,N,C)
    mp = p.mean(0); eps = 1e-12
    pred_ent = -(mp*np.log(mp+eps)).sum(-1)
    exp_ent = -(p*np.log(p+eps)).sum(-1).mean(0)
    mi = pred_ent - exp_ent
    msp = 1 - mp.max(-1)
    var_ratio = 1 - (p.argmax(-1)[:, :, None] == np.arange(p.shape[-1])).mean(0).max(-1)
    mean_kl = (p*(np.log(p+eps) - np.log(mp[None]+eps))).sum(-1).mean(0)
    logit_cov_tr = np.array([np.trace(np.cov(L[:, j, :].T)) for j in range(L.shape[1])])
    prob_cov_tr = np.array([np.trace(np.cov(p[:, j, :].T)) for j in range(L.shape[1])])
    energy = (-1.0*np.log(np.exp(L.mean(0)).sum(-1) + eps))  # -logsumexp of mean logits (scalar baseline)
    return {"predictive_entropy": pred_ent, "mutual_information": mi, "expected_entropy": exp_ent,
            "max_softmax_unc": msp, "variation_ratio": var_ratio, "mean_member_kl": mean_kl,
            "logit_cov_tr": logit_cov_tr, "prob_cov_tr": prob_cov_tr, "energy": energy}


def auroc(id_s, ood_s):
    a = np.concatenate([id_s, ood_s]); order = a.argsort(); ranks = np.empty(len(a)); ranks[order] = np.arange(1, len(a)+1)
    n1, n2 = len(ood_s), len(id_s)
    return (ranks[len(id_s):].sum() - n1*(n1+1)/2) / (n1*n2)


def fpr95(id_s, ood_s):
    t = np.quantile(ood_s, 0.05); return float((id_s >= t).mean())


def main():
    t0 = time.time()
    task = CIFAROpenOODPnC(dataset="cifar10", epochs=300, perturbation_sizes=[SCALE], n_directions=K,
                           n_perturbations=M_POOL, subset_size=CALIB, chunk_size=CHUNK, target_stage_idx=SI,
                           target_block_idx=BI, random_directions=True, seed=SEED, lambda_reg=LAM,
                           posthoc_calibrate=True, bootstrap_frac=0.05)
    benchmark, x_tr, _, x_va, y_va, n_cls = _load_cifar_openood_context(task)
    model = PreActResNet18(n_classes=n_cls, rngs=nnx.Rngs(SEED))
    with open(task.input().path, "rb") as f:
        nnx.update(model, pickle.load(f)["state"])
    ens, _ = _build_single_block_pnc_ensemble(nnx.clone(model), x_tr, target_stage_idx=SI, target_block_idx=BI,
        n_directions=K, n_perturbations=M_POOL, perturbation_scale=SCALE, subset_size=CALIB, chunk_size=CHUNK,
        lambda_reg=LAM, random_directions=True, seed=SEED, bootstrap_frac=0.05, bootstrap_seed=SEED)
    print(f"[es] built M={M_POOL} pool ({time.time()-t0:.0f}s)", flush=True)

    rng = np.random.RandomState(7)
    def take(x): return np.asarray(x)[rng.choice(len(x), min(N_EVAL, len(x)), replace=False)]
    id_x = take(benchmark["id_test"]["inputs"])
    L = {"id": member_logits(ens, id_x)}
    for d in NEAR: L[d] = member_logits(ens, take(benchmark["near_ood"][d]["inputs"]))
    for d in FAR: L[d] = member_logits(ens, take(benchmark["far_ood"][d]["inputs"]))
    print(f"[es] got member logits on {len(L)} datasets ({time.time()-t0:.0f}s)", flush=True)

    # fit anchor temperature on ID val (mixture NLL)
    val_L = member_logits(ens, np.asarray(x_va))
    T = float(_fit_posthoc_temperature(jnp.asarray(val_L), jnp.asarray(np.asarray(y_va).astype(int))))
    print(f"[es] fitted T={T:.3f}", flush=True)

    # --- Section 14: ensemble-size convergence (predictive_entropy score) ---
    def macro_auroc(subL, group, T=1.0):
        ids = scores_from_logits(subL["id"], T)["predictive_entropy"]
        return float(np.mean([auroc(ids, scores_from_logits(subL[d], T)["predictive_entropy"]) for d in group]))
    es_curve = []
    for M in M_GRID:
        nsub = 1 if M == M_POOL else N_SUBSETS
        na, fa, pe, mi, lct = [], [], [], [], []
        for s in range(nsub):
            idx = np.arange(M_POOL)[:M] if M == M_POOL else np.random.RandomState(100+s).choice(M_POOL, M, replace=False)
            subL = {d: L[d][idx] for d in L}
            na.append(macro_auroc(subL, NEAR, T)); fa.append(macro_auroc(subL, FAR, T))
            sc = scores_from_logits(subL["id"], T); pe.append(sc["predictive_entropy"].mean()); mi.append(sc["mutual_information"].mean())
            lct.append(sc["logit_cov_tr"].mean())
        es_curve.append({"M": M, "near_auroc_mean": float(np.mean(na))*100, "near_auroc_std": float(np.std(na))*100,
                         "far_auroc_mean": float(np.mean(fa))*100, "far_auroc_std": float(np.std(fa))*100,
                         "id_pred_ent": float(np.mean(pe)), "id_MI": float(np.mean(mi)), "id_logit_cov_tr": float(np.mean(lct))})
        print(f"[es] M={M:3d} nearAUROC={np.mean(na)*100:.2f}±{np.std(na)*100:.2f} farAUROC={np.mean(fa)*100:.2f} "
              f"id_MI={np.mean(mi):.4f}", flush=True)

    # --- Section 15: OOD-score analysis at M=50, T=1 and T=fitted ---
    score_analysis = {}
    for Tcond, Tval in [("T1", 1.0), ("Tfit", T)]:
        L50 = {d: L[d][:50] for d in L}
        sid = scores_from_logits(L50["id"], Tval)
        per = {}
        for score in sid:
            def dsauroc(d):
                so = scores_from_logits(L50[d], Tval)[score]; return auroc(sid[score], so)*100, fpr95(sid[score], so)*100
            near = [dsauroc(d) for d in NEAR]; far = [dsauroc(d) for d in FAR]
            per[score] = {"near_auroc": float(np.mean([a for a, _ in near])), "near_fpr95": float(np.mean([f for _, f in near])),
                          "far_auroc": float(np.mean([a for a, _ in far])), "far_fpr95": float(np.mean([f for _, f in far]))}
        score_analysis[Tcond] = per
    # best score per condition
    for Tcond in score_analysis:
        best = max(score_analysis[Tcond], key=lambda s: score_analysis[Tcond][s]["near_auroc"] + score_analysis[Tcond][s]["far_auroc"])
        print(f"[es] {Tcond}: best score = {best} (near {score_analysis[Tcond][best]['near_auroc']:.2f} / far {score_analysis[Tcond][best]['far_auroc']:.2f})", flush=True)

    meta = {"seed": SEED, "block": f"s{SI}b{BI}", "scale": SCALE, "M_pool": M_POOL, "n_eval": N_EVAL,
            "fitted_T": T, "near": NEAR, "far": FAR, "runtime_sec": time.time()-t0}
    json.dump({"meta": meta, "ensemble_size_curve": es_curve, "ood_score_analysis": score_analysis},
              open(OUT / "ensemble_size_ood_score_raw.json", "w"), indent=2)
    print(f"[es] wrote ensemble_size_ood_score_raw.json in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
