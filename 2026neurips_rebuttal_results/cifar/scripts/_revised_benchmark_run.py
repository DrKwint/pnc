#!/usr/bin/env python3
"""Phase 5 / Section 16: revised benchmark run — anchor P&C on FULL OpenOOD sets, 3 seeds,
computing BOTH predictive_entropy (submitted score) and logit_cov_tr (revised score).

For seeds 0,1,2: build the anchor ensemble (s3b0, K=20, M=50, ps=25, bf=0.05), fit temperature on the
ID val split, then evaluate on the FULL ID test + Near {cifar100,tiny_imagenet} + Far {mnist,svhn,textures,
places365}. Report ID acc/NLL/ECE and, for BOTH scores, macro Near/Far AUROC + FPR95 (matching the submitted
protocol: predictive_entropy = temperature-scaled mixture entropy; logit_cov = trace of member logit covariance,
temperature-invariant). Cross-checks predictive_entropy AUROC against the cached submitted JSON.
Resumable: writes per-seed results incrementally. Writes revised_benchmark_raw.json.
"""
from __future__ import annotations
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import sys, json, time, glob
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
RAW = OUT / "revised_benchmark_raw.json"
SEEDS = [0, 1, 2]; CALIB = 1024; CHUNK = 1024; SCALE = 25.0; LAM = 1e-3; K = 20; M = 50; SI, BI = 3, 0
BATCH = 250
NEAR = ["cifar100", "tiny_imagenet"]; FAR = ["mnist", "svhn", "textures", "places365"]


def softmax(z, T=1.0):
    z = z / T; z = z - z.max(-1, keepdims=True); e = np.exp(z); return e / e.sum(-1, keepdims=True)
def mlog(ens, x):
    return np.concatenate([np.asarray(ens.predict(jnp.asarray(x[i:i+BATCH]))) for i in range(0, len(x), BATCH)], axis=1)
def auroc(a, b):
    s = np.concatenate([a, b]); o = s.argsort(); r = np.empty(len(s)); r[o] = np.arange(1, len(s)+1)
    n1, n2 = len(b), len(a); return (r[len(a):].sum()-n1*(n1+1)/2)/(n1*n2)
def fpr95(a, b): return float((a >= np.quantile(b, 0.05)).mean())
def pe_score(L, T):
    m = softmax(L, T).mean(0); eps = 1e-12; return -(m*np.log(m+eps)).sum(-1)
def lc_score(L):
    return np.array([np.trace(np.cov(L[:, j, :].T)) for j in range(L.shape[1])])
def ece_score(probs, labels, nb=15):
    conf = probs.max(1); pred = probs.argmax(1); acc = (pred == labels).astype(float)
    bins = np.linspace(0, 1, nb+1); e = 0.0
    for i in range(nb):
        m = (conf > bins[i]) & (conf <= bins[i+1])
        if m.sum() > 0: e += m.mean()*abs(acc[m].mean()-conf[m].mean())
    return float(e)


def cached_pe_auroc(seed):
    fs = glob.glob(f"results/cifar10/openood_v1p5_pnc_single_block_vcal_s3b0_k20_n50_ps25.0_lr0.001_bf0.05_*_seed{seed}_random.json")
    if not fs: return None
    e = json.load(open(fs[0]))["25.0"]
    return {"near_auroc": e["near_ood_auroc"]*100, "far_auroc": e["far_ood_auroc"]*100}


def main():
    all_res = json.load(open(RAW)) if RAW.exists() else {"per_seed": {}}
    for seed in SEEDS:
        if str(seed) in all_res["per_seed"]:
            print(f"[rb] seed {seed} cached, skip", flush=True); continue
        t0 = time.time()
        task = CIFAROpenOODPnC(dataset="cifar10", epochs=300, perturbation_sizes=[SCALE], n_directions=K,
                               n_perturbations=M, subset_size=CALIB, chunk_size=CHUNK, target_stage_idx=SI,
                               target_block_idx=BI, random_directions=True, seed=seed, lambda_reg=LAM,
                               posthoc_calibrate=True, bootstrap_frac=0.05)
        benchmark, x_tr, _, x_va, y_va, n_cls = _load_cifar_openood_context(task)
        model = PreActResNet18(n_classes=n_cls, rngs=nnx.Rngs(seed))
        with open(task.input().path, "rb") as f:
            nnx.update(model, pickle.load(f)["state"])
        ens, _ = _build_single_block_pnc_ensemble(nnx.clone(model), x_tr, target_stage_idx=SI, target_block_idx=BI,
            n_directions=K, n_perturbations=M, perturbation_scale=SCALE, subset_size=CALIB, chunk_size=CHUNK,
            lambda_reg=LAM, random_directions=True, seed=seed, bootstrap_frac=0.05, bootstrap_seed=seed)
        # temperature on ID val
        T = float(_fit_posthoc_temperature(jnp.asarray(mlog(ens, np.asarray(x_va))), jnp.asarray(np.asarray(y_va).astype(int))))
        # ID test (full)
        Lid = mlog(ens, np.asarray(benchmark["id_test"]["inputs"])); yid = np.asarray(benchmark["id_test"]["targets"]).astype(int)
        mp = softmax(Lid, T).mean(0); eps = 1e-12
        acc = float((mp.argmax(-1) == yid).mean())*100; nll = float(-np.log(mp[np.arange(len(yid)), yid]+eps).mean())
        ece = ece_score(mp, yid)
        id_pe = pe_score(Lid, T); id_lc = lc_score(Lid)
        # OOD (full)
        def macro(group, fam, which):
            aus, frs = [], []
            for d in group:
                Lo = mlog(ens, np.asarray(benchmark[fam][d]["inputs"]))
                so = pe_score(Lo, T) if which == "pe" else lc_score(Lo)
                si = id_pe if which == "pe" else id_lc
                aus.append(auroc(si, so)*100); frs.append(fpr95(si, so)*100)
            return float(np.mean(aus)), float(np.mean(frs))
        res = {"T": T, "id_acc": acc, "id_nll": nll, "id_ece": ece}
        for which in ["pe", "lc"]:
            na, nf = macro(NEAR, "near_ood", which); fa, ff = macro(FAR, "far_ood", which)
            res[which] = {"near_auroc": na, "near_fpr95": nf, "far_auroc": fa, "far_fpr95": ff}
        res["cached_pe_check"] = cached_pe_auroc(seed)
        res["runtime_sec"] = time.time()-t0
        all_res["per_seed"][str(seed)] = res
        json.dump(all_res, open(RAW, "w"), indent=2)
        ck = res["cached_pe_check"]
        print(f"[rb] seed {seed} T={T:.3f} acc={acc:.2f} | pe near/far AUROC {res['pe']['near_auroc']:.2f}/{res['pe']['far_auroc']:.2f} "
              f"(cached {ck['near_auroc']:.2f}/{ck['far_auroc']:.2f}) | lc {res['lc']['near_auroc']:.2f}/{res['lc']['far_auroc']:.2f} "
              f"({res['runtime_sec']:.0f}s)", flush=True)

    # aggregate
    seeds = sorted(all_res["per_seed"].keys())
    def agg(path):
        vals = [_get(all_res["per_seed"][s], path) for s in seeds]
        return float(np.mean(vals)), float(np.std(vals))
    def _get(d, path):
        for p in path.split("."): d = d[p]
        return d
    summary = {}
    for m in ["id_acc", "id_nll", "id_ece"]:
        summary[m] = agg(m)
    for which in ["pe", "lc"]:
        summary[which] = {k: agg(f"{which}.{k}") for k in ["near_auroc", "near_fpr95", "far_auroc", "far_fpr95"]}
    all_res["summary_3seed"] = summary
    json.dump(all_res, open(RAW, "w"), indent=2)
    def pm(t): return f"{t[0]:.2f}±{t[1]:.2f}"
    print(f"\n[rb] 3-SEED SUMMARY (acc {pm(summary['id_acc'])} nll {pm(summary['id_nll'])}):")
    for which in ["pe", "lc"]:
        s = summary[which]
        print(f"  {which}: Near AUROC {pm(s['near_auroc'])} FPR95 {pm(s['near_fpr95'])} | Far AUROC {pm(s['far_auroc'])} FPR95 {pm(s['far_fpr95'])}", flush=True)


if __name__ == "__main__":
    main()
