#!/usr/bin/env python3
"""Phase 4 / Section 6: directional uncertainty — P&C covariance vs Deep Ensemble & error direction.

Per eval image, compare the leading eigenvectors of the P&C member-logit covariance (M=50, s3b0 anchor)
against (a) the Deep Ensemble covariance (5 independently-trained models, seeds 0-4), and (b) the ID
error direction. Reports covariance trace, principal angles between leading eigenspaces, MI, and
error-direction alignment; plus a distance-matched (Mahalanobis-binned) comparison. Writes
directional_uncertainty_raw.json.
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
SEED = 0; CALIB = 1024; CHUNK = 256; SCALE = 25.0; LAM = 1e-3; K = 20; SI, BI = 3, 0; M = 50
N_EVAL = 400; BATCH = 200
RECIPE = "_e300_optsgd_lr1e-01_wd5e-04_bs128_wu5_mom0p9_n1_augfcco8_ls0"
NEAR = ["cifar100", "tiny_imagenet"]; FAR = ["mnist", "svhn", "textures", "places365"]


def softmax(z):
    z = z - z.max(-1, keepdims=True); e = np.exp(z); return e / e.sum(-1, keepdims=True)
def mlogits(ens, x):
    return np.concatenate([np.asarray(ens.predict(jnp.asarray(x[i:i+BATCH]))) for i in range(0, len(x), BATCH)], axis=1)


def main():
    t0 = time.time()
    task = CIFAROpenOODPnC(dataset="cifar10", epochs=300, perturbation_sizes=[SCALE], n_directions=K,
                           n_perturbations=M, subset_size=CALIB, chunk_size=CHUNK, target_stage_idx=SI,
                           target_block_idx=BI, random_directions=True, seed=SEED, lambda_reg=LAM,
                           posthoc_calibrate=True, bootstrap_frac=0.05)
    benchmark, x_tr, _, x_va, y_va, n_cls = _load_cifar_openood_context(task)
    base = PreActResNet18(n_classes=n_cls, rngs=nnx.Rngs(SEED))
    with open(task.input().path, "rb") as f:
        nnx.update(base, pickle.load(f)["state"])
    ens, _ = _build_single_block_pnc_ensemble(nnx.clone(base), x_tr, target_stage_idx=SI, target_block_idx=BI,
        n_directions=K, n_perturbations=M, perturbation_scale=SCALE, subset_size=CALIB, chunk_size=CHUNK,
        lambda_reg=LAM, random_directions=True, seed=SEED, bootstrap_frac=0.05, bootstrap_seed=SEED)
    # Deep ensemble: 5 base models
    de_models = []
    for s in range(5):
        m = PreActResNet18(n_classes=n_cls, rngs=nnx.Rngs(s))
        with open(f"results/cifar10/preact_resnet18_train{RECIPE}_seed{s}.pkl", "rb") as f:
            nnx.update(m, pickle.load(f)["state"])
        de_models.append(m)
    @nnx.jit
    def _fwd(m, x): return m(x, use_running_average=True)
    def de_logits(x):
        outs = []
        for m in de_models:
            outs.append(np.concatenate([np.asarray(_fwd(m, jnp.asarray(x[i:i+BATCH]))) for i in range(0, len(x), BATCH)], 0))
        return np.stack(outs, 0)  # (5,N,C)
    print(f"[dir] built P&C + DE ({time.time()-t0:.0f}s)", flush=True)

    rng = np.random.RandomState(7)
    def take_xy(src, key, d=None):
        arr = src["inputs"]; N = min(N_EVAL, len(arr)); sel = rng.choice(len(arr), N, replace=False)
        return np.asarray(arr)[sel], (np.asarray(src["targets"])[sel] if src.get("targets") is not None else None)
    evalsets = {"id_test": ("id", benchmark["id_test"])}
    for d in NEAR: evalsets[d] = ("near", benchmark["near_ood"][d])
    for d in FAR: evalsets[d] = ("far", benchmark["far_ood"][d])

    def lead_subspace(cov, k):
        w, V = np.linalg.eigh(cov); return V[:, ::-1][:, :k]  # top-k eigenvectors
    def princ_cos(A, B):
        try: return float(np.mean(np.cos(subspace_angles(A, B))))
        except Exception: return float("nan")

    results = {}
    for name, (regime, src) in evalsets.items():
        X, Y = take_xy(src, name); N = len(X)
        Lp = mlogits(ens, X)            # (M,N,C)
        Ld = de_logits(X)              # (5,N,C)
        pcov_tr = np.zeros(N); dcov_tr = np.zeros(N)
        cos1 = np.zeros(N); ang12 = np.zeros(N); ang13 = np.zeros(N)
        err_align_p = np.full(N, np.nan); err_align_d = np.full(N, np.nan)
        mp_p = softmax(Lp).mean(0); mp_d = softmax(Ld).mean(0); eps = 1e-12
        MI_p = (-(mp_p*np.log(mp_p+eps)).sum(-1)) - (-(softmax(Lp)*np.log(softmax(Lp)+eps)).sum(-1).mean(0))
        MI_d = (-(mp_d*np.log(mp_d+eps)).sum(-1)) - (-(softmax(Ld)*np.log(softmax(Ld)+eps)).sum(-1).mean(0))
        for j in range(N):
            Cp = np.cov(Lp[:, j, :].T); Cd = np.cov(Ld[:, j, :].T)
            pcov_tr[j] = np.trace(Cp); dcov_tr[j] = np.trace(Cd)
            Vp = lead_subspace(Cp, 3); Vd = lead_subspace(Cd, 3)
            cos1[j] = abs(float(Vp[:, 0] @ Vd[:, 0]))
            ang12[j] = princ_cos(Vp[:, :2], Vd[:, :2]); ang13[j] = princ_cos(Vp[:, :3], Vd[:, :3])
            if regime == "id" and Y is not None:
                e = np.zeros(n_cls); e[int(Y[j])] = 1.0
                gerr = mp_p[j] - e  # logit-loss gradient direction (== -error prob direction)
                gerr = gerr / (np.linalg.norm(gerr) + eps)
                err_align_p[j] = abs(float(Vp[:, 0] @ gerr)); err_align_d[j] = abs(float(Vd[:, 0] @ gerr))
        results[name] = {
            "regime": regime, "n": N,
            "pnc_cov_trace_med": float(np.median(pcov_tr)), "de_cov_trace_med": float(np.median(dcov_tr)),
            "cov_trace_ratio_pnc_over_de": float(np.median(pcov_tr) / (np.median(dcov_tr) + eps)),
            "lead_eigvec_cos_pnc_de_med": float(np.median(cos1)),
            "princ_cos_top2_med": float(np.nanmedian(ang12)), "princ_cos_top3_med": float(np.nanmedian(ang13)),
            "MI_pnc_med": float(np.median(MI_p)), "MI_de_med": float(np.median(MI_d)),
            "err_align_pnc_med": float(np.nanmedian(err_align_p)) if regime == "id" else None,
            "err_align_de_med": float(np.nanmedian(err_align_d)) if regime == "id" else None,
        }
        r = results[name]
        print(f"[dir] {name:14} ({regime}) cov_tr P&C/DE={r['cov_trace_ratio_pnc_over_de']:.2f} "
              f"lead_cos={r['lead_eigvec_cos_pnc_de_med']:.3f} top3_cos={r['princ_cos_top3_med']:.3f} "
              f"MI P&C/DE={r['MI_pnc_med']:.4f}/{r['MI_de_med']:.4f}", flush=True)

    json.dump({"meta": {"seed": SEED, "M": M, "de_members": 5, "n_eval": N_EVAL, "runtime_sec": time.time()-t0},
               "results": results}, open(OUT / "directional_uncertainty_raw.json", "w"), indent=2)
    print(f"[dir] wrote directional_uncertainty_raw.json in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
