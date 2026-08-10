#!/usr/bin/env python3
"""Phase 2 / Section 5: static distance vs transfer defect as predictors of disagreement.

Per eval image (ID + Near + Far), computes:
  static:  regularized Mahalanobis distance (block-input GAP rep, calib-fit),
           ridge leverage (image-mean patch leverage in the original conv2 design),
  transfer: downstream-projected residual ||J.R_v|| (member-mean; the Section-4 bridge variable),
            raw local residual ||R_v||_F (member-mean),
  targets:  mutual information, logit-covariance trace, predictive entropy (M=50 ensemble).

Nested OLS (numpy) predicts each disagreement target from {distance, leverage, transfer, combined,
+dataset FE}; reports pooled R^2, within-dataset R^2, and incremental dR^2 in both directions.
Writes distance_transfer_raw.json + CIFAR_DISTANCE_TRANSFER.md.
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
from scipy import stats
import pickle

REPO = Path(__file__).resolve().parents[3]
os.chdir(REPO); sys.path.insert(0, str(REPO))
from models import PreActResNet18
from cifar_tasks import (_load_cifar_openood_context, _build_single_block_pnc_ensemble,
                         compute_cifar_block_preacts, CIFAROpenOODPnC)
from pnc import extract_patches, flatten_conv_kernel_to_patches

OUT = REPO / "results" / "neurips_2026_rebuttal" / "cifar"
SEED = 0; CALIB = 1024; CHUNK = 256; PHI_BATCH = 128
N_EVAL = 256; M_CHECK = 8; SCALE = 25.0; LAM = 1e-3; K = 20; SI, BI = 3, 0
SHRINK = 0.10


def softmax(z):
    z = z - z.max(-1, keepdims=True); e = np.exp(z); return e / e.sum(-1, keepdims=True)


def ols_r2(y, X):
    """OLS R^2 with intercept; X is (n,k) design (no intercept col)."""
    A = np.column_stack([np.ones(len(y)), X])
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ beta
    ss = ((y - y.mean())**2).sum()
    return float(1 - (resid @ resid) / (ss + 1e-30))


def main():
    t0 = time.time()
    task = CIFAROpenOODPnC(dataset="cifar10", epochs=300, perturbation_sizes=[SCALE], n_directions=K,
                           n_perturbations=50, subset_size=CALIB, chunk_size=CHUNK, target_stage_idx=SI,
                           target_block_idx=BI, random_directions=True, seed=SEED, lambda_reg=LAM,
                           posthoc_calibrate=True, bootstrap_frac=0.0)
    benchmark, x_tr, _, x_va, y_va, n_cls = _load_cifar_openood_context(task)
    model = PreActResNet18(n_classes=n_cls, rngs=nnx.Rngs(SEED))
    with open(task.input().path, "rb") as f:
        nnx.update(model, pickle.load(f)["state"])
    stages = [model.stage1, model.stage2, model.stage3, model.stage4]
    blk = stages[SI][BI]
    w1_orig = np.asarray(blk.conv1.kernel[...], np.float32)
    w2_orig = np.asarray(blk.conv2.kernel[...], np.float32)
    Theta = np.asarray(flatten_conv_kernel_to_patches(jnp.asarray(w2_orig)), np.float64)
    cout = w2_orig.shape[-1]

    ens, _ = _build_single_block_pnc_ensemble(nnx.clone(model), x_tr, target_stage_idx=SI, target_block_idx=BI,
        n_directions=K, n_perturbations=50, perturbation_scale=SCALE, subset_size=CALIB, chunk_size=CHUNK,
        lambda_reg=LAM, random_directions=True, seed=SEED, bootstrap_frac=0.0, bootstrap_seed=SEED)

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
    def block_input_rep(x):  # GAP of stage3 output = input to corrected block (256-d for s3)
        h = model.stem(jnp.asarray(x, jnp.float32))
        for s in range(SI):
            for b in range(len(stages[s])):
                h = stages[s][b](h, use_running_average=True)
        return np.asarray(jnp.mean(h, axis=(1, 2)), np.float64)

    def downstream(hb):
        h = hb
        for j in range(BI + 1, len(stages[SI])):
            h = stages[SI][j](h, use_running_average=True)
        for s in range(SI + 1, 4):
            for b2 in range(len(stages[s])):
                h = stages[s][b2](h, use_running_average=True)
        h = model.final_bn(h, use_running_average=True); h = jax.nn.relu(h)
        return model.fc(jnp.mean(h, axis=(1, 2)))

    @nnx.jit
    def base_block_out(x):
        h = model.stem(x)
        for s in range(4):
            for b2 in range(len(stages[s])):
                if s == SI and b2 == BI:
                    return stages[s][b2](h, use_running_average=True)
                h = stages[s][b2](h, use_running_average=True)

    # calibration: block-input rep (Mahalanobis fit) and original design Gram (leverage)
    cal_rep = np.concatenate([block_input_rep(x_tr[i:i+256]) for i in range(0, CALIB, 256)], 0)
    mu = cal_rep.mean(0); Xc = cal_rep - mu; cov = (Xc.T @ Xc) / (len(cal_rep) - 1)
    D = cov.shape[0]; cov_s = (1-SHRINK)*cov + SHRINK*(np.trace(cov)/D)*np.eye(D); prec = np.linalg.inv(cov_s)
    cal_h = np.concatenate([np.asarray(c) for c in ens.chunks], 0)
    Xcal = Phi(w1_orig, cal_h); Gorig = Xcal.T @ Xcal + LAM*np.eye(Xcal.shape[1]); Ginv = np.linalg.inv(Gorig)

    rng = np.random.RandomState(7)
    def take(x): return np.asarray(x)[rng.choice(len(x), min(N_EVAL, len(x)), replace=False)]
    evalsets = {"id_test": ("id", take(benchmark["id_test"]["inputs"])),
                "cifar100": ("near", take(benchmark["near_ood"]["cifar100"]["inputs"])),
                "tiny_imagenet": ("near", take(benchmark["near_ood"]["tiny_imagenet"]["inputs"])),
                "svhn": ("far", take(benchmark["far_ood"]["svhn"]["inputs"])),
                "textures": ("far", take(benchmark["far_ood"]["textures"]["inputs"]))}

    rows = []  # per-image records
    for name, (regime, X) in evalsets.items():
        h_in = compute_cifar_block_preacts(model, jnp.asarray(X), CHUNK, SI, BI, jnp.asarray(w1_orig))[0][0]
        rep = block_input_rep(X)
        maha = np.sqrt(np.maximum(np.einsum('ni,ij,nj->n', rep-mu, prec, rep-mu), 0))
        phi0 = Phi(w1_orig, h_in)
        N = len(X); npatch = phi0.shape[0] // N
        # ridge leverage per patch = sum(phi * (Ginv @ phi^T)^T), image-mean
        lev_patch = np.einsum('rp,pq,rq->r', phi0, Ginv, phi0)
        lev_img = lev_patch.reshape(N, npatch).mean(1)
        hb0 = np.asarray(base_block_out(jnp.asarray(X)), np.float64)
        all_logits = np.asarray(ens.predict(jnp.asarray(X)), np.float64)  # (M,N,C)
        base_logits = np.asarray(downstream(jnp.asarray(hb0, jnp.float32)), np.float64)
        # transfer variables (member-mean over M_CHECK)
        jproj = np.zeros(N); rfrob = np.zeros(N)
        for mi in range(min(M_CHECK, len(ens.members_w1))):
            w1p = np.asarray(ens.members_w1[mi], np.float32)
            w2c = np.asarray(ens.members_w2[mi][0], np.float64); b2c = np.asarray(ens.members_w2[mi][1], np.float64)
            phiv = Phi(w1p, h_in); onse = np.ones((phiv.shape[0], 1))
            Tva = np.concatenate([b2c, np.asarray(w2c).transpose(2, 0, 1, 3).reshape(-1, cout)], 0)
            R = (np.concatenate([onse, phiv], 1) @ Tva - phi0 @ Theta).reshape(N, -1, cout)
            jz = np.asarray(jax.jvp(downstream, (jnp.asarray(hb0, jnp.float32),),
                                    (jnp.asarray(R.reshape(hb0.shape), jnp.float32),))[1], np.float64)
            for j in range(N):
                jproj[j] += np.linalg.norm(jz[j]); rfrob[j] += np.linalg.norm(R[j])
        jproj /= M_CHECK; rfrob /= M_CHECK
        # disagreement targets (M=50)
        p = softmax(all_logits); mp = p.mean(0); eps = 1e-12
        pred_ent = -(mp*np.log(mp+eps)).sum(-1); mi_arr = pred_ent - (-(p*np.log(p+eps)).sum(-1).mean(0))
        lct = np.array([np.trace(np.cov(all_logits[:, j, :].T)) for j in range(N)])
        for j in range(N):
            rows.append(dict(dataset=name, regime=regime, maha=float(maha[j]), log_maha=float(np.log10(maha[j]+1e-9)),
                             leverage=float(lev_img[j]), jproj=float(jproj[j]), rfrob=float(rfrob[j]),
                             MI=float(mi_arr[j]), logit_cov_tr=float(lct[j]), pred_ent=float(pred_ent[j])))
        print(f"[dt] {name:14} N={N} med maha={np.median(maha):.2f} jproj={np.median(jproj):.3f} MI={np.median(mi_arr):.3f}", flush=True)

    import numpy as _np
    D = rows
    def col(k): return _np.array([r[k] for r in D])
    def z(a): return (a - a.mean())/(a.std()+1e-12)
    targets = ["MI", "logit_cov_tr", "pred_ent"]
    dist = _np.column_stack([z(col("log_maha")), z(col("leverage"))])   # static block
    transf = _np.column_stack([z(col("jproj")), z(col("rfrob"))])       # transfer block
    ds_dummies = None
    import pandas as pd
    ds_dummies = pd.get_dummies(pd.Series(col("dataset")), drop_first=True).to_numpy(float)
    analysis = {}
    for tgt in targets:
        y = z(col(tgt))
        r2_dist = ols_r2(y, dist); r2_tr = ols_r2(y, transf)
        r2_both = ols_r2(y, _np.column_stack([dist, transf]))
        r2_ds = ols_r2(y, ds_dummies)
        r2_dist_ds = ols_r2(y, _np.column_stack([dist, ds_dummies]))
        r2_tr_ds = ols_r2(y, _np.column_stack([transf, ds_dummies]))
        r2_both_ds = ols_r2(y, _np.column_stack([dist, transf, ds_dummies]))
        # within-dataset mean R^2
        wd_dist, wd_tr = [], []
        for name in set(col("dataset")):
            m = col("dataset") == name
            if m.sum() > 10:
                wd_dist.append(ols_r2(z(col(tgt)[m]), _np.column_stack([z(col("log_maha")[m]), z(col("leverage")[m])])))
                wd_tr.append(ols_r2(z(col(tgt)[m]), _np.column_stack([z(col("jproj")[m]), z(col("rfrob")[m])])))
        analysis[tgt] = {
            "R2_distance": r2_dist, "R2_transfer": r2_tr, "R2_both": r2_both, "R2_datasetFE": r2_ds,
            "dR2_transfer_given_distance": r2_both - r2_dist, "dR2_distance_given_transfer": r2_both - r2_tr,
            "dR2_transfer_given_distance_and_FE": r2_both_ds - r2_dist_ds,
            "dR2_distance_given_transfer_and_FE": r2_both_ds - r2_tr_ds,
            "within_dataset_R2_distance_mean": float(_np.mean(wd_dist)),
            "within_dataset_R2_transfer_mean": float(_np.mean(wd_tr)),
        }

    meta = {"seed": SEED, "block": f"s{SI}b{BI}", "scale": SCALE, "n_eval_per_dataset": N_EVAL,
            "m_check": M_CHECK, "n_total": len(rows), "runtime_sec": time.time()-t0}
    json.dump({"meta": meta, "analysis": analysis}, open(OUT / "distance_transfer_raw.json", "w"), indent=2)
    import csv
    with open(OUT / "distance_transfer_per_example.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    for tgt in targets:
        a = analysis[tgt]
        print(f"[dt] {tgt}: R2 dist={a['R2_distance']:.3f} transfer={a['R2_transfer']:.3f} both={a['R2_both']:.3f} | "
              f"dR2(tr|dist,FE)={a['dR2_transfer_given_distance_and_FE']:.3f} dR2(dist|tr,FE)={a['dR2_distance_given_transfer_and_FE']:.3f} | "
              f"within-ds R2 dist={a['within_dataset_R2_distance_mean']:.3f} tr={a['within_dataset_R2_transfer_mean']:.3f}", flush=True)
    print(f"[dt] wrote distance_transfer_raw.json in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
