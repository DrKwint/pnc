#!/usr/bin/env python3
"""Phase 4 (CIFAR distance-disagreement mechanism) -- EXTRACTION stage.

Builds the anchor single-block PnC ensemble (s3b0/ps25/bf0.05, seed 0), then for
the exact ID calibration subset + every eval split (ID test, near-OOD, far-OOD)
computes, per example:
  * regularized Mahalanobis distance to the calibration representation distribution,
    in TWO representations:
      - block_input  (256-d, GAP of stage3 output = input to the corrected stage4[0]
                      block; PRIMARY -- most analogous to the correction geometry),
      - penult       (512-d penultimate features; secondary robustness check),
  * disagreement quantities from the 50-member ensemble vs the base predictor:
      - predictive_entropy (of ensemble-mean probs),
      - mutual_information (pred entropy - mean member entropy),
      - kl_to_base (mean_m KL(p_member || p_base)),
      - prob_l2   (mean_m || p_member - p_base ||_2),
      - logit_l2  (mean_m || z_member - z_base ||_2).
Disagreement is computed on RAW (untempered) logits -- documented; a single global
temperature is a monotone rescale that would only damp MI, and Spearman is rank-based.

Writes: mechanism_cifar_per_example.csv  (+ _raw.npz with the fitted estimator).
Subsampling: balanced N_SUB per dataset with a fixed seed; counts recorded.
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
import csv

REPO = Path(__file__).resolve().parents[3]
os.chdir(REPO); sys.path.insert(0, str(REPO))
from cifar_tasks import CIFAROpenOODPnC, _load_cifar_openood_context, _build_single_block_pnc_ensemble  # noqa
from models import PreActResNet18  # noqa
from util import seed_everything  # noqa

OUT = REPO / "results" / "neurips_2026_rebuttal" / "cifar"
SEED = 0
N_SUB = 2500            # per-dataset subsample cap
BATCH = 200
ANCHOR = dict(dataset="cifar10", epochs=300, n_directions=20, n_perturbations=50,
              perturbation_sizes=[25.0], subset_size=1024, chunk_size=1024,
              target_stage_idx=3, target_block_idx=0, random_directions=True,
              seed=SEED, lambda_reg=1e-3, posthoc_calibrate=True, bootstrap_frac=0.05)
SHRINKAGE = 0.10        # Ledoit-Wolf-style diagonal shrinkage for the covariance


def block_input_rep(model, x):
    """GAP of stage3 output = input activations to the corrected block stage4[0]."""
    h = model.stem(x)
    for blk in model.stage1: h = blk(h, use_running_average=True)
    for blk in model.stage2: h = blk(h, use_running_average=True)
    for blk in model.stage3: h = blk(h, use_running_average=True)
    return jnp.mean(h, axis=(1, 2))  # (batch, 256)


@nnx.jit
def _rep_both(model, x):
    return block_input_rep(model, x), model.features(x, use_running_average=True)


@nnx.jit
def _base_logits(model, x):
    return model(x, use_running_average=True)


def batched(fn, x, bs=BATCH):
    outs = [np.asarray(fn(jnp.asarray(x[i:i+bs]))) for i in range(0, len(x), bs)]
    return np.concatenate(outs, 0)


def ens_logits(ens, x, bs=BATCH):
    outs = []
    for i in range(0, len(x), bs):
        outs.append(np.asarray(ens.predict(jnp.asarray(x[i:i+bs]))))  # (M, b, C)
    return np.concatenate(outs, axis=1)  # (M, N, C)


def softmax_np(z):
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z); return e / e.sum(-1, keepdims=True)


def disagreement(ens_lg, base_lg):
    # ens_lg (M,N,C), base_lg (N,C) -- raw logits
    p = softmax_np(ens_lg)                      # (M,N,C)
    mp = p.mean(0)                              # (N,C)
    eps = 1e-12
    pred_ent = -(mp * np.log(mp + eps)).sum(-1)                     # (N,)
    mem_ent = -(p * np.log(p + eps)).sum(-1).mean(0)               # (N,)
    mi = pred_ent - mem_ent                                         # (N,)
    pb = softmax_np(base_lg)                                        # (N,C)
    kl = (p * (np.log(p + eps) - np.log(pb[None] + eps))).sum(-1).mean(0)   # (N,)
    prob_l2 = np.linalg.norm(p - pb[None], axis=-1).mean(0)                 # (N,)
    logit_l2 = np.linalg.norm(ens_lg - base_lg[None], axis=-1).mean(0)      # (N,)
    return dict(predictive_entropy=pred_ent, mutual_information=mi,
                kl_to_base=kl, prob_l2=prob_l2, logit_l2=logit_l2)


def fit_maha(reps):
    mu = reps.mean(0)
    Xc = reps - mu
    cov = (Xc.T @ Xc) / (len(reps) - 1)
    D = cov.shape[0]
    shrunk = (1 - SHRINKAGE) * cov + SHRINKAGE * (np.trace(cov) / D) * np.eye(D)
    prec = np.linalg.inv(shrunk)
    return mu.astype(np.float64), prec.astype(np.float64)


def maha_dist(reps, mu, prec):
    Xc = reps.astype(np.float64) - mu
    d2 = np.einsum('ni,ij,nj->n', Xc, prec, Xc)
    return np.sqrt(np.maximum(d2, 0.0))


def main():
    t0 = time.time()
    seed_everything(SEED)
    task = CIFAROpenOODPnC(**ANCHOR)
    benchmark, x_tr, _, x_va, y_va, n_cls = _load_cifar_openood_context(task)

    model = PreActResNet18(n_classes=n_cls, rngs=nnx.Rngs(SEED))
    with open(task.input().path, "rb") as f:
        ckpt = pickle.load(f)
    nnx.update(model, ckpt["state"])

    # exact calibration subset used by the correction
    actual_sub = min(len(x_tr), ANCHOR["subset_size"])
    idx = np.random.RandomState(SEED).choice(len(x_tr), actual_sub, replace=False)
    x_cal = x_tr[idx]
    print(f"[mech] calibration subset: {len(x_cal)} images (exact P&C set)", flush=True)

    # build anchor ensemble
    ens, build_meta = _build_single_block_pnc_ensemble(nnx.clone(model), x_tr, **{
        k: ANCHOR[k] for k in ("target_stage_idx", "target_block_idx", "n_directions",
                               "n_perturbations", "subset_size", "chunk_size",
                               "lambda_reg", "random_directions", "seed")},
        perturbation_scale=25.0, bootstrap_frac=0.05, bootstrap_seed=SEED)
    print(f"[mech] anchor ensemble built ({time.time()-t0:.0f}s)", flush=True)

    # fit Mahalanobis on calibration reps (both representations)
    cal_bi, cal_pen = [], []
    for i in range(0, len(x_cal), BATCH):
        bi, pen = _rep_both(model, jnp.asarray(x_cal[i:i+BATCH]))
        cal_bi.append(np.asarray(bi)); cal_pen.append(np.asarray(pen))
    cal_bi = np.concatenate(cal_bi); cal_pen = np.concatenate(cal_pen)
    mu_bi, prec_bi = fit_maha(cal_bi)
    mu_pen, prec_pen = fit_maha(cal_pen)
    print(f"[mech] Mahalanobis fit: block_input D={cal_bi.shape[1]}, penult D={cal_pen.shape[1]}", flush=True)

    # eval splits
    splits = [("id_test", "id", benchmark["id_test"]["inputs"], benchmark["id_test"]["targets"])]
    for fam, regime in (("near_ood", "near"), ("far_ood", "far")):
        for dkey, d in benchmark[fam].items():
            splits.append((dkey, regime, d["inputs"], d["targets"]))

    rng_sub = np.random.RandomState(1234)
    rows = []
    counts = {}
    for name, regime, X, Y in splits:
        n = len(X)
        take = min(N_SUB, n)
        sel = rng_sub.choice(n, take, replace=False)
        Xs = X[sel]
        counts[name] = int(take)
        # representations + distances
        bi = batched(lambda xb: _rep_both(model, xb)[0], Xs)
        pen = batched(lambda xb: _rep_both(model, xb)[1], Xs)
        db = maha_dist(bi, mu_bi, prec_bi)
        dp = maha_dist(pen, mu_pen, prec_pen)
        # disagreement
        el = ens_logits(ens, Xs)
        bl = batched(lambda xb: _base_logits(model, xb), Xs)
        dis = disagreement(el, bl)
        base_pred = softmax_np(bl).argmax(-1)
        correct = (base_pred == np.asarray(Y[sel]).astype(int)).astype(int) if regime == "id" else -1
        for j in range(take):
            rows.append(dict(
                dataset=name, regime=regime,
                maha_block_input=float(db[j]), log10_dist=float(np.log10(db[j] + 1e-12)),
                maha_penult=float(dp[j]),
                predictive_entropy=float(dis["predictive_entropy"][j]),
                mutual_information=float(dis["mutual_information"][j]),
                kl_to_base=float(dis["kl_to_base"][j]),
                prob_l2=float(dis["prob_l2"][j]),
                logit_l2=float(dis["logit_l2"][j]),
                base_correct=int(correct[j]) if regime == "id" else -1,
            ))
        print(f"[mech] {name:14} ({regime}) n={take}  med_dist={np.median(db):.3f}  "
              f"med_predent={np.median(dis['predictive_entropy']):.4f}  "
              f"med_MI={np.median(dis['mutual_information']):.4f}", flush=True)

    # write CSV
    csv_path = OUT / "mechanism_cifar_per_example.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    np.savez(OUT / "mechanism_cifar_raw.npz",
             mu_bi=mu_bi, prec_bi=prec_bi, mu_pen=mu_pen, prec_pen=prec_pen,
             shrinkage=SHRINKAGE, counts=json.dumps(counts))
    meta = dict(seed=SEED, anchor=ANCHOR, n_sub=N_SUB, per_dataset_counts=counts,
                shrinkage=SHRINKAGE, representation_primary="block_input(stage3 GAP,256d)",
                representation_secondary="penultimate(512d)",
                temperature_applied=False, runtime_sec=time.time()-t0,
                calibration_images=int(len(x_cal)))
    json.dump(meta, open(OUT / "mechanism_cifar_extract_meta.json", "w"), indent=2)
    print(f"[mech] wrote {csv_path} ({len(rows)} rows) in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
