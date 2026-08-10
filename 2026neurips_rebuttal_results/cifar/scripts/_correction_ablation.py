#!/usr/bin/env python3
"""Phase 5 -- Correction vs no-correction ablation (CIFAR-10, anchor lineage, seed 0).

For a grid of perturbation scales, builds the anchor single-block PnC ensemble
(s3b0, K20, M50, bf0.05) ONCE per scale, then evaluates TWO variants that share the
IDENTICAL perturbed conv1 (same directions/coefficients/scale/members):
  * corrected   : the fitted affine conv2 correction (w2_new, b2_new),
  * uncorrected : original conv2 (w2_orig, bias 0)  [members_w2 override].

Reports per (scale, variant), on balanced subsamples:
  ID accuracy, ID NLL, Near/Far AUROC + FPR95 (predictive_entropy score),
  ID predictive entropy, ID mutual information, Near/Far predictive entropy & MI,
  hidden perturbation magnitude (mean member conv1 L2 delta = uncorrected block-output
  shift), corrected block-output shift, and mean logit/prob change-from-base on ID.

The key claim: correction lets the hidden (conv1) perturbation be LARGE while keeping ID
outputs close to the base (ID acc/NLL preserved, small ID logit change) yet preserving
OOD disagreement -- whereas without correction, large perturbations wreck ID too.
"""
from __future__ import annotations
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import sys, json, time, csv
from pathlib import Path
import numpy as np
import jax.numpy as jnp
from flax import nnx
import pickle

REPO = Path(__file__).resolve().parents[3]
os.chdir(REPO); sys.path.insert(0, str(REPO))
from cifar_tasks import CIFAROpenOODPnC, _load_cifar_openood_context, _build_single_block_pnc_ensemble  # noqa
from models import PreActResNet18  # noqa
from util import seed_everything  # noqa

OUT = REPO / "results" / "neurips_2026_rebuttal" / "cifar"
SEED = 0
N_SUB = 2000
BATCH = 200
SCALES = [5.0, 12.5, 25.0, 50.0, 100.0]   # anchor = 25.0
BASE = dict(target_stage_idx=3, target_block_idx=0, n_directions=20, n_perturbations=50,
            subset_size=1024, chunk_size=1024, lambda_reg=1e-3, random_directions=True, seed=SEED)


def softmax_np(z):
    z = z - z.max(-1, keepdims=True); e = np.exp(z); return e / e.sum(-1, keepdims=True)


def ens_logits(ens, x, bs=BATCH):
    outs = [np.asarray(ens.predict(jnp.asarray(x[i:i+bs]))) for i in range(0, len(x), bs)]
    return np.concatenate(outs, axis=1)  # (M,N,C)


def base_logits(model, x, bs=BATCH):
    @nnx.jit
    def f(m, xb): return m(xb, use_running_average=True)
    return np.concatenate([np.asarray(f(model, jnp.asarray(x[i:i+bs]))) for i in range(0, len(x), bs)], 0)


def dis_scalars(ens_lg, base_lg):
    p = softmax_np(ens_lg); mp = p.mean(0); eps = 1e-12
    pred_ent = -(mp * np.log(mp + eps)).sum(-1)
    mi = pred_ent - (-(p * np.log(p + eps)).sum(-1).mean(0))
    pb = softmax_np(base_lg)
    logit_chg = np.linalg.norm(ens_lg - base_lg[None], axis=-1).mean(0)
    prob_chg = np.linalg.norm(p - pb[None], axis=-1).mean(0)
    return mp, pred_ent, mi, logit_chg, prob_chg


def auroc(id_s, ood_s):
    # OOD = positive (higher score). Mann-Whitney.
    all_s = np.concatenate([id_s, ood_s]); order = all_s.argsort()
    ranks = np.empty(len(all_s)); ranks[order] = np.arange(1, len(all_s) + 1)
    r_ood = ranks[len(id_s):].sum()
    n1, n2 = len(ood_s), len(id_s)
    return (r_ood - n1 * (n1 + 1) / 2) / (n1 * n2)


def fpr95(id_s, ood_s):
    t = np.quantile(ood_s, 0.05)          # 95% of OOD have score >= t
    return float((id_s >= t).mean())


def main():
    t0 = time.time()
    seed_everything(SEED)
    task = CIFAROpenOODPnC(dataset="cifar10", epochs=300, perturbation_sizes=[25.0],
                           posthoc_calibrate=True, bootstrap_frac=0.05, **BASE)
    benchmark, x_tr, _, x_va, y_va, n_cls = _load_cifar_openood_context(task)
    model = PreActResNet18(n_classes=n_cls, rngs=nnx.Rngs(SEED))
    with open(task.input().path, "rb") as f:
        ckpt = pickle.load(f)
    nnx.update(model, ckpt["state"])

    # balanced subsamples (fixed)
    rng = np.random.RandomState(1234)
    def sub(X):
        k = min(N_SUB, len(X)); return X[rng.choice(len(X), k, replace=False)]
    rng2 = np.random.RandomState(1234)
    sel_id = rng2.choice(len(benchmark["id_test"]["inputs"]), min(N_SUB, len(benchmark["id_test"]["inputs"])), replace=False)
    id_x = benchmark["id_test"]["inputs"][sel_id]
    id_y = np.asarray(benchmark["id_test"]["targets"])[sel_id].astype(int)
    near = {k: sub(v["inputs"]) for k, v in benchmark["near_ood"].items()}
    far = {k: sub(v["inputs"]) for k, v in benchmark["far_ood"].items()}

    base_id_lg = base_logits(model, id_x)

    rows = []
    for scale in SCALES:
        ens, _ = _build_single_block_pnc_ensemble(nnx.clone(model), x_tr, perturbation_scale=scale,
                                                  bootstrap_frac=0.05, bootstrap_seed=SEED, **BASE)
        raw_arr, corr_arr = ens.compute_shift_diagnostics(ens.chunks, ens.T_orig_chunks, label=f"scale{scale}")
        hidden_mag = float(raw_arr.mean())      # uncorrected block-output shift = hidden perturbation effect
        corr_mag = float(corr_arr.mean())
        # save corrected members_w2, prepare uncorrected override
        corrected_w2 = ens.members_w2
        C_out = ens.w2_orig.shape[3]
        uncorrected_w2 = [(ens.w2_orig, jnp.zeros(C_out)) for _ in range(len(ens.members_w2))]

        for variant, mw2 in [("corrected", corrected_w2), ("uncorrected", uncorrected_w2)]:
            ens.members_w2 = mw2
            # ID
            el = ens_logits(ens, id_x)
            mp, pe, mi, lchg, pchg = dis_scalars(el, base_id_lg)
            acc = float((mp.argmax(-1) == id_y).mean())
            nll = float(-np.log(mp[np.arange(len(id_y)), id_y] + 1e-12).mean())
            id_score = pe.copy()  # predictive_entropy OOD score on ID
            # OOD families
            fam_res = {}
            for fam_name, fam in [("near", near), ("far", far)]:
                aurocs, fprs, pes, mis = [], [], [], []
                for dk, X in fam.items():
                    elo = ens_logits(ens, X); blo = base_logits(model, X)
                    _, peo, mio, _, _ = dis_scalars(elo, blo)
                    aurocs.append(auroc(id_score, peo)); fprs.append(fpr95(id_score, peo))
                    pes.append(float(peo.mean())); mis.append(float(mio.mean()))
                fam_res[fam_name] = (float(np.mean(aurocs)) * 100, float(np.mean(fprs)) * 100,
                                     float(np.mean(pes)), float(np.mean(mis)))
            rows.append(dict(
                scale=scale, variant=variant,
                hidden_pert_mag=round(hidden_mag, 4), corrected_shift=round(corr_mag, 4),
                id_acc=round(acc * 100, 3), id_nll=round(nll, 4),
                id_pred_entropy=round(float(pe.mean()), 4), id_mutual_info=round(float(mi.mean()), 4),
                id_logit_change=round(float(lchg.mean()), 4), id_prob_change=round(float(pchg.mean()), 4),
                near_auroc=round(fam_res["near"][0], 3), near_fpr95=round(fam_res["near"][1], 3),
                far_auroc=round(fam_res["far"][0], 3), far_fpr95=round(fam_res["far"][1], 3),
                near_pred_entropy=round(fam_res["near"][2], 4), near_mutual_info=round(fam_res["near"][3], 4),
                far_pred_entropy=round(fam_res["far"][2], 4), far_mutual_info=round(fam_res["far"][3], 4),
            ))
            print(f"[abl] scale={scale:6} {variant:11} hidden={hidden_mag:.2f} idacc={acc*100:.2f} "
                  f"idnll={nll:.3f} idlogitchg={lchg.mean():.2f} nearAUROC={fam_res['near'][0]:.1f} "
                  f"farAUROC={fam_res['far'][0]:.1f}", flush=True)
        ens.members_w2 = corrected_w2

    fields = list(rows[0].keys())
    with open(OUT / "correction_ablation_cifar.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
    json.dump(dict(seed=SEED, scales=SCALES, n_sub=N_SUB, anchor_scale=25.0,
                   temperature_applied=False, runtime_sec=time.time() - t0,
                   note="OOD score=predictive_entropy; AUROC macro-mean over datasets; subsampled N_SUB/dataset"),
              open(OUT / "correction_ablation_cifar_meta.json", "w"), indent=2)
    print(f"[abl] wrote correction_ablation_cifar.csv ({len(rows)} rows) in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
