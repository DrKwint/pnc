"""Part A — why does class-conditional Mahalanobis beat P&C on ImageNet OOD?

All quantities are computed on the *same* examples so geometry scores, P&C uncertainty
scores and P&C mechanism quantities can be correlated directly:

  geometry      class-conditional Mahalanobis (M1), predicted-label variant (M2),
                unconditional (M3), nearest-centroid Euclidean (M4), whitened global
                distance, feature norm
  P&C scores    predictive entropy, expected member entropy, mutual information,
                mean logit variance, mean probability variance
                (mean pairwise KL is omitted: over 190 member pairs x 1000 classes x
                136k images it is the dominant cost, and mutual information is the same
                disagreement quantity in the form the protocol already scores)
  P&C mechanism hidden perturbation change ||y_v − y||, output-visible change
                ||z_v − z0||, post-correction residual ||r_S|| (exact, not the
                infinitesimal approximation), and ridge leverage x̂ᵀ(XᵀX+λI)⁻¹x̂

Nothing here replaces the headline P&C predictive-entropy result (spec A3).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

from . import full_cache as fc
from . import full_ood as fo
from . import full_oodmetrics as fm
from . import pnc_core as pc
from .memprobe import reset_cuda
from .vit_adapter import ViTPnCAdapter

SRC = Path("results/neurips_2026_rebuttal/imagenet_vit")
FRONTIER = Path("results/neurips_2026_rebuttal/imagenet_vit_preservation_frontier")
OUT = Path("results/neurips_2026_rebuttal/imagenet_vit_geometry_scod_llla")
GROUPS = {k: v[2] for k, v in fo.DATASETS.items()}
DS = ["ssb_hard", "ninco", "inaturalist", "textures", "openimage_o"]
LAYERS = ["cls_block8", "cls_block9", "cls_block10", "cls_block11", "cls_final_ln"]


# ------------------------------------------------------------ Mahalanobis family
def fit_gaussian(feats: np.ndarray, targets: np.ndarray | None, n_classes: int = 1000):
    D = feats.shape[1]
    if targets is None:                                   # M3: one global Gaussian
        mu = feats.astype(np.float64).mean(0, keepdims=True)
        cen = feats.astype(np.float64) - mu
        cov = cen.T @ cen / len(feats) + 1e-6 * np.eye(D)
        return mu, np.linalg.inv(cov)
    means = np.zeros((n_classes, D), dtype=np.float64)
    for c in range(n_classes):
        m = targets == c
        if m.sum():
            means[c] = feats[m].astype(np.float64).mean(0)
    cen = feats.astype(np.float64) - means[targets]
    cov = cen.T @ cen / len(feats) + 1e-6 * np.eye(D)
    return means, np.linalg.inv(cov)


def maha(feats: np.ndarray, means: np.ndarray, prec: np.ndarray | None,
         chunk: int = 8192) -> np.ndarray:
    """min_c (x−μ_c)ᵀP(x−μ_c); with prec=None this is squared Euclidean (M4)."""
    dev = "cuda"
    Mu = torch.as_tensor(means, dtype=torch.float64, device=dev)
    P = (torch.as_tensor(prec, dtype=torch.float64, device=dev) if prec is not None
         else torch.eye(Mu.shape[1], dtype=torch.float64, device=dev))
    PMu = P @ Mu.T
    muPmu = (Mu * PMu.T).sum(1)
    out = []
    for s in range(0, len(feats), chunk):
        x = torch.as_tensor(feats[s:s + chunk], dtype=torch.float64, device=dev)
        q = (x * (x @ P)).sum(1)[:, None] - 2.0 * (x @ PMu) + muPmu[None, :]
        out.append(q.min(1).values.cpu().numpy())
        del x, q
    del Mu, P, PMu, muPmu
    reset_cuda()
    return np.concatenate(out)


# ------------------------------------------------- P&C scores + mechanism terms
@torch.inference_mode()
def pnc_quantities(ad, x_cls: torch.Tensor, members: list, T: float, Ginv: torch.Tensor,
                   chunk: int = 4096) -> dict:
    """Member-level P&C scores and mechanism terms on one cached image set."""
    n = x_cls.shape[0]
    acc_p = torch.zeros(n, 1000, dtype=torch.float64)
    acc_ent = torch.zeros(n, dtype=torch.float64)
    acc_logit_m = torch.zeros(n, 1000, dtype=torch.float64)
    acc_logit_m2 = torch.zeros(n, 1000, dtype=torch.float64)
    acc_p2 = torch.zeros(n, 1000, dtype=torch.float64)
    hid = torch.zeros(n, dtype=torch.float64)
    vis = torch.zeros(n, dtype=torch.float64)
    res = torch.zeros(n, dtype=torch.float64)
    lev = torch.zeros(n, dtype=torch.float64)
    M = len(members)
    for mi, mem in enumerate(members):
        for s in range(0, n, chunk):
            x = x_cls[s:s + chunk]
            h = ad.block.ln_2(x[:, None, :])[:, 0]
            y0 = torch.nn.functional.gelu(h @ ad.W1 + ad.b1)
            z0 = y0 @ ad.W2 + ad.b2
            yv = torch.nn.functional.gelu(h @ mem["W1v"] + ad.b1)
            zu = yv @ ad.W2 + ad.b2
            zc = yv @ mem["W2"] + mem["b2"]
            lg = ad.head(ad.enc.ln((x + zc)[:, None, :])[:, 0])
            p = torch.softmax(lg.double() / T, -1).cpu()
            acc_p[s:s + chunk] += p
            acc_p2[s:s + chunk] += p ** 2
            acc_ent[s:s + chunk] += -(p * torch.log(p + 1e-12)).sum(-1)
            l64 = lg.double().cpu()
            acc_logit_m[s:s + chunk] += l64
            acc_logit_m2[s:s + chunk] += l64 ** 2
            hid[s:s + chunk] += (yv - y0).double().norm(dim=1).cpu() / M
            vis[s:s + chunk] += (zu - z0).double().norm(dim=1).cpu() / M
            res[s:s + chunk] += (zc - z0).double().norm(dim=1).cpu() / M
            # Leverage is a property of the correction design, so it is evaluated for
            # member 0 only: the n x 3073 x 3073 product is the dominant cost here and
            # averaging it over all 20 members would multiply the whole analysis by 20
            # for a quantity that varies little across members.
            if mi == 0:
                xa = pc.SufficientStats.augment(yv).double()
                lev[s:s + chunk] = ((xa @ Ginv) * xa).sum(1).cpu()
                del xa
            del x, h, y0, z0, yv, zu, zc, lg, p, l64
    pbar = (acc_p / M).numpy()
    ent_bar = -np.sum(pbar * np.log(pbar + 1e-12), -1)
    exp_ent = (acc_ent / M).numpy()
    logit_var = ((acc_logit_m2 / M) - (acc_logit_m / M) ** 2).clamp(min=0).mean(-1).numpy()
    prob_var = ((acc_p2 / M) - (acc_p / M) ** 2).clamp(min=0).mean(-1).numpy()
    return {"predictive_entropy": ent_bar, "expected_member_entropy": exp_ent,
            "mutual_information": ent_bar - exp_ent, "logit_variance": logit_var,
            "prob_variance": prob_var,
            "hidden_perturbation_change": hid.numpy(),
            "output_visible_change": vis.numpy(),
            "post_correction_residual": res.numpy(),
            "ridge_leverage": lev.numpy()}


def load_primary_members(ad, seed: int = 0):
    cfg = json.loads((FRONTIER / "selection" / "frozen_configs.json").read_text())
    c = cfg["configs"]["primary"]
    z = np.load(FRONTIER / "raw" / f"members_{'primary'}_seed{seed}.npz")
    U = pc.perturbation_basis(seed, c["K"])
    out = []
    for m in range(len(z["coefficients"])):
        dW1 = torch.as_tensor(pc.member_dW1(U, z["coefficients"][m], float(z["scale"])),
                              device=ad.device, dtype=ad.dtype)
        out.append({"W1v": ad.W1 + dW1,
                    "W2": torch.as_tensor(z["W2"][m], device=ad.device, dtype=ad.dtype),
                    "b2": torch.as_tensor(z["b2"][m], device=ad.device, dtype=ad.dtype)})
        del dW1
    return out, c


def correction_ginv(ad, members, lam: float, n_cal: int = 32768) -> torch.Tensor:
    """(XᵀX + λI)⁻¹ for member 0's correction design — the leverage denominator."""
    corr = fc.load_cache(SRC / "raw" / "cache_correction.npz")
    h = torch.as_tensor(corr["h"][:n_cal], device=ad.device, dtype=ad.dtype)
    with torch.inference_mode():
        y = torch.nn.functional.gelu(h @ members[0]["W1v"] + ad.b1)
        X = pc.SufficientStats.augment(y).double()
        G = X.T @ X + lam * torch.eye(X.shape[1], device=ad.device, dtype=torch.float64)
    Ginv = torch.linalg.inv(G)
    del h, y, X, G
    reset_cuda()
    return Ginv


def _ood_metrics(id_s, ood_s):
    per = {d: fm.binary_ood_metrics(id_s, ood_s[d]) for d in DS}
    agg = {g: fm.aggregate_family_metrics(
        id_s, {k: v for k, v in ood_s.items() if GROUPS[k] == g})
        for g in ("near", "far")}
    return {"per_dataset": {d: {"auroc": per[d]["auroc"], "fpr95": per[d]["fpr95"]}
                            for d in DS},
            "near_auroc": agg["near"]["mean_auroc"], "near_fpr95": agg["near"]["mean_fpr95"],
            "far_auroc": agg["far"]["mean_auroc"], "far_fpr95": agg["far"]["mean_fpr95"]}


def run():
    ad = ViTPnCAdapter(device="cuda", dtype=torch.float32)
    T = json.loads((SRC / "metrics" / "temperature.json").read_text())["temperature"]
    members, cfg = load_primary_members(ad)
    print(f"P&C primary: r={cfg['r_target']} lam={cfg['lambda']} M={len(members)}")
    Ginv = correction_ginv(ad, members, cfg["lambda"])

    # ---- calibration-side Gaussians (A4: M1-M4), fitted on the 32,768 ID pool ----
    corr = fc.load_cache(SRC / "raw" / "cache_correction.npz")
    Xc = fc.cache_to_gpu(corr, ad)
    phi_c = fc.features_from_cache(ad, Xc).numpy()
    y_true = corr["labels"]
    with torch.inference_mode():
        y_pred = ad.head(torch.as_tensor(phi_c, device=ad.device,
                                         dtype=ad.dtype)).argmax(-1).cpu().numpy()
    del Xc
    reset_cuda()
    print(f"calibration features {phi_c.shape}; base train-pool accuracy "
          f"{(y_pred == y_true).mean()*100:.2f}%")
    fits = {
        "M1_true_label": fit_gaussian(phi_c, y_true),
        "M2_pred_label": fit_gaussian(phi_c, y_pred),
        "M3_unconditional": fit_gaussian(phi_c, None),
        "M4_nearest_centroid": (fit_gaussian(phi_c, y_true)[0], None),
    }
    scores = {}
    t0 = time.perf_counter()
    for name in ["val50k"] + DS:
        cache = "cache_val50k.npz" if name == "val50k" else f"cache_ood_{name}.npz"
        c = fc.load_cache(SRC / "raw" / cache)
        X = fc.cache_to_gpu(c, ad)
        phi = fc.features_from_cache(ad, X).numpy()
        s = {}
        for k, (mu, prec) in fits.items():
            s[k] = maha(phi, mu, prec)
        s["whitened_global"] = maha(phi, fits["M3_unconditional"][0],
                                    fits["M3_unconditional"][1])
        s["feature_norm"] = -np.linalg.norm(phi, axis=1)
        q = pnc_quantities(ad, X, members, T, Ginv)
        s.update(q)
        scores[name] = s
        print(f"  {name:<12} scored ({len(phi):,})", flush=True)
        del X, phi
        reset_cuda()
    print(f"scoring took {time.perf_counter()-t0:.0f}s")

    keys = list(scores["val50k"])
    id_s = scores["val50k"]
    ood_s = {d: scores[d] for d in DS}
    results = {"config": cfg, "temperature": T, "scores": {}}
    print(f"\n{'score':<30}{'Near AUROC':>12}{'Far AUROC':>11}{'Near FPR95':>12}{'Far FPR95':>11}")
    for k in keys:
        m = _ood_metrics(id_s[k], {d: ood_s[d][k] for d in DS})
        results["scores"][k] = m
        print(f"  {k:<28}{m['near_auroc']*100:>11.2f}{m['far_auroc']*100:>11.2f}"
              f"{m['near_fpr95']*100:>12.2f}{m['far_fpr95']*100:>11.2f}")

    # ---- A3/A5: Spearman correlations against M1 ----
    corr_out = {}
    pooled_ood = {k: np.concatenate([ood_s[d][k] for d in DS]) for k in keys}
    groups = {"ID": id_s, "pooled_OOD": pooled_ood,
              "near": {k: np.concatenate([ood_s[d][k] for d in DS if GROUPS[d] == "near"])
                       for k in keys},
              "far": {k: np.concatenate([ood_s[d][k] for d in DS if GROUPS[d] == "far"])
                      for k in keys},
              "all": {k: np.concatenate([id_s[k]] + [ood_s[d][k] for d in DS])
                      for k in keys}}
    for gname, g in groups.items():
        corr_out[gname] = {k: float(spearmanr(g["M1_true_label"], g[k]).statistic)
                           for k in keys if k != "M1_true_label"}
    for d in DS:
        corr_out[d] = {k: float(spearmanr(ood_s[d]["M1_true_label"], ood_s[d][k]).statistic)
                       for k in keys if k != "M1_true_label"}
    results["spearman_vs_M1"] = corr_out

    # ---- A5: adjacent-stage correlations along the pipeline ----
    chain = ["M1_true_label", "hidden_perturbation_change", "output_visible_change",
             "post_correction_residual", "logit_variance", "predictive_entropy"]
    g = groups["all"]
    results["pipeline_chain"] = {
        f"{chain[i]}->{chain[i+1]}": float(spearmanr(g[chain[i]], g[chain[i+1]]).statistic)
        for i in range(len(chain) - 1)}

    fc.write_json(OUT / "metrics" / "geometry_analysis.json", results)
    np.savez_compressed(OUT / "predictions" / "geometry_scores_val50k.npz",
                        **{k: v.astype(np.float32) for k, v in id_s.items()})
    for d in DS:
        np.savez_compressed(OUT / "predictions" / f"geometry_scores_{d}.npz",
                            **{k: v.astype(np.float32) for k, v in ood_s[d].items()})
    print("\n=== A5 pipeline chain (Spearman, ID+OOD pooled) ===")
    for k, v in results["pipeline_chain"].items():
        print(f"  {k:<62} {v:+.4f}")
    print(f"\nwrote {OUT/'metrics'/'geometry_analysis.json'}")
    return results
