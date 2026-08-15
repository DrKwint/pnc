"""Spec §4 — which covariance directions make Mahalanobis effective?

Fit the shared covariance Sigma = Q Lambda Q^T on the frozen 32,768-image ID pool. For each
evaluation example, fix the class c* chosen by the headline Mahalanobis detector and write

    d_M^2(x) = sum_j <phi(x) - mu_{c*}, q_j>^2 / lambda_j

Modes are sorted by decreasing ID variance and split into eight bands of 96. Each band is
scored alone, and cumulative high-variance / low-variance prefixes are scored too. No band
is selected using OOD; the whole curve is reported.

Also emits the per-mode statistics that §5 needs to align P&C's perturbation response
against the same eigenbasis.
"""
from __future__ import annotations

import time

import numpy as np
import torch

from . import full_cache as fc
from . import fu_common as F
from .memprobe import reset_cuda
from .vit_adapter import ViTPnCAdapter

BAND = 96
CUMUL = [96, 192, 384, 768]


def run():
    ad = ViTPnCAdapter(device="cuda", dtype=torch.float32)
    phi_c, y_true, _ = F.calibration_features(ad)
    means, prec = F.fit_gaussian(phi_c, y_true)
    D = phi_c.shape[1]

    # eigendecomposition of the *same* regularised covariance the detector inverts
    cov = np.linalg.inv(prec)
    lam, Q = np.linalg.eigh(cov)
    order = np.argsort(lam)[::-1]                    # decreasing ID variance
    lam, Q = lam[order], Q[:, order]
    print(f"covariance spectrum: lambda_1 {lam[0]:.4g}  lambda_768 {lam[-1]:.4g}  "
          f"condition {lam[0]/lam[-1]:.4g}")

    Qg = torch.as_tensor(Q, dtype=torch.float64, device="cuda")
    lg = torch.as_tensor(lam, dtype=torch.float64, device="cuda")

    t0 = time.perf_counter()
    contrib = {}          # per-set (n, 768) squared whitened coordinates
    for name in F.SETS:
        c = fc.load_cache(F.cache_path(name))
        X = fc.cache_to_gpu(c, ad)
        phi = fc.features_from_cache(ad, X).numpy()
        _, cstar = F.maha(phi, means, prec, return_class=True)
        out = []
        for s in range(0, len(phi), 8192):
            x = torch.as_tensor(phi[s:s + 8192], dtype=torch.float64, device="cuda")
            mu = torch.as_tensor(means[cstar[s:s + 8192]], dtype=torch.float64,
                                 device="cuda")
            a = (x - mu) @ Qg                        # projection onto eigenbasis
            out.append(((a ** 2) / lg).cpu().numpy())
            del x, mu, a
        contrib[name] = np.concatenate(out)
        print(f"  {name:<12} {contrib[name].shape}  ({time.perf_counter()-t0:.0f}s)",
              flush=True)
        del X, phi
        reset_cuda()

    results = {"n_cal": len(phi_c), "band_width": BAND,
               "eigenvalues": lam.tolist(), "bands": [], "cumulative": [],
               "per_mode": {}}

    # ---- per-band, scored alone ----
    print(f"\n{'band (ID-variance rank)':<26}{'lambda range':>26}"
          f"{'Near AUROC':>12}{'Far AUROC':>11}{'share of d^2':>14}")
    total_id = contrib["val50k"].sum(1).mean()
    for b in range(D // BAND):
        lo, hi = b * BAND, (b + 1) * BAND
        s = {n: contrib[n][:, lo:hi].sum(1) for n in F.SETS}
        m = F.ood_metrics(s["val50k"], {d: s[d] for d in F.DS})
        share = float(contrib["val50k"][:, lo:hi].sum(1).mean() / total_id)
        row = {"band": f"{lo+1}-{hi}", "lambda_hi": float(lam[lo]),
               "lambda_lo": float(lam[hi - 1]), "share_of_id_d2": share,
               "near_auroc": m["near_auroc"], "far_auroc": m["far_auroc"],
               "near_fpr95": m["near_fpr95"], "far_fpr95": m["far_fpr95"]}
        results["bands"].append(row)
        print(f"  {row['band']:<24}{lam[lo]:>12.4g}{lam[hi-1]:>13.4g}"
              f"{m['near_auroc']*100:>12.2f}{m['far_auroc']*100:>11.2f}{share*100:>13.2f}%")

    # ---- cumulative prefixes ----
    print(f"\n{'cumulative subset':<30}{'Near AUROC':>12}{'Far AUROC':>11}")
    for k in CUMUL:
        for tag, sl in (("highest-variance", slice(0, k)),
                        ("lowest-variance", slice(D - k, D))):
            s = {n: contrib[n][:, sl].sum(1) for n in F.SETS}
            m = F.ood_metrics(s["val50k"], {d: s[d] for d in F.DS})
            row = {"subset": f"{tag} {k}", "k": k, "which": tag,
                   "near_auroc": m["near_auroc"], "far_auroc": m["far_auroc"],
                   "near_fpr95": m["near_fpr95"], "far_fpr95": m["far_fpr95"]}
            results["cumulative"].append(row)
            print(f"  {row['subset']:<28}{m['near_auroc']*100:>12.2f}"
                  f"{m['far_auroc']*100:>11.2f}")
            if k == D and tag == "lowest-variance":
                break

    # ---- per-mode statistics (inputs to §5) ----
    ood_all = np.concatenate([contrib[d] for d in F.DS])
    id_mean = contrib["val50k"].mean(0)
    ood_mean = ood_all.mean(0)
    per_mode_auroc = np.zeros(D)
    for j in range(D):
        m = F.ood_metrics(contrib["val50k"][:, j], {d: contrib[d][:, j] for d in F.DS})
        per_mode_auroc[j] = 0.5 * (m["near_auroc"] + m["far_auroc"])
    results["per_mode"] = {
        "id_mean_contribution": id_mean.tolist(),
        "ood_mean_contribution": ood_mean.tolist(),
        "separation": (ood_mean - id_mean).tolist(),
        "auroc": per_mode_auroc.tolist()}

    np.savez_compressed(F.OUT / "raw" / "covariance_eigenbasis.npz",
                        Q=Q.astype(np.float32), eigenvalues=lam.astype(np.float64),
                        id_mean_contribution=id_mean, ood_mean_contribution=ood_mean,
                        per_mode_auroc=per_mode_auroc)
    F.write_json(F.OUT / "metrics" / "mahalanobis_spectrum.json", results)
    print(f"\nwrote {F.OUT/'metrics'/'mahalanobis_spectrum.json'}")
    return results


if __name__ == "__main__":
    run()
