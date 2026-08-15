"""Spec §5 — align P&C's perturbation response with the covariance eigenmodes.

For a fixed evaluation subset we run the frozen members and take the *exact* final-CLS
change after the last LayerNorm,

    dphi_m(x) = phi_m(x) - phi_0(x),

project it onto the ID covariance eigenbasis, a_{m,j} = q_j^T dphi_m(x), and compare the
response energy E_j = E_{x,m}[a_{m,j}^2] against lambda_j, 1/lambda_j and the mode's
Mahalanobis OOD separation.

This is the exact tail forward pass, not the infinitesimal approximation — the cached CLS
residual makes it cheap enough that there is no reason to linearise.
"""
from __future__ import annotations

import time

import numpy as np
import torch
from scipy.stats import spearmanr

from . import full_cache as fc
from . import fu_common as F
from .memprobe import reset_cuda
from .vit_adapter import ViTPnCAdapter

N_ID = 8192
N_OOD = 5000
SUBSET_SEED = 20260815
BAND = 96


def subset_idx(n: int, k: int, seed: int) -> np.ndarray:
    if n <= k:
        return np.arange(n)
    return np.sort(np.random.default_rng(seed).choice(n, k, replace=False))


def response_energy(ad, members, Q: np.ndarray, sets: list) -> dict:
    """E_j and per-example projections, accumulated over examples and members."""
    D = Q.shape[0]
    Qg = torch.as_tensor(Q, dtype=torch.float64, device="cuda")
    acc = {}
    for name in sets:
        c = fc.load_cache(F.cache_path(name))
        n = len(c["x_resid_cls"])                    # OOD caches carry no labels
        idx = subset_idx(n, N_ID if name == "val50k" else N_OOD, SUBSET_SEED)
        X = torch.as_tensor(c["x_resid_cls"][idx], device=ad.device, dtype=ad.dtype)
        phi0 = F.member_phi(ad, X, None)
        e = torch.zeros(D, dtype=torch.float64)
        n_terms = 0
        for mem in members:
            dphi = (F.member_phi(ad, X, mem) - phi0).double().cuda()
            a = dphi @ Qg
            e += (a ** 2).sum(0).cpu()
            n_terms += a.shape[0]
            del dphi, a
        acc[name] = {"energy": (e / n_terms).numpy(), "n": len(idx),
                     "dphi_norm2": float((e.sum() / n_terms).item())}
        print(f"  {name:<12} n={len(idx):>5,}  mean ||dphi||^2 = "
              f"{acc[name]['dphi_norm2']:.4f}", flush=True)
        del X, phi0
        reset_cuda()
    return acc


def run():
    ad = ViTPnCAdapter(device="cuda", dtype=torch.float32)
    z = np.load(F.OUT / "raw" / "covariance_eigenbasis.npz")
    Q = z["Q"].astype(np.float64)
    lam = z["eigenvalues"]
    sep = z["ood_mean_contribution"] - z["id_mean_contribution"]
    mode_auroc = z["per_mode_auroc"]
    D = len(lam)

    members, cfg = F.load_primary_members(ad)
    print(f"P&C primary: r={cfg['r_target']} lam={cfg['lambda']} K={cfg['K']} "
          f"M={len(members)}")
    t0 = time.perf_counter()
    acc = response_energy(ad, members, Q, F.SETS)
    print(f"response pass took {time.perf_counter()-t0:.0f}s")

    E_id = acc["val50k"]["energy"]
    E_ood = np.mean([acc[d]["energy"] for d in F.DS], axis=0)
    E_all = 0.5 * (E_id + E_ood)
    rank = np.arange(1, D + 1)

    def sp(a, b):
        return float(spearmanr(a, b).statistic)

    results = {"n_id": acc["val50k"]["n"], "n_ood_each": N_OOD, "config": cfg,
               "band_width": BAND,
               "curves": {"eigenvalue": lam.tolist(),
                          "inv_eigenvalue_norm": (1.0 / lam / (1.0 / lam).sum()).tolist(),
                          "maha_separation": sep.tolist(),
                          "maha_mode_auroc": mode_auroc.tolist(),
                          "pnc_energy_id": E_id.tolist(),
                          "pnc_energy_ood": E_ood.tolist(),
                          "pnc_energy_norm": (E_all / E_all.sum()).tolist(),
                          "pnc_whitened_energy_norm":
                              ((E_all / lam) / (E_all / lam).sum()).tolist()},
               "spearman": {
                   "E_vs_lambda": sp(E_all, lam),
                   "E_vs_inv_lambda": sp(E_all, 1.0 / lam),
                   "E_vs_maha_separation": sp(E_all, sep),
                   "E_vs_maha_mode_auroc": sp(E_all, mode_auroc),
                   "E_vs_rank": sp(E_all, rank),
                   "lambda_vs_maha_mode_auroc": sp(lam, mode_auroc),
                   "maha_separation_vs_inv_lambda": sp(sep, 1.0 / lam)},
               "bands": []}

    print(f"\n=== Spearman ===")
    for k, v in results["spearman"].items():
        print(f"  {k:<34} {v:+.4f}")

    print(f"\n{'band':<12}{'lambda mean':>13}{'P&C energy share':>19}"
          f"{'whitened share':>17}{'Maha mode AUROC':>18}")
    for b in range(D // BAND):
        lo, hi = b * BAND, (b + 1) * BAND
        share = float(E_all[lo:hi].sum() / E_all.sum())
        wshare = float((E_all[lo:hi] / lam[lo:hi]).sum() / (E_all / lam).sum())
        row = {"band": f"{lo+1}-{hi}", "lambda_mean": float(lam[lo:hi].mean()),
               "pnc_energy_share": share, "pnc_whitened_share": wshare,
               "maha_mode_auroc_mean": float(mode_auroc[lo:hi].mean()),
               "maha_separation_share": float(sep[lo:hi].sum() / sep.sum()),
               "id_variance_share": float(lam[lo:hi].sum() / lam.sum())}
        results["bands"].append(row)
        print(f"  {row['band']:<10}{row['lambda_mean']:>13.5f}{share*100:>18.2f}%"
              f"{wshare*100:>16.2f}%{row['maha_mode_auroc_mean']*100:>17.2f}")

    F.write_json(F.OUT / "metrics" / "spectral_alignment.json", results)
    np.savez_compressed(F.OUT / "raw" / "pnc_response_energy.npz",
                        energy_id=E_id, energy_ood=E_ood,
                        **{f"energy_{d}": acc[d]["energy"] for d in F.DS})
    print(f"\nwrote {F.OUT/'metrics'/'spectral_alignment.json'}")
    return results


if __name__ == "__main__":
    run()
