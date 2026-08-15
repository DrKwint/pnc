"""Spec §3 — random-projection Mahalanobis dimensionality control.

A K=20 P&C perturbation subspace and a 768-dimensional Mahalanobis model differ in more
than dimensionality. This asks how much Mahalanobis performance survives when the CLS
geometry itself is viewed through only K random directions, which separates

  H1  20 dimensions cannot represent the relevant geometry
  H2  20 dimensions suffice, but P&C's weight perturbations do not map onto the relevant
      feature-space directions

All means/covariances are fitted on the same frozen 32,768-image ID calibration pool. No
OOD data enters any choice.
"""
from __future__ import annotations

import time

import numpy as np
import torch

from . import full_cache as fc
from . import fu_common as F
from .memprobe import reset_cuda
from .vit_adapter import ViTPnCAdapter

KS = [5, 20, 40, 80, 160, 320]
N_PROJ = 20
SEED0 = 20260815


def orthonormal(D: int, K: int, seed: int) -> np.ndarray:
    g = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(g.standard_normal((D, K)))
    return Q[:, :K]


def run():
    ad = ViTPnCAdapter(device="cuda", dtype=torch.float32)
    phi_c, y_true, _ = F.calibration_features(ad)
    phi = {}
    for name in F.SETS:
        c = fc.load_cache(F.cache_path(name))
        X = fc.cache_to_gpu(c, ad)
        phi[name] = fc.features_from_cache(ad, X).numpy()
        del X
        reset_cuda()
    print(f"calibration {phi_c.shape}; eval sets "
          f"{ {k: v.shape[0] for k, v in phi.items()} }")

    results = {"n_projections": N_PROJ, "seed0": SEED0, "n_cal": len(phi_c),
               "variants": {}}
    t0 = time.perf_counter()
    for variant, targets in (("class_conditional", y_true), ("unconditional", None)):
        rows = []
        for K in KS:
            near, far, nf, ff = [], [], [], []
            for j in range(N_PROJ):
                R = orthonormal(phi_c.shape[1], K, SEED0 + 1000 * K + j)
                mu, prec = F.fit_gaussian(phi_c @ R, targets)
                s = {n: F.maha(phi[n] @ R, mu, prec) for n in F.SETS}
                m = F.ood_metrics(s["val50k"], {d: s[d] for d in F.DS})
                near.append(m["near_auroc"]); far.append(m["far_auroc"])
                nf.append(m["near_fpr95"]); ff.append(m["far_fpr95"])
            row = {"K": K,
                   "near_auroc_mean": float(np.mean(near)),
                   "near_auroc_std": float(np.std(near)),
                   "far_auroc_mean": float(np.mean(far)),
                   "far_auroc_std": float(np.std(far)),
                   "near_fpr95_mean": float(np.mean(nf)),
                   "far_fpr95_mean": float(np.mean(ff)),
                   "near_auroc_min": float(np.min(near)),
                   "near_auroc_max": float(np.max(near))}
            rows.append(row)
            print(f"  {variant:<18} K={K:<4} Near {row['near_auroc_mean']*100:6.2f} "
                  f"± {row['near_auroc_std']*100:.2f}   "
                  f"Far {row['far_auroc_mean']*100:6.2f} ± "
                  f"{row['far_auroc_std']*100:.2f}   ({time.perf_counter()-t0:.0f}s)",
                  flush=True)
        # full 768-d reference in the same code path
        mu, prec = F.fit_gaussian(phi_c, targets)
        s = {n: F.maha(phi[n], mu, prec) for n in F.SETS}
        m = F.ood_metrics(s["val50k"], {d: s[d] for d in F.DS})
        rows.append({"K": 768, "near_auroc_mean": m["near_auroc"], "near_auroc_std": 0.0,
                     "far_auroc_mean": m["far_auroc"], "far_auroc_std": 0.0,
                     "near_fpr95_mean": m["near_fpr95"], "far_fpr95_mean": m["far_fpr95"],
                     "near_auroc_min": m["near_auroc"], "near_auroc_max": m["near_auroc"]})
        print(f"  {variant:<18} K=768  Near {m['near_auroc']*100:6.2f}          "
              f"Far {m['far_auroc']*100:6.2f}   (full, identity projection)")
        results["variants"][variant] = rows

    F.write_json(F.OUT / "metrics" / "random_projection_mahalanobis.json", results)
    print(f"\nwrote {F.OUT/'metrics'/'random_projection_mahalanobis.json'}")
    return results


if __name__ == "__main__":
    run()
