"""Spec §21 — layerwise Mahalanobis at blocks 8-11 and the final LayerNorm.

An independent class-conditional shared-covariance detector is fitted at each depth on the
same frozen 32,768-image ID *training* pool (never on the evaluation split), with the same
regularisation convention everywhere. No layer is selected using OOD; all are reported.

The question is whether the geometry Mahalanobis exploits is already present well before the
final-block FFN that P&C perturbs.
"""
from __future__ import annotations

import time

import numpy as np

from . import fu_common as F

LAYERS = ["block8", "block9", "block10", "block11", "final_ln"]
EVAL_KEY = {"block8": "cls_block8", "block9": "cls_block9", "block10": "cls_block10",
            "block11": "cls_block11", "final_ln": "cls_final_ln"}


def run():
    train = np.load(F.OUT / "raw" / "cls_layers_correction.npz")
    y_true = train["labels"]
    y_pred = train["predicted_labels"]
    print(f"training pool {len(y_true):,}; base top-1 "
          f"{100*(y_pred == y_true).mean():.3f}%")

    ev = {}
    for n in F.SETS:
        p = F.PREV / "raw" / f"cls_layers_{n}.npz"
        if not p.exists():
            raise SystemExit(f"missing evaluation layer cache {p}")
        ev[n] = np.load(p)

    results = {"n_cal": int(len(y_true)), "layers": {},
               "regularisation": "cov + 1e-6 I, identical at every layer",
               "fit_pool": "32,768 ID training images (correction_rows.npy)",
               "OOD data used for layer selection": "NO"}
    t0 = time.perf_counter()
    print(f"\n{'layer':<12}{'Near AUROC':>12}{'Far AUROC':>11}{'Near FPR95':>12}"
          f"{'Far FPR95':>11}{'uncond Near':>13}")
    for L in LAYERS:
        phi_c = train[L].astype(np.float32)
        mu, prec = F.fit_gaussian(phi_c, y_true)
        s = {n: F.maha(ev[n][EVAL_KEY[L]].astype(np.float32), mu, prec) for n in F.SETS}
        m = F.ood_metrics(s["val50k"], {d: s[d] for d in F.DS})
        mu0, prec0 = F.fit_gaussian(phi_c, None)
        s0 = {n: F.maha(ev[n][EVAL_KEY[L]].astype(np.float32), mu0, prec0)
              for n in F.SETS}
        m0 = F.ood_metrics(s0["val50k"], {d: s0[d] for d in F.DS})
        results["layers"][L] = {"class_conditional": m, "unconditional": m0,
                                "dim": int(phi_c.shape[1])}
        print(f"  {L:<10}{m['near_auroc']*100:>12.2f}{m['far_auroc']*100:>11.2f}"
              f"{m['near_fpr95']*100:>12.2f}{m['far_fpr95']*100:>11.2f}"
              f"{m0['near_auroc']*100:>13.2f}", flush=True)
        np.savez_compressed(F.OUT / "predictions" / f"layerwise_maha_{L}.npz",
                            **{n: s[n].astype(np.float32) for n in F.SETS})
        del phi_c, mu, prec, s, s0

    best = max(LAYERS, key=lambda L: results["layers"][L]["class_conditional"]["near_auroc"])
    results["strongest_near"] = best
    results["monotone_in_depth"] = bool(all(
        results["layers"][LAYERS[i]]["class_conditional"]["near_auroc"]
        <= results["layers"][LAYERS[i + 1]]["class_conditional"]["near_auroc"] + 1e-12
        for i in range(len(LAYERS) - 1)))
    results["seconds"] = time.perf_counter() - t0
    F.write_json(F.OUT / "metrics" / "layerwise_mahalanobis.json", results)
    print(f"\n  strongest Near: {best}; monotone in depth: "
          f"{results['monotone_in_depth']}")
    print(f"wrote {F.OUT/'metrics'/'layerwise_mahalanobis.json'}")
    return results


if __name__ == "__main__":
    run()
