"""Spec §17-19 — give LLLA the same ID-only temperature calibration P&C gets.

The completed LLLA-Kron result is preserved unchanged: its prior precision stays at the
value chosen by ID-only NLL in the previous round (1e9), and is *not* refitted here. After
the prior is frozen, one scalar probability temperature is fitted on the separate
8,192-image temperature pool — the same pool and the same criterion P&C used — by
minimising ID NLL of

    p_T(c|x) = p(c|x)^(1/T) / sum_j p(j|x)^(1/T).

Both the raw and the temperature-calibrated variants are reported. Prior and temperature
are never tuned jointly, and no OOD data enters either choice.
"""
from __future__ import annotations

import json

import numpy as np
import torch
from scipy.optimize import minimize_scalar

from experiments.banking77_pnc.pnc_metrics import clf_metrics

from . import full_cache as fc
from . import fu_common as F
from . import llla_kron as lk
from .memprobe import reset_cuda

FROZEN_PRIOR = 1e9


def temper(p: np.ndarray, T: float) -> np.ndarray:
    lp = np.log(np.clip(p, 1e-300, None)) / T
    lp -= lp.max(-1, keepdims=True)
    e = np.exp(lp)
    return e / e.sum(-1, keepdims=True)


def fit_temperature(p: np.ndarray, y: np.ndarray) -> tuple[float, dict]:
    def nll(logT):
        q = temper(p, float(np.exp(logT)))
        return -np.log(np.clip(q[np.arange(len(y)), y], 1e-300, None)).mean()
    res = minimize_scalar(nll, bounds=(np.log(0.05), np.log(20.0)), method="bounded",
                          options={"xatol": 1e-5})
    T = float(np.exp(res.x))
    return T, {"nll_at_T": float(res.fun), "nll_at_1": float(nll(0.0)),
               "converged": bool(res.success)}


def run(structures=("kron",)):
    prev = json.loads((F.PREV / "metrics" / "llla_results.json").read_text())
    out = {"frozen_prior_precision": FROZEN_PRIOR,
           "temperature_pool": "cache_temperature.npz (8,192 ID train images)",
           "rule": "prior frozen from the previous round; T fitted afterwards on ID NLL",
           "variants": {}}

    for st in structures:
        name = f"LLLA-{'Kron' if st == 'kron' else 'Diag'}"
        print(f"\n=== {name} ===")
        ad, la, rows, best, info = lk.fit(st)
        head = lk.head_module(ad)
        if st == "kron":
            assert best["prior_precision"] == FROZEN_PRIOR, (
                f"refit selected {best['prior_precision']}, expected {FROZEN_PRIOR}")
        la.prior_precision = torch.tensor(float(best["prior_precision"]),
                                          device=ad.device)
        print(f"  prior precision held at {best['prior_precision']:g}")

        tmp = fc.load_cache(F.SRC / "raw" / "cache_temperature.npz")
        Xt = fc.cache_to_gpu(tmp, ad)
        phi_t = fc.features_from_cache(ad, Xt)
        phi_t = torch.from_numpy(phi_t.cpu().numpy()).to(ad.device, ad.dtype)
        p_t = lk.glm_probit_probs(la, head, phi_t)
        T, tinfo = fit_temperature(p_t, tmp["labels"])
        print(f"  fitted temperature T = {T:.4f}  (pool NLL {tinfo['nll_at_1']:.4f} "
              f"-> {tinfo['nll_at_T']:.4f})")
        del Xt, phi_t
        reset_cuda()

        val = fc.load_cache(F.cache_path("val50k"))
        Xv = fc.cache_to_gpu(val, ad)
        phi_v = fc.features_from_cache(ad, Xv)
        phi_v = torch.from_numpy(phi_v.cpu().numpy()).to(ad.device, ad.dtype)
        p_id = lk.glm_probit_probs(la, head, phi_v)
        del Xv
        reset_cuda()

        variants = {name: p_id, f"{name}+Temp": temper(p_id, T)}
        ood_p = {}
        for d in F.DS:
            c = fc.load_cache(F.cache_path(d))
            X = fc.cache_to_gpu(c, ad)
            phi = fc.features_from_cache(ad, X)
            phi = torch.from_numpy(phi.cpu().numpy()).to(ad.device, ad.dtype)
            ood_p[d] = lk.glm_probit_probs(la, head, phi)
            del X, phi
            reset_cuda()

        for vname, pv in variants.items():
            m = clf_metrics(pv, val["labels"])
            id_s = F.entropy(pv)
            oo = {d: F.entropy(ood_p[d] if vname.endswith("Temp") is False
                               else temper(ood_p[d], T)) for d in F.DS}
            om = F.ood_metrics(id_s, oo)
            rec = {"temperature": T if vname.endswith("Temp") else 1.0,
                   "prior_precision": best["prior_precision"],
                   "top1": m["accuracy"], "nll": m["nll"], "ece": m["ece"], **om}
            out["variants"][vname] = rec
            print(f"  {vname:<18} top-1 {m['accuracy']*100:.3f}%  NLL {m['nll']:.4f}  "
                  f"ECE {m['ece']:.4f}  Near {om['near_auroc']*100:.2f}  "
                  f"Far {om['far_auroc']*100:.2f}")
            if vname.endswith("+Temp"):
                np.savez_compressed(
                    F.OUT / "predictions" / "llla-kron-temp_scores.npz",
                    id_entropy=id_s.astype(np.float32),
                    **{f"ood_{d}": oo[d].astype(np.float32) for d in F.DS})
        out["variants"][name]["matches_previous_round"] = {
            "near_auroc_prev": prev["LLLA-Kron"]["aggregate"]["near"]["mean_auroc"],
            "far_auroc_prev": prev["LLLA-Kron"]["aggregate"]["far"]["mean_auroc"]}
        del la, p_id, ood_p
        reset_cuda()

    F.write_json(F.OUT / "metrics" / "llla_temperature.json", out)
    print(f"\nwrote {F.OUT/'metrics'/'llla_temperature.json'}")
    return out


if __name__ == "__main__":
    import sys
    run(("kron", "diag") if "--diag" in sys.argv else ("kron",))
