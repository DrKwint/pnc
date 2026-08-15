"""Spec §2 — P&C rank sweep K in {5,20,40,80} under a matched preservation budget.

This is a diagnostic, not a re-selection of the headline configuration: it asks whether a
richer perturbation subspace makes P&C's OOD behaviour more Mahalanobis-like.

Two controls make the comparison across K meaningful:

* **Nested orthonormal bases.** One orthonormal K=80 basis per seed; smaller ranks are its
  first K rows. Orthonormalisation is Cholesky/Gram-Schmidt, which is lower-triangular, so
  row k depends only on draws 1..k and the subspaces are genuinely nested. Member
  coefficients are the first K columns of one (M, 80) draw, so they nest too.
* **Matched realised perturbation.** Scale is set so the *median* realised
  ||dW1||_F/||W1||_F equals the target r at every K, so larger K does not silently receive
  more perturbation energy just because it has more coefficients.

Scale and ridge are selected per K on the ID selection pool alone, against the same
primary preservation budget (dAcc_ID >= -0.50 pp by paired-bootstrap LCB) used by the
frozen frontier experiment. No OOD data enters any choice.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch
from scipy.linalg import solve_triangular
from scipy.stats import spearmanr

from experiments.banking77_pnc.pnc_metrics import clf_metrics

from . import full_cache as fc
from . import fu_common as F
from . import pnc_core as pc
from .frontier import FrontierSearcher, best_lambda, boundary, pick_scale
from .memprobe import reset_cuda
from .vit_adapter import ViTPnCAdapter

KS = [5, 20, 40, 80]
K_MAX = 80
R_GRID = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0]
R_EXTEND = [5.0, 6.0]
LAM_GRID = [100.0, 300.0, 1000.0, 3000.0, 10000.0]
BUDGET = "primary"                       # dAcc_ID >= -0.50 pp, same as the frozen frontier
SEEDS_COARSE = (0,)
SEEDS_CONFIRM = (0, 10, 42)
SEEDS_FINAL = (0, 10, 42, 123, 2026)
M_SEARCH = 10
M_FINAL = 20

_CACHE: dict[int, np.ndarray] = {}


def nested_basis(seed: int, kmax: int = K_MAX, chunk: int = 262144) -> np.ndarray:
    """Orthonormal (kmax, D_FLAT) basis whose first K rows are themselves orthonormal."""
    if seed in _CACHE:
        return _CACHE[seed]
    D = pc.D_FLAT
    G = np.random.RandomState(seed).normal(size=(kmax, D)).astype(np.float32)
    G /= np.linalg.norm(G, axis=1, keepdims=True) + 1e-12
    L = np.linalg.cholesky((G.astype(np.float64) @ G.astype(np.float64).T))
    U = np.empty_like(G)
    for s in range(0, D, chunk):                     # chunked to bound peak RSS
        U[:, s:s + chunk] = solve_triangular(
            L, G[:, s:s + chunk].astype(np.float64), lower=True).astype(np.float32)
    del G
    if len(_CACHE) >= 2:
        _CACHE.pop(next(iter(_CACHE)))
    _CACHE[seed] = U
    return U


def nested_coefficients(seed: int, M: int, K: int) -> np.ndarray:
    return np.random.RandomState(seed + 1).normal(size=(M, K_MAX)).astype(np.float32)[:, :K]


class KSearcher(FrontierSearcher):
    """FrontierSearcher with a nested orthonormal basis at an arbitrary rank."""

    def __init__(self, K: int, **kw):
        self.K = K
        super().__init__(**kw)
        self._basis = {s: (nested_basis(s)[:K], nested_coefficients(s, self.M, K))
                       for s in self.seeds}


def _search_one_k(K: int, out: dict) -> dict:
    print(f"\n=== K={K}: ID-only scale/ridge search "
          f"(M={M_SEARCH}, seeds={SEEDS_COARSE}) ===", flush=True)
    s = KSearcher(K, m=M_SEARCH, seeds=SEEDS_COARSE)
    rows, grid = [], list(R_GRID)
    i = 0
    while i < len(grid):
        r = grid[i]
        rows.extend(s.evaluate_scale(r, LAM_GRID))
        i += 1
        # extend upward only once the planned grid is exhausted and the top still passes
        if i == len(grid) and grid is not None:
            top = max(R_GRID + [g for g in grid])
            if best_lambda([x for x in rows if x["r_target"] == top], BUDGET) is not None:
                nxt = [x for x in R_EXTEND if x > top]
                if nxt:
                    grid.append(nxt[0])
                    print(f"    top of grid r={top} still passes -> extending to "
                          f"{nxt[0]}", flush=True)
    b = boundary(rows, BUDGET)
    print(f"  K={K} coarse boundary: largest passing r={b['largest_passing_r']}, "
          f"smallest failing above={b['smallest_failing_r_above']}")

    confirm = []
    if b["largest_passing_r"] is not None:
        print(f"  K={K}: boundary confirmation at seeds {SEEDS_CONFIRM}", flush=True)
        s3 = KSearcher(K, m=M_SEARCH, seeds=SEEDS_CONFIRM)
        for r in [b["largest_passing_r"]] + (
                [b["smallest_failing_r_above"]] if b["smallest_failing_r_above"] else []):
            confirm.extend(s3.evaluate_scale(r, LAM_GRID))
        del s3
        reset_cuda()
    del s
    reset_cuda()

    sel = pick_scale(confirm, BUDGET) if confirm else None
    src = "confirm"
    if sel is None:                       # confirmation demoted the coarse boundary
        sel = pick_scale(rows, BUDGET)
        src = "coarse"
    out[str(K)] = {"K": K, "coarse_rows": rows, "confirm_rows": confirm,
                   "coarse_boundary": b, "selected": sel, "selected_from": src,
                   "budget": BUDGET, "grid_r": grid, "grid_lambda": LAM_GRID,
                   "OOD data accessed before selection": "NO"}
    if sel:
        print(f"  K={K} SELECTED r={sel['r_target']} lambda={sel['lambda']:g}  "
              f"realized r={sel['realized_r_median']:.4f}  "
              f"dAcc={sel['delta_top1_pp']:+.3f} pp (LCB {sel['lcb_pp']:+.3f})  "
              f"NLL {sel['nll']:.4f}")
    else:
        print(f"  K={K} NO configuration passes the {BUDGET} budget")
    return out


def stage_search():
    out = {}
    for K in KS:
        _search_one_k(K, out)
        F.write_json(F.OUT / "id_selection" / "ksweep_selection.json", out)
    print(f"\nwrote {F.OUT/'id_selection'/'ksweep_selection.json'}")


# ------------------------------------------------------------------ final stage
def _build_members(ad, h, z0, Theta0, seed, K, r, lam, M=M_FINAL):
    U = nested_basis(seed)[:K]
    co = nested_coefficients(seed, M, K)
    W1n = float(ad.W1.norm())
    scale = pc.base_scale(U, co, W1n, target_rel=r)
    members, rs = [], []
    for m in range(M):
        dW1 = torch.as_tensor(pc.member_dW1(U, co[m], scale), device=ad.device,
                              dtype=ad.dtype)
        rs.append(float(torch.linalg.norm(dW1.double()) / W1n))
        W1v = ad.W1 + dW1
        y = torch.nn.functional.gelu(h @ W1v + ad.b1)
        X = pc.SufficientStats.augment(y)
        G = (X.T @ X).double().cpu().numpy()
        C = (X.T @ z0).double().cpu().numpy()
        Theta, _, _ = pc.cho_solve_shared(G, C, lam, w_prior=Theta0)
        members.append({"W1v": W1v,
                        "W2": torch.as_tensor(Theta[1:], device=ad.device, dtype=ad.dtype),
                        "b2": torch.as_tensor(Theta[0], device=ad.device, dtype=ad.dtype)})
        del y, X, dW1, G, C
    return members, float(np.median(rs)), scale


@torch.inference_mode()
def _scores(ad, X, members, T, chunk=4096):
    n, M = X.shape[0], len(members)
    sp = torch.zeros(n, 1000, dtype=torch.float64)
    sH = torch.zeros(n, dtype=torch.float64)
    lm = torch.zeros(n, 1000, dtype=torch.float64)
    l2 = torch.zeros(n, 1000, dtype=torch.float64)
    hid = torch.zeros(n, dtype=torch.float64)
    for mem in members:
        for s in range(0, n, chunk):
            x = X[s:s + chunk]
            h = ad.block.ln_2(x[:, None, :])[:, 0]
            y0 = torch.nn.functional.gelu(h @ ad.W1 + ad.b1)
            yv = torch.nn.functional.gelu(h @ mem["W1v"] + ad.b1)
            lg = ad.head(ad.enc.ln((x + yv @ mem["W2"] + mem["b2"])[:, None, :])[:, 0])
            p = torch.softmax(lg.double() / T, -1).cpu()
            sp[s:s + chunk] += p
            sH[s:s + chunk] += -(p * torch.log(p + 1e-12)).sum(-1)
            l64 = lg.double().cpu()
            lm[s:s + chunk] += l64
            l2[s:s + chunk] += l64 ** 2
            hid[s:s + chunk] += (yv - y0).double().norm(dim=1).cpu() / M
            del x, h, y0, yv, lg, p, l64
    pbar = (sp / M).numpy()
    ent = -np.sum(pbar * np.log(pbar + 1e-12), -1)
    exp_ent = (sH / M).numpy()
    return {"pbar": pbar, "predictive_entropy": ent,
            "expected_member_entropy": exp_ent,
            "mutual_information": ent - exp_ent,
            "logit_variance": ((l2 / M) - (lm / M) ** 2).clamp(min=0).mean(-1).numpy(),
            "hidden_perturbation_change": hid.numpy()}


def stage_final():
    sel_all = json.loads((F.OUT / "id_selection" / "ksweep_selection.json").read_text())
    ad = ViTPnCAdapter(device="cuda", dtype=torch.float32)
    T = F.temperature()
    corr = fc.load_cache(F.SRC / "raw" / "cache_correction.npz")
    Theta0 = ad.theta().detach().double().cpu().numpy()
    maha = {n: F.load_scores(n)["M1_true_label"] for n in F.SETS}

    caches = {}
    for n in F.SETS:
        c = fc.load_cache(F.cache_path(n))
        caches[n] = (fc.cache_to_gpu(c, ad), c.get("labels"))
    yval = caches["val50k"][1]
    base_logits = torch.from_numpy(np.load(F.SRC / "raw" / "base_val_logits.npy"))
    base_m = clf_metrics(torch.softmax(base_logits.double() / T, -1).numpy(), yval)
    print(f"base on 50k val (T={T}): top-1 {base_m['accuracy']*100:.3f}%  "
          f"NLL {base_m['nll']:.4f}")

    results = {"temperature": T, "M": M_FINAL, "seeds": list(SEEDS_FINAL),
               "base": {"top1": base_m["accuracy"], "nll": base_m["nll"],
                        "ece": base_m["ece"]}, "K": {}}
    for K in KS:
        sel = sel_all[str(K)]["selected"]
        if sel is None:
            results["K"][str(K)] = {"selected": None}
            continue
        r, lam = sel["r_target"], sel["lambda"]
        n_cal = sel["n_cal"]
        h = torch.as_tensor(corr["h"][:n_cal], device=ad.device, dtype=ad.dtype)
        z0 = torch.as_tensor(corr["z0"][:n_cal], device=ad.device, dtype=ad.dtype)
        print(f"\n=== K={K} final: r={r} lambda={lam:g} M={M_FINAL} "
              f"seeds={SEEDS_FINAL} ===", flush=True)
        per_seed, realized, seed0 = [], [], None
        t0 = time.perf_counter()
        for seed in SEEDS_FINAL:
            members, rmed, scale = _build_members(ad, h, z0, Theta0, seed, K, r, lam)
            realized.append(rmed)
            S = {n: _scores(ad, caches[n][0], members, T) for n in F.SETS}
            m = clf_metrics(S["val50k"]["pbar"], yval)
            row = {"seed": seed, "realized_r_median": rmed, "scale": scale,
                   "top1": m["accuracy"], "nll": m["nll"], "ece": m["ece"]}
            for key in ("predictive_entropy", "mutual_information", "logit_variance"):
                om = F.ood_metrics(S["val50k"][key], {d: S[d][key] for d in F.DS})
                row[key] = {"near_auroc": om["near_auroc"], "far_auroc": om["far_auroc"],
                            "near_fpr95": om["near_fpr95"], "far_fpr95": om["far_fpr95"]}
            per_seed.append(row)
            if seed == SEEDS_FINAL[0]:
                allm = np.concatenate([maha[n] for n in F.SETS])
                seed0 = {
                    "spearman_maha_vs_hidden_response": float(spearmanr(
                        allm, np.concatenate([S[n]["hidden_perturbation_change"]
                                              for n in F.SETS])).statistic),
                    "spearman_maha_vs_predictive_entropy": float(spearmanr(
                        allm, np.concatenate([S[n]["predictive_entropy"]
                                              for n in F.SETS])).statistic)}
                np.savez_compressed(
                    F.OUT / "predictions" / f"ksweep_K{K}_seed0.npz",
                    **{f"{n}_{k}": S[n][k].astype(np.float32) for n in F.SETS
                       for k in ("predictive_entropy", "mutual_information",
                                 "logit_variance", "hidden_perturbation_change")})
            del members, S
            reset_cuda()
            print(f"    seed {seed}: top-1 {row['top1']*100:.3f}%  NLL {row['nll']:.4f}  "
                  f"Near {row['predictive_entropy']['near_auroc']*100:.2f}  "
                  f"Far {row['predictive_entropy']['far_auroc']*100:.2f}  "
                  f"({time.perf_counter()-t0:.0f}s)", flush=True)
        agg = {"K": K, "r_target": r, "lambda": lam, "n_cal": n_cal,
               "realized_r_median": float(np.mean(realized)),
               "realized_r_min": float(np.min(realized)),
               "realized_r_max": float(np.max(realized)),
               "top1": float(np.mean([p["top1"] for p in per_seed])),
               "top1_std": float(np.std([p["top1"] for p in per_seed], ddof=1)),
               "nll": float(np.mean([p["nll"] for p in per_seed])),
               "ece": float(np.mean([p["ece"] for p in per_seed])),
               "delta_top1_pp": (np.mean([p["top1"] for p in per_seed])
                                 - base_m["accuracy"]) * 100,
               "per_seed": per_seed, "seed0_diagnostics": seed0}
        for key in ("predictive_entropy", "mutual_information", "logit_variance"):
            for met in ("near_auroc", "far_auroc", "near_fpr95", "far_fpr95"):
                v = [p[key][met] for p in per_seed]
                agg[f"{key}_{met}"] = float(np.mean(v))
                agg[f"{key}_{met}_std"] = float(np.std(v, ddof=1))
        results["K"][str(K)] = agg
        del h, z0
        reset_cuda()
        F.write_json(F.OUT / "metrics" / "ksweep_final.json", results)

    print(f"\n{'K':<5}{'r':>6}{'lam':>8}{'real r':>9}{'ID top1':>10}{'dAcc pp':>9}"
          f"{'Near':>8}{'Far':>8}{'MI Near':>9}{'rho(M,resp)':>13}{'rho(M,H)':>10}")
    for K in KS:
        a = results["K"].get(str(K))
        if not a or a.get("selected") is None and "top1" not in a:
            print(f"  {K:<3}  no passing configuration")
            continue
        d = a["seed0_diagnostics"]
        print(f"  {K:<3}{a['r_target']:>6.2f}{a['lambda']:>8.0f}"
              f"{a['realized_r_median']:>9.4f}{a['top1']*100:>10.3f}"
              f"{a['delta_top1_pp']:>9.3f}{a['predictive_entropy_near_auroc']*100:>8.2f}"
              f"{a['predictive_entropy_far_auroc']*100:>8.2f}"
              f"{a['mutual_information_near_auroc']*100:>9.2f}"
              f"{d['spearman_maha_vs_hidden_response']:>13.4f}"
              f"{d['spearman_maha_vs_predictive_entropy']:>10.4f}")
    print(f"\nwrote {F.OUT/'metrics'/'ksweep_final.json'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["search", "final"])
    a = ap.parse_args()
    stage_search() if a.stage == "search" else stage_final()


if __name__ == "__main__":
    main()
