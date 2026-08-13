"""Preservation-frontier search: how far can r go before the *corrected* model stops
preserving ImageNet accuracy?

Implements `ID_SELECTION_RULE.md`. Two things make the grid affordable:

* **Gram reuse.** For a fixed (scale, seed, member) the design ``X = [1, gelu(h W1v + b1)]``
  and the target ``z0`` do not depend on the ridge, so ``G = XᵀX`` and ``C = Xᵀz0`` are
  accumulated once and all eight λ values are eight Cholesky solves on the same statistics.
  That turns an 8× GPU cost into 1× GPU + 8× cheap CPU.
* **Multinomial bootstrap.** The paired difference ``d_i`` takes only a handful of distinct
  values (seed-averaged correctness minus base correctness), so a bootstrap replicate is a
  multinomial draw over the value counts rather than 8,192 index draws. This is exactly
  equivalent to resampling examples and makes 10,000 replicates per configuration free.

Member directions are drawn once per seed and reused at every scale (only the scalar
multiplier changes), so scale comparisons are paired.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from experiments.banking77_pnc.pnc_metrics import clf_metrics  # REUSED evaluator

from . import full_cache as fc
from . import pnc_core as pc
from .memprobe import GIB, reset_cuda
from .vit_adapter import ViTPnCAdapter

SRC = Path("results/neurips_2026_rebuttal/imagenet_vit")            # reused artefacts
OUT = Path("results/neurips_2026_rebuttal/imagenet_vit_preservation_frontier")

LAMBDAS = [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0, 10000.0]
COARSE_R = [0.50, 0.75, 1.00, 1.25, 1.50, 2.00, 2.50, 3.00, 4.00]
EXTEND_UP = [6.0, 8.0]
EXTEND_DOWN = [0.375, 0.25]
BUDGETS = {"strict": 0.0025, "primary": 0.0050, "relaxed": 0.0100}
PRIMARY = "primary"
SEEDS = (0, 10, 42)
M_SEARCH = 10
K = 20
N_CAL = 32768
BOOTSTRAP_N = 10000
BOOTSTRAP_SEED = 20260814


# ---------------------------------------------------------------- bootstrap
def paired_bootstrap_lcb(d: np.ndarray, n_rep: int = BOOTSTRAP_N,
                         seed: int = BOOTSTRAP_SEED, alpha: float = 0.05) -> dict:
    """One-sided 95% lower bound on mean(d) by paired nonparametric bootstrap.

    ``d`` is the per-example paired difference (seed-averaged P&C correctness minus base
    correctness). It takes few distinct values, so each replicate is drawn as a multinomial
    over the empirical value counts — identical in distribution to resampling example
    indices with replacement, but O(#values) instead of O(n) per replicate.
    """
    vals, counts = np.unique(d, return_counts=True)
    n = int(counts.sum())
    rng = np.random.RandomState(seed)
    draws = rng.multinomial(n, counts / n, size=n_rep)        # (n_rep, #values)
    means = draws @ vals / n
    return {"point": float(d.mean()),
            "lcb": float(np.percentile(means, 100 * alpha)),
            "ucb": float(np.percentile(means, 100 * (1 - alpha))),
            "boot_mean": float(means.mean()), "boot_std": float(means.std(ddof=1)),
            "n_examples": n, "n_replicates": n_rep, "bootstrap_seed": seed}


# ---------------------------------------------------------------- searcher
class FrontierSearcher:
    """Holds the reused caches; evaluates a whole ridge curve at one scale per seed."""

    def __init__(self, temperature: float | None = None, m: int = M_SEARCH,
                 n_cal: int = N_CAL, seeds=SEEDS):
        self.ad = ViTPnCAdapter(device="cuda", dtype=torch.float32)
        self.M, self.n_cal, self.seeds = m, n_cal, tuple(seeds)

        corr = fc.load_cache(SRC / "raw" / "cache_correction.npz")
        self.h = torch.as_tensor(corr["h"][:n_cal], device=self.ad.device,
                                 dtype=self.ad.dtype)
        self.z0 = torch.as_tensor(corr["z0"][:n_cal], device=self.ad.device,
                                  dtype=self.ad.dtype)
        sel = fc.load_cache(SRC / "raw" / "cache_selection.npz")
        self.Xsel = torch.as_tensor(sel["x_resid_cls"], device=self.ad.device,
                                    dtype=self.ad.dtype)
        self.ysel = sel["labels"]
        self.h_sel = self.ad.block.ln_2(self.Xsel[:, None, :])[:, 0]
        self.z0_sel = (torch.nn.functional.gelu(self.h_sel @ self.ad.W1 + self.ad.b1)
                       @ self.ad.W2 + self.ad.b2)

        if temperature is None:
            temperature = json.loads(
                (SRC / "metrics" / "temperature.json").read_text())["temperature"]
        self.T = float(temperature)

        self.base_logits = fc.logits_from_cache(self.ad, self.Xsel)
        bm = clf_metrics(torch.softmax(self.base_logits.double() / self.T, -1).numpy(),
                         self.ysel)
        self.base_top1, self.base_nll, self.base_ece = bm["accuracy"], bm["nll"], bm["ece"]
        self.base_correct = (self.base_logits.argmax(-1).numpy() == self.ysel).astype(
            np.float64)
        self.base_pred = self.base_logits.argmax(-1).numpy()
        self.Theta0 = self.ad.theta().detach().double().cpu().numpy()
        self.W1_norm = float(self.ad.W1.norm())
        # basis + member directions drawn once per seed and reused at every scale
        self._basis = {s: (pc.perturbation_basis(s, K),
                           pc.member_coefficients(s, self.M, K)) for s in self.seeds}
        print(f"  base on ID selection pool (T={self.T}): top-1 {self.base_top1*100:.3f}%  "
              f"NLL {self.base_nll:.4f}  ECE {self.base_ece:.4f}", flush=True)

    def scale_for(self, seed: int, r: float) -> float:
        U, co = self._basis[seed]
        return pc.base_scale(U, co, self.W1_norm, target_rel=r)

    # ---- one scale, one seed, the whole ridge curve ----
    @torch.inference_mode()
    def _one_seed(self, r: float, seed: int, lambdas) -> dict:
        U, co = self._basis[seed]
        scale = self.scale_for(seed, r)
        N = self.Xsel.shape[0]
        acc = {lam: torch.zeros(N, 1000, device=self.ad.device, dtype=torch.float64)
               for lam in lambdas}
        acc_u = torch.zeros(N, 1000, device=self.ad.device, dtype=torch.float64)
        mse = {lam: [] for lam in lambdas}
        mse_u, calib, held, rs, finite = [], [], [], [], True
        base_gpu = self.base_logits.to(self.ad.device).double()

        for m in range(self.M):
            dW1 = torch.as_tensor(pc.member_dW1(U, co[m], scale), device=self.ad.device,
                                  dtype=self.ad.dtype)
            rs.append(float(torch.linalg.norm(dW1.double()) / self.W1_norm))
            W1v = self.ad.W1 + dW1
            # --- correction statistics: computed once, shared by every ridge ---
            y_c = torch.nn.functional.gelu(self.h @ W1v + self.ad.b1)
            Xc = pc.SufficientStats.augment(y_c)
            G = (Xc.T @ Xc).double().cpu().numpy()
            C = (Xc.T @ self.z0).double().cpu().numpy()
            yty = float((self.z0.double() ** 2).sum())
            del y_c, Xc
            # --- selection-pool activations for this member ---
            y_s = torch.nn.functional.gelu(self.h_sel @ W1v + self.ad.b1)
            Xs = pc.SufficientStats.augment(y_s)
            zu = y_s @ self.ad.W2 + self.ad.b2
            lu = self.ad.head(self.ad.enc.ln((self.Xsel + zu)[:, None, :])[:, 0])
            acc_u += torch.softmax(lu.double() / self.T, -1)
            mse_u.append(((lu.double() - base_gpu) ** 2).mean(-1).cpu().numpy())
            for lam in lambdas:
                Theta, _, _ = pc.cho_solve_shared(G, C, lam, w_prior=self.Theta0)
                finite &= bool(np.isfinite(Theta).all())
                if m == 0:
                    calib.append(pc.relative_residual({"G": G, "C": C, "yty": yty}, Theta))
                    Th = torch.as_tensor(Theta, device=self.ad.device, dtype=torch.float64)
                    pred = Xs.double() @ Th
                    held.append(float(torch.linalg.norm(pred - self.z0_sel.double())
                                      / (torch.linalg.norm(self.z0_sel.double()) + 1e-30)))
                W2c = torch.as_tensor(Theta[1:], device=self.ad.device, dtype=self.ad.dtype)
                b2c = torch.as_tensor(Theta[0], device=self.ad.device, dtype=self.ad.dtype)
                z = y_s @ W2c + b2c
                lg = self.ad.head(self.ad.enc.ln((self.Xsel + z)[:, None, :])[:, 0])
                acc[lam] += torch.softmax(lg.double() / self.T, -1)
                mse[lam].append(((lg.double() - base_gpu) ** 2).mean(-1).cpu().numpy())
                del W2c, b2c, z, lg
            del dW1, W1v, y_s, Xs, zu, lu
        out = {"scale": scale, "realized_r_median": float(np.median(rs)),
               "realized_r_min": float(np.min(rs)), "realized_r_max": float(np.max(rs)),
               "all_finite": finite, "lambdas": {}}
        pu = (acc_u / self.M).cpu().numpy()
        out["uncorrected"] = self._metrics(pu, mse_u)
        for i, lam in enumerate(lambdas):
            p = (acc[lam] / self.M).cpu().numpy()
            d = self._metrics(p, mse[lam])
            d["calib_residual"] = calib[i]
            d["heldout_cls_residual"] = held[i]
            d["correct"] = (p.argmax(-1) == self.ysel).astype(np.float64)
            out["lambdas"][lam] = d
            del p
        del acc, acc_u, base_gpu
        reset_cuda()
        return out

    def _metrics(self, pbar: np.ndarray, mse_list) -> dict:
        m = clf_metrics(pbar, self.ysel)
        pred = pbar.argmax(-1)
        top5 = torch.from_numpy(pbar).topk(5, -1).indices.numpy()
        v = np.concatenate(mse_list)
        q = np.percentile(v, [50, 90, 95, 99])
        ent = -np.sum(pbar * np.log(pbar + 1e-12), -1)
        return {"top1": m["accuracy"],
                "top5": float((top5 == self.ysel[:, None]).any(-1).mean()),
                "nll": m["nll"], "ece": m["ece"], "brier": m["brier"],
                "base_agreement": float((pred == self.base_pred).mean()),
                "mean_pred_entropy": float(ent.mean()),
                "logit_mse_mean": float(v.mean()), "logit_mse_median": float(q[0]),
                "logit_mse_p90": float(q[1]), "logit_mse_p95": float(q[2]),
                "logit_mse_p99": float(q[3])}

    # ---- one scale, aggregated over seeds ----
    def evaluate_scale(self, r: float, lambdas=LAMBDAS) -> list[dict]:
        t0 = time.perf_counter()
        per_seed = {s: self._one_seed(r, s, lambdas) for s in self.seeds}
        rows = []
        for lam in lambdas:
            ds = [per_seed[s]["lambdas"][lam] for s in self.seeds]
            corr_mat = np.stack([d.pop("correct") for d in ds])          # (S, N)
            d_paired = corr_mat.mean(0) - self.base_correct
            boot = paired_bootstrap_lcb(d_paired)
            agg = {k: float(np.mean([d[k] for d in ds]))
                   for k in ds[0] if isinstance(ds[0][k], float)}
            std = {f"{k}_std": float(np.std([d[k] for d in ds], ddof=1))
                   for k in ("top1", "nll", "ece")}
            per_seed_top1 = [d["top1"] for d in ds]
            row = {
                "r_target": r,
                "realized_r_median": float(np.mean(
                    [per_seed[s]["realized_r_median"] for s in self.seeds])),
                "realized_r_min": float(np.min(
                    [per_seed[s]["realized_r_min"] for s in self.seeds])),
                "realized_r_max": float(np.max(
                    [per_seed[s]["realized_r_max"] for s in self.seeds])),
                "lambda": lam, "M": self.M, "n_cal": self.n_cal,
                "seeds": list(self.seeds), **agg, **std,
                "delta_top1": agg["top1"] - self.base_top1,
                "delta_top1_pp": (agg["top1"] - self.base_top1) * 100,
                "lcb": boot["lcb"], "lcb_pp": boot["lcb"] * 100,
                "ucb_pp": boot["ucb"] * 100,
                "bootstrap_point_pp": boot["point"] * 100,
                "per_seed_top1": per_seed_top1,
                "seed_spread_pp": (max(per_seed_top1) - min(per_seed_top1)) * 100,
                "all_finite": all(per_seed[s]["all_finite"] for s in self.seeds),
                "base_top1": self.base_top1, "base_nll": self.base_nll,
                "base_ece": self.base_ece,
            }
            row["pathology"] = self._pathology(row)
            for name, eps in BUDGETS.items():
                row[f"pass_{name}"] = bool(row["lcb"] >= -eps and not row["pathology"])
            rows.append(row)
        # uncorrected control (diagnostic only; identical directions and scale)
        u = [per_seed[s]["uncorrected"] for s in self.seeds]
        unc = {f"unc_{k}": float(np.mean([x[k] for x in u]))
               for k in u[0] if isinstance(u[0][k], float)}
        for row in rows:
            row.update(unc)
            row["unc_delta_top1_pp"] = (unc["unc_top1"] - self.base_top1) * 100
        print(f"    r={r:<6g} done in {time.perf_counter()-t0:.0f}s", flush=True)
        return rows

    def _pathology(self, row: dict) -> str:
        why = []
        if not row["all_finite"] or not np.isfinite(row["nll"]) or not np.isfinite(row["ece"]):
            why.append("non_finite")
        if row["nll"] > 2.0 * self.base_nll:
            why.append(f"nll_{row['nll']:.3f}_vs_base_{self.base_nll:.3f}")
        if row["ece"] > 3.0 * self.base_ece:
            why.append(f"ece_{row['ece']:.3f}_vs_base_{self.base_ece:.3f}")
        return ";".join(why)


# ---------------------------------------------------------------- selection
def best_lambda(rows: list[dict], budget: str) -> dict | None:
    """§12: among ridges passing this budget at this scale — lowest NLL, then ECE, then λ."""
    ok = [r for r in rows if r[f"pass_{budget}"]]
    if not ok:
        return None
    best = min(r["nll"] for r in ok)
    tied = [r for r in ok if r["nll"] - best < 0.002]
    tied.sort(key=lambda r: (round(r["ece"], 6), r["lambda"]))
    return tied[0]


def pick_scale(all_rows: list[dict], budget: str) -> dict | None:
    """§13: largest r with any passing ridge, using that scale's ID-selected ridge."""
    by_r: dict[float, list[dict]] = {}
    for r in all_rows:
        by_r.setdefault(r["r_target"], []).append(r)
    passing = [r for r in sorted(by_r) if best_lambda(by_r[r], budget) is not None]
    return best_lambda(by_r[max(passing)], budget) if passing else None


def boundary(all_rows: list[dict], budget: str) -> dict:
    """Largest passing scale and the smallest failing scale above it."""
    by_r: dict[float, list[dict]] = {}
    for r in all_rows:
        by_r.setdefault(r["r_target"], []).append(r)
    passes = {r: best_lambda(by_r[r], budget) is not None for r in sorted(by_r)}
    p = [r for r, ok in passes.items() if ok]
    f = [r for r, ok in passes.items() if not ok]
    largest_pass = max(p) if p else None
    smallest_fail_above = min([r for r in f if largest_pass is not None
                               and r > largest_pass], default=None)
    return {"largest_passing_r": largest_pass,
            "smallest_failing_r_above": smallest_fail_above,
            "tested": {str(r): bool(ok) for r, ok in passes.items()}}
