"""Final-block CLS-conditioned P&C: member construction and ensemble evaluation (spec §3).

Construction reuses the preflight machinery unchanged — the perturbation basis /
coefficient / scale conventions of ``banking77_pnc.construct`` and the original-centred
ridge solve of ``pnc_theory.linalg.ridge_solve``, in its streamed sufficient-statistic form
(bitwise identical, gated in ``validate.py``).

One correction row per calibration image: the final-block CLS post-GELU activation entering
W2, with the base model's own W2 output for that row as the target. No patch tokens.

Scale is parameterised by the *realized* relative Frobenius norm

    r = ||dW1||_F / ||W1||_F

so configurations are comparable across implementations rather than reported as an
opaque multiplier (spec §8).
"""
from __future__ import annotations

import time

import numpy as np
import torch

from . import pnc_core as pc

RIDGE_DEFAULT = 1e-3


def scale_for_r(U: np.ndarray, coeffs: np.ndarray, W1_norm: float, r: float) -> float:
    """Numerical scale whose median realized ||dW1||_F/||W1||_F equals `r`."""
    return pc.base_scale(U, coeffs, W1_norm, target_rel=r)


def realized_r(U: np.ndarray, coeffs: np.ndarray, scale: float, W1_norm: float) -> dict:
    rs = [float(np.linalg.norm(scale * (coeffs[m] @ U)) / W1_norm) for m in range(len(coeffs))]
    return {"median": float(np.median(rs)), "min": float(np.min(rs)),
            "max": float(np.max(rs)), "per_member": rs}


class MemberFactory:
    """Builds P&C members from a cached (h, z0) correction set.

    The cache makes construction independent of the ViT prefix, so a member costs one
    GELU + two Gram accumulations + one shared Cholesky.
    """

    def __init__(self, adapter, h: torch.Tensor, z0: torch.Tensor, seed: int, K: int = 20,
                 M: int = 20):
        self.ad = adapter
        self.h = h                      # (N, 768) on GPU
        self.z0 = z0                    # (N, 768) on GPU
        self.seed, self.K, self.M = seed, K, M
        self.U = pc.perturbation_basis(seed, K)
        self.coeffs = pc.member_coefficients(seed, M, K)
        self.W1_norm = float(adapter.W1.norm())
        self.Theta0 = adapter.theta().detach().double().cpu().numpy()

    def scale(self, r: float) -> float:
        return scale_for_r(self.U, self.coeffs, self.W1_norm, r)

    def dW1(self, m: int, scale: float) -> torch.Tensor:
        return torch.as_tensor(pc.member_dW1(self.U, self.coeffs[m], scale),
                               device=self.ad.device, dtype=self.ad.dtype)

    @torch.inference_mode()
    def build(self, r: float, lam: float, n_cal: int, m_list=None) -> dict:
        """Construct members at scale r, ridge lam, using the first n_cal cached rows."""
        scale = self.scale(r)
        h = self.h[:n_cal]
        z0 = self.z0[:n_cal]
        members, diag = [], {"calib_residual": [], "solve_s": [], "gpu_s": []}
        for m in (m_list if m_list is not None else range(self.M)):
            t0 = time.perf_counter()
            W1v = self.ad.W1 + self.dW1(m, scale)
            y = torch.nn.functional.gelu(h @ W1v + self.ad.b1)
            X = pc.SufficientStats.augment(y)
            G = (X.T @ X).double().cpu().numpy()
            C = (X.T @ z0).double().cpu().numpy()
            yty = float((z0.double() ** 2).sum())
            torch.cuda.synchronize()
            t_gpu = time.perf_counter() - t0
            t1 = time.perf_counter()
            Theta, _, _ = pc.cho_solve_shared(G, C, lam, w_prior=self.Theta0)
            t_solve = time.perf_counter() - t1
            diag["calib_residual"].append(
                pc.relative_residual({"G": G, "C": C, "yty": yty}, Theta))
            diag["solve_s"].append(t_solve)
            diag["gpu_s"].append(t_gpu)
            members.append({
                "m": int(m), "coeff": self.coeffs[m],
                "W1v": W1v,
                "W2": torch.as_tensor(Theta[1:], device=self.ad.device, dtype=self.ad.dtype),
                "b2": torch.as_tensor(Theta[0], device=self.ad.device, dtype=self.ad.dtype),
                "finite": bool(np.isfinite(Theta).all()),
            })
            del y, X
        return {"members": members, "scale": scale, "r": r, "lam": lam, "n_cal": n_cal,
                "realized_r": realized_r(self.U, self.coeffs, scale, self.W1_norm),
                "calib_residual_median": float(np.median(diag["calib_residual"])),
                "calib_residual_per_member": diag["calib_residual"],
                "construct_gpu_s_median": float(np.median(diag["gpu_s"])),
                "construct_solve_s_median": float(np.median(diag["solve_s"])),
                "construct_total_s": float(np.sum(diag["gpu_s"]) + np.sum(diag["solve_s"])),
                "all_finite": all(m["finite"] for m in members)}


@torch.inference_mode()
def member_logits(adapter, X: torch.Tensor, member: dict | None, corrected: bool = True,
                  chunk: int = 8192) -> torch.Tensor:
    """Logits for one member over a cached CLS residual. `member=None` -> base model.

    ``corrected=False`` applies the identical W1 mutation with the **original** W2/b2 —
    the matched uncorrected ablation of spec §18.
    """
    if member is None:
        W1 = W2 = b2 = None
    else:
        W1 = member["W1v"]
        W2 = member["W2"] if corrected else None
        b2 = member["b2"] if corrected else None
    out = []
    for s in range(0, X.shape[0], chunk):
        xb = X[s:s + chunk][:, None, :]
        out.append(adapter.tail(xb, W1=W1, W2=W2, b2=b2, cls_only=True).cpu())
    return torch.cat(out)


@torch.inference_mode()
def ensemble_probs(adapter, X: torch.Tensor, members: list, T: float = 1.0,
                   corrected: bool = True, chunk: int = 8192, return_members: bool = False):
    """Temperature-scaled mean softmax over members, evaluated sequentially.

    Follows the existing classification protocol: a single shared temperature is applied to
    each member's logits before the softmax, and the member probabilities are averaged.
    Peak memory is independent of M — one member's logits exist at a time.
    """
    acc = None
    per_member = []
    for mem in members:
        lg = member_logits(adapter, X, mem, corrected=corrected, chunk=chunk)
        p = torch.softmax(lg.double() / T, dim=-1)
        acc = p if acc is None else acc + p
        if return_members:
            per_member.append(p.float())
        del lg, p
    pbar = (acc / len(members)).float()
    return (pbar, per_member) if return_members else pbar


def predictive_entropy(probs: np.ndarray) -> np.ndarray:
    p = np.asarray(probs, dtype=np.float64)
    return -np.sum(p * np.log(p + 1e-12), axis=-1)


def member_disagreement(per_member: list) -> dict:
    """Exploratory member statistics (spec §20). Headline score stays predictive entropy."""
    P = np.stack([p.numpy().astype(np.float64) for p in per_member])     # (M, N, C)
    pbar = P.mean(0)
    ent = lambda q: -np.sum(q * np.log(q + 1e-12), -1)
    pred_ent = ent(pbar)
    exp_ent = ent(P).mean(0)
    return {"predictive_entropy": pred_ent,
            "mean_member_entropy": exp_ent,
            "mutual_information": pred_ent - exp_ent,
            "logit_variance": P.var(0).mean(-1)}


def logit_mse_distribution(base_logits: torch.Tensor, member_logit_list: list) -> dict:
    """Per (member, image) logit MSE vs the base model, summarised by quantiles.

    The preflight showed this distribution is heavy-tailed, so the mean alone is not a
    usable preservation statistic (spec §8: always report mean, median and p99).
    """
    b = base_logits.double().numpy()
    mses = np.concatenate([((lg.double().numpy() - b) ** 2).mean(-1)
                           for lg in member_logit_list])
    q = np.percentile(mses, [50, 90, 95, 99])
    return {"mean": float(mses.mean()), "median": float(q[0]), "p90": float(q[1]),
            "p95": float(q[2]), "p99": float(q[3]), "max": float(mses.max()),
            "n_pairs": int(mses.size)}
