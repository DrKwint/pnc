"""Shared plumbing for the geometry follow-up round.

Everything upstream (`imagenet_vit`, `imagenet_vit_baselines`,
`imagenet_vit_geometry_scod_llla`) is consumed read-only; new artifacts land under
`imagenet_vit_geometry_followup`.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from . import full_cache as fc
from . import full_ood as fo
from . import full_oodmetrics as fm
from . import pnc_core as pc
from .memprobe import reset_cuda

SRC = Path("results/neurips_2026_rebuttal/imagenet_vit")
FRONTIER = Path("results/neurips_2026_rebuttal/imagenet_vit_preservation_frontier")
BASE = Path("results/neurips_2026_rebuttal/imagenet_vit_baselines")
PREV = Path("results/neurips_2026_rebuttal/imagenet_vit_geometry_scod_llla")
OUT = Path("results/neurips_2026_rebuttal/imagenet_vit_geometry_followup")

DS = ["ssb_hard", "ninco", "inaturalist", "textures", "openimage_o"]
SETS = ["val50k"] + DS
GROUPS = {k: v[2] for k, v in fo.DATASETS.items()}
NEAR = [d for d in DS if GROUPS[d] == "near"]
FAR = [d for d in DS if GROUPS[d] == "far"]


def cache_path(name: str) -> Path:
    return SRC / "raw" / ("cache_val50k.npz" if name == "val50k"
                          else f"cache_ood_{name}.npz")


def temperature() -> float:
    return json.loads((SRC / "metrics" / "temperature.json").read_text())["temperature"]


def load_scores(name: str) -> dict:
    """Per-example scores saved by the previous round's Part A."""
    z = np.load(PREV / "predictions" / f"geometry_scores_{name}.npz")
    return {k: z[k].astype(np.float64) for k in z.files}


def ood_metrics(id_s: np.ndarray, ood_s: dict) -> dict:
    """Near/Far macro aggregates plus per-dataset AUROC/FPR95, higher score = more OOD."""
    per = {d: fm.binary_ood_metrics(id_s, ood_s[d]) for d in DS if d in ood_s}
    out = {"per_dataset": {d: {"auroc": per[d]["auroc"], "fpr95": per[d]["fpr95"]}
                           for d in per}}
    for g, members in (("near", NEAR), ("far", FAR)):
        sub = {k: ood_s[k] for k in members if k in ood_s}
        if sub:
            a = fm.aggregate_family_metrics(id_s, sub)
            out[f"{g}_auroc"] = a["mean_auroc"]
            out[f"{g}_fpr95"] = a["mean_fpr95"]
    return out


# ------------------------------------------------------------------ base model
@torch.inference_mode()
def base_probs(ad, X: torch.Tensor, T: float | None = None, chunk: int = 8192):
    """Base-model probabilities from a cached CLS residual; T=None means raw softmax."""
    out = []
    for s in range(0, X.shape[0], chunk):
        lg = ad.tail(X[s:s + chunk][:, None, :], cls_only=True).double()
        if T is not None:
            lg = lg / T
        out.append(torch.softmax(lg, -1).cpu())
    return torch.cat(out).numpy()


def entropy(p: np.ndarray) -> np.ndarray:
    return -np.sum(p * np.log(p + 1e-12), -1)


# ------------------------------------------------------------------ P&C members
def frozen_config(name: str = "primary") -> dict:
    cfg = json.loads((FRONTIER / "selection" / "frozen_configs.json").read_text())
    return cfg["configs"][name]


def load_primary_members(ad, seed: int = 0, name: str = "primary"):
    c = frozen_config(name)
    z = np.load(FRONTIER / "raw" / f"members_{name}_seed{seed}.npz")
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


@torch.inference_mode()
def member_logits(ad, X: torch.Tensor, mem: dict, chunk: int = 8192) -> torch.Tensor:
    out = []
    for s in range(0, X.shape[0], chunk):
        x = X[s:s + chunk]
        h = ad.block.ln_2(x[:, None, :])[:, 0]
        yv = torch.nn.functional.gelu(h @ mem["W1v"] + ad.b1)
        zc = yv @ mem["W2"] + mem["b2"]
        out.append(ad.head(ad.enc.ln((x + zc)[:, None, :])[:, 0]).cpu())
        del x, h, yv, zc
    return torch.cat(out)


@torch.inference_mode()
def member_phi(ad, X: torch.Tensor, mem: dict | None, chunk: int = 8192) -> torch.Tensor:
    """Post-final-LayerNorm CLS feature under a member (mem=None -> base model)."""
    out = []
    for s in range(0, X.shape[0], chunk):
        x = X[s:s + chunk]
        h = ad.block.ln_2(x[:, None, :])[:, 0]
        if mem is None:
            z = torch.nn.functional.gelu(h @ ad.W1 + ad.b1) @ ad.W2 + ad.b2
        else:
            z = torch.nn.functional.gelu(h @ mem["W1v"] + ad.b1) @ mem["W2"] + mem["b2"]
        out.append(ad.enc.ln((x + z)[:, None, :])[:, 0].cpu())
        del x, h, z
    return torch.cat(out)


# ------------------------------------------------------------------ Gaussian fits
def fit_gaussian(feats: np.ndarray, targets: np.ndarray | None, n_classes: int = 1000,
                 reg: float = 1e-6):
    """Class-conditional (or global) mean(s) + shared precision, matching the frozen
    Mahalanobis baseline's convention exactly."""
    D = feats.shape[1]
    if targets is None:
        mu = feats.astype(np.float64).mean(0, keepdims=True)
        cen = feats.astype(np.float64) - mu
        cov = cen.T @ cen / len(feats) + reg * np.eye(D)
        return mu, np.linalg.inv(cov)
    means = np.zeros((n_classes, D), dtype=np.float64)
    for c in range(n_classes):
        m = targets == c
        if m.sum():
            means[c] = feats[m].astype(np.float64).mean(0)
    cen = feats.astype(np.float64) - means[targets]
    cov = cen.T @ cen / len(feats) + reg * np.eye(D)
    return means, np.linalg.inv(cov)


def maha(feats: np.ndarray, means: np.ndarray, prec: np.ndarray | None,
         chunk: int = 8192, return_class: bool = False):
    """min_c (x-mu_c)^T P (x-mu_c). prec=None gives squared Euclidean."""
    dev = "cuda"
    Mu = torch.as_tensor(means, dtype=torch.float64, device=dev)
    P = (torch.as_tensor(prec, dtype=torch.float64, device=dev) if prec is not None
         else torch.eye(Mu.shape[1], dtype=torch.float64, device=dev))
    PMu = P @ Mu.T
    muPmu = (Mu * PMu.T).sum(1)
    vals, args = [], []
    for s in range(0, len(feats), chunk):
        x = torch.as_tensor(feats[s:s + chunk], dtype=torch.float64, device=dev)
        q = (x * (x @ P)).sum(1)[:, None] - 2.0 * (x @ PMu) + muPmu[None, :]
        mn = q.min(1)
        vals.append(mn.values.cpu().numpy())
        if return_class:
            args.append(mn.indices.cpu().numpy())
        del x, q, mn
    del Mu, P, PMu, muPmu
    reset_cuda()
    v = np.concatenate(vals)
    return (v, np.concatenate(args)) if return_class else v


def calibration_features(ad):
    """CLS features, true labels and base predictions on the frozen 32,768 ID pool."""
    corr = fc.load_cache(SRC / "raw" / "cache_correction.npz")
    X = fc.cache_to_gpu(corr, ad)
    phi = fc.features_from_cache(ad, X).numpy()
    with torch.inference_mode():
        pred = ad.head(torch.as_tensor(phi, device=ad.device,
                                       dtype=ad.dtype)).argmax(-1).cpu().numpy()
    del X
    reset_cuda()
    return phi, corr["labels"], pred


def write_json(path: Path, obj):
    fc.write_json(path, obj)
