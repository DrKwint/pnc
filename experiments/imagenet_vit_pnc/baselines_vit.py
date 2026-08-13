"""Matched post-hoc uncertainty baselines on the frozen ImageNet ViT-B/16.

Ports the repository's own implementations (see `BASELINE_AUDIT.md`); nothing here is a
new method. Both fitted baselines act on the final normalized CLS representation
φ(x) = `encoder.ln(x)[:, 0]` ∈ R⁷⁶⁸ — the representation the classifier consumes and the
one ReAct clips.

  Mahalanobis   `pnc_core/openood_eval.py::_fit_mahalanobis` — single-layer,
                class-conditional Gaussian with a shared covariance and a fixed 1e-6 ridge.
                No OOD-fitted logistic regressor.

  Laplace-KFAC  `pnc_core/ensembles.py::LaplaceEnsemble` +
                `pnc_core/laplace.py::compute_kfac_factors(is_classification=True)`.
                Kronecker factors A = E[â âᵀ] and S = E[d_pre d_preᵀ] with the MC Fisher
                (`y ~ Categorical(p)`, `d_pre = p − onehot(y)`), sampled exactly as the
                source does: eigendecompose both factors, `eig = N·outer(eig_A, eig_S) + λ`,
                `ΔW = U_A (Z ⊙ eig^{-1/2}) U_Sᵀ`.

The dense last-layer Laplace of `LLLAEnsemble` is *not* implemented: at 768→1000 its
covariance is 769,000² = 2.37 TB. That is recorded as MEMORY_INFEASIBLE, not silently
swapped for this Kronecker form.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from experiments.banking77_pnc.pnc_metrics import clf_metrics  # REUSED evaluator

from . import full_cache as fc
from .memprobe import GIB, cpu_peak_rss_gib, reset_cuda

SRC = Path("results/neurips_2026_rebuttal/imagenet_vit")
FRONTIER = Path("results/neurips_2026_rebuttal/imagenet_vit_preservation_frontier")
OUT = Path("results/neurips_2026_rebuttal/imagenet_vit_baselines")

# Spec §12's log grid. The repo's own canonical grid is [1, 10, 100] (mnist_tasks.py:240),
# which is tuned for small MNIST/UCI/gym heads and is catastrophic at ViT scale (top-1
# 9.3-93.4%). §12 sanctions this fallback grid; it is extended upward to 1e8 because the
# ID-NLL optimum otherwise sits on its top edge, and reporting a baseline pinned at a grid
# boundary would understate it. All selection remains ID-only.
PRIOR_PRECISION_GRID = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0, 1e4,
                        3e4, 1e5, 3e5, 1e6, 1e7, 1e8]
M_SAMPLES = 20                      # matches P&C's M = 20 (spec §23)


# --------------------------------------------------------------- Mahalanobis
def fit_mahalanobis(features: np.ndarray, targets: np.ndarray, n_classes: int = 1000):
    """Verbatim `pnc_core.openood_eval._fit_mahalanobis` (float64, 1e-6 ridge)."""
    D = features.shape[1]
    class_means = np.zeros((n_classes, D), dtype=np.float64)
    for c in range(n_classes):
        mask = targets == c
        if mask.sum() > 0:
            class_means[c] = features[mask].astype(np.float64).mean(axis=0)
    centered = features.astype(np.float64) - class_means[targets]
    cov = (centered.T @ centered) / len(features) + 1e-6 * np.eye(D)
    precision = np.linalg.inv(cov)
    return class_means, precision


def mahalanobis_scores(features: np.ndarray, class_means: np.ndarray,
                       precision: np.ndarray, chunk: int = 8192) -> np.ndarray:
    """Minimum squared Mahalanobis distance over classes; higher = more OOD.

    Expanded as (x−μ)ᵀP(x−μ) = xᵀPx − 2xᵀPμ + μᵀPμ. Forming the (n, C, D) difference
    tensor directly would be 25 GB for one chunk at C=1000, D=768; this costs
    O(nD² + nCD) and a few MB.
    """
    P = torch.as_tensor(precision, dtype=torch.float64, device="cuda")
    Mu = torch.as_tensor(class_means, dtype=torch.float64, device="cuda")
    PMu = (P @ Mu.T)                                         # (D, C)
    muPmu = (Mu * (P @ Mu.T).T).sum(1)                       # (C,)
    out = []
    for s in range(0, len(features), chunk):
        x = torch.as_tensor(features[s:s + chunk], dtype=torch.float64, device="cuda")
        xPx = (x * (x @ P)).sum(1)                           # (n,)
        m = xPx[:, None] - 2.0 * (x @ PMu) + muPmu[None, :]  # (n, C)
        out.append(m.min(dim=1).values.cpu().numpy())
        del x, xPx, m
    del P, Mu, PMu, muPmu
    reset_cuda()
    return np.concatenate(out)


# ------------------------------------------------------------- Laplace (KFAC)
@torch.inference_mode()
def kfac_factors(adapter, feats: torch.Tensor, seed: int = 0):
    """A = E[â âᵀ] and S = E[d_pre d_preᵀ] with the MC Fisher, as in `laplace.py`."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    N, D = feats.shape
    ones = torch.ones(N, 1, device=feats.device, dtype=feats.dtype)
    a_hat = torch.cat([feats, ones], 1)                        # (N, D+1)
    A = (a_hat.double().T @ a_hat.double()) / N

    logits = adapter.head(feats)
    probs = torch.softmax(logits.double(), -1)
    idx = torch.multinomial(probs.cpu(), 1, generator=g).squeeze(1)
    onehot = torch.zeros_like(probs)
    onehot[torch.arange(N), idx.to(probs.device)] = 1.0
    d_pre = probs - onehot                                     # (N, K)
    S = (d_pre.T @ d_pre) / N
    return A.cpu().numpy(), S.cpu().numpy(), int(N)


class LaplaceKFACHead:
    """Samples classifier heads from the KFAC posterior, exactly as `LaplaceEnsemble`."""

    def __init__(self, adapter, A, S, prior_precision: float, data_size: int,
                 n_models: int = M_SAMPLES, seed: int = 0):
        self.ad, self.lam, self.N, self.n_models = adapter, float(prior_precision), \
            int(data_size), n_models
        A = np.asarray(A) + np.eye(A.shape[0]) * 1e-6
        S = np.asarray(S) + np.eye(S.shape[0]) * 1e-6
        U_A, eig_A, _ = np.linalg.svd(A)
        U_S, eig_S, _ = np.linalg.svd(S)
        dev, dt = adapter.device, adapter.dtype
        self.U_A = torch.as_tensor(U_A, device=dev, dtype=dt)
        self.U_S = torch.as_tensor(U_S, device=dev, dtype=dt)
        eig = self.N * np.outer(np.maximum(eig_A, 0.0), np.maximum(eig_S, 0.0)) + self.lam
        self.std = torch.as_tensor(1.0 / np.sqrt(eig), device=dev, dtype=dt)
        head = adapter.model.heads[-1] if hasattr(adapter.model.heads, "__getitem__") \
            else adapter.model.heads
        self.W_map = head.weight.T.detach().clone()            # (D, K)
        self.b_map = head.bias.detach().clone()                # (K,)
        self.W_full = torch.cat([self.W_map, self.b_map[None, :]], 0)   # (D+1, K)
        self.seed = seed

    def sample(self, i: int):
        g = torch.Generator(device="cpu").manual_seed(self.seed * 1000 + i)
        Z = torch.randn(self.W_full.shape, generator=g).to(self.W_full.device,
                                                           self.W_full.dtype)
        dW = self.U_A @ (Z * self.std) @ self.U_S.T
        W_new = self.W_full + dW
        return W_new[:-1, :], W_new[-1, :]

    @torch.inference_mode()
    def predict_probs(self, feats: torch.Tensor, T: float = 1.0, chunk: int = 8192):
        """Mean softmax over `n_models` posterior head samples."""
        acc = None
        for i in range(self.n_models):
            W, b = self.sample(i)
            parts = []
            for s in range(0, feats.shape[0], chunk):
                parts.append(torch.softmax((feats[s:s + chunk] @ W + b).double() / T, -1))
            p = torch.cat(parts)
            acc = p if acc is None else acc + p
            del p, parts
        return (acc / self.n_models).cpu().numpy()


def predictive_entropy(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64)
    return -np.sum(p * np.log(p + 1e-12), axis=-1)


def id_metrics(probs: np.ndarray, labels: np.ndarray, base_pred: np.ndarray) -> dict:
    m = clf_metrics(probs, labels)
    pred = probs.argmax(-1)
    top5 = torch.from_numpy(probs).topk(5, -1).indices.numpy()
    return {"top1": m["accuracy"],
            "top5": float((top5 == labels[:, None]).any(-1).mean()),
            "nll": m["nll"], "ece": m["ece"], "brier": m["brier"],
            "base_agreement": float((pred == base_pred).mean()),
            "mean_pred_entropy": float(predictive_entropy(probs).mean())}


def write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str))
