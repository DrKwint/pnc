"""Post-hoc OOD baselines on the identical checkpoint, preprocessing and evaluator (§21-22).

All three read the same cached CLS residual as P&C, so they see exactly the same images
through exactly the same pipeline; only the score differs.

    MSP              -max softmax probability
    Energy           -logsumexp(logits / T) * T          (T = 1, the canonical form)
    ReAct + Energy   Energy after clipping the penultimate feature at the ID p-th
                     percentile, following Sun et al. (2021)

ReAct needs one ID-derived constant (the clipping threshold). Per §22 it is taken from the
designated ID **training** calibration pool, never from validation and never from OOD.

Every score is oriented so that **higher means more OOD**, matching
``banking77_pnc.pnc_metrics.ood_metrics``.
"""
from __future__ import annotations

import numpy as np
import torch

from . import full_cache as fc


def msp_score(logits: torch.Tensor) -> np.ndarray:
    p = torch.softmax(logits.double(), -1)
    return (-p.max(-1).values).numpy()


def energy_score(logits: torch.Tensor, T: float = 1.0) -> np.ndarray:
    return (-T * torch.logsumexp(logits.double() / T, dim=-1)).numpy()


def react_threshold(features: torch.Tensor, percentile: float = 90.0) -> float:
    """ID activation percentile used as the ReAct clipping level."""
    return float(np.percentile(features.double().numpy().ravel(), percentile))


@torch.inference_mode()
def react_energy_score(adapter, X: torch.Tensor, c: float, T: float = 1.0,
                       chunk: int = 8192) -> np.ndarray:
    """Energy computed after clipping penultimate features at `c`."""
    out = []
    for s in range(0, X.shape[0], chunk):
        xb = X[s:s + chunk][:, None, :]
        f = adapter.features(xb, cls_only=True).clamp(max=c)
        out.append(adapter.head(f).cpu())
    return energy_score(torch.cat(out), T=T)


@torch.inference_mode()
def base_scores(adapter, X: torch.Tensor, react_c: float | None = None,
                chunk: int = 8192) -> dict:
    """MSP / Energy / ReAct+Energy for one cached image set, plus the base logits."""
    logits = fc.logits_from_cache(adapter, X, chunk=chunk)
    out = {"logits": logits,
           "msp": msp_score(logits),
           "energy": energy_score(logits)}
    if react_c is not None:
        out["react_energy"] = react_energy_score(adapter, X, react_c, chunk=chunk)
    return out
