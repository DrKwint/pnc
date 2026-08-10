"""Dense full-covariance GGN, eigensystem, prior-precision grid, and posterior samples for SLL.

G_S = F^T F,  F = [F_1; ...; F_N] stacked Fisher-weighted selected Jacobians (F_n = dztilde/dw_sub,
[C,S]).  Posterior q(w_S) = N(w_S0, (G_S + lambda_S I)^{-1}).  Samples via the eigensystem:
    G_S = U diag(d) U^T,   Delta w_m = U diag((d+lambda)^{-1/2}) eps_m,  eps fixed ~ N(0,I).
Everything is S x S (S <= 2048) so this is tiny; no dense all-network covariance is ever formed.
"""
from __future__ import annotations

import time
import numpy as np
import jax.numpy as jnp


def build_dense_ggn(fns, x_calib, S: int, batch: int = 1, log_every: int = 256):
    """G_S = sum_n F_n^T F_n via per-example Fisher-weighted selected Jacobians (F_n = dztilde/dw_sub,
    [C,S]). Per-example (not vmapped) -- vmapping jacrev over a batch OOMs / triggers cuDNN autotune
    failures. Returns (G[S,S], stats)."""
    sel_jac_z = fns["sel_jac_ztilde"]; w0 = fns["w_sub0"]
    N = int(len(x_calib)); Xd = jnp.asarray(x_calib)
    G = np.zeros((S, S), dtype=np.float64)
    t0 = time.time()
    for i in range(N):
        Fn = np.asarray(sel_jac_z(w0, Xd[i]))                  # (C, S)
        G += Fn.T.astype(np.float64) @ Fn.astype(np.float64)
        if log_every and i % log_every == 0:
            print(f"  [dense-ggn] {i}/{N} ({time.time()-t0:.0f}s)", flush=True)
    G = 0.5 * (G + G.T)
    sym_err = float(np.max(np.abs(G - G.T)))
    ev = np.linalg.eigvalsh(G)
    stats = dict(build_seconds=round(time.time() - t0, 1), sym_error=sym_err,
                 eig_min=float(ev[0]), eig_max=float(ev[-1]),
                 numerical_rank=int(np.sum(ev > ev[-1] * S * np.finfo(np.float64).eps)),
                 cond_estimate=float(ev[-1] / max(ev[ev > 0].min(), 1e-300)) if np.any(ev > 0) else np.inf)
    return G, stats


def eigensystem(G: np.ndarray):
    """G = U diag(d) U^T, d ascending -> return d (S,), U (S,S)."""
    d, U = np.linalg.eigh(G)
    d = np.clip(d, 0.0, None)
    return d, U


def prior_grid(d: np.ndarray):
    """Data-adaptive grid lambda_S = m * {1e-4..100}, m = median strictly-positive eigenvalue,
    plus an absolute PD floor."""
    pos = d[d > 0]
    m = float(np.median(pos)) if len(pos) else 1.0
    mults = np.array([1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0])
    floor = max(1e-6, 1e-8 * float(d.max()) if d.max() > 0 else 1e-6)
    grid = np.maximum(m * mults, floor)
    return grid, m, floor


def sample_perturbations(d: np.ndarray, U: np.ndarray, lam: float, M: int, sample_seed: int):
    """Delta w = U diag((d+lam)^{-1/2}) eps,  eps ~ N(0,I) fixed (common random numbers)."""
    S = len(d)
    rng = np.random.RandomState(sample_seed)
    eps = rng.standard_normal((S, M)).astype(np.float64)      # fixed across lambda candidates
    scale = 1.0 / np.sqrt(d + lam)                             # (S,)
    dW = U @ (scale[:, None] * eps)                            # (S, M)
    return dW.astype(np.float32), eps.astype(np.float32)


def posterior_cov_diag_of_logits(J_S: np.ndarray, d: np.ndarray, U: np.ndarray, lam: float):
    """Exact diagonal of J_S (G+lam I)^{-1} J_S^T  (C,) for a single example. For validation gate 7."""
    # (G+lam)^{-1} = U diag(1/(d+lam)) U^T ; JS: (C,S)
    A = J_S @ U                                                # (C,S)
    return np.sum((A ** 2) / (d + lam)[None, :], axis=1)       # (C,)
