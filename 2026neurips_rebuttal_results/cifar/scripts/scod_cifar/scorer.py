"""SCOD posterior-predictive OOD score (JAX).

At a test input x, let L_x = d ztilde / d w (C x P), the Fisher-weighted parameter Jacobian.
With the recovered training-Fisher eigensystem (eigenvalues lambda_j, eigenvectors u_j = U[:, j])
SCOD applies the shrinkage

    s_j^2 = lambda_j / (lambda_j + 1 / (2 M_eps))

and returns the RESIDUAL Fisher energy outside the shrunk training eigenspace:

    score(x) = ||L_x||_F^2  -  sum_j s_j^2 * ||L_x u_j||^2 .

Since u_j are orthonormal and 0 <= s_j^2 <= 1, score(x) >= 0 (gate 9). Orientation: larger score
= more epistemically uncertain / more OOD (energy in directions the training Fisher does not
explain), which matches the evaluator's "higher = more OOD" convention -- no negation.

Efficiency: per example we compute only  total = ||L_x||_F^2  and  captured_j = ||L_x u_j||^2
(j = 1..k_max). Every (k, M_eps) score is then a cheap post-hoc reduction of (total, captured),
so rank/M_eps sensitivity grids need no re-evaluation of Jacobians.
"""
from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp


def make_score_feature_fn(ztilde_fn, U: jnp.ndarray):
    """Return jitted feat(flat_w, x1) -> (total, captured[k_max]).

    total    = ||L_x||_F^2
    captured = [ ||L_x u_j||^2 ]_j   (sum over the C output coords of (L_x U)^2)
    """
    Ud = jnp.asarray(U)  # (P, k_max) resident on device

    @jax.jit
    def feat(flat_w, x1):
        Lx = jax.jacrev(ztilde_fn)(flat_w, x1)   # (C, P)
        proj = Lx @ Ud                            # (C, k)
        total = jnp.sum(Lx * Lx)
        captured = jnp.sum(proj * proj, axis=0)   # (k,)
        return total, captured

    return feat


def compute_score_features(ztilde_fn, flat_w, X, U, log_every: int = 1000, tag: str = ""):
    """Per-example (total, captured) features for all images in X. Returns (total[N], captured[N,k])."""
    import time
    feat = make_score_feature_fn(ztilde_fn, U)
    N = int(X.shape[0])
    Xd = jnp.asarray(X)
    totals = np.zeros(N, dtype=np.float64)
    captured = np.zeros((N, U.shape[1]), dtype=np.float64)
    t0 = time.time()
    for i in range(N):
        t, c = feat(flat_w, Xd[i])
        totals[i] = float(t)
        captured[i] = np.asarray(c, dtype=np.float64)
        if log_every and i % log_every == 0:
            print(f"  [score{(' '+tag) if tag else ''}] {i}/{N} ({time.time()-t0:.0f}s)", flush=True)
    return totals, captured


def shrinkage_sq(eigvals: np.ndarray, Meps: float) -> np.ndarray:
    """s_j^2 = lambda_j / (lambda_j + 1/(2 Meps))."""
    ev = np.asarray(eigvals, dtype=np.float64)
    return ev / (ev + 1.0 / (2.0 * float(Meps)))


def score_from_features(total: np.ndarray, captured: np.ndarray, eigvals: np.ndarray,
                        k: int, Meps: float) -> np.ndarray:
    """SCOD residual score for the first-k eigendirections at a given Meps. Vectorised over N."""
    k = min(int(k), captured.shape[1])
    s2 = shrinkage_sq(eigvals[:k], Meps)                 # (k,)
    subtract = captured[:, :k] @ s2                       # (N,)
    score = np.asarray(total, dtype=np.float64) - subtract
    return np.clip(score, 0.0, None)                      # nonnegative (numerical guard)
