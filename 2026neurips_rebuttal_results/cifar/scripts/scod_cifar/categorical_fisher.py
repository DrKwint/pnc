"""Categorical-logit Fisher for SCOD (JAX/Flax adaptation).

SCOD (Sketching Curvature for OOD Detection; Sharma et al.) requires the square-root
action of the per-example Fisher w.r.t. the model outputs. For a categorical distribution
parameterised by logits z in R^C with p = softmax(z), the Fisher w.r.t. the logits is

    F_z = diag(p) - p p^T.

Rather than materialise F_z, SCOD differentiates a Fisher-weighted output whose parameter
Jacobian G satisfies  G^T G = J_z^T F_z J_z. With p DETACHED from autograd,

    zbar_c = sum_c p_c z_c          (a scalar per example)
    ztilde_c = sqrt(p_c) (z_c - zbar)

has, w.r.t. z (p held constant),   d ztilde_c / d z_j = sqrt(p_c) (delta_cj - p_j),
so  (J_ztilde^T J_ztilde)_{ij} = sum_c p_c (delta_ci - p_i)(delta_cj - p_j)
                                = p_i delta_ij - p_i p_j = (diag(p) - p p^T)_{ij}.  QED.

This is the JAX analogue of the official (PyTorch) implementation's detached-probability
transform; `stop_gradient` plays the role of `.detach()`.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp


def fisher_weighted_logits(logits: jnp.ndarray, eps: float = 1e-12) -> jnp.ndarray:
    """Fisher square-root transform of logits (last axis = classes).

    Returns ztilde with the same shape as `logits` such that the parameter Jacobian G of
    ztilde satisfies G^T G = J_z^T (diag(p) - p p^T) J_z. Probabilities are stop-gradiented
    so the transform is the exact Fisher factor (not softmax curvature of p itself).
    """
    probs = jax.nn.softmax(jax.lax.stop_gradient(logits), axis=-1)
    mean_logit = jnp.sum(probs * logits, axis=-1, keepdims=True)
    return jnp.sqrt(jnp.clip(probs, a_min=eps)) * (logits - mean_logit)


def categorical_fisher_matrix(logits: jnp.ndarray) -> jnp.ndarray:
    """Exact F_z = diag(p) - p p^T for a single logit vector (reference / testing)."""
    probs = jax.nn.softmax(logits, axis=-1)
    return jnp.diag(probs) - jnp.outer(probs, probs)


# ---------------------------------------------------------------------------
# Section 6.1 — Fisher factor test:  || J_ztilde,z a ||_2^2 ~= a^T (diag(p) - p p^T) a
# ---------------------------------------------------------------------------
def _jvp_ztilde(logits: jnp.ndarray, a: jnp.ndarray) -> jnp.ndarray:
    """Directional derivative J_ztilde,z @ a via forward-mode AD (p detached inside)."""
    _, jvp = jax.jvp(fisher_weighted_logits, (logits,), (a,))
    return jvp


def fisher_factor_test(dtype=jnp.float64, n_trials: int = 20, C: int = 10, seed: int = 0):
    """Numerically verify the Fisher square-root identity. Returns (max_rel_err, tol, ok)."""
    from jax import config as _cfg
    if dtype == jnp.float64:
        _cfg.update("jax_enable_x64", True)
    key = jax.random.PRNGKey(seed)
    tol = 1e-5 if dtype == jnp.float64 else 1e-4
    max_rel = 0.0
    for _ in range(n_trials):
        key, k1, k2 = jax.random.split(key, 3)
        z = (jax.random.normal(k1, (C,)) * 3.0).astype(dtype)
        a = jax.random.normal(k2, (C,)).astype(dtype)
        lhs = float(jnp.sum(_jvp_ztilde(z, a) ** 2))          # ||J a||^2
        F = categorical_fisher_matrix(z)
        rhs = float(a @ F @ a)                                 # a^T F a
        denom = max(abs(rhs), 1e-30)
        max_rel = max(max_rel, abs(lhs - rhs) / denom)
    return max_rel, tol, (max_rel < tol)


if __name__ == "__main__":
    for dt, name in [(jnp.float64, "float64"), (jnp.float32, "float32")]:
        err, tol, ok = fisher_factor_test(dtype=dt)
        print(f"[fisher_factor_test/{name}] max_rel_err={err:.2e} tol={tol:.0e} -> {'PASS' if ok else 'FAIL'}")
