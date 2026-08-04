"""SCOD sketch + score (JAX port).

Builds a randomized low-rank sketch of the dataset Fisher

    F = sum_n J~_n^T J~_n          (P x P, never materialised)

where J~_n is the Fisher-weighted output Jacobian of example n (see
``scod_distribution``).  We only ever need F's action on a vector, which is a
JVP followed by a VJP over the whole calibration set:

    F omega = J~^T ( J~ omega )         (T = N * out_dim intermediate)

so one matvec costs one forward-mode + one reverse-mode pass over the data — no
(out_dim x P) Jacobian is formed.  The top eigensystem is recovered with the
single-pass Nystrom sketch of Tropp et al. (2017), which is exactly SCOD's
"random sketch" with ``num_samples = 6*num_eigs + 4``.

At test time the ``posterior_pred`` score is the square root of the residual
Fisher energy of J~(x) after eigenvalue-dependent shrinkage controlled by Meps:

    u(x)^2 = Meps * [ ||J~(x)||_F^2  -  sum_i  s_i(Meps) * ||J~(x) u_i||^2 ]
    s_i    = lambda_i / (lambda_i + 1/Meps)   in [0,1]

s_i -> 1 for well-constrained (large-lambda) directions, so their energy is
"explained" and removed; energy left in weakly-constrained / null directions
raises the score (higher = more atypical / OOD).  The global Meps prefactor is
irrelevant to AUROC/Spearman (rank) and cancels in the ID-median normalisation of
the Gaussian variant, but is kept for fidelity to the posterior form (Meps = 1/eps,
eps = prior precision).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree
from scipy.linalg import solve_triangular

from scod_distribution import make_param_fn, fisher_weighted_output


class SCODModel:
    """Pure-function view giving the stacked Fisher-weighted output t(theta) and its
    JVP/VJP, for a batch of inputs held fixed at construction of each closure."""

    def __init__(self, model, fixed_sigma2=None):
        self.f, self.params, self.P = make_param_fn(model)
        self.theta0, self.unravel = ravel_pytree(self.params)
        self.theta0 = jnp.asarray(self.theta0)
        self.fixed_sigma2 = fixed_sigma2

    def t_flat(self, theta_vec, X):
        """Stacked transformed outputs, shape (N*out,), out = k (Case A) or 2k (Case B)."""
        params = self.unravel(theta_vec)
        mean, var = self.f(params, X)
        t = fisher_weighted_output(mean, var, fixed_sigma2=self.fixed_sigma2)
        return t.reshape(-1)

    # --- per-test-point Fisher-weighted Jacobian (small batches only) ---------
    def jac(self, X):
        """J~ for a batch, shape (N, out, P).  Materialises the Jacobian — only for
        test scoring in small batches (out*P per point)."""
        def t_one(theta, x):
            params = self.unravel(theta)
            mean, var = self.f(params, x[None, :])
            return fisher_weighted_output(mean[0], var[0], fixed_sigma2=self.fixed_sigma2)
        jac_one = jax.jacrev(lambda th, x: t_one(th, x))
        leaves = jax.vmap(lambda x: jac_one(self.theta0, x))(X)  # (N, out, P) already flat
        return leaves


# ---------------------------------------------------------------------------
# exact Fisher (validation only — tiny models)
# ---------------------------------------------------------------------------
def exact_fisher(scod: SCODModel, X):
    """F = J^T J with J = d t_flat/d theta (T x P). For small models/tests only."""
    J = jax.jacrev(lambda th: scod.t_flat(th, X))(scod.theta0)  # (T, P)
    return np.asarray(J).T @ np.asarray(J), np.asarray(J)


# ---------------------------------------------------------------------------
# randomized Nystrom sketch of F
# ---------------------------------------------------------------------------
def build_sketch(scod: SCODModel, X, *, num_eigs_max=100, num_samples=None,
                 sketch_seed=0, col_chunk=16):
    """Return (eigs, basis): eigs (m,) descending, basis (P, m), m<=num_eigs_max.

    Single-pass Nystrom (Tropp 2017) on the PSD dataset Fisher F.
    """
    P = scod.P
    m = min(num_eigs_max, P)
    ell = int(num_samples if num_samples is not None else 6 * num_eigs_max + 4)
    ell = min(ell, P)
    rng = np.random.RandomState(sketch_seed)
    Omega = jnp.asarray(rng.randn(P, ell).astype(np.float32))

    theta = scod.theta0

    def tfun(th):
        return scod.t_flat(th, X)

    def F_matvec(omega):
        _, jvp_out = jax.jvp(tfun, (theta,), (omega,))   # J omega  (T,)
        _, vjp = jax.vjp(tfun, theta)
        return vjp(jvp_out)[0]                            # J^T J omega  (P,)

    # Y = F Omega, columns processed in chunks to bound memory
    cols = []
    for s in range(0, ell, col_chunk):
        block = Omega[:, s:s + col_chunk]
        cols.append(jax.vmap(F_matvec, in_axes=1, out_axes=1)(block))
    Y = jnp.concatenate(cols, axis=1)                    # (P, ell)

    # single-pass Nystrom eigendecomposition (float64 for the small ell x ell solve)
    Y64 = np.asarray(Y, np.float64)
    Om64 = np.asarray(Omega, np.float64)
    nu = np.sqrt(P) * np.finfo(np.float64).eps * np.linalg.norm(Y64)
    Yv = Y64 + nu * Om64
    B = Om64.T @ Yv                                       # (ell, ell) symmetric PSD
    B = 0.5 * (B + B.T)
    # escalate jitter until the (ell x ell) Nystrom core factorizes; ill-conditioned
    # Grams (e.g. high-out_dim envs) can need more than a single 1e-10 shift.
    scale = np.trace(B) / B.shape[0]
    C = None
    for j in (0.0, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2):
        try:
            C = np.linalg.cholesky(B + j * scale * np.eye(B.shape[0]))
            break
        except np.linalg.LinAlgError:
            continue
    if C is None:
        raise np.linalg.LinAlgError("Nystrom core not PD even at 1e-2 relative jitter")
    E = solve_triangular(C, Yv.T, lower=True).T     # E = Yv C^{-T}, stable triangular solve
    U, S, _ = np.linalg.svd(E, full_matrices=False)
    eigs = np.maximum(0.0, S ** 2 - nu)
    order = np.argsort(eigs)[::-1][:m]
    return np.asarray(eigs[order]), np.asarray(U[:, order])


# ---------------------------------------------------------------------------
# posterior_pred score
# ---------------------------------------------------------------------------
def scod_energy(scod: SCODModel, X, basis, *, point_chunk=64):
    """Per-point Fisher energy, computed ONCE and shared across all (k, Meps).

    Returns (fro2, pe):
      fro2 : (N,)      ||J~(x)||_F^2
      pe   : (N, m)    per-eigendirection energy  sum_out (J~(x) u_i)^2
    so that for any rank k and shrinkage s_i,
      u(x)^2 = Meps * ( fro2 - sum_{i<k} s_i * pe[:, i] ).
    """
    U = jnp.asarray(np.asarray(basis))                   # (P, m)
    fro2s, pes = [], []
    N = X.shape[0]
    for i in range(0, N, point_chunk):
        Jb = scod.jac(X[i:i + point_chunk])              # (b, out, P)
        fro2s.append(jnp.sum(Jb ** 2, axis=(1, 2)))      # (b,)
        proj = jnp.einsum("bop,pm->bom", Jb, U)          # (b, out, m)
        pes.append(jnp.sum(proj ** 2, axis=1))           # (b, m)
    return np.asarray(jnp.concatenate(fro2s)), np.asarray(jnp.concatenate(pes))


def scod_energy_jvp(scod: SCODModel, X, basis, *, n_probes=256, probe_seed=0,
                    point_chunk=1024, dir_chunk=64):
    """Memory-safe Fisher energy via JVPs — never materialises (out x P) Jacobians.

    pe[:, i] = ||J~(x) u_i||^2  is EXACT (JVP of t along each eigenvector u_i).
    fro2     = ||J~(x)||_F^2    is a Hutchinson estimate  E_xi[ ||J~(x) xi||^2 ]
               with xi Rademacher(P), n_probes probes (needed because the exact
               Frobenius norm would otherwise cost `out` reverse passes per point).
    Both are per-point and batchable; scales to out_dim=348 (Humanoid).
    """
    U = np.asarray(basis)                                   # (P, m)
    P, m = U.shape
    rng = np.random.RandomState(probe_seed)
    probes = rng.choice([-1.0, 1.0], size=(n_probes, P)).astype(np.float32)
    dirs = np.concatenate([U.T, probes], axis=0)           # (m+n_probes, P)

    def jvp_along(theta, Xb, V):
        # V: (d, P) directions -> (d, batch, out) tangents of t
        def one(v):
            _, tan = jax.jvp(lambda th: scod.t_flat(th, Xb), (theta,), (v,))
            return tan.reshape(Xb.shape[0], -1)
        return jax.vmap(one)(V)                             # (d, batch, out)

    theta = scod.theta0
    fro2_all, pe_all = [], []
    N = X.shape[0]
    for i in range(0, N, point_chunk):
        Xb = jnp.asarray(X[i:i + point_chunk])
        sq_dir = []                                        # per-direction ||.||^2 -> (d, b)
        for s in range(0, dirs.shape[0], dir_chunk):
            tan = jvp_along(theta, Xb, jnp.asarray(dirs[s:s + dir_chunk]))  # (dc, b, out)
            sq_dir.append(np.asarray(jnp.sum(tan ** 2, axis=2)))            # (dc, b)
        sq = np.concatenate(sq_dir, axis=0)                # (m+n_probes, b)
        pe_all.append(sq[:m].T)                            # (b, m)  exact
        fro2_all.append(sq[m:].mean(axis=0))               # (b,)    Hutchinson mean
    return np.concatenate(fro2_all), np.concatenate(pe_all)


def score_from_energy(fro2, pe, eigs, *, k, Meps):
    """Native SCOD score u(x) (>=0) from cached energy, for a given (k, Meps)."""
    lam = np.asarray(eigs)[:k]
    s = lam / (lam + 1.0 / Meps)                          # shrinkage in [0,1]
    explained = np.asarray(pe)[:, :k] @ s                 # (N,)
    return np.sqrt(np.maximum(Meps * (np.asarray(fro2) - explained), 0.0))


def posterior_pred_scores(scod: SCODModel, X, basis, eigs, *, k, Meps, point_chunk=64):
    """Convenience: native SCOD scores u(x) for one (k, Meps)."""
    fro2, pe = scod_energy(scod, X, basis, point_chunk=point_chunk)
    return score_from_energy(fro2, pe, eigs, k=k, Meps=Meps)


# ---------------------------------------------------------------------------
# validation: sketch eigs vs exact Fisher; score sanity
# ---------------------------------------------------------------------------
def _validate(seed=0):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from flax import nnx
    from pnc_core.models import ProbabilisticRegressionModel

    rng = np.random.RandomState(seed)
    model = ProbabilisticRegressionModel(4, 3, nnx.Rngs(params=seed),
                                         hidden_dims=[8, 6], activation=nnx.tanh)
    scod = SCODModel(model)
    X = jnp.asarray(rng.randn(200, 4).astype(np.float32))

    F, _ = exact_fisher(scod, X)
    w_exact = np.linalg.eigvalsh(F)[::-1]                # descending

    eigs, basis = build_sketch(scod, X, num_eigs_max=min(20, scod.P), sketch_seed=100 + seed)
    kcmp = min(10, len(eigs))
    rel = np.abs(eigs[:kcmp] - w_exact[:kcmp]) / (w_exact[:kcmp] + 1e-12)
    # subspace alignment of the top eigenvector
    wv, Ve = np.linalg.eigh(F)
    top_exact = Ve[:, -1]
    align = abs(float(top_exact @ np.asarray(basis)[:, 0]))
    print(f"[SCOD sketch] P={scod.P}  top-{kcmp} eigval rel-err max={rel.max():.2e}  "
          f"top-eigvec |cos|={align:.6f}")

    u = posterior_pred_scores(scod, X[:32], basis, eigs, k=kcmp, Meps=float(X.shape[0]))
    print(f"[SCOD score]  finite={np.all(np.isfinite(u))}  nonneg={np.all(u >= 0)}  "
          f"range=[{u.min():.3e},{u.max():.3e}]")
    return rel.max(), align, u


if __name__ == "__main__":
    rel, align, u = _validate()
    assert rel < 1e-3, f"sketch eigenvalues off: {rel}"
    assert align > 0.999, f"top eigenvector misaligned: {align}"
    assert np.all(np.isfinite(u)) and np.all(u >= 0), "scores must be finite & nonneg"
    print("PASS: Nystrom sketch recovers the exact Fisher eigensystem; scores valid.")
