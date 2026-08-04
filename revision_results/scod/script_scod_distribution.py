"""SCOD distribution-family adapter (JAX port).

The MuJoCo base model (`models.ProbabilisticRegressionModel`) is a heteroscedastic
diagonal Gaussian: it returns ``(mean, var)`` with ``var = softplus(head) + 1e-6``.
This is SCOD "Case B" (heteroscedastic diagonal Gaussian), but parameterised by the
variance itself rather than log-variance or log-std.

SCOD needs, per input x, the *Fisher-weighted output Jacobian*  J~(x) = L(x)^T J_theta,w
where J_theta,w = d theta / d w are the Jacobians of the output-distribution parameters
theta w.r.t. the network weights w, and L L^T = F_theta is the Fisher of the likelihood
in theta-coordinates.  Then the dataset Fisher is  sum_n J~_n^T J~_n, and SCOD sketches
its top eigenspace.

For a diagonal Gaussian with parameters theta_j = (mu_j, s_j) where s_j = sigma_j^2 is the
*variance* (exactly what the model outputs), the Fisher is diagonal:

    I(mu_j)  = 1 / sigma_j^2
    I(s_j)   = 1 / (2 sigma_j^4)          (s = sigma^2)

so L = diag( 1/sigma_j , 1/(sqrt(2) sigma_j^2) ).  We realise L^T J_theta,w by autodiff of
the *Fisher-weighted output map* (Fisher factors held constant / detached):

    t_mu_j(w) = mu_j(w)      / sigma_j^detach
    t_s_j (w) = sigma_j^2(w) / (sqrt(2) * (sigma_j^2)^detach)

d t / d w evaluated at the current w equals L^T J_theta,w exactly, because the detached
Fisher factors are constants w.r.t. the differentiation.  The softplus that produces var
is captured automatically by autodiff of sigma_j^2(w); no manual Jacobian factor is needed.

(Equivalence to the spec's log-variance form: with rho = log sigma^2, I(rho) = 1/2, giving
t_rho = rho / sqrt(2); the s = sigma^2 coordinate used here has I(s) = 1/(2 sigma^4), i.e.
the same KL quadratic form under the change of variables ds = sigma^2 d rho.  Case A, a
fixed diagonal noise Sigma_a, is the mean-only special case t_mu_j = mu_j / sigma_{a,j}.)

The finite-difference unit test at the bottom confirms  ||J~ dw||^2 ~= 2 KL(p_w || p_{w+dw}).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax.flatten_util import ravel_pytree

SQRT2 = np.sqrt(2.0)


# ---------------------------------------------------------------------------
# pure-function view of the nnx model:  params_vector -> (mean, var) for one x
# ---------------------------------------------------------------------------
def make_param_fn(model):
    """Return (f, params, n_params) where f(params, x) -> (mean, var).

    ``params`` is the nnx ``State`` pytree of all ``nnx.Param`` leaves (the SCOD
    parameter set).  We differentiate w.r.t. this pytree directly (the standard
    flax-nnx functional pattern) rather than a raveled vector, because ravelling
    then re-merging the Variable pytree does not round-trip cleanly.
    """
    graphdef, params = nnx.split(model, nnx.Param)
    theta0, _ = ravel_pytree(params)

    def f(params_state, x):
        m = nnx.merge(graphdef, params_state)
        mean, var = m(x)
        return mean, var

    return f, params, int(theta0.shape[0])


def _param_names(params) -> list[str]:
    """Flat, ordered leaf names matching the ravel order (for provenance)."""
    flat = jax.tree_util.tree_leaves_with_path(params)
    names = []
    for path, leaf in flat:
        key = ".".join(str(getattr(p, "key", getattr(p, "idx", p))) for p in path)
        names.append(f"{key}:{tuple(leaf.shape)}")
    return names


# ---------------------------------------------------------------------------
# Fisher-weighted output map and its Jacobian
# ---------------------------------------------------------------------------
def fisher_weighted_output(mean, var, *, fixed_sigma2=None):
    """Transformed coords t(w) whose d/dw equals L^T J_theta,w.

    mean, var : (..., k) model outputs (var = sigma^2).
    fixed_sigma2 : if given (Case A, fixed diagonal noise), use it in place of the
        model's var for the mean-Fisher factor and drop the variance coordinates.
    Returns t : (..., k)   for Case A
            t : (..., 2k)  for Case B  [mean coords, then variance coords]
    """
    if fixed_sigma2 is not None:
        sigma = jnp.sqrt(fixed_sigma2)
        return mean / jax.lax.stop_gradient(sigma)
    sigma = jnp.sqrt(var)
    t_mu = mean / jax.lax.stop_gradient(sigma)
    t_s = var / (SQRT2 * jax.lax.stop_gradient(var))
    return jnp.concatenate([t_mu, t_s], axis=-1)


def per_example_fisher_jac(f, params, x, *, fixed_sigma2=None):
    """Fisher-weighted Jacobian J~(x) = d t / d w, shape (out_dim, n_params).

    out_dim = k (Case A) or 2k (Case B). x is a single input, shape (in_dim,).
    Differentiates w.r.t. the ``params`` pytree, then ravels each output row into
    a flat parameter vector (consistent ordering across calls).
    """
    def t_of_params(p):
        mean, var = f(p, x[None, :])
        return fisher_weighted_output(mean[0], var[0], fixed_sigma2=fixed_sigma2)

    jac_tree = jax.jacrev(t_of_params)(params)      # pytree; each leaf: (out_dim, *pshape)
    leaves = jax.tree_util.tree_leaves(jac_tree)
    out_dim = leaves[0].shape[0]
    # reshape each leaf (out_dim, *pshape) -> (out_dim, prod(pshape)) in C order and
    # concatenate in tree_leaves order; this matches ravel_pytree(params) column order.
    return jnp.concatenate([lf.reshape(out_dim, -1) for lf in leaves], axis=1)


# ---------------------------------------------------------------------------
# finite-difference validation against the Gaussian KL quadratic form
# ---------------------------------------------------------------------------
def _diag_gaussian_kl(mu1, var1, mu2, var2, eps=1e-12):
    """KL( N(mu1,var1) || N(mu2,var2) ) summed over a diagonal Gaussian.

    Computed in float64: the local quadratic signal (O(dw^2) ~ 1e-9) is otherwise
    lost to catastrophic cancellation between the two O(1e-5) large terms in float32.
    """
    mu1 = np.asarray(mu1, np.float64); var1 = np.asarray(var1, np.float64) + eps
    mu2 = np.asarray(mu2, np.float64); var2 = np.asarray(var2, np.float64) + eps
    return float(np.sum(0.5 * (np.log(var2 / var1)
                               + (var1 + (mu1 - mu2) ** 2) / var2 - 1.0)))


def finite_difference_check(seed: int = 0, n_dirs: int = 8, h: float = 1e-4,
                            case: str = "B", verbose: bool = True):
    """Confirm ||J~ dw||^2 ~= 2 KL(p_w || p_{w+dw}) on a tiny model.

    Returns the worst relative error across n_dirs random directions.
    """
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    jax.config.update("jax_enable_x64", True)   # validation only; real pipeline stays f32
    from pnc_core.models import ProbabilisticRegressionModel

    rng = np.random.RandomState(seed)
    in_dim, out_dim = 3, 2
    # smooth activation for the FD check: ReLU's kinks make forward-difference outputs
    # jump to a different linear piece at finite h, a test artifact unrelated to the
    # (activation-independent) Fisher-factor algebra being validated.
    model = ProbabilisticRegressionModel(in_dim, out_dim, nnx.Rngs(params=seed),
                                         hidden_dims=[5, 4], activation=nnx.tanh)
    f, params, P = make_param_fn(model)
    params = jax.tree.map(lambda a: jnp.asarray(a, jnp.float64), params)  # f64 for FD
    theta0, unravel = ravel_pytree(params)
    x = jnp.asarray(rng.randn(in_dim), jnp.float64)

    fixed = None
    if case == "A":
        # fixed diagonal noise = the model's var at theta0 (frozen)
        _, v0 = f(params, x[None, :])
        fixed = jax.lax.stop_gradient(v0[0])

    Jt = per_example_fisher_jac(f, params, x, fixed_sigma2=fixed)  # (out, P)
    mean0, var0 = f(params, x[None, :])
    mean0, var0 = mean0[0], var0[0]

    Jt64 = np.asarray(Jt, np.float64)
    worst = 0.0
    for _ in range(n_dirs):
        dw = rng.randn(P)
        dw = dw / np.linalg.norm(dw) * h
        quad = float(np.sum((Jt64 @ dw) ** 2))                   # ||J~ dw||^2 (float64)
        params1 = unravel(theta0 + jnp.asarray(dw, theta0.dtype))
        mean1, var1 = f(params1, x[None, :])
        if case == "A":
            kl = _diag_gaussian_kl(mean0, fixed, mean1[0], fixed)
        else:
            kl = _diag_gaussian_kl(mean0, var0, mean1[0], var1[0])
        target = 2.0 * kl
        rel = abs(quad - target) / (abs(target) + 1e-30)
        worst = max(worst, rel)
    if verbose:
        print(f"[SCOD Fisher FD check | case {case}] worst rel err over {n_dirs} dirs "
              f"@ h={h}: {worst:.3e}  (params={P})")
    return worst


if __name__ == "__main__":
    wB = finite_difference_check(case="B")
    wA = finite_difference_check(case="A")
    tol = 1e-3
    assert wB < tol, f"Case B Fisher check failed: {wB}"
    assert wA < tol, f"Case A Fisher check failed: {wA}"
    print(f"PASS: both Fisher-weighted Jacobians reproduce the Gaussian KL quadratic form "
          f"(< {tol}).")
