"""SCOD Fisher validation tests (Section 6).

6.1 Fisher factor test         -- ||J_ztilde a||^2 == a^T (diag(p)-p p^T) a
6.2 Tiny-network reference test -- sketch+scorer vs an EXACT empirical-Fisher eigendecomposition
6.3 Model immutability          -- verified in run_build_sketch (weights unchanged after sketching)

Run:  python -m experiments.scod_cifar.tests
Exit code 0 iff all tests pass.
"""
from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

from experiments.scod_cifar.categorical_fisher import fisher_factor_test, fisher_weighted_logits
from experiments.scod_cifar.sketch import build_fisher_sketch
from experiments.scod_cifar.scorer import compute_score_features, score_from_features


# ----------------------------- tiny classifier -----------------------------
def _tiny_params(key, d_in, d_hid, C):
    k1, k2, k3, k4 = jax.random.split(key, 4)
    return dict(
        W1=jax.random.normal(k1, (d_hid, d_in)) * 0.5,
        b1=jax.random.normal(k2, (d_hid,)) * 0.1,
        W2=jax.random.normal(k3, (C, d_hid)) * 0.5,
        b2=jax.random.normal(k4, (C,)) * 0.1,
    )


def _tiny_logits(p, x):
    h = jnp.tanh(p["W1"] @ x + p["b1"])
    return p["W2"] @ h + p["b2"]


def tiny_network_reference_test(seed=0, d_in=12, d_hid=10, C=4, N=6, n_test=60, verbose=True):
    """Compare sketched SCOD scores against an exact empirical-Fisher eigendecomposition.

    P = tiny (<1000). Numerical rank of A = (1/N) sum_i L_i^T L_i is <= N*C. We set the sketch
    rank k = numerical rank and T = 6k+4, then require score correlation > 0.99 (Section 6.2).
    """
    key = jax.random.PRNGKey(seed)
    kp, kx, kt = jax.random.split(key, 3)
    params = _tiny_params(kp, d_in, d_hid, C)
    flat_w, unravel = ravel_pytree(params)
    P = int(flat_w.shape[0])
    X = jax.random.normal(kx, (N, d_in))          # calibration
    Xtest = jax.random.normal(kt, (n_test, d_in)) * 1.3

    def ztilde(fw, x1):
        return fisher_weighted_logits(_tiny_logits(unravel(fw), x1))

    # ---- exact empirical Fisher A = (1/N) sum_i L_i^T L_i ----
    jac = jax.jit(jax.jacrev(ztilde))
    Ls = np.stack([np.asarray(jac(flat_w, X[i])) for i in range(N)])   # (N, C, P)
    A = np.einsum("ncp,ncq->pq", Ls, Ls) / N                            # (P, P)
    evals, evecs = np.linalg.eigh(A)
    evals = evals[::-1]; evecs = evecs[:, ::-1]
    tol = evals[0] * P * np.finfo(np.float64).eps
    num_rank = int(np.sum(evals > tol))
    k = num_rank
    T = 6 * k + 4

    # ---- sketched recovery ----
    res = build_fisher_sketch(ztilde, flat_w, np.asarray(X), C=C, T=T, k=k,
                              sketch_seed=123, block_cols=max(4, T), log_every=0)
    # eigenvalue agreement (top-k)
    lam_exact = evals[:k]
    lam_rel = np.max(np.abs(res.eigvals - lam_exact) / np.maximum(lam_exact, 1e-12))

    # ---- score agreement (posterior_pred residual, Meps=5000) ----
    Meps = 5000.0
    U_exact = evecs[:, :k]
    tot_e, cap_e = compute_score_features(ztilde, flat_w, np.asarray(Xtest), U_exact, log_every=0)
    tot_s, cap_s = compute_score_features(ztilde, flat_w, np.asarray(Xtest), res.basis, log_every=0)
    sc_exact = score_from_features(tot_e, cap_e, lam_exact, k, Meps)
    sc_sketch = score_from_features(tot_s, cap_s, res.eigvals, k, Meps)
    corr = float(np.corrcoef(sc_exact, sc_sketch)[0, 1])
    nonneg = bool(np.all(sc_sketch >= -1e-9) and np.all(sc_exact >= -1e-9))

    ok = (corr > 0.99) and nonneg and (lam_rel < 1e-3)
    if verbose:
        print(f"[tiny_ref] P={P} num_rank={num_rank} k={k} T={T}")
        print(f"[tiny_ref] eigenvalue top-k max rel err = {lam_rel:.2e} (tol 1e-3)")
        print(f"[tiny_ref] score correlation = {corr:.6f} (need > 0.99)")
        print(f"[tiny_ref] scores nonnegative = {nonneg}")
        print(f"[tiny_ref] -> {'PASS' if ok else 'FAIL'}")
    return dict(P=P, num_rank=num_rank, k=k, T=T, eig_rel_err=lam_rel, score_corr=corr,
                nonneg=nonneg, ok=ok)


def run_all():
    results = {}
    print("=== 6.1 Fisher factor test ===")
    for dt, name in [(jnp.float64, "float64"), (jnp.float32, "float32")]:
        err, tol, ok = fisher_factor_test(dtype=dt)
        results[f"fisher_factor_{name}"] = ok
        print(f"  {name}: max_rel_err={err:.2e} tol={tol:.0e} -> {'PASS' if ok else 'FAIL'}")
    print("=== 6.2 Tiny-network reference test ===")
    r = tiny_network_reference_test()
    results["tiny_reference"] = r["ok"]
    allok = all(results.values())
    print(f"\n=== ALL SCOD FISHER TESTS: {'PASS' if allok else 'FAIL'} ===")
    return allok, results


if __name__ == "__main__":
    import sys
    ok, _ = run_all()
    sys.exit(0 if ok else 1)
