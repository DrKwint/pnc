"""Numerically verify what ridge centre each PRODUCTION P&C solver implements.

The audit must not infer the centre from prose or even from reading code. This
script calls each experiment's actual solver on a small synthetic problem and
compares the result against the two closed forms

    original-centred:  (X^T X + lam I) Theta_hat = X^T Z + lam Theta
    zero-centred:      (X^T X + lam I) Theta_hat = X^T Z

reporting the relative error against each. Exactly one should be ~0.

Backends are selected with --backends because they live in different virtualenvs:
`jax_ensemble` and `cifar_conv` need .venv (JAX); `vit` needs .venv_vit (torch);
`numpy_ridge` (DistilBERT's solver) runs anywhere.

    .venv/bin/python -m experiments.pnc_protocol.verify_ridge_center \
        --backends numpy_ridge,jax_ensemble,cifar_conv
    .venv_vit/bin/python -m experiments.pnc_protocol.verify_ridge_center --backends vit
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "scripts"))

LAM = 0.37                      # nothing special; large enough that the centres differ
SEED = 20260815


def _closed_forms(X, Z, Theta, lam):
    p = X.shape[1]
    G = X.T @ X + lam * np.eye(p)
    return (np.linalg.solve(G, X.T @ Z + lam * Theta),   # original-centred
            np.linalg.solve(G, X.T @ Z))                 # zero-centred


def _verdict(got, orig, zero, tol=1e-5):
    """Classify which closed form the solver reproduces.

    `tol` is loose because two production solvers (JAX PJSVD, CIFAR conv) run in
    float32; what identifies the centre is not the absolute residual but the
    ratio between the two, which is ~5 orders of magnitude in every case here.
    """
    r_o = np.linalg.norm(got - orig) / max(np.linalg.norm(orig), 1e-300)
    r_z = np.linalg.norm(got - zero) / max(np.linalg.norm(zero), 1e-300)
    if r_o < tol and r_o < r_z / 100.0:
        centre = "original"
    elif r_z < tol and r_z < r_o / 100.0:
        centre = "zero"
    else:
        centre = "NEITHER"
    return {"rel_err_vs_original": float(r_o), "rel_err_vs_zero": float(r_z),
            "ratio_zero_over_original": float(r_z / max(r_o, 1e-300)),
            "implements_centre": centre}


def _toy(n=48, p=7, d=4, seed=SEED):
    g = np.random.default_rng(seed)
    return (g.standard_normal((n, p)), g.standard_normal((n, d)),
            g.standard_normal((p, d)))


# ------------------------------------------------------------------ backends
def b_numpy_ridge():
    """pnc_theory.linalg.ridge_solve — the DistilBERT / Banking77 solver."""
    from pnc_theory.linalg import ridge_solve
    X, Z, Theta = _toy()
    orig, zero = _closed_forms(X, Z, Theta, LAM)
    out = {}
    out["as_called_by_distilbert(w_prior=Theta)"] = _verdict(
        ridge_solve(X, Z, LAM, w_prior=Theta), orig, zero)
    out["library_default(w_prior=None)"] = _verdict(
        ridge_solve(X, Z, LAM), orig, zero)
    return out


def b_jax_ensemble():
    """pnc_core.ensembles._ls_or_ridge_solve — the MuJoCo / PJSVD solver."""
    import jax.numpy as jnp
    from pnc_core.ensembles import PJSVDEnsemble, _ls_or_ridge_solve
    X, Z, Theta = _toy()
    orig, zero = _closed_forms(X, Z, Theta, LAM)
    out = {}
    out["ridge_toward_orig=True"] = _verdict(
        np.asarray(_ls_or_ridge_solve(jnp.asarray(X), jnp.asarray(Z), LAM,
                                      jnp.asarray(Theta)), np.float64), orig, zero)
    out["ridge_toward_orig=False"] = _verdict(
        np.asarray(_ls_or_ridge_solve(jnp.asarray(X), jnp.asarray(Z), LAM, None),
                   np.float64), orig, zero)
    # the constructor default is what new runs get
    import inspect
    sig = inspect.signature(PJSVDEnsemble.__init__)
    out["constructor_default_ridge_toward_orig"] = bool(
        sig.parameters["ridge_toward_orig"].default)
    return out


def b_cifar_conv():
    """pnc_core.pnc._ridge_regression_solve — the CIFAR conv-block solver.

    CIFAR parameterises the *delta*: it accumulates H = M^T M and
    b = M^T (T - Y w2_orig), then solves (H + lam I) Delta = b and returns
    w2_orig + Delta. Penalising ||Delta||^2 IS penalising ||Theta_hat - Theta||^2,
    so this should come out original-centred. conv2 has use_bias=False, so the
    original bias is 0 and the implicit bias centre is exact.
    """
    import jax.numpy as jnp
    from pnc_core.pnc import _ridge_regression_solve
    g = np.random.default_rng(SEED + 1)
    n, feat, cout = 60, 9, 5
    Y = g.standard_normal((n, feat))              # perturbed post-activation patches
    w2 = g.standard_normal((feat, cout))          # original conv2 (flattened patches)
    T = g.standard_normal((n, cout))              # original targets
    M = np.concatenate([np.ones((n, 1)), Y], 1)   # bias first, as the code builds it
    H = M.T @ M
    b = M.T @ (T - Y @ w2)                        # residual vs ORIGINAL map
    # the production call, with w2 shaped as a trivial 1x1 conv kernel
    w2_k = jnp.asarray(w2.reshape(1, 1, feat, cout))
    w2_new, b2_new = _ridge_regression_solve(jnp.asarray(H), jnp.asarray(b), LAM, w2_k)
    got = np.concatenate([np.asarray(b2_new, np.float64),
                          np.asarray(w2_new, np.float64).reshape(feat, cout)], 0)
    Theta = np.concatenate([np.zeros((1, cout)), w2], 0)   # original bias is 0
    orig, zero = _closed_forms(M, T, Theta, LAM)
    return {"solve_chunked_conv2_correction path": _verdict(got, orig, zero),
            "note": "conv2 use_bias=False, so Theta = [0; w2_orig] exactly"}


def b_vit():
    """experiments.imagenet_vit_pnc.pnc_core.cho_solve_shared — the ViT solver."""
    from experiments.imagenet_vit_pnc import pnc_core as pc
    X, Z, Theta = _toy()
    orig, zero = _closed_forms(X, Z, Theta, LAM)
    G, C = X.T @ X, X.T @ Z
    out = {}
    got, _, _ = pc.cho_solve_shared(G, C, LAM, w_prior=Theta)
    out["as_called_by_vit(w_prior=Theta0)"] = _verdict(np.asarray(got), orig, zero)
    got0, _, _ = pc.cho_solve_shared(G, C, LAM, w_prior=None)
    out["library_default(w_prior=None)"] = _verdict(np.asarray(got0), orig, zero)
    return out


BACKENDS = {"numpy_ridge": b_numpy_ridge, "jax_ensemble": b_jax_ensemble,
            "cifar_conv": b_cifar_conv, "vit": b_vit}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backends", default=",".join(BACKENDS))
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    res = {}
    for name in a.backends.split(","):
        name = name.strip()
        if not name:
            continue
        try:
            res[name] = BACKENDS[name]()
            print(f"\n=== {name} ===")
            print(json.dumps(res[name], indent=2))
        except Exception as e:                      # a missing venv must not hide the rest
            res[name] = {"error": f"{type(e).__name__}: {e}"}
            print(f"\n=== {name} === SKIPPED: {type(e).__name__}: {e}")
    if a.out:
        p = Path(a.out)
        p.parent.mkdir(parents=True, exist_ok=True)
        prev = json.loads(p.read_text()) if p.exists() else {}
        prev.update(res)
        p.write_text(json.dumps(prev, indent=2))
        print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
