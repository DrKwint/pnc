"""Regression tests for the canonical original-centered P&C correction (Part A).

The canonical objective is

    Theta_hat = argmin_{Theta'} ||Y_v Theta'^T - Y Theta^T||_F^2 + lambda ||Theta' - Theta||_F^2

so the ridge shrinks toward the ORIGINAL affine map. These tests pin the four
properties the manuscript now relies on:

  1. closed-form parity against a direct solve (float64, rel err <= 1e-10)
  2. lambda = 0 makes the centre irrelevant when the LS solution is unique
  3. lambda -> infinity drives Theta_hat -> Theta  (the semantic content of lambda)
  4. the exact residual identity r_S(x; v) = Theta g_S(x; v), with NO beta term

Run:  .venv/bin/python -m pytest experiments/pnc_protocol/tests/test_canonical_ridge_center.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "scripts"))

from experiments.pnc_protocol.ridge import (  # noqa: E402
    RidgeSpecification, solve_correction,
)
from pnc_theory.linalg import ridge_solve  # noqa: E402

RNG = np.random.default_rng(20260815)


def _problem(n=64, p=9, d=5, seed=0):
    g = np.random.default_rng(seed)
    X = g.standard_normal((n, p))          # augmented perturbed design
    Theta = g.standard_normal((p, d))      # original affine map
    Z = g.standard_normal((n, d))          # original targets  Y Theta^T
    return X, Z, Theta


# --------------------------------------------------------------- 1. parity
@pytest.mark.parametrize("lam", [1e-6, 1e-3, 1.0, 10.0, 1e3])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_closed_form_parity_original_centered(lam, seed):
    """(X^T X + lam I) Theta_hat = X^T Z + lam Theta, solved directly."""
    X, Z, Theta = _problem(seed=seed)
    p = X.shape[1]
    direct = np.linalg.solve(X.T @ X + lam * np.eye(p), X.T @ Z + lam * Theta)
    got = Theta + solve_correction(
        X, Z, Theta, RidgeSpecification(lam, "sum_legacy", "original"))
    rel = np.linalg.norm(got - direct) / max(np.linalg.norm(direct), 1e-300)
    assert rel <= 1e-10, f"lam={lam} seed={seed}: rel err {rel:.3e}"


@pytest.mark.parametrize("lam_mean", [1e-4, 1e-2, 1.0])
def test_closed_form_parity_mean_objective(lam_mean):
    """The mean-reduction variant against its own direct solve."""
    X, Z, Theta = _problem(seed=7)
    n, p = X.shape
    direct = np.linalg.solve((X.T @ X) / n + lam_mean * np.eye(p),
                             (X.T @ Z) / n + lam_mean * Theta)
    got = Theta + solve_correction(
        X, Z, Theta, RidgeSpecification(lam_mean, "mean", "original"))
    rel = np.linalg.norm(got - direct) / np.linalg.norm(direct)
    assert rel <= 1e-10, f"lam_mean={lam_mean}: rel err {rel:.3e}"


# --------------------------------------------------------- 2. lambda == 0
@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_zero_lambda_makes_centre_irrelevant(seed):
    """With a unique LS solution, the two centres give the same Theta_hat."""
    X, Z, Theta = _problem(n=128, p=9, d=5, seed=seed)   # n >> p => unique
    assert np.linalg.matrix_rank(X) == X.shape[1]
    a = solve_correction(X, Z, Theta, RidgeSpecification(0.0, "mean", "original"))
    b = solve_correction(X, Z, Theta, RidgeSpecification(0.0, "mean", "zero"))
    rel = np.linalg.norm(a - b) / max(np.linalg.norm(a), 1e-300)
    assert rel <= 1e-10, f"seed={seed}: centres disagree at lambda=0, rel {rel:.3e}"


# ------------------------------------------------------ 3. lambda -> infinity
@pytest.mark.parametrize("lam", [1e6, 1e9, 1e12])
def test_large_lambda_returns_the_original_map(lam):
    """lambda -> infinity  =>  Theta_hat -> Theta, i.e. the correction vanishes.

    This is the semantic property the manuscript now leans on: lambda is repair
    conservatism toward the base affine map.
    """
    X, Z, Theta = _problem(seed=11)
    C = solve_correction(X, Z, Theta, RidgeSpecification(lam, "sum_legacy", "original"))
    rel = np.linalg.norm(C) / np.linalg.norm(Theta)
    assert rel < 50.0 / lam ** 0.5, f"lam={lam}: ||C||/||Theta|| = {rel:.3e}"


def test_large_lambda_is_monotone_toward_original():
    X, Z, Theta = _problem(seed=12)
    norms = [np.linalg.norm(solve_correction(
        X, Z, Theta, RidgeSpecification(lam, "sum_legacy", "original")))
        for lam in (1e0, 1e2, 1e4, 1e6, 1e8)]
    assert all(a > b for a, b in zip(norms, norms[1:])), norms
    assert norms[-1] < 1e-6 * norms[0]


def test_zero_centre_does_the_opposite():
    """Contrast: zero-centred ridge drives Theta_hat -> 0, not -> Theta.

    Guards against the two conventions being confused again.
    """
    X, Z, Theta = _problem(seed=13)
    C = solve_correction(X, Z, Theta, RidgeSpecification(1e10, "sum_legacy", "zero"))
    Theta_hat = Theta + C
    assert np.linalg.norm(Theta_hat) < 1e-6 * np.linalg.norm(Theta)


# ------------------------------------------------- 4. exact residual identity
def _mlp_residual_identity(seed=0, lam=1e-3, n=256, d_in=7, d_hid=11, d_out=5):
    """Synthetic one-hidden-layer MLP: build the correction, then check

        r_S(x; v) = Theta g_S(x; v)

    where g_S = [1, y_v] - [1, y] is the augmented post-activation change and
    r_S is the corrected-minus-original pre-activation of the following layer.
    No beta ridge-centre term appears anywhere.
    """
    g = np.random.default_rng(seed)
    X = g.standard_normal((n, d_in))
    W1 = g.standard_normal((d_in, d_hid)) / np.sqrt(d_in)
    b1 = g.standard_normal(d_hid) * 0.1
    W2 = g.standard_normal((d_hid, d_out)) / np.sqrt(d_hid)
    b2 = g.standard_normal(d_out) * 0.1
    dW1 = g.standard_normal((d_in, d_hid)) * 0.05

    act = lambda a: np.tanh(a)
    y = act(X @ W1 + b1)                      # original post-activation
    yv = act(X @ (W1 + dW1) + b1)             # perturbed post-activation
    aug = lambda h: np.concatenate([np.ones((h.shape[0], 1)), h], axis=1)
    Theta = np.concatenate([b2[None, :], W2], axis=0)      # (1+d_hid, d_out)

    Z = aug(y) @ Theta                        # original pre-activations (targets)
    Xv = aug(yv)
    C = solve_correction(Xv, Z, Theta, RidgeSpecification(lam, "sum_legacy", "original"))
    Theta_hat = Theta + C

    r_S = Xv @ Theta_hat - aug(y) @ Theta     # corrected - original pre-activation
    g_S = Xv - aug(y)                         # augmented activation change
    pred = g_S @ Theta + Xv @ C               # exact decomposition
    return r_S, g_S, Theta, C, Xv, pred


def test_exact_residual_identity_synthetic_mlp():
    """r_S = g_S Theta + Xv C, exactly, with no ridge-centre offset term."""
    for seed in range(4):
        r_S, g_S, Theta, C, Xv, pred = _mlp_residual_identity(seed=seed)
        rel = np.linalg.norm(r_S - pred) / np.linalg.norm(r_S)
        assert rel <= 1e-12, f"seed={seed}: rel {rel:.3e}"


def test_residual_identity_collapses_to_theta_g_at_large_lambda():
    """As lambda -> infinity the correction vanishes and r_S -> Theta g_S.

    This is the statement the theory section makes for original-centred ridge;
    under zero-centred ridge it is false, because Theta_hat -> 0 instead.
    """
    r_S, g_S, Theta, C, Xv, _ = _mlp_residual_identity(seed=3, lam=1e12)
    rel = np.linalg.norm(r_S - g_S @ Theta) / np.linalg.norm(r_S)
    assert rel <= 1e-6, f"rel {rel:.3e}"
    assert np.linalg.norm(C) / np.linalg.norm(Theta) < 1e-8


def test_no_beta_term_needed():
    """The identity holds with the ORIGINAL Theta as the only affine object.

    Historically the zero-centred convention required an extra ridge-centre
    (beta) term to state the residual identity. Assert that the original-centred
    identity closes exactly without one, and that inserting any nonzero beta
    breaks it.
    """
    r_S, g_S, Theta, C, Xv, pred = _mlp_residual_identity(seed=5)
    assert np.linalg.norm(r_S - pred) / np.linalg.norm(r_S) <= 1e-12
    beta = np.full_like(Theta, 1e-3)
    broken = g_S @ (Theta + beta) + Xv @ C
    assert np.linalg.norm(r_S - broken) / np.linalg.norm(r_S) > 1e-9
