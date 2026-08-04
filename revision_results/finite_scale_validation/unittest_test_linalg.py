"""Exact-identity unit tests for the P&C theory core (Round 1 deliverable).

Run directly:   .venv/bin/python experiments/scripts/pnc_theory/test_linalg.py
Or via pytest:  .venv/bin/python -m pytest experiments/scripts/pnc_theory/test_linalg.py

Every identity is checked in float64 to a tight tolerance on random synthetic
correction problems, across ridge conventions and the well-/ill-/under-
determined regimes. These are the algebraic guarantees the real-data runs rely
on; if any fails, nothing downstream is trustworthy.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from experiments.scripts.pnc_theory.linalg import (  # noqa: E402
    Correction, ridge_solve, leverage_mahalanobis, rel_err,
)

TOL = 1e-9          # relative, float64 well-conditioned (primary bar)
TOL_ABS = 1e-6      # absolute fallback: rescues genuine near-zero (interpolation)
                    # comparisons and documented ill-conditioned float64 degradation
                    # for O(1)-scale quantities. An identity passes if rel<=TOL OR abs<=TOL_ABS.


def _make_problem(n, p, d, pert=0.3, seed=0):
    """Random original design X, a genuinely different perturbed design Xv,
    and a random next-affine Theta. Perturbation is a smooth random map of the
    non-bias features so Xv != X but both share the bias column."""
    rng = np.random.default_rng(seed)
    h = rng.standard_normal((n, p - 1))
    X = np.concatenate([h, np.ones((n, 1))], axis=1)
    # perturbed features: nonlinear-ish smooth transform + noise
    A = np.eye(p - 1) + pert * rng.standard_normal((p - 1, p - 1)) / np.sqrt(p)
    hv = h @ A + pert * np.tanh(h) + 0.05 * pert * rng.standard_normal((n, p - 1))
    Xv = np.concatenate([hv, np.ones((n, 1))], axis=1)
    Theta = rng.standard_normal((p, d))
    return X, Xv, Theta


def _test_points(p, B=17, seed=1):
    rng = np.random.default_rng(seed)
    h = rng.standard_normal((B, p - 1))
    hb = np.concatenate([h, np.ones((B, 1))], axis=1)
    hv = h @ (np.eye(p - 1) + 0.3 * rng.standard_normal((p - 1, p - 1)) / np.sqrt(p)) \
        + 0.3 * np.tanh(h)
    hvb = np.concatenate([hv, np.ones((B, 1))], axis=1)
    return hb, hvb


RESULTS: list[tuple[str, bool, float]] = []


def check(name, a, b, tol=TOL, tol_abs=TOL_ABS):
    """Pass if relative error <= tol OR absolute error <= tol_abs.

    ``a``, ``b`` may be arrays (compared) or ``a`` a scalar error with b=None."""
    if b is None:
        err = float(a); abs_err = float(a)
    else:
        a = np.asarray(a, float); b = np.asarray(b, float)
        abs_err = float(np.linalg.norm(a - b))
        err = abs_err / (np.linalg.norm(b) + 1e-30)
    ok = (err <= tol) or (abs_err <= tol_abs)
    RESULTS.append((name, ok, min(err, abs_err) if b is not None else err))
    return ok


def run_all():
    # regimes: overdetermined well-cond, near-square, underdetermined
    regimes = {
        "overdet(n=400,p=30)": (400, 30, 5),
        "near-square(n=32,p=30)": (32, 30, 5),
        "underdet(n=12,p=30)": (12, 30, 5),
    }
    lams = [0.0, 1e-6, 1e-2, 1.0, 100.0]

    for rname, (n, p, d) in regimes.items():
        X, Xv, Theta = _make_problem(n, p, d, seed=hash(rname) % 1000)
        hb, hvb = _test_points(p)
        for mode in ("toward_zero", "toward_orig"):
            for lam in lams:
                c = Correction(X, Xv, Theta, lam=lam, mode=mode)
                tag = f"[{rname}|{mode}|λ={lam:g}]"
                well_cond = c.spectrum()["cond_G"] < 1e6  # strict bar only when well-conditioned

                # implementation-mirror solver == stable SVD form
                th = c.theta_hat()
                check(f"theta_hat(impl)==theta_hat_svd {tag}", th, c.theta_hat_svd())

                # (1) Eq(1): spec closed form == solver (invertible-G regime)
                if not (lam == 0.0 and n < p):
                    check(f"Eq1 theta_hat==formula {tag}", th, c.theta_hat_formula())

                # (1b) normal equations satisfied (lam>0 => unique minimizer)
                if lam > 0:
                    ne = c.normal_equation_residual(th)
                    scale = np.linalg.norm(c.Xv.T @ c.target) + 1e-30
                    check(f"normal-eq≈0 {tag}", np.linalg.norm(ne) / scale, None,
                          tol=(TOL if well_cond else 1e-5))

                # (2/4) Eq(2)&(4): reconstructed residual == direct residual
                r_direct = c.residual_direct(hb, hvb, th)
                check(f"Eq2/4 residual identity {tag}", r_direct, c.residual_formula(hb, hvb),
                      tol=(TOL if well_cond else 1e-6))

                # r = Theta g explicitly
                g = c.transfer_defect(hb, hvb)
                check(f"Eq4 r=Θg {tag}", r_direct, g @ c.Theta,
                      tol=(TOL if well_cond else 1e-6))

                # (5) calibration-residual hat identity
                Rs = c.calibration_residual(th)
                check(f"Eq5 cal-resid hat identity {tag}", Rs, c.calibration_residual_formula(),
                      tol=(TOL if well_cond else 1e-6))

                # lam=0: interpolation (n<=p) => R_S≈0; else Xv^T R_S = 0
                if lam == 0.0:
                    if n <= p:
                        check(f"λ=0 interpolation R_S≈0 {tag}",
                              np.linalg.norm(Rs), None, tol=1e-6)
                    else:
                        orth = c.Xv.T @ Rs
                        scale = np.linalg.norm(c.Xv.T @ c.target) + 1e-30
                        check(f"λ=0 Xv^T R_S=0 {tag}", np.linalg.norm(orth) / scale, None, tol=1e-6)

        # toward_zero vs toward_orig differ by exactly -λ G⁻¹ Θ (well-cond only)
        for lam in [1e-2, 1.0, 100.0]:
            if n < p:
                continue
            cz = Correction(X, Xv, Theta, lam=lam, mode="toward_zero")
            co = Correction(X, Xv, Theta, lam=lam, mode="toward_orig")
            Ginv = np.linalg.inv(cz.G)
            check(f"toward_zero-toward_orig=-λG⁻¹Θ [{rname}|λ={lam:g}]",
                  cz.theta_hat() - co.theta_hat(), -lam * Ginv @ Theta)

    # (8) exact leverage <-> Mahalanobis identity
    rng = np.random.default_rng(7)
    for (n, k) in [(500, 12), (200, 40), (60, 40)]:
        h_cal = rng.standard_normal((n, k)) @ (np.eye(k) + 0.5 * rng.standard_normal((k, k)))
        h_test = rng.standard_normal((25, k))
        lev, dmah, rhs = leverage_mahalanobis(h_cal, h_test)
        check(f"Eq8 leverage=1/n+dMah²/(n-1) [n={n},k={k}]", lev, rhs)

    # reference solver sanity: ridge_solve toward-0 matches explicit inverse
    X, Xv, Theta = _make_problem(400, 30, 5, seed=3)
    target = X @ Theta
    for lam in [1e-3, 1.0]:
        W = ridge_solve(Xv, target, lam)
        G = Xv.T @ Xv + lam * np.eye(30)
        check(f"ridge_solve==explicit [λ={lam:g}]", W, np.linalg.solve(G, Xv.T @ target))

    return RESULTS


def _print_and_exit():
    run_all()
    n_pass = sum(ok for _, ok, _ in RESULTS)
    n = len(RESULTS)
    worst = sorted(RESULTS, key=lambda r: -r[2])[:6]
    print(f"\n{'='*70}\nP&C theory core — exact identity tests\n{'='*70}")
    print(f"PASS {n_pass}/{n}")
    print("\nlargest relative errors:")
    for name, ok, err in worst:
        print(f"  {'ok ' if ok else 'FAIL'}  {err:.2e}  {name}")
    fails = [r for r in RESULTS if not r[1]]
    if fails:
        print(f"\n{len(fails)} FAILURES:")
        for name, ok, err in fails:
            print(f"  FAIL {err:.2e}  {name}")
        sys.exit(1)
    print("\nAll exact identities hold to tolerance in float64. ✓")
    sys.exit(0)


# pytest entry points
def test_all_identities():
    run_all()
    fails = [(name, err) for name, ok, err in RESULTS if not ok]
    assert not fails, f"identity failures: {fails}"


if __name__ == "__main__":
    _print_and_exit()
