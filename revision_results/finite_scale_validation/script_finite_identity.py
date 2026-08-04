"""Part 0 — the exact finite transfer-defect identity.

The claim under test.  A single P&C correction interface fits

    ThetaHat = argmin_W  ||Xv W - X0 Theta||^2 + lam ||W - W_prior||^2

on the calibration set, then applies it to an evaluation point whose perturbed
representation is ``Av``.  The brief asserts that the resulting output-visible
change is *exactly* the part of that point's hidden response which the
calibration responses cannot reconstruct:

    Q               = solve(Gv, Av.T)          Gv = Xv.T Xv + Lambda
    Alpha           = Xv @ Q                                   (B, N)
    PredictedDelta  = Alpha.T @ DeltaX                         (N, d+1)
    TransferDefect  = DeltaA - PredictedDelta
    PredictedResidual = TransferDefect @ Theta
    ActualResidual  = Av @ ThetaHat - A0 @ Theta

    ActualResidual == PredictedResidual                        (*)

**(*) is not unconditionally true.**  Working through the algebra with
``Gv = Xv.T Xv + lam I``:

    PredictedResidual - ActualResidual = lam * Av Gv^-1 Theta       (ridge -> 0)
    PredictedResidual - ActualResidual = 0                          (ridge -> Theta)

so the identity as stated holds *exactly* only when the ridge shrinks toward the
original following map (``ridge_toward_orig=True``), or trivially when
``lam == 0``.  For the shipped default ``lam == 0`` there is a second wrinkle:
the implementation calls ``lstsq`` (minimum-norm pseudoinverse) rather than
inverting a Gram matrix, and post-ReLU designs are routinely rank deficient, so
``Gv`` is singular and ``solve(Gv, Av.T)`` is undefined.  The correct general
statement replaces the ridge bias term with its stable SVD form,

    ActualResidual = (DeltaA - Alpha.T DeltaX - (Av - Av V diag(f2) V.T)) @ Theta

with ``Xv = U diag(s) V.T`` and ``f2 = s^2 / (s^2 + lam)`` (``f2 = 1`` on the
row space at ``lam = 0``).  That form is exact in every regime and reduces to
``lam * Av Gv^-1`` when ``Gv`` is invertible.

This module implements *both*: the literal formula from the brief
(:func:`predicted_delta_primal`, :func:`predicted_residual_literal`) and the
regime-correct one (:func:`transfer_defect`, :func:`predicted_residual`), and
:func:`identity_report` reports the error of each so the distinction is visible
rather than hidden behind a passing tolerance.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_EPS = 1e-300


# ──────────────────────────────────────────────────────────────────────────
# elementary pieces
# ──────────────────────────────────────────────────────────────────────────

def regularization_matrix(p: int, lam: float, dtype=np.float64) -> np.ndarray:
    """The exact penalty the implementation applies: uniform ``lam * I``.

    ``ensembles._ls_or_ridge_solve`` builds ``h_aug.T @ h_aug + lambda_reg *
    jnp.eye(D)`` where ``D`` includes the bias column, so the intercept is
    regularized identically to every other coefficient.  Kept as a function so
    an implementation change to a non-uniform ``Lambda`` has one place to land.
    """
    return lam * np.eye(p, dtype=dtype)


def gram(Xv: np.ndarray, lam: float) -> np.ndarray:
    p = Xv.shape[1]
    return Xv.T @ Xv + regularization_matrix(p, lam, Xv.dtype)


def _svd(Xv: np.ndarray):
    U, s, Vt = np.linalg.svd(Xv, full_matrices=False)
    return U, s, Vt


def _filters(s: np.ndarray, lam: float, n: int, p: int, dtype=np.float64):
    """``f1 = s/(s^2+lam)`` (solve gain) and ``f2 = s*f1`` (hat gain).

    At ``lam == 0`` the solve gain becomes the pseudoinverse ``1/s`` truncated
    at the numerical rank, matching ``lstsq``.
    """
    smax = s.max() if s.size else 0.0
    rtol = smax * max(n, p) * np.finfo(dtype).eps
    if lam == 0.0:
        f1 = np.zeros_like(s)
        pos = s > rtol
        f1[pos] = 1.0 / s[pos]
    else:
        f1 = s / (s ** 2 + lam)
    return f1, s * f1, rtol


# ──────────────────────────────────────────────────────────────────────────
# reconstruction weights and predicted response
# ──────────────────────────────────────────────────────────────────────────

def alpha_primal(Xv: np.ndarray, Av: np.ndarray, lam: float) -> np.ndarray:
    """``Alpha = Xv @ solve(Gv, Av.T)`` — the literal primal form, shape (B, N).

    Requires ``Gv`` invertible (``lam > 0``, or ``lam == 0`` with full column
    rank).  Raises ``np.linalg.LinAlgError`` otherwise, deliberately: silently
    falling back would hide the rank deficiency the analysis cares about.
    """
    G = gram(Xv, lam)
    return Xv @ np.linalg.solve(G, Av.T)


def alpha_dual(Xv: np.ndarray, Av: np.ndarray, lam: float) -> np.ndarray:
    """Kernel/dual form ``Alpha.T = (Av Xv.T) (Kv + lam I)^-1``, shape (B, N).

    Uses the push-through identity ``Xv (Xv.T Xv + lam I)^-1 = (Xv Xv.T + lam
    I)^-1 Xv``; exact for ``lam > 0``.  Costs O(B^2) memory, so only use it on
    small calibration subsets.
    """
    Kv = Xv @ Xv.T
    kx = Av @ Xv.T
    W = np.linalg.solve(Kv + regularization_matrix(Kv.shape[0], lam, Kv.dtype), kx.T)
    return W  # (B, N)


def alpha_svd(Xv: np.ndarray, Av: np.ndarray, lam: float, svd=None) -> np.ndarray:
    """``Alpha = U diag(f1) V.T Av.T`` — stable in every regime, shape (B, N)."""
    U, s, Vt = _svd(Xv) if svd is None else svd
    f1, _, _ = _filters(s, lam, *Xv.shape, dtype=Xv.dtype)
    return U @ (f1[:, None] * (Vt @ Av.T))


def alpha_norm_sq(Xv: np.ndarray, Av: np.ndarray, lam: float, svd=None) -> np.ndarray:
    """``||alpha_v(x)||^2`` per evaluation row without forming the (B, N) matrix."""
    U, s, Vt = _svd(Xv) if svd is None else svd
    f1, _, _ = _filters(s, lam, *Xv.shape, dtype=Xv.dtype)
    proj = Av @ Vt.T                                   # (N, r)
    return (proj ** 2) @ (f1 ** 2)


def ridge_leverage(Xv: np.ndarray, Av: np.ndarray, lam: float, svd=None) -> np.ndarray:
    """``h_lam(x) = Av Gv^-1 Av.T`` diagonal, per evaluation row.

    For ``lam > 0`` a component of ``Av`` outside the row space of ``Xv``
    contributes ``||.||^2 / lam``; at ``lam == 0`` only the pseudoinverse
    row-space part is finite and out-of-range mass is dropped.
    """
    U, s, Vt = _svd(Xv) if svd is None else svd
    _, _, rtol = _filters(s, lam, *Xv.shape, dtype=Xv.dtype)
    proj = Av @ Vt.T
    denom = s ** 2 + lam
    if lam == 0.0:
        denom = np.where(s > rtol, s ** 2, np.inf)
    lev = (proj ** 2) @ (1.0 / denom)
    if lam > 0 and Xv.shape[1] > s.shape[0]:
        null_sq = np.sum(Av ** 2, axis=1) - np.sum(proj ** 2, axis=1)
        lev = lev + np.maximum(null_sq, 0.0) / lam
    return lev


def predicted_delta_primal(Xv: np.ndarray, DeltaX: np.ndarray, Av: np.ndarray,
                           lam: float) -> np.ndarray:
    """``PredictedDelta = Alpha.T @ DeltaX`` via the literal primal solve.

    Computed as ``(Av Gv^-1) @ (Xv.T DeltaX)`` — algebraically identical to
    forming the (B, N) ``Alpha`` but O(p^2) instead of O(B N).
    """
    G = gram(Xv, lam)
    M = Xv.T @ DeltaX                                   # (p, p)
    return np.linalg.solve(G, Av.T).T @ M


def predicted_delta_dual(Xv: np.ndarray, DeltaX: np.ndarray, Av: np.ndarray,
                         lam: float) -> np.ndarray:
    return alpha_dual(Xv, Av, lam).T @ DeltaX


def predicted_delta_svd(Xv: np.ndarray, DeltaX: np.ndarray, Av: np.ndarray,
                        lam: float, svd=None) -> np.ndarray:
    U, s, Vt = _svd(Xv) if svd is None else svd
    f1, _, _ = _filters(s, lam, *Xv.shape, dtype=Xv.dtype)
    UtdX = U.T @ DeltaX                                 # (r, p)
    return (Av @ Vt.T) @ (f1[:, None] * UtdX)


# ──────────────────────────────────────────────────────────────────────────
# transfer defect and residuals
# ──────────────────────────────────────────────────────────────────────────

def transfer_defect_literal(DeltaA: np.ndarray, PredictedDelta: np.ndarray) -> np.ndarray:
    return DeltaA - PredictedDelta


def ridge_bias_term(Xv: np.ndarray, Av: np.ndarray, lam: float, svd=None) -> np.ndarray:
    """``Av - Av V diag(f2) V.T`` — the shrinkage/out-of-range term.

    Equals ``lam * Av Gv^-1`` whenever ``Gv`` is invertible; at ``lam == 0`` it
    is the projection of ``Av`` orthogonal to the row space of ``Xv``, which the
    minimum-norm ``lstsq`` solution cannot see.
    """
    U, s, Vt = _svd(Xv) if svd is None else svd
    _, f2, _ = _filters(s, lam, *Xv.shape, dtype=Xv.dtype)
    proj = Av @ Vt.T
    return Av - (proj * f2) @ Vt


def transfer_defect(DeltaA: np.ndarray, PredictedDelta: np.ndarray,
                    Xv: np.ndarray, Av: np.ndarray, lam: float,
                    toward_orig: bool, svd=None) -> np.ndarray:
    """Regime-correct hidden transfer defect ``g_v(x)``, shape (N, d+1)."""
    g = DeltaA - PredictedDelta
    if not (toward_orig and lam > 0):
        g = g - ridge_bias_term(Xv, Av, lam, svd=svd)
    return g


def predicted_residual(TransferDefect: np.ndarray, Theta: np.ndarray) -> np.ndarray:
    return TransferDefect @ Theta


def actual_residual(Av: np.ndarray, ThetaHat: np.ndarray,
                    A0: np.ndarray, Theta: np.ndarray) -> np.ndarray:
    """``Av ThetaHat - A0 Theta`` — the ground-truth corrected-interface change."""
    return Av @ ThetaHat - A0 @ Theta


# ──────────────────────────────────────────────────────────────────────────
# reporting
# ──────────────────────────────────────────────────────────────────────────

def rel_fro(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, np.float64); b = np.asarray(b, np.float64)
    return float(np.linalg.norm(a - b) / (np.linalg.norm(a) + _EPS))


@dataclass
class IdentityReport:
    n_eval: int
    n_cal: int
    p: int
    q: int
    lam: float
    mode: str
    dtype: str
    numerical_rank: int
    cond_Xv: float
    cond_G: float
    # literal (brief-as-written) formula
    literal_max_abs: float
    literal_mean_abs: float
    literal_rel_fro: float
    literal_available: bool
    # regime-correct formula
    exact_max_abs: float
    exact_mean_abs: float
    exact_rel_fro: float
    # cross-checks
    primal_vs_dual: float
    primal_vs_svd: float
    residual_scale: float

    def as_row(self) -> dict:
        return self.__dict__.copy()


def identity_report(X0, Xv, A0, Av, Theta, ThetaHat, lam: float,
                    toward_orig: bool, dtype=np.float64,
                    check_dual: bool = True, dual_max_b: int = 1024) -> IdentityReport:
    """Run every Part-0 numerical check for one member at one dtype."""
    cast = lambda a: np.asarray(a, dtype)                              # noqa: E731
    X0, Xv, A0, Av = map(cast, (X0, Xv, A0, Av))
    Theta, ThetaHat = cast(Theta), cast(ThetaHat)
    DeltaX, DeltaA = Xv - X0, Av - A0
    n, p = Xv.shape
    q = Theta.shape[1]

    svd = _svd(Xv)
    _, s, _ = svd
    _, _, rtol = _filters(s, lam, n, p, dtype=dtype)
    rank = int((s > rtol).sum())
    g_eig = s ** 2 + lam
    cond_G = float(g_eig.max() / g_eig.min()) if g_eig.min() > 0 else np.inf
    cond_Xv = float(s.max() / s[s > rtol].min()) if rank else np.inf

    actual = actual_residual(Av, ThetaHat, A0, Theta)
    scale = float(np.linalg.norm(actual))

    # ── literal formula from the brief ──────────────────────────────────
    literal_ok = True
    try:
        pd_primal = predicted_delta_primal(Xv, DeltaX, Av, lam)
    except np.linalg.LinAlgError:
        literal_ok = False
        pd_primal = predicted_delta_svd(Xv, DeltaX, Av, lam, svd=svd)
    if literal_ok and lam == 0.0 and rank < p:
        # solve() on a singular Gram silently returns garbage on some LAPACK
        # paths rather than raising; treat rank deficiency as "unavailable".
        literal_ok = False
    lit_res = predicted_residual(transfer_defect_literal(DeltaA, pd_primal), Theta)
    lit_err = np.abs(lit_res - actual)

    # ── regime-correct formula ──────────────────────────────────────────
    pd_svd = predicted_delta_svd(Xv, DeltaX, Av, lam, svd=svd)
    g = transfer_defect(DeltaA, pd_svd, Xv, Av, lam, toward_orig, svd=svd)
    exact_res = predicted_residual(g, Theta)
    exact_err = np.abs(exact_res - actual)

    # ── primal / dual / svd cross-checks ────────────────────────────────
    pd_dual_err = np.nan
    if check_dual and lam > 0 and n <= dual_max_b:
        pd_dual = predicted_delta_dual(Xv, DeltaX, Av, lam)
        pd_dual_err = rel_fro(pd_svd, pd_dual)
    primal_vs_svd = rel_fro(pd_svd, pd_primal) if literal_ok else np.nan

    return IdentityReport(
        n_eval=int(Av.shape[0]), n_cal=n, p=p, q=q, lam=float(lam),
        mode="toward_orig" if toward_orig else "toward_zero",
        dtype=np.dtype(dtype).name, numerical_rank=rank,
        cond_Xv=cond_Xv, cond_G=cond_G,
        literal_max_abs=float(lit_err.max()), literal_mean_abs=float(lit_err.mean()),
        literal_rel_fro=rel_fro(actual, lit_res), literal_available=bool(literal_ok),
        exact_max_abs=float(exact_err.max()), exact_mean_abs=float(exact_err.mean()),
        exact_rel_fro=rel_fro(actual, exact_res),
        primal_vs_dual=float(pd_dual_err), primal_vs_svd=float(primal_vs_svd),
        residual_scale=scale,
    )
