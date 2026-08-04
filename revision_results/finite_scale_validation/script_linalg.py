"""Exact float64 linear algebra for the revised P&C correction theory.

Pure numpy, GPU-free, deterministic. Everything here is unit-tested against
random synthetic problems to ~1e-10 relative error (see tests/).

The single central object is :class:`Correction`, which holds one finite-sample
ridge-regression correction problem (one ensemble member, one perturbed layer's
correction interface) and exposes every quantity the spec's "shared analysis
unit" (Section 0.5) asks for, in both ridge conventions:

    mode="toward_zero"  -> matches the DEFAULT implementation (ridge to 0)
    mode="toward_orig"  -> matches the spec Eq (1)-(4) (ridge to Theta)

All identities are checked in the accompanying tests; the methods here are the
reference implementations those tests validate.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

Mode = Literal["toward_zero", "toward_orig"]
Reduction = Literal["mean", "sum_legacy"]
_EPS = 1e-30


def _as_f64(a) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(a, dtype=np.float64))


def sum_lambda_to_mean(lambda_sum: float, n_rows: int) -> float:
    """Convert a summed-objective ridge value to its mean-objective equivalent."""
    if n_rows <= 0:
        raise ValueError(f"n_rows must be positive, got {n_rows}")
    return float(lambda_sum) / float(n_rows)


def mean_lambda_to_sum(lambda_mean: float, n_rows: int) -> float:
    """Convert a mean-objective ridge value to its summed-objective equivalent."""
    if n_rows <= 0:
        raise ValueError(f"n_rows must be positive, got {n_rows}")
    return float(lambda_mean) * float(n_rows)


def ridge_solve(Xv: np.ndarray, target: np.ndarray, lam: float,
                w_prior: np.ndarray | None = None,
                objective_reduction: Reduction = "sum_legacy") -> np.ndarray:
    """Reference ridge/pseudoinverse solve of  ``Xv @ W ≈ target``.

    ``objective_reduction`` selects the data-term normalization and is **explicit** —
    it is never inferred from the magnitude of ``lam``:

    ``"sum_legacy"`` (default, unchanged historical behaviour)
        min ||Xv W - target||^2 + lam ||W - w_prior||^2
        W = (Xv^T Xv + lam I)^-1 (Xv^T target + lam w_prior)

    ``"mean"``
        min (1/n) ||Xv W - target||^2 + lam ||W - w_prior||^2
        W = (Xv^T Xv/n + lam I)^-1 (Xv^T target/n + lam w_prior)

    The two coincide under ``lam_sum = n * lam_mean`` (:func:`mean_lambda_to_sum`);
    that identity is the migration's numerical gate, so ``"mean"`` is implemented
    literally (dividing by n) rather than by rescaling into the legacy path.

    The default stays ``"sum_legacy"`` so every existing call site is byte-identical.
    New experiments must go through the high-level API, which requires an explicit
    :class:`RidgeSpecification`.

    lam == 0 is minimum-norm least squares in both modes (w_prior ignored).
    """
    if objective_reduction not in ("mean", "sum_legacy"):
        raise ValueError(f"objective_reduction must be 'mean' or 'sum_legacy', "
                         f"got {objective_reduction!r}")
    Xv = _as_f64(Xv)
    target = _as_f64(target)
    if lam > 0.0:
        n, p = Xv.shape
        if objective_reduction == "mean":
            G = (Xv.T @ Xv) / n + lam * np.eye(p)
            rhs = (Xv.T @ target) / n
        else:
            G = Xv.T @ Xv + lam * np.eye(p)
            rhs = Xv.T @ target
        if w_prior is not None:
            rhs = rhs + lam * _as_f64(w_prior)
        return np.linalg.solve(G, rhs)
    W, *_ = np.linalg.lstsq(Xv, target, rcond=None)
    return W


@dataclass
class Correction:
    """One exact correction problem, all quantities in float64.

    Parameters
    ----------
    X, Xv : (n, p)  original / perturbed bias-augmented designs (bias LAST col).
    Theta : (p, d)  original next affine layer, code layout (preact = X @ Theta).
    lam   : ridge value (absolute, as in the implementation).
    mode  : "toward_zero" (default impl) or "toward_orig" (spec Eq 1-4).
    """
    X: np.ndarray
    Xv: np.ndarray
    Theta: np.ndarray
    lam: float = 0.0
    mode: Mode = "toward_zero"
    # cached
    _svd: tuple | None = field(default=None, repr=False, compare=False)

    def __post_init__(self):
        self.X = _as_f64(self.X)
        self.Xv = _as_f64(self.Xv)
        self.Theta = _as_f64(self.Theta)
        self.lam = float(self.lam)
        assert self.X.shape == self.Xv.shape, (self.X.shape, self.Xv.shape)
        assert self.Theta.shape[0] == self.X.shape[1], (self.Theta.shape, self.X.shape)
        self.n, self.p = self.Xv.shape
        self.d = self.Theta.shape[1]
        self.dX = self.Xv - self.X
        self.target = self.X @ self.Theta
        self.G = self.Xv.T @ self.Xv + self.lam * np.eye(self.p)
        # thin SVD of Xv = U S V^T (U: n×r, s: r, Vt: r×p; r=min(n,p)).
        # Every analytic quantity below is expressed through the filter factors
        #   f1 = s/(s^2+lam)   (solve gain, -> 1/s at lam=0 = pseudoinverse)
        #   f2 = s^2/(s^2+lam) (hat gain)
        # which is exact and numerically stable across the well-/ill-/under-
        # determined regimes (G^-1 is singular at lam=0 when n<p).
        self.U, self.s, self.Vt = np.linalg.svd(self.Xv, full_matrices=False)
        self.V = self.Vt.T
        smax = self.s.max() if self.s.size else 0.0
        # rank tolerance for the pseudoinverse (lam=0) path
        self._rtol = smax * max(self.n, self.p) * np.finfo(np.float64).eps
        pos = self.s > self._rtol
        if self.lam == 0.0:
            f1 = np.zeros_like(self.s)
            f1[pos] = 1.0 / self.s[pos]                 # pseudoinverse: 1/s
        else:
            f1 = self.s / (self.s ** 2 + self.lam)
        self.f1 = f1                                    # solve filter s/(s^2+lam)
        self.f2 = self.s * self.f1                      # hat filter  s^2/(s^2+lam)

    # ── SVD helpers ───────────────────────────────────────────────────────
    def _apply_Ginv(self, w: np.ndarray) -> np.ndarray:
        """G^-1 @ w with w shape (p, k). Row-space part via V diag(1/(s^2+lam)) V^T;
        null-space part (only exists when n<p) via (1/lam)(I - V V^T) for lam>0
        (dropped for lam=0, i.e. pseudoinverse on the row space)."""
        w = _as_f64(w)
        Vt_w = self.Vt @ w                              # (r, k)
        denom = self.s ** 2 + self.lam
        row = self.V @ (Vt_w / denom[:, None])          # (p, k)
        if self.lam > 0 and self.p > self.s.shape[0]:
            null = w - self.V @ Vt_w                     # (I - V V^T) w
            row = row + null / self.lam
        return row

    # ── correction weights ────────────────────────────────────────────────
    @property
    def w_prior(self) -> np.ndarray | None:
        return self.Theta if (self.mode == "toward_orig" and self.lam > 0) else None

    def theta_hat(self) -> np.ndarray:
        """Fitted correction via the reference solver (matches implementation)."""
        return ridge_solve(self.Xv, self.target, self.lam, self.w_prior)

    def theta_hat_svd(self) -> np.ndarray:
        """Fitted correction via the stable SVD filter form (all regimes).

        toward_zero:  Theta_hat = V diag(f1) U^T target
        toward_orig:  Theta_hat = Theta + V diag(f1) U^T (target - Xv Theta)
        with f1 = s/(s^2+lam) (= 1/s at lam=0 on the row space = min-norm).
        Exact for n>=p and for the n<p / rank-deficient / ridge cases alike.
        """
        if self.mode == "toward_orig" and self.lam > 0:
            r0 = self.target - self.Xv @ self.Theta
            delta = self.V @ (self.f1[:, None] * (self.U.T @ r0))
            return self.Theta + delta
        return self.V @ (self.f1[:, None] * (self.U.T @ self.target))

    # keep the spec's algebraic closed form as an explicit cross-check for the
    # invertible-G regime (used only where G^-1 is well defined).
    def theta_hat_formula(self) -> np.ndarray:
        """Spec Eq (1) closed form via G^-1 (valid when G is invertible).

        At lambda=0 the G^-1 form is only valid for full column rank; when Xv is
        rank-deficient (common for post-ReLU designs) G is singular and the
        spec's closed form reduces to the pseudoinverse = the stable SVD form."""
        if self.lam == 0.0:
            return self.theta_hat_svd()
        Ginv = self._apply_Ginv(np.eye(self.p))
        base = self.Theta - Ginv @ (self.Xv.T @ (self.dX @ self.Theta))
        if self.mode == "toward_zero" and self.lam > 0:
            base = base - self.lam * (Ginv @ self.Theta)
        return base

    # ── hat matrix / leverage (Section 0.5, 4.x) ──────────────────────────
    def hat_diag(self) -> np.ndarray:
        """Ridge leverages of the calibration rows: diag(Xv G^-1 Xv^T)
        = sum_k U[i,k]^2 * f2_k  (stable, no n×n matrix)."""
        return (self.U ** 2) @ self.f2

    def test_leverage(self, hvb: np.ndarray) -> np.ndarray:
        """Ridge self-leverage h^lam(x) = hvb G^-1 hvb^T for test rows hvb (B,p).

        Row-space part sum_k (V^T hvb)_k^2 / (s_k^2+lam); for lam>0 a novel
        (out-of-row-space) direction adds ||(I-VV^T)hvb||^2/lam (correctly ->inf
        as lam->0). At lam=0 only the pseudoinverse row-space part is returned."""
        hvb = _as_f64(np.atleast_2d(hvb))
        Vt_h = hvb @ self.V                              # (B, r)
        denom = self.s ** 2 + self.lam
        if self.lam == 0.0:
            denom = np.where(self.s > self._rtol, self.s ** 2, np.inf)
        lev = (Vt_h ** 2) @ (1.0 / denom)
        if self.lam > 0 and self.p > self.s.shape[0]:
            null_sq = np.sum(hvb ** 2, axis=1) - np.sum(Vt_h ** 2, axis=1)
            lev = lev + np.maximum(null_sq, 0.0) / self.lam
        return lev

    def alpha(self, hvb: np.ndarray) -> np.ndarray:
        """Reconstruction weights alpha_v(x) = Xv G^-1 hvb^T = U diag(f1) V^T hvb^T,
        shape (B, n). Stable in all regimes."""
        hvb = _as_f64(np.atleast_2d(hvb))
        Vt_h = hvb @ self.V                              # (B, r)
        return (self.U @ (self.f1[:, None] * Vt_h.T)).T  # (B, n)

    # ── residuals / transfer defect (Eq 2,3,4 and impl variant 2') ────────
    def residual_direct(self, hb: np.ndarray, hvb: np.ndarray,
                        theta_hat: np.ndarray | None = None) -> np.ndarray:
        """Exact post-correction residual r_v(x) = Theta_hat hvb - Theta hb.

        Uses the *actual* fitted correction (implementation) by default so this
        is the ground-truth residual, not a reconstruction."""
        hb = _as_f64(np.atleast_2d(hb))
        hvb = _as_f64(np.atleast_2d(hvb))
        th = self.theta_hat() if theta_hat is None else _as_f64(theta_hat)
        return hvb @ th - hb @ self.Theta

    def transfer_defect(self, hb: np.ndarray, hvb: np.ndarray) -> np.ndarray:
        """Representation-level transfer defect g_v(x)  (Eq 3), shape (B, p).

        toward_orig:  g = dhb - dX^T alpha                    (all regimes, exact)
        toward_zero:  g = dhb - dX^T alpha - (hvb - hvb V diag(f2) V^T)
                      where the bias term equals lam*hvb G^-1 for n>=p and adds
                      the out-of-row-space component hvb(I-VV^T) when n<p.
        alpha_v(x) = Xv G^-1 hvb^T (Eq for the reconstruction weights).
        """
        hb = _as_f64(np.atleast_2d(hb))
        hvb = _as_f64(np.atleast_2d(hvb))
        dhb = hvb - hb
        alpha = self.alpha(hvb)                          # (B, n)
        g = dhb - alpha @ self.dX                        # (B, p)
        # The clean toward-orig form (no bias term) is exact only when a genuine
        # ridge-toward-Theta solve occurred (lam>0). At lam=0 both modes collapse
        # to the min-norm solve, which carries the out-of-row-space bias term.
        if not (self.mode == "toward_orig" and self.lam > 0):
            Vt_h = hvb @ self.V                          # (B, r)
            hvb_proj = (Vt_h * self.f2) @ self.Vt        # hvb V diag(f2) V^T
            g = g - (hvb - hvb_proj)
        return g

    def residual_formula(self, hb: np.ndarray, hvb: np.ndarray) -> np.ndarray:
        """Reconstructed residual via r = Theta g_v  (Eq 4 / 2, impl variant 2')."""
        g = self.transfer_defect(hb, hvb)
        return g @ self.Theta

    # ── calibration residual / hat identity (Eq 5) ────────────────────────
    def calibration_residual(self, theta_hat: np.ndarray | None = None) -> np.ndarray:
        """R_{S,v} = Xv Theta_hat - X Theta   (fitted minus original on cal set)."""
        th = self.theta_hat() if theta_hat is None else _as_f64(theta_hat)
        return self.Xv @ th - self.target

    def calibration_residual_formula(self) -> np.ndarray:
        """Eq (5):  R_S = (I - H) dX Theta   (toward_orig, all regimes),
        with H = Xv G^-1 Xv^T = U diag(f2) U^T. For toward_zero an extra
        -lam Xv G^-1 Theta = -lam U diag(f1) V^T Theta term appears."""
        H_dX = self.U @ (self.f2[:, None] * (self.U.T @ self.dX))   # H @ dX
        R = (self.dX - H_dX) @ self.Theta
        if self.mode == "toward_zero":
            R = R - self.lam * (self.U @ (self.f1[:, None] * (self.Vt @ self.Theta)))
        return R

    def normal_equation_residual(self, theta_hat: np.ndarray | None = None) -> np.ndarray:
        """Gradient of the (regularized) objective at Theta_hat — should be ~0.

        toward_orig: Xv^T(Xv W - target) + lam (W - Theta)
        toward_zero: Xv^T(Xv W - target) + lam  W
        """
        th = self.theta_hat() if theta_hat is None else _as_f64(theta_hat)
        grad = self.Xv.T @ (self.Xv @ th - self.target)
        if self.lam > 0:
            grad = grad + self.lam * (th - (self.Theta if self.mode == "toward_orig" else 0.0))
        return grad

    # ── spectral diagnostics (Section 0.5, Round 1.4) ─────────────────────
    def svd(self):
        return self.U, self.s, self.Vt

    def spectrum(self) -> dict:
        s = self.s
        smax = float(s.max())
        rtol = self._rtol
        rank = int((s > rtol).sum())
        s_nonzero = s[s > 0]
        # G eigenvalues are s^2 + lam
        g_eig = s ** 2 + self.lam
        logdet_G = float(np.sum(np.log(g_eig)))
        return {
            "n": self.n, "p": self.p, "d": self.d,
            "n_over_p": self.n / self.p,
            "s": s,
            "s_min": float(s.min()), "s_max": smax,
            "s_min_nonzero": float(s_nonzero.min()) if s_nonzero.size else 0.0,
            "numerical_rank": rank,
            "cond_Xv": float(smax / s[s > 0].min()) if (s > 0).any() else np.inf,
            "cond_G": float(g_eig.max() / g_eig.min()),
            "trace_G": float(np.sum(g_eig)),
            "stable_rank_Xv": float((s ** 2).sum() / (smax ** 2)),
            "eff_rank_Xv": float((s.sum() ** 2) / (s ** 2).sum()),  # participation ratio of s
            "logdet_G": logdet_G,
            "ridge_gain_max": float(np.max(s / (s ** 2 + self.lam))) if self.lam > 0 else np.inf,
        }

    def correction_norm(self, theta_hat: np.ndarray | None = None) -> float:
        th = self.theta_hat() if theta_hat is None else _as_f64(theta_hat)
        return float(np.linalg.norm(th - self.Theta))


# ── stand-alone identities ────────────────────────────────────────────────

def leverage_mahalanobis(h_cal: np.ndarray, h_test: np.ndarray):
    """Exact OLS(lambda=0) leverage vs Mahalanobis distance identity, Eq (8).

    For the intercept-augmented design X = [h, 1] with G = X^T X (lambda=0),
    the leverage of a test row equals

        h_S(x) = 1/n + d_Mah(x)^2 / (n-1)

    with d_Mah^2 = (h-mu)^T Sigma^-1 (h-mu), Sigma = (1/(n-1)) Z^T Z the sample
    covariance of the *calibration* h (Z = centered h_cal). Uses the pseudo
    inverse so it stays valid when Sigma is rank-deficient.

    Returns (leverage, d_mah_sq, rhs) each shape (B,), where rhs is the Eq (8)
    prediction — leverage and rhs must agree to machine precision.
    """
    h_cal = _as_f64(h_cal)
    h_test = _as_f64(np.atleast_2d(h_test))
    n = h_cal.shape[0]
    mu = h_cal.mean(axis=0)
    Z = h_cal - mu
    Sigma = (Z.T @ Z) / (n - 1)
    Sigma_pinv = np.linalg.pinv(Sigma)
    diff = h_test - mu
    d_mah_sq = np.einsum("ij,jk,ik->i", diff, Sigma_pinv, diff)

    # leverage directly from the augmented design (via pinv of Gram)
    Xcal = np.concatenate([h_cal, np.ones((n, 1))], axis=1)
    G = Xcal.T @ Xcal
    Ginv = np.linalg.pinv(G)
    Xtest = np.concatenate([h_test, np.ones((h_test.shape[0], 1))], axis=1)
    leverage = np.einsum("ij,jk,ik->i", Xtest, Ginv, Xtest)

    rhs = 1.0 / n + d_mah_sq / (n - 1)
    return leverage, d_mah_sq, rhs


def rel_err(a: np.ndarray, b: np.ndarray) -> float:
    """Frobenius relative error ||a-b|| / (||a|| + eps)."""
    a = _as_f64(a); b = _as_f64(b)
    return float(np.linalg.norm(a - b) / (np.linalg.norm(a) + _EPS))
