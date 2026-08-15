"""Normalized correction-strength and spectral-support diagnostics.

The mean objective removes the trivial dependence of lambda on the correction-row count,
but it does **not** normalize feature scale or Gram geometry. These diagnostics supply the
comparison coordinates, computed from the mean Gram

    G_mean = X^T X / n.

None of these quantities is a selection rule, and none has a universal optimum;
``lambda_gram_normalized`` in particular is a comparison coordinate only.

Every scalar here is computed by :mod:`experiments.lambda_study.core`, which owns the
sufficient statistics and the spectral primitives; this module owns only the *objective
semantics* (mean vs legacy sum) and the artifact field names.

Naming, because two vocabularies meet here
------------------------------------------
This module reports the names the P&C experiment protocol asks for; ``core`` reports the
names the large-lambda study used. They refer to the same two quantities::

    diagnostics.lambda_mean             == core.lambda_row    (lambda_sum / n)
    diagnostics.lambda_gram_normalized  == core.lambda_mean   (lambda_sum / (tr G / p))

Both spellings are emitted so a join across `protocol_v2/correction/diagnostics.csv` and
`lambda_study/spectral/spectral_metrics.csv` is unambiguous.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.lambda_study import core  # noqa: E402
from experiments.pnc_protocol.ridge import RidgeSpecification  # noqa: E402

# Above this augmented width, an exact eigendecomposition of G_mean is not attempted.
EXACT_EIG_MAX_P = 6000


@dataclass
class CorrectionDiagnostics:
    """Every field the P&C experiment protocol's artifact manifest requires."""
    objective_reduction: str
    ridge_center: str
    lambda_mean: float
    lambda_sum_equivalent: float
    lambda_gram_normalized: float
    core_lambda_row: float          # alias of lambda_mean, in core's vocabulary
    core_lambda_mean: float         # alias of lambda_gram_normalized, in core's vocabulary
    n_rows: int
    p_augmented: int
    n_over_p: float
    mean_gram_eigenvalue: float
    c_train: float
    c_val: float
    correction_norm_ratio: float
    calibration_residual: float
    validation_residual: float
    condition_number: float
    solver_dtype: str
    effective_dimension: float | None = None
    effective_dimension_fraction: float | None = None
    effective_dimension_method: str = "not_computed"
    effective_dimension_omitted_reason: str | None = None
    correction_row_policy: str | None = None
    correction_row_hash: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def compute(X, Z, Theta, spec: RidgeSpecification, C, *,
            X_val=None, Z_val=None, st: core.SuffStats | None = None,
            st_val: core.SuffStats | None = None, mu=None,
            row_policy: str | None = None, row_hash: str | None = None,
            exact_eig_max_p: int = EXACT_EIG_MAX_P) -> CorrectionDiagnostics:
    """Diagnostics for one solved correction.

    ``C`` is the correction (``Theta_hat - Theta``) already solved under ``spec``.
    Held-out ``c_val`` is NaN when no validation design is supplied.

    ``st``, ``st_val`` and ``mu`` are the member's sufficient statistics and Gram
    eigenvalues. They depend only on the member, never on lambda, so a caller sweeping a
    lambda grid should build them once and pass them in — otherwise this function rebuilds
    a (p, p) Gram and re-runs ``eigvalsh`` on every call, which at p=3073 is ~2 s of
    lambda-invariant work per lambda.
    """
    Theta = np.asarray(Theta, np.float64)
    if st is None:
        st = core.suff_stats(X, Z, Theta, bias_col=None)
    if st_val is None and X_val is not None and Z_val is not None:
        st_val = core.suff_stats(np.asarray(X_val, np.float64), Z_val, Theta, bias_col=None)

    n, p = st.n, st.p
    lam_mean = spec.lambda_mean(n)
    lam_sum = spec.lambda_sum(n)
    g_mean = st.gram_mean_eig / n          # tr(G_mean)/p = tr(X^T X)/(n p)
    Ct = np.asarray(C, np.float64)

    # --- effective dimension on the MEAN Gram, guarded by size -------------
    d_eff = d_eff_frac = None
    method, omitted = "not_computed", None
    cond = float("nan")
    if p <= exact_eig_max_p:
        if mu is None:
            mu = core.eigensystem(st.G, want_vectors=False).mu
        mu_mean = np.asarray(mu, np.float64) / n           # eigenvalues of G/n
        rank = int((mu_mean > float(mu_mean.max()) * core.RANK_TOL_REL).sum())
        d_eff, d_eff_frac = core.effective_dimension(mu_mean, lam_mean, rank)
        method = "exact_eigvalsh"
        cond = core.condition_number(st, lam_mean, mu_mean)
    else:
        omitted = (f"p_augmented={p} exceeds exact_eig_max_p={exact_eig_max_p}; "
                   "no validated stochastic estimator is wired in, so the diagnostic is "
                   "omitted rather than approximated")

    c_tr, resid_tr = core.fraction_and_residual(st, Ct)
    if st_val is not None:
        c_va, resid_va = core.fraction_and_residual(st_val, Ct)
    else:
        c_va = resid_va = float("nan")

    return CorrectionDiagnostics(
        objective_reduction=spec.objective_reduction,
        ridge_center=spec.center,
        lambda_mean=lam_mean,
        lambda_sum_equivalent=lam_sum,
        lambda_gram_normalized=lam_mean / g_mean if g_mean > 0 else float("inf"),
        core_lambda_row=lam_mean,
        core_lambda_mean=lam_mean / g_mean if g_mean > 0 else float("inf"),
        n_rows=int(n), p_augmented=int(p), n_over_p=float(n) / p,
        mean_gram_eigenvalue=float(g_mean),
        c_train=c_tr, c_val=c_va,
        correction_norm_ratio=core.correction_norm_ratio(Ct, Theta),
        calibration_residual=resid_tr, validation_residual=resid_va,
        condition_number=cond,
        solver_dtype="float64",
        effective_dimension=d_eff,
        effective_dimension_fraction=d_eff_frac,
        effective_dimension_method=method,
        effective_dimension_omitted_reason=omitted,
        correction_row_policy=row_policy,
        correction_row_hash=row_hash,
    )
