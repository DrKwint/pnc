"""Explicit correction-objective semantics for P&C.

The historical solvers all minimize a **sum** over correction rows, which makes a raw
lambda incomparable across experiments with different row counts (Banking77 n=9000,
MuJoCo n=409, CIFAR n = N*H*W). The canonical objective going forward is the **mean**
reconstruction objective

    min_C  (1/n) ||X C^T - D||_F^2  +  lambda_mean ||C||_F^2

with kernel and bias alike ridged toward their original values, i.e.

    (X^T X / n + lambda_mean I) C^T = X^T D / n.

A :class:`RidgeSpecification` carries the value *together with* its semantics, so a bare
float can never reach the high-level API and be silently reinterpreted.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from pnc_theory.linalg import (  # noqa: E402
    Reduction, mean_lambda_to_sum, ridge_solve, sum_lambda_to_mean,
)

Center = Literal["original", "zero"]

__all__ = ["RidgeSpecification", "solve_correction", "sum_lambda_to_mean",
           "mean_lambda_to_sum", "Reduction", "Center"]


@dataclass(frozen=True)
class RidgeSpecification:
    """A ridge value *and* the semantics needed to interpret it.

    ``value`` is in the units implied by ``objective_reduction``: a mean-objective
    lambda when ``"mean"``, a summed-objective lambda when ``"sum_legacy"``.
    """
    value: float
    objective_reduction: Reduction = "mean"
    center: Center = "original"

    def __post_init__(self):
        if self.objective_reduction not in ("mean", "sum_legacy"):
            raise ValueError(f"unknown objective_reduction {self.objective_reduction!r}")
        if self.center not in ("original", "zero"):
            raise ValueError(f"unknown center {self.center!r}")
        if not np.isfinite(self.value) or self.value < 0:
            raise ValueError(f"ridge value must be finite and non-negative, got {self.value}")

    # -- coordinate conversions (require n, because that is the whole point) ----
    def lambda_mean(self, n_rows: int) -> float:
        return (self.value if self.objective_reduction == "mean"
                else sum_lambda_to_mean(self.value, n_rows))

    def lambda_sum(self, n_rows: int) -> float:
        return (self.value if self.objective_reduction == "sum_legacy"
                else mean_lambda_to_sum(self.value, n_rows))

    def as_mean(self, n_rows: int) -> "RidgeSpecification":
        return RidgeSpecification(self.lambda_mean(n_rows), "mean", self.center)

    def to_dict(self, n_rows: int | None = None) -> dict:
        d = {"value": float(self.value),
             "objective_reduction": self.objective_reduction,
             "center": self.center}
        if n_rows is not None:
            d.update({"n_rows": int(n_rows),
                      "lambda_mean": self.lambda_mean(n_rows),
                      "lambda_sum_equivalent": self.lambda_sum(n_rows)})
        return d


def solve_correction(X, Z, Theta, spec: RidgeSpecification) -> np.ndarray:
    """Return the correction ``C = Theta_hat - Theta`` under ``spec``.

    ``X`` (n, p) augmented perturbed design, ``Z`` (n, d) original targets,
    ``Theta`` (p, d) original affine parameters in the code layout ``preact = X @ Theta``.
    Bias placement is whatever the caller used to build ``X`` and ``Theta``; this
    function never reorders columns.
    """
    X = np.asarray(X, dtype=np.float64)
    Z = np.asarray(Z, dtype=np.float64)
    Theta = np.asarray(Theta, dtype=np.float64)
    if X.shape[0] != Z.shape[0]:
        raise ValueError(f"row mismatch: X {X.shape}, Z {Z.shape}")
    if X.shape[1] != Theta.shape[0]:
        raise ValueError(f"feature mismatch: X {X.shape}, Theta {Theta.shape}")
    prior = Theta if spec.center == "original" else None
    Th = ridge_solve(X, Z, spec.value, w_prior=prior,
                     objective_reduction=spec.objective_reduction)
    return Th - Theta
