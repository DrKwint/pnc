"""Sum/mean equivalence, diagnostics and selector tests (brief §2.3, §11)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "scripts"))

from pnc_theory.linalg import (  # noqa: E402
    mean_lambda_to_sum, ridge_solve, sum_lambda_to_mean,
)
from experiments.pnc_protocol.ridge import RidgeSpecification, solve_correction  # noqa: E402
from experiments.pnc_protocol import diagnostics as D  # noqa: E402
from experiments.pnc_protocol.selection import (  # noqa: E402
    CLASSIFICATION_DEFAULT_BUDGET, Candidate, IDPreservationBudget,
    PreservationConstraint, select_weakest_admissible,
)

PARAM_RTOL, PRED_ATOL = 1e-9, 1e-6


def _problem(n, p, d, seed=0, duplicate_rows=False):
    rng = np.random.RandomState(seed)
    X = rng.normal(size=(n, p))
    X[:, -1] = 1.0                                   # bias column (last, MuJoCo convention)
    if duplicate_rows:                               # bootstrap-style duplicates
        idx = rng.randint(0, n, n)
        X = X[idx]
    Theta = rng.normal(size=(p, d)) * 0.1
    Z = X @ Theta + rng.normal(size=(len(X), d)) * 0.3
    return X, Z, Theta


# --------------------------------------------------------------- equivalence
@pytest.mark.parametrize("name,n,p", [
    ("full_rank", 200, 20),
    ("underdetermined", 12, 30),
    ("bootstrapped_duplicates", 200, 20),
])
def test_sum_mean_equivalence(name, n, p):
    X, Z, Theta = _problem(n, p, 7, seed=hash(name) % 1000,
                           duplicate_rows=(name == "bootstrapped_duplicates"))
    n_rows = X.shape[0]
    for lam_sum in (1e-4, 1.0, 1e3, 1e6):
        lam_mean = sum_lambda_to_mean(lam_sum, n_rows)
        a = ridge_solve(X, Z, lam_sum, w_prior=Theta, objective_reduction="sum_legacy")
        b = ridge_solve(X, Z, lam_mean, w_prior=Theta, objective_reduction="mean")
        rel = np.abs(a - b).max() / max(np.abs(a).max(), 1e-30)
        assert rel <= PARAM_RTOL, f"{name} lam_sum={lam_sum}: param rel err {rel:.3e}"
        assert np.abs(X @ a - X @ b).max() <= PRED_ATOL


def test_lambda_conversions_roundtrip():
    for lam, n in ((1e-3, 9000), (1e4, 9000), (0.1, 409)):
        assert mean_lambda_to_sum(sum_lambda_to_mean(lam, n), n) == pytest.approx(lam)
    with pytest.raises(ValueError):
        sum_lambda_to_mean(1.0, 0)


def test_default_reduction_is_unchanged_legacy():
    """The low-level solver must not change behaviour for existing call sites."""
    X, Z, Theta = _problem(50, 8, 3, seed=5)
    explicit = ridge_solve(X, Z, 2.5, w_prior=Theta, objective_reduction="sum_legacy")
    default = ridge_solve(X, Z, 2.5, w_prior=Theta)
    assert np.array_equal(explicit, default)


def test_reduction_must_be_explicit_not_inferred():
    with pytest.raises(ValueError):
        ridge_solve(np.eye(4), np.eye(4), 1.0, objective_reduction="auto")
    with pytest.raises(ValueError):
        RidgeSpecification(1.0, objective_reduction="whatever")


def test_spec_carries_semantics():
    s = RidgeSpecification(1e4, "sum_legacy", "original")
    assert s.lambda_mean(9000) == pytest.approx(1e4 / 9000)
    assert s.as_mean(9000).objective_reduction == "mean"
    assert s.to_dict(9000)["lambda_sum_equivalent"] == pytest.approx(1e4)


# ------------------------------------------------------------ centering/limits
def test_center_original_vs_zero():
    X, Z, Theta = _problem(80, 10, 4, seed=11)
    big = RidgeSpecification(1e8, "mean", "original")
    C = solve_correction(X, Z, Theta, big)
    assert np.abs(C).max() < 1e-4, "large lambda toward original must approach zero correction"
    Cz = solve_correction(X, Z, Theta, RidgeSpecification(1e8, "mean", "zero"))
    assert np.abs(Cz + Theta).max() < 1e-4, "toward zero must approach -Theta"


def test_small_lambda_matches_least_squares():
    X, Z, Theta = _problem(300, 12, 5, seed=3)
    C = solve_correction(X, Z, Theta, RidgeSpecification(1e-12, "mean", "original"))
    ls, *_ = np.linalg.lstsq(X, Z, rcond=None)
    assert np.abs((Theta + C) - ls).max() < 1e-5


def test_bias_row_is_regularized_like_the_kernel():
    """Both conventions ridge every row including the bias — verify no row is exempt."""
    X, Z, Theta = _problem(60, 6, 2, seed=9)
    C = solve_correction(X, Z, Theta, RidgeSpecification(1e6, "mean", "original"))
    assert np.abs(C).max() < 1e-2                 # every row shrunk, bias included
    assert np.abs(C[-1]).max() <= np.abs(C).max()


# ---------------------------------------------------------------- diagnostics
def test_diagnostics_fields_and_monotone_effective_dimension():
    X, Z, Theta = _problem(400, 25, 6, seed=13)
    Xv, Zv, _ = _problem(150, 25, 6, seed=14)
    prev = np.inf
    for lam in (1e-6, 1e-3, 1e-1, 1.0, 1e2):
        spec = RidgeSpecification(lam, "mean", "original")
        C = solve_correction(X, Z, Theta, spec)
        d = D.compute(X, Z, Theta, spec, C, X_val=Xv, Z_val=Zv, row_policy="one_per_example")
        assert d.effective_dimension is not None and d.effective_dimension < prev
        prev = d.effective_dimension
        assert 0.0 <= d.effective_dimension_fraction <= 1.0
        assert d.lambda_gram_normalized == pytest.approx(lam / d.mean_gram_eigenvalue)
        assert d.n_over_p == pytest.approx(X.shape[0] / X.shape[1])
        assert np.isfinite(d.c_train) and np.isfinite(d.c_val)
    assert d.c_train < 0.999                       # strong ridge leaves mismatch behind


def test_effective_dimension_omitted_not_approximated_when_large():
    X, Z, Theta = _problem(60, 40, 3, seed=17)
    spec = RidgeSpecification(1e-2, "mean", "original")
    C = solve_correction(X, Z, Theta, spec)
    d = D.compute(X, Z, Theta, spec, C, exact_eig_max_p=10)
    assert d.effective_dimension is None
    assert d.effective_dimension_method == "not_computed"
    assert "exceeds exact_eig_max_p" in d.effective_dimension_omitted_reason


def test_c_train_stable_when_mismatch_is_tiny():
    rng = np.random.RandomState(2)
    X = rng.normal(size=(50, 5)); X[:, -1] = 1.0
    Theta = rng.normal(size=(5, 2))
    Z = X @ Theta                                   # D == 0 exactly
    spec = RidgeSpecification(1e-3, "mean", "original")
    C = solve_correction(X, Z, Theta, spec)
    d = D.compute(X, Z, Theta, spec, C)
    assert np.isfinite(d.c_train) or np.isnan(d.c_train)   # must not raise


def test_rank_deficient_design_is_handled():
    rng = np.random.RandomState(4)
    B = rng.normal(size=(80, 3))
    X = np.hstack([B, B @ rng.normal(size=(3, 4)), np.ones((80, 1))])   # rank 3 + bias
    Theta = rng.normal(size=(X.shape[1], 2))
    Z = X @ Theta + rng.normal(size=(80, 2)) * 0.1
    spec = RidgeSpecification(1e-3, "mean", "original")
    d = D.compute(X, Z, Theta, spec, solve_correction(X, Z, Theta, spec))
    assert d.effective_dimension <= X.shape[1]


# ------------------------------------------------------------------ selection
def _cand(key, strength, acc=0.0, nll=0.0, ece=0.0, agree=1.0, seeds=(0, 1, 2)):
    return Candidate(key, strength, {s: {"accuracy_drop_pp": acc, "nll_increase": nll,
                                         "ece_increase": ece, "base_agreement": agree}
                                     for s in seeds})


def test_selects_largest_admissible_strength():
    r = select_weakest_admissible(
        [_cand("a", 1.0), _cand("b", 10.0), _cand("c", 100.0, acc=5.0)],
        CLASSIFICATION_DEFAULT_BUDGET)
    assert r.status == "ok" and r.selected.key == "b"
    assert not r.fell_back_to_strongest
    assert "c" in r.rejected


def test_single_constraint_violation_rejects():
    r = select_weakest_admissible([_cand("a", 1.0), _cand("b", 10.0, ece=0.5)],
                                  CLASSIFICATION_DEFAULT_BUDGET)
    assert r.selected.key == "a" and "b" in r.rejected


def test_per_seed_guard_blocks_averaging_over_one_bad_seed():
    c = Candidate("uneven", 10.0, {0: {"accuracy_drop_pp": 0.0, "nll_increase": 0.0,
                                       "ece_increase": 0.0, "base_agreement": 1.0},
                                   1: {"accuracy_drop_pp": 0.0, "nll_increase": 0.0,
                                       "ece_increase": 0.0, "base_agreement": 1.0},
                                   2: {"accuracy_drop_pp": 0.6, "nll_increase": 0.0,
                                       "ece_increase": 0.0, "base_agreement": 1.0}})
    # mean 0.2 pp passes, but seed 2 exceeds 2x the 0.25 pp budget
    r = select_weakest_admissible([_cand("safe", 1.0), c], CLASSIFICATION_DEFAULT_BUDGET)
    assert r.selected.key == "safe" and "uneven" in r.rejected


def test_falls_back_to_strongest_and_flags_it():
    r = select_weakest_admissible(
        [_cand("strong", 0.1), _cand("weak", 10.0, nll=1.0)],
        CLASSIFICATION_DEFAULT_BUDGET)
    assert r.status == "ok" and r.selected.key == "strong"
    assert r.fell_back_to_strongest and "STRONGEST" in r.message


def test_fails_loudly_when_nothing_passes():
    r = select_weakest_admissible([_cand("a", 1.0, nll=9.0), _cand("b", 10.0, nll=9.0)],
                                  CLASSIFICATION_DEFAULT_BUDGET)
    assert r.status == "no_admissible_candidate" and r.selected is None
    assert "least-bad" in r.message


def test_deterministic_tie_break():
    a = select_weakest_admissible([_cand("z", 5.0), _cand("a", 5.0)],
                                  CLASSIFICATION_DEFAULT_BUDGET)
    b = select_weakest_admissible([_cand("a", 5.0), _cand("z", 5.0)],
                                  CLASSIFICATION_DEFAULT_BUDGET)
    assert a.selected.key == b.selected.key == "z"


def test_regression_budget_must_be_supplied_not_inherited():
    reg = IDPreservationBudget(task="regression", constraints=(
        PreservationConstraint("rmse_increase", max_degradation=0.02),))
    with pytest.raises(ValueError, match="missing"):
        select_weakest_admissible([_cand("a", 1.0)], reg)   # classification metrics only
