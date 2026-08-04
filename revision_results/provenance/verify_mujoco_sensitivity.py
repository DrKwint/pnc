#!/usr/bin/env python
"""Reconstruct the rebuttal's MuJoCo one-factor sensitivity classification.

Source of truth
---------------
results/neurips_2026_rebuttal/mujoco_sensitivity/aggregates/far_sensitivity_raw.csv
(11 environments x 27 seeds x 7 factors, 14,520 rows, status == 'ok' for all).

What this reproduces
--------------------
The rebuttal quoted a per-cell classification of every (environment, metric) cell
of every swept factor into {sharp, flat, materially worse}:

    Scale               0 sharp / 38 flat /  6 worse
    Rank                0 sharp / 44 flat /  0 worse
    Bootstrap fraction  4 sharp / 31 flat /  9 worse
    Ridge               0 sharp / 43 flat /  1 worse
    Total               4 sharp / 156 flat / 16 worse

44 cells per factor = 11 environments x 4 metrics, and 176 total = 4 factors x 44,
so the cell definition is (environment, metric). NO SAVED SCRIPT PRODUCES THESE
COUNTS anywhere in the repository (searched: every .py/.md/.csv/.txt). This module
is therefore a *reconstruction* under the practical tolerances the rebuttal stated,
not a recovery of the original code. Every classification rule is spelled out below
so that a disagreement with the quoted counts is attributable to a rule, not to data.

Cell definition
---------------
For one (factor, environment, metric):
  1. Average the metric over seeds at each swept value of the factor  -> a curve.
  2. The anchor is the swept value equal to that environment's submitted
     configuration (row where `relative_to_anchor == 1.0`, cross-checked against
     `anchors.json`).
  3. A swept value is "materially worse than the anchor" if it exceeds the
     rebuttal's stated practical tolerance in the harmful direction:
         AUROC, Spearman  (higher better) : anchor - value  > 0.02
         NLL              (lower  better) : value  - anchor > 0.10
         RMSE             (lower  better) : value  > 1.05 * anchor   (5% relative)
  4. Classify the cell:
         sharp  : an *immediately adjacent* swept value (in sorted numeric order)
                  is materially worse  -> the anchor sits on a cliff
         worse  : some non-adjacent swept value is materially worse, but the
                  anchor's neighbourhood is within tolerance
         flat   : no swept value is materially worse than the anchor

Also emitted
------------
  * seed-paired anchor-vs-value differences with 95% t confidence intervals
  * the full bootstrap-fraction double-descent curves (`bootfull` factor)
  * the interpolation-threshold evidence: per-member calibration rows n vs the
    bias-augmented correction dimension p (= 201), both nominal (with-replacement
    draw size) and unique (distinct rows actually drawn)

Usage
-----
    .venv/bin/python revision_results/provenance/verify_mujoco_sensitivity.py
Writes into revision_results/mujoco_sensitivity/ and revision_results/layer_scope/.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "results/neurips_2026_rebuttal/mujoco_sensitivity/aggregates/far_sensitivity_raw.csv"
OUT = ROOT / "revision_results/mujoco_sensitivity"
LAYER_OUT = ROOT / "revision_results/layer_scope"

# metric -> (direction, tolerance kind, tolerance)
#   direction "hi" = higher is better; "lo" = lower is better
METRICS = {
    "id_rmse": ("lo", "rel", 0.05),
    "far_nll": ("lo", "abs", 0.10),
    "far_auroc": ("hi", "abs", 0.02),
    "far_spearman": ("hi", "abs", 0.02),
}
# the four factors the rebuttal classified (bootfull/calib/layer are reported separately)
FACTORS = ["scale", "rank", "bootstrap", "ridge"]


def materially_worse(value: float, anchor: float, metric: str) -> bool:
    direction, kind, tol = METRICS[metric]
    if not np.isfinite(value) or not np.isfinite(anchor):
        return False
    if direction == "hi":
        return (anchor - value) > tol
    if kind == "rel":
        return value > (1.0 + tol) * abs(anchor)
    return (value - anchor) > tol


def classify(curve: pd.Series, anchor_x: float, metric: str) -> str:
    """curve: index = numeric factor value (sorted), values = seed-mean metric."""
    xs = list(curve.index)
    if anchor_x not in curve.index:
        return "no_anchor"
    anchor = curve.loc[anchor_x]
    worse = {x: materially_worse(curve.loc[x], anchor, metric) for x in xs if x != anchor_x}
    if not any(worse.values()):
        return "flat"
    i = xs.index(anchor_x)
    neighbours = [xs[j] for j in (i - 1, i + 1) if 0 <= j < len(xs) and j != i]
    if any(worse.get(x, False) for x in neighbours):
        return "sharp"
    return "worse"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    LAYER_OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(RAW)
    assert (df.status == "ok").all(), "raw file contains non-ok rows"

    # ---------------------------------------------------------------- cells
    cells, paired = [], []
    for factor in FACTORS:
        sub = df[df.factor == factor]
        for env, g in sub.groupby("environment"):
            anchor_rows = g[np.isclose(g.relative_to_anchor, 1.0)]
            anchor_x = (
                float(anchor_rows.factor_value_numeric.iloc[0]) if len(anchor_rows) else float("nan")
            )
            for metric in METRICS:
                curve = g.groupby("factor_value_numeric")[metric].mean().sort_index()
                n_seeds = g.groupby("factor_value_numeric")[metric].size().min()
                verdict = classify(curve, anchor_x, metric)
                cells.append(
                    dict(
                        factor=factor,
                        environment=env,
                        metric=metric,
                        anchor_value=anchor_x,
                        anchor_metric=curve.get(anchor_x, np.nan),
                        curve_min=curve.min(),
                        curve_max=curve.max(),
                        curve_range=curve.max() - curve.min(),
                        n_swept_values=len(curve),
                        min_seeds_per_value=int(n_seeds),
                        verdict=verdict,
                    )
                )
                # seed-paired differences vs the anchor, 95% t interval
                wide = g.pivot_table(index="seed", columns="factor_value_numeric", values=metric)
                if anchor_x in wide.columns:
                    for x in wide.columns:
                        if x == anchor_x:
                            continue
                        d = (wide[x] - wide[anchor_x]).dropna()
                        if len(d) < 2:
                            continue
                        se = d.std(ddof=1) / math.sqrt(len(d))
                        t = 1.96 if len(d) > 30 else 2.06  # ~t.975 at df~25
                        paired.append(
                            dict(
                                factor=factor,
                                environment=env,
                                metric=metric,
                                anchor_value=anchor_x,
                                factor_value=x,
                                n_paired_seeds=len(d),
                                mean_diff=d.mean(),
                                ci95_lo=d.mean() - t * se,
                                ci95_hi=d.mean() + t * se,
                                materially_worse=materially_worse(
                                    wide[x].mean(), wide[anchor_x].mean(), metric
                                ),
                            )
                        )

    cells_df = pd.DataFrame(cells)
    cells_df.to_csv(OUT / "cell_classification.csv", index=False)
    pd.DataFrame(paired).to_csv(OUT / "seed_paired_differences_ci95.csv", index=False)

    counts = (
        cells_df.pivot_table(index="factor", columns="verdict", aggfunc="size", fill_value=0)
        .reindex(FACTORS)
        .reindex(columns=["sharp", "flat", "worse"], fill_value=0)
    )
    counts.loc["TOTAL"] = counts.sum()
    quoted = {
        "scale": (0, 38, 6),
        "rank": (0, 44, 0),
        "bootstrap": (4, 31, 9),
        "ridge": (0, 43, 1),
        "TOTAL": (4, 156, 16),
    }
    counts["quoted_sharp"] = [quoted[i][0] for i in counts.index]
    counts["quoted_flat"] = [quoted[i][1] for i in counts.index]
    counts["quoted_worse"] = [quoted[i][2] for i in counts.index]
    counts.to_csv(OUT / "cell_classification_counts_vs_quoted.csv")
    print("=== reconstructed cell counts vs rebuttal-quoted ===")
    print(counts.to_string())

    # ------------------------------------------------- bootstrap / calib curves
    boot = df[df.factor.isin(["bootfull", "bootstrap"])]
    boot_curve = (
        boot.groupby(["factor", "environment", "factor_value_numeric"])
        .agg(
            n_seeds=("seed", "nunique"),
            per_member_rows=("per_member_calibration_size", "mean"),
            unique_rows=("unique_calibration_rows", "mean"),
            p=("feature_dimension_p", "mean"),
            nominal_n_over_p=("nominal_n_over_p", "mean"),
            unique_n_over_p=("unique_n_over_p", "mean"),
            id_rmse=("id_rmse", "mean"),
            far_nll=("far_nll", "mean"),
            far_auroc=("far_auroc", "mean"),
            far_spearman=("far_spearman", "mean"),
            gram_condition=("gram_condition", "mean"),
        )
        .reset_index()
    )
    boot_curve.to_csv(OUT / "bootstrap_double_descent_curves.csv", index=False)

    calib = df[df.factor == "calib"]
    calib_curve = (
        calib.groupby(["environment", "factor_value_numeric"])
        .agg(
            n_seeds=("seed", "nunique"),
            calibration_pool_size=("calibration_pool_size", "mean"),
            bootstrap_fraction=("bootstrap_fraction", "mean"),
            per_member_rows=("per_member_calibration_size", "mean"),
            unique_rows=("unique_calibration_rows", "mean"),
            p=("feature_dimension_p", "mean"),
            nominal_n_over_p=("nominal_n_over_p", "mean"),
            unique_n_over_p=("unique_n_over_p", "mean"),
            id_rmse=("id_rmse", "mean"),
            far_nll=("far_nll", "mean"),
            far_auroc=("far_auroc", "mean"),
        )
        .reset_index()
    )
    calib_curve.to_csv(OUT / "calibration_pool_curves.csv", index=False)

    # interpolation threshold: worst ID RMSE cell vs n/p
    peak = boot_curve.loc[boot_curve.groupby(["factor", "environment"]).id_rmse.idxmax()]
    peak[
        [
            "factor",
            "environment",
            "factor_value_numeric",
            "per_member_rows",
            "unique_rows",
            "p",
            "nominal_n_over_p",
            "unique_n_over_p",
            "id_rmse",
            "far_nll",
            "far_auroc",
        ]
    ].to_csv(OUT / "interpolation_threshold_peaks.csv", index=False)

    # ------------------------------------------------------------ layer scope
    lay = df[df.factor == "layer"]
    per_env = (
        lay.groupby(["environment", "layer_scope"])[
            ["near_auroc", "mid_auroc", "far_auroc", "id_rmse", "far_nll", "far_spearman"]
        ]
        .agg(["mean", "std", "count"])
    )
    per_env.to_csv(LAYER_OUT / "mujoco_layer_scope_by_environment.csv")
    lay.to_csv(LAYER_OUT / "mujoco_layer_scope_per_seed.csv", index=False)
    macro = (
        lay.groupby(["environment", "layer_scope"])[["near_auroc", "mid_auroc", "far_auroc"]]
        .mean()
        .groupby("layer_scope")
        .mean()
    )
    micro = lay.groupby("layer_scope")[["near_auroc", "mid_auroc", "far_auroc"]].mean()
    print("\n=== layer scope: macro-average over environments ===")
    print(macro.to_string())
    print("=== layer scope: pooled over all env-seed rows ===")
    print(micro.to_string())
    print("quoted: single 0.681 / 0.873 / 0.976 ; multi 0.692 / 0.893 / 0.981")
    pd.concat({"macro_avg_over_envs": macro, "pooled_over_rows": micro}).to_csv(
        LAYER_OUT / "mujoco_layer_scope_aggregates.csv"
    )

    # mid-tier availability
    mid = df.groupby("environment").mid_auroc.apply(lambda s: int(s.notna().sum()))
    mid.rename("rows_with_mid_tier").to_frame().assign(
        has_distinct_mid_tier=lambda d: d.rows_with_mid_tier > 0
    ).to_csv(ROOT / "revision_results/shift_tiers/mid_tier_availability.csv")
    print("\n=== environments WITHOUT a distinct Mid tier ===")
    print(sorted(mid[mid == 0].index.tolist()))


if __name__ == "__main__":
    main()
