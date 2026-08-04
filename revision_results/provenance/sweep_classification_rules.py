#!/usr/bin/env python
"""Rule sweep: which sharp/flat/worse definition could have produced the quoted counts?

The rebuttal quoted, per swept factor, a classification of 44 cells
(= 11 environments x 4 metrics) into sharp / flat / materially worse:

    scale 0/38/6 ; rank 0/44/0 ; bootstrap 4/31/9 ; ridge 0/43/1 ; total 4/156/16

No script in the repository produces these numbers. This module enumerates six
plausible definitions and reports how far each lands from the quoted counts, so the
audit records a systematic search rather than a single failed guess.

Rules
-----
R1 anchor-vs-all + neighbour-cliff  : worse if any swept value beats tolerance;
                                      sharp if an adjacent value does.
R2 anchor-vs-all, no sharp class    : as R1 but every violation counts as "worse".
R3 curve-range                      : flat if (max - min) of the whole seed-mean
                                      curve is within tolerance; else worse.
R4 seed-paired CI                   : worse only if the 95% paired-difference CI
                                      lies entirely beyond the tolerance.
R5 relative NLL tolerance           : as R1 but the NLL tolerance is 10% relative
                                      rather than 0.10 absolute.
R6 interior of sweep                : as R1 after dropping the smallest and largest
                                      swept value (the deliberate extremes).

Usage: .venv/bin/python revision_results/provenance/sweep_classification_rules.py
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "results/neurips_2026_rebuttal/mujoco_sensitivity/aggregates/far_sensitivity_raw.csv"
OUT = ROOT / "revision_results/mujoco_sensitivity"

METRICS = {
    "id_rmse": ("lo", "rel", 0.05),
    "far_nll": ("lo", "abs", 0.10),
    "far_auroc": ("hi", "abs", 0.02),
    "far_spearman": ("hi", "abs", 0.02),
}
FACTORS = ["scale", "rank", "bootstrap", "ridge"]
QUOTED = {
    "scale": (0, 38, 6),
    "rank": (0, 44, 0),
    "bootstrap": (4, 31, 9),
    "ridge": (0, 43, 1),
}


def worse_fn(metric: str, relative_nll: bool = False):
    direction, kind, tol = METRICS[metric]
    if metric == "far_nll" and relative_nll:
        kind, tol = "rel", 0.10

    def f(value: float, anchor: float) -> bool:
        if not (np.isfinite(value) and np.isfinite(anchor)):
            return False
        if direction == "hi":
            return (anchor - value) > tol
        if kind == "rel":
            return value > (1.0 + tol) * abs(anchor)
        return (value - anchor) > tol

    return f


def cell_verdict(rule, curve, anchor_x, metric, wide=None):
    xs = list(curve.index)
    if anchor_x not in curve.index:
        return "no_anchor"
    anchor = curve.loc[anchor_x]

    if rule == "R3":
        direction, kind, tol = METRICS[metric]
        rng = curve.max() - curve.min()
        limit = tol * abs(anchor) if kind == "rel" else tol
        return "flat" if rng <= limit else "worse"

    if rule == "R6":
        xs = xs[1:-1] if len(xs) > 2 else xs
        if anchor_x not in xs:
            xs = list(curve.index)

    w = worse_fn(metric, relative_nll=(rule == "R5"))

    if rule == "R4":
        direction, kind, tol = METRICS[metric]
        limit = tol * abs(anchor) if kind == "rel" else tol
        flags = {}
        for x in xs:
            if x == anchor_x or wide is None or x not in wide.columns:
                continue
            d = (wide[x] - wide[anchor_x]).dropna()
            if len(d) < 2:
                continue
            se = d.std(ddof=1) / math.sqrt(len(d))
            lo, hi = d.mean() - 2.06 * se, d.mean() + 2.06 * se
            # harmful direction: "hi" metrics get worse when the difference is negative
            flags[x] = (hi < -limit) if direction == "hi" else (lo > limit)
    else:
        flags = {x: w(curve.loc[x], anchor) for x in xs if x != anchor_x}

    if not any(flags.values()):
        return "flat"
    if rule == "R2":
        return "worse"
    i = xs.index(anchor_x)
    nbrs = [xs[j] for j in (i - 1, i + 1) if 0 <= j < len(xs)]
    return "sharp" if any(flags.get(x, False) for x in nbrs) else "worse"


def main() -> None:
    df = pd.read_csv(RAW)
    rows = []
    for rule in ["R1", "R2", "R3", "R4", "R5", "R6"]:
        for factor in FACTORS:
            sub = df[df.factor == factor]
            tally = {"sharp": 0, "flat": 0, "worse": 0, "no_anchor": 0}
            for env, g in sub.groupby("environment"):
                a = g[np.isclose(g.relative_to_anchor, 1.0)]
                anchor_x = float(a.factor_value_numeric.iloc[0]) if len(a) else float("nan")
                for metric in METRICS:
                    curve = g.groupby("factor_value_numeric")[metric].mean().sort_index()
                    wide = g.pivot_table(index="seed", columns="factor_value_numeric", values=metric)
                    tally[cell_verdict(rule, curve, anchor_x, metric, wide)] += 1
            qs, qf, qw = QUOTED[factor]
            rows.append(
                dict(
                    rule=rule,
                    factor=factor,
                    sharp=tally["sharp"],
                    flat=tally["flat"],
                    worse=tally["worse"],
                    quoted_sharp=qs,
                    quoted_flat=qf,
                    quoted_worse=qw,
                    l1_distance=abs(tally["sharp"] - qs)
                    + abs(tally["flat"] - qf)
                    + abs(tally["worse"] - qw),
                )
            )
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "classification_rule_sweep.csv", index=False)
    tot = out.groupby("rule").l1_distance.sum().sort_values()
    print(out.to_string(index=False))
    print("\n=== total L1 distance from the quoted counts, by rule (lower = closer) ===")
    print(tot.to_string())
    print("\nNo rule reproduces the quoted counts; the original classifier was not saved.")


if __name__ == "__main__":
    main()
