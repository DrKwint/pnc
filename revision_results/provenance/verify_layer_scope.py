#!/usr/bin/env python
"""Locate and verify the single-block vs multi-block AUROC averages quoted in the rebuttal.

Quoted
------
    single block : Near 0.681 , Mid 0.873 , Far 0.976
    multi  block : Near 0.692 , Mid 0.893 , Far 0.981

Two candidate sources exist in the tree and they disagree:

A) results/neurips_2026_rebuttal/priority1/sensitivity_mujoco.csv, rows sweep=='layer'
   HalfCheetah-v5 ONLY, seeds {0,10,42}, anchor = the earlier priority1 canonical
   config (bf=0.1, lambda=0, subset_size=10000).

B) results/neurips_2026_rebuttal/mujoco_sensitivity/aggregates/far_sensitivity_raw.csv,
   rows factor=='layer'  -- 11 environments x 27 seeds on the corrected per-env anchors.

This script computes both. (A) reproduces all six quoted numbers to three decimals;
(B) does not, and is the larger and later experiment. The rebuttal's phrase
"average AUROC" therefore denotes a single-environment three-seed average, not an
average over environments.

Usage: .venv/bin/python revision_results/provenance/verify_layer_scope.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "revision_results/layer_scope"

QUOTED = {
    "single block": {"near": 0.681, "mid": 0.873, "far": 0.976},
    "multi block": {"near": 0.692, "mid": 0.893, "far": 0.981},
}


def source_a() -> pd.DataFrame:
    src = ROOT / "results/neurips_2026_rebuttal/priority1/sensitivity_mujoco.csv"
    d = pd.read_csv(src)
    lay = d[d.sweep == "layer"].copy()
    # `value` distinguishes the block scope; the `layer_scope` column is stale for these rows.
    lay["block_scope"] = lay.value.map(
        lambda v: "single block" if str(v).startswith("single") else "multi block"
    )
    lay.to_csv(OUT / "sourceA_priority1_halfcheetah_layer_rows.csv", index=False)
    return lay.groupby("block_scope")[
        ["auroc_ood_near", "auroc_ood_mid", "auroc_ood_far"]
    ].agg(["mean", "std", "count"])


def source_b() -> pd.DataFrame:
    src = ROOT / "results/neurips_2026_rebuttal/mujoco_sensitivity/aggregates/far_sensitivity_raw.csv"
    d = pd.read_csv(src)
    lay = d[d.factor == "layer"].copy()
    lay["block_scope"] = lay.layer_scope.map({"first": "single block", "multi": "multi block"})
    per_env = lay.groupby(["environment", "block_scope"])[
        ["near_auroc", "mid_auroc", "far_auroc"]
    ].mean()
    per_env.to_csv(OUT / "sourceB_11env_27seed_layer_by_environment.csv")
    return per_env.groupby("block_scope").mean()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    a, b = source_a(), source_b()

    rows = []
    for scope, q in QUOTED.items():
        for tier, col_a, col_b in [
            ("near", "auroc_ood_near", "near_auroc"),
            ("mid", "auroc_ood_mid", "mid_auroc"),
            ("far", "auroc_ood_far", "far_auroc"),
        ]:
            va = a.loc[scope, (col_a, "mean")]
            vb = b.loc[scope, col_b]
            rows.append(
                dict(
                    block_scope=scope,
                    tier=tier,
                    quoted=q[tier],
                    sourceA_halfcheetah_3seed=round(float(va), 4),
                    sourceA_matches=abs(round(float(va), 3) - q[tier]) < 1e-9,
                    sourceB_11env_27seed=round(float(vb), 4),
                    sourceB_matches=abs(round(float(vb), 3) - q[tier]) < 1e-9,
                )
            )
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "layer_scope_quoted_value_checks.csv", index=False)
    print(out.to_string(index=False))
    print(f"\nsource A n per cell: {int(a.loc['single block', ('auroc_ood_near', 'count')])} "
          "(HalfCheetah-v5, seeds 0/10/42)")
    print("source A reproduces all six quoted values; source B (11 envs x 27 seeds) does not.")


if __name__ == "__main__":
    main()
