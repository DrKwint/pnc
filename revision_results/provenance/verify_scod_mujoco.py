#!/usr/bin/env python
"""Normalize and verify the MuJoCo SCOD-vs-P&C comparison quoted in the rebuttal.

Sources
-------
SCOD : results/posthoc_mujoco/scod/predictions/<env>/<seed>/_metrics.json
       (11 environments x seeds {0, 10, 200} = 33 runs, 0 failures)
P&C  : results/neurips_2026_rebuttal/mujoco_sensitivity/aggregates/far_sensitivity_raw.csv
       restricted to the anchor rows (relative_to_anchor == 1.0), i.e. each
       environment's submitted configuration, averaged over the 27 available seeds.

Quoted in the rebuttal
----------------------
    mean Far AUROC : P&C 0.922 , SCOD 0.512
    Ant-v5      P&C 0.998 , SCOD 0.110
    Hopper-v5   P&C 0.981 , SCOD 0.028
    Humanoid-v5 P&C 0.344 , SCOD 0.267
    Pusher-v5   P&C 1.000 , SCOD 0.995
    Far/ID SCOD mean-score ratio : Hopper ~0.32 , Ant ~0.50
    P&C exceeds SCOD on 11/11 environments

Caveat this script makes explicit
---------------------------------
The two columns are NOT seed-matched: SCOD is 3 seeds {0,10,200}, P&C is 27 seeds.
A seed-matched restriction of P&C to {0,10,200} is emitted alongside so the
comparison can be reported either way.

Usage: .venv/bin/python revision_results/provenance/verify_scod_mujoco.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SCOD_DIR = ROOT / "results/posthoc_mujoco/scod/predictions"
PNC_RAW = ROOT / "results/neurips_2026_rebuttal/mujoco_sensitivity/aggregates/far_sensitivity_raw.csv"
OUT = ROOT / "revision_results/scod"

QUOTED_PAIRS = {
    "Ant-v5": (0.998, 0.110),
    "Hopper-v5": (0.981, 0.028),
    "Humanoid-v5": (0.344, 0.267),
    "Pusher-v5": (1.000, 0.995),
}


def load_scod() -> pd.DataFrame:
    rows = []
    for mf in sorted(SCOD_DIR.glob("*/*/_metrics.json")):
        d = json.load(open(mf))
        sel, m = d["selected"], d["metrics"]
        idm = m.get("id_eval", {})
        rows.append(
            dict(
                environment=d["environment"],
                seed=d["model_seed"],
                id_rmse=d.get("id_rmse"),
                id_nll_gaussian=idm.get("nll_gaussian"),
                id_mean_uncertainty=idm.get("mean_uncertainty"),
                near_auroc=m.get("ood_near", {}).get("auroc_native"),
                mid_auroc=m.get("ood_mid", {}).get("auroc_native"),
                far_auroc=m.get("ood_far", {}).get("auroc_native"),
                far_spearman=m.get("ood_far", {}).get("spearman_native"),
                near_nll_gaussian=m.get("ood_near", {}).get("nll_gaussian"),
                mid_nll_gaussian=m.get("ood_mid", {}).get("nll_gaussian"),
                far_nll_gaussian=m.get("ood_far", {}).get("nll_gaussian"),
                far_mean_uncertainty=m.get("ood_far", {}).get("mean_uncertainty"),
                near_mean_uncertainty=m.get("ood_near", {}).get("mean_uncertainty"),
                selected_num_eigs=sel.get("selected_num_eigs"),
                selected_Meps=sel.get("selected_Meps"),
                selected_Meps_factor=sel.get("selected_Meps_factor"),
                selected_alpha=sel.get("selected_alpha"),
                num_train_examples=sel.get("num_train_examples"),
                num_samples=sel.get("num_samples"),
                sketch_seed=sel.get("sketch_seed"),
                used_ood_for_selection=sel.get("used_ood_for_selection"),
                selection_metric=sel.get("selection_metric"),
                score_subsampled=sel.get("score_subsampled"),
                git_tag=sel.get("git_tag"),
                source_file=str(mf.relative_to(ROOT)),
            )
        )
    df = pd.DataFrame(rows)
    df["far_over_id_score_ratio"] = df.far_mean_uncertainty / df.id_mean_uncertainty
    return df


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    scod = load_scod()
    scod.to_csv(OUT / "scod_mujoco_per_seed.csv", index=False)

    pnc = pd.read_csv(PNC_RAW)
    anchor = pnc[np.isclose(pnc.relative_to_anchor, 1.0)]

    def summarise(d, cols):
        g = d.groupby("environment")[cols]
        out = g.mean().add_suffix("_mean").join(g.std().add_suffix("_std"))
        out["n_seeds"] = d.groupby("environment").seed.nunique()
        return out

    s = summarise(scod, ["far_auroc", "near_auroc", "mid_auroc", "far_spearman",
                         "id_rmse", "far_over_id_score_ratio"]).add_prefix("scod_")
    p_all = summarise(anchor, ["far_auroc", "near_auroc", "mid_auroc", "far_spearman",
                               "id_rmse"]).add_prefix("pnc_")
    matched = anchor[anchor.seed.isin([0, 10, 200])]
    p_m = summarise(matched, ["far_auroc"]).add_prefix("pnc_seedmatched_")

    tab = s.join(p_all).join(p_m).reset_index()
    tab["pnc_minus_scod_far_auroc"] = tab.pnc_far_auroc_mean - tab.scod_far_auroc_mean
    tab["pnc_wins_far"] = tab.pnc_minus_scod_far_auroc > 0
    tab = tab.sort_values("scod_far_auroc_mean", ascending=False)
    tab.to_csv(OUT / "scod_vs_pnc_mujoco_by_environment.csv", index=False)

    print("=== per-environment Far AUROC ===")
    print(
        tab[
            [
                "environment",
                "pnc_far_auroc_mean",
                "pnc_n_seeds",
                "pnc_seedmatched_far_auroc_mean",
                "scod_far_auroc_mean",
                "scod_n_seeds",
                "scod_far_over_id_score_ratio_mean",
                "pnc_wins_far",
            ]
        ].to_string(index=False, float_format=lambda v: f"{v:.4f}")
    )

    checks = []
    checks.append(("mean Far AUROC P&C (27 seeds)", tab.pnc_far_auroc_mean.mean(), 0.922))
    checks.append(
        ("mean Far AUROC P&C (seed-matched 0/10/200)", tab.pnc_seedmatched_far_auroc_mean.mean(), 0.922)
    )
    checks.append(("mean Far AUROC SCOD (3 seeds)", tab.scod_far_auroc_mean.mean(), 0.512))
    for env, (qp, qs) in QUOTED_PAIRS.items():
        r = tab[tab.environment == env].iloc[0]
        checks.append((f"{env} P&C Far AUROC", r.pnc_far_auroc_mean, qp))
        checks.append((f"{env} SCOD Far AUROC", r.scod_far_auroc_mean, qs))
    for env, q in (("Hopper-v5", 0.32), ("Ant-v5", 0.50)):
        r = tab[tab.environment == env].iloc[0]
        checks.append((f"{env} SCOD Far/ID score ratio", r.scod_far_over_id_score_ratio_mean, q))
    checks.append(("environments where P&C > SCOD (of 11)", float(tab.pnc_wins_far.sum()), 11.0))

    cdf = pd.DataFrame(checks, columns=["quantity", "recomputed", "rebuttal_quoted"])
    cdf["abs_diff"] = (cdf.recomputed - cdf.rebuttal_quoted).abs()
    cdf["matches_at_quoted_precision"] = cdf.abs_diff < 0.001
    cdf.to_csv(OUT / "scod_quoted_value_checks.csv", index=False)
    print("\n=== quoted-value checks ===")
    print(cdf.to_string(index=False, float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    main()
