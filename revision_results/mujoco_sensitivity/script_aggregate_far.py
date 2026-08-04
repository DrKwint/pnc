"""Aggregate the 11-env Far-OOD sensitivity sweep into per-environment and
across-environment summaries + the reviewer table, for the metric set:
ID RMSE, Far NLL, Far AUROC, Far Spearman (uncertainty↔error rank corr on Far).

Sensitivity = per environment, range(metric) across the factor's values; then we
aggregate the env-level ranges (median/IQR/max/worst-env). We do NOT pool rows
across environments before computing sensitivity.
"""
from __future__ import annotations
import csv
from collections import defaultdict
from pathlib import Path
import numpy as np

OUT = Path(__file__).resolve().parents[1]
RAW = OUT / "aggregates" / "far_sensitivity_raw.csv"
FACTORS = ["scale", "rank", "bootstrap", "calib", "ridge", "layer"]
ENVS = ["Ant-v5", "HalfCheetah-v5", "Hopper-v5", "Humanoid-v5", "Walker2d-v5",
        "HumanoidStandup-v5", "Swimmer-v5", "Reacher-v5", "Pusher-v5",
        "InvertedPendulum-v5", "InvertedDoublePendulum-v5"]
# metric key, label, tolerance-kind
METRICS = [("id_rmse", "ID RMSE", "rel"), ("far_nll", "Far NLL", "abs"),
           ("far_auroc", "Far AUROC", "abs"), ("far_spearman", "Far Spearman", "abs")]
AUROC_TOL = 0.02


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return np.nan


def load():
    rows = list(csv.DictReader(open(RAW)))
    for r in rows:
        for k in ("id_rmse", "id_nll", "far_nll", "far_auroc", "far_spearman", "factor_value_numeric"):
            r[k] = _f(r.get(k))
    return rows


def main():
    rows = load()
    envs = [e for e in ENVS if any(r["environment"] == e for r in rows)]
    seeds = sorted(set(r["seed"] for r in rows))
    # seed-average per (env,factor,value)
    sa = defaultdict(lambda: defaultdict(list))
    order = defaultdict(dict)
    anchor = defaultdict(dict)
    for r in rows:
        key = (r["environment"], r["factor"], r["factor_value"])
        for m, _, _ in METRICS:
            sa[key][m].append(r[m])
        order[(r["environment"], r["factor"])][r["factor_value"]] = r["factor_value_numeric"]
        if r["factor"] == "scale" and "1.0x" in r["factor_value"]:
            for m, _, _ in METRICS:
                anchor[r["environment"]][m] = anchor[r["environment"]].get(m, []) + [r[m]]
    savg = {k: {m: np.nanmean(v) for m, v in d.items()} for k, d in sa.items()}
    anc = {e: {m: np.nanmean(v) for m, v in d.items()} for e, d in anchor.items()}

    # ---- by-environment CSV ----
    with open(OUT / "aggregates" / "far_by_environment.csv", "w", newline="") as f:
        cols = ["environment", "factor", "factor_value"] + [m for m, _, _ in METRICS]
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
        for (env, fac, val), d in sorted(savg.items()):
            w.writerow({"environment": env, "factor": fac, "factor_value": val,
                        **{m: d.get(m, np.nan) for m, _, _ in METRICS}})

    # ---- across-env: per (env,factor) range for each metric ----
    ranges = defaultdict(lambda: defaultdict(dict))  # factor -> metric -> {env: range}
    with open(OUT / "aggregates" / "far_across_envs.csv", "w", newline="") as f:
        cols = ["environment", "factor"] + [f"{m}_range" for m, _, _ in METRICS] + [f"{m}_anchor" for m, _, _ in METRICS]
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
        for env in envs:
            for fac in FACTORS:
                vals = [savg[(env, fac, v)] for v in order.get((env, fac), {})]
                if not vals:
                    continue
                row = {"environment": env, "factor": fac}
                for m, _, kind in METRICS:
                    xs = [d[m] for d in vals if np.isfinite(d.get(m, np.nan))]
                    a = anc.get(env, {}).get(m, np.nan)
                    if not xs:
                        rng = np.nan
                    elif kind == "rel" and np.isfinite(a) and a:
                        rng = (max(xs) - min(xs)) / a
                    else:
                        rng = max(xs) - min(xs)
                    ranges[fac][m][env] = rng
                    row[f"{m}_range"] = rng; row[f"{m}_anchor"] = a
                w.writerow(row)

    # ---- primary reviewer table: median across-env range per factor per metric ----
    hdr = "| Factor | " + " | ".join(f"{lbl} med(range)" for _, lbl, _ in METRICS) + " | Worst env (AUROC) |"
    sep = "|---|" + "---:|" * len(METRICS) + "---|"
    lines = [f"# 11-env Far-OOD sensitivity — median across-env range per factor",
             f"\nEnvironments: {len(envs)} | seeds: {', '.join(seeds)} | metrics: ID RMSE (relative), Far NLL, Far AUROC, Far Spearman.\n",
             hdr, sep]
    tcsv = [["factor"] + [f"{m}_median_range" for m, _, _ in METRICS] + ["far_auroc_worst_range", "far_auroc_worst_env"]]
    for fac in FACTORS:
        cells = []
        for m, _, _ in METRICS:
            vals = [v for v in ranges[fac][m].values() if np.isfinite(v)]
            cells.append(np.median(vals) if vals else np.nan)
        au = ranges[fac]["far_auroc"]
        worst_env = max(au, key=lambda e: (au[e] if np.isfinite(au[e]) else -1)) if au else "-"
        worst = au.get(worst_env, np.nan)
        lines.append("| " + fac + " | " + " | ".join(f"{c:.3f}" if np.isfinite(c) else "NA" for c in cells) + f" | {worst_env} ({worst:.3f}) |")
        tcsv.append([fac] + [f"{c:.4f}" for c in cells] + [f"{worst:.4f}", worst_env])
    (OUT / "tables" / "far_sensitivity_table.md").write_text("\n".join(lines))
    with open(OUT / "tables" / "far_sensitivity_table.csv", "w", newline="") as f:
        csv.writer(f).writerows(tcsv)
    print(f"envs={len(envs)} seeds={seeds}")
    print("\n".join(lines[3:]))


if __name__ == "__main__":
    main()
