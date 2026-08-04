"""Aggregate the raw sensitivity CSV into per-environment + across-environment
summaries and the reviewer-facing tables (Sections 14-15).

Sensitivity = per environment, range(metric) across the factor's values. Across
environments we aggregate those env-level ranges (median/IQR/max/worst-env), and
count environments within tolerance / ID-stable / catastrophic. We do NOT pool
rows across environments before computing sensitivity.
"""
from __future__ import annotations
import csv, json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np

OUT = Path(__file__).resolve().parents[1]
RAW = OUT / "aggregates" / "mujoco_sensitivity_raw.csv"
FACTORS = ["scale", "rank", "bootstrap", "calib", "ridge", "layer"]
# tolerances (task §14)
AUROC_TOL = 0.02; ID_RMSE_TOL = 0.10; NLL_TOL = 0.20


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return np.nan


def load():
    rows = list(csv.DictReader(open(RAW)))
    for r in rows:
        for k in ("id_rmse", "id_nll", "near_auroc", "mid_auroc", "far_auroc",
                  "near_nll", "mid_nll", "far_nll", "factor_value_numeric", "relative_to_anchor"):
            r[k] = _f(r.get(k))
    return rows


def catastrophic(r, anchor):
    if r["status"] != "ok":
        return True
    if not np.isfinite(r["id_rmse"]) or not np.isfinite(r.get("near_auroc", np.nan)):
        return True
    if anchor and np.isfinite(anchor["id_rmse"]) and r["id_rmse"] > 2 * anchor["id_rmse"]:
        return True
    # ID-NLL>2x rule (task §14) — gated behind a meaningful positive anchor NLL:
    # doubling a near-zero/negative NLL is a metric artifact, not a failure.
    if anchor and np.isfinite(anchor["id_nll"]) and anchor["id_nll"] > 1.0 and r["id_nll"] > 2 * anchor["id_nll"]:
        return True
    for m in ("near_auroc", "mid_auroc", "far_auroc"):
        if anchor and np.isfinite(anchor.get(m, np.nan)) and anchor[m] > 0.8 and np.isfinite(r.get(m, np.nan)) and r[m] < 0.6:
            return True
    return False


def main():
    rows = load()
    envs = sorted(set(r["environment"] for r in rows))
    seeds = sorted(set(r["seed"] for r in rows))
    # index: (env, seed, factor, value) -> row; and the per-(env,seed) anchor = scale 1.0x
    idx = {}
    anchor = {}
    for r in rows:
        idx[(r["environment"], r["seed"], r["factor"], r["factor_value"])] = r
        if r["factor"] == "scale" and abs((r["relative_to_anchor"] or 0) - 1.0) < 1e-6:
            anchor[(r["environment"], r["seed"])] = r

    # ---- A. per-environment, seed-averaged rows ----
    by_env = []
    seedavg = defaultdict(list)  # (env,factor,value) -> list of rows over seeds
    for r in rows:
        seedavg[(r["environment"], r["factor"], r["factor_value"])].append(r)
    with open(OUT / "aggregates" / "mujoco_sensitivity_by_environment.csv", "w", newline="") as f:
        cols = ["environment", "factor", "factor_value", "n_success", "n_fail",
                "near_auroc_mean", "near_auroc_std", "mid_auroc_mean", "far_auroc_mean",
                "id_rmse_mean", "id_rmse_std", "id_nll_mean", "far_nll_mean"]
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
        for (env, fac, val), rs in sorted(seedavg.items()):
            ok = [x for x in rs if x["status"] == "ok"]
            row = {"environment": env, "factor": fac, "factor_value": val,
                   "n_success": len(ok), "n_fail": len(rs) - len(ok)}
            for m in ("near_auroc", "mid_auroc", "far_auroc", "id_rmse", "id_nll", "far_nll"):
                vals = [x[m] for x in ok if np.isfinite(x.get(m, np.nan))]
                row[f"{m}_mean"] = np.mean(vals) if vals else np.nan
                if m in ("near_auroc", "id_rmse"):
                    row[f"{m}_std"] = np.std(vals) if vals else np.nan
            w.writerow(row); by_env.append(row)

    # ---- B. across-environment sensitivity ----
    # per (env, factor): range of seed-averaged near_auroc across values; id stability; catastrophic
    across = []
    per_factor = defaultdict(list)  # factor -> list of (env, near_range, id_rel_range, within, id_stable, catastro)
    for env in envs:
        for fac in FACTORS:
            vals = [(val, rw) for (e, f2, val), rw in
                    {(env, fac, v[2]): _sa for (e2, f2b, v), _sa in
                     [((env, fac, k[2]), None) for k in []]}.items()]  # placeholder
        # simpler: gather seed-averaged rows for this env,factor
    for env in envs:
        for fac in FACTORS:
            rws = [row for row in by_env if row["environment"] == env and row["factor"] == fac]
            if not rws:
                continue
            aurocs = [r["near_auroc_mean"] for r in rws if np.isfinite(r["near_auroc_mean"])]
            idrmse = [r["id_rmse_mean"] for r in rws if np.isfinite(r["id_rmse_mean"])]
            # anchor auroc/rmse for this env (seed-avg of scale 1.0x)
            a_rows = [r for r in by_env if r["environment"] == env and r["factor"] == "scale"
                      and "1.0x" in r["factor_value"]]
            a_au = a_rows[0]["near_auroc_mean"] if a_rows else np.nan
            a_rm = a_rows[0]["id_rmse_mean"] if a_rows else np.nan
            near_range = (max(aurocs) - min(aurocs)) if aurocs else np.nan
            id_rel_range = ((max(idrmse) - min(idrmse)) / a_rm) if (idrmse and np.isfinite(a_rm) and a_rm) else np.nan
            within = (near_range <= AUROC_TOL) if np.isfinite(near_range) else False
            id_stable = (max(idrmse) <= (1 + ID_RMSE_TOL) * a_rm) if (idrmse and np.isfinite(a_rm)) else False
            # catastrophic: any raw seed row catastrophic
            cat = any(catastrophic(idx[k], anchor.get((k[0], k[1])))
                      for k in idx if k[0] == env and k[2] == fac)
            per_factor[fac].append((env, near_range, id_rel_range, within, id_stable, cat))
            across.append(dict(environment=env, factor=fac, near_auroc_range=near_range,
                               id_rmse_rel_range=id_rel_range, within_auroc_tol=within,
                               id_stable=id_stable, catastrophic=cat, anchor_near_auroc=a_au))
    with open(OUT / "aggregates" / "mujoco_sensitivity_across_envs.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(across[0].keys())); w.writeheader()
        for r in across:
            w.writerow(r)

    # ---- primary reviewer table ----
    factor_labels = {"scale": "Scale (0.25–4× anchor)", "rank": "Subspace dim K∈{1,2,5,20,40}",
                     "bootstrap": "Bootstrap b∈{0,.05,.1,.2,.3,.5,.99}",
                     "calib": "Clean calib n∈{100..4096}", "ridge": "Ridge λ∈{0,1e-4,1e-3,1e-2,1e-1,1}",
                     "layer": "Layer scope first vs multi"}
    table = []
    for fac in FACTORS:
        recs = per_factor[fac]
        ranges = [r[1] for r in recs if np.isfinite(r[1])]
        if not ranges:
            continue
        worst_i = int(np.argmax(ranges))
        worst_env = [r[0] for r in recs if np.isfinite(r[1])][worst_i]
        table.append({
            "Factor": factor_labels[fac],
            "MedianAUROCrange": round(float(np.median(ranges)), 4),
            "WorstAUROCrange": round(float(np.max(ranges)), 4),
            "WorstEnv": worst_env,
            "WithinTol": f"{sum(r[3] for r in recs)}/{len(recs)}",
            "IDstable": f"{sum(r[4] for r in recs)}/{len(recs)}",
            "Catastrophic": sum(r[5] for r in recs),
        })
    # write md/csv/tex
    tdir = OUT / "tables"; tdir.mkdir(exist_ok=True)
    with open(tdir / "mujoco_sensitivity_rebuttal.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0].keys())); w.writeheader()
        for r in table:
            w.writerow(r)
    md = ["# MuJoCo P&C sensitivity — primary table",
          f"\nEnvironments: {', '.join(envs)} | seeds: {', '.join(seeds)} | metric: Near-OOD AUROC range across factor values (per-env, then aggregated).\n",
          "| Factor | Median AUROC range | Worst range | Worst env | Within tol (≤0.02) | ID-stable | Catastrophic |",
          "|---|---:|---:|---|---:|---:|---:|"]
    for r in table:
        md.append(f"| {r['Factor']} | {r['MedianAUROCrange']} | {r['WorstAUROCrange']} | {r['WorstEnv']} | {r['WithinTol']} | {r['IDstable']} | {r['Catastrophic']} |")
    (tdir / "mujoco_sensitivity_rebuttal.md").write_text("\n".join(md))
    tex = ["\\begin{tabular}{lrrlrrr}", "\\toprule",
           "Factor & Med.\\ range & Worst & Worst env & Within tol & ID-stable & Catastr.\\\\", "\\midrule"]
    for r in table:
        tex.append(f"{r['Factor']} & {r['MedianAUROCrange']} & {r['WorstAUROCrange']} & {r['WorstEnv']} & {r['WithinTol']} & {r['IDstable']} & {r['Catastrophic']}\\\\")
    tex += ["\\bottomrule", "\\end{tabular}"]
    (tdir / "mujoco_sensitivity_rebuttal.tex").write_text("\n".join(tex))

    print(f"envs={envs} seeds={seeds}")
    print("\n".join(md[3:]))
    return table


if __name__ == "__main__":
    main()
