#!/usr/bin/env python3
"""Priority 5, item 12 — expose the OMITTED Near/Mid AUROC for MuJoCo.

The submitted table reports only Far AUROC. Every cached result JSON already contains
auroc_ood_near / auroc_ood_mid. This script extracts the full Near/Mid/Far NLL AND AUROC:

(a) direct P&C canonical extraction (config reproduced exactly in Phase 0), per-seed size
    selection by nll_val, mean+/-std over seeds; and
(b) all-methods table via the paper's own selection machinery (json_to_tex_table.method_rows),
    dumping every tier metric it computes but does not print.

Outputs:
  results/neurips_2026_rebuttal/priority5_full_mujoco_tables.csv        (all methods, part b)
  results/neurips_2026_rebuttal/priority5_pnc_tiers.csv                 (P&C canonical, part a)
CPU-only (reads cached JSONs); safe to run alongside GPU jobs.
"""
import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))
os.chdir(_REPO)

import csv
import json
import math
import statistics as st

ENVS = ["Ant-v5", "HalfCheetah-v5", "Hopper-v5"]
SEEDS = [0, 10, 42, 100, 200]
TIER_METRICS = ["rmse_id", "nll_id",
                "nll_ood_near", "nll_ood_mid", "nll_ood_far",
                "auroc_ood_near", "auroc_ood_mid", "auroc_ood_far"]
OUT = Path("results/neurips_2026_rebuttal")

PNC_CANON = ("pjsvd_multi_least_squares_random_projected_residual_prob_bf0.1_"
             "k20_n50_ps5.0-10.0-20.0-50.0_h200-200-200-200_act-relu_seed{seed}.json")


def mean_std(xs):
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    if not xs:
        return (float("nan"), float("nan"))
    return (st.mean(xs), st.pstdev(xs) if len(xs) > 1 else 0.0)


# ---- Part (a): P&C canonical, per-seed nll_val size selection --------------------
def extract_pnc_canonical():
    rows = []
    per_env_seed = {}
    for env in ENVS:
        for seed in SEEDS:
            p = Path("results") / env / PNC_CANON.format(seed=seed)
            if not p.exists():
                continue
            m = json.loads(p.read_text())
            sel = min(m.keys(), key=lambda k: m[k]["nll_val"])
            per_env_seed.setdefault(env, {})[seed] = (sel, m[sel])
    # per-seed rows + aggregate
    for env in ENVS:
        d = per_env_seed.get(env, {})
        for seed, (sel, mv) in sorted(d.items()):
            row = {"env": env, "seed": seed, "selected_size": sel}
            for f in TIER_METRICS:
                row[f] = mv.get(f)
            rows.append(row)
        # aggregate
        agg = {"env": env, "seed": "mean+/-std", "selected_size": ""}
        for f in TIER_METRICS:
            mu, sd = mean_std([d[s][1].get(f) for s in d])
            agg[f] = f"{mu:.4f}+/-{sd:.4f}"
        rows.append(agg)
    return rows


# ---- Part (b): all methods via paper machinery ----------------------------------
def extract_all_methods():
    from pnc_core.json_to_tex_table import load_env_results, method_rows
    # Collapse hyperparameter knobs to one row per method-family, but KEEP the
    # method-defining tokens (family, scope, mode, vcal, prob) separate so
    # PnC-Low vs PnC-Random remain distinct rows (as in the submitted table).
    max_over = {"k", "n", "act", "T", "full", "grid", "ens", "lam",
                "dr", "lreg", "bf", "sws", "prel2"}
    rows = []
    for env in ENVS:
        env_dir = Path("results") / env
        if not env_dir.is_dir():
            continue
        groups = load_env_results(env_dir, "gym", seed_filter=set(SEEDS))
        for label, metrics in method_rows(groups, "gym", max_over):
            row = {"env": env, "method": label,
                   "n_seeds": len(metrics.get("nll_id", metrics.get("rmse_id", [])))}
            for f in TIER_METRICS:
                mu, sd = mean_std(metrics.get(f, []))
                row[f] = f"{mu:.4f}+/-{sd:.4f}" if not math.isnan(mu) else ""
            rows.append(row)
    return rows


def write_csv(path, rows, cols):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main():
    a = extract_pnc_canonical()
    write_csv(OUT / "priority5_pnc_tiers.csv", a,
              ["env", "seed", "selected_size"] + TIER_METRICS)
    print(f"[part a] P&C canonical tiers -> priority5_pnc_tiers.csv ({len(a)} rows)")
    for r in a:
        if r["seed"] == "mean+/-std":
            print(f"  {r['env']:16s} near_auroc={r['auroc_ood_near']}  "
                  f"mid_auroc={r['auroc_ood_mid']}  far_auroc={r['auroc_ood_far']}")

    try:
        b = extract_all_methods()
        write_csv(OUT / "priority5_full_mujoco_tables.csv", b,
                  ["env", "method", "n_seeds"] + TIER_METRICS)
        print(f"\n[part b] all-methods full-tier table -> priority5_full_mujoco_tables.csv ({len(b)} rows)")
        for env in ENVS:
            print(f"\n### {env} (method: near_auroc / mid_auroc / far_auroc) ###")
            for r in [x for x in b if x["env"] == env]:
                print(f"  {r['method'][:40]:40s} {r['auroc_ood_near']} / {r['auroc_ood_mid']} / {r['auroc_ood_far']}")
    except Exception as e:
        print(f"\n[part b] FAILED ({type(e).__name__}: {e}); part (a) is the primary deliverable.")


if __name__ == "__main__":
    main()
