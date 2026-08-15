"""Revision closure Parts C.5, C.6 and 5: MuJoCo zero- vs original-centred ridge.

Reads the historical zero-centred runs and the Part C original-centred reruns, which
differ in exactly one flag and therefore share base checkpoints, splits, seeds, K, M,
bootstrap fraction, target layers, the perturbation-size grid and the evaluation data.

Selection (C.5) reuses the manuscript's own rule: per (env, seed) pick the perturbation
size with the lowest ID-validation NLL, falling back to ID NLL. That rule is ID-only, so
applying it independently to each centre is not a shifted-data comparison. The script
reports, per environment, whether the original-centred solve selected a different size —
i.e. whether any hyperparameter needed reselection.

Outputs:
  mujoco_original_centered_per_env.csv    per (env, seed, centre) selected-config metrics
  mujoco_original_centered_ranks.csv      cross-environment mean ranks and win counts
  mujoco_original_centered_summary.md     narrative summary + paired differences
  mujoco_center_paired.csv                per-metric paired differences with bootstrap CIs
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

RESULTS = Path("results")
OUT = Path("2026neurips_rebuttal_results/revision_experiment_closure")
ENVS = ["Ant-v5", "HalfCheetah-v5", "Hopper-v5", "Humanoid-v5"]
# The manuscript's headline gym table covers these three. Humanoid is reported
# separately: P&C sits at or below chance there (Far AUROC 0.22-0.47) under BOTH
# centres, so its large swings say nothing about the ridge centre and would
# dominate any pooled mean.
HEADLINE_ENVS = ["Ant-v5", "HalfCheetah-v5", "Hopper-v5"]
SEEDS = [0, 10, 42, 100, 200]
METRICS = [("rmse_id", False), ("nll_id", False), ("nll_ood_near", False),
           ("nll_ood_mid", False), ("nll_ood_far", False), ("auroc_ood_far", True),
           ("auroc_ood_near", True), ("auroc_ood_mid", True),
           ("spearman_ood_far", True)]
BOOT_N, BOOT_SEED = 10000, 20260815

# Humanoid ran with --compute-l2 False (the uncorrected-l2 path OOMs there), which
# inserts a _nol2 token into the filename. Both spellings are tried so Humanoid is not
# silently dropped from the pairing.
STEMS = [("pjsvd_multi_least_squares_random_projected_residual{l2}_prob_lreg0.0001"
          "{tok}_bf0.1_k20_n50_ps5.0-10.0-20.0-50.0_h200-200-200-200_act-relu_seed{s}.json")
         .replace("{l2}", v) for v in ("", "_nol2")]


def _val_nll(m: dict) -> float:
    for k in ("nll_val", "nll_id"):
        v = m.get(k)
        if isinstance(v, (int, float)) and not math.isnan(float(v)):
            return float(v)
    return float("inf")


def load(env: str, seed: int, centre: str) -> tuple[str, dict] | None:
    tok = "_ridgeorig" if centre == "original" else ""
    for stem in STEMS:
        p = RESULTS / env / stem.format(tok=tok, s=seed)
        if p.exists():
            break
    else:
        return None
    d = json.load(open(p))
    best = min(d, key=lambda k: _val_nll(d[k]))       # ID-only selection rule
    return best, d[best]


def collect() -> list[dict]:
    rows = []
    for env in ENVS:
        for seed in SEEDS:
            for centre in ("zero", "original"):
                got = load(env, seed, centre)
                if got is None:
                    continue
                size, m = got
                r = {"env": env, "seed": seed, "ridge_center": centre,
                     "selected_size": float(size), "val_nll": _val_nll(m)}
                for k, _ in METRICS:
                    v = m.get(k)
                    r[k] = float(v) if isinstance(v, (int, float)) else ""
                rows.append(r)
    return rows


def paired(rows: list[dict], envs: list[str] | None = None) -> list[dict]:
    envs = envs or ENVS
    idx = {(r["env"], r["seed"], r["ridge_center"]): r for r in rows}
    pairs = [(idx[(e, s, "zero")], idx[(e, s, "original")])
             for e in envs for s in SEEDS
             if (e, s, "zero") in idx and (e, s, "original") in idx]
    rng = np.random.default_rng(BOOT_SEED)
    out = []
    for k, higher in METRICS:
        d = np.array([o[k] - z[k] for z, o in pairs
                      if isinstance(z.get(k), float) and isinstance(o.get(k), float)],
                     dtype=float)
        d = d[np.isfinite(d)]
        if d.size == 0:
            continue
        boot = np.array([rng.choice(d, d.size, replace=True).mean()
                         for _ in range(BOOT_N)])
        out.append({"metric": k, "n_pairs": int(d.size),
                    "mean_diff": float(d.mean()), "median_diff": float(np.median(d)),
                    "max_abs_diff": float(np.abs(d).max()),
                    "ci95_lo": float(np.percentile(boot, 2.5)),
                    "ci95_hi": float(np.percentile(boot, 97.5)),
                    "excludes_zero": bool(np.percentile(boot, 2.5) > 0
                                          or np.percentile(boot, 97.5) < 0),
                    "higher_is_better": higher,
                    "n_replicates": BOOT_N, "seed": BOOT_SEED,
                    "direction": "original minus zero",
                    "env_scope": "headline3" if envs == HEADLINE_ENVS else "all4"})
    return out


def ranks(rows: list[dict]) -> list[dict]:
    """Cross-environment mean rank and win count of each centre, per metric."""
    out = []
    for k, higher in METRICS:
        per_centre = defaultdict(list)
        wins = defaultdict(int)
        n = 0
        for env in ENVS:
            for seed in SEEDS:
                vals = {c: next((r[k] for r in rows if r["env"] == env
                                 and r["seed"] == seed and r["ridge_center"] == c
                                 and isinstance(r.get(k), float)), None)
                        for c in ("zero", "original")}
                if any(v is None for v in vals.values()):
                    continue
                n += 1
                order = sorted(vals, key=lambda c: -vals[c] if higher else vals[c])
                for rank, c in enumerate(order, 1):
                    per_centre[c].append(rank)
                wins[order[0]] += 1
        if not n:
            continue
        for c in ("zero", "original"):
            out.append({"metric": k, "ridge_center": c, "n_cells": n,
                        "mean_rank": float(np.mean(per_centre[c])),
                        "wins": wins[c], "higher_is_better": higher})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    o = Path(a.out)
    o.mkdir(parents=True, exist_ok=True)

    rows = collect()
    have = {(r["env"], r["seed"], r["ridge_center"]) for r in rows}
    print(f"loaded {len(rows)} selected-config rows "
          f"({len([1 for x in have if x[2]=='original'])} original, "
          f"{len([1 for x in have if x[2]=='zero'])} zero)")

    cols = ["env", "seed", "ridge_center", "selected_size", "val_nll"] + \
           [k for k, _ in METRICS]
    with (o / "mujoco_original_centered_per_env.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, cols)
        w.writeheader()
        w.writerows(rows)

    rk = ranks(rows)
    with (o / "mujoco_original_centered_ranks.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, list(rk[0]))
        w.writeheader()
        w.writerows(rk)

    pr = paired(rows)
    pr_h = paired(rows, HEADLINE_ENVS)
    with (o / "mujoco_center_paired.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, list(pr[0]))
        w.writeheader()
        w.writerows(pr_h + pr)

    # ---- reselection check (C.5) ----
    resel = []
    for env in ENVS:
        for seed in SEEDS:
            z = next((r for r in rows if r["env"] == env and r["seed"] == seed
                      and r["ridge_center"] == "zero"), None)
            ov = next((r for r in rows if r["env"] == env and r["seed"] == seed
                       and r["ridge_center"] == "original"), None)
            if z and ov:
                resel.append({"env": env, "seed": seed,
                              "size_zero": z["selected_size"],
                              "size_original": ov["selected_size"],
                              "changed": z["selected_size"] != ov["selected_size"]})
    n_ch = sum(r["changed"] for r in resel)

    L = ["# MuJoCo: original-centred vs zero-centred ridge", "",
         "Same base checkpoints, splits, seeds, K=20, M=50, bootstrap 0.1, target layers, "
         "perturbation-size grid and evaluation data. The only change is the ridge centre.",
         "", f"Pairs available: **{len(resel)}** of {len(ENVS)*len(SEEDS)} "
         f"(env x seed).", "",
         "## Hyperparameter reselection (C.5)", "",
         f"Perturbation size is selected per (env, seed) by lowest ID-validation NLL — the "
         f"manuscript's own ID-only rule, applied independently under each centre. "
         f"**{n_ch} of {len(resel)}** cells selected a different size under the "
         f"original-centred solve.", "",
         "| env | seed | size (zero) | size (original) | changed |", "|---|---|---|---|---|"]
    for r in resel:
        L.append(f"| {r['env']} | {r['seed']} | {r['size_zero']:g} | "
                 f"{r['size_original']:g} | {'**yes**' if r['changed'] else 'no'} |")

    L += ["", "## Paired differences (original minus zero), headline envs only", "",
          "Ant-v5, HalfCheetah-v5, Hopper-v5 — the three environments in the manuscript's "
          "gym table. Bootstrap: 10,000 replicates over the (env, seed) pairs, "
          "seed 20260815, percentile interval.", "",
          "| metric | n | mean | median | max abs | 95% CI | excludes 0 |",
          "|---|---|---|---|---|---|---|"]
    for r in pr_h:
        L.append(f"| {r['metric']} | {r['n_pairs']} | {r['mean_diff']:+.4f} | "
                 f"{r['median_diff']:+.4f} | {r['max_abs_diff']:.4f} | "
                 f"[{r['ci95_lo']:+.4f}, {r['ci95_hi']:+.4f}] | "
                 f"{'**yes**' if r['excludes_zero'] else 'no'} |")
    L += ["", "## Paired differences including Humanoid-v5", "",
          "Humanoid is included for completeness only. P&C's Far AUROC there is "
          "0.22-0.47 under **both** centres, i.e. at or below chance, so its swings are "
          "not informative about the ridge centre and they dominate any pooled mean.", "",
          "| metric | n | mean | median | max abs | 95% CI | excludes 0 |",
          "|---|---|---|---|---|---|---|"]
    for r in pr:
        L.append(f"| {r['metric']} | {r['n_pairs']} | {r['mean_diff']:+.4f} | "
                 f"{r['median_diff']:+.4f} | {r['max_abs_diff']:.4f} | "
                 f"[{r['ci95_lo']:+.4f}, {r['ci95_hi']:+.4f}] | "
                 f"{'**yes**' if r['excludes_zero'] else 'no'} |")

    L += ["", "## Cross-environment ranks and wins", "",
          "| metric | centre | mean rank | wins |", "|---|---|---|---|"]
    for r in rk:
        L.append(f"| {r['metric']} | {r['ridge_center']} | {r['mean_rank']:.2f} | "
                 f"{r['wins']}/{r['n_cells']} |")
    L += ["", "The purpose of this comparison is provenance and consistency, not to argue "
          "that either centre is empirically superior."]
    (o / "mujoco_original_centered_summary.md").write_text("\n".join(L) + "\n")

    print(f"\nreselection: {n_ch}/{len(resel)} cells changed size")
    print(f"\n{'metric':<18}{'mean':>10}{'median':>10}{'max|d|':>10}   95% CI")
    for r in pr:
        print(f"  {r['metric']:<16}{r['mean_diff']:>+10.4f}{r['median_diff']:>+10.4f}"
              f"{r['max_abs_diff']:>10.4f}   [{r['ci95_lo']:+.4f}, {r['ci95_hi']:+.4f}]"
              f"{'  *' if r['excludes_zero'] else ''}")
    print(f"\nwrote {o}/mujoco_original_centered_*.csv|md")


if __name__ == "__main__":
    main()
