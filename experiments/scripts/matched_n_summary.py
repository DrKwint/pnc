#!/usr/bin/env python3
"""Summarize the matched-n PJSVD vs Deep Ensemble experiment.

Reads results/{ENV}/standard_ensemble_n*_seed*.json and
results/{ENV}/pjsvd_multi_least_squares_random_projected_residual_prob_k20_n*_ps20.0_seed*.json
files, computes median + IQR across seeds for each (method, n) combination,
and prints a Pareto-style table.

Usage:
    python experiments/scripts/matched_n_summary.py
    python experiments/scripts/matched_n_summary.py --metric rmse_id
"""

import argparse
import glob
import json
import statistics
from collections import defaultdict
from pathlib import Path


def load_pjsvd(env: str, n: int, seed: int) -> dict | None:
    """Load PJSVD-multi result for a given (env, n, seed) at scale=20."""
    candidates = sorted(glob.glob(
        f"results/{env}/pjsvd_multi_least_squares_random_projected_residual_prob_k20_n{n}_ps*_h200-200-200-200_act-relu_seed{seed}.json"
    ))
    candidates = [c for c in candidates if "vcal" not in c]
    if not candidates:
        return None
    # Pick the file that contains scale 20.0
    for fname in candidates:
        with open(fname) as f:
            d = json.load(f)
        if "20.0" in d:
            return d["20.0"]
    return None


def load_de(env: str, n: int, seed: int) -> dict | None:
    """Load Deep Ensemble result for a given (env, n, seed)."""
    fname = f"results/{env}/standard_ensemble_n{n}_h200-200-200-200_act-relu_seed{seed}.json"
    if not Path(fname).exists():
        return None
    with open(fname) as f:
        return json.load(f)


def median_iqr(values: list[float]) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    if len(values) == 1:
        return values[0], 0.0
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    def pct(p):
        idx = (n - 1) * p
        lo = int(idx)
        hi = lo + 1 if lo + 1 < n else lo
        return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (idx - lo)
    return pct(0.5), pct(0.75) - pct(0.25)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 10, 200])
    p.add_argument("--envs", nargs="+", default=["HalfCheetah-v5", "Hopper-v5", "Ant-v5"])
    p.add_argument(
        "--metrics", nargs="+",
        default=["rmse_id", "nll_id", "nll_ood_far", "auroc_ood_far"],
    )
    args = p.parse_args()

    rows = []
    for env in args.envs:
        for method, n_values in [("PJSVD-Multi (k=20, scale=20)", [5, 10, 20, 50]),
                                  ("Deep Ensemble", [5, 10, 20])]:
            for n in n_values:
                metrics_by_key = defaultdict(list)
                for seed in args.seeds:
                    if "PJSVD" in method:
                        d = load_pjsvd(env, n, seed)
                    else:
                        d = load_de(env, n, seed)
                    if d is None:
                        continue
                    for m in args.metrics:
                        if m in d:
                            metrics_by_key[m].append(d[m])
                if not metrics_by_key:
                    continue
                row = {"env": env, "method": method, "n": n,
                       "n_seeds": len(metrics_by_key.get(args.metrics[0], []))}
                for m in args.metrics:
                    med, iqr = median_iqr(metrics_by_key.get(m, []))
                    row[m] = (med, iqr)
                rows.append(row)

    # Print one section per env
    for env in args.envs:
        print(f"\n=== {env} ===")
        env_rows = [r for r in rows if r["env"] == env]
        if not env_rows:
            print("  (no data)")
            continue
        col_widths = {m: max(20, max(len(m), 16)) for m in args.metrics}
        header = f"{'method':<35} {'n':>4} {'seeds':>6}  " + "  ".join(
            f"{m:>{col_widths[m]}}" for m in args.metrics
        )
        print(header)
        print("-" * len(header))
        for r in env_rows:
            cells = []
            for m in args.metrics:
                med, iqr = r[m]
                cells.append(f"{med:8.4f} ({iqr:6.4f})".rjust(col_widths[m]))
            print(f"{r['method']:<35} {r['n']:>4} {r['n_seeds']:>6}  " + "  ".join(cells))


if __name__ == "__main__":
    main()
