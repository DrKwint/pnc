#!/usr/bin/env python3
"""Generate a matched-cost Pareto plot for PJSVD vs Deep Ensemble.

Uses inference cost = number of forward passes per prediction as the x-axis
proxy. For PJSVD-Multi, each member is an independent forward pass through
the same base model, so cost ≈ n_perturbations. For Deep Ensemble, each
member is an independent base model, so cost ≈ n_baseline (still measured
in forward passes). Far NLL and Far AUROC go on y-axes.

Inputs: results/{env}/standard_ensemble_n*_seed*.json,
        results/{env}/pjsvd_multi_..._n*_ps*_seed*.json
Outputs: experiments/figures/pareto_far_nll.png,
         experiments/figures/pareto_far_auroc.png

Usage:
    python experiments/scripts/matched_cost_pareto.py
    python experiments/scripts/matched_cost_pareto.py --seeds 0,10,200
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import statistics
from pathlib import Path


def load_pjsvd_best_val(env: str, n: int, seed: int) -> dict | None:
    """Load PJSVD-multi result for (env, n, seed), picking the scale with
    best `nll_val` (fallback `nll_id`).
    """
    files = glob.glob(
        f"results/{env}/pjsvd_multi_least_squares_random_projected_residual"
        f"*_prob_k20_n{n}_ps*_h200-200-200-200_act-relu_seed{seed}.json"
    )
    files = [f for f in files if "vcal" not in f and "_none_" not in f]
    if not files:
        return None

    best = None
    best_key = float("inf")
    for fname in files:
        try:
            with open(fname) as f:
                data = json.load(f)
        except json.JSONDecodeError:
            continue
        if not isinstance(data, dict):
            continue
        for config, metrics in data.items():
            if not isinstance(metrics, dict):
                continue
            key = metrics.get("nll_val")
            if key is None:
                key = metrics.get("nll_id")
            if key is None:
                continue
            if key < best_key:
                best_key = key
                best = metrics
    return best


def load_de(env: str, n: int, seed: int) -> dict | None:
    fname = (
        f"results/{env}/standard_ensemble_n{n}"
        f"_h200-200-200-200_act-relu_seed{seed}.json"
    )
    if not Path(fname).exists():
        return None
    with open(fname) as f:
        return json.load(f)


def median_iqr(xs: list[float]) -> tuple[float, float]:
    xs = [x for x in xs if x is not None]
    if not xs:
        return float("nan"), 0.0
    if len(xs) == 1:
        return xs[0], 0.0
    xs_sorted = sorted(xs)
    n = len(xs_sorted)

    def pct(p: float) -> float:
        idx = (n - 1) * p
        lo = int(idx)
        hi = min(lo + 1, n - 1)
        return xs_sorted[lo] + (xs_sorted[hi] - xs_sorted[lo]) * (idx - lo)

    return pct(0.5), pct(0.75) - pct(0.25)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seeds", default="0,10,200", help="Comma-separated seed subset."
    )
    parser.add_argument(
        "--envs",
        default="HalfCheetah-v5,Hopper-v5,Ant-v5",
        help="Comma-separated env list.",
    )
    parser.add_argument(
        "--out-dir",
        default="experiments/figures",
        help="Where to save plots.",
    )
    parser.add_argument(
        "--linear-y",
        action="store_true",
        help="Use a linear y-axis instead of log (paper-ready version).",
    )
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    envs = args.envs.split(",")

    pjsvd_ns = [5, 10, 20, 50]
    de_ns = [5, 10, 20]

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    # Aggregate
    results: dict[str, dict[str, list[tuple[int, float, float]]]] = {}
    for env in envs:
        env_rows = {"PJSVD-Multi": [], "Deep Ensemble": []}
        for n in pjsvd_ns:
            nlls = []
            aurocs = []
            for seed in seeds:
                d = load_pjsvd_best_val(env, n, seed)
                if d is None:
                    continue
                nlls.append(d.get("nll_ood_far"))
                aurocs.append(d.get("auroc_ood_far"))
            if nlls:
                env_rows["PJSVD-Multi"].append(
                    (n, median_iqr(nlls)[0], median_iqr(aurocs)[0])
                )
        for n in de_ns:
            nlls = []
            aurocs = []
            for seed in seeds:
                d = load_de(env, n, seed)
                if d is None:
                    continue
                nlls.append(d.get("nll_ood_far"))
                aurocs.append(d.get("auroc_ood_far"))
            if nlls:
                env_rows["Deep Ensemble"].append(
                    (n, median_iqr(nlls)[0], median_iqr(aurocs)[0])
                )
        results[env] = env_rows

    # Print as text first
    print("\n=== Matched-Cost Pareto (median over seeds) ===")
    for env, env_rows in results.items():
        print(f"\n{env}")
        for method, rows in env_rows.items():
            for n, nll, auroc in rows:
                print(
                    f"  {method:18s} n={n:>3}  "
                    f"Far NLL={nll:>7.3f}  Far AUROC={auroc:.3f}"
                )

    # Plot
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\nmatplotlib not available, skipping plot.")
        return

    for metric, metric_label, metric_key in [
        ("nll", "Far NLL (lower = better)", 1),
        ("auroc", "Far AUROC (higher = better)", 2),
    ]:
        fig, axes = plt.subplots(1, len(envs), figsize=(4.5 * len(envs), 4), sharey=False)
        if len(envs) == 1:
            axes = [axes]
        for ax, env in zip(axes, envs):
            rows = results[env]
            for method, style in [
                ("PJSVD-Multi", {"color": "#1f77b4", "marker": "o", "linestyle": "-"}),
                ("Deep Ensemble", {"color": "#d62728", "marker": "s", "linestyle": "--"}),
            ]:
                pts = rows.get(method, [])
                if not pts:
                    continue
                xs = [p[0] for p in pts]
                ys = [p[metric_key] for p in pts]
                ax.plot(xs, ys, label=method, linewidth=1.8, markersize=7, **style)
            ax.set_title(env)
            ax.set_xlabel("Inference cost (n forward passes)")
            ax.set_xscale("log")
            ax.grid(alpha=0.3)
            if metric == "nll" and not args.linear_y:
                ax.set_yscale("log")
        axes[0].set_ylabel(metric_label)
        axes[-1].legend(loc="best", fontsize=9)
        fig.suptitle(
            f"Matched-cost comparison: PJSVD vs Deep Ensemble ({metric_label})",
            fontsize=11,
        )
        fig.tight_layout()
        out_path = Path(args.out_dir) / f"pareto_far_{metric}.png"
        fig.savefig(out_path, dpi=140, bbox_inches="tight")
        print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
