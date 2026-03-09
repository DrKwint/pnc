#!/usr/bin/env python3
"""
plot_error_variance.py — Error-Variance scatter plots from saved experiment sidecars.

Each experiment run writes a .npz sidecar alongside its .json metrics file.
For gym experiments the sidecar contains:
    sq_error_id, pred_var_id    — per-sample squared error & predictive variance (ID split)
    sq_error_ood, pred_var_ood  — same for OOD split

This script:
  1. Discovers all .npz sidecars under results/<env>/
  2. Loads each and plots error vs predictive variance on a shared axes
  3. Overlays multiple experiments / seeds with distinct colours/markers

Usage:
    python plot_error_variance.py                              # all envs
    python plot_error_variance.py --env HalfCheetah-v5       # one env
    python plot_error_variance.py --split id                  # ID only
    python plot_error_variance.py --methods pjsvd standard   # filter by method keyword
    python plot_error_variance.py --out fig.png               # save instead of show
    python plot_error_variance.py --log                       # log-log axes
    python plot_error_variance.py --bins 40                   # bin & plot median ± IQR
"""

import argparse
import re
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


# ---------------------------------------------------------------------------
# Sidecar discovery
# ---------------------------------------------------------------------------

def _friendly(stem: str) -> str:
    """Best-effort human-readable label from a sidecar filename stem."""
    # Strip config suffixes added for multi-config files
    stem = re.sub(r"_seed\d+", "", stem)
    stem = re.sub(r"_prior([\d.]+)", r" prior=\1", stem)
    stem = re.sub(r"_ps([\d.]+)", r" size=\1", stem)
    stem = stem.replace("standard_ensemble", "Deep Ensemble")
    stem = stem.replace("mc_dropout", "MC Dropout")
    stem = stem.replace("ml_pjsvd", "ML-PJSVD")
    stem = stem.replace("pjsvd", "PJSVD")
    stem = stem.replace("laplace_priors", "Laplace")
    stem = stem.replace("swag", "SWAG")
    stem = re.sub(r"_n(\d+)", r" (n=\1)", stem)
    stem = re.sub(r"_k(\d+)", r" k=\1", stem)
    return stem.strip()


def _discover(results_dir: Path, env: str | None, methods: list[str]) -> list[dict]:
    """Return a list of {path, label, env} dicts for every matching .npz sidecar."""
    entries = []
    for env_dir in sorted(results_dir.iterdir()):
        if not env_dir.is_dir() or env_dir.name.lower() == "mnist":
            continue
        if env and env_dir.name != env:
            continue
        for npz in sorted(env_dir.glob("*.npz")):
            if npz.stem.startswith("data_"):
                continue   # skip data collection npz files
            label = _friendly(npz.stem)
            if methods and not any(m.lower() in label.lower() for m in methods):
                continue
            entries.append({"path": npz, "label": label, "env": env_dir.name})
    return entries


# ---------------------------------------------------------------------------
# Binned summary (median ± IQR)
# ---------------------------------------------------------------------------

def _bin_summary(x: np.ndarray, y: np.ndarray, n_bins: int = 30):
    """Bin x, compute median and IQR of y within each bin."""
    # Use log-spaced bins to handle heavy-tailed variance distributions
    x_min, x_max = np.percentile(x, 1), np.percentile(x, 99)
    if x_min <= 0:
        x_min = x[x > 0].min() if (x > 0).any() else 1e-8
    edges = np.logspace(np.log10(x_min), np.log10(x_max), n_bins + 1)
    centers, medians, q25s, q75s = [], [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (x >= lo) & (x < hi)
        if mask.sum() < 5:
            continue
        centers.append(np.sqrt(lo * hi))   # geometric mean
        medians.append(np.median(y[mask]))
        q25s.append(np.percentile(y[mask], 25))
        q75s.append(np.percentile(y[mask], 75))
    return (np.array(centers), np.array(medians),
            np.array(q25s),    np.array(q75s))


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _compute_norm_constants(entries: list[dict]) -> tuple[float, float]:
    """Compute normalisation constants from the pooled ID data of all entries.

    Returns (var_scale, err_scale): divide raw values by these before plotting
    so that the reference y=x line and cross-method comparisons share a scale.
    """
    all_var, all_err = [], []
    for entry in entries:
        data = np.load(entry["path"])
        if "pred_var_id" in data:
            all_var.append(data["pred_var_id"])
        if "sq_error_id" in data:
            all_err.append(data["sq_error_id"])
    var_scale = float(np.concatenate(all_var).mean()) if all_var else 1.0
    err_scale = float(np.concatenate(all_err).mean()) if all_err else 1.0
    # Guard against degenerate data
    var_scale = var_scale if var_scale > 0 else 1.0
    err_scale = err_scale if err_scale > 0 else 1.0
    return var_scale, err_scale


def _plot_entries(entries: list[dict], split: str, log: bool,
                  bins: int | None, ax: plt.Axes,
                  var_scale: float = 1.0, err_scale: float = 1.0,
                  cmap_name: str = "tab10"):
    cmap   = cm.get_cmap(cmap_name, max(len(entries), 1))
    colors = [cmap(i) for i in range(len(entries))]

    for i, entry in enumerate(entries):
        data  = np.load(entry["path"])
        label = entry["label"]
        color = colors[i]

        splits_to_plot = []
        if split in ("id", "both"):
            if "pred_var_id" in data and "sq_error_id" in data:
                splits_to_plot.append(("ID", data["pred_var_id"], data["sq_error_id"],
                                       color, "o", "-"))
        if split in ("ood", "both"):
            if "pred_var_ood" in data and "sq_error_ood" in data:
                splits_to_plot.append(("OOD", data["pred_var_ood"], data["sq_error_ood"],
                                       color, "^", "--"))

        for (split_name, pred_var, sq_err, col, marker, ls) in splits_to_plot:
            lbl = f"{label} ({split_name})" if split == "both" else label

            # Normalise
            pv = pred_var / var_scale
            se = sq_err   / err_scale

            if bins is not None:
                cx, med, q25, q75 = _bin_summary(pv, se, n_bins=bins)
                if len(cx) == 0:
                    continue
                ax.plot(cx, med, color=col, linestyle=ls, linewidth=1.5,
                        marker=marker, markersize=4, label=lbl)
                ax.fill_between(cx, q25, q75, color=col, alpha=0.15)
            else:
                # Subsample for readability if too many points
                n = len(pv)
                idx = np.random.choice(n, min(n, 2000), replace=False)
                ax.scatter(pv[idx], se[idx],
                           c=[col], alpha=0.25, s=6, marker=marker)
                ax.scatter([], [], c=[col], label=lbl,
                           marker=marker, s=20)  # legend proxy

    if log:
        ax.set_xscale("log")
        ax.set_yscale("log")

    # Perfect calibration reference: E[error] ∝ variance (holds on normalised axes too)
    xlim = ax.get_xlim()
    xs   = np.logspace(np.log10(max(xlim[0], 1e-10)), np.log10(xlim[1]), 100)
    ax.plot(xs, xs, "k--", linewidth=0.8, alpha=0.5, label="y = x (ideal)")
    ax.set_xlim(xlim)

    ax.set_xlabel("Predictive Variance (normalised)", fontsize=11)
    ax.set_ylabel("Squared Error (normalised)", fontsize=11)
    # Legend outside the axes, to the right
    ax.legend(fontsize=7, ncol=1, loc="upper left",
              bbox_to_anchor=(1.01, 1.0), borderaxespad=0)
    ax.grid(True, which="both", alpha=0.3, linewidth=0.5)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Error-Variance scatter/line plots from experiment sidecar .npz files.")
    parser.add_argument("--results_dir", default="results",
                        help="Root results directory (default: results/)")
    parser.add_argument("--env",     default=None,
                        help="Filter to a single environment name")
    parser.add_argument("--methods", nargs="*", default=None,
                        help="Show only methods whose label contains these keywords")
    parser.add_argument("--split",   choices=["id", "ood", "both"], default="both",
                        help="Which data split to plot (default: both)")
    parser.add_argument("--log",     action="store_true",
                        help="Use log-log axes")
    parser.add_argument("--bins",    type=int, default=30,
                        help="Number of bins for median±IQR summary (0 = scatter, default: 30)")
    parser.add_argument("--out",     default=None,
                        help="Save figure to this path instead of showing interactively")
    parser.add_argument("--seed",    type=int, default=0,
                        help="NumPy random seed for scatter subsampling")
    args = parser.parse_args()

    np.random.seed(args.seed)

    root    = Path(args.results_dir)
    entries = _discover(root, args.env, args.methods or [])

    if not entries:
        print("No sidecar .npz files found matching your filters.")
        return

    # Group by env for separate subplots
    envs = list(dict.fromkeys(e["env"] for e in entries))
    n    = len(envs)
    fig, axes = plt.subplots(1, n, figsize=(10 * n, 5), squeeze=False)
    fig.suptitle("Error vs. Predictive Variance", fontsize=13, fontweight="bold")

    bins_arg = args.bins if args.bins > 0 else None

    for col_idx, env_name in enumerate(envs):
        env_entries = [e for e in entries if e["env"] == env_name]
        var_scale, err_scale = _compute_norm_constants(env_entries)
        ax = axes[0][col_idx]
        ax.set_title(env_name, fontsize=11)
        _plot_entries(env_entries, split=args.split, log=args.log,
                      bins=bins_arg, ax=ax,
                      var_scale=var_scale, err_scale=err_scale)

    # Use tight_layout with rect to leave room for the outside legend
    plt.tight_layout(rect=[0, 0, 0.87, 1])

    if args.out:
        plt.savefig(args.out, dpi=150, bbox_inches="tight")
        print(f"Figure saved to {args.out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
