"""Sensitivity plots (Section 16): one figure per factor (metric relative to the
env anchor, one line per env + bold median + zero line), plus an env×factor
heatmap of max Near-AUROC change."""
from __future__ import annotations
import csv
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parents[1]
RAW = OUT / "aggregates" / "mujoco_sensitivity_raw.csv"
PLOTS = OUT / "plots"; PLOTS.mkdir(exist_ok=True)
ENVS = ["Ant-v5", "HalfCheetah-v5", "Hopper-v5", "Humanoid-v5"]


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return np.nan


def load():
    rows = list(csv.DictReader(open(RAW)))
    # seed-average per (env, factor, value)
    agg = defaultdict(lambda: defaultdict(list))
    order = defaultdict(dict)
    for r in rows:
        key = (r["environment"], r["factor"], r["factor_value"])
        for m in ("near_auroc", "id_rmse", "id_residual", "far_auroc", "mid_auroc"):
            agg[key][m].append(_f(r.get(m)))
        order[(r["environment"], r["factor"])][r["factor_value"]] = _f(r.get("factor_value_numeric"))
    sa = {k: {m: np.nanmean(v) for m, v in d.items()} for k, d in agg.items()}
    return sa, order


def _series(sa, order, env, factor, metric):
    vals = order[(env, factor)]
    # For the scale sweep, use the RELATIVE multiplier (0.25..4x) as x, not the raw
    # env-specific scale — the anchor is env/seed-specific, so raw scales are not
    # comparable across environments and can shift across seeds.
    if factor == "scale":
        vals = {k: float(k.rstrip("x")) for k in vals}
    items = sorted(vals.items(), key=lambda kv: (kv[1] if np.isfinite(kv[1]) else 0))
    xs = [v for _, v in items]; labels = [k for k, _ in items]
    ys = [sa.get((env, factor, k), {}).get(metric, np.nan) for k, _ in items]
    return np.array(xs, float), ys, labels


def anchor_val(sa, env, metric):
    # env anchor = scale 1.0x
    for k in sa:
        if k[0] == env and k[1] == "scale" and "1.0x" in k[2]:
            return sa[k].get(metric, np.nan)
    return np.nan


def plot_factor(sa, order, factor, metric, fname, relative="diff", logx=False, title=""):
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    med_x = None; med_stack = []
    for env in ENVS:
        if (env, factor) not in order:
            continue
        xs, ys, labels = _series(sa, order, env, factor, metric)
        a = anchor_val(sa, env, metric)
        if relative == "diff":
            yr = [y - a for y in ys]
        elif relative == "ratio":
            yr = [y / a if (a and np.isfinite(a)) else np.nan for y in ys]
        else:
            yr = ys
        ax.plot(range(len(xs)) if factor in ("layer",) else xs, yr, "o-", alpha=0.6, label=env)
        med_stack.append(yr); med_x = (range(len(xs)) if factor in ("layer",) else xs)
    if med_stack:
        med = np.nanmedian(np.array(med_stack), axis=0)
        ax.plot(list(med_x), med, "k-", lw=2.8, label="median", zorder=5)
    ax.axhline(0 if relative == "diff" else 1, color="gray", ls="--", lw=1)
    if logx and factor not in ("layer",):
        ax.set_xscale("symlog", linthresh=1e-4)
    if factor == "scale":
        ax.set_xscale("log", base=2); ax.set_xticks([0.25, 0.5, 1, 2, 4])
        ax.set_xticklabels(["0.25×", "0.5×", "1×", "2×", "4×"])
    xlab = {"scale": "perturbation scale (× env anchor)", "layer": "layer scope (0=first, 1=multi)"}
    ax.set_xlabel(xlab.get(factor, factor + " value"))
    ax.set_ylabel(f"{metric} {'(Δ vs anchor)' if relative=='diff' else ('(ratio vs anchor)' if relative=='ratio' else '')}")
    ax.set_title(title or f"{factor}: {metric}")
    ax.legend(fontsize=7, ncol=2); fig.tight_layout()
    fig.savefig(PLOTS / fname, dpi=110); plt.close(fig)


def heatmap(sa, order):
    factors = ["scale", "rank", "bootstrap", "calib", "ridge", "layer"]
    M = np.full((len(ENVS), len(factors)), np.nan)
    for i, env in enumerate(ENVS):
        for j, fac in enumerate(factors):
            _, ys, _ = _series(sa, order, env, fac, "near_auroc")
            ys = [y for y in ys if np.isfinite(y)]
            if ys:
                M[i, j] = max(ys) - min(ys)
    fig, ax = plt.subplots(figsize=(7, 4))
    im = ax.imshow(M, cmap="magma", aspect="auto")
    ax.set_xticks(range(len(factors))); ax.set_xticklabels(factors, rotation=30)
    ax.set_yticks(range(len(ENVS))); ax.set_yticklabels(ENVS)
    for i in range(len(ENVS)):
        for j in range(len(factors)):
            if np.isfinite(M[i, j]):
                ax.text(j, i, f"{M[i,j]:.03f}", ha="center", va="center",
                        color="white" if M[i, j] < np.nanmax(M) * 0.6 else "black", fontsize=8)
    ax.set_title("Max Near-AUROC change across factor values")
    fig.colorbar(im, label="AUROC range"); fig.tight_layout()
    fig.savefig(PLOTS / "heatmap_env_factor_auroc.png", dpi=110); plt.close(fig)


def main():
    sa, order = load()
    plot_factor(sa, order, "scale", "near_auroc", "scale_near_auroc_relative.png", "diff")
    plot_factor(sa, order, "scale", "id_rmse", "scale_id_rmse_relative.png", "ratio")
    plot_factor(sa, order, "rank", "near_auroc", "rank_near_auroc_relative.png", "diff")
    plot_factor(sa, order, "bootstrap", "near_auroc", "bootstrap_near_auroc_relative.png", "diff")
    plot_factor(sa, order, "calib", "id_residual", "calibration_size_id_residual.png", "none", logx=False,
                title="calibration size: held-out ID local residual (interpolation peak near n/p=1)")
    plot_factor(sa, order, "calib", "near_auroc", "calibration_size_near_auroc_relative.png", "diff")
    plot_factor(sa, order, "ridge", "id_residual", "ridge_id_residual.png", "none", logx=True)
    plot_factor(sa, order, "ridge", "near_auroc", "ridge_near_auroc_relative.png", "diff", logx=True)
    plot_factor(sa, order, "layer", "near_auroc", "layer_scope_near_auroc.png", "none")
    heatmap(sa, order)
    print("wrote plots to", PLOTS)


if __name__ == "__main__":
    main()
