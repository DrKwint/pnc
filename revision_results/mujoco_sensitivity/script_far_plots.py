"""Plots for the 11-env Far-OOD sensitivity study: env×factor heatmap of Far-AUROC
range (key figure) + per-factor relative curves for Far AUROC / Far NLL / Far
Spearman / ID RMSE, and the calibration-size interpolation residual."""
from __future__ import annotations
import csv
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parents[1]
RAW = OUT / "aggregates" / "far_sensitivity_raw.csv"
PLOTS = OUT / "plots"; PLOTS.mkdir(exist_ok=True)
ENVS = ["Ant-v5", "HalfCheetah-v5", "Hopper-v5", "Humanoid-v5", "Walker2d-v5",
        "HumanoidStandup-v5", "Swimmer-v5", "Reacher-v5", "Pusher-v5",
        "InvertedPendulum-v5", "InvertedDoublePendulum-v5"]
FACTORS = ["scale", "rank", "bootstrap", "calib", "ridge", "layer"]


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return np.nan


def load():
    rows = list(csv.DictReader(open(RAW)))
    sa = defaultdict(lambda: defaultdict(list)); order = defaultdict(dict)
    for r in rows:
        k = (r["environment"], r["factor"], r["factor_value"])
        for m in ("far_auroc", "far_nll", "far_spearman", "id_rmse", "id_residual"):
            sa[k][m].append(_f(r.get(m)))
        order[(r["environment"], r["factor"])][r["factor_value"]] = _f(r.get("factor_value_numeric"))
    return {k: {m: np.nanmean(v) for m, v in d.items()} for k, d in sa.items()}, order


def series(sa, order, env, fac, m):
    vals = order.get((env, fac), {})
    if fac == "scale":
        vals = {k: float(k.rstrip("x")) for k in vals}
    items = sorted(vals.items(), key=lambda kv: kv[1] if np.isfinite(kv[1]) else 0)
    return [v for _, v in items], [sa.get((env, fac, k), {}).get(m, np.nan) for k, _ in items]


def heatmap(sa, order):
    M = np.full((len(ENVS), len(FACTORS)), np.nan)
    for i, e in enumerate(ENVS):
        for j, fc in enumerate(FACTORS):
            _, ys = series(sa, order, e, fc, "far_auroc")
            ys = [y for y in ys if np.isfinite(y)]
            if ys:
                M[i, j] = max(ys) - min(ys)
    fig, ax = plt.subplots(figsize=(7.5, 6))
    im = ax.imshow(M, cmap="magma", aspect="auto", vmin=0)
    ax.set_xticks(range(len(FACTORS))); ax.set_xticklabels(FACTORS, rotation=30)
    ax.set_yticks(range(len(ENVS))); ax.set_yticklabels(ENVS, fontsize=8)
    for i in range(len(ENVS)):
        for j in range(len(FACTORS)):
            if np.isfinite(M[i, j]):
                ax.text(j, i, f"{M[i,j]:.02f}", ha="center", va="center", fontsize=7,
                        color="white" if M[i, j] < np.nanmax(M) * 0.55 else "black")
    ax.set_title("Far-OOD AUROC range across factor values (11 envs)")
    fig.colorbar(im, label="AUROC range"); fig.tight_layout()
    fig.savefig(PLOTS / "far_heatmap_env_factor_auroc.png", dpi=120); plt.close(fig)


def curves(sa, order, m, fname, relative, title, ylab):
    fig, ax = plt.subplots(figsize=(7, 4.5)); stack = []; xref = None
    for e in ENVS:
        xs, ys = series(sa, order, e, m if False else "scale", m) if False else (None, None)
    for e in ENVS:
        pass
    # per-factor small multiples instead: plot the chosen metric vs each factor is too many;
    # here 'm' is the factor; plot metric 'metric' — handled by caller via fname mapping.


def factor_metric(sa, order, fac, metric, fname, relative, ylab, logx_scale=False):
    fig, ax = plt.subplots(figsize=(7, 4.5)); stack = []; xr = None
    for e in ENVS:
        xs, ys = series(sa, order, e, fac, metric)
        if not xs:
            continue
        a = sa.get((e, "scale", "1.0x"), {}).get(metric, np.nan)
        if relative == "diff":
            yr = [y - a for y in ys]
        elif relative == "ratio":
            yr = [y / a if (a and np.isfinite(a)) else np.nan for y in ys]
        else:
            yr = ys
        xx = range(len(xs)) if fac == "layer" else xs
        ax.plot(list(xx), yr, "o-", alpha=0.5, lw=1)
        stack.append(yr); xr = list(xx)
    if stack:
        L = max(len(s) for s in stack)
        st = np.array([s + [np.nan] * (L - len(s)) for s in stack], float)
        ax.plot(xr[:st.shape[1]], np.nanmedian(st, 0), "k-", lw=3, label="median", zorder=5)
    ax.axhline(0 if relative == "diff" else (1 if relative == "ratio" else np.nan), color="gray", ls="--", lw=1)
    if fac == "scale":
        ax.set_xscale("log", base=2); ax.set_xticks([0.25, 0.5, 1, 2, 4]); ax.set_xticklabels(["0.25×", "0.5×", "1×", "2×", "4×"])
    if logx_scale and fac in ("ridge",):
        ax.set_xscale("symlog", linthresh=1e-4)
    ax.set_xlabel(fac + (" (× anchor)" if fac == "scale" else (" (0=first,1=multi)" if fac == "layer" else " value")))
    ax.set_ylabel(ylab); ax.set_title(f"{fac}: {ylab}"); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(PLOTS / fname, dpi=115); plt.close(fig)


def main():
    sa, order = load()
    heatmap(sa, order)
    factor_metric(sa, order, "scale", "far_auroc", "far_scale_auroc.png", "diff", "Far AUROC (Δ vs anchor)")
    factor_metric(sa, order, "scale", "far_nll", "far_scale_nll.png", "diff", "Far NLL (Δ vs anchor)")
    factor_metric(sa, order, "scale", "far_spearman", "far_scale_spearman.png", "diff", "Far Spearman (Δ vs anchor)")
    factor_metric(sa, order, "rank", "far_auroc", "far_rank_auroc.png", "diff", "Far AUROC (Δ vs anchor)")
    factor_metric(sa, order, "ridge", "far_auroc", "far_ridge_auroc.png", "diff", "Far AUROC (Δ vs anchor)", logx_scale=True)
    factor_metric(sa, order, "calib", "id_residual", "far_calib_id_residual.png", "none", "held-out ID local residual")
    factor_metric(sa, order, "calib", "far_auroc", "far_calib_auroc.png", "diff", "Far AUROC (Δ vs anchor)")
    factor_metric(sa, order, "bootstrap", "far_auroc", "far_bootstrap_auroc.png", "diff", "Far AUROC (Δ vs anchor)")
    factor_metric(sa, order, "layer", "far_auroc", "far_layer_auroc.png", "none", "Far AUROC")
    print("wrote far plots to", PLOTS)


if __name__ == "__main__":
    main()
