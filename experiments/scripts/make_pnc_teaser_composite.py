#!/usr/bin/env python3
"""
Paper-ready PnC teaser composite figure (Ant-v5).

Layout (asymmetric):
  Left (~50%):   (A) ID / OOD frontier
  Top-right:     (B) ID: hidden perturbation vs predictive change
  Bottom-right:  (C) Far-OOD: hidden perturbation vs predictive change

Usage:
    .venv/bin/python experiments/scripts/make_pnc_teaser_composite.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from plot_pnc_correction_frontier import collect_method_points

# ── Config ────────────────────────────────────────────────────────────────

ENV = "Ant-v5"
SEEDS = [0, 10, 42, 100, 200]
RESULTS_ROOT = Path("results")
# Analysis at size=20 for both PnC variants (consistent scale)
ANALYSIS_JSON = Path("experiments/figures/pnc_subspace_analysis_ant_size20_data.json")

METHOD_ORDER = ["Subspace", "SWAG", "Laplace", "PnC (no corr.)", "PnC (LS corr.)"]
COLORS = {
    "Subspace":        "#2ca02c",
    "SWAG":            "#1f77b4",
    "Laplace":         "#9467bd",
    "PnC (no corr.)":  "#ff7f0e",
    "PnC (LS corr.)":  "#d62728",
}
MARKERS = {
    "Subspace": "s", "SWAG": "D", "Laplace": "^",
    "PnC (no corr.)": "o", "PnC (LS corr.)": "o",
}

FRONTIER_SPECS = [
    {"label": "Subspace",       "include": ["subspace"],                       "exclude": ["hybrid", "cifar", "openood"]},
    {"label": "SWAG",           "include": ["swag"],                           "exclude": ["hybrid", "cifar", "openood"]},
    {"label": "Laplace",        "include": ["laplace"],                        "exclude": ["hybrid", "cifar", "openood", "vcal"]},
    {"label": "PnC (no corr.)", "include": ["pjsvd_multi", "none"],            "exclude": ["hybrid", "cifar", "openood"]},
    {"label": "PnC (LS corr.)", "include": ["pjsvd_multi", "least_squares"],   "exclude": ["hybrid", "cifar", "openood"]},
]

# ── Styling ───────────────────────────────────────────────────────────────

FONT = 9
plt.rcParams.update({
    "font.size": FONT,
    "axes.titlesize": 10,
    "axes.labelsize": FONT,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 7,
    "axes.grid": True,
    "grid.alpha": 0.25,
})


def _lighter(hex_color: str, factor: float = 0.45) -> str:
    """Return a lighter version of a hex color."""
    rgb = np.array(mcolors.to_rgb(hex_color))
    return mcolors.to_hex(rgb + (1 - rgb) * factor)


# ── Panel A: frontier ─────────────────────────────────────────────────────

SHORT_NAMES = {
    "Subspace": "Sub.",
    "SWAG": "SWAG",
    "Laplace": "Lapl.",
    "PnC (no corr.)": "PnC w/o Corr",
    "PnC (LS corr.)": "PnC",
}

PNC_LABELS = {
    "PnC (no corr.)":  "PnC\nw/o\nCorr",
    "PnC (LS corr.)":  "PnC",
}

LABEL_OFFSETS = {
    "Subspace":        ( 6,  -1),
    "SWAG":            ( 0,  12),
    "Laplace":         ( 6,   6),
    "PnC (no corr.)":  ( 8,   0),
    "PnC (LS corr.)":  (-2,   8),
}


def draw_frontier(ax):
    env_dir = RESULTS_ROOT / ENV
    points = {}
    for spec in FRONTIER_SPECS:
        pt = collect_method_points(
            env_dir=env_dir, method_spec=spec, seeds=SEEDS,
            x_metric="rmse_id", y_metric="nll_ood_far",
            selection_metric="nll_val", selection_fallback="nll_id",
            require_vcal=False, forbid_vcal=False, verbose=False,
            l2_metric="uncorrected_l2_id_h",
        )
        if pt is not None:
            points[pt["label"]] = pt

    for label in METHOD_ORDER:
        if label not in points:
            continue
        pt = points[label]
        c = COLORS[label]
        mk = MARKERS[label]
        x, y = pt["x_q50"], pt["y_q50"]
        xerr = [[x - pt["x_q25"]], [pt["x_q75"] - x]]
        yerr = [[y - pt["y_q25"]], [pt["y_q75"] - y]]

        ax.errorbar(x, y, xerr=xerr, yerr=yerr, fmt=mk, color=c,
                    capsize=3, markersize=7, label=label, zorder=3)

        txt = PNC_LABELS.get(label, label)
        off = LABEL_OFFSETS.get(label, (6, -8))
        va = "top" if off[1] < 0 else "bottom"
        ha = "right" if off[0] < 0 else ("center" if off[0] == 0 else "left")
        ax.annotate(txt, (x, y), xytext=off, textcoords="offset points",
                    fontsize=7.5, va=va, ha=ha, color=c)

    ax.set_xlabel(r"ID RMSE $\downarrow$", labelpad=2)
    ax.set_ylabel(r"Far NLL $\downarrow$", labelpad=2)
    ax.set_title("(A)  ID / OOD frontier")


# ── Panels B & C: dh vs dy bars ──────────────────────────────────────────

def draw_dh_dy(ax, data, region_key, title, show_legend=True):
    """Paired bar chart of ||Δh|| and ||Δy|| for all methods.

    region_key: "" for ID, "_ood" for OOD-far.
    """
    h_key = "h_l2" if region_key == "" else f"h_l2{region_key}"
    dy_key = "dy_median" if region_key == "" else f"dy_median{region_key}"

    methods = [m for m in METHOD_ORDER if m in data]
    x_pos = np.arange(len(methods))
    width = 0.36

    for i, m in enumerate(methods):
        c = COLORS[m]
        c_light = _lighter(c)
        dh = data[m].get(h_key, {}).get("median", 0)
        dy = data[m].get(dy_key, {}).get("median", 0)

        kw_h = {"label": r"$\|\Delta h\|$"} if i == 0 else {}
        kw_y = {"label": r"$\|\Delta y\|$"} if i == 0 else {}

        ax.bar(x_pos[i] - width / 2, dh, width, color=c_light, edgecolor=c,
               linewidth=0.8, **kw_h)
        ax.bar(x_pos[i] + width / 2, dy, width, color=c, edgecolor=c,
               linewidth=0.8, **kw_y)

    ax.set_xticks(x_pos)
    short = [SHORT_NAMES.get(m, m) for m in methods]
    ax.set_xticklabels(short, rotation=30, ha="right")
    ax.set_ylabel("Magnitude", labelpad=2)
    ax.set_title(title)
    if show_legend:
        ax.legend(loc="upper left", framealpha=0.85)
    ax.set_ylim(bottom=0)


# ── Composite ─────────────────────────────────────────────────────────────

def main():
    if not ANALYSIS_JSON.exists():
        print(f"Analysis data not found: {ANALYSIS_JSON}")
        print("Run: .venv/bin/python experiments/scripts/analyze_perturbation_subspaces.py"
              " --env Ant-v5 --seeds 0 --pnc-size 20.0")
        sys.exit(1)
    with open(ANALYSIS_JSON) as f:
        data = json.load(f)

    fig = plt.figure(figsize=(7.0, 2.5))
    gs = gridspec.GridSpec(
        1, 3, width_ratios=[1.2, 1, 1], wspace=0.45,
        left=0.07, right=0.99, top=0.87, bottom=0.25,
    )

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])

    draw_frontier(ax_a)
    draw_dh_dy(ax_b, data, "",     r"(B)  ID: $\|\Delta h\|$ vs $\|\Delta y\|$")
    draw_dh_dy(ax_c, data, "_ood", r"(C)  OOD: $\|\Delta h\|$ vs $\|\Delta y\|$",
               show_legend=False)

    # Share y-axis between B and C for direct comparison
    y_max = max(ax_b.get_ylim()[1], ax_c.get_ylim()[1])
    ax_b.set_ylim(0, y_max * 1.08)
    ax_c.set_ylim(0, y_max * 1.08)

    out_dir = Path("experiments/figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ["png", "pdf"]:
        p = out_dir / f"pnc_teaser_ant_composite.{ext}"
        fig.savefig(p, dpi=300 if ext == "png" else None, bbox_inches="tight")
        print(f"Saved {p}")
    plt.close()


if __name__ == "__main__":
    main()
