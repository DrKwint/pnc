#!/usr/bin/env python3
"""Draw Figure 3 — mechanism-level diagnostics for P&C on Ant-v5 — in one pass.

A single matplotlib figure with all three panels, exported as one flat vector
PDF. Nothing is composited, overlaid or pasted in from another PDF; every mark
on the page is drawn here from a data file.

  (A) Correction frontier          figures/fig3_panelA_source.csv
  (B) Disagreement vs distance     figures/fig3_panelB_source.csv
  (C) Probes predict disagreement  pnc_repro/artifacts/panel_c_diagnostic_Ant-v5_seed0.csv

Panel C's y-axis reads "P&C disagreement D(x)". The submitted figure said
"Finite P&C disagreement D(x)"; the final manuscript drops "Finite". Its three
annotations are recomputed from the canonical CSV on every run and reproduce the
submitted values exactly (rho_s +0.73, beta_1 +0.83±0.01, t +117.7).

Provenance of the three inputs differs, and the difference matters:

  * Panel C is canonical. `panel_c_diagnostic_Ant-v5_seed0.csv` is the per-example
    diagnostic written by the original run: 4000 rows, 1000 per tier.
  * Panels A and B are **recovered**. Their run is a bespoke configuration
    (perturbation scales 4/8/16/32, bootstrap_frac 0.3, lambda 0.1, calibration
    size 1024) that was never cached under `results/`, and the script that drew
    them is gone. `scripts/extract_fig3_panelAB_data.py` reads their data back
    out of the submitted PDF's vector content stream and inverts the axis
    transform; the recovery is accurate to ~1e-6 relative, verified against the
    cached hidden-space Mahalanobis distances (see figure3c_audit.md).
    Panel B's rho_s / beta_1 annotation cannot be recomputed, because only the
    plotted subsample survives, so it is carried through verbatim from the
    submitted panel via the recovery JSON.

Panel geometry, colours, opacities, line widths and type sizes were likewise
read out of the submitted PDF, so the regenerated figure lands on the same page
size and the same axes rectangles.

Run:  .venv/bin/python scripts/make_fig3.py
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

from make_fig3c_panel import (  # noqa: E402  (same directory)
    FIG_DIR, REPO_ROOT, draw_panel_c, load_panel_c, panel_c_stats,
    scatter_subsets, style,
)

PANEL_A_CSV = FIG_DIR / "fig3_panelA_source.csv"
PANEL_B_CSV = FIG_DIR / "fig3_panelB_source.csv"
RECOVERY_JSON = FIG_DIR / "fig3_panelAB_recovery.json"

PAGE_W_PT = 516.654296875
PAGE_H_PT = 202.16250610351562
# Axes rectangles as (x0, y0, x1, y1) in points from the page's bottom-left.
PANEL_RECTS = {
    "A": (33.45, 31.7025, 153.66, 165.9825),
    "B": (208.94, 31.7025, 329.15, 165.9825),
    "C": (384.43, 31.7025, 504.64, 165.9825),
}
SUPTITLE_X_PT = 255.65     # centre of the "Ant-v5" figure title
SUPTITLE_Y_PT = 188.12     # its baseline

SERIES = {
    "no_correction": {"color": "#ff7f0e", "label": "No correction"},
    "ls_correction": {"color": "#d62728", "label": "LS correction"},
}
TIERS = ["id", "near", "mid", "far"]
TIER_LABELS = {"id": "ID", "near": "Near", "mid": "Mid", "far": "Far"}
TIER_COLORS = {"id": "#2ca02c", "near": "#1f77b4", "mid": "#9467bd", "far": "#d62728"}

FS_STATS = 6.2


def axes_fraction(rect):
    x0, y0, x1, y1 = rect
    return [x0 / PAGE_W_PT, y0 / PAGE_H_PT,
            (x1 - x0) / PAGE_W_PT, (y1 - y0) / PAGE_H_PT]


def apply_recovered_ticks(axis, cal):
    """Reinstate the tick labels the submitted panel used.

    Panel A's x-axis spans less than a decade and was labelled 0.4 / 0.6 / 0.8 / 1
    in plain decimals. Matplotlib's default log formatter would instead print
    3x10^-1, 4x10^-1, ... and collide. Where the recovered labels are exact
    powers of ten the default formatter is already right, so leave it alone.
    """
    values = cal.get("values") or []
    if cal["scale"] != "log" or not values:
        return
    if all(abs(np.log10(v) - round(np.log10(v))) < 1e-9 for v in values):
        return
    # Sub-decade log axis. Leave the locators alone — the submitted panel has a
    # single major tick (and so a single vertical grid line) at 1.0, with 0.4 /
    # 0.6 / 0.8 labelled on minor ticks. Only the labelling differs from the
    # default, which would print "4x10^-1" and collide.
    def fmt(v, _pos):
        return f"{v:g}" if any(abs(v - t) <= 1e-9 * max(1.0, abs(t)) for t in values) else ""
    axis.set_major_formatter(mticker.FuncFormatter(fmt))
    axis.set_minor_formatter(mticker.FuncFormatter(fmt))


def grid(ax):
    ax.grid(True, which="major", color="#b0b0b0", alpha=0.22, linewidth=0.4)
    ax.set_axisbelow(True)


def stats_box(ax, lines):
    ax.text(0.982, 0.038, "\n".join(lines), transform=ax.transAxes,
            ha="right", va="bottom", fontsize=FS_STATS, color="0.15",
            linespacing=1.04,
            bbox=dict(boxstyle="round,pad=0.22", facecolor="white",
                      edgecolor="0.75", alpha=0.9, linewidth=0.4),
            zorder=6)


# ── panel A ──────────────────────────────────────────────────────────────────


def draw_panel_a(ax, rows, limits):
    for key, cfg in SERIES.items():
        pts = sorted((r for r in rows if r["series"] == key),
                     key=lambda r: int(r["perturbation_scale"]))
        x = [float(r["id_rmse"]) for r in pts]
        y = [float(r["far_nll"]) for r in pts]
        ax.plot(x, y, "-", color=cfg["color"], lw=1.2, alpha=0.9, zorder=2)
        ax.plot(x, y, "o", color=cfg["color"], markersize=4.2, linestyle="",
                label=cfg["label"], zorder=3)
        for r in pts:
            # va="center" centres the full ascent+descent box, which sits
            # 0.35 pt high against the recovered ink bbox centre ("p" descends)
            ax.annotate(f"ps={int(r['perturbation_scale'])}",
                        xy=(float(r["label_x"]), float(r["label_y"])),
                        xytext=(0, -0.35), textcoords="offset points",
                        fontsize=5.8, color=cfg["color"], ha="left", va="center",
                        annotation_clip=False, zorder=4,
                        bbox=dict(boxstyle="square,pad=0.15", facecolor="white",
                                  edgecolor="none", alpha=0.8))

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(*limits["x"]["axis_limits"])
    ax.set_ylim(*limits["y"]["axis_limits"])
    apply_recovered_ticks(ax.xaxis, limits["x"])
    apply_recovered_ticks(ax.yaxis, limits["y"])
    grid(ax)
    ax.set_xlabel(r"ID RMSE $\downarrow$")
    ax.set_ylabel(r"Far NLL $\downarrow$")
    ax.set_title("(A) Correction frontier")
    ax.legend(loc="upper right", fontsize=6.0, framealpha=0.9, borderpad=0.25,
              labelspacing=0.19, handlelength=1.0, handletextpad=0.3,
              borderaxespad=0.5)


# ── panel B ──────────────────────────────────────────────────────────────────


def draw_panel_b(ax, rows, limits, annotation):
    def pick(role):
        return [r for r in rows if r["role"] == role]

    for tier in TIERS:
        pts = [r for r in pick("scatter") if r["tier"] == tier]
        ax.plot([float(r["distance_mahal"]) for r in pts],
                [float(r["disagreement"]) for r in pts],
                marker="o", linestyle="", markersize=1.2,
                color=TIER_COLORS[tier], alpha=0.10, zorder=1)

    band = sorted(pick("sem_band"), key=lambda r: float(r["distance_mahal"]))
    ax.fill_between([float(r["distance_mahal"]) for r in band],
                    [float(r["sem_lo"]) for r in band],
                    [float(r["sem_hi"]) for r in band],
                    color="0.25", alpha=0.18, linewidth=0, zorder=4)

    for tier in TIERS:
        pts = sorted((r for r in pick("tier_bin_mean") if r["tier"] == tier),
                     key=lambda r: float(r["distance_mahal"]))
        ax.plot([float(r["distance_mahal"]) for r in pts],
                [float(r["disagreement"]) for r in pts],
                marker="o", linestyle="", markersize=3.2,
                color=TIER_COLORS[tier], markeredgecolor="white",
                markeredgewidth=0.35, label=TIER_LABELS[tier], zorder=3)

    trend = sorted(pick("pooled_trend"), key=lambda r: float(r["distance_mahal"]))
    ax.plot([float(r["distance_mahal"]) for r in trend],
            [float(r["disagreement"]) for r in trend],
            "-", color="0.1", lw=1.5, zorder=5)

    ax.set_xscale("log")
    ax.set_xlim(*limits["x"]["axis_limits"])
    ax.set_ylim(*limits["y"]["axis_limits"])
    apply_recovered_ticks(ax.xaxis, limits["x"])
    grid(ax)
    ax.set_xlabel("Mahalanobis distance to calibration (hidden)")
    ax.set_ylabel("Prediction disagreement to base")
    ax.set_title("(B) Disagreement vs distance")
    ax.legend(loc="upper left", framealpha=0.9, borderpad=0.3, labelspacing=0.25,
              handlelength=1.0, handletextpad=0.4, borderaxespad=0.5)
    stats_box(ax, [
        rf"$\rho_s = {annotation['rho_s']:+.2f}$",
        rf"$\beta_1 = {annotation['beta_1']:+.2f}{{\pm}}{annotation['beta_1_se']:.2f}$",
    ])


# ── figure ───────────────────────────────────────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-stem", type=Path,
                    default=FIG_DIR / "fig3_pnc_bridge_Ant-v5")
    args = ap.parse_args()

    for path in (PANEL_A_CSV, PANEL_B_CSV, RECOVERY_JSON):
        if not path.exists():
            raise SystemExit(
                f"missing {path.relative_to(REPO_ROOT)} — run "
                "scripts/extract_fig3_panelAB_data.py first (needs PyMuPDF)")

    a_rows = list(csv.DictReader(PANEL_A_CSV.open()))
    b_rows = list(csv.DictReader(PANEL_B_CSV.open()))
    recovery = json.loads(RECOVERY_JSON.read_text())

    c_data = load_panel_c()
    c_stats = panel_c_stats(c_data)
    c_subsets = scatter_subsets(c_data)

    with plt.rc_context(style()):
        fig = plt.figure(figsize=(PAGE_W_PT / 72.0, PAGE_H_PT / 72.0))
        ax_a = fig.add_axes(axes_fraction(PANEL_RECTS["A"]))
        ax_b = fig.add_axes(axes_fraction(PANEL_RECTS["B"]))
        ax_c = fig.add_axes(axes_fraction(PANEL_RECTS["C"]))

        draw_panel_a(ax_a, a_rows, recovery["panel_A"]["calibration"])
        draw_panel_b(ax_b, b_rows, recovery["panel_B"]["calibration"],
                     recovery["panel_B_printed_annotation"])
        draw_panel_c(ax_c, c_data, c_stats, c_subsets)

        fig.text(SUPTITLE_X_PT / PAGE_W_PT, SUPTITLE_Y_PT / PAGE_H_PT, "Ant-v5",
                 ha="center", va="baseline", fontsize=8.5)

        fig.savefig(args.out_stem.with_suffix(".pdf"))
        fig.savefig(args.out_stem.with_suffix(".png"), dpi=400)
        plt.close(fig)

    ols = c_stats["regime_controlled_ols"]
    print(f"panel A: {len(a_rows)} frontier points")
    print(f"panel B: {sum(r['role'] == 'scatter' for r in b_rows)} scatter, "
          f"{sum(r['role'] == 'tier_bin_mean' for r in b_rows)} tier bin means, "
          f"{sum(r['role'] == 'pooled_trend' for r in b_rows)} trend vertices")
    print(f"panel C: n={c_stats['n_total']}  rho_s={c_stats['pooled_spearman']['rho']:+.4f}  "
          f"beta_1={ols['beta_1']:+.4f}+-{ols['beta_1_se']:.4f}  t={ols['t_stat']:+.4f}")
    print(f"wrote {args.out_stem.with_suffix('.pdf')} / .png")


if __name__ == "__main__":
    main()
