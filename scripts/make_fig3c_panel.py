#!/usr/bin/env python3
"""Regenerate Figure 3 panel (C) — "Probes predict disagreement" — for Ant-v5.

The submitted panel's y-axis reads "Finite P&C disagreement D(x)". The final
manuscript calls the quantity "P&C disagreement D(x)", so the panel has to be
re-rendered with the corrected label. The original three-panel plotting script
(which produced ``pnc_repro/figures/pnc_bridge_<env>.pdf``) is not in this repo
or anywhere on this machine, so panel C is rebuilt here from its canonical
per-example data:

    pnc_repro/artifacts/panel_c_diagnostic_Ant-v5_seed0.csv   (4000 rows)

Quantities (columns of that CSV; see pnc_repro/figures/notes.txt):
  x = ``sensitivity_sketch``   random-probe sketch
        s_hat(x) = (1 / (J eps^2)) * sum_j || f_probe_j(x) - f_base(x) ||^2
      over J=16 finite-difference probes u_j ~ N(0, I) at eps = 0.01 * ps = 0.32.
  y = ``finite_disagreement``  the finite-scale P&C ensemble disagreement D(x)
      at the full operating perturbation scale ps=32.
  tier = ``regime`` in {id, near, mid, far}, 1000 examples each.

Reported statistics (all recomputed here from the CSV, not copied):
  rho_s : pooled Spearman rank correlation between s_hat(x) and D(x), n=4000.
  beta_1: slope of the regime-controlled log-log OLS
              log10 D(x) = beta_0 + beta_1 * log10 s_hat(x) + regime dummies
          with ID as the dropped baseline; "+-" is the OLS standard error.
  t     : beta_1 / SE(beta_1).

Drawing recipe recovered from the submitted vector PDF so the regenerated panel
is visually identical apart from the label (see figure3c_audit.md):
  - scatter: 500 examples per tier, drawn with a single np.random.default_rng(0)
    sampled in the order id, near, mid, far; tab10 colours; ms=1.4, alpha=0.18.
  - trend: 12 quantile bins on the pooled s_hat; x = geometric centre of the bin
    edges, y = mean D(x) in the bin, with a +-SEM band.
  - axes autoscale from the plotted subsample (log-log, default 5% margins).

Outputs (repo-root relative):
  figures/fig3c_panelC_source.csv        normalized per-example plot input
  figures/fig3c_panelC_stats.json        recomputed statistics + provenance
  figures/fig3c_probes_predict_disagreement.{pdf,png}   standalone panel

The full three-panel Figure 3 is drawn by ``scripts/make_fig3.py``, which imports
the panel-C drawing from this module. This script only produces panel C on its
own, plus the normalized inputs and the recomputed statistics.

Run:  .venv/bin/python scripts/make_fig3c_panel.py
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_CSV = REPO_ROOT / "pnc_repro" / "artifacts" / "panel_c_diagnostic_Ant-v5_seed0.csv"
SUBMITTED_PDF = REPO_ROOT / "pnc_repro" / "figures" / "pnc_bridge_Ant-v5.pdf"
FIG_DIR = REPO_ROOT / "figures"

ENV = "Ant-v5"
TIERS = ["id", "near", "mid", "far"]
TIER_LABELS = {"id": "ID", "near": "Near", "mid": "Mid", "far": "Far"}
# tab10 green/blue/purple/red, read back out of the submitted PDF.
TIER_COLORS = {
    "id": "#2ca02c",
    "near": "#1f77b4",
    "mid": "#9467bd",
    "far": "#d62728",
}

N_SCATTER_PER_TIER = 500
N_TREND_BINS = 12
SCATTER_RNG_SEED = 0

# Panel geometry of the submitted figure, in PostScript points. Reproducing it
# lets the regenerated panel drop straight back into the composite.
PAGE_W_PT = 516.654296875
PAGE_H_PT = 202.16250610351562
PANEL_C_SPLIT_PT = 354.4          # x of the boundary between panels B and C
AXES_X0_PT, AXES_X1_PT = 384.43, 504.64
AXES_Y0_PT, AXES_Y1_PT = 31.7025, 165.9825   # measured from the page bottom

# Type sizes, also read back out of the submitted PDF.
FS_TITLE = 8.0
FS_LABEL = 7.5
FS_TICK = 6.5
FS_STATS = 6.2


# ── data ─────────────────────────────────────────────────────────────────────


def load_panel_c() -> dict[str, dict[str, np.ndarray]]:
    """Per-tier ``s_hat`` / ``D`` arrays from the canonical diagnostic CSV."""
    rows = list(csv.DictReader(SOURCE_CSV.open()))
    out: dict[str, dict[str, np.ndarray]] = {}
    for tier in TIERS:
        sel = [r for r in rows if r["regime"] == tier]
        out[tier] = {
            "s_hat": np.array([float(r["sensitivity_sketch"]) for r in sel], np.float64),
            "D": np.array([float(r["finite_disagreement"]) for r in sel], np.float64),
        }
    return out


def pooled(data: dict[str, dict[str, np.ndarray]]):
    s = np.concatenate([data[t]["s_hat"] for t in TIERS])
    d = np.concatenate([data[t]["D"] for t in TIERS])
    tier_id = np.concatenate([np.full(data[t]["s_hat"].size, i) for i, t in enumerate(TIERS)])
    return s, d, tier_id


def panel_c_stats(data: dict[str, dict[str, np.ndarray]]) -> dict:
    """Pooled Spearman plus the regime-controlled log-log OLS slope."""
    s, d, tier_id = pooled(data)
    rho, rho_p = spearmanr(s, d)

    x = np.log10(s)
    y = np.log10(d)
    n = x.size
    # ID is the dropped baseline; one dummy per remaining tier.
    design = np.column_stack(
        [np.ones(n), x] + [(tier_id == i).astype(float) for i in range(1, len(TIERS))]
    )
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    resid = y - design @ beta
    dof = n - design.shape[1]
    cov = (resid @ resid / dof) * np.linalg.inv(design.T @ design)
    se = np.sqrt(np.diag(cov))
    slope, slope_se = float(beta[1]), float(se[1])

    return {
        "n_total": int(n),
        "n_by_tier": {t: int(data[t]["s_hat"].size) for t in TIERS},
        "pooled_spearman": {"rho": float(rho), "p": float(rho_p)},
        "within_tier_spearman": {
            t: float(spearmanr(data[t]["s_hat"], data[t]["D"])[0]) for t in TIERS
        },
        "regime_controlled_ols": {
            "model": "log10(D) ~ 1 + log10(s_hat) + tier dummies (ID = baseline)",
            "scale": "log-log (base 10), no epsilon offset needed; all values > 0",
            "beta_1": slope,
            "beta_1_se": slope_se,
            "beta_1_se_definition": "OLS standard error (homoskedastic), not a CI half-width",
            "t_stat": slope / slope_se,
            "dof": int(dof),
            "intercept": float(beta[0]),
            "tier_offsets": {t: float(beta[2 + i]) for i, t in enumerate(TIERS[1:])},
        },
    }


def trend(data: dict[str, dict[str, np.ndarray]]):
    """12 pooled quantile bins: geometric bin centre vs mean D, with SEM."""
    s, d, _ = pooled(data)
    edges = np.quantile(s, np.linspace(0.0, 1.0, N_TREND_BINS + 1))
    centers = np.sqrt(edges[:-1] * edges[1:])          # geometric, matching the log x-axis
    idx = np.clip(np.digitize(s, edges[1:-1]), 0, N_TREND_BINS - 1)
    mean = np.array([d[idx == b].mean() for b in range(N_TREND_BINS)])
    sem = np.array([d[idx == b].std(ddof=1) / np.sqrt((idx == b).sum())
                    for b in range(N_TREND_BINS)])
    n_in_bin = np.array([int((idx == b).sum()) for b in range(N_TREND_BINS)])
    return centers, mean, sem, edges, n_in_bin


def scatter_subsets(data: dict[str, dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    """Indices of the plotted subsample, tier by tier, from one shared RNG."""
    rng = np.random.default_rng(SCATTER_RNG_SEED)
    return {
        t: np.sort(rng.choice(data[t]["s_hat"].size,
                              min(N_SCATTER_PER_TIER, data[t]["s_hat"].size),
                              replace=False))
        for t in TIERS
    }


# ── drawing ──────────────────────────────────────────────────────────────────


def style() -> dict:
    return {
        "font.size": FS_TICK,
        "axes.titlesize": FS_TITLE,
        "axes.labelsize": FS_LABEL,
        "xtick.labelsize": FS_TICK,
        "ytick.labelsize": FS_TICK,
        "legend.fontsize": FS_TICK,
        "axes.labelpad": 2.5,
        "axes.titlepad": 3.0,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.minor.width": 0.6,
        "ytick.minor.width": 0.6,
        "xtick.major.pad": 2.0,
        "ytick.major.pad": 2.0,
        "xtick.minor.pad": 2.0,
        "ytick.minor.pad": 2.0,
        "xtick.major.size": 2.0,
        "ytick.major.size": 2.0,
        "xtick.minor.size": 2.0,
        "ytick.minor.size": 2.0,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "pdf.fonttype": 42,
    }


def draw_panel_c(ax, data, stats, subsets) -> None:
    for tier in TIERS:
        sel = subsets[tier]
        ax.plot(
            data[tier]["s_hat"][sel], data[tier]["D"][sel],
            marker="o", linestyle="", markersize=1.4,
            color=TIER_COLORS[tier], alpha=0.18,
            label=TIER_LABELS[tier], zorder=1,
        )

    centers, mean, sem, _, _ = trend(data)
    ax.fill_between(centers, mean - sem, mean + sem,
                    color="0.25", alpha=0.18, linewidth=0, zorder=4)
    ax.plot(centers, mean, "-", color="0.1", lw=1.5, zorder=5)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, which="major", color="#b0b0b0", alpha=0.22, linewidth=0.4)
    ax.set_axisbelow(True)

    ax.set_xlabel(r"Random-probe sketch $\hat{s}(x)$")
    # The submitted panel said "Finite P&C disagreement D(x)"; the final
    # manuscript calls the quantity "P&C disagreement D(x)".
    ax.set_ylabel(r"P&C disagreement $D(x)$")
    ax.set_title("(C) Probes predict disagreement")

    ax.legend(
        loc="upper left", framealpha=0.9, borderpad=0.3, labelspacing=0.25,
        handlelength=1.0, handletextpad=0.4, borderaxespad=0.5,
    )

    rho = stats["pooled_spearman"]["rho"]
    b1 = stats["regime_controlled_ols"]["beta_1"]
    se = stats["regime_controlled_ols"]["beta_1_se"]
    t = stats["regime_controlled_ols"]["t_stat"]
    annot = "\n".join([
        rf"$\rho_s = {rho:+.2f}$",
        rf"$\beta_1 = {b1:+.2f}{{\pm}}{se:.2f}$",
        rf"$t = {t:+.1f}$",
    ])
    ax.text(
        0.982, 0.038, annot, transform=ax.transAxes, ha="right", va="bottom",
        fontsize=FS_STATS, color="0.15", linespacing=1.04,
        bbox=dict(boxstyle="round,pad=0.22", facecolor="white",
                  edgecolor="0.75", alpha=0.9, linewidth=0.4),
        zorder=6,
    )


def render_standalone(data, stats, subsets, out_stem: Path) -> None:
    """Panel C alone, on the same footprint it occupies inside Figure 3."""
    width_pt = PAGE_W_PT - PANEL_C_SPLIT_PT
    with plt.rc_context(style()):
        fig = plt.figure(figsize=(width_pt / 72.0, PAGE_H_PT / 72.0))
        ax = fig.add_axes([
            (AXES_X0_PT - PANEL_C_SPLIT_PT) / width_pt,
            AXES_Y0_PT / PAGE_H_PT,
            (AXES_X1_PT - AXES_X0_PT) / width_pt,
            (AXES_Y1_PT - AXES_Y0_PT) / PAGE_H_PT,
        ])
        draw_panel_c(ax, data, stats, subsets)
        fig.savefig(out_stem.with_suffix(".pdf"))
        fig.savefig(out_stem.with_suffix(".png"), dpi=400)
        plt.close(fig)


# ── outputs ──────────────────────────────────────────────────────────────────


def write_source_csv(data, subsets, path: Path) -> None:
    """Exactly what the panel consumes: every example, flagged if plotted."""
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["env", "seed", "tier", "example_index",
                    "sensitivity_sketch_s_hat", "finite_disagreement_D",
                    "in_scatter_subsample"])
        for tier in TIERS:
            plotted = set(subsets[tier].tolist())
            for i, (s, d) in enumerate(zip(data[tier]["s_hat"], data[tier]["D"])):
                w.writerow([ENV, 0, tier, i, repr(s), repr(d), int(i in plotted)])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=FIG_DIR)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    data = load_panel_c()
    stats = panel_c_stats(data)
    subsets = scatter_subsets(data)
    centers, mean, sem, edges, n_in_bin = trend(data)

    panel_stem = args.out_dir / "fig3c_probes_predict_disagreement"
    render_standalone(data, stats, subsets, panel_stem)
    write_source_csv(data, subsets, args.out_dir / "fig3c_panelC_source.csv")

    stats_out = dict(stats)
    stats_out["provenance"] = {
        "canonical_source": str(SOURCE_CSV.relative_to(REPO_ROOT)),
        "submitted_figure": str(SUBMITTED_PDF.relative_to(REPO_ROOT)),
        "env": ENV, "seed": 0,
        "probe_config": {"n_probes_J": 16, "probe_eps": 0.32,
                         "eps_frac_of_ps": 0.01, "pnc_perturbation_scale": 32.0},
        "pnc_config": {"bootstrap_frac": 0.3, "correction_lambda": 0.1,
                       "calibration_size": 1024,
                       "target_layer": "even-indexed (l1, l3, ...); LS on next layer"},
        "scatter": {"per_tier": N_SCATTER_PER_TIER,
                    "rng": f"np.random.default_rng({SCATTER_RNG_SEED}), drawn in tier order "
                           + ", ".join(TIERS)},
    }
    stats_out["trend"] = {
        "n_bins": N_TREND_BINS,
        "binning": "quantile bins on pooled s_hat over all 4000 examples",
        "x": "geometric centre of the bin edges", "y": "mean D(x) in bin",
        "band": "+/- SEM",
        "bin_edges": [float(v) for v in edges],
        "bin_centers": [float(v) for v in centers],
        "bin_mean_D": [float(v) for v in mean],
        "bin_sem_D": [float(v) for v in sem],
        "bin_n": [int(v) for v in n_in_bin],
    }
    (args.out_dir / "fig3c_panelC_stats.json").write_text(json.dumps(stats_out, indent=2) + "\n")

    ols = stats["regime_controlled_ols"]
    print(f"n = {stats['n_total']} "
          + " ".join(f"{t}={stats['n_by_tier'][t]}" for t in TIERS))
    print(f"rho_s  = {stats['pooled_spearman']['rho']:+.4f}  (printed {stats['pooled_spearman']['rho']:+.2f})")
    print(f"beta_1 = {ols['beta_1']:+.4f} +- {ols['beta_1_se']:.4f}  "
          f"(printed {ols['beta_1']:+.2f}+-{ols['beta_1_se']:.2f})")
    print(f"t      = {ols['t_stat']:+.4f}  (printed {ols['t_stat']:+.1f})")
    print(f"wrote {panel_stem.with_suffix('.pdf')} / .png")

    print("full Figure 3: run scripts/make_fig3.py")


if __name__ == "__main__":
    main()
