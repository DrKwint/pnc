"""Manuscript tables and figures (spec §25-27).

Every table is generated from the raw result JSON/CSV, so nothing is transcribed by hand.

Figure colours use slots 1 and 2 of the validated reference categorical palette in fixed
order (blue #2a78d6, orange #eb6834), with line style carrying the second distinction so
identity never rests on hue alone. The frontier figure is deliberately three stacked panels
sharing one x-axis rather than one panel with twin y-axes: accuracy, agreement and logit
MSE have unrelated scales, and a dual-axis chart invites false visual correlation.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

C_PNC, C_UNC = "#2a78d6", "#eb6834"          # categorical slots 1 and 2
INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"

MAIN_ROWS = [
    ("Base / MSP", "MSP"),
    ("Energy", "Energy"),
    ("ReAct + Energy", "ReAct+Energy"),
    ("Uncorrected perturb.", "Uncorrected"),
    ("P&C", "P&C"),
]


def _fmt(mean, std, pct=True, nd=2):
    v = mean * 100 if pct else mean
    s = std * 100 if pct else std
    return f"{v:.{nd}f} ± {s:.{nd}f}" if s > 0 else f"{v:.{nd}f}"


def build_main_table(out: Path):
    idm = json.loads((out / "metrics" / "id_final.json").read_text())
    ood = json.loads((out / "metrics" / "ood_results.json").read_text())
    base = json.loads((out / "metrics" / "base_id_metrics.json").read_text())
    agg, summ = idm["aggregate"], ood["summary"]

    rows = []
    for label, key in MAIN_ROWS:
        if key in ("MSP", "Energy", "ReAct+Energy"):
            # deterministic post-hoc scores on the untouched base model: one ID row,
            # no artificial seed variance
            idp = {"top1": (base["top1"], 0.0), "nll": (idm["base"]["nll"], 0.0),
                   "ece": (idm["base"]["ece"], 0.0)}
        else:
            m = agg["pnc" if key == "P&C" else "uncorrected"]
            idp = {"top1": (m["top1"]["mean"], m["top1"]["std"]),
                   "nll": (m["nll"]["mean"], m["nll"]["std"]),
                   "ece": (m["ece"]["mean"], m["ece"]["std"])}
        s = summ[key]
        rows.append({
            "method": label,
            "id_acc": _fmt(*idp["top1"]),
            "id_nll": _fmt(*idp["nll"], pct=False, nd=4),
            "id_ece": _fmt(*idp["ece"], pct=False, nd=4),
            "near_auroc": _fmt(s["near_mean_auroc"]["mean"], s["near_mean_auroc"]["std"]),
            "near_fpr95": _fmt(s["near_mean_fpr95"]["mean"], s["near_mean_fpr95"]["std"]),
            "far_auroc": _fmt(s["far_mean_auroc"]["mean"], s["far_mean_auroc"]["std"]),
            "far_fpr95": _fmt(s["far_mean_fpr95"]["mean"], s["far_mean_fpr95"]["std"]),
        })

    tdir = out / "tables"
    tdir.mkdir(parents=True, exist_ok=True)
    cols = ["method", "id_acc", "id_nll", "id_ece", "near_auroc", "near_fpr95",
            "far_auroc", "far_fpr95"]
    with (tdir / "imagenet_vit_main.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, cols)
        w.writeheader()
        w.writerows(rows)

    hdr = ["Method", "ID Acc", "ID NLL", "ID ECE", "Near AUROC", "Near FPR95",
           "Far AUROC", "Far FPR95"]
    md = ["| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
    md += ["| " + " | ".join(r[c] for c in cols) + " |" for r in rows]
    n_seeds = len(idm["seeds"])
    md += ["", f"ImageNet-1k, ViT-B/16 ({base['checkpoint']['weight_enum']}). "
               f"Accuracy/AUROC/FPR95 in %. P&C and the uncorrected ablation are "
               f"mean ± std over {n_seeds} construction seeds at M=20; MSP, Energy and "
               f"ReAct+Energy are deterministic post-hoc scores on the same base model. "
               f"Near = SSB-hard, NINCO. Far = iNaturalist, Textures, OpenImage-O "
               f"(macro mean over datasets)."]
    (tdir / "imagenet_vit_main.md").write_text("\n".join(md) + "\n")

    tex = [r"\begin{tabular}{lccccccc}", r"\toprule",
           " & ".join(hdr) + r" \\", r"\midrule"]
    for r in rows:
        tex.append(" & ".join(r[c].replace("±", r"$\pm$").replace("&", r"\&")
                              for c in cols) + r" \\")
    tex += [r"\bottomrule", r"\end{tabular}"]
    (tdir / "imagenet_vit_main.tex").write_text("\n".join(tex) + "\n")
    print("\n".join(md[:len(rows) + 2]))
    return rows


def build_hyperparameter_table(out: Path):
    """All Stage-B configurations, ID metrics only (spec §26)."""
    src = list(csv.DictReader((out / "selection" / "all_stage_b.csv").open()))
    cols = ["r_target", "r_realized_median", "n_cal", "lambda", "top1", "nll", "ece",
            "base_agreement", "logit_mse_median", "logit_mse_p99", "passes_gate"]
    dst = out / "tables" / "hyperparameter_search.csv"
    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("w", newline="") as fh:
        w = csv.DictWriter(fh, cols)
        w.writeheader()
        for r in src:
            w.writerow({c: r[c] for c in cols})
    print(f"  wrote {dst} ({len(src)} configurations)")


def build_ncal_ridge_diagnostic(out: Path):
    """n_cal x lambda interpolation/generalisation diagnostic (spec §27)."""
    src = list(csv.DictReader((out / "selection" / "all_stage_b.csv").open()))
    cols = ["r_target", "n_cal", "lambda", "calib_residual", "heldout_cls_residual",
            "uncorrected_heldout_cls_residual", "residual_ratio",
            "logit_mse_median", "logit_mse_p99", "unc_logit_mse_p99", "passes_gate"]
    dst = out / "tables" / "ncal_ridge_diagnostic.csv"
    with dst.open("w", newline="") as fh:
        w = csv.DictWriter(fh, cols)
        w.writeheader()
        for r in src:
            w.writerow({c: r[c] for c in cols})
    print(f"  wrote {dst}")


def frontier_figure(out: Path):
    """ID-stability frontier vs perturbation scale. No OOD quantity appears (spec §26)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = sorted(csv.DictReader((out / "selection" / "all_stage_a.csv").open()),
                  key=lambda r: float(r["r_target"]))
    r = np.array([float(x["r_target"]) for x in rows])
    top1 = np.array([float(x["top1"]) for x in rows]) * 100
    agree = np.array([float(x["base_agreement"]) for x in rows]) * 100
    cmed = np.array([float(x["logit_mse_median"]) for x in rows])
    cp99 = np.array([float(x["logit_mse_p99"]) for x in rows])
    umed = np.array([float(x["unc_logit_mse_median"]) for x in rows])
    up99 = np.array([float(x["unc_logit_mse_p99"]) for x in rows])
    sel = json.loads((out / "selection" / "selected_config.json").read_text())
    base_top1 = sel["base_top1_selection_pool"] * 100

    fig, axes = plt.subplots(3, 1, figsize=(6.2, 7.4), sharex=True)
    for ax in axes:
        ax.set_xscale("log")
        # explicit ticks at the tested scales: the default log minor ticks render as
        # "3x10^-1 4x10^-1" and collide
        ax.set_xticks(r)
        ax.set_xticklabels([f"{v:g}" for v in r])
        ax.xaxis.set_minor_locator(plt.NullLocator())
        ax.xaxis.set_minor_formatter(plt.NullFormatter())
        ax.grid(True, which="major", color=GRID, lw=0.6, zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.tick_params(colors=MUTED, labelsize=9)

    a = axes[0]
    a.axhline(base_top1, color=MUTED, lw=1.2, ls=":", zorder=1)
    a.axhline(base_top1 - 0.25, color="#d03b3b", lw=1.2, ls="--", zorder=1)
    a.plot(r, top1, "-o", color=C_PNC, lw=2, ms=6, zorder=3)
    a.annotate("base", (r[0], base_top1), textcoords="offset points", xytext=(2, 4),
               color=MUTED, fontsize=8)
    a.annotate("gate: −0.25 pp", (r[-1], base_top1 - 0.25), textcoords="offset points",
               xytext=(-4, -12), ha="right", color="#d03b3b", fontsize=8)
    a.set_ylabel("ID top-1 (%)", color=INK, fontsize=10)

    b = axes[1]
    b.axhline(99.0, color="#d03b3b", lw=1.2, ls="--", zorder=1)
    b.plot(r, agree, "-o", color=C_PNC, lw=2, ms=6, zorder=3)
    b.annotate("gate: 99%", (r[-1], 99.0), textcoords="offset points", xytext=(-4, 4),
               ha="right", color="#d03b3b", fontsize=8)
    b.set_ylabel("base agreement (%)", color=INK, fontsize=10)

    c = axes[2]
    c.set_yscale("log")
    c.plot(r, cmed, "-o", color=C_PNC, lw=2, ms=6, label="P&C, median", zorder=3)
    c.plot(r, cp99, "--s", color=C_PNC, lw=2, ms=6, label="P&C, p99", zorder=3)
    c.plot(r, umed, "-o", color=C_UNC, lw=2, ms=6, label="uncorrected, median", zorder=2)
    c.plot(r, up99, "--s", color=C_UNC, lw=2, ms=6, label="uncorrected, p99", zorder=2)
    c.set_ylabel("logit MSE vs base", color=INK, fontsize=10)
    c.set_xlabel(r"perturbation scale  $r=\|\Delta W_1\|_F/\|W_1\|_F$", color=INK,
                 fontsize=10)
    c.legend(frameon=False, fontsize=8, ncol=2, labelcolor=INK)

    sel_r = sel["r_target"]
    for ax in axes:
        ax.axvline(sel_r, color=MUTED, lw=1.0, ls="-", alpha=0.5, zorder=1)
    # placed in the middle panel: the top panel's headroom is taken by the "base" label
    axes[1].annotate(f"selected\nr={sel_r:g}", (sel_r, agree.min()),
                     textcoords="offset points", xytext=(6, 6), color=MUTED, fontsize=8)

    fig.suptitle("ID-stability frontier (ID data only)", color=INK, fontsize=11, y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fdir = out / "figures"
    fdir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(fdir / f"id_stability_frontier.{ext}", dpi=200,
                    facecolor="#fcfcfb")
    plt.close(fig)
    print(f"  wrote {fdir/'id_stability_frontier.png'} (+.pdf)")


def posthoc_figure(out: Path):
    """Post-hoc: OOD vs perturbation scale, P&C against the matched uncorrected ensemble.

    Marked post hoc because it reads OOD data across configurations; per §29 it cannot and
    does not revise the frozen selection. Two panels (Near, Far) rather than twin axes.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    src = out / "metrics" / "posthoc_ood_sensitivity" / "scale_sensitivity.csv"
    rows = list(csv.DictReader(src.open()))
    sel = json.loads((out / "selection" / "selected_config.json").read_text())
    gate_max = 0.5          # largest ID-stable scale from Stage A

    def series(variant, col):
        rs = sorted((x for x in rows if x["variant"] == variant),
                    key=lambda x: float(x["r_target"]))
        return (np.array([float(x["r_target"]) for x in rs]),
                np.array([float(x[col]) for x in rs]) * 100)

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.9))
    for ax, col, name in zip(axes, ("near_auroc", "far_auroc"), ("Near", "Far")):
        rr, y_pnc = series("P&C", col)
        _, y_unc = series("uncorrected", col)
        ax.axvspan(rr.min(), gate_max, color="#2a78d6", alpha=0.06, zorder=0)
        ax.plot(rr, y_pnc, "-o", color=C_PNC, lw=2, ms=6, label="P&C", zorder=3)
        ax.plot(rr, y_unc, "--s", color=C_UNC, lw=2, ms=6, label="uncorrected", zorder=3)
        ax.axvline(sel["r_target"], color=MUTED, lw=1.0, alpha=0.6, zorder=1)
        ax.set_xscale("log")
        ax.set_xticks(rr)
        ax.set_xticklabels([f"{v:g}" for v in rr])
        ax.xaxis.set_minor_locator(plt.NullLocator())
        ax.grid(True, color=GRID, lw=0.6, zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.tick_params(colors=MUTED, labelsize=9)
        ax.set_title(f"{name} OOD", color=INK, fontsize=10)
        ax.set_xlabel(r"perturbation scale $r$", color=INK, fontsize=10)
        ax.set_ylabel("AUROC (%)", color=INK, fontsize=10)
    axes[0].annotate("ID-stable\n(gate)", (0.14, axes[0].get_ylim()[0]),
                     textcoords="offset points", xytext=(2, 8), color=MUTED, fontsize=8)
    axes[0].annotate(f"selected\nr={sel['r_target']:g}", (sel["r_target"],
                     axes[0].get_ylim()[0]), textcoords="offset points", xytext=(5, 26),
                     color=MUTED, fontsize=8)
    axes[0].legend(frameon=False, fontsize=9, labelcolor=INK, loc="upper left")
    fig.suptitle("Post-hoc: the correction is what lets the perturbation grow",
                 color=INK, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fdir = out / "figures"
    fdir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(fdir / f"posthoc_ood_vs_scale.{ext}", dpi=200, facecolor="#fcfcfb")
    plt.close(fig)
    print(f"  wrote {fdir/'posthoc_ood_vs_scale.png'} (+.pdf)")


def build_all(out: Path):
    print("== main table ==")
    build_main_table(out)
    print("\n== hyperparameter search ==")
    build_hyperparameter_table(out)
    build_ncal_ridge_diagnostic(out)
    print("\n== figures ==")
    frontier_figure(out)
    if (out / "metrics" / "posthoc_ood_sensitivity" / "scale_sensitivity.csv").exists():
        posthoc_figure(out)


if __name__ == "__main__":
    build_all(Path("results/neurips_2026_rebuttal/imagenet_vit"))
