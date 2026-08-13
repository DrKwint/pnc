"""Tables and the preservation/diversity frontier figure (spec §19-20).

Colours are slots 1 and 2 of the validated reference categorical palette in fixed order
(blue = P&C, orange = matched uncorrected), with marker/line style carrying the second
distinction so identity never rests on hue alone. The frontier is three stacked panels
sharing one x-axis rather than one panel with several y-scales: ID accuracy change,
logit deviation and AUROC are unrelated quantities and a shared axis would invite false
visual correlation.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from .frontier import BUDGETS, OUT

C_PNC, C_UNC = "#2a78d6", "#eb6834"
INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"
BUDGET_COLOR = {"strict": "#d03b3b", "primary": "#ec835a", "relaxed": "#fab219"}


def _grid() -> list[dict]:
    rows = list(csv.DictReader((OUT / "selection" / "frontier_grid.csv").open()))
    for r in rows:
        for k, v in list(r.items()):
            if k in ("all_finite", "pass_strict", "pass_primary", "pass_relaxed"):
                r[k] = v == "True"
            elif k in ("pathology",):
                pass
            else:
                try:
                    r[k] = float(v)
                except (TypeError, ValueError):
                    pass
    return rows


def _fallback(candidates):
    """Most-preserving ridge among NON-pathological ones.

    Never fall back to a CALIBRATION_PATHOLOGY configuration: those buy accuracy by
    wrecking calibration and the protocol rejects them, so showing them as "the best
    available" at a failing scale would misrepresent the frontier.
    """
    clean = [c for c in candidates if not c["pathology"]]
    return max(clean or candidates, key=lambda x: x["lcb_pp"])


def _best_per_scale(rows, budget):
    from .frontier import best_lambda
    by_r = {}
    for r in rows:
        by_r.setdefault(r["r_target"], []).append(r)
    return {r: best_lambda(by_r[r], budget) for r in sorted(by_r)}


def table1_frontier(out: Path = OUT):
    """Table 1 — ID preservation frontier, one row per scale at its best ridge."""
    rows = _grid()
    by_r = {}
    for r in rows:
        by_r.setdefault(r["r_target"], []).append(r)
    # "best lambda" for the table is the PRIMARY-budget choice where one exists,
    # otherwise the ridge with the highest LCB (the most preserving available)
    from .frontier import best_lambda
    out_rows = []
    for r in sorted(by_r):
        b = best_lambda(by_r[r], "primary") or _fallback(by_r[r])
        out_rows.append({
            "r_target": r, "realized_r_median": round(b["realized_r_median"], 4),
            "best_lambda": b["lambda"],
            "corrected_top1": round(b["top1"] * 100, 3),
            "delta_top1_pp": round(b["delta_top1_pp"], 3),
            "lcb_95_pp": round(b["lcb_pp"], 3),
            "nll": round(b["nll"], 4), "ece": round(b["ece"], 4),
            "base_agreement": round(b["base_agreement"], 4),
            "uncorrected_delta_top1_pp": round(b["unc_delta_top1_pp"], 3),
            "pass_strict": any(x["pass_strict"] for x in by_r[r]),
            "pass_primary": any(x["pass_primary"] for x in by_r[r]),
            "pass_relaxed": any(x["pass_relaxed"] for x in by_r[r]),
            "pathology": b["pathology"],
        })
    d = out / "tables"
    d.mkdir(parents=True, exist_ok=True)
    with (d / "table1_preservation_frontier.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, list(out_rows[0]))
        w.writeheader()
        w.writerows(out_rows)
    hdr = ["r", "realized r", "λ", "top-1", "Δ pp", "95% LCB pp", "NLL", "ECE",
           "agree", "unc Δ pp", "S", "P", "R"]
    md = ["| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
    for r in out_rows:
        md.append("| " + " | ".join([
            f"{r['r_target']:g}", f"{r['realized_r_median']:.3f}", f"{r['best_lambda']:g}",
            f"{r['corrected_top1']:.3f}", f"{r['delta_top1_pp']:+.3f}",
            f"{r['lcb_95_pp']:+.3f}", f"{r['nll']:.4f}", f"{r['ece']:.4f}",
            f"{r['base_agreement']:.4f}", f"{r['uncorrected_delta_top1_pp']:+.3f}",
            "✓" if r["pass_strict"] else "·", "✓" if r["pass_primary"] else "·",
            "✓" if r["pass_relaxed"] else "·"]) + " |")
    (d / "table1_preservation_frontier.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    return out_rows


def table2_operating_points(out: Path = OUT):
    frozen = json.loads((out / "selection" / "frozen_configs.json").read_text())
    idf = json.loads((out / "metrics" / "id_final.json").read_text())
    ood = json.loads((out / "metrics" / "ood_results.json").read_text())
    rows = []
    for name in ("strict", "primary", "relaxed"):
        cfg = frozen["configs"].get(name)
        if cfg is None or name not in idf["configs"]:
            continue
        a = idf["configs"][name]["aggregate"]["pnc"]
        s = ood["summary"][f"P&C[{name}]"]
        rows.append({
            "budget": name, "max_loss_pp": cfg["max_top1_loss_pp"],
            "r": cfg["r_target"], "realized_r": round(cfg["realized_r_median"], 4),
            "lambda": cfg["lambda"],
            "id_acc": f"{a['top1']['mean']*100:.3f} ± {a['top1']['std']*100:.3f}",
            "id_nll": f"{a['nll']['mean']:.4f}", "id_ece": f"{a['ece']['mean']:.4f}",
            "base_agreement": f"{a['base_agreement']['mean']:.4f}",
            "near_auroc": f"{s['near_mean_auroc']['mean']*100:.2f} ± {s['near_mean_auroc']['std']*100:.2f}",
            "near_fpr95": f"{s['near_mean_fpr95']['mean']*100:.2f}",
            "far_auroc": f"{s['far_mean_auroc']['mean']*100:.2f} ± {s['far_mean_auroc']['std']*100:.2f}",
            "far_fpr95": f"{s['far_mean_fpr95']['mean']*100:.2f}",
        })
    for m in ("MSP", "Energy", "ReAct+Energy"):
        s = ood["summary"][m]
        rows.append({"budget": m, "max_loss_pp": "", "r": "", "realized_r": "",
                     "lambda": "",
                     "id_acc": f"{idf['base']['top1']*100:.3f}",
                     "id_nll": f"{idf['base']['nll']:.4f}",
                     "id_ece": f"{idf['base']['ece']:.4f}", "base_agreement": "1.0000",
                     "near_auroc": f"{s['near_mean_auroc']['mean']*100:.2f}",
                     "near_fpr95": f"{s['near_mean_fpr95']['mean']*100:.2f}",
                     "far_auroc": f"{s['far_mean_auroc']['mean']*100:.2f}",
                     "far_fpr95": f"{s['far_mean_fpr95']['mean']*100:.2f}"})
    d = out / "tables"
    with (d / "table2_operating_points.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    hdr = ["Config", "budget pp", "r", "λ", "ID Acc", "ID NLL", "ID ECE", "agree",
           "Near AUROC", "Near FPR95", "Far AUROC", "Far FPR95"]
    cols = ["budget", "max_loss_pp", "r", "lambda", "id_acc", "id_nll", "id_ece",
            "base_agreement", "near_auroc", "near_fpr95", "far_auroc", "far_fpr95"]
    md = ["| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
    md += ["| " + " | ".join(str(r[c]) for c in cols) + " |" for r in rows]
    (d / "table2_operating_points.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    return rows


def table3_ablation(out: Path = OUT):
    frozen = json.loads((out / "selection" / "frozen_configs.json").read_text())
    idf = json.loads((out / "metrics" / "id_final.json").read_text())
    ood = json.loads((out / "metrics" / "ood_results.json").read_text())
    rows = []
    for name in ("strict", "primary", "relaxed"):
        if name not in idf["configs"]:
            continue
        cfg = frozen["configs"][name]
        p = idf["configs"][name]["aggregate"]["pnc"]
        u = idf["configs"][name]["aggregate"]["uncorrected"]
        sp = ood["summary"][f"P&C[{name}]"]
        su = ood["summary"][f"Uncorrected[{name}]"]
        rows.append({
            "budget": name, "r": cfg["r_target"], "lambda": cfg["lambda"],
            "pnc_id_acc": round(p["top1"]["mean"] * 100, 3),
            "unc_id_acc": round(u["top1"]["mean"] * 100, 3),
            "pnc_nll": round(p["nll"]["mean"], 4), "unc_nll": round(u["nll"]["mean"], 4),
            "pnc_agreement": round(p["base_agreement"]["mean"], 4),
            "unc_agreement": round(u["base_agreement"]["mean"], 4),
            "pnc_near_auroc": round(sp["near_mean_auroc"]["mean"] * 100, 2),
            "unc_near_auroc": round(su["near_mean_auroc"]["mean"] * 100, 2),
            "pnc_far_auroc": round(sp["far_mean_auroc"]["mean"] * 100, 2),
            "unc_far_auroc": round(su["far_mean_auroc"]["mean"] * 100, 2),
            "median_logit_mse_ratio": round(
                u["logit_mse_median"]["mean"] / p["logit_mse_median"]["mean"], 3),
            "p99_logit_mse_ratio": round(
                u["logit_mse_p99"]["mean"] / p["logit_mse_p99"]["mean"], 3),
        })
    d = out / "tables"
    with (d / "table3_correction_ablation.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    hdr = ["budget", "r", "λ", "P&C acc", "unc acc", "P&C NLL", "unc NLL",
           "P&C Near", "unc Near", "P&C Far", "unc Far", "med MSE ratio", "p99 ratio"]
    cols = ["budget", "r", "lambda", "pnc_id_acc", "unc_id_acc", "pnc_nll", "unc_nll",
            "pnc_near_auroc", "unc_near_auroc", "pnc_far_auroc", "unc_far_auroc",
            "median_logit_mse_ratio", "p99_logit_mse_ratio"]
    md = ["| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
    md += ["| " + " | ".join(str(r[c]) for c in cols) + " |" for r in rows]
    (d / "table3_correction_ablation.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    return rows


def frontier_figure(out: Path = OUT, with_ood: bool = True):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = _grid()
    best = _best_per_scale(rows, "primary")
    scales = sorted({r["r_target"] for r in rows})
    frozen = json.loads((out / "selection" / "frozen_configs.json").read_text())

    # per scale, the most-preserving available ridge (for the part of the curve that fails)
    by_r = {}
    for r in rows:
        by_r.setdefault(r["r_target"], []).append(r)
    pick = {r: (best[r] or _fallback(by_r[r])) for r in scales}
    x = np.array(scales)
    dtop = np.array([pick[r]["delta_top1_pp"] for r in scales])
    lcb = np.array([pick[r]["lcb_pp"] for r in scales])
    udtop = np.array([pick[r]["unc_delta_top1_pp"] for r in scales])
    cmed = np.array([pick[r]["logit_mse_median"] for r in scales])
    umed = np.array([pick[r]["unc_logit_mse_median"] for r in scales])

    n_panels = 3 if with_ood and (out / "metrics" / "ood_results.json").exists() else 2
    fig, axes = plt.subplots(n_panels, 1, figsize=(6.6, 3.0 * n_panels), sharex=True)
    # label a readable subset: the bisection points 2.125/2.25 sit on top of 2.5
    label_at = {0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 6.0, 8.0}
    for ax in np.atleast_1d(axes):
        ax.set_xscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{v:g}" if v in label_at else "" for v in x])
        ax.xaxis.set_minor_locator(plt.NullLocator())
        ax.grid(True, color=GRID, lw=0.6, zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.tick_params(colors=MUTED, labelsize=9)

    # Panel A — corrected ID preservation vs budgets
    a = axes[0]
    for name, eps in BUDGETS.items():
        a.axhline(-eps * 100, color=BUDGET_COLOR[name], lw=1.2, ls="--", zorder=1)
        a.annotate(f"{name} −{eps*100:g} pp", (x[-1], -eps * 100),
                   textcoords="offset points", xytext=(-3, 3), ha="right",
                   color=BUDGET_COLOR[name], fontsize=8)
    a.axhline(0, color=MUTED, lw=1.0, ls=":", zorder=1)
    a.fill_between(x, lcb, dtop, color=C_PNC, alpha=0.15, zorder=2, lw=0)
    a.plot(x, dtop, "-o", color=C_PNC, lw=2, ms=6, zorder=3, label="P&C (best ID-selected λ)")
    a.plot(x, lcb, "-", color=C_PNC, lw=1, alpha=0.7, zorder=3, label="95% lower bound")
    a.set_ylabel("ID top-1 change (pp)", color=INK, fontsize=10)
    a.legend(frameon=False, fontsize=8, labelcolor=INK, loc="lower left")

    # Panel B — correction benefit on ID
    b = axes[1]
    b.plot(x, dtop, "-o", color=C_PNC, lw=2, ms=6, label="P&C", zorder=3)
    b.plot(x, udtop, "--s", color=C_UNC, lw=2, ms=6, label="uncorrected", zorder=3)
    b.axhline(0, color=MUTED, lw=1.0, ls=":", zorder=1)
    # symlog: the uncorrected collapse at r=4 (-10.8 pp) would otherwise flatten the
    # whole r <= 2.5 region where the two curves actually cross
    b.set_yscale("symlog", linthresh=0.5, linscale=0.6)
    b.set_ylabel("ID top-1 change (pp)", color=INK, fontsize=10)
    b.legend(frameon=False, fontsize=9, labelcolor=INK, loc="lower left")
    bt = b.twinx() if False else None    # never a dual axis; MSE lives in its own figure

    # Panel C — OOD ranking at the frozen points
    if n_panels == 3:
        ood = json.loads((out / "metrics" / "ood_results.json").read_text())
        c = axes[2]
        pts = []
        for name in ("strict", "primary", "relaxed"):
            key = f"P&C[{name}]"
            if key not in ood["summary"]:
                continue
            cfg = frozen["configs"][name]
            pts.append((cfg["r_target"],
                        ood["summary"][key]["near_mean_auroc"]["mean"] * 100,
                        ood["summary"][f"Uncorrected[{name}]"]["near_mean_auroc"]["mean"] * 100,
                        name))
        if pts:
            pr = [p[0] for p in pts]
            c.plot(pr, [p[1] for p in pts], "-o", color=C_PNC, lw=2, ms=8, label="P&C",
                   zorder=3)
            c.plot(pr, [p[2] for p in pts], "--s", color=C_UNC, lw=2, ms=8,
                   label="uncorrected", zorder=3)
            for r_, y_, _, name in pts:
                c.annotate(name, (r_, y_), textcoords="offset points", xytext=(0, 9),
                           ha="center", color=BUDGET_COLOR[name], fontsize=8)
        c.set_ylabel("Near AUROC (%)", color=INK, fontsize=10)
        c.legend(frameon=False, fontsize=9, labelcolor=INK, loc="lower left")

    np.atleast_1d(axes)[-1].set_xlabel(
        r"perturbation scale  $r=\|\Delta W_1\|_F/\|W_1\|_F$", color=INK, fontsize=10)
    fig.suptitle("Preservation / diversity frontier (scale chosen by corrected-model "
                 "ID budget)", color=INK, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    d = out / "figures"
    d.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(d / f"preservation_frontier.{ext}", dpi=200, facecolor="#fcfcfb")
    plt.close(fig)
    print(f"  wrote {d/'preservation_frontier.png'} (+.pdf)")


def build_all(out: Path = OUT):
    print("== Table 1: ID preservation frontier ==")
    table1_frontier(out)
    if (out / "metrics" / "ood_results.json").exists():
        print("\n== Table 2: operating points ==")
        table2_operating_points(out)
        print("\n== Table 3: correction ablation ==")
        table3_ablation(out)
    print("\n== figure ==")
    frontier_figure(out)


if __name__ == "__main__":
    build_all()
