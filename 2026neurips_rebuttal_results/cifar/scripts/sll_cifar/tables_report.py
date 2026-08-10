"""Aggregate SLL-Backbone metrics into tables + SLL_CIFAR_REPORT.md.

Final comparison table includes LLLA, SCOD-1024 (from the completed SCOD run), SLL-Backbone (new),
P&C, and Standard (Deep) Ensemble. Mean +/- sample std over 3 checkpoint seeds. Documents the
degeneracy of the faithful variance selection and the resource-adapted predictive-variance
selection actually used.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / "results" / "neurips_2026_rebuttal" / "cifar" / "sll"
NEAR = ["cifar100", "tiny_imagenet"]; FAR = ["mnist", "svhn", "textures", "places365"]

# reproduced submitted comparators + completed SCOD (Acc, NearAUROC, NearFPR95, FarAUROC, FarFPR95)
COMPARATORS = [
    ("LLLA (n=50)", 95.77, 88.97, 54.10, 93.04, 28.40),
    ("SCOD-1024", None, None, None, None, None),         # filled from scod_aggregate.json
    ("SLL-Backbone", None, None, None, None, None),      # filled from this run
    ("P&C s3b0 (M=50)", 95.59, 91.55, 33.08, 95.09, 18.15),
    ("Standard Ensemble (n=5)", 96.56, 91.10, 40.40, 94.63, 19.50),
]


def ms(vals):
    vals = [v for v in vals if v is not None]
    if not vals: return (None, None)
    return (float(np.mean(vals)), float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0)


def fmt(t, dp=2):
    if t is None or t[0] is None: return "n/a"
    return f"{t[0]:.{dp}f}±{t[1]:.{dp}f}" if t[1] else f"{t[0]:.{dp}f}"


def load_scod():
    p = REPO / "results" / "scod_cifar" / "tables" / "scod_aggregate.json"
    if not p.exists(): return None
    a = json.load(open(p))
    return dict(acc=a["acc"][0], near_au=a["near"]["auroc"][0], near_fp=a["near"]["fpr95"][0],
               far_au=a["far"]["auroc"][0], far_fp=a["far"]["fpr95"][0])


def main():
    seeds = [0, 1, 2]
    M = {s: json.load(open(ROOT / "metrics" / f"sll_backbone_seed{s}.json")) for s in seeds
         if (ROOT / "metrics" / f"sll_backbone_seed{s}.json").exists()}
    seeds = sorted(M.keys())
    tdir = ROOT / "tables"; tdir.mkdir(parents=True, exist_ok=True)

    acc = ms([M[s]["id_metrics"]["accuracy"] for s in seeds])
    nll = ms([M[s]["id_metrics"]["nll"] for s in seeds])
    ece = ms([M[s]["id_metrics"]["ece"] for s in seeds]); brier = ms([M[s]["id_metrics"]["brier"] for s in seeds])
    near_au = ms([M[s]["near"]["auroc"] for s in seeds]); near_fp = ms([M[s]["near"]["fpr95"] for s in seeds])
    far_au = ms([M[s]["far"]["auroc"] for s in seeds]); far_fp = ms([M[s]["far"]["fpr95"] for s in seeds])
    S = M[seeds[0]]["S"]

    # main table
    scod = load_scod()
    rows = ["| Method | Acc | Near AUROC | Near FPR95 | Far AUROC | Far FPR95 |", "|---|---:|---:|---:|---:|---:|"]
    for name, a, na, nf, fa, ff in COMPARATORS:
        if name == "SLL-Backbone":
            rows.append(f"| **SLL-Backbone (S={S})** | {fmt(acc)} | {fmt(near_au)} | {fmt(near_fp)} "
                        f"| {fmt(far_au)} | {fmt(far_fp)} |")
        elif name == "SCOD-1024" and scod:
            rows.append(f"| SCOD-1024 | {scod['acc']:.2f} | {scod['near_au']:.2f} | {scod['near_fp']:.2f} "
                        f"| {scod['far_au']:.2f} | {scod['far_fp']:.2f} |")
        else:
            rows.append(f"| {name} | {a:.2f} | {na:.2f} | {nf:.2f} | {fa:.2f} | {ff:.2f} |")
    (tdir / "cifar10_with_sll.md").write_text(
        "# CIFAR-10 OpenOOD v1.5 -- SLL-Backbone vs post-hoc / ensemble methods\n\n"
        f"SLL-Backbone: full-covariance linearized Laplace over S={S} backbone weights selected by "
        "predictive-variance contribution (ID-only). Probit predictive entropy. Mean ± sample-std, "
        "3 seeds.\n\n" + "\n".join(rows) + "\n")

    # per-dataset
    def per_ds(metric):
        allds = NEAR + FAR
        lines = ["| Method | " + " | ".join(allds) + " |", "|" + "---|" * (len(allds) + 1)]
        cells = [fmt(ms([M[s]["per_dataset"][d][metric] for s in seeds])) for d in allds]
        lines.append(f"| SLL-Backbone (S={S}) | " + " | ".join(cells) + " |")
        return "\n".join(lines)
    (tdir / "sll_per_dataset.md").write_text(
        f"# SLL-Backbone per-dataset (mean±std, 3 seeds)\n\n## AUROC\n{per_ds('auroc')}\n\n## FPR95\n{per_ds('fpr95')}\n")

    # aggregate
    agg = dict(S=S, acc=acc, nll=nll, ece=ece, brier=brier,
               near=dict(auroc=near_au, fpr95=near_fp), far=dict(auroc=far_au, fpr95=far_fp),
               lam_star=[M[s]["lam_star"] for s in seeds], T_star=[M[s]["T_star"] for s in seeds])
    (tdir / "sll_aggregate.md").write_text(
        f"# SLL-Backbone aggregate (3 seeds)\n\n"
        f"| | Acc | NLL | ECE | Brier | Near AUROC | Near FPR95 | Far AUROC | Far FPR95 |\n"
        f"|---|---|---|---|---|---|---|---|---|\n"
        f"| SLL-Backbone (S={S}) | {fmt(acc)} | {fmt(nll,3)} | {fmt(ece,3)} | {fmt(brier,3)} | "
        f"{fmt(near_au)} | {fmt(near_fp)} | {fmt(far_au)} | {fmt(far_fp)} |\n")
    json.dump(agg, open(tdir / "sll_aggregate.json", "w"), indent=2, default=float)
    print(open(tdir / "cifar10_with_sll.md").read())
    return agg, S


if __name__ == "__main__":
    main()
