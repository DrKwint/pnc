"""Stage 3 -- aggregate per-seed SCOD metrics into the paper tables (Section 16).

Produces (mean +/- sample-std over the 3 checkpoint seeds, matching the submitted convention):
  * tables/cifar10_with_scod.{md,tex,csv} -- main table: SCOD-1024 row alongside the reproduced
    submitted comparators (MSP, Mahalanobis, MC-Dropout, LLLA, SWAG, P&C, Deep Ensemble).
  * tables/scod_per_dataset_auroc.md / _fpr95.md -- per-OOD-dataset breakdown.
  * tables/scod_sensitivity.md -- rank (k in 5,10,20) and Meps grid (ID-declared, NOT OOD-selected).
Accuracy/NLL are the UNCHANGED base classifier's (SCOD does not alter predictions).
"""
from __future__ import annotations

import argparse, json
from pathlib import Path
import numpy as np, yaml

REPO = Path(__file__).resolve().parents[2]

# Reproduced submitted comparators (from results/.../CIFAR_REVISED_BENCHMARK.md 16.1; 3-seed).
SUBMITTED = [
    ("PreActResNet-18 / MSP", 95.74, 87.70, 66.30, 91.50, 38.30),
    ("Mahalanobis",           95.74, 87.98, 66.30, 93.25, 38.30),
    ("MC Dropout (n=32)",     95.76, 87.25, 71.00, 91.34, 42.50),
    ("LLLA (n=50)",           95.77, 88.97, 54.10, 93.04, 28.40),
    ("SWAG (n=50)",           95.37, 90.03, 44.70, 94.19, 22.10),
    ("P&C s3b0 (M=50)",       95.59, 91.55, 33.08, 95.09, 18.15),
    ("Deep Ensemble (n=5)",   96.56, 91.10, 40.40, 94.63, 19.50),
]


def _cfg(p):
    with open(p) as f:
        return yaml.safe_load(f)


def mean_std(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return (None, None)
    mu = float(np.mean(vals))
    sd = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
    return (mu, sd)


def fmt(ms, dp=2):
    mu, sd = ms
    return "n/a" if mu is None else (f"{mu:.{dp}f}±{sd:.{dp}f}" if sd else f"{mu:.{dp}f}")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--config", required=True)
    args = ap.parse_args(); cfg = _cfg(args.config)
    root = REPO / cfg["root"]; tdir = root / "tables"; tdir.mkdir(parents=True, exist_ok=True)
    seeds = cfg["model_seeds"]
    k_prim = int(cfg["reported_num_eigs"]); meps_prim = int(float(cfg["Meps"]))
    prim_key = f"tempered_k{k_prim}_Meps{meps_prim}"
    near_ds = list(cfg["near_ood"]); far_ds = list(cfg["far_ood"])

    metrics = {s: json.load(open(root / "metrics" / f"seed{s}_metrics.json")) for s in seeds}

    def collect(fam, key):
        return mean_std([metrics[s]["variants"][prim_key][fam][key] for s in seeds])
    acc = mean_std([metrics[s]["base_id"]["id_test_acc"] for s in seeds])
    nll = mean_std([metrics[s]["base_id"]["id_test_nll"] for s in seeds])
    near_au, near_fp = collect("near", "auroc"), collect("near", "fpr95")
    far_au, far_fp = collect("far", "auroc"), collect("far", "fpr95")

    # ---------- main table ----------
    header = "| Method | Acc % ↑ | Near AUROC ↑ | Near FPR95 ↓ | Far AUROC ↑ | Far FPR95 ↓ |"
    sep = "|---|---:|---:|---:|---:|---:|"
    rows = [header, sep]
    for name, a, na, nf, fa, ff in SUBMITTED:
        rows.append(f"| {name} | {a:.2f} | {na:.2f} | {nf:.2f} | {fa:.2f} | {ff:.2f} |")
        if name.startswith("SWAG"):  # insert SCOD just before P&C for adjacency
            rows.append(f"| **SCOD-1024 (k={k_prim})** | {fmt(acc)} | {fmt(near_au)} | {fmt(near_fp)} "
                        f"| {fmt(far_au)} | {fmt(far_fp)} |")
    md = ("# CIFAR-10 OpenOOD v1.5 -- SCOD-1024 vs submitted methods\n\n"
          f"Primary SCOD variant: **SCOD-1024**, k={k_prim}, Meps={meps_prim}, T=6k_max+4="
          f"{cfg['num_samples']}, tempered posterior_pred score, base-classifier temperature on ID-val. "
          "Mean ± sample-std over 3 checkpoint seeds. SCOD Acc/NLL are the unchanged base classifier's. "
          "Comparator rows are the reproduced submitted 3-seed table.\n\n"
          + "\n".join(rows) + "\n\n"
          f"SCOD-1024 ID: Acc {fmt(acc)}, NLL {fmt(nll,3)}. Score orientation: higher = more OOD.\n")
    (tdir / "cifar10_with_scod.md").write_text(md)

    # csv
    import csv
    with open(tdir / "cifar10_with_scod.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["method", "acc", "near_auroc", "near_fpr95", "far_auroc", "far_fpr95"])
        for name, a, na, nf, fa, ff in SUBMITTED:
            w.writerow([name, a, na, nf, fa, ff])
        w.writerow([f"SCOD-1024_k{k_prim}", fmt(acc), fmt(near_au), fmt(near_fp), fmt(far_au), fmt(far_fp)])

    # tex
    tex = ["\\begin{tabular}{lrrrrr}", "\\toprule",
           "Method & Acc & Near AUROC & Near FPR95 & Far AUROC & Far FPR95 \\\\", "\\midrule"]
    for name, a, na, nf, fa, ff in SUBMITTED:
        tex.append(f"{name} & {a:.2f} & {na:.2f} & {nf:.2f} & {fa:.2f} & {ff:.2f} \\\\")
    tex.append(f"SCOD-1024 (k={k_prim}) & {fmt(acc)} & {fmt(near_au)} & {fmt(near_fp)} & {fmt(far_au)} & {fmt(far_fp)} \\\\")
    tex += ["\\bottomrule", "\\end{tabular}"]
    (tdir / "cifar10_with_scod.tex").write_text("\n".join(tex))

    # ---------- per-dataset tables ----------
    def per_ds_table(metric):
        allds = near_ds + far_ds
        head = "| Method | " + " | ".join(allds) + " |"
        rows = [head, "|" + "---|" * (len(allds) + 1)]
        cells = []
        for d in allds:
            ms = mean_std([metrics[s]["variants"][prim_key]["per_dataset"][d][metric] for s in seeds])
            cells.append(fmt(ms))
        rows.append(f"| SCOD-1024 (k={k_prim}) | " + " | ".join(cells) + " |")
        return "\n".join(rows)
    (tdir / "scod_per_dataset_auroc.md").write_text(
        f"# SCOD-1024 per-dataset AUROC (mean±std, 3 seeds)\n\n" + per_ds_table("auroc") + "\n")
    (tdir / "scod_per_dataset_fpr95.md").write_text(
        f"# SCOD-1024 per-dataset FPR95 (mean±std, 3 seeds)\n\n" + per_ds_table("fpr95") + "\n")

    # ---------- sensitivity grid (k x Meps), ID-declared ----------
    k_list = cfg["sensitivity"]["num_eigs"]; meps_list = cfg["sensitivity"]["Meps"]
    lines = ["# SCOD-1024 sensitivity (rank k x Meps) -- 3-seed mean Near/Far AUROC",
             "*ID-predeclared grid; NOT selected on OOD. Primary row is k=10, Meps=5000.*", "",
             "| variant | Near AUROC | Near FPR95 | Far AUROC | Far FPR95 |", "|---|---:|---:|---:|---:|"]
    for k in k_list:
        for m in meps_list:
            key = f"tempered_k{k}_Meps{int(m)}"
            if key not in metrics[seeds[0]]["variants"]:
                continue
            na = mean_std([metrics[s]["variants"][key]["near"]["auroc"] for s in seeds])
            nf = mean_std([metrics[s]["variants"][key]["near"]["fpr95"] for s in seeds])
            fa = mean_std([metrics[s]["variants"][key]["far"]["auroc"] for s in seeds])
            ff = mean_std([metrics[s]["variants"][key]["far"]["fpr95"] for s in seeds])
            star = " **(primary)**" if (k == k_prim and int(m) == meps_prim) else ""
            lines.append(f"| k={k}, Meps={int(m)}{star} | {fmt(na)} | {fmt(nf)} | {fmt(fa)} | {fmt(ff)} |")
    # untempered diagnostic
    ukey = f"untempered_k{k_prim}_Meps{meps_prim}"
    if ukey in metrics[seeds[0]]["variants"]:
        na = mean_std([metrics[s]["variants"][ukey]["near"]["auroc"] for s in seeds])
        fa = mean_std([metrics[s]["variants"][ukey]["far"]["auroc"] for s in seeds])
        nf = mean_std([metrics[s]["variants"][ukey]["near"]["fpr95"] for s in seeds])
        ff = mean_std([metrics[s]["variants"][ukey]["far"]["fpr95"] for s in seeds])
        lines.append(f"| untempered (k={k_prim}, diag) | {fmt(na)} | {fmt(nf)} | {fmt(fa)} | {fmt(ff)} |")
    (tdir / "scod_sensitivity.md").write_text("\n".join(lines) + "\n")

    # machine-readable aggregate
    agg = dict(primary=prim_key, seeds=seeds, acc=acc, nll=nll,
               near=dict(auroc=near_au, fpr95=near_fp), far=dict(auroc=far_au, fpr95=far_fp))
    json.dump(agg, open(tdir / "scod_aggregate.json", "w"), indent=2)
    print(md)
    print(f"\n[aggregate] wrote tables to {tdir}")


if __name__ == "__main__":
    main()
