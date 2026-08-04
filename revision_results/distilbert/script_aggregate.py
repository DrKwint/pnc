"""Aggregate Banking77 P&C raw metrics into per-method summaries + comparison tables."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

OUT = Path("results/banking77_distilbert_pnc")
METHOD_ORDER = ["base_msp", "base_energy", "base_entropy", "mc_dropout",
                "uncorrected", "head_pnc", "ffn_pnc"]
LABEL = {"base_msp": "MSP", "base_energy": "Energy", "base_entropy": "Entropy",
         "mc_dropout": "MC Dropout", "uncorrected": "Uncorrected perturb",
         "head_pnc": "Head P&C", "ffn_pnc": "Internal FFN P&C"}


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return np.nan


def load(raw):
    return list(csv.DictReader(open(raw)))


def agg(vals):
    v = np.array([x for x in vals if np.isfinite(x)])
    if not len(v):
        return None
    m, sd = v.mean(), v.std(ddof=1) if len(v) > 1 else 0.0
    se = sd / np.sqrt(len(v)) if len(v) > 1 else 0.0
    return {"mean": float(m), "sd": float(sd), "se": float(se), "n": int(len(v)),
            "ci95": [float(m - 1.96 * se), float(m + 1.96 * se)]}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--config", default=None); ap.parse_args()
    rows = load(OUT / "metrics" / "raw.csv")
    seeds = sorted({r["seed"] for r in rows})

    # by (method, split, metric) -> aggregate across seeds
    by = {}
    for r in rows:
        for k, v in r.items():
            if k.startswith(("id_", "ood_")):
                by.setdefault((r["method"], r["split"], k), []).append(_f(v))
    summary = {f"{m}|{s}|{k}": agg(vs) for (m, s, k), vs in by.items()}
    (OUT / "metrics" / "across_seeds.json").write_text(json.dumps(summary, indent=2))

    # by_environment-style CSV: method x split x metric means
    with open(OUT / "metrics" / "by_dataset.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["method", "split", "metric", "mean", "se", "n"])
        for (m, s, k), a in sorted(summary.items() if False else
                                   [((m, s, k), summary[f"{m}|{s}|{k}"]) for (m, s, k) in by]):
            if a:
                w.writerow([m, s, k, f"{a['mean']:.4f}", f"{a['se']:.4f}", a["n"]])

    # main comparison table: methods x {ID acc, ID NLL, ID ECE, Near AUROC, Far AUROC, Far FPR95}
    def cell(method, split, metric):
        a = summary.get(f"{method}|{split}|{metric}")
        return f"{a['mean']:.3f}" if a else "—"
    hdr = ["Method", "ID Acc", "ID NLL", "ID ECE", "Near AUROC", "Cross AUROC", "Far AUROC", "Far FPR95"]
    md = ["| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
    tex = ["\\begin{tabular}{l" + "r" * (len(hdr) - 1) + "}", " & ".join(hdr) + " \\\\", "\\hline"]
    tcsv = [hdr]
    for m in METHOD_ORDER:
        if not any(r["method"] == m for r in rows):
            continue
        cells = [LABEL[m], cell(m, "id_eval", "id_accuracy"), cell(m, "id_eval", "id_nll"),
                 cell(m, "id_eval", "id_ece"), cell(m, "near", "ood_auroc"),
                 cell(m, "cross", "ood_auroc"), cell(m, "far", "ood_auroc"),
                 cell(m, "far", "ood_fpr95")]
        md.append("| " + " | ".join(cells) + " |")
        tex.append(" & ".join(cells) + " \\\\")
        tcsv.append(cells)
    tex.append("\\end{tabular}")
    (OUT / "tables").mkdir(exist_ok=True)
    (OUT / "tables" / "banking77_pnc.md").write_text(
        f"# Banking77 DistilBERT post-hoc P&C — {len(seeds)} construction seeds\n\n" + "\n".join(md) + "\n")
    (OUT / "tables" / "banking77_pnc.tex").write_text("\n".join(tex) + "\n")
    with open(OUT / "tables" / "banking77_pnc.csv", "w", newline="") as f:
        csv.writer(f).writerows(tcsv)
    print("\n".join(md))
    print(f"\n[aggregate] {len(seeds)} seeds -> tables/banking77_pnc.{{md,tex,csv}}")


if __name__ == "__main__":
    main()
