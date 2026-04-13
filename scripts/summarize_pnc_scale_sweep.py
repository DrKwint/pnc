#!/usr/bin/env python3
"""Summarize a PnC scale sweep result file (one Luigi run, multiple scales).

Usage: python scripts/summarize_pnc_scale_sweep.py <path-to-sweep-json>
"""

import json
import sys
from pathlib import Path


def main():
    if len(sys.argv) < 2:
        # default: most recent sweep file
        candidates = sorted(
            Path("results/cifar10").glob("openood_v1p5_pnc_single_block*ps*-*.json"),
            key=lambda p: p.stat().st_mtime,
        )
        if not candidates:
            print("No sweep files found.")
            sys.exit(1)
        path = candidates[-1]
        print(f"# Auto-selected most recent sweep: {path.name}")
    else:
        path = Path(sys.argv[1])

    with open(path) as f:
        data = json.load(f)

    print(f"\n## Scale sweep: {path.name}\n")
    print("| Scale | ID Acc | ID NLL | ID ECE | Near AUROC | Near FPR95 | Far AUROC | Far FPR95 |")
    print("|---:|---:|---:|---:|---:|---:|---:|---:|")

    scales = sorted(
        (k for k in data.keys() if isinstance(data[k], dict) and "id_metrics" in data[k]),
        key=lambda s: float(s),
    )
    for scale in scales:
        sub = data[scale]
        idm = sub["id_metrics"]
        primary = sub.get("protocol", {}).get("primary_score", "predictive_entropy")
        near = sub["near_ood"]["aggregate"].get(primary, {})
        far = sub["far_ood"]["aggregate"].get(primary, {})
        print(
            f"| {scale} | {idm.get('accuracy',0)*100:.2f} | {idm.get('nll',0):.4f} | "
            f"{idm.get('ece',0):.4f} | {near.get('mean_auroc',0)*100:.2f} | "
            f"{near.get('mean_fpr95',0)*100:.2f} | {far.get('mean_auroc',0)*100:.2f} | "
            f"{far.get('mean_fpr95',0)*100:.2f} |"
        )

    # Identify which scale is best on each OOD metric
    print("\n### Best scale per metric (single seed)\n")
    metric_keys = [
        ("ID Acc (↑)", "accuracy", True, "id"),
        ("ID NLL (↓)", "nll", False, "id"),
        ("ID ECE (↓)", "ece", False, "id"),
        ("Near AUROC (↑)", "mean_auroc", True, "near_ood"),
        ("Near FPR95 (↓)", "mean_fpr95", False, "near_ood"),
        ("Far AUROC (↑)", "mean_auroc", True, "far_ood"),
        ("Far FPR95 (↓)", "mean_fpr95", False, "far_ood"),
    ]
    for label, key, higher_better, source in metric_keys:
        scored = []
        for scale in scales:
            sub = data[scale]
            primary = sub.get("protocol", {}).get("primary_score", "predictive_entropy")
            if source == "id":
                v = sub["id_metrics"].get(key)
            else:
                v = sub[source]["aggregate"].get(primary, {}).get(key)
            if v is not None:
                scored.append((scale, v))
        if not scored:
            continue
        best = max(scored, key=lambda t: t[1]) if higher_better else min(scored, key=lambda t: t[1])
        print(f"- **{label}**: scale={best[0]} ({best[1]*100:.2f}%)" if "AUROC" in label or "FPR95" in label or "Acc" in label else f"- **{label}**: scale={best[0]} ({best[1]:.4f})")


if __name__ == "__main__":
    main()
