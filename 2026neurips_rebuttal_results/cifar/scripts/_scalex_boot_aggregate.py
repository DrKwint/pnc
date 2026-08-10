#!/usr/bin/env python3
"""Aggregate the perturbation-scale-multiplier and bootstrap-fraction one-factor sweeps.

Reads the main raw CSV plus any per-seed shard CSVs (sensitivity_shard_seed*.csv), then builds
two 3-seed tables reporting the 5 requested metrics: accuracy, near AUROC, near FPR95, far AUROC,
far FPR95 (predictive-entropy score, full OpenOOD sets — identical pipeline to the submitted table).

Grids (all non-varying factors fixed at the original-paper anchor: s3b0, K=20, M=50, ps=25,
lambda=1e-3, subset=1024, bf=0.05, random dirs, temp on ID-val):
  * perturbation scale  {0.5,0.75,1.0,1.25,1.5} x 25  ->  ps {12.5,18.75,25,31.25,37.5}
  * bootstrap fraction  {0.01,0.03,0.05,0.07,0.1}
The center cell (ps=25, bf=0.05) == anchor `scale_ps25.0` and is shared by both grids (reused, not re-run).
"""
from __future__ import annotations
import csv, glob, json
from pathlib import Path
import numpy as np

OUT = Path(__file__).resolve().parent
MAIN = OUT / "sensitivity_cifar_raw.csv"
SEEDS = [0, 1, 2]
METRICS = ["id_acc", "near_auroc", "near_fpr95", "far_auroc", "far_fpr95"]

# (cell_key, label) for each grid; center anchor cell reused in both.
SCALE_GRID = [("scale_ps12.5", "0.50x (ps=12.5)"), ("scale_ps18.75", "0.75x (ps=18.75)"),
              ("scale_ps25.0", "1.00x (ps=25)"), ("scale_ps31.25", "1.25x (ps=31.25)"),
              ("scale_ps37.5", "1.50x (ps=37.5)")]
BOOT_GRID = [("boot_bf0.01", "bf=0.01"), ("boot_bf0.03", "bf=0.03"),
             ("scale_ps25.0", "bf=0.05 (anchor)"), ("boot_bf0.07", "bf=0.07"),
             ("boot_bf0.1", "bf=0.1")]


def load_rows():
    rows = {}  # (cell_key, seed) -> row dict (last write wins; shards are authoritative for their seed)
    files = [MAIN] + sorted(OUT.glob("sensitivity_shard_seed*.csv"))
    for path in files:
        if not Path(path).exists():
            continue
        with open(path) as f:
            for r in csv.DictReader(f):
                try:
                    seed = int(r["seed"])
                except (ValueError, KeyError):
                    continue
                rows[(r["cell_key"], seed)] = r
    return rows


def cell_stats(rows, cell_key):
    """Return dict metric -> (mean, std, n_ok) over the 3 seeds, plus status list."""
    vals = {m: [] for m in METRICS}
    statuses = []
    for s in SEEDS:
        r = rows.get((cell_key, s))
        if r is None:
            statuses.append(f"s{s}:MISSING")
            continue
        statuses.append(f"s{s}:{r.get('status', '?')}")
        if r.get("status") != "ok":
            continue
        for m in METRICS:
            v = r.get(m, "")
            if v not in ("", None):
                vals[m].append(float(v))
    stats = {}
    for m in METRICS:
        a = vals[m]
        stats[m] = (float(np.mean(a)), float(np.std(a)), len(a)) if a else (None, None, 0)
    return stats, statuses


def fmt(t):
    mean, std, n = t
    return f"{mean:.2f}±{std:.2f}" if mean is not None else "n/a"


def build_table(rows, grid, title):
    lines = [f"### {title}", "",
             "| setting | Acc | Near-AUROC | Near-FPR95 | Far-AUROC | Far-FPR95 | seeds |",
             "|---|---|---|---|---|---|---|"]
    table_json = []
    for cell_key, label in grid:
        stats, statuses = cell_stats(rows, cell_key)
        lines.append(f"| {label} | " + " | ".join(fmt(stats[m]) for m in METRICS) +
                     f" | {stats['id_acc'][2]}/3 |")
        table_json.append({"cell_key": cell_key, "label": label,
                           "metrics": {m: {"mean": stats[m][0], "std": stats[m][1], "n": stats[m][2]}
                                       for m in METRICS},
                           "seed_status": statuses})
    return "\n".join(lines), table_json


def main():
    rows = load_rows()
    scale_md, scale_j = build_table(rows, SCALE_GRID, "Perturbation-scale multiplier (bf fixed = 0.05)")
    boot_md, boot_j = build_table(rows, BOOT_GRID, "Bootstrap fraction (ps fixed = 25)")

    md = f"""# CIFAR-10 One-Factor Sweeps: Perturbation Scale & Bootstrap Fraction

**Anchor (original-paper config, held fixed for all non-varying factors):** single-block P&C on
`s3b0` (stage_idx=3, block_idx=0), K=20 directions, M=50 members, ps=25, lambda=1e-3, subset_size=1024,
random directions, bootstrap_frac=0.05, temperature fit on the ID-val split. Score = predictive-entropy;
full OpenOOD v1.5 sets (Near = cifar100, tiny_imagenet; Far = mnist, svhn, textures, places365).
3 seeds (0,1,2), mean±std. The center cell (ps=25, bf=0.05) is the shared anchor, reused in both grids.

{scale_md}

{boot_md}

*Metrics are macro-mean over the OOD datasets, matching the submitted protocol. AUROC/FPR95 in %.*
"""
    (OUT / "sweep_scalex_boot_summary.md").write_text(md)
    json.dump({"scale": scale_j, "boot": boot_j}, open(OUT / "sweep_scalex_boot_agg.json", "w"), indent=2)
    print(md)
    print(f"\nWrote {OUT/'sweep_scalex_boot_summary.md'} and {OUT/'sweep_scalex_boot_agg.json'}")


if __name__ == "__main__":
    main()
