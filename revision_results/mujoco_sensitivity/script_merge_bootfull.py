"""Fold the full-pool bootstrap sweep (far_bootfull_raw.csv, factor='bootfull')
into the main sensitivity CSV (far_sensitivity_raw.csv).

Idempotent + safe to re-run while the bootfull sweep is still producing data:
  1. back up far_sensitivity_raw.csv -> .pre_bootfull_merge.bak (once; not overwritten)
  2. drop any existing factor=='bootfull' rows from the main CSV
  3. append the current contents of far_bootfull_raw.csv
Both files share the identical CSV_COLS schema, so this is a pure row union.

Usage:  .venv/bin/python results/neurips_2026_rebuttal/mujoco_sensitivity/scripts/merge_bootfull.py
"""
from __future__ import annotations
import csv
import shutil
from pathlib import Path

AGG = Path(__file__).resolve().parents[1] / "aggregates"
MAIN = AGG / "far_sensitivity_raw.csv"
BOOTFULL = AGG / "far_bootfull_raw.csv"
BACKUP = AGG / "far_sensitivity_raw.csv.pre_bootfull_merge.bak"


def main() -> None:
    if not BOOTFULL.exists():
        raise SystemExit(f"no bootfull data at {BOOTFULL}")
    with open(MAIN, newline="") as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames
        main_rows = list(reader)
    with open(BOOTFULL, newline="") as f:
        bf_reader = csv.DictReader(f)
        if bf_reader.fieldnames != cols:
            raise SystemExit("schema mismatch between main and bootfull CSVs — refusing to merge")
        bf_rows = list(bf_reader)

    if not BACKUP.exists():                       # preserve the pre-merge main once
        shutil.copy2(MAIN, BACKUP)

    kept = [r for r in main_rows if r.get("factor") != "bootfull"]   # drop stale bootfull
    merged = kept + bf_rows
    with open(MAIN, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(merged)

    n_bf_combos = len({(r["environment"], r["seed"]) for r in bf_rows})
    print(f"merged: {len(kept)} non-bootfull rows + {len(bf_rows)} bootfull rows "
          f"({n_bf_combos} env-seed combos) = {len(merged)} total")
    print(f"backup (pre-merge main): {BACKUP.name}")


if __name__ == "__main__":
    main()
