#!/usr/bin/env python
"""Recover the submission-era CIFAR-10 OpenOOD results from git and normalize them.

Why this exists
---------------
`results/cifar10/` holds 137 result JSONs in git HEAD (70fb480) but EVERY ONE of them
is deleted in the working tree. They are the only CIFAR artifacts on this machine:
the rebuttal-era CIFAR runs (sensitivity cross-product, SCOD comparison, efficiency
accounting) were executed on a different machine and were never copied back here.

What this produces
------------------
  revision_results/cifar_sensitivity/cifar10_openood_recovered_from_git.csv
      one row per (result file, perturbation scale) with accuracy, ID NLL/ECE,
      Near/Far AUROC and FPR95, fitted temperature, and the OpenOOD protocol flags.
  revision_results/cifar_sensitivity/raw_json_from_git/
      the P&C and baseline OpenOOD JSONs themselves, byte-identical to HEAD.

Both P&C file families are kept: single-block (`s3b1` = stage 4 / block 1) and
multi-block. Scores are `predictive_entropy` of the ensemble-mean softmax.

Usage: .venv/bin/python revision_results/provenance/extract_cifar_from_git.py
"""
from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "revision_results/cifar_sensitivity"
RAWDIR = OUT / "raw_json_from_git"


def git_files() -> list[str]:
    r = subprocess.run(["git", "ls-tree", "-r", "HEAD", "--name-only"], cwd=ROOT,
                       capture_output=True, text=True, check=True)
    return [p for p in r.stdout.splitlines() if p.startswith("results/cifar10/openood_")]


def blob(rel: str) -> bytes:
    return subprocess.run(["git", "show", f"HEAD:{rel}"], cwd=ROOT,
                          capture_output=True, check=True).stdout


def flatten(name: str, payload: dict) -> list[dict]:
    """A result JSON is either a metrics dict or a {scale: metrics} mapping."""
    blocks = (
        {"": payload}
        if "id_metrics" in payload
        else {k: v for k, v in payload.items() if isinstance(v, dict)}
    )
    seed = (re.search(r"_seed(\d+)", name) or [None, ""])[1]
    method = re.sub(r"_e300.*", "", name).replace("openood_v1p5_", "")
    rows = []
    for scale, v in blocks.items():
        idm = v.get("id_metrics", {})
        proto = v.get("protocol", {})
        near = v.get("near_ood", {}).get("aggregate", {}).get("predictive_entropy", {})
        far = v.get("far_ood", {}).get("aggregate", {}).get("predictive_entropy", {})
        rows.append(
            dict(
                source_file=name,
                method=method,
                seed=seed,
                perturbation_scale=scale,
                ensemble_name=v.get("ensemble_name"),
                accuracy=idm.get("accuracy"),
                id_nll=idm.get("nll"),
                id_ece=idm.get("ece"),
                id_brier=idm.get("brier"),
                near_ood_auroc=v.get("near_ood_auroc"),
                near_ood_fpr95=near.get("mean_fpr95"),
                far_ood_auroc=v.get("far_ood_auroc"),
                far_ood_fpr95=far.get("mean_fpr95"),
                posthoc_temperature=v.get("posthoc_temperature"),
                posthoc_calibrate=v.get("posthoc_calibrate"),
                primary_score=proto.get("primary_score"),
                temperature_fit_split=proto.get("temperature_fit_split"),
                ood_validation_used=proto.get("ood_validation_used"),
                ood_tuning_used=proto.get("ood_tuning_used"),
            )
        )
    return rows


def main() -> None:
    RAWDIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for rel in git_files():
        raw = blob(rel)
        name = Path(rel).name
        (RAWDIR / name).write_bytes(raw)
        rows.append(flatten(name, json.loads(raw)))
    df = pd.DataFrame([r for group in rows for r in group])
    df = df.sort_values(["method", "seed", "perturbation_scale"])
    dst = OUT / "cifar10_openood_recovered_from_git.csv"
    df.to_csv(dst, index=False)
    print(f"{len(df)} rows from {len(rows)} JSON files -> {dst.relative_to(ROOT)}")

    pnc = df[df.method.str.startswith("pnc")]
    print("\n=== CIFAR P&C (submission-era, recovered) ===")
    print(
        pnc[["method", "seed", "perturbation_scale", "accuracy", "near_ood_auroc",
             "near_ood_fpr95", "far_ood_auroc", "far_ood_fpr95"]]
        .to_string(index=False, float_format=lambda v: f"{v:.4f}")
    )

    # closest available comparison to the rebuttal's 3-checkpoint P&C row
    sb25 = pnc[(pnc.method.str.contains("single_block")) & (pnc.perturbation_scale == "25.0")]
    if len(sb25):
        m = sb25[["accuracy", "near_ood_auroc", "near_ood_fpr95",
                  "far_ood_auroc", "far_ood_fpr95"]].mean() * 100
        print(f"\nsingle-block scale=25, n={len(sb25)} checkpoints, mean:")
        print(m.round(2).to_string())
        print("rebuttal quoted P&C: 95.69 acc / 90.99 nearAUROC / 37.39 nearFPR95 "
              "/ 94.83 farAUROC / 19.52 farFPR95")
        print("rebuttal quoted SCOD: 95.74 / 89.69 / 39.40 / 92.56 / 21.41  "
              "-- NO SCOD CIFAR ARTIFACT EXISTS ON THIS MACHINE")


if __name__ == "__main__":
    main()
