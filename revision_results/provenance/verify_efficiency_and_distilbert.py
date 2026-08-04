#!/usr/bin/env python
"""Verify the efficiency-accounting and DistilBERT numbers quoted in the rebuttal.

Efficiency (MuJoCo)
-------------------
Source: results/neurips_2026_rebuttal/efficiency_v2/efficiency_v2_rows.csv
        (+ the per-env JSONs beside it). Ant-v5 / HalfCheetah-v5 / Hopper-v5,
        seed 0, 5000 training steps, NVIDIA TITAN X (Pascal).
Claim under test: "P&C is ~47-62x cheaper to construct than a matched 50-member
Deep Ensemble."  construction_total_s = base_train_s + build_s.

Efficiency (CIFAR-10)
---------------------
Source: results/cifar10/inference_cost.json (RECOVERED FROM GIT HEAD 70fb480 -
        the working tree deletes every results/cifar10 file). Quoted inference
        values are checked against `warm_per_sample_ms`.

DistilBERT / Banking77 / CLINC-OOS
----------------------------------
Source: results/banking77_distilbert_pnc/metrics/raw.csv (per-seed, 5 construction
        seeds) aggregated to results/banking77_distilbert_pnc/tables/banking77_pnc.md.

Usage: .venv/bin/python revision_results/provenance/verify_efficiency_and_distilbert.py
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
EFF = ROOT / "revision_results/efficiency"
DIS = ROOT / "revision_results/distilbert"


def mujoco_construction() -> pd.DataFrame:
    src = ROOT / "results/neurips_2026_rebuttal/efficiency_v2/efficiency_v2_rows.csv"
    df = pd.read_csv(src)
    piv = df.pivot_table(index="env", columns="method", values="construction_total_s")
    out = pd.DataFrame(
        {
            "pnc_construction_total_s": piv["pnc"],
            "deep_ensemble_M50_construction_total_s": piv["deep_ensemble"],
            "swag_construction_total_s": piv["swag"],
            "laplace_construction_total_s": piv["laplace"],
            "single_base_train_s": piv["single_base"],
        }
    )
    out["de_over_pnc_ratio"] = (
        out.deep_ensemble_M50_construction_total_s / out.pnc_construction_total_s
    )
    base = df[df.method == "pnc"].set_index("env")
    out["pnc_base_train_s"] = base.base_train_s
    out["pnc_posthoc_build_s"] = base.build_s
    out["pnc_minimal_ckpt_mb"] = base.minimal_ckpt_mb
    out["pnc_storage_mb"] = base.storage_mb
    out["de_storage_mb"] = df[df.method == "deep_ensemble"].set_index("env").storage_mb
    out["device"] = base.device
    out["n_trained_nets_de"] = df[df.method == "deep_ensemble"].set_index("env").n_trained_nets
    return out.reset_index()


def cifar_inference() -> pd.DataFrame:
    """results/cifar10/inference_cost.json is deleted in the working tree; read it from git."""
    blob = subprocess.run(
        ["git", "show", "HEAD:results/cifar10/inference_cost.json"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    d = json.loads(blob)
    rows = [
        dict(
            method=k,
            cold_warmup_s=v.get("cold_warmup_s"),
            warm_total_s=v.get("warm_total_s"),
            warm_per_sample_ms=v.get("warm_per_sample_ms"),
            n_forward_passes=v.get("n_forward_passes"),
            train_cost_factor=v.get("train_cost_factor"),
            note=v.get("note", ""),
        )
        for k, v in d["methods"].items()
    ]
    df = pd.DataFrame(rows)
    df["n_bench_samples"] = d.get("n_bench_samples")
    return df


def distilbert() -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = pd.read_csv(ROOT / "results/banking77_distilbert_pnc/metrics/raw.csv")
    test = raw[raw.split != "id_val"] if "split" in raw else raw
    per_seed = raw.copy()
    agg = (
        raw.groupby(["method", "split"])
        .agg(
            n_seeds=("seed", "nunique"),
            id_accuracy_mean=("id_accuracy", "mean"),
            id_accuracy_std=("id_accuracy", "std"),
            id_nll_mean=("id_nll", "mean"),
            id_nll_std=("id_nll", "std"),
            id_ece_mean=("id_ece", "mean"),
            id_ece_std=("id_ece", "std"),
            ood_auroc_mean=("ood_auroc", "mean"),
            ood_auroc_std=("ood_auroc", "std"),
            ood_fpr95_mean=("ood_fpr95", "mean"),
            temperature=("temperature", "mean"),
            selected_scale=("selected_scale", "mean"),
        )
        .reset_index()
    )
    return per_seed, agg


def main() -> None:
    EFF.mkdir(parents=True, exist_ok=True)
    DIS.mkdir(parents=True, exist_ok=True)

    mj = mujoco_construction()
    mj.to_csv(EFF / "mujoco_construction_normalized.csv", index=False)
    print("=== MuJoCo matched-M construction (seed 0, 5000 steps, TITAN X Pascal) ===")
    print(
        mj[
            [
                "env",
                "pnc_base_train_s",
                "pnc_posthoc_build_s",
                "pnc_construction_total_s",
                "deep_ensemble_M50_construction_total_s",
                "de_over_pnc_ratio",
            ]
        ].to_string(index=False, float_format=lambda v: f"{v:.2f}")
    )
    lo, hi = mj.de_over_pnc_ratio.min(), mj.de_over_pnc_ratio.max()
    print(f"observed ratio range: {lo:.1f}x - {hi:.1f}x   (rebuttal quoted 47-62x)")

    ci = cifar_inference()
    ci.to_csv(EFF / "cifar_inference_cost_from_git_HEAD.csv", index=False)
    quoted = {
        "P&C": 7.53,
        "Deep Ensemble": 7.26,
        "SWAG": 7.33,
        "Laplace": 7.68,
    }
    print("\n=== CIFAR inference (results/cifar10/inference_cost.json @ git HEAD) ===")
    print(ci[["method", "warm_per_sample_ms", "n_forward_passes"]].to_string(index=False))
    print("rebuttal quoted:", quoted)

    per_seed, agg = distilbert()
    per_seed.to_csv(DIS / "distilbert_per_seed_raw.csv", index=False)
    agg.to_csv(DIS / "distilbert_aggregated_by_method_split.csv", index=False)

    checks = []
    tab = pd.read_csv(ROOT / "results/banking77_distilbert_pnc/tables/banking77_pnc.csv")
    key = [c for c in tab.columns if c.lower() in ("method", "unnamed: 0")][0]
    tab = tab.set_index(key)
    quoted_dist = {
        ("Internal FFN P&C", "ID Acc"): 0.925,
        ("Internal FFN P&C", "ID NLL"): 0.301,
        ("Internal FFN P&C", "ID ECE"): 0.031,
        ("Internal FFN P&C", "Near AUROC"): 0.907,
        ("Internal FFN P&C", "Cross AUROC"): 0.967,
        ("Internal FFN P&C", "Far AUROC"): 0.981,
        ("Uncorrected perturb", "ID Acc"): 0.921,
        ("Uncorrected perturb", "ID NLL"): 1.079,
        ("Uncorrected perturb", "ID ECE"): 0.525,
        ("MC Dropout", "Near AUROC"): 0.917,
        ("MC Dropout", "Cross AUROC"): 0.969,
        ("MC Dropout", "Far AUROC"): 0.982,
        ("Energy", "Near AUROC"): 0.929,
        ("Energy", "Cross AUROC"): 0.980,
        ("Energy", "Far AUROC"): 0.990,
    }
    for (m, col), q in quoted_dist.items():
        got = tab.loc[m, col] if (m in tab.index and col in tab.columns) else float("nan")
        checks.append(dict(method=m, metric=col, table_value=got, rebuttal_quoted=q,
                           matches=abs(float(got) - q) < 5e-4))
    cdf = pd.DataFrame(checks)
    cdf.to_csv(DIS / "distilbert_quoted_value_checks.csv", index=False)
    print("\n=== DistilBERT quoted-value checks (vs tables/banking77_pnc.csv) ===")
    print(cdf.to_string(index=False))
    print("all match:", bool(cdf.matches.all()))


if __name__ == "__main__":
    main()
