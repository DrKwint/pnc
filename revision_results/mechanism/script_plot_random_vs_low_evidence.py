#!/usr/bin/env python3
"""Generate figures supporting the 5 hypotheses for Random-vs-Low PnC.

Panels:
  (a) OOD-far AUROC gap (aggregate over existing 5-seed runs)
  (b) OOD-far predictive-variance ratio (var_ood_far / var_id)
  (c) Δh_mean ID vs OOD at perturbed layer 1 (shows Low's shrink-OOD effect)
  (d) Effective output rank (participation ratio of SVs of ΔY) on OOD
  (e) Median pairwise cosine similarity in output-delta space (OOD)
  (f) Intervention: Low vs Low(σ=1) vs Random on pred-std-ratio and eff_rank_ood
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _agg_load(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def _collect_diag(log_dir: Path, env: str, pert_size: float):
    prefix = f"random_vs_low_diag_{env}_seed"
    suffix = f"_ps{pert_size}.json"
    return sorted(log_dir.glob(f"{prefix}*{suffix}"))


def _collect_intervention(log_dir: Path, env: str, pert_size: float):
    prefix = f"random_vs_low_diag_{env}_seed"
    suffix = f"_ps{pert_size}_low_flat_sigma+random.json"
    return sorted(log_dir.glob(f"{prefix}*{suffix}"))


def _mean_std_of_field(paths: List[Path], family: str, field: str):
    vals = []
    for p in paths:
        j = json.loads(p.read_text())
        if family in j and field in j[family]:
            vals.append(j[family][field])
    if not vals:
        return np.nan, np.nan
    arr = np.array(vals)
    return float(arr.mean()), float(arr.std())


def _layer_field(paths: List[Path], family: str, layer_idx: int, tag: str, field: str):
    vals = []
    for p in paths:
        j = json.loads(p.read_text())
        for e in j.get(family, {}).get("per_layer", []):
            if e["layer_idx"] == layer_idx and e["tag"] == tag and field in e:
                vals.append(e[field])
    if not vals:
        return np.nan, np.nan
    arr = np.array(vals)
    return float(arr.mean()), float(arr.std())


def _bar_pair(ax, labels, low_mu, low_sd, rand_mu, rand_sd, title, ylabel,
              low_color="#6baed6", rand_color="#de2d26"):
    x = np.arange(len(labels))
    w = 0.35
    ax.bar(x - w/2, low_mu, w, yerr=low_sd, capsize=4, label="Low",
           color=low_color, edgecolor="black", alpha=0.9)
    ax.bar(x + w/2, rand_mu, w, yerr=rand_sd, capsize=4, label="Random",
           color=rand_color, edgecolor="black", alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--log-dir", type=Path,
                   default=Path("experiments/logs/random_vs_low_diag"))
    p.add_argument("--agg-json", type=Path,
                   default=Path("experiments/logs/random_vs_low_aggregate.json"))
    p.add_argument("--out", type=Path,
                   default=Path("experiments/figures/random_vs_low_evidence.png"))
    p.add_argument("--envs", nargs="+",
                   default=["Ant-v5", "Hopper-v5"])
    p.add_argument("--pert-sizes", type=float, nargs="+",
                   default=[50.0, 5.0])
    args = p.parse_args()

    env_labels = [e.replace("-v5", "") for e in args.envs]

    # (a) AUROC gap from aggregate JSON
    agg = _agg_load(args.agg_json)
    auroc_low, auroc_low_sd, auroc_rand, auroc_rand_sd = [], [], [], []
    nll_low, nll_low_sd, nll_rand, nll_rand_sd = [], [], [], []
    varr_low, varr_low_sd, varr_rand, varr_rand_sd = [], [], [], []
    if agg is not None:
        for env in args.envs:
            for row in agg:
                if row.get("env") != env:
                    continue
                if row.get("metric") == "auroc_ood_far":
                    auroc_low.append(row["low_mean"]); auroc_low_sd.append(row["low_std"])
                    auroc_rand.append(row["rand_mean"]); auroc_rand_sd.append(row["rand_std"])
                if row.get("metric") == "nll_ood_far":
                    nll_low.append(row["low_mean"]); nll_low_sd.append(row["low_std"])
                    nll_rand.append(row["rand_mean"]); nll_rand_sd.append(row["rand_std"])
                if row.get("metric") == "var_ratio":
                    varr_low.append(row["low_mean"]); varr_low_sd.append(row["low_std"])
                    varr_rand.append(row["rand_mean"]); varr_rand_sd.append(row["rand_std"])

    # (c) Δh ID vs OOD at layer 1 (diagnostic runs)
    dh_id_low, dh_id_rand, dh_ood_low, dh_ood_rand = [], [], [], []
    dh_id_low_sd, dh_id_rand_sd, dh_ood_low_sd, dh_ood_rand_sd = [], [], [], []
    # (d) eff_rank_ood
    eff_low, eff_rand = [], []
    eff_low_sd, eff_rand_sd = [], []
    # (e) median_cos_ood
    cos_low, cos_rand = [], []
    cos_low_sd, cos_rand_sd = [], []
    # (f) intervention
    flat_eff, flat_eff_sd, flat_cos, flat_cos_sd, flat_ratio, flat_ratio_sd = [], [], [], [], [], []
    low_ratio, low_ratio_sd, rand_ratio, rand_ratio_sd = [], [], [], []
    low_eff_for_int, low_eff_for_int_sd, rand_eff_for_int, rand_eff_for_int_sd = [], [], [], []

    for env, ps in zip(args.envs, args.pert_sizes):
        paths = _collect_diag(args.log_dir, env, ps)
        # layer 1 (index 0) in gym code
        li_mu, li_sd = _layer_field(paths, "low", 0, "id", "dh_mean")
        dh_id_low.append(li_mu); dh_id_low_sd.append(li_sd)
        li_mu, li_sd = _layer_field(paths, "low", 0, "ood", "dh_mean")
        dh_ood_low.append(li_mu); dh_ood_low_sd.append(li_sd)
        ri_mu, ri_sd = _layer_field(paths, "random", 0, "id", "dh_mean")
        dh_id_rand.append(ri_mu); dh_id_rand_sd.append(ri_sd)
        ri_mu, ri_sd = _layer_field(paths, "random", 0, "ood", "dh_mean")
        dh_ood_rand.append(ri_mu); dh_ood_rand_sd.append(ri_sd)

        lmu, lsd = _mean_std_of_field(paths, "low", "eff_rank_ood")
        eff_low.append(lmu); eff_low_sd.append(lsd)
        rmu, rsd = _mean_std_of_field(paths, "random", "eff_rank_ood")
        eff_rand.append(rmu); eff_rand_sd.append(rsd)

        lmu, lsd = _mean_std_of_field(paths, "low", "median_cos_ood")
        cos_low.append(lmu); cos_low_sd.append(lsd)
        rmu, rsd = _mean_std_of_field(paths, "random", "median_cos_ood")
        cos_rand.append(rmu); cos_rand_sd.append(rsd)

        int_paths = _collect_intervention(args.log_dir, env, ps)
        fmu, fsd = _mean_std_of_field(int_paths, "low_flat_sigma", "eff_rank_ood")
        flat_eff.append(fmu); flat_eff_sd.append(fsd)
        fmu, fsd = _mean_std_of_field(int_paths, "low_flat_sigma", "median_cos_ood")
        flat_cos.append(fmu); flat_cos_sd.append(fsd)
        fmu, fsd = _mean_std_of_field(int_paths, "low_flat_sigma", "pred_std_ratio")
        flat_ratio.append(fmu); flat_ratio_sd.append(fsd)

        # For intervention panel, only use seeds that appear in intervention runs
        # (to show a fair 3-way comparison)
        int_seeds = [int(p.name.split("_seed")[1].split("_")[0]) for p in int_paths]
        base_paths = [p for p in paths
                      if int(p.name.split("_seed")[1].split("_ps")[0]) in int_seeds]
        lr, lrs = _mean_std_of_field(base_paths, "low", "pred_std_ratio")
        low_ratio.append(lr); low_ratio_sd.append(lrs)
        lr, lrs = _mean_std_of_field(base_paths, "random", "pred_std_ratio")
        rand_ratio.append(lr); rand_ratio_sd.append(lrs)
        lr, lrs = _mean_std_of_field(base_paths, "low", "eff_rank_ood")
        low_eff_for_int.append(lr); low_eff_for_int_sd.append(lrs)
        lr, lrs = _mean_std_of_field(base_paths, "random", "eff_rank_ood")
        rand_eff_for_int.append(lr); rand_eff_for_int_sd.append(lrs)

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))

    # (a) AUROC
    _bar_pair(axes[0, 0], env_labels, auroc_low, auroc_low_sd, auroc_rand, auroc_rand_sd,
              "(a) H1: OOD-far AUROC", "AUROC (↑)")

    # (b) var_ratio
    _bar_pair(axes[0, 1], env_labels, varr_low, varr_low_sd, varr_rand, varr_rand_sd,
              "(b) H1: Predictive var_ratio (OOD/ID)", "var_ratio")
    axes[0, 1].axhline(1.0, color="gray", linestyle="--", alpha=0.5)

    # (c) Δh layer 1: ID vs OOD
    ax = axes[0, 2]
    xlabels = [f"{e}\nID" for e in env_labels] + [f"{e}\nOOD" for e in env_labels]
    low_vals = dh_id_low + dh_ood_low
    low_sd = dh_id_low_sd + dh_ood_low_sd
    rand_vals = dh_id_rand + dh_ood_rand
    rand_sd = dh_id_rand_sd + dh_ood_rand_sd
    _bar_pair(ax, xlabels, low_vals, low_sd, rand_vals, rand_sd,
              "(c) H2: Δh at layer l1 (log-scale)", "||Δh||₂ mean")
    ax.set_yscale("log")

    # (d) eff_rank_ood
    _bar_pair(axes[1, 0], env_labels, eff_low, eff_low_sd, eff_rand, eff_rand_sd,
              "(d) H3: Effective output rank (OOD)", "Participation ratio of σ(ΔY)")

    # (e) median_cos_ood
    _bar_pair(axes[1, 1], env_labels, cos_low, cos_low_sd, cos_rand, cos_rand_sd,
              "(e) H3: Median pairwise cosine sim (OOD)", "cos(ΔYᵢ, ΔYⱼ)")

    # (f) Intervention — show Low vs Low(σ=1) vs Random on eff_rank_ood
    ax = axes[1, 2]
    x = np.arange(len(env_labels))
    w = 0.25
    ax.bar(x - w, low_eff_for_int, w, yerr=low_eff_for_int_sd, capsize=4, label="Low",
           color="#6baed6", edgecolor="black")
    ax.bar(x, flat_eff, w, yerr=flat_eff_sd, capsize=4, label="Low (σ=1)",
           color="#9ecae1", edgecolor="black", hatch="//")
    ax.bar(x + w, rand_eff_for_int, w, yerr=rand_eff_for_int_sd, capsize=4, label="Random",
           color="#de2d26", edgecolor="black")
    ax.set_xticks(x)
    ax.set_xticklabels(env_labels)
    ax.set_ylabel("eff_rank_ood")
    ax.set_title("(f) H5: σ-rescaling intervention")
    ax.legend(fontsize=8)

    fig.suptitle("Random-Proj vs Low-Proj PnC — evidence summary (5 seeds/env)", fontsize=14)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=130, bbox_inches="tight")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
