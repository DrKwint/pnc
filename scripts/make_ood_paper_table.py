#!/usr/bin/env python3
"""Produce the paper-ready OOD comparison table from multi-seed result JSONs.

Differs from `aggregate_ood_results.py` in:
- Methods are grouped (single-model OOD baselines | UQ methods | reference) and ordered.
- Deep Ensemble is marked as a 5×-cost reference row, not a peer.
- Mahalanobis caveat is appended.
- Bolds the best entry per metric within the post-hoc-method group only.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path

# Display order and grouping
GROUPS = [
    ("Post-hoc OOD baselines (frozen single model)", [
        ("PreAct ResNet-18", "predictive_entropy"),
        ("MSP",              "max_softmax_uncertainty"),
        ("Energy",           "energy_score"),
        ("Mahalanobis",      "mahalanobis"),
        ("ReAct+Energy",     "energy_score"),
    ]),
    ("Post-hoc UQ methods (single model)", [
        ("LLLA",             "predictive_entropy"),
        ("Epinet",           "predictive_entropy"),
        ("PnC scale=25.0",   "predictive_entropy"),
        ("PnC scale=50.0",   "predictive_entropy"),
        ("Multi-block PnC scale=7.0", "predictive_entropy"),
    ]),
    ("Train-time UQ methods (single model)", [
        ("MC Dropout",       "predictive_entropy"),
        ("SWAG",             "predictive_entropy"),
    ]),
    ("Reference: 5× training cost", [
        ("Standard Ensemble", "predictive_entropy"),
    ]),
]

PNC_SCALE_DISPLAY_OVERRIDES = {
    "PnC scale=25.0": ("PnC scale=25.0 (scale=25.0)", "PnC scale=25.0"),
    "PnC scale=50.0": ("PnC scale=50.0 (scale=50.0)", "PnC scale=50.0"),
    "Multi-block PnC scale=7.0": ("Multi-block PnC scale=7.0 (scale=7.0)", "Multi-block PnC scale=7.0"),
}

# Methods that use 5x training compute
HIGH_COST_METHODS = {"Standard Ensemble"}


def _strip_seed(stem: str) -> str:
    return re.sub(r"_seed\d+(?:_[A-Za-z0-9.]+)*$", "", stem)


def _seed_from_path(path: Path) -> int | None:
    m = re.search(r"_seed(\d+)", path.stem)
    return int(m.group(1)) if m else None


def _mean_std(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    mu = sum(values) / len(values)
    if len(values) == 1:
        return mu, None
    var = sum((x - mu) ** 2 for x in values) / (len(values) - 1)
    return mu, math.sqrt(var)


def _fmt_pct(mu, sd, dp=1):
    if mu is None:
        return "---"
    if sd is None:
        return f"{mu*100:.{dp}f}"
    return f"{mu*100:.{dp}f} ± {sd*100:.{dp}f}"


def _fmt_raw(mu, sd, dp=4):
    if mu is None:
        return "---"
    if sd is None:
        return f"{mu:.{dp}f}"
    return f"{mu:.{dp}f} ± {sd:.{dp}f}"


def _load_method_seeds(results_dir: Path) -> dict[str, dict[int, dict]]:
    """Map ensemble_name (with optional scale suffix) -> seed -> result-dict."""
    seeds_by_name: dict[str, dict[int, dict]] = defaultdict(dict)
    for path in sorted(results_dir.glob("openood_v1p5_*.json")):
        try:
            with open(path) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue

        seed = _seed_from_path(path)
        if seed is None:
            continue

        if "id_metrics" in data:
            name = data.get("ensemble_name", "?")
            seeds_by_name[name][seed] = data
        else:
            for scale_key, sub in data.items():
                if isinstance(sub, dict) and "id_metrics" in sub:
                    name = sub.get("ensemble_name", "?")
                    seeds_by_name[name][seed] = sub
    return seeds_by_name


def _row_stats(seed_to_data: dict[int, dict], primary: str) -> dict:
    accs, nlls, eces = [], [], []
    naas, nfps, faas, ffps = [], [], [], []
    for seed in sorted(seed_to_data.keys()):
        d = seed_to_data[seed]
        idm = d.get("id_metrics", {})
        accs.append(idm.get("accuracy"))
        nlls.append(idm.get("nll"))
        eces.append(idm.get("ece"))
        near_agg = d.get("near_ood", {}).get("aggregate", {}).get(primary, {})
        far_agg = d.get("far_ood", {}).get("aggregate", {}).get(primary, {})
        naas.append(near_agg.get("mean_auroc"))
        nfps.append(near_agg.get("mean_fpr95"))
        faas.append(far_agg.get("mean_auroc"))
        ffps.append(far_agg.get("mean_fpr95"))
    drop = lambda lst: [v for v in lst if v is not None]
    return {
        "n_seeds": len(seed_to_data),
        "accuracy": _mean_std(drop(accs)),
        "nll": _mean_std(drop(nlls)),
        "ece": _mean_std(drop(eces)),
        "near_auroc": _mean_std(drop(naas)),
        "near_fpr95": _mean_std(drop(nfps)),
        "far_auroc": _mean_std(drop(faas)),
        "far_fpr95": _mean_std(drop(ffps)),
    }


def _resolve_method(target_name: str, all_seeds: dict[str, dict[int, dict]]) -> dict[int, dict]:
    """Find the seeds dict for a given target method name (handles scale suffix variants)."""
    if target_name in all_seeds:
        return all_seeds[target_name]
    if target_name in PNC_SCALE_DISPLAY_OVERRIDES:
        for variant in PNC_SCALE_DISPLAY_OVERRIDES[target_name]:
            if variant in all_seeds:
                return all_seeds[variant]
    # Substring fallback
    for k in all_seeds:
        if target_name == k or k.startswith(target_name):
            return all_seeds[k]
    return {}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir", nargs="?", default="results/cifar10")
    args = parser.parse_args()

    seeds_by_name = _load_method_seeds(Path(args.results_dir))

    # Infer dataset name from the directory (results/cifar10 -> CIFAR-10)
    dataset_token = Path(args.results_dir).name
    if dataset_token.lower() == "cifar10":
        dataset_label = "CIFAR-10"
    elif dataset_token.lower() == "cifar100":
        dataset_label = "CIFAR-100"
    else:
        dataset_label = dataset_token

    print(f"# {dataset_label} OOD Detection — Paper Table\n")
    print(
        "All methods use frozen ID hyperparameters; no OOD data was used for tuning, "
        "model selection, or threshold choice. Mean ± std reported across N seeds."
    )
    print(
        "\n*Note: the Mahalanobis baseline reported here uses a paper-clean variant — "
        "single penultimate layer, per-class means + shared covariance fitted on ID train, "
        "no OOD-trained logistic combiner. The original Lee et al. (2018) numbers use OOD "
        "validation data and are not directly comparable.*\n"
    )

    print("| Method | Train cost | N | Acc% ↑ | NLL ↓ | ECE ↓ | Near AUROC ↑ | Near FPR95 ↓ | Far AUROC ↑ | Far FPR95 ↓ |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")

    for group_name, methods in GROUPS:
        print(f"| **{group_name}** |  |  |  |  |  |  |  |  |  |")
        for target_name, primary in methods:
            seeds = _resolve_method(target_name, seeds_by_name)
            if not seeds:
                print(f"| {target_name} | 1× | 0 | --- | --- | --- | --- | --- | --- | --- |")
                continue
            stats = _row_stats(seeds, primary)
            cost = "5×" if target_name in HIGH_COST_METHODS else "1×"
            print(
                f"| {target_name} | {cost} | {stats['n_seeds']} | "
                f"{_fmt_pct(*stats['accuracy'], dp=2)} | "
                f"{_fmt_raw(*stats['nll'], dp=4)} | "
                f"{_fmt_raw(*stats['ece'], dp=4)} | "
                f"{_fmt_pct(*stats['near_auroc'], dp=2)} | "
                f"{_fmt_pct(*stats['near_fpr95'], dp=2)} | "
                f"{_fmt_pct(*stats['far_auroc'], dp=2)} | "
                f"{_fmt_pct(*stats['far_fpr95'], dp=2)} |"
            )

    print("\n*Train cost is the multiplier of base-model training compute.*")


if __name__ == "__main__":
    main()
