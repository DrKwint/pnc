#!/usr/bin/env python3
"""Aggregate multi-seed OpenOOD result JSONs into mean ± std tables.

For each canonical method, groups all matching seed files, computes mean and std
across seeds for every metric, and emits a markdown table. Handles both flat
result files (one method per file) and nested PnC files (per-perturbation-scale).
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path


def _strip_seed(stem: str) -> str:
    """Remove `_seed<N>` (and optional trailing tags) so different seeds collapse."""
    return re.sub(r"_seed\d+(?:_[A-Za-z0-9.]+)*$", "", stem)


def _load_records(results_dir: Path) -> dict[str, list[tuple[Path, dict]]]:
    """Group result JSONs by canonical (seed-stripped) stem."""
    groups: dict[str, list[tuple[Path, dict]]] = defaultdict(list)
    for path in sorted(results_dir.glob("openood_v1p5_*.json")):
        try:
            with open(path) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        canonical = _strip_seed(path.stem)
        groups[canonical].append((path, data))
    return groups


def _flatten_records(records: list[tuple[Path, dict]]) -> list[tuple[Path, dict, str | None]]:
    """Yield (path, metric_dict, scale_label_or_none) for each scoring entry.

    PnC tasks save nested JSONs keyed by perturbation scale; this function unrolls
    them so each (scale) appears as a separate row.
    """
    out = []
    for path, data in records:
        if "id_metrics" in data:
            out.append((path, data, None))
        else:
            for scale_key, sub in data.items():
                if isinstance(sub, dict) and "id_metrics" in sub:
                    out.append((path, sub, scale_key))
    return out


def _seed_from_path(path: Path) -> int | None:
    m = re.search(r"_seed(\d+)", path.stem)
    return int(m.group(1)) if m else None


def _mean_std(values: list[float]) -> tuple[float, float | None]:
    if not values:
        return float("nan"), None
    mu = sum(values) / len(values)
    if len(values) == 1:
        return mu, None
    var = sum((x - mu) ** 2 for x in values) / (len(values) - 1)
    return mu, math.sqrt(var)


def _fmt(mu: float, sd: float | None, scale: float = 1.0, dp: int = 1) -> str:
    if math.isnan(mu):
        return "---"
    if sd is None:
        return f"{mu*scale:.{dp}f}"
    return f"{mu*scale:.{dp}f} ± {sd*scale:.{dp}f}"


def _collect_metrics(
    rows: list[tuple[Path, dict, str | None]],
) -> dict[str, dict[int, dict]]:
    """Group by canonical (display name + scale) → seed → metrics dict."""
    by_method: dict[str, dict[int, dict]] = defaultdict(dict)
    for path, sub, scale_label in rows:
        seed = _seed_from_path(path)
        if seed is None:
            continue
        name = sub.get("ensemble_name", "?")
        if scale_label is not None:
            name = f"{name} (scale={scale_label})"
        by_method[name][seed] = sub
    return by_method


def _extract_row(seed_to_data: dict[int, dict]) -> dict[str, list[float]]:
    """For one method, pull every metric across all seeds into parallel lists."""
    accs, nlls, eces = [], [], []
    near_aurocs, near_fpr95s = [], []
    far_aurocs, far_fpr95s = [], []
    primary_scores = set()
    for seed in sorted(seed_to_data.keys()):
        d = seed_to_data[seed]
        idm = d.get("id_metrics", {})
        accs.append(idm.get("accuracy"))
        nlls.append(idm.get("nll"))
        eces.append(idm.get("ece"))
        primary = d.get("protocol", {}).get("primary_score", "?")
        primary_scores.add(primary)
        near_agg = d.get("near_ood", {}).get("aggregate", {}).get(primary, {})
        far_agg = d.get("far_ood", {}).get("aggregate", {}).get(primary, {})
        near_aurocs.append(near_agg.get("mean_auroc"))
        near_fpr95s.append(near_agg.get("mean_fpr95"))
        far_aurocs.append(far_agg.get("mean_auroc"))
        far_fpr95s.append(far_agg.get("mean_fpr95"))
    return {
        "accuracy": [v for v in accs if v is not None],
        "nll": [v for v in nlls if v is not None],
        "ece": [v for v in eces if v is not None],
        "near_auroc": [v for v in near_aurocs if v is not None],
        "near_fpr95": [v for v in near_fpr95s if v is not None],
        "far_auroc": [v for v in far_aurocs if v is not None],
        "far_fpr95": [v for v in far_fpr95s if v is not None],
        "n_seeds": len(seed_to_data),
        "seeds": sorted(seed_to_data.keys()),
        "primary_score": ", ".join(sorted(primary_scores)),
    }


def main():
    parser = argparse.ArgumentParser(description="Aggregate multi-seed OOD results.")
    parser.add_argument("results_dir", nargs="?", default="results/cifar10")
    parser.add_argument("--markdown", action="store_true", help="Emit markdown table")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    grouped = _load_records(results_dir)
    rows = []
    for canonical, records in grouped.items():
        rows.extend(_flatten_records(records))

    by_method = _collect_metrics(rows)
    aggregated = {name: _extract_row(seed_data) for name, seed_data in by_method.items()}

    if args.markdown:
        print("| Method | Primary Score | N | Seeds | Acc% | NLL | ECE | Near AUROC | Near FPR95 | Far AUROC | Far FPR95 |")
        print("|---|---|---|---|---|---|---|---|---|---|---|")
        for name in sorted(aggregated.keys()):
            r = aggregated[name]
            n = r["n_seeds"]
            seeds = ",".join(str(s) for s in r["seeds"])
            acc_mu, acc_sd = _mean_std(r["accuracy"])
            nll_mu, nll_sd = _mean_std(r["nll"])
            ece_mu, ece_sd = _mean_std(r["ece"])
            na_mu, na_sd = _mean_std(r["near_auroc"])
            nf_mu, nf_sd = _mean_std(r["near_fpr95"])
            fa_mu, fa_sd = _mean_std(r["far_auroc"])
            ff_mu, ff_sd = _mean_std(r["far_fpr95"])
            print(
                f"| {name} | {r['primary_score']} | {n} | {seeds} | "
                f"{_fmt(acc_mu, acc_sd, 100, 2)} | "
                f"{_fmt(nll_mu, nll_sd, 1, 4)} | "
                f"{_fmt(ece_mu, ece_sd, 1, 4)} | "
                f"{_fmt(na_mu, na_sd, 100, 2)} | "
                f"{_fmt(nf_mu, nf_sd, 100, 2)} | "
                f"{_fmt(fa_mu, fa_sd, 100, 2)} | "
                f"{_fmt(ff_mu, ff_sd, 100, 2)} |"
            )
        return

    # Plain text table
    header = (
        f"{'Method':<35s} | {'N':>2s} | "
        f"{'Acc%':>13s} {'NLL':>16s} {'ECE':>16s} | "
        f"{'Near AUROC':>15s} {'Near FPR95':>15s} | "
        f"{'Far AUROC':>15s} {'Far FPR95':>15s}"
    )
    sep = "-" * len(header)
    print(sep)
    print(header)
    print(sep)
    for name in sorted(aggregated.keys()):
        r = aggregated[name]
        n = r["n_seeds"]
        acc_mu, acc_sd = _mean_std(r["accuracy"])
        nll_mu, nll_sd = _mean_std(r["nll"])
        ece_mu, ece_sd = _mean_std(r["ece"])
        na_mu, na_sd = _mean_std(r["near_auroc"])
        nf_mu, nf_sd = _mean_std(r["near_fpr95"])
        fa_mu, fa_sd = _mean_std(r["far_auroc"])
        ff_mu, ff_sd = _mean_std(r["far_fpr95"])
        print(
            f"{name:<35s} | {n:>2d} | "
            f"{_fmt(acc_mu, acc_sd, 100, 2):>13s} "
            f"{_fmt(nll_mu, nll_sd, 1, 4):>16s} "
            f"{_fmt(ece_mu, ece_sd, 1, 4):>16s} | "
            f"{_fmt(na_mu, na_sd, 100, 2):>15s} "
            f"{_fmt(nf_mu, nf_sd, 100, 2):>15s} | "
            f"{_fmt(fa_mu, fa_sd, 100, 2):>15s} "
            f"{_fmt(ff_mu, ff_sd, 100, 2):>15s}"
        )
    print(sep)
    print(f"\nTotal methods: {len(aggregated)}")


if __name__ == "__main__":
    main()
