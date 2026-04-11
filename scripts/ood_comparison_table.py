#!/usr/bin/env python3
"""Generate a compact OOD comparison table from OpenOOD result JSONs."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def _load_results(results_dir: Path) -> list[dict]:
    """Load all openood_v1p5_*.json files, return list of (method, data) dicts."""
    rows = []
    for p in sorted(results_dir.glob("openood_v1p5_*.json")):
        with open(p) as f:
            data = json.load(f)
        # For PnC tasks that nest by perturbation size, unwrap
        if "id_metrics" not in data:
            for scale_key, sub in data.items():
                if isinstance(sub, dict) and "id_metrics" in sub:
                    rows.append({"path": str(p), "scale": scale_key, **sub})
        else:
            rows.append({"path": str(p), **data})
    return rows


def _fmt(val, fmt=".1f"):
    if val is None or (isinstance(val, float) and (val != val)):  # NaN check
        return "---"
    return f"{val:{fmt}}"


def _is_higher_better(metric: str) -> bool:
    return "accuracy" in metric or "auroc" in metric


def _maybe_bold(text: str, bold: bool) -> str:
    return f"*{text}*" if bold and text != "---" else text


def _best_metric_indices(
    rows: list[dict], metrics: list[str], bold_pct: float, ignore_names: set[str] | None = None
) -> dict[str, set[int]]:
    ignore_names = ignore_names or set()
    aliases = {
        "Deep Ensemble": {"Deep Ensemble", "Standard Ensemble"},
        "Standard Ensemble": {"Deep Ensemble", "Standard Ensemble"},
    }

    def display_name(row: dict) -> str:
        name = row.get("ensemble_name", "?")
        scale = row.get("scale")
        return f"{name} (s={scale})" if scale is not None else name

    def is_ignored(name: str) -> bool:
        for target in ignore_names:
            for candidate in aliases.get(target, {target}):
                if name == candidate or name.startswith(f"{candidate} ("):
                    return True
        return False

    scored_by_metric: dict[str, list[tuple[int, float]]] = {metric: [] for metric in metrics}
    for idx, row in enumerate(rows):
        if is_ignored(display_name(row)):
            continue

        id_m = row.get("id_metrics", {})
        primary = row.get("protocol", {}).get("primary_score", "?")
        near_agg = row.get("near_ood", {}).get("aggregate", {}).get(primary, {})
        far_agg = row.get("far_ood", {}).get("aggregate", {}).get(primary, {})

        values = {
            "accuracy": id_m.get("accuracy"),
            "nll": id_m.get("nll"),
            "ece": id_m.get("ece"),
            "near_auroc": near_agg.get("mean_auroc"),
            "near_fpr95": near_agg.get("mean_fpr95"),
            "far_auroc": far_agg.get("mean_auroc"),
            "far_fpr95": far_agg.get("mean_fpr95"),
        }

        for metric, value in values.items():
            if metric in scored_by_metric and value is not None and not math.isnan(value):
                scored_by_metric[metric].append((idx, value))

    winners: dict[str, set[int]] = {}
    for metric in metrics:
        scored = scored_by_metric[metric]
        if not scored:
            winners[metric] = set()
            continue
        target = (
            max(value for _, value in scored)
            if _is_higher_better(metric)
            else min(value for _, value in scored)
        )
        margin = abs(target) * (bold_pct / 100.0)
        if _is_higher_better(metric):
            threshold = target - margin
            winners[metric] = {
                idx
                for idx, value in scored
                if value >= threshold or math.isclose(value, target, rel_tol=1e-12, abs_tol=1e-12)
            }
        else:
            threshold = target + margin
            winners[metric] = {
                idx
                for idx, value in scored
                if value <= threshold or math.isclose(value, target, rel_tol=1e-12, abs_tol=1e-12)
            }
    return winners


def main():
    parser = argparse.ArgumentParser(description="Generate a compact OOD comparison table.")
    parser.add_argument(
        "results_dir",
        nargs="?",
        default="results/cifar10",
        help="Directory containing openood_v1p5_*.json files.",
    )
    parser.add_argument(
        "--bold",
        type=float,
        default=None,
        metavar="K",
        help="Bold values within K percent of the best value in each metric.",
    )
    parser.add_argument(
        "--bold-ignore-deep-ensemble",
        action="store_true",
        help="Exclude the 'Deep Ensemble' row when determining which values get bolded.",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    rows = _load_results(results_dir)
    if not rows:
        print(f"No openood_v1p5_*.json files found in {results_dir}")
        return

    bold_metrics = ["accuracy", "nll", "ece", "near_auroc", "near_fpr95", "far_auroc", "far_fpr95"]
    winners = (
        _best_metric_indices(
            rows,
            bold_metrics,
            args.bold,
            {"Deep Ensemble"} if args.bold_ignore_deep_ensemble else set(),
        )
        if args.bold is not None
        else {metric: set() for metric in bold_metrics}
    )

    # Header
    header = (
        f"{'Method':<22s} | {'Primary Score':<22s} | "
        f"{'Acc%':>5s} {'NLL':>6s} {'ECE':>6s} | "
        f"{'Near AUROC':>10s} {'Near FPR95':>10s} | "
        f"{'Far AUROC':>9s} {'Far FPR95':>9s}"
    )
    sep = "-" * len(header)

    print(sep)
    print(header)
    print(sep)

    for idx, row in enumerate(rows):
        name = row.get("ensemble_name", "?")
        scale = row.get("scale")
        if scale is not None:
            name = f"{name} (s={scale})"

        primary = row.get("protocol", {}).get("primary_score", "?")
        id_m = row.get("id_metrics", {})
        acc = id_m.get("accuracy", None)
        nll = id_m.get("nll", None)
        ece = id_m.get("ece", None)

        near_agg = row.get("near_ood", {}).get("aggregate", {}).get(primary, {})
        far_agg = row.get("far_ood", {}).get("aggregate", {}).get(primary, {})

        near_auroc = near_agg.get("mean_auroc")
        near_fpr95 = near_agg.get("mean_fpr95")
        far_auroc = far_agg.get("mean_auroc")
        far_fpr95 = far_agg.get("mean_fpr95")

        acc_text = _maybe_bold(_fmt(acc and acc * 100), idx in winners["accuracy"])
        nll_text = _maybe_bold(_fmt(nll, ".3f"), idx in winners["nll"])
        ece_text = _maybe_bold(_fmt(ece, ".3f"), idx in winners["ece"])
        near_auroc_text = _maybe_bold(_fmt(near_auroc and near_auroc * 100), idx in winners["near_auroc"])
        near_fpr95_text = _maybe_bold(_fmt(near_fpr95 and near_fpr95 * 100), idx in winners["near_fpr95"])
        far_auroc_text = _maybe_bold(_fmt(far_auroc and far_auroc * 100), idx in winners["far_auroc"])
        far_fpr95_text = _maybe_bold(_fmt(far_fpr95 and far_fpr95 * 100), idx in winners["far_fpr95"])

        print(
            f"{name:<22s} | {primary:<22s} | "
            f"{acc_text:>5s} {nll_text:>6s} {ece_text:>6s} | "
            f"{near_auroc_text:>10s} {near_fpr95_text:>10s} | "
            f"{far_auroc_text:>9s} {far_fpr95_text:>9s}"
        )

    print(sep)

    # Protocol compliance check
    print("\nProtocol compliance:")
    all_clean = True
    for row in rows:
        proto = row.get("protocol", {})
        name = row.get("ensemble_name", row.get("path", "?"))
        issues = []
        if proto.get("ood_validation_used") is not False:
            issues.append("ood_validation_used != false")
        if proto.get("ood_tuning_used") is not False:
            issues.append("ood_tuning_used != false")
        if issues:
            print(f"  WARNING {name}: {', '.join(issues)}")
            all_clean = False
    if all_clean:
        print("  All methods: ood_validation_used=false, ood_tuning_used=false")


if __name__ == "__main__":
    main()
