#!/usr/bin/env python3
"""
Plot a perturbation-method Pareto frontier for one Gym environment.

Default plot:
    x-axis = ID RMSE (lower is better)
    y-axis = Far AUROC (higher is better)

Supported methods:
    - Subspace
    - SWAG
    - PnC (no correction)
    - PnC (LS correction)
    - optional Deep Ensemble reference

The script searches results/{env}/ for matching JSON files, chooses the best
candidate per seed using `nll_val` (fallback `nll_id`), then aggregates over
seeds with median + IQR and plots one point per method.

Examples
--------
# Hopper, ID RMSE vs Far AUROC
python plot_pnc_correction_frontier.py \
    --env Hopper-v5 \
    --results-root results \
    --out experiments/figures/hopper_rmse_vs_far_auroc.png

# Ant, ID RMSE vs Far NLL
python plot_pnc_correction_frontier.py \
    --env Ant-v5 \
    --y-metric nll_ood_far \
    --y-label "Far NLL (lower = better)" \
    --results-root results \
    --out experiments/figures/ant_rmse_vs_far_nll.png

# Restrict to a known seed subset
python plot_pnc_correction_frontier.py \
    --env Hopper-v5 \
    --seeds 0,10,42,100,200 \
    --results-root results \
    --out experiments/figures/hopper_rmse_vs_far_auroc.png
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


METHOD_SPECS = [
    {
        "label": "Subspace",
        "include": ["subspace"],
        "exclude": ["hybrid", "cifar", "openood"],
    },
    {
        "label": "SWAG",
        "include": ["swag"],
        "exclude": ["hybrid", "cifar", "openood"],
    },
    {
        "label": "PnC (no corr.)",
        "include": ["pjsvd_multi", "none"],
        "exclude": ["hybrid", "cifar", "openood"],
    },
    {
        "label": "PnC (LS corr.)",
        "include": ["pjsvd_multi", "least_squares"],
        "exclude": ["hybrid", "cifar", "openood"],
    },
    {
        "label": "Laplace",
        "include": ["laplace"],
        "exclude": ["hybrid", "cifar", "openood", "vcal"],
    },
    {
        "label": "Deep Ensemble",
        "include": ["standard_ensemble"],
        "exclude": ["hybrid", "cifar", "openood"],
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", required=True, help="Environment name, e.g. Hopper-v5")
    parser.add_argument("--results-root", default="results", help="Root results directory")
    parser.add_argument(
        "--seeds",
        default="0,10,42,100,200",
        help="Comma-separated seed list. Missing seeds are skipped.",
    )
    parser.add_argument(
        "--x-metric",
        default="rmse_id",
        help="Metric for x-axis. Default: rmse_id",
    )
    parser.add_argument(
        "--y-metric",
        default="auroc_ood_far",
        help="Metric for y-axis. Default: auroc_ood_far",
    )
    parser.add_argument(
        "--x-label",
        default="ID RMSE (lower = better)",
        help="Axis label for x-axis",
    )
    parser.add_argument(
        "--y-label",
        default="Far AUROC (higher = better)",
        help="Axis label for y-axis",
    )
    parser.add_argument(
        "--selection-metric",
        default="nll_val",
        help="Metric used to choose the best config per seed",
    )
    parser.add_argument(
        "--selection-fallback",
        default="nll_id",
        help="Fallback metric if selection metric is missing",
    )
    parser.add_argument(
        "--exclude-deep-ensemble",
        action="store_true",
        help="Do not plot Deep Ensemble reference point",
    )
    parser.add_argument(
        "--require-vcal",
        action="store_true",
        help="Only consider files whose stem includes 'vcal'",
    )
    parser.add_argument(
        "--forbid-vcal",
        action="store_true",
        help="Ignore files whose stem includes 'vcal'",
    )
    parser.add_argument("--title", default=None, help="Optional plot title")
    parser.add_argument(
        "--annotate-n",
        action="store_true",
        help="Append seed count to point labels",
    )
    parser.add_argument(
        "--annotate-size",
        action="store_true",
        help="Append chosen perturbation size to PnC labels",
    )
    parser.add_argument(
        "--annotate-l2",
        action="store_true",
        help="Append median Unc-L2 value to point labels",
    )
    parser.add_argument(
        "--l2-metric",
        default="uncorrected_l2_id_h",
        help="L2 metric key used by --annotate-l2. Default: uncorrected_l2_id_h",
    )
    parser.add_argument(
        "--l2-metric-pnc",
        default=None,
        help="Override --l2-metric for PnC methods (e.g. corrected_l2_id_z)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print chosen file/config per seed",
    )
    parser.add_argument("--out", required=True, help="Output PNG path")
    return parser.parse_args()


def percentile(xs: list[float], p: float) -> float:
    xs = sorted(xs)
    if not xs:
        raise ValueError("empty input")
    if len(xs) == 1:
        return xs[0]
    idx = (len(xs) - 1) * p
    lo = int(math.floor(idx))
    hi = int(math.ceil(idx))
    if lo == hi:
        return xs[lo]
    w = idx - lo
    return xs[lo] * (1.0 - w) + xs[hi] * w


def summarize(xs: list[float]) -> dict[str, float]:
    vals = [x for x in xs if x is not None and math.isfinite(x)]
    if not vals:
        raise ValueError("no finite values")
    return {
        "q25": percentile(vals, 0.25),
        "q50": percentile(vals, 0.50),
        "q75": percentile(vals, 0.75),
        "n": float(len(vals)),
    }


def looks_like_metrics_dict(d: dict[str, Any]) -> bool:
    metric_keys = {
        "rmse_id",
        "nll_id",
        "nll_val",
        "auroc_ood_far",
        "nll_ood_far",
        "rmse_ood_far",
        "ece_id",
        "var_id",
    }
    return any(k in d for k in metric_keys)


def iter_metric_dicts(obj: Any, prefix: str = "") -> list[tuple[str, dict[str, Any]]]:
    """
    Return a list of (config_name, metrics_dict) from a JSON object.
    Supports:
      1) top-level metrics dict
      2) top-level dict of config_name -> metrics dict
    """
    out: list[tuple[str, dict[str, Any]]] = []

    if isinstance(obj, dict):
        if looks_like_metrics_dict(obj):
            out.append((prefix or "<root>", obj))
            return out

        for k, v in obj.items():
            if isinstance(v, dict) and looks_like_metrics_dict(v):
                name = f"{prefix}:{k}" if prefix else str(k)
                out.append((name, v))

    return out


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def candidate_files(
    env_dir: Path,
    seed: int,
    include: list[str],
    exclude: list[str],
    require_vcal: bool,
    forbid_vcal: bool,
) -> list[Path]:
    # Glob with `*seed{seed}*.json` erroneously matches `seed100` when we ask
    # for `seed10` (prefix collision). Post-filter with a word-boundary regex
    # requiring ``_seed{seed}`` followed by a non-digit, so seed tokens are
    # matched exactly.
    seed_re = re.compile(rf"_seed{seed}(?![0-9])")
    files = sorted(
        p for p in env_dir.glob(f"*seed{seed}*.json")
        if seed_re.search(p.stem)
    )
    selected = []

    for path in files:
        stem = path.stem.lower()

        if require_vcal and "vcal" not in stem:
            continue
        if forbid_vcal and "vcal" in stem:
            continue

        if any(s not in stem for s in include):
            continue
        if any(s in stem for s in exclude):
            continue

        selected.append(path)

    return selected


def select_best_record(
    files: list[Path],
    selection_metric: str,
    selection_fallback: str,
    restrict_cfg: str | None = None,
) -> tuple[Path, str, dict[str, Any]] | None:
    """Pick the (file, cfg) pair that minimises ``selection_metric``.

    ``restrict_cfg`` pins the config key (e.g. ``"50.0"`` for PnC ps). Useful
    when the caller wants a fixed slice across seeds rather than the per-seed
    best.
    """
    best: tuple[float, Path, str, dict[str, Any]] | None = None

    for path in files:
        try:
            data = load_json(path)
        except (OSError, json.JSONDecodeError):
            continue

        for cfg_name, metrics in iter_metric_dicts(data):
            if not isinstance(metrics, dict):
                continue
            if restrict_cfg is not None and cfg_name != restrict_cfg:
                continue

            key = metrics.get(selection_metric)
            if key is None:
                key = metrics.get(selection_fallback)
            if key is None or not isinstance(key, (int, float)) or not math.isfinite(key):
                continue

            if best is None or key < best[0]:
                best = (float(key), path, cfg_name, metrics)

    if best is None:
        return None

    _, path, cfg_name, metrics = best
    return path, cfg_name, metrics


def collect_method_points(
    env_dir: Path,
    method_spec: dict[str, Any],
    seeds: list[int],
    x_metric: str,
    y_metric: str,
    selection_metric: str,
    selection_fallback: str,
    require_vcal: bool,
    forbid_vcal: bool,
    verbose: bool,
    l2_metric: str = "uncorrected_l2_id_h",
    restrict_cfg: str | None = None,
) -> dict[str, Any] | None:
    xs: list[float] = []
    ys: list[float] = []
    l2s: list[float] = []
    sizes: list[str] = []
    chosen: list[tuple[int, str, str]] = []

    for seed in seeds:
        files = candidate_files(
            env_dir=env_dir,
            seed=seed,
            include=method_spec["include"],
            exclude=method_spec["exclude"],
            require_vcal=require_vcal,
            forbid_vcal=forbid_vcal,
        )
        if not files:
            continue

        best = select_best_record(
            files, selection_metric, selection_fallback,
            restrict_cfg=restrict_cfg,
        )
        if best is None:
            continue

        path, cfg_name, metrics = best
        x_val = metrics.get(x_metric)
        y_val = metrics.get(y_metric)

        if not isinstance(x_val, (int, float)) or not math.isfinite(x_val):
            continue
        if not isinstance(y_val, (int, float)) or not math.isfinite(y_val):
            continue

        xs.append(float(x_val))
        ys.append(float(y_val))
        chosen.append((seed, path.name, cfg_name))

        # Collect L2 metric if present
        l2_val = metrics.get(l2_metric)
        if isinstance(l2_val, (int, float)) and math.isfinite(l2_val):
            l2s.append(float(l2_val))

        # Record the config name (perturbation size key for PnC)
        sizes.append(cfg_name)

        if verbose:
            print(
                f"[{method_spec['label']}] seed={seed}  "
                f"x={x_val:.4f}  y={y_val:.4f}  "
                f"file={path.name}  cfg={cfg_name}"
            )

    if not xs or not ys:
        return None

    xsum = summarize(xs)
    ysum = summarize(ys)

    result: dict[str, Any] = {
        "label": method_spec["label"],
        "x_q25": xsum["q25"],
        "x_q50": xsum["q50"],
        "x_q75": xsum["q75"],
        "y_q25": ysum["q25"],
        "y_q50": ysum["q50"],
        "y_q75": ysum["q75"],
        "n": int(min(xsum["n"], ysum["n"])),
        "chosen": chosen,
    }

    # L2 summary (None if no seeds had L2 data)
    if l2s:
        l2sum = summarize(l2s)
        result["l2_q50"] = l2sum["q50"]
    else:
        result["l2_q50"] = None

    # Most common chosen config name (perturbation size for PnC)
    if sizes:
        from collections import Counter
        result["size_label"] = Counter(sizes).most_common(1)[0][0]
    else:
        result["size_label"] = None

    return result


def main() -> None:
    args = parse_args()

    if args.require_vcal and args.forbid_vcal:
        raise ValueError("Cannot use both --require-vcal and --forbid-vcal")

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    env_dir = Path(args.results_root) / args.env
    if not env_dir.exists():
        raise FileNotFoundError(f"Environment results directory not found: {env_dir}")

    method_specs = []
    for spec in METHOD_SPECS:
        if args.exclude_deep_ensemble and spec["label"] == "Deep Ensemble":
            continue
        method_specs.append(spec)

    points = []
    for spec in method_specs:
        is_pnc = "pjsvd" in " ".join(spec["include"])
        l2_key = args.l2_metric_pnc if (is_pnc and args.l2_metric_pnc) else args.l2_metric
        point = collect_method_points(
            env_dir=env_dir,
            method_spec=spec,
            seeds=seeds,
            x_metric=args.x_metric,
            y_metric=args.y_metric,
            selection_metric=args.selection_metric,
            selection_fallback=args.selection_fallback,
            require_vcal=args.require_vcal,
            forbid_vcal=args.forbid_vcal,
            verbose=args.verbose,
            l2_metric=l2_key,
        )
        if point is not None:
            points.append(point)

    if not points:
        raise RuntimeError("No matching method points were found.")

    plt.figure(figsize=(7, 5.5))

    for point in points:
        x = point["x_q50"]
        y = point["y_q50"]
        xerr = [[x - point["x_q25"]], [point["x_q75"] - x]]
        yerr = [[y - point["y_q25"]], [point["y_q75"] - y]]

        plt.errorbar(
            x,
            y,
            xerr=xerr,
            yerr=yerr,
            fmt="o",
            capsize=4,
        )

        label = point["label"]
        if args.annotate_n:
            label = f"{label} (n={point['n']})"
        if args.annotate_size and point.get("size_label"):
            sl = point["size_label"]
            # Only annotate size for PnC methods (nested configs keyed by perturbation scale)
            if sl != "<root>" and "PnC" in point["label"]:
                label = f"{label}\nsize={sl}"
        if args.annotate_l2 and point.get("l2_q50") is not None:
            label = f"{label}\nL2={point['l2_q50']:.2f}"

        plt.annotate(
            label,
            (x, y),
            xytext=(6, -6),
            textcoords="offset points",
            fontsize=8,
            va="top",
        )

    plt.xlabel(args.x_label)
    plt.ylabel(args.y_label)

    if args.title:
        plt.title(args.title)
    else:
        plt.title(f"{args.env}: perturbation-method frontier")

    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    print(f"Saved plot to {out_path}")


if __name__ == "__main__":
    main()