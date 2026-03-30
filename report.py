#!/usr/bin/env python3
"""
report.py — Compile Luigi experiment results into a compact table report.

Results from multiple seeds are aggregated: each cell shows mean ± stddev.
When only one seed is present, stddev is omitted.

Usage:
    python report.py                          # auto-discover results/
    python report.py --results_dir results    # explicit path
    python report.py --env HalfCheetah-v5    # filter to one env
    python report.py --fmt md                 # markdown output (default: text)
    python report.py --out report.md --fmt md # save to file
"""

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path


# ---------------------------------------------------------------------------
# Metrics configuration
# ---------------------------------------------------------------------------

GYM_METRICS = [
    ("rmse_id", "RMSE-ID"),
    ("nll_id", "NLL-ID"),
    ("ece_id", "ECE-ID"),
    ("rmse_ood", "RMSE-OOD"),
    ("nll_ood", "NLL-OOD"),
    ("ece_ood", "ECE-OOD"),
    ("auroc", "AUROC"),
    ("aupr", "AUPR"),
    ("uncorrected_l2_id", "Uncorr-L2-ID"),
    ("uncorrected_l2_ood", "Uncorr-L2-OOD"),
    ("corrected_l2_id", "Corr-L2-ID"),
    ("corrected_l2_ood", "Corr-L2-OOD"),
    ("train_time", "Train (s)"),
    ("eval_time", "Eval (s)"),
]

CLF_METRICS = [
    ("accuracy", "Accuracy"),
    ("brier", "Brier"),
    ("entropy", "Entropy"),
    ("ece", "ECE"),
    ("train_time", "Train (s)"),
    ("eval_time", "Eval (s)"),
]

UCI_METRICS = [
    ("rmse", "RMSE"),
    ("nll", "NLL"),
    ("ece", "ECE"),
    ("var", "Var"),
    ("train_time", "Train (s)"),
    ("eval_time", "Eval (s)"),
]


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------


def _mean_std(values: list[float]) -> tuple[float, float | None]:
    if not values:
        return float("nan"), None
    if len(values) == 1:
        return values[0], None
    n = len(values)
    mu = sum(values) / n
    var = sum((x - mu) ** 2 for x in values) / (n - 1)
    return mu, math.sqrt(var)


def _fmt_cell(mean: float, std: float | None, width: int = 16) -> str:
    """Format a mean ± std (or just mean) right-justified into `width` chars."""
    if math.isnan(mean):
        return "-".center(width)
    if std is None:
        return f"{mean:.4f}".rjust(width)
    return f"{mean:.4f}±{std:.4f}".rjust(width)


def _fmt_cell_md(mean: float, std: float | None) -> str:
    if math.isnan(mean):
        return "-"
    if std is None:
        return f"{mean:.4f}"
    return f"{mean:.4f}±{std:.4f}"


# ---------------------------------------------------------------------------
# File-name helpers
# ---------------------------------------------------------------------------


def _canonical(stem: str) -> str:
    """Strip the _seed<N> suffix to get a canonical (seed-invariant) key."""
    return re.sub(r"_seed\d+", "", stem).replace("__", "_")


def _is_nested(data: dict) -> bool:
    return bool(data) and isinstance(next(iter(data.values())), dict)


def _extract(stem: str, key: str) -> str:
    if key == "act":
        m = re.search(r"act-([a-z]+)", stem)
        return m.group(1) if m else "?"
    m = re.search(rf"(?<![a-z]){key}(\d[\d.]*)", stem)
    return m.group(1) if m else "?"


def _friendly_name(canonical: str) -> str | None:
    """Human-readable method name from a canonical filename stem."""
    act = _extract(canonical, "act")
    act_suffix = f" [{act}]" if act != "?" else ""

    backend = ""
    if "_activation_covariance" in canonical:
        backend = "-Cov"
    elif "_projected_residual" in canonical:
        backend = "-Proj"

    if canonical.startswith("standard_ensemble"):
        return f"Deep Ensemble (n={_extract(canonical, 'n')}){act_suffix}"
    if canonical.startswith("mc_dropout"):
        return f"MC Dropout (n={_extract(canonical, 'n')}){act_suffix}"
    if canonical.startswith("swag"):
        return f"SWAG (n={_extract(canonical, 'n')}){act_suffix}"
    if canonical.startswith("laplace"):
        return f"Laplace (n={_extract(canonical, 'n')}){act_suffix}"
    if canonical.startswith("pjsvd"):
        scope = (
            "First"
            if "_first_" in canonical
            else "Multi"
            if "_multi_" in canonical
            else ""
        )
        mode = (
            "Affine"
            if "_affine" in canonical
            else "LS"
            if "_least_squares" in canonical
            else ""
        )
        full = " (Full)" if "_full" in canonical else ""
        k = _extract(canonical, "k")
        n = _extract(canonical, "n")
        family = ""
        if "_low" in canonical:
            family = "Low"
        elif "_random" in canonical:
            family = "Random"
        elif "_all" in canonical:
            family = "All"
        
        family_str = f" ({family}{backend})" if (family or backend) else ""
        if not scope and not mode:
            return f"PJSVD{family_str} (k={k}, n={n}){act_suffix}"
        return f"PJSVD-{scope}-{mode}{full}{family_str} (k={k}, n={n}){act_suffix}"
    if canonical.startswith("ml_pjsvd"):
        return f"ML-PJSVD{backend} (k={_extract(canonical, 'k')}, n={_extract(canonical, 'n')}){act_suffix}"
    if canonical.startswith("ensemble_pjsvd"):
        return (
            f"Ensemble+PJSVD{backend} (m={_extract(canonical, 'm')}, "
            f"k={_extract(canonical, 'k')}, n={_extract(canonical, 'n')}){act_suffix}"
        )
    if canonical.startswith("subspace_inference"):
        n = _extract(canonical, "n")
        t = _extract(canonical, "T")
        t_str = f" (T={t})" if t != "?" else ""
        return f"Subspace (n={n}){t_str}{act_suffix}"
    if canonical.startswith("base_model"):
        return None  # checkpoint — skip
    return canonical + act_suffix


def _config_label(canonical: str, config_val: str) -> str:
    act = _extract(canonical, "act")
    act_suffix = f" [{act}]" if act != "?" else ""
    if canonical.startswith("laplace"):
        return f"  ↳ prior={config_val}{act_suffix}"
    if "pjsvd" in canonical:
        return f"  ↳ size={config_val}{act_suffix}"
    return f"  ↳ [{config_val}]{act_suffix}"


# ---------------------------------------------------------------------------
# Aggregation: load all JSON files, group by canonical name + config key
# ---------------------------------------------------------------------------


def _load_env_results(env_dir: Path) -> dict:
    """
    Returns:
        {
          canonical_stem: {
            "_flat": { metric_key: [val_seed0, val_seed1, ...] }   (non-nested)
            OR
            config_val: { metric_key: [val_seed0, val_seed1, ...] } (nested)
          }
        }
    """
    groups: dict[str, dict] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )

    for path in sorted(env_dir.glob("*.json")):
        stem = path.stem
        can = _canonical(stem)
        if _friendly_name(can) is None:
            continue  # skip checkpoints etc.
        with open(path) as f:
            data = json.load(f)

        if _is_nested(data):
            for config_val, metrics in data.items():
                if "uncorrected_l2_id_h" in metrics:
                    metrics["uncorrected_l2_id"] = metrics["uncorrected_l2_id_h"]
                    metrics["uncorrected_l2_ood"] = metrics["uncorrected_l2_ood_h"]
                if "corrected_l2_id_z" in metrics:
                    metrics["corrected_l2_id"] = metrics["corrected_l2_id_z"]
                    metrics["corrected_l2_ood"] = metrics["corrected_l2_ood_z"]
                for k, v in metrics.items():
                    groups[can][config_val][k].append(v)
        else:
            if "uncorrected_l2_id_h" in data:
                data["uncorrected_l2_id"] = data["uncorrected_l2_id_h"]
                data["uncorrected_l2_ood"] = data["uncorrected_l2_ood_h"]
            if "corrected_l2_id_z" in data:
                data["corrected_l2_id"] = data["corrected_l2_id_z"]
                data["corrected_l2_ood"] = data["corrected_l2_ood_z"]
            for k, v in data.items():
                groups[can]["_flat"][k].append(v)

    return dict(groups)


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------

W = 17  # column width for text layout


def _render_header(metrics_cfg, fmt):
    if fmt == "md":
        return (
            "| Method | "
            + " | ".join(h for _, h in metrics_cfg)
            + " |\n"
            + "|"
            + "|".join(["---"] * (len(metrics_cfg) + 1))
            + "|"
        )
    else:
        return f"  {'Method':<44}" + "".join(f"{h:>{W}}" for _, h in metrics_cfg)


def _render_row(label, metrics_cfg, agg_dict, fmt):
    """Render one row, computing mean ± std from the list of seed values."""
    cells = []
    for k, _ in metrics_cfg:
        vals = agg_dict.get(k, [])
        mu, sd = _mean_std(vals) if vals else (float("nan"), None)
        cells.append((mu, sd))

    if fmt == "md":
        vals_str = " | ".join(_fmt_cell_md(mu, sd) for mu, sd in cells)
        return f"| {label} | {vals_str} |"
    else:
        vals_str = "".join(_fmt_cell(mu, sd, W) for mu, sd in cells)
        return f"  {label:<44}{vals_str}"


def _sep(metrics_cfg):
    return "-" * (46 + W * len(metrics_cfg))


# ---------------------------------------------------------------------------
# Section builders
# ---------------------------------------------------------------------------


def _build_section(title: str, env_dir: Path, metrics_cfg: list, fmt: str) -> str:
    groups = _load_env_results(env_dir)
    if not groups:
        return f"  (no results in {env_dir})\n"

    lines = []
    sep = _sep(metrics_cfg)

    if fmt == "md":
        lines.append(f"\n## {title}\n")
        lines.append(_render_header(metrics_cfg, fmt))
    else:
        lines.append(f"\n{'=' * len(sep)}")
        lines.append(f"  {title}")
        lines.append(sep)
        lines.append(_render_header(metrics_cfg, fmt))
        lines.append(sep)

    # Sort: flat-only methods first, then nested (multi-config)
    flat_items = [(c, d) for c, d in groups.items() if "_flat" in d]
    nested_items = [(c, d) for c, d in groups.items() if "_flat" not in d]

    for can, group in flat_items:
        label = _friendly_name(can) or can
        lines.append(_render_row(label, metrics_cfg, group["_flat"], fmt))

    if nested_items and fmt != "md":
        lines.append("")

    for can, group in nested_items:
        base = _friendly_name(can) or can
        if fmt != "md":
            lines.append(f"  ── {base} ──")
        for config_val, agg in group.items():
            n_seeds = max(len(v) for v in agg.values()) if agg else 0
            suffix = (
                f" (n={n_seeds} seed{'s' if n_seeds != 1 else ''})"
                if n_seeds > 1
                else ""
            )
            label = _config_label(can, config_val).strip() + suffix
            lines.append(_render_row(label, metrics_cfg, agg, fmt))
        if fmt != "md":
            lines.append("")

    if fmt != "md":
        lines.append(sep)

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Print a compact report of experiment results."
    )
    parser.add_argument(
        "--results_dir",
        default="results",
        help="Root results directory (default: results/)",
    )
    parser.add_argument(
        "--env", default=None, help="Filter to a single environment name"
    )
    parser.add_argument(
        "--fmt",
        choices=["text", "md"],
        default="text",
        help="Output format: 'text' (default) or 'md' (markdown)",
    )
    parser.add_argument(
        "--out", default=None, help="Write report to this file instead of stdout"
    )
    args = parser.parse_args()

    root = Path(args.results_dir)
    if not root.exists():
        print(f"Results directory '{root}' not found.")
        return

    parts = []

    if args.fmt == "md":
        parts.append("# Experiment Results Report\n")
        parts.append(
            "*Cells show mean ± stddev across seeds (when >1 seed available).*\n"
        )
    else:
        parts.append("\n" + "#" * 72)
        parts.append("  EXPERIMENT RESULTS REPORT")
        parts.append("  Cells: mean ± stddev across seeds  (single seed → value only)")
        parts.append("#" * 72)

    found = False

    envs = []
    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        if d.name.lower() == "mnist":
            continue
        if d.name == "uci":
            for sub_d in sorted(d.iterdir()):
                if sub_d.is_dir():
                    envs.append((f"uci-{sub_d.name}", sub_d, UCI_METRICS))
        else:
            envs.append((d.name, d, GYM_METRICS))

    for env_name, env_dir, metrics_cfg in envs:
        if args.env and env_name != args.env:
            continue
        parts.append(_build_section(env_name, env_dir, metrics_cfg, args.fmt))
        found = True

    mnist_dir = root / "mnist"
    if mnist_dir.exists() and (args.env is None or args.env.upper() == "MNIST"):
        section = _build_section(
            "MNIST Classification", mnist_dir, CLF_METRICS, args.fmt
        )
        if section.strip():
            parts.append(section)
            found = True

    if not found:
        parts.append("  No results found.")

    report = "\n".join(parts)

    if args.out:
        Path(args.out).write_text(report)
        print(f"Report written to {args.out}")
    else:
        print(report)


if __name__ == "__main__":
    main()
