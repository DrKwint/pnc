#!/usr/bin/env python3
"""
Build tables directly from Gym result JSONs.

For nested methods, the script picks:
- the Laplace prior with the best ID NLL
- the PJSVD Low size with the best ID NLL
- the PJSVD Random size with the best ID NLL

Usage:
    python json_to_tex_table.py
    python json_to_tex_table.py --results-dir results --env Ant-v5
    python json_to_tex_table.py --fmt text
    python json_to_tex_table.py --out gym_tables.tex
    python json_to_tex_table.py --out gym_tables.txt
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path


METRICS = [
    ("rmse_id", r"ID RMSE $\downarrow$"),
    ("nll_ood_near", r"Near NLL $\downarrow$"),
    ("nll_ood_mid", r"Mid NLL $\downarrow$"),
    ("nll_ood_far", r"Far NLL $\downarrow$"),
    ("auroc_ood_far", r"Far AUROC $\uparrow$"),
]

METHOD_ORDER = [
    "MC Dropout",
    "Deep Ensemble",
    "Subspace",
    "SWAG",
    "Laplace",
    "PJSVD Low",
    "PJSVD Random",
]


def mean_std(values: list[float]) -> tuple[float, float | None]:
    if not values:
        return float("nan"), None
    if len(values) == 1:
        return values[0], None
    mu = sum(values) / len(values)
    var = sum((x - mu) ** 2 for x in values) / (len(values) - 1)
    return mu, math.sqrt(var)


def fmt_cell(values: list[float], include_std: bool) -> str:
    mu, sd = mean_std(values)
    if math.isnan(mu):
        return "--"
    if sd is None or not include_std:
        return f"{mu:.4f}"
    return rf"{mu:.4f} $\pm$ {sd:.4f}"


def fmt_cell_text(values: list[float], include_std: bool) -> str:
    mu, sd = mean_std(values)
    if math.isnan(mu):
        return "--"
    if sd is None or not include_std:
        return f"{mu:.4f}"
    return f"{mu:.4f} +/- {sd:.4f}"


def is_higher_better(metric: str) -> bool:
    return "auroc" in metric or "aupr" in metric


def best_metric_indices(
    rows: list[tuple[str, dict[str, list[float]]]], bold_pct: float
) -> dict[str, set[int]]:
    winners: dict[str, set[int]] = {}
    for metric, _ in METRICS:
        scored = []
        for idx, (_, metrics) in enumerate(rows):
            mu, _ = mean_std(metrics.get(metric, []))
            if not math.isnan(mu):
                scored.append((idx, mu))
        if not scored:
            winners[metric] = set()
            continue
        target = (
            max(value for _, value in scored)
            if is_higher_better(metric)
            else min(value for _, value in scored)
        )
        margin = abs(target) * (bold_pct / 100.0)
        if is_higher_better(metric):
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


def maybe_bold_tex(text: str, bold: bool) -> str:
    return rf"\textbf{{{text}}}" if bold and text != "--" else text


def maybe_bold_text(text: str, bold: bool) -> str:
    return f"*{text}*" if bold and text != "--" else text


def numeric_key(value: str) -> tuple[int, float | str]:
    try:
        return (0, float(value))
    except ValueError:
        return (1, value)


def canonical_stem(path: Path) -> str:
    return re.sub(r"_seed\d+(?:_T[\d.]+)?$", "", path.stem)


def method_name(stem: str) -> str | None:
    if stem.endswith(".npz"):
        return None
    if stem.startswith("mc_dropout"):
        return "MC Dropout"
    if stem.startswith("standard_ensemble"):
        return "Deep Ensemble"
    if stem.startswith("subspace_inference"):
        return "Subspace"
    if stem.startswith("swag"):
        return "SWAG"
    if stem.startswith("laplace"):
        return "Laplace"
    if stem.startswith("pjsvd_"):
        if "_random_" in stem:
            return "PJSVD Random"
        if "_low_" in stem:
            return "PJSVD Low"
    return None


def load_env_results(env_dir: Path) -> dict[str, dict]:
    groups: dict[str, dict] = defaultdict(
        lambda: {
            "flat": defaultdict(list),
            "configs": defaultdict(lambda: defaultdict(list)),
        }
    )

    for path in sorted(env_dir.glob("*.json")):
        if path.name.endswith(".npz.json"):
            continue

        stem = canonical_stem(path)
        method = method_name(stem)
        if method is None:
            continue

        with open(path) as f:
            data = json.load(f)

        first = next(iter(data.values())) if isinstance(data, dict) and data else None
        if isinstance(first, dict):
            for config, metrics in data.items():
                for metric, value in metrics.items():
                    if isinstance(value, (int, float)):
                        groups[method]["configs"][str(config)][metric].append(float(value))
        else:
            for metric, value in data.items():
                if isinstance(value, (int, float)):
                    groups[method]["flat"][metric].append(float(value))

    return groups


def choose_best_config(configs: dict[str, dict[str, list[float]]]) -> str:
    ranked = []
    for config, metrics in configs.items():
        mu, _ = mean_std(metrics.get("nll_id", []))
        ranked.append((mu, numeric_key(config), config))
    ranked.sort()
    return ranked[0][2]


def method_rows(env_groups: dict[str, dict]) -> list[tuple[str, dict[str, list[float]]]]:
    rows = []
    for method in METHOD_ORDER:
        group = env_groups.get(method)
        if not group:
            continue

        if group["configs"]:
            config = choose_best_config(group["configs"])
            label = (
                f"Laplace (prior={config})"
                if method == "Laplace"
                else f"{method} (size={config})"
            )
            rows.append((label, group["configs"][config]))
        else:
            rows.append((method, group["flat"]))
    return rows


def render_env_table(
    env: str,
    rows: list[tuple[str, dict[str, list[float]]]],
    include_std: bool,
    bold_pct: float,
) -> str:
    winners = best_metric_indices(rows, bold_pct)
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{{env}. Laplace prior and PJSVD sizes selected by best ID NLL.}}",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        "Method & " + " & ".join(header for _, header in METRICS) + r" \\",
        r"\midrule",
    ]

    for row_idx, (label, metrics) in enumerate(rows):
        cells = [
            maybe_bold_tex(
                fmt_cell(metrics.get(metric, []), include_std),
                row_idx in winners.get(metric, set()),
            )
            for metric, _ in METRICS
        ]
        lines.append(f"{label} & " + " & ".join(cells) + r" \\")

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ]
    )
    return "\n".join(lines)


def render_env_table_text(
    env: str,
    rows: list[tuple[str, dict[str, list[float]]]],
    include_std: bool,
    bold_pct: float,
) -> str:
    winners = best_metric_indices(rows, bold_pct)
    headers = ["Method"] + [header.replace(r" $\downarrow$", " down").replace(r" $\uparrow$", " up") for _, header in METRICS]
    body = []
    for row_idx, (label, metrics) in enumerate(rows):
        body.append(
            [label]
            + [
                maybe_bold_text(
                    fmt_cell_text(metrics.get(metric, []), include_std),
                    row_idx in winners.get(metric, set()),
                )
                for metric, _ in METRICS
            ]
        )

    widths = [len(h) for h in headers]
    for row in body:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def fmt_row(row: list[str]) -> str:
        return "  ".join(
            cell.ljust(widths[i]) if i == 0 else cell.rjust(widths[i])
            for i, cell in enumerate(row)
        )

    sep = "-" * len(fmt_row(["-" * w for w in widths]))
    lines = [env, sep, fmt_row(headers), sep]
    for row in body:
        lines.append(fmt_row(row))
    lines.append(sep)
    return "\n".join(lines)


def build_tables(
    results_dir: Path,
    env_filter: str | None,
    fmt: str,
    include_std: bool,
    bold_pct: float,
) -> str:
    env_dirs = sorted(p for p in results_dir.iterdir() if p.is_dir())
    if env_filter is not None:
        env_dirs = [p for p in env_dirs if p.name == env_filter]

    tables = []
    for env_dir in env_dirs:
        groups = load_env_results(env_dir)
        rows = method_rows(groups)
        if rows:
            if fmt == "tex":
                tables.append(render_env_table(env_dir.name, rows, include_std, bold_pct))
            else:
                tables.append(render_env_table_text(env_dir.name, rows, include_std, bold_pct))

    return "\n\n".join(tables) + ("\n" if tables else "")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build tables from Gym JSON results.")
    parser.add_argument("--results-dir", default="results", help="Results directory to read.")
    parser.add_argument("--env", default=None, help="Optional env name filter.")
    parser.add_argument(
        "--fmt",
        choices=["tex", "text"],
        default=None,
        help="Output format. Defaults to tex unless --out ends in .txt.",
    )
    parser.add_argument(
        "--include-std",
        action="store_true",
        help="Include standard deviations in table cells. Off by default.",
    )
    parser.add_argument(
        "--bold",
        type=float,
        default=0.0,
        metavar="K",
        help="Bold values within K percent of the best value in each column. Default: 0.",
    )
    parser.add_argument("--out", default=None, help="Optional output path.")
    args = parser.parse_args()

    fmt = args.fmt
    if fmt is None:
        fmt = "text" if args.out and args.out.endswith(".txt") else "tex"

    tex = build_tables(Path(args.results_dir), args.env, fmt, args.include_std, args.bold)
    if args.out:
        Path(args.out).write_text(tex)
    else:
        print(tex, end="")


if __name__ == "__main__":
    main()
