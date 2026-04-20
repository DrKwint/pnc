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
    python json_to_tex_table.py --fmt text --max-over vcal
    python json_to_tex_table.py --fmt text --max-over vcal,prob,scope,family,backend,k,n,act,T,full,grid,mode,ens,lam,prel2
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path


PROFILES = {
    "gym": {
        "metrics": [
            ("rmse_id", r"ID RMSE $\downarrow$"),
            ("nll_ood_near", r"Near NLL $\downarrow$"),
            ("nll_ood_mid", r"Mid NLL $\downarrow$"),
            ("nll_ood_far", r"Far NLL $\downarrow$"),
            ("auroc_ood_far", r"Far AUROC $\uparrow$"),
        ],
        "method_order": [
            "MC Dropout",
            "Deep Ensemble",
            "Subspace",
            "SWAG",
            "Laplace",
            "PJSVD Low",
            "PJSVD Random",
        ],
        "selection_metric": "nll_val",
        "selection_metric_fallback": "nll_id",
        "caption": "{env}. Laplace prior and PJSVD sizes selected by best validation NLL (fallback: ID NLL for runs without nll_val).",
    },
    "cifar": {
        "metrics": [
            ("accuracy", r"Accuracy $\uparrow$"),
            ("nll", r"NLL $\downarrow$"),
            ("ece", r"ECE $\downarrow$"),
            ("brier", r"Brier $\downarrow$"),
            ("posthoc_temperature", r"Temp $\downarrow$"),
        ],
        "method_order": [
            "Single Model",
            "MC Dropout",
            "Deep Ensemble",
            "SWAG",
            "Laplace",
            "Epinet",
            "PnC Single Block",
            "PnC Multi Block",
        ],
        "selection_metric": "nll",
        "caption": "{env}. Sweep settings selected by best NLL.",
    },
}

GYM_MAX_OVER_CHOICES = {
    "vcal",
    "prob",
    "scope",
    "family",
    "backend",
    "k",
    "n",
    "act",
    "T",
    "full",
    "grid",
    "mode",
    "ens",
    "lam",
    "prel2",
    "dr",
    "lreg",
    "esn",
    "bf",
}


REQUIRE_KEYS = {
    # binary markers: value in {yes, no}
    "vcal",
    "prob",
    "full",
    "prel2",
    "esn",
    # choice dims: value is the choice name, or "none" for absent
    "scope",    # first | multi | none
    "family",   # low | random | all | none
    "backend",  # cov | proj | activation_covariance | projected_residual | none
    "mode",     # least_squares | affine | none
    # value dims: exact value match
    "k",
    "n",
    "T",
    "lam",
    "act",
    "dr",
    "lreg",
    "bf",  # bootstrap_frac; 'none' matches stems without a _bf token
}


def _parse_yes_no(value: str) -> bool | None:
    v = value.lower()
    if v in ("yes", "y", "on", "1", "true"):
        return True
    if v in ("no", "n", "off", "0", "false"):
        return False
    return None


# Keys whose dimension only shows up in certain method families. For stems
# from other families (where the dimension has no meaning), a --require clause
# on that key is auto-satisfied so it doesn't inadvertently drop unrelated
# rows. Example: --require lreg=0.0001 filters PJSVD rows but keeps MC Dropout,
# Deep Ensemble, etc. Keys not listed here are treated as universally applicable
# (e.g. vcal, act).
_REQUIRE_KEY_SCOPE: dict[str, tuple[str, ...]] = {
    "lreg": ("pjsvd",),
    "full": ("pjsvd",),
    "mode": ("pjsvd",),
    "scope": ("pjsvd",),
    "family": ("pjsvd", "hybrid_pnc_de"),
    "backend": ("pjsvd", "hybrid_pnc_de"),
    "k": ("pjsvd", "hybrid_pnc_de"),
    "prob": ("pjsvd", "hybrid_pnc_de"),
    "T": ("subspace",),
    "lam": ("evidential",),
    "esn": ("evidential",),
    "dr": ("mc_dropout",),
    "ens": ("hybrid_pnc_de",),
    "prel2": ("laplace", "swag", "subspace"),
    "bf": ("pjsvd", "hybrid_pnc_de"),
}


def _clause_applies_to_stem(stem: str, key: str) -> bool:
    """Whether a --require clause on `key` should be evaluated for this stem."""
    scope = _REQUIRE_KEY_SCOPE.get(key)
    if scope is None:
        return True
    return any(stem.startswith(prefix) for prefix in scope)


def _stem_satisfies_clause(stem: str, key: str, value: str) -> bool:
    """Check one key=value require clause against a canonical stem."""
    if key in ("vcal", "prob", "full", "esn"):
        has = f"_{key}" in stem
        want = _parse_yes_no(value)
        return want is not None and has == want
    if key == "prel2":
        has = ".pre_l2" in stem
        want = _parse_yes_no(value)
        return want is not None and has == want
    if key == "scope":
        if value == "first":
            return "_first_" in stem
        if value == "multi":
            return "_multi_" in stem
        if value == "none":
            return "_first_" not in stem and "_multi_" not in stem
        return False
    if key == "family":
        if value == "none":
            return not any(m in stem for m in ("_low", "_random", "_all"))
        if value in ("low", "random", "all"):
            return f"_{value}" in stem
        return False
    if key == "backend":
        aliases = {
            "cov": "_activation_covariance",
            "activation_covariance": "_activation_covariance",
            "proj": "_projected_residual",
            "projected_residual": "_projected_residual",
        }
        if value == "none":
            return not any(m in stem for m in set(aliases.values()))
        if value in aliases:
            return aliases[value] in stem
        return False
    if key == "mode":
        if value == "none":
            return not any(m in stem for m in ("_least_squares", "_affine"))
        if value in ("least_squares", "affine"):
            return f"_{value}" in stem
        return False
    if key == "k":
        return bool(re.search(rf"_k{re.escape(value)}(?![\d.])", stem))
    if key == "n":
        return bool(re.search(rf"_n{re.escape(value)}(?![\d.a-zA-Z])", stem))
    if key == "T":
        return bool(re.search(rf"_T{re.escape(value)}(?![\d.])", stem))
    if key == "lam":
        return bool(re.search(rf"_lam{re.escape(value)}(?![\d.eE+-])", stem))
    if key == "act":
        return bool(re.search(rf"_act-{re.escape(value)}(?![a-z])", stem))
    if key == "dr":
        return bool(re.search(rf"_dr{re.escape(value)}(?![\d.])", stem))
    if key == "lreg":
        return bool(re.search(rf"_lreg{re.escape(value)}(?![\d.eE+-])", stem))
    if key == "bf":
        if value == "none":
            return "_bf" not in stem
        return bool(re.search(rf"_bf{re.escape(value)}(?![\d.eE+-])", stem))
    return False


def stem_satisfies_require(stem: str, require: dict[str, str]) -> bool:
    """Return True iff canonical stem satisfies every applicable require clause.

    Clauses whose key is scoped to a specific method family (see
    ``_REQUIRE_KEY_SCOPE``) are skipped for stems from unrelated families. That
    keeps e.g. ``--require lreg=0.0001`` from dropping MC Dropout rows, which
    never carry an ``_lreg`` token.
    """
    for key, value in require.items():
        if not _clause_applies_to_stem(stem, key):
            continue
        if not _stem_satisfies_clause(stem, key, value):
            return False
    return True


def mean_std(values: list[float]) -> tuple[float, float | None]:
    if not values:
        return float("nan"), None
    if len(values) == 1:
        return values[0], None
    mu = sum(values) / len(values)
    var = sum((x - mu) ** 2 for x in values) / (len(values) - 1)
    return mu, math.sqrt(var)


def median_iqr(values: list[float]) -> tuple[float, float | None]:
    """Median and inter-quartile range. With small n, IQR is more robust to
    outliers than the standard deviation. Falls back to (single value, None)
    for n=1 and (nan, None) for n=0."""
    if not values:
        return float("nan"), None
    if len(values) == 1:
        return values[0], None
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    # Use linear interpolation for percentiles (numpy default)
    def pct(p):
        idx = (n - 1) * p
        lo = int(math.floor(idx))
        hi = int(math.ceil(idx))
        if lo == hi:
            return sorted_vals[lo]
        return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (idx - lo)
    return pct(0.5), pct(0.75) - pct(0.25)


def fmt_cell(values: list[float], include_std: bool, mode: str = "mean_std") -> str:
    if mode == "median_iqr":
        m, sp = median_iqr(values)
        if math.isnan(m):
            return "--"
        if sp is None or not include_std:
            return f"{m:.3f}"
        return rf"{m:.3f} (IQR {sp:.3f})"
    mu, sd = mean_std(values)
    if math.isnan(mu):
        return "--"
    if sd is None or not include_std:
        return f"{mu:.3f}"
    return rf"{mu:.3f} $\pm$ {sd:.3f}"


def fmt_cell_text(values: list[float], include_std: bool, mode: str = "mean_std") -> str:
    if mode == "median_iqr":
        m, sp = median_iqr(values)
        if math.isnan(m):
            return "--"
        if sp is None or not include_std:
            return f"{m:.3f}"
        return f"{m:.3f} (IQR {sp:.3f})"
    mu, sd = mean_std(values)
    if math.isnan(mu):
        return "--"
    if sd is None or not include_std:
        return f"{mu:.3f}"
    return f"{mu:.3f} +/- {sd:.3f}"


def is_higher_better(metric: str) -> bool:
    return "auroc" in metric or "aupr" in metric


def beat_baseline_indices(
    rows: list[tuple[str, dict[str, list[float]]]],
    metrics,
    baseline_substr: str,
    ignore_for_bolding: set[str] | None = None,
) -> dict[str, set[int]]:
    """Per metric, return indices of rows whose mean ties or beats the baseline.

    The baseline is the first row whose label contains `baseline_substr` (case
    insensitive). If no row matches, no indices are bolded.
    """
    ignore_for_bolding = ignore_for_bolding or set()

    def is_ignored(label: str) -> bool:
        return any(label == name or label.startswith(f"{name} (") for name in ignore_for_bolding)

    needle = baseline_substr.lower()
    baseline_idx: int | None = None
    for idx, (label, _) in enumerate(rows):
        if needle in label.lower():
            baseline_idx = idx
            break

    winners: dict[str, set[int]] = {metric: set() for metric, _ in metrics}
    if baseline_idx is None:
        return winners

    _, baseline_metrics = rows[baseline_idx]
    for metric, _ in metrics:
        baseline_mu, _ = mean_std(baseline_metrics.get(metric, []))
        if math.isnan(baseline_mu):
            continue
        higher = is_higher_better(metric)
        picked: set[int] = set()
        for idx, (label, row_metrics) in enumerate(rows):
            if is_ignored(label):
                continue
            mu, _ = mean_std(row_metrics.get(metric, []))
            if math.isnan(mu):
                continue
            tie = math.isclose(mu, baseline_mu, rel_tol=1e-12, abs_tol=1e-12)
            if tie or (mu >= baseline_mu if higher else mu <= baseline_mu):
                picked.add(idx)
        winners[metric] = picked
    return winners


def best_metric_indices(
    rows: list[tuple[str, dict[str, list[float]]]],
    metrics,
    bold_pct: float,
    ignore_for_bolding: set[str] | None = None,
) -> dict[str, set[int]]:
    ignore_for_bolding = ignore_for_bolding or set()

    def is_ignored(label: str) -> bool:
        return any(label == name or label.startswith(f"{name} (") for name in ignore_for_bolding)

    winners: dict[str, set[int]] = {}
    for metric, _ in metrics:
        scored = []
        for idx, (label, metrics) in enumerate(rows):
            if is_ignored(label):
                continue
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


def disqualify_outlier_rows(
    rows: list[tuple[str, dict[str, list[float]]]],
    metrics,
    ratio: float,
) -> tuple[list[tuple[str, dict[str, list[float]]]], list[tuple[str, str, float, float]]]:
    """Drop rows whose mean on any lower-is-better metric exceeds ``ratio`` ×
    the median of that column across all rows.

    Higher-is-better metrics (AUROC/AUPR) are bounded in [0, 1] so the
    multiplicative "10×" check is not meaningful and is skipped. Columns
    whose median is ≤ 0 are also skipped (happens for NLL where most rows
    have negative values — "10× a negative" is not well-defined).

    Returns the surviving rows and a list of (label, metric, value, median)
    tuples describing each disqualification, useful for a caption footnote.
    """
    if ratio <= 0:
        return rows, []
    if len(rows) < 3:
        return rows, []
    drop: set[int] = set()
    dropped_info: list[tuple[str, str, float, float]] = []
    for metric_key, _ in metrics:
        if is_higher_better(metric_key):
            continue
        values = []
        for _, row_metrics in rows:
            mu, _ = mean_std(row_metrics.get(metric_key, []))
            if not math.isnan(mu):
                values.append(mu)
        if len(values) < 3:
            continue
        sorted_vals = sorted(values)
        mid = len(sorted_vals) // 2
        median = (
            sorted_vals[mid]
            if len(sorted_vals) % 2 == 1
            else 0.5 * (sorted_vals[mid - 1] + sorted_vals[mid])
        )
        if median <= 0:
            continue
        threshold = ratio * median
        for idx, (label, row_metrics) in enumerate(rows):
            if idx in drop:
                continue
            mu, _ = mean_std(row_metrics.get(metric_key, []))
            if not math.isnan(mu) and mu > threshold:
                drop.add(idx)
                dropped_info.append((label, metric_key, float(mu), float(median)))
    survivors = [row for idx, row in enumerate(rows) if idx not in drop]
    return survivors, dropped_info


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
    return re.sub(r"_seed\d+", "", path.stem).replace("__", "_")


def extract_stem_value(stem: str, key: str) -> str:
    if key == "act":
        match = re.search(r"act-([a-z]+)", stem)
        return match.group(1) if match else "?"
    match = re.search(rf"(?<![a-z]){key}(\d[\d.]*)", stem)
    return match.group(1) if match else "?"


def gym_friendly_name(stem: str) -> str | None:
    # Skip internal ablation/scale-sweep artefacts: these files hold nested
    # dicts keyed by non-standard sweep values (e.g. SWAG scale multipliers)
    # and do not belong in the main paper table.
    if "scale_sweep" in stem:
        return None
    act = extract_stem_value(stem, "act")
    act_suffix = f" [{act}]" if act != "?" else ""
    calib_suffix = " + VCal" if "_vcal" in stem else ""
    prob_suffix = " + Prob" if "_prob" in stem else ""

    backend = ""
    if "_activation_covariance" in stem:
        backend = "-Cov"
    elif "_projected_residual" in stem:
        backend = "-Proj"

    if stem.startswith("standard_ensemble"):
        return f"Deep Ensemble{calib_suffix} (n={extract_stem_value(stem, 'n')}){act_suffix}"
    if stem.startswith("mc_dropout"):
        dr_match = re.search(r"_dr([\d.]+)", stem)
        dr_str = f", dr={dr_match.group(1)}" if dr_match else ""
        return (
            f"MC Dropout{calib_suffix} (n={extract_stem_value(stem, 'n')}{dr_str})"
            f"{act_suffix}"
        )
    if stem.startswith("swag"):
        return f"SWAG{calib_suffix} (n={extract_stem_value(stem, 'n')}){act_suffix}"
    if stem.startswith("laplace"):
        return f"Laplace{calib_suffix} (n={extract_stem_value(stem, 'n')}){act_suffix}"
    if stem.startswith("evidential"):
        lam_match = re.search(r"lam([\d.e+-]+)", stem)
        lam_str = f" (lam={lam_match.group(1)})" if lam_match else ""
        esn_suffix = " + ESN" if "_esn" in stem else ""
        return f"Evidential{calib_suffix}{esn_suffix}{lam_str}{act_suffix}"
    if stem.startswith("hybrid_pnc_de"):
        nde_match = re.search(r"nDE(\d+)", stem)
        npnc_match = re.search(r"nPnC(\d+)", stem)
        nde = nde_match.group(1) if nde_match else "?"
        npnc = npnc_match.group(1) if npnc_match else "?"
        backend = ""
        if "_activation_covariance" in stem:
            backend = "-Cov"
        elif "_projected_residual" in stem:
            backend = "-Proj"
        bf_match = re.search(r"_bf([\d.eE+-]+)", stem)
        bf_suffix = f" + Boot(f={bf_match.group(1)})" if bf_match else ""
        return (
            f"Hybrid PnC+DE{calib_suffix}{prob_suffix}{bf_suffix}"
            f" (Random{backend}) (M={nde}, K={npnc}){act_suffix}"
        )
    if stem.startswith("pjsvd"):
        scope = (
            "First"
            if "_first_" in stem
            else "Multi"
            if "_multi_" in stem
            else ""
        )
        mode = (
            "Affine"
            if "_affine" in stem
            else "LS"
            if "_least_squares" in stem
            else ""
        )
        full = " (Full)" if "_full" in stem else ""
        k = extract_stem_value(stem, "k")
        n = extract_stem_value(stem, "n")
        family = ""
        if "_low" in stem:
            family = "Low"
        elif "_random" in stem:
            family = "Random"
        elif "_all" in stem:
            family = "All"
        family_str = f" ({family}{backend})" if (family or backend) else ""
        lreg_match = re.search(r"_lreg([\d.eE+-]+)", stem)
        lreg_str = f", λ={lreg_match.group(1)}" if lreg_match else ""
        bf_match = re.search(r"_bf([\d.eE+-]+)", stem)
        bf_suffix = f" + Boot(f={bf_match.group(1)})" if bf_match else ""
        if not scope and not mode:
            return (
                f"PJSVD{calib_suffix}{prob_suffix}{bf_suffix}{family_str} "
                f"(k={k}, n={n}{lreg_str}){act_suffix}"
            )
        return (
            f"PJSVD-{scope}-{mode}{full}{calib_suffix}{prob_suffix}{bf_suffix}"
            f"{family_str} (k={k}, n={n}{lreg_str}){act_suffix}"
        )
    if stem.startswith("subspace_inference"):
        n = extract_stem_value(stem, "n")
        t = extract_stem_value(stem, "T")
        t_suffix = f" (T={t})" if t != "?" else ""
        return f"Subspace{calib_suffix} (n={n}){t_suffix}{act_suffix}"
    if stem.startswith("base_model") or stem.startswith("data_") or stem.endswith(".npz"):
        return None
    return None


def gym_sort_key(stem: str, label: str) -> tuple[int, str, str]:
    if stem.startswith("mc_dropout"):
        family = 0
    elif stem.startswith("standard_ensemble"):
        family = 1
    elif stem.startswith("subspace_inference"):
        family = 2
    elif stem.startswith("swag"):
        family = 3
    elif stem.startswith("laplace"):
        family = 4
    elif stem.startswith("evidential"):
        family = 5
    elif stem.startswith("pjsvd"):
        family = 6
    elif stem.startswith("hybrid_pnc_de"):
        family = 7
    else:
        family = 8
    return (family, label, stem)


def normalize_gym_stem(stem: str, max_over: set[str]) -> str:
    normalized = stem
    if "prel2" in max_over:
        # Strip the .pre_l2 suffix first so downstream regexes (e.g. _T\d[\d.]*)
        # don't greedily consume the leading dot.
        normalized = normalized.replace(".pre_l2", "")
    if "esn" in max_over:
        normalized = normalized.replace("_esn", "")
    if "vcal" in max_over:
        normalized = normalized.replace("_vcal", "")
    if "prob" in max_over:
        normalized = normalized.replace("_prob", "")
    if "scope" in max_over:
        normalized = normalized.replace("_first_", "_")
        normalized = normalized.replace("_multi_", "_")
    if "mode" in max_over:
        normalized = normalized.replace("_least_squares", "")
        normalized = normalized.replace("_affine", "")
        normalized = normalized.replace("_none", "")
    if "family" in max_over:
        normalized = normalized.replace("_low", "")
        normalized = normalized.replace("_random", "")
        normalized = normalized.replace("_all", "")
    if "backend" in max_over:
        normalized = normalized.replace("_activation_covariance", "")
        normalized = normalized.replace("_projected_residual", "")
    if "full" in max_over:
        normalized = normalized.replace("_full", "")
    if "ens" in max_over:
        normalized = re.sub(r"_nDE\d+_nPnC\d+", "", normalized)
    if "k" in max_over:
        normalized = re.sub(r"_k\d[\d.]*", "", normalized)
    if "n" in max_over:
        normalized = re.sub(r"_n\d[\d.]*", "", normalized)
    if "lam" in max_over:
        normalized = re.sub(r"_lam[\d.e+-]+", "", normalized)
    if "dr" in max_over:
        normalized = re.sub(r"_dr[\d.]+", "", normalized)
    if "lreg" in max_over:
        normalized = re.sub(r"_lreg[\d.eE+-]+", "", normalized)
    if "bf" in max_over:
        normalized = re.sub(r"_bf[\d.eE+-]+", "", normalized)
    if "act" in max_over:
        normalized = re.sub(r"_act-[a-z]+", "", normalized)
    if "T" in max_over:
        normalized = re.sub(r"_T\d[\d.]*", "", normalized)
    if "grid" in max_over:
        normalized = re.sub(r"_ps[\d.-]+", "", normalized)
        normalized = re.sub(r"_priors[\d.-]+", "", normalized)
    return normalized.replace("__", "_")


def active_max_over_note(max_over: set[str]) -> str | None:
    if not max_over:
        return None
    collapsed = ", ".join(sorted(max_over))
    remaining = ", ".join(sorted(GYM_MAX_OVER_CHOICES - max_over))
    return (
        f"selection note: collapsed over {collapsed}; rows still split on remaining "
        f"stem distinctions such as {remaining}"
    )


def method_name(stem: str, profile: str) -> str | None:
    if stem.endswith(".npz"):
        return None
    if profile == "gym":
        return gym_friendly_name(stem)

    if stem.startswith("baseline_preact_resnet18"):
        return "Single Model"
    if stem.startswith("baseline_mc_dropout"):
        return "MC Dropout"
    if stem.startswith("baseline_standard_ensemble"):
        return "Deep Ensemble"
    if stem.startswith("baseline_swag"):
        return "SWAG"
    if stem.startswith("baseline_llla"):
        return "Laplace"
    if stem.startswith("pjsvd_"):
        if "_random_" in stem:
            return "PJSVD Random"
        if "_low_" in stem:
            return "PJSVD Low"
    if stem.startswith("baseline_epinet"):
        return "Epinet"
    if stem.startswith("pnc_single_block"):
        return "PnC Single Block"
    if stem.startswith("pnc_multi_block"):
        return "PnC Multi Block"
    return None


def detect_profile(env_dir: Path) -> str:
    for path in sorted(env_dir.glob("*.json")):
        try:
            with open(path) as f:
                data = json.load(f)
        except Exception:
            continue
        if isinstance(data, dict):
            if {"accuracy", "nll", "ece"} & set(data.keys()):
                return "cifar"
            first = next(iter(data.values())) if data else None
            if isinstance(first, dict) and {"accuracy", "nll", "ece"} & set(first.keys()):
                return "cifar"
    return "gym"


def config_label(stem: str, method: str, profile: str, config: str) -> str:
    if profile == "gym":
        if stem.startswith("laplace") or method.startswith("Laplace"):
            return f"{method} (prior={config})"
        if stem.startswith("pjsvd") or method.startswith("PJSVD"):
            return f"{method} (size={config})"
        return f"{method} ({config})"

    if method == "Laplace":
        return f"Laplace (prior={config})"
    if method == "Epinet":
        return f"Epinet (ps={config})"
    if method.startswith("PnC"):
        return f"{method} (scale={config})"
    return f"{method} ({config})"


def config_from_stem(stem: str, method: str, profile: str) -> str | None:
    def extract(pattern: str) -> str | None:
        match = re.search(pattern, stem)
        return match.group(1) if match else None

    if profile == "gym":
        if method.startswith("Laplace"):
            return extract(r"prior([^_]+)")
        if method.startswith("PJSVD"):
            return extract(r"ps([^_]+)")
        return None

    if method == "MC Dropout":
        n = extract(r"_n(\d+)")
        dr = extract(r"_dr([0-9.]+)")
        parts = []
        if n is not None:
            parts.append(f"n={n}")
        if dr is not None:
            parts.append(f"dr={dr}")
        return ", ".join(parts) if parts else None
    if method == "Deep Ensemble":
        n = extract(r"_n(\d+)")
        return f"n={n}" if n is not None else None
    if method == "SWAG":
        n = extract(r"_n(\d+)")
        sws = extract(r"_sws([^_]+)")
        parts = []
        if n is not None:
            parts.append(f"n={n}")
        if sws is not None:
            parts.append(f"sws={sws}")
        return ", ".join(parts) if parts else None
    if method == "Laplace":
        prec = extract(r"_prec([^_]+)")
        return prec
    if method == "Epinet":
        ps = extract(r"_ps([^_]+)")
        return ps
    return None


def load_env_results(
    env_dir: Path,
    profile: str,
    seed_filter: set[int] | None = None,
    require: dict[str, str] | None = None,
) -> dict[str, dict]:
    groups: dict[str, dict] = defaultdict(
        lambda: {
            "label": None,
            "flat": defaultdict(list),
            "configs": defaultdict(lambda: defaultdict(list)),
        }
    )

    for path in sorted(env_dir.glob("*.json")):
        if path.name.endswith(".npz.json"):
            continue

        if seed_filter is not None:
            m = re.search(r"_seed(\d+)", path.stem)
            if m is None or int(m.group(1)) not in seed_filter:
                continue

        stem = canonical_stem(path)
        if require and not stem_satisfies_require(stem, require):
            continue
        method = method_name(stem, profile)
        if method is None:
            continue
        group_key = stem if profile == "gym" else method
        groups[group_key]["label"] = method

        with open(path) as f:
            data = json.load(f)

        first = next(iter(data.values())) if isinstance(data, dict) and data else None
        if isinstance(first, dict):
            for config, metrics in data.items():
                for metric, value in metrics.items():
                    if isinstance(value, (int, float)):
                        groups[group_key]["configs"][str(config)][metric].append(float(value))
        else:
            config = config_from_stem(stem, method, profile)
            for metric, value in data.items():
                if isinstance(value, (int, float)):
                    if config is None:
                        groups[group_key]["flat"][metric].append(float(value))
                    else:
                        groups[group_key]["configs"][config][metric].append(float(value))

    return groups


def _select_value(metrics: dict[str, list[float]], primary: str, fallback: str | None) -> float:
    """Return the mean of `primary` if present and finite, else fall back."""
    mu, _ = mean_std(metrics.get(primary, []))
    if not math.isnan(mu):
        return mu
    if fallback is not None:
        mu_fb, _ = mean_std(metrics.get(fallback, []))
        return mu_fb
    return float("nan")


def choose_best_config(
    configs: dict[str, dict[str, list[float]]],
    selection_metric: str,
    fallback_metric: str | None = None,
) -> str:
    ranked = []
    for config, metrics in configs.items():
        mu = _select_value(metrics, selection_metric, fallback_metric)
        ranked.append((mu, numeric_key(config), config))
    ranked.sort()
    return ranked[0][2]


def choose_best_candidate(
    candidates: list[tuple[tuple[str, dict[str, list[float]]], tuple[int, str, str]]],
    selection_metric: str,
    fallback_metric: str | None = None,
) -> tuple[str, dict[str, list[float]]]:
    ranked = []
    for row, sort_key in candidates:
        label, metrics = row
        mu = _select_value(metrics, selection_metric, fallback_metric)
        ranked.append((mu, sort_key, label, metrics))
    ranked.sort()
    _, _, label, metrics = ranked[0]
    return label, metrics


def method_rows(
    env_groups: dict[str, dict], profile: str, max_over: set[str]
) -> list[tuple[str, dict[str, list[float]]]]:
    if profile == "gym":
        profile_cfg = PROFILES[profile]
        grouped_keys: dict[str, list[str]] = defaultdict(list)
        for key in env_groups:
            bucket = normalize_gym_stem(key, max_over) if max_over else key
            grouped_keys[bucket].append(key)

        rows_with_order = []
        for bucket, keys in grouped_keys.items():
            candidates = []
            for key in keys:
                group = env_groups[key]
                label = group["label"] or key
                sort_key = gym_sort_key(key, label)
                if group["configs"]:
                    config = choose_best_config(
                        group["configs"],
                        profile_cfg["selection_metric"],
                        profile_cfg.get("selection_metric_fallback"),
                    )
                    row = (config_label(key, label, profile, config), group["configs"][config])
                else:
                    row = (label, group["flat"])
                candidates.append((row, sort_key))
            bucket_sort_key = min(sort_key for _, sort_key in candidates)
            rows_with_order.append(
                (
                    bucket_sort_key,
                    choose_best_candidate(
                        candidates,
                        profile_cfg["selection_metric"],
                        profile_cfg.get("selection_metric_fallback"),
                    ),
                )
            )

        rows_with_order.sort(key=lambda item: item[0])
        return [row for _, row in rows_with_order]

    profile_cfg = PROFILES[profile]
    rows = []
    for method in profile_cfg["method_order"]:
        group = env_groups.get(method)
        if not group:
            continue

        if group["configs"]:
            config = choose_best_config(
                group["configs"],
                profile_cfg["selection_metric"],
                profile_cfg.get("selection_metric_fallback"),
            )
            label = config_label("", group["label"] or method, profile, config)
            rows.append((label, group["configs"][config]))
        else:
            rows.append((group["label"] or method, group["flat"]))
    return rows


def render_env_table(
    env: str,
    rows: list[tuple[str, dict[str, list[float]]]],
    metric_specs,
    caption: str,
    include_std: bool,
    bold_pct: float,
    ignore_for_bolding: set[str] | None = None,
    stat_mode: str = "mean_std",
    bold_vs: str | None = None,
    dropped_info: list[tuple[str, str, float, float]] | None = None,
) -> str:
    if bold_vs:
        winners = beat_baseline_indices(rows, metric_specs, bold_vs, ignore_for_bolding)
    else:
        winners = best_metric_indices(rows, metric_specs, bold_pct, ignore_for_bolding)
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{{caption.format(env=env)}}}",
        r"\begin{tabular}{" + ("l" + "c" * len(metric_specs)) + "}",
        r"\toprule",
        "Method & " + " & ".join(header for _, header in metric_specs) + r" \\",
        r"\midrule",
    ]

    for row_idx, (label, row_metrics) in enumerate(rows):
        cells = [
            maybe_bold_tex(
                fmt_cell(row_metrics.get(metric, []), include_std, mode=stat_mode),
                row_idx in winners.get(metric, set()),
            )
            for metric, _ in metric_specs
        ]
        lines.append(f"{label} & " + " & ".join(cells) + r" \\")

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
        ]
    )
    if dropped_info:
        note = "Disqualified (mean > threshold $\\times$ col median): " + "; ".join(
            f"{lbl} [{m}={v:.3f} vs med {med:.3f}]"
            for lbl, m, v, med in dropped_info
        )
        lines.append(r"\footnote{" + note + "}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def render_env_table_text(
    env: str,
    rows: list[tuple[str, dict[str, list[float]]]],
    metric_specs,
    include_std: bool,
    bold_pct: float,
    max_over: set[str],
    ignore_for_bolding: set[str] | None = None,
    stat_mode: str = "mean_std",
    bold_vs: str | None = None,
    dropped_info: list[tuple[str, str, float, float]] | None = None,
) -> str:
    if bold_vs:
        winners = beat_baseline_indices(rows, metric_specs, bold_vs, ignore_for_bolding)
    else:
        winners = best_metric_indices(rows, metric_specs, bold_pct, ignore_for_bolding)
    headers = ["Method"] + [header.replace(r" $\downarrow$", " down").replace(r" $\uparrow$", " up") for _, header in metric_specs]
    body = []
    for row_idx, (label, row_metrics) in enumerate(rows):
        body.append(
            [label]
            + [
                maybe_bold_text(
                    fmt_cell_text(row_metrics.get(metric, []), include_std, mode=stat_mode),
                    row_idx in winners.get(metric, set()),
                )
                for metric, _ in metric_specs
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
    lines = [env]
    note = active_max_over_note(max_over)
    if note is not None:
        lines.append(note)
    lines.extend([sep, fmt_row(headers), sep])
    for row in body:
        lines.append(fmt_row(row))
    lines.append(sep)
    if dropped_info:
        lines.append("disqualified rows (mean > threshold × col median):")
        for label, metric, mu, median in dropped_info:
            lines.append(f"  - {label} | {metric}={mu:.3f} vs median={median:.3f}")
    return "\n".join(lines)


def build_tables(
    results_dir: Path,
    env_filter: str | None,
    fmt: str,
    include_std: bool,
    bold_pct: float,
    ignore_deep_ensemble_for_bolding: bool,
    max_over: set[str],
    stat_mode: str = "mean_std",
    seed_filter: set[int] | None = None,
    exclude: list[str] | None = None,
    bold_vs: str | None = None,
    require: dict[str, str] | None = None,
    disqualify_ratio: float = 0.0,
) -> str:
    env_dirs = sorted(p for p in results_dir.iterdir() if p.is_dir())
    if env_filter is not None:
        env_dirs = [p for p in env_dirs if p.name == env_filter]

    exclude_lower = [e.lower() for e in (exclude or [])]

    tables = []
    for env_dir in env_dirs:
        profile = detect_profile(env_dir)
        profile_cfg = PROFILES[profile]
        groups = load_env_results(
            env_dir, profile, seed_filter=seed_filter, require=require
        )
        rows = method_rows(groups, profile, max_over if profile == "gym" else set())
        if exclude_lower:
            rows = [
                (label, metrics)
                for label, metrics in rows
                if not any(token in label.lower() for token in exclude_lower)
            ]
        dropped_info: list[tuple[str, str, float, float]] = []
        if disqualify_ratio > 0:
            rows, dropped_info = disqualify_outlier_rows(
                rows, profile_cfg["metrics"], disqualify_ratio
            )
        ignore_for_bolding = {"Deep Ensemble"} if ignore_deep_ensemble_for_bolding else set()
        if rows:
            if fmt == "tex":
                tables.append(
                    render_env_table(
                        env_dir.name,
                        rows,
                        profile_cfg["metrics"],
                        profile_cfg["caption"],
                        include_std,
                        bold_pct,
                        ignore_for_bolding=ignore_for_bolding,
                        stat_mode=stat_mode,
                        bold_vs=bold_vs,
                        dropped_info=dropped_info,
                    )
                )
            else:
                tables.append(
                    render_env_table_text(
                        env_dir.name,
                        rows,
                        profile_cfg["metrics"],
                        include_std,
                        bold_pct,
                        max_over if profile == "gym" else set(),
                        ignore_for_bolding=ignore_for_bolding,
                        stat_mode=stat_mode,
                        bold_vs=bold_vs,
                        dropped_info=dropped_info,
                    )
                )

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
    parser.add_argument(
        "--bold-ignore-deep-ensemble",
        action="store_true",
        help="Exclude the 'Deep Ensemble' row when determining which values get bolded.",
    )
    parser.add_argument(
        "--max-over",
        default="",
        help=(
            "Comma-separated Gym stem parameters to optimize over when picking one row per algorithm. "
            f"Choices: {', '.join(sorted(GYM_MAX_OVER_CHOICES))}."
        ),
    )
    parser.add_argument("--out", default=None, help="Optional output path.")
    parser.add_argument(
        "--stat-mode",
        choices=["mean_std", "median_iqr"],
        default="mean_std",
        help="Per-cell statistic. mean_std=mean ± stddev (default); median_iqr=median (IQR) — more robust with small n.",
    )
    parser.add_argument(
        "--seeds",
        default=None,
        help=(
            "Optional comma-separated list of seeds to restrict aggregation to. "
            "Useful to fairly compare methods with different total seed counts by filtering to a common subset."
        ),
    )
    parser.add_argument(
        "--exclude",
        default="",
        help=(
            "Comma-separated substrings; any row whose label contains one (case-insensitive) is dropped. "
            "Examples: --exclude Evidential; --exclude PJSVD,Hybrid."
        ),
    )
    parser.add_argument(
        "--bold-vs",
        default=None,
        help=(
            "Substring matching a baseline row label (case-insensitive). Bold every row whose mean "
            "ties or beats that baseline in each metric column, respecting metric direction. "
            "Overrides --bold when set. Example: --bold-vs 'Deep Ensemble'."
        ),
    )
    parser.add_argument(
        "--require",
        default="",
        help=(
            "Comma-separated key=value clauses; only stems satisfying every clause are loaded. "
            "Binary markers (vcal, prob, full, prel2, esn) take yes/no. Choice dims (scope, family, "
            "backend, mode) take a choice name or 'none'. Value dims (k, n, T, lam, act, dr, lreg) "
            "take an exact value. Keys that belong to a specific method family (e.g. lreg → PJSVD, "
            "lam → Evidential, dr → MC Dropout) are auto-scoped: a clause on them is silently "
            "skipped for stems from unrelated families, so the filter doesn't drop unrelated rows. "
            "Examples: --require vcal=no; --require family=low,mode=least_squares; "
            "--require lreg=0.0001 (keeps non-PJSVD rows too)."
        ),
    )
    parser.add_argument(
        "--disqualify-ratio",
        type=float,
        default=0.0,
        metavar="K",
        help=(
            "Per-env, disqualify any row whose mean on any lower-is-better metric is greater "
            "than K × the column median. Higher-is-better metrics (AUROC/AUPR) and columns "
            "with non-positive medians are skipped. A footnote lists which rows were dropped "
            "and why. Default 0 disables; 10 matches the paper-table convention."
        ),
    )
    args = parser.parse_args()

    fmt = args.fmt
    if fmt is None:
        fmt = "text" if args.out and args.out.endswith(".txt") else "tex"

    max_over = {item.strip() for item in args.max_over.split(",") if item.strip()}
    invalid = sorted(max_over - GYM_MAX_OVER_CHOICES)
    if invalid:
        parser.error(
            f"Invalid --max-over value(s): {', '.join(invalid)}. "
            f"Valid choices: {', '.join(sorted(GYM_MAX_OVER_CHOICES))}."
        )

    seed_filter: set[int] | None = None
    if args.seeds is not None:
        seed_filter = {int(s.strip()) for s in args.seeds.split(",") if s.strip()}

    exclude = [item.strip() for item in args.exclude.split(",") if item.strip()]

    require: dict[str, str] = {}
    for item in args.require.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            parser.error(f"Invalid --require token '{item}'; expected key=value.")
        key, _, value = item.partition("=")
        key = key.strip()
        value = value.strip()
        if key not in REQUIRE_KEYS:
            parser.error(
                f"Unknown --require key '{key}'. "
                f"Valid keys: {', '.join(sorted(REQUIRE_KEYS))}."
            )
        if not value:
            parser.error(f"Empty value for --require '{key}'; expected key=value.")
        require[key] = value

    tex = build_tables(
        Path(args.results_dir),
        args.env,
        fmt,
        args.include_std,
        args.bold,
        args.bold_ignore_deep_ensemble,
        max_over,
        stat_mode=args.stat_mode,
        seed_filter=seed_filter,
        exclude=exclude,
        bold_vs=args.bold_vs,
        require=require,
        disqualify_ratio=args.disqualify_ratio,
    )
    if args.out:
        Path(args.out).write_text(tex)
    else:
        print(tex, end="")


if __name__ == "__main__":
    main()
