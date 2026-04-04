"""Metrics and utility functions for mechanism experiment analysis."""

import json
from pathlib import Path
from typing import Any, Optional, Tuple

import jax.numpy as jnp
import numpy as np
import pandas as pd
from scipy import stats


def load_trials_from_jsonl(jsonl_path: Path) -> list[dict]:
    """Load trial results from JSONL file."""
    trials = []
    with open(jsonl_path, "r") as f:
        for line in f:
            if line.strip():
                trials.append(json.loads(line))
    return trials


def load_all_results(results_dir: Path) -> pd.DataFrame:
    """Load all trial results into a single DataFrame."""
    all_trials = []
    csv_files = list(results_dir.glob("*_summary.csv"))

    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        all_trials.append(df)

    if not all_trials:
        raise ValueError(f"No trial results found in {results_dir}")

    return pd.concat(all_trials, ignore_index=True)


def compute_family_statistics(
    df: pd.DataFrame, damage_metric: str = "rmse_after"
) -> dict:
    """Compute per-family statistics.

    Args:
        df: Trial results DataFrame
        damage_metric: Column name to use for damage measurement

    Returns:
        dict with keys: low_singular, random, high_singular
        Each contains: mean, std, median, min, max
    """
    stats_dict = {}

    for family in ["low_singular", "random", "high_singular"]:
        family_data = df[df["family"] == family][damage_metric]

        stats_dict[family] = {
            "mean": float(family_data.mean()),
            "std": float(family_data.std()),
            "median": float(family_data.median()),
            "min": float(family_data.min()),
            "max": float(family_data.max()),
            "count": len(family_data),
        }

    return stats_dict


def test_family_ordering(df: pd.DataFrame, damage_metric: str = "rmse_after") -> dict:
    """Statistical test of family ordering: low < random < high.

    Computes pairwise Mann-Whitney U tests and reports p-values.
    """
    families = ["low_singular", "random", "high_singular"]
    data_by_family = {f: df[df["family"] == f][damage_metric].values for f in families}

    results = {}

    # Test low_singular < random
    u_low_rand, p_low_rand = stats.mannwhitneyu(
        data_by_family["low_singular"], data_by_family["random"], alternative="less"
    )
    results["low_vs_random"] = {
        "u_statistic": float(u_low_rand),
        "p_value": float(p_low_rand),
        "low_mean": float(np.mean(data_by_family["low_singular"])),
        "random_mean": float(np.mean(data_by_family["random"])),
    }

    # Test random < high_singular
    u_rand_high, p_rand_high = stats.mannwhitneyu(
        data_by_family["random"], data_by_family["high_singular"], alternative="less"
    )
    results["random_vs_high"] = {
        "u_statistic": float(u_rand_high),
        "p_value": float(p_rand_high),
        "random_mean": float(np.mean(data_by_family["random"])),
        "high_mean": float(np.mean(data_by_family["high_singular"])),
    }

    # Test low_singular < high_singular
    u_low_high, p_low_high = stats.mannwhitneyu(
        data_by_family["low_singular"],
        data_by_family["high_singular"],
        alternative="less",
    )
    results["low_vs_high"] = {
        "u_statistic": float(u_low_high),
        "p_value": float(p_low_high),
        "low_mean": float(np.mean(data_by_family["low_singular"])),
        "high_mean": float(np.mean(data_by_family["high_singular"])),
    }

    return results


def compute_singular_rank_correlation(
    df: pd.DataFrame, damage_metric: str = "rmse_after"
) -> dict:
    """Compute correlation between singular value rank and damage metric.

    Should be negative: lower singular values → less damage.
    """
    # Singular value rank: lower rank = smaller singular value (safer)
    # We'll use -dir_rank so that lower singular values have higher rank values
    df_copy = df.copy()
    df_copy["rank_score"] = -df_copy[
        "dir_rank"
    ].values  # Negate so low-singular has high score

    # Compute Spearman correlation
    corr, p_val = stats.spearmanr(
        df_copy["rank_score"].values, df_copy[damage_metric].values
    )

    return {
        "spearman_correlation": float(corr),
        "p_value": float(p_val),
        "interpretation": "Negative correlation indicates lower singular values cause less damage",
    }


def compute_correction_benefit(df: pd.DataFrame) -> dict:
    """Compute the benefit of correction: reduction in damage after correction.

    Benefit = (rmse_before - rmse_after) / rmse_before
    (positive = improvement)
    """
    df_copy = df.copy()

    # Avoid division by zero
    df_copy["correction_benefit"] = (df_copy["rmse_before"] - df_copy["rmse_after"]) / (
        df_copy["rmse_before"] + 1e-8
    )

    # Per family
    benefits_by_family = {}
    for family in ["low_singular", "random", "high_singular"]:
        family_benefits = df_copy[df_copy["family"] == family]["correction_benefit"]
        benefits_by_family[family] = {
            "mean_benefit": float(family_benefits.mean()),
            "std_benefit": float(family_benefits.std()),
            "median_benefit": float(family_benefits.median()),
        }

    return {
        "by_family": benefits_by_family,
        "overall_mean": float(df_copy["correction_benefit"].mean()),
    }


def compute_calibration_vs_held_out_correlation(df: pd.DataFrame) -> dict:
    """Correlation between calibration residual and held-out damage.

    Should be positive if correction success on calibration predicts held-out success.
    """
    corr, p_val = stats.spearmanr(
        df["calibration_residual"].values, df["rmse_after"].values
    )

    return {
        "spearman_correlation": float(corr),
        "p_value": float(p_val),
        "interpretation": "Positive correlation indicates calibration surrogate predicts held-out result",
    }


def summarize_results(df: pd.DataFrame, output_path: Optional[Path] = None) -> dict:
    """Compute comprehensive summary statistics."""
    summary = {
        "family_statistics": compute_family_statistics(df),
        "family_ordering_tests": test_family_ordering(df),
        "singular_rank_correlation": compute_singular_rank_correlation(df),
        "correction_benefit": compute_correction_benefit(df),
        "calibration_correlation": compute_calibration_vs_held_out_correlation(df),
        "total_trials": len(df),
        "environments": df["environment"].unique().tolist(),
        "layers_tested": df["layer"].unique().tolist(),
    }

    if output_path:
        with open(output_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"Saved summary to {output_path}")

    return summary


def print_summary(summary: dict) -> None:
    """Pretty-print summary statistics."""
    print(f"\n{'=' * 70}")
    print("MECHANISM EXPERIMENT SUMMARY")
    print(f"{'=' * 70}\n")

    print("FAMILY STATISTICS (rmse_after):")
    for family, stats_info in summary["family_statistics"].items():
        print(f"\n  {family.upper()}:")
        print(f"    Mean:   {stats_info['mean']:.6f}")
        print(f"    Std:    {stats_info['std']:.6f}")
        print(f"    Median: {stats_info['median']:.6f}")
        print(f"    Count:  {stats_info['count']}")

    print("\n\nFAMILY ORDERING TESTS (Mann-Whitney U, alternative='less'):")
    for test_name, test_result in summary["family_ordering_tests"].items():
        print(f"\n  {test_name}:")
        print(
            f"    p-value: {test_result['p_value']:.6f} {'✓' if test_result['p_value'] < 0.05 else '✗'}"
        )

    print("\n\nSINGULAR RANK CORRELATION:")
    corr_info = summary["singular_rank_correlation"]
    print(f"  Spearman r: {corr_info['spearman_correlation']:.4f}")
    print(f"  p-value:    {corr_info['p_value']:.6f}")

    print("\n\nCORRECTION BENEFIT:")
    benefit_info = summary["correction_benefit"]
    print(f"  Overall mean benefit: {benefit_info['overall_mean']:.4f}")
    for family, benefit_stats in benefit_info["by_family"].items():
        print(
            f"  {family}: {benefit_stats['mean_benefit']:.4f} ± {benefit_stats['std_benefit']:.4f}"
        )

    print("\n\nCALIBRATION-HELD-OUT CORRELATION:")
    cal_info = summary["calibration_correlation"]
    print(f"  Spearman r: {cal_info['spearman_correlation']:.4f}")
    print(f"  p-value:    {cal_info['p_value']:.6f}")

    print(f"\n\nTOTAL TRIALS: {summary['total_trials']}")
    print(f"ENVIRONMENTS: {summary['environments']}")
    print(f"LAYERS: {summary['layers_tested']}")
    print(f"\n{'=' * 70}\n")


if __name__ == "__main__":
    # Example usage
    results_dir = Path("results/mechanism")
    if results_dir.exists():
        df = load_all_results(results_dir)
        summary = summarize_results(df, output_path=results_dir / "summary.json")
        print_summary(summary)
    else:
        print(f"Results directory {results_dir} not found")
