#!/usr/bin/env python3
"""Comprehensive analysis of mechanism experiment results."""

from pathlib import Path
import sys


def main():
    print("\n" + "=" * 70)
    print("MECHANISM EXPERIMENT ANALYSIS")
    print("=" * 70 + "\n")

    results_dir = Path("results/mechanism")

    # Import analysis utilities
    try:
        from mechanism_metrics import (
            load_all_results,
            summarize_results,
            print_summary,
            compute_family_statistics,
            test_family_ordering,
            compute_singular_rank_correlation,
            compute_correction_benefit,
            compute_calibration_vs_held_out_correlation,
        )
    except ImportError as e:
        print(f"Error importing analysis utilities: {e}")
        sys.exit(1)

    # Load results
    try:
        df = load_all_results(results_dir)
        print(f"✓ Loaded {len(df)} trials from {results_dir}\n")
    except Exception as e:
        print(f"Error loading results: {e}")
        sys.exit(1)

    # Generate comprehensive summary
    summary = summarize_results(
        df, output_path=results_dir / "comprehensive_summary.json"
    )
    print_summary(summary)

    # Per-environment analysis
    print("\n" + "=" * 70)
    print("PER-ENVIRONMENT ANALYSIS")
    print("=" * 70 + "\n")

    for env in sorted(df["environment"].unique()):
        env_data = df[df["environment"] == env]
        print(f"\n{env}:")
        print(f"  Total trials: {len(env_data)}")
        print(f"  Layers tested: {sorted(env_data['layer'].unique())}")

        # Family stats
        family_stats = compute_family_statistics(env_data, damage_metric="rmse_after")
        for family in ["low_singular", "random", "high_singular"]:
            stats_info = family_stats[family]
            print(
                f"  {family:15s}: mean={stats_info['mean']:.6f}, std={stats_info['std']:.6f}"
            )

    # Per-layer analysis
    print("\n" + "=" * 70)
    print("PER-LAYER ANALYSIS")
    print("=" * 70 + "\n")

    for layer in sorted(df["layer"].unique()):
        layer_data = df[df["layer"] == layer]
        print(f"\nLayer {layer}:")
        print(f"  Total trials: {len(layer_data)}")

        family_stats = compute_family_statistics(layer_data, damage_metric="rmse_after")
        for family in ["low_singular", "random", "high_singular"]:
            stats_info = family_stats[family]
            print(
                f"  {family:15s}: mean={stats_info['mean']:.6f}, std={stats_info['std']:.6f}"
            )

    # Scale-dependent analysis
    print("\n" + "=" * 70)
    print("PERTURBATION SCALE ANALYSIS")
    print("=" * 70 + "\n")

    scales = sorted(df["pert_scale"].unique())
    print(f"Scales tested: {scales}\n")

    for scale in scales:
        scale_data = df[df["pert_scale"] == scale]
        family_stats = compute_family_statistics(scale_data, damage_metric="rmse_after")
        print(f"\nScale {scale:.4f}:")
        for family in ["low_singular", "random", "high_singular"]:
            stats_info = family_stats[family]
            print(f"  {family:15s}: mean={stats_info['mean']:.6f}")

    # Export comprehensive results
    print("\n" + "=" * 70)
    print("EXPORTING DETAILED RESULTS")
    print("=" * 70 + "\n")

    # Full DataFrame CSV
    export_path = results_dir / "detailed_results.csv"
    df.to_csv(export_path, index=False)
    print(f"✓ Exported detailed results to {export_path}")

    # Summary statistics CSV
    summary_stats = []
    for env in sorted(df["environment"].unique()):
        for layer in sorted(df["layer"].unique()):
            env_layer_data = df[(df["environment"] == env) & (df["layer"] == layer)]
            if len(env_layer_data) == 0:
                continue

            for family in ["low_singular", "random", "high_singular"]:
                family_data = env_layer_data[env_layer_data["family"] == family]
                if len(family_data) == 0:
                    continue

                rmse_after = family_data["rmse_after"]
                summary_stats.append(
                    {
                        "environment": env,
                        "layer": layer,
                        "family": family,
                        "num_trials": len(family_data),
                        "rmse_after_mean": float(rmse_after.mean()),
                        "rmse_after_std": float(rmse_after.std()),
                        "rmse_after_min": float(rmse_after.min()),
                        "rmse_after_max": float(rmse_after.max()),
                    }
                )

    import pandas as pd

    summary_df = pd.DataFrame(summary_stats)
    summary_path = results_dir / "family_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"✓ Exported family summary to {summary_path}")

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
