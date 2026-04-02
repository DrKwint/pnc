"""Plotting functions for mechanism experiment results."""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Any

# Set up plotting style
plt.style.use("default")
sns.set_palette("husl")


def print_summary_statistics(df: pd.DataFrame) -> None:
    """Print summary statistics for the experiment results."""
    print("\n" + "=" * 80)
    print("MECHANISM EXPERIMENT SUMMARY STATISTICS")
    print("=" * 80)

    print(f"Total trials: {len(df)}")
    print(f"Environments: {sorted(df['environment'].unique())}")
    print(f"Layers: {sorted(df['layer'].unique())}")
    print(f"Families: {sorted(df['family'].unique())}")

    # Per-family statistics
    print("\nPer-family held-out RMSE after correction:")
    for family in ["low_singular", "random", "high_singular"]:
        fam_data = df[df["family"] == family]
        if len(fam_data) > 0:
            mean_rmse = fam_data["rmse_after"].mean()
            std_rmse = fam_data["rmse_after"].std()
            print(".4f")

    # Test family ordering hypothesis
    print("\nFamily ordering test (lower is better):")
    families = ["low_singular", "random", "high_singular"]
    for i, fam1 in enumerate(families[:-1]):
        for fam2 in families[i + 1 :]:
            data1 = df[df["family"] == fam1]["rmse_after"]
            data2 = df[df["family"] == fam2]["rmse_after"]
            if len(data1) > 0 and len(data2) > 0:
                mean_diff = data1.mean() - data2.mean()
                direction = "better" if mean_diff < 0 else "worse"
                print(".4f")

    # Singular value correlation
    print("\nSingular value vs damage correlation:")
    valid_data = df[df["family"].isin(["low_singular", "high_singular"])]
    if len(valid_data) > 0:
        corr = np.corrcoef(valid_data["singular_value"], valid_data["rmse_after"])[0, 1]
        print(".4f")

    print("=" * 80)


def generate_all_plots(
    df: pd.DataFrame, plots_dir: Path, env_name: str
) -> Dict[str, plt.Figure]:
    """Generate all analysis plots for the mechanism experiment."""
    figures = {}

    # Create plots directory
    plots_dir.mkdir(exist_ok=True)

    # Simple plots for each layer
    for layer_idx in sorted(df["layer"].unique()):
        layer_df = df[df["layer"] == layer_idx]
        layer_suffix = f"_layer{layer_idx}"

        # 1. Family ordering
        fig1 = plot_family_ordering_simple(
            layer_df,
            layer_idx,
            plots_dir / f"{env_name}_family_ordering{layer_suffix}.png",
        )
        figures[f"family_ordering{layer_suffix}"] = fig1

        # 2. Singular rank trend
        fig2 = plot_singular_rank_trend_simple(
            layer_df,
            layer_idx,
            plots_dir / f"{env_name}_singular_trend{layer_suffix}.png",
        )
        figures[f"singular_trend{layer_suffix}"] = fig2

        # 3. Before/after correction
        fig3 = plot_before_after_simple(
            layer_df,
            layer_idx,
            plots_dir / f"{env_name}_correction_benefit{layer_suffix}.png",
        )
        figures[f"correction_benefit{layer_suffix}"] = fig3

        # 4. Calibration vs held-out
        fig4 = plot_calibration_vs_held_out_simple(
            layer_df,
            layer_idx,
            plots_dir / f"{env_name}_calibration_correlation{layer_suffix}.png",
        )
        figures[f"calibration_correlation{layer_suffix}"] = fig4

    return figures


def plot_family_ordering_simple(
    df: pd.DataFrame, layer_idx: int, save_path: Optional[Path] = None
) -> plt.Figure:
    """Simple family ordering plot."""
    fig, ax = plt.subplots(figsize=(8, 6))

    families = ["low_singular", "random", "high_singular"]
    colors = ["green", "gray", "red"]

    for i, (family, color) in enumerate(zip(families, colors)):
        fam_data = df[df["family"] == family]
        if len(fam_data) > 0:
            ax.bar(
                i,
                fam_data["rmse_after"].mean(),
                color=color,
                alpha=0.7,
                label=family.replace("_", " ").title(),
            )

    ax.set_ylabel("Held-out RMSE After Correction")
    ax.set_title(f"Family Ordering - Layer {layer_idx}")
    ax.set_xticks(range(len(families)))
    ax.set_xticklabels([f.replace("_", " ").title() for f in families])
    ax.legend()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved family ordering plot to {save_path}")

    return fig


def plot_singular_rank_trend_simple(
    df: pd.DataFrame, layer_idx: int, save_path: Optional[Path] = None
) -> plt.Figure:
    """Simple singular rank trend plot."""
    fig, ax = plt.subplots(figsize=(8, 6))

    families = ["low_singular", "random", "high_singular"]
    colors = ["green", "gray", "red"]

    for family, color in zip(families, colors):
        fam_data = df[df["family"] == family]
        if len(fam_data) > 0:
            ax.scatter(
                fam_data["dir_rank"],
                fam_data["rmse_after"],
                color=color,
                alpha=0.6,
                label=family.replace("_", " ").title(),
            )

    ax.set_xlabel("Direction Rank")
    ax.set_ylabel("Held-out RMSE After Correction")
    ax.set_title(f"Singular Value Rank vs Damage - Layer {layer_idx}")
    ax.legend()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved singular rank trend plot to {save_path}")

    return fig


def plot_before_after_simple(
    df: pd.DataFrame, layer_idx: int, save_path: Optional[Path] = None
) -> plt.Figure:
    """Simple before/after correction plot."""
    fig, ax = plt.subplots(figsize=(8, 6))

    families = ["low_singular", "random", "high_singular"]
    colors = ["green", "gray", "red"]

    for i, (family, color) in enumerate(zip(families, colors)):
        fam_data = df[df["family"] == family]
        if len(fam_data) > 0:
            before = fam_data["rmse_before"].mean()
            after = fam_data["rmse_after"].mean()
            ax.bar(
                i - 0.2,
                before,
                0.4,
                color="lightcoral",
                alpha=0.7,
                label="Before" if i == 0 else "",
            )
            ax.bar(
                i + 0.2,
                after,
                0.4,
                color=color,
                alpha=0.7,
                label="After" if i == 0 else "",
            )

    ax.set_ylabel("RMSE")
    ax.set_title(f"Before/After Correction - Layer {layer_idx}")
    ax.set_xticks(range(len(families)))
    ax.set_xticklabels([f.replace("_", " ").title() for f in families])
    ax.legend()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved before/after correction plot to {save_path}")

    return fig


def plot_calibration_vs_held_out_simple(
    df: pd.DataFrame, layer_idx: int, save_path: Optional[Path] = None
) -> plt.Figure:
    """Simple calibration vs held-out plot."""
    fig, ax = plt.subplots(figsize=(8, 6))

    families = ["low_singular", "random", "high_singular"]
    colors = ["green", "gray", "red"]

    for family, color in zip(families, colors):
        fam_data = df[df["family"] == family]
        if len(fam_data) > 0:
            ax.scatter(
                fam_data["calibration_residual"],
                fam_data["rmse_after"],
                color=color,
                alpha=0.6,
                label=family.replace("_", " ").title(),
            )

    ax.set_xlabel("Calibration Residual")
    ax.set_ylabel("Held-out RMSE After Correction")
    ax.set_title(f"Calibration vs Held-Out - Layer {layer_idx}")
    ax.legend()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved calibration vs held-out plot to {save_path}")

    return fig
