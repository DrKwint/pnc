"""Analysis script for mechanism experiment results."""

import json
import pandas as pd
from pathlib import Path
from mechanism_plot import generate_all_plots, print_summary_statistics


def load_experiment_results(
    results_dir: Path, env_name: str, seed: int = 0
) -> pd.DataFrame:
    """Load experiment results from JSON lines file."""
    jsonl_path = results_dir / f"{env_name}_seed{seed}_trials.jsonl"

    if not jsonl_path.exists():
        raise FileNotFoundError(f"Results file not found: {jsonl_path}")

    # Load JSON lines
    trials = []
    with open(jsonl_path, "r") as f:
        for line in f:
            trials.append(json.loads(line.strip()))

    # Convert to DataFrame
    df = pd.DataFrame(trials)

    # Flatten metrics column
    if "metrics" in df.columns:
        metrics_df = pd.json_normalize(df["metrics"])
        df = pd.concat([df.drop("metrics", axis=1), metrics_df], axis=1)

    # Rename columns to match expected format
    column_mapping = {
        "layer_idx": "layer",
        "direction_family": "family",
        "singular_value_rank": "dir_rank",
        "perturbation_scale": "pert_scale",
    }
    df = df.rename(columns=column_mapping)

    print(f"Loaded {len(df)} trials from {jsonl_path}")

    return df


def analyze_mechanism_experiment(
    env_name: str = "HalfCheetah-v5",
    seed: int = 0,
    results_dir: Path = Path("results/mechanism"),
    plots_dir: Path = Path("results/mechanism/plots"),
):
    """Run complete analysis of mechanism experiment results."""
    print(f"Analyzing mechanism experiment: {env_name} (seed {seed})")

    # Load results
    df = load_experiment_results(results_dir, env_name, seed)

    # Print summary statistics
    print_summary_statistics(df)

    # Generate all plots
    figures = generate_all_plots(df, plots_dir, env_name)

    print(f"\nAnalysis complete. Results saved to {plots_dir}")
    return df, figures


if __name__ == "__main__":
    # Analyze HalfCheetah results
    df, figures = analyze_mechanism_experiment("HalfCheetah-v5")

    # Could also analyze Hopper if available
    try:
        df_hopper, figures_hopper = analyze_mechanism_experiment("Hopper-v5")
    except FileNotFoundError:
        print("Hopper-v5 results not found, skipping...")
