#!/usr/bin/env python3
"""Full mechanism experiment on HalfCheetah-v5 and Hopper-v5."""

from pathlib import Path
from mechanism_experiment import MechanismExperiment

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("MECHANISM EXPERIMENT - FULL RUN")
    print("=" * 70 + "\n")

    for env in ["HalfCheetah-v5", "Hopper-v5"]:
        print("\n" + "-" * 70)
        print(f"Running experiment on {env}...")
        print("-" * 70 + "\n")

        # Create experiment with full parameters
        exp = MechanismExperiment(
            env=env,
            seed=0,
            num_directions_per_family=20,  # Full specification
            num_scales=4,
            scale_range=(0.01, 0.2),  # 1%, 5%, 10%, 20%
            activation="relu",
        )

        print(f"Running mechanism experiment on {env}...\n")
        trials = exp.run_all_trials()

        print("\n" + "=" * 70)
        print("Saving results...")
        exp.save_results(output_dir=Path("results/mechanism"))

        print(f"\n✓ {env}: {len(trials)} trials completed")

    print("\n" + "=" * 70)
    print("MACHINE EXPERIMENT COMPLETE")
    print("=" * 70 + "\n")
