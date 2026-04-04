"""Mechanism experiment: test whether PJSVD-identified safe directions preserve nominal behavior."""

import json
import pickle
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from data import collect_data, OODPolicyWrapper
from models import TransitionModel
from training import train_generic
from pjsvd import get_full_span_affine_residuals
from util import seed_everything, _get_activation


@dataclass
class MechanismTrialResult:
    """Result from a single perturb-and-correct trial."""

    trial_id: int
    environment: str
    layer_idx: int  # 0 for l1→l2, 1 for l2→l3
    direction_family: str  # "low_singular" | "random" | "high_singular"
    singular_value_rank: int
    singular_value: float
    perturbation_scale: float  # The scale multiplier applied
    metrics: dict  # Keys: movement_l2, rmse_before, rmse_after, nll_before, nll_after, calibration_residual


class MechanismExperiment:
    """Orchestrates the perturb-and-correct mechanism experiment."""

    def __init__(
        self,
        env: str,
        data_dir: Path = Path("results"),
        steps: int = 10000,
        seed: int = 0,
        num_directions_per_family: int = 20,
        num_scales: int = 4,
        scale_range: Tuple[float, float] = (0.01, 0.2),
        activation: str = "relu",
    ):
        """
        Args:
            env: MuJoCo environment name (e.g., "HalfCheetah-v5")
            data_dir: Base directory for data
            steps: Data collection steps per role
            seed: Random seed
            num_directions_per_family: Number of directions per family (low/random/high)
            num_scales: Number of perturbation scales to test
            scale_range: (min_scale, max_scale) for perturbations (as fraction of typical weight magnitude)
            activation: Activation function name
        """
        self.env = env
        self.data_dir = Path(data_dir)
        self.steps = steps
        self.seed = seed
        self.num_directions_per_family = num_directions_per_family
        self.num_scales = num_scales
        self.scale_range = scale_range
        self.activation = activation

        self.act_fn = _get_activation(activation)

        # Results storage
        self.trials: list[MechanismTrialResult] = []
        self.trial_id_counter = 0

        seed_everything(seed)

    def load_or_collect_data(self) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Load or collect nominal data. Returns (inputs, targets)."""
        data_dir = self.data_dir / self.env
        data_dir.mkdir(parents=True, exist_ok=True)

        id_train_path = (
            data_dir / f"mechanism_data_id_train_seed{self.seed}_steps{self.steps}.npz"
        )

        if id_train_path.exists():
            print(f"Loading existing data from {id_train_path}")
            d = np.load(id_train_path)
            inputs = jnp.array(d["inputs"])
            targets = jnp.array(d["targets"])
        else:
            print(f"Collecting new data: {self.env}")
            inputs_list, targets_list = collect_data(
                self.env, self.steps, OODPolicyWrapper(), seed=self.seed
            )
            inputs = jnp.array(inputs_list)
            targets = jnp.array(targets_list)
            # Save for reuse
            np.savez(id_train_path, inputs=np.array(inputs), targets=np.array(targets))

        return inputs, targets

    def split_data_roles(
        self,
        inputs: jnp.ndarray,
        targets: jnp.ndarray,
        train_split: float = 0.7,
        cal_split: float = 0.15,
        held_out_split: float = 0.15,
    ) -> dict:
        """Split data into training, calibration, and held-out roles.

        Returns:
            dict with keys: x_train, y_train, x_cal, y_cal, x_held_out, y_held_out
        """
        n = len(inputs)
        n_train = int(n * train_split)
        n_cal = int(n * cal_split)
        # Remainder goes to held-out

        # Sequential split to preserve trajectory structure
        rng = np.random.RandomState(self.seed + 1000)
        idx = rng.permutation(n)

        idx_train = idx[:n_train]
        idx_cal = idx[n_train : n_train + n_cal]
        idx_held_out = idx[n_train + n_cal :]

        # Verify no overlap
        assert len(set(idx_train) & set(idx_cal)) == 0, "Train/cal overlap!"
        assert len(set(idx_train) & set(idx_held_out)) == 0, "Train/held-out overlap!"
        assert len(set(idx_cal) & set(idx_held_out)) == 0, "Cal/held-out overlap!"

        print(
            f"Data split: train={len(idx_train)}, cal={len(idx_cal)}, held_out={len(idx_held_out)}"
        )

        return {
            "x_train": inputs[idx_train],
            "y_train": targets[idx_train],
            "x_cal": inputs[idx_cal],
            "y_cal": targets[idx_cal],
            "x_held_out": inputs[idx_held_out],
            "y_held_out": targets[idx_held_out],
        }

    def train_dynamics_model(self, x_train, y_train, x_val, y_val) -> TransitionModel:
        """Train a dynamics model on the training data."""

        def loss_fn(m, x, y):
            pred = m(x)
            return jnp.mean((pred - y) ** 2)

        model = TransitionModel(
            x_train.shape[1],
            y_train.shape[1],
            nnx.Rngs(params=self.seed),
            activation=self.act_fn,
        )

        print(f"\n=== Training baseline dynamics model ({self.env}) ===")
        model = train_generic(
            model,
            x_train,
            y_train,
            x_val,
            y_val,
            loss_fn=loss_fn,
            steps=2000,
            batch_size=64,
            lr=1e-3,
            patience=15,
            log_prefix="Mechanism experiment",
        )
        return model  # type: ignore

    def extract_layer_output(
        self,
        model: TransitionModel,
        inputs: jnp.ndarray,
        layer_idx: int,
    ) -> jnp.ndarray:
        """Extract outputs at a given layer.

        layer_idx=0: output of l2 (after second hidden layer)
        layer_idx=1: output of l3 (final model output)
        """

        def forward_to_layer(x):
            h1 = model.activation(model.l1(x))
            if layer_idx == 0:
                return model.activation(model.l2(h1))
            elif layer_idx == 1:
                h2 = model.activation(model.l2(h1))
                return model.l3(h2)
            else:
                raise ValueError(f"Invalid layer_idx: {layer_idx}")

        outputs = jax.vmap(forward_to_layer)(inputs)
        return outputs

    def compute_layer_jacobian(
        self,
        model: TransitionModel,
        x_batch: jnp.ndarray,
        layer_idx: int,
    ) -> jnp.ndarray:
        """Compute Jacobian of layer output w.r.t. layer parameters, averaged over batch.

        For layer_idx=0: Jacobian of l2 output w.r.t. l2.kernel (64×64 matrix)
        For layer_idx=1: Jacobian of l3 output w.r.t. l3.kernel (64×out_features matrix)

        Returns: Jacobian matrix of shape (output_dim, weight_dim)
        """

        def forward_batch(W_flat):
            # Reshape flattened weights back to matrix
            if layer_idx == 0:
                W = W_flat.reshape(model.l2.kernel.shape)
                h1 = model.activation(jax.vmap(model.l1)(x_batch))
                return jax.vmap(lambda h: model.activation(jnp.dot(h, W)))(h1)
            elif layer_idx == 1:
                W = W_flat.reshape(model.l3.kernel.shape)
                h1 = model.activation(jax.vmap(model.l1)(x_batch))
                h2 = model.activation(
                    jax.vmap(lambda h: jnp.dot(h, model.l2.kernel))(h1)
                )
                return jax.vmap(lambda h: jnp.dot(h, W))(h2)

        # Get current weight matrix
        if layer_idx == 0:
            W_current = model.l2.kernel
        elif layer_idx == 1:
            W_current = model.l3.kernel

        W_flat = W_current.flatten()

        # Compute Jacobian using JAX's jacfwd
        jac_fn = jax.jacfwd(forward_batch)
        J_batch = jac_fn(W_flat)  # Shape: (batch_size, output_dim, weight_dim)

        # Average over batch dimension
        J = jnp.mean(J_batch, axis=0)  # (output_dim, weight_dim)

        return J

    def build_pjsvd_operator(
        self,
        model: TransitionModel,
        x_cal: jnp.ndarray,
        y_cal: jnp.ndarray,
        layer_idx: int,
    ) -> dict:
        """Build theorem-aligned direction families using projected residual operator R = (I - P_A)J.

        Returns dict with keys: low_directions, high_directions, random_directions, singular_values
        All directions are in flattened weight space: (weight_dim, num_directions)
        """
        print(f"  Building PJSVD operator for layer {layer_idx}...")

        # Get weight matrix shape and dimension
        if layer_idx == 0:
            W_shape = model.l2.kernel.shape  # (64, 64)
        elif layer_idx == 1:
            W_shape = model.l3.kernel.shape  # (64, out_features)
        weight_dim = int(jnp.prod(jnp.array(W_shape)))
        num_dirs = self.num_directions_per_family

        # ===== Step 1: Build calibration correction geometry =====
        # Get layer outputs on calibration data
        h_cal = self.extract_layer_output(
            model, x_cal, layer_idx
        )  # (n_cal, layer_out_dim)

        # Build correction subspace: span of [all-ones vector, activation columns]
        # This is the affine correction space P_A
        n_cal, layer_out_dim = h_cal.shape

        # Create augmented matrix: [h_cal.T, ones_vector]
        # h_cal.T has shape (layer_out_dim, n_cal)
        ones_col = jnp.ones((layer_out_dim, 1))  # Bias column
        Y_aug = jnp.concatenate(
            [h_cal.T, ones_col], axis=1
        )  # (layer_out_dim, n_cal + 1)

        # Compute orthonormal basis for correction subspace
        Q_corr, _ = jnp.linalg.qr(
            Y_aug
        )  # (layer_out_dim, min(layer_out_dim, n_cal + 1))

        # Projector P_A onto correction subspace
        P_A = Q_corr @ Q_corr.T  # (layer_out_dim, layer_out_dim)

        print(f"    Correction subspace rank: {Q_corr.shape[1]} / {layer_out_dim}")

        # ===== Step 2: Build Jacobian J of layer outputs w.r.t. layer parameters =====
        # Use a representative sample for Jacobian computation
        x_sample = x_cal[: min(32, len(x_cal))]  # Use up to 32 samples for efficiency

        # Compute Jacobian averaged over the batch
        J = self.compute_layer_jacobian(
            model, x_sample, layer_idx
        )  # (layer_out_dim, weight_dim)

        print(f"    Jacobian shape: {J.shape}")

        # ===== Step 3: Construct projected residual operator R = (I - P_A) @ J =====
        I_minus_P = jnp.eye(layer_out_dim) - P_A
        R = I_minus_P @ J  # (layer_out_dim, weight_dim)

        print(f"    Residual operator R shape: {R.shape}")

        # ===== Step 4: SVD of R to get theorem-aligned directions =====
        # R = U @ S @ Vh, where Vh contains right singular vectors
        U, S, Vh = jnp.linalg.svd(R, full_matrices=False)

        # Right singular vectors (in parameter space) - these are our directions
        V = Vh.T  # (weight_dim, min(layer_out_dim, weight_dim))

        singular_values = S
        print(
            f"    Singular values: min={singular_values.min():.6f}, max={singular_values.max():.6f}"
        )
        print(f"    Available directions: {V.shape[1]}")

        # ===== Step 5: Define three direction families =====
        available_dirs = V.shape[1]
        dirs_needed = min(
            num_dirs, available_dirs // 3
        )  # Ensure we have enough for all families

        if dirs_needed < num_dirs:
            print(
                f"    Warning: Only {available_dirs} directions available, using {dirs_needed} per family"
            )
            num_dirs = dirs_needed

        # Safe directions: smallest singular values (right singular vectors)
        low_indices = jnp.argsort(singular_values)[:num_dirs]
        low_directions = V[:, low_indices]  # (weight_dim, num_dirs)
        low_singular_values = singular_values[low_indices]

        # Constrained directions: largest singular values (right singular vectors)
        high_indices = jnp.argsort(singular_values)[-num_dirs:]
        high_directions = V[:, high_indices]  # (weight_dim, num_dirs)
        high_singular_values = singular_values[high_indices]

        # Random directions: random unit vectors in parameter space
        key = jax.random.PRNGKey(self.seed + layer_idx + 200)
        random_dirs_flat = jax.random.normal(key, (weight_dim, num_dirs))
        # Orthogonalize to ensure independence
        random_dirs_flat, _ = jnp.linalg.qr(random_dirs_flat)
        random_directions = random_dirs_flat[:, :num_dirs]

        # Ensure all directions are unit norm
        low_directions = low_directions / (
            jnp.linalg.norm(low_directions, axis=0, keepdims=True) + 1e-8
        )
        high_directions = high_directions / (
            jnp.linalg.norm(high_directions, axis=0, keepdims=True) + 1e-8
        )
        random_directions = random_directions / (
            jnp.linalg.norm(random_directions, axis=0, keepdims=True) + 1e-8
        )

        print(f"    Low singular values: {low_singular_values}")
        print(f"    High singular values: {high_singular_values}")
        print(f"    Direction families created: {num_dirs} each")

        return {
            "singular_values": singular_values,
            "low_directions": low_directions,  # (weight_dim, num_dirs)
            "high_directions": high_directions,  # (weight_dim, num_dirs)
            "random_directions": random_directions,  # (weight_dim, num_dirs)
            "low_singular_values": low_singular_values,
            "high_singular_values": high_singular_values,
        }

    def perturb_and_correct_trial(
        self,
        model: TransitionModel,
        direction: jnp.ndarray,
        scale: float,
        layer_idx: int,
        x_cal: jnp.ndarray,
        y_cal: jnp.ndarray,
        x_held_out: jnp.ndarray,
        y_held_out: jnp.ndarray,
    ) -> dict:
        """Run a single perturb-and-correct trial and return metrics.

        Args:
            model: Original unperturbed model (TransitionModel)
            direction: Perturbation direction (unit norm, shape = weight matrix shape)
            scale: Scale factor for perturbation
            layer_idx: Which layer to perturb (0 or 1)
            x_cal, y_cal: Calibration data
            x_held_out, y_held_out: Held-out evaluation data

        Returns:
            dict with keys: movement_l2, rmse_before, rmse_after, nll_before, nll_after,
                           calibration_residual
        """
        # Copy model for perturbation: create new instance and copy state
        state_copy = nnx.state(model)
        model_pert = TransitionModel(
            model.l1.in_features,
            model.l3.out_features,
            nnx.Rngs(params=self.seed + self.trial_id_counter + 10000),
            activation=self.act_fn,
        )
        nnx.update(model_pert, state_copy)

        # Get weight matrix and perturb it
        if layer_idx == 0:
            W_orig = model_pert.l2.kernel
            W_shape = W_orig.shape  # Should be (64, 64)
        elif layer_idx == 1:
            W_orig = model_pert.l3.kernel
            W_shape = W_orig.shape  # Should be (64, out_features)
        else:
            raise ValueError(f"Invalid layer_idx: {layer_idx}")

        # Reshape direction to match weight matrix shape
        direction_reshaped = direction.reshape(W_shape)

        # Measure typical weight magnitude for scale interpretation
        typical_magnitude = jnp.linalg.norm(W_orig)

        # Apply perturbation
        perturbation = scale * typical_magnitude * direction_reshaped
        W_pert = W_orig + perturbation

        # Update the model
        if layer_idx == 0:
            model_pert.l2.kernel = W_pert
        elif layer_idx == 1:
            model_pert.l3.kernel = W_pert

        # === Evaluate on held-out data before correction ===
        pred_orig = model(x_held_out)
        pred_pert_uncorr = model_pert(x_held_out)

        movement_l2 = float(jnp.linalg.norm(pred_pert_uncorr - pred_orig, ord=2))
        rmse_uncorr = float(jnp.sqrt(jnp.mean((pred_pert_uncorr - y_held_out) ** 2)))
        rmse_orig = float(jnp.sqrt(jnp.mean((pred_orig - y_held_out) ** 2)))
        nll_uncorr = float(jnp.mean((pred_pert_uncorr - y_held_out) ** 2))

        # === Least-squares correction on calibration data ===
        # Simple approach: fit global scale and bias to minimize prediction error on calibration
        # pred_corrected = scale * pred_perturbed + bias
        pred_cal_pert = model_pert(x_cal)  # (n_cal, out_features)
        y_cal_flat = y_cal.reshape(-1)  # Flatten
        pred_cal_flat = pred_cal_pert.reshape(-1)

        # Fit scale and bias globally
        X_aug = jnp.column_stack([pred_cal_flat, jnp.ones(len(pred_cal_flat))])
        coeffs, _, _, _ = jnp.linalg.lstsq(X_aug, y_cal_flat, rcond=None)
        scale = coeffs[0]
        bias = coeffs[1]

        # Compute calibration residual
        pred_cal_corrected_flat = scale * pred_cal_flat + bias
        pred_cal_corrected = pred_cal_corrected_flat.reshape(y_cal.shape)
        calibration_residual = float(jnp.linalg.norm(pred_cal_corrected - y_cal, ord=2))

        # === Apply correction to held-out data ===
        pred_pert_corr_flat = scale * pred_pert_uncorr.reshape(-1) + bias
        pred_pert_corr = pred_pert_corr_flat.reshape(y_held_out.shape)
        rmse_corr = float(jnp.sqrt(jnp.mean((pred_pert_corr - y_held_out) ** 2)))
        nll_corr = float(jnp.mean((pred_pert_corr - y_held_out) ** 2))

        return {
            "movement_l2": movement_l2,
            "rmse_before": rmse_uncorr,
            "rmse_after": rmse_corr,
            "rmse_orig": rmse_orig,
            "nll_before": nll_uncorr,
            "nll_after": nll_corr,
            "calibration_residual": calibration_residual,
        }

    def run_all_trials(self):
        """Run the complete mechanism experiment."""
        print(f"\n{'=' * 70}")
        print(f"MECHANISM EXPERIMENT: {self.env}")
        print(f"{'=' * 70}\n")

        # Load data
        inputs_all, targets_all = self.load_or_collect_data()
        print(f"Loaded data: {inputs_all.shape} inputs, {targets_all.shape} targets\n")

        # Split data
        data_roles = self.split_data_roles(inputs_all, targets_all)

        # Train baseline model
        model = self.train_dynamics_model(
            data_roles["x_train"],
            data_roles["y_train"],
            data_roles["x_cal"],
            data_roles["y_cal"],
        )

        # Generate perturbation scales (linearly spaced in the range)
        scales = jnp.linspace(self.scale_range[0], self.scale_range[1], self.num_scales)
        print(f"\nPerturbation scales: {scales}\n")

        # Run trials for each layer
        for layer_idx in [0, 1]:
            print(f"\n{'=' * 70}")
            print(f"Layer {layer_idx}")
            print(f"{'=' * 70}")

            # Build operator
            operator = self.build_pjsvd_operator(
                model, data_roles["x_cal"], data_roles["y_cal"], layer_idx
            )
            singular_values = operator["singular_values"]

            # Run trials for each direction family
            for family_name, directions in [
                ("low_singular", operator["low_directions"]),
                ("random", operator["random_directions"]),
                ("high_singular", operator["high_directions"]),
            ]:
                print(f"\n  {family_name.upper()}")

                # Get appropriate singular values for this family
                if family_name == "low_singular":
                    family_singular_values = operator["low_singular_values"]
                elif family_name == "high_singular":
                    family_singular_values = operator["high_singular_values"]
                else:  # random
                    family_singular_values = jnp.zeros(
                        directions.shape[1]
                    )  # No meaningful singular values for random

                for dir_idx in range(directions.shape[1]):
                    direction = directions[:, dir_idx]

                    for scale_idx, scale in enumerate(scales):
                        metrics = self.perturb_and_correct_trial(
                            model,
                            direction,
                            float(scale),
                            layer_idx,
                            data_roles["x_cal"],
                            data_roles["y_cal"],
                            data_roles["x_held_out"],
                            data_roles["y_held_out"],
                        )

                        result = MechanismTrialResult(
                            trial_id=self.trial_id_counter,
                            environment=self.env,
                            layer_idx=layer_idx,
                            direction_family=family_name,
                            singular_value_rank=dir_idx,
                            singular_value=float(family_singular_values[dir_idx]),
                            perturbation_scale=float(scale),
                            metrics=metrics,
                        )
                        self.trials.append(result)
                        self.trial_id_counter += 1

                n_trials = directions.shape[1] * self.num_scales
                print(f"    Completed {n_trials} trials")

        print(f"\n{'=' * 70}")
        print(f"Total trials completed: {len(self.trials)}")
        print(f"{'=' * 70}\n")
        return self.trials

    def save_results(self, output_dir: Path = Path("results/mechanism")):
        """Save trial results to JSON and CSV."""
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save all trials as JSON lines
        json_path = output_dir / f"{self.env}_seed{self.seed}_trials.jsonl"
        with open(json_path, "w") as f:
            for trial in self.trials:
                trial_dict = asdict(trial)
                f.write(json.dumps(trial_dict) + "\n")

        print(f"Saved {len(self.trials)} trials to {json_path}")

        # Save summary CSV
        try:
            import pandas as pd

            df_list = []
            for trial in self.trials:
                row = {
                    "trial_id": trial.trial_id,
                    "environment": trial.environment,
                    "layer": trial.layer_idx,
                    "family": trial.direction_family,
                    "dir_rank": trial.singular_value_rank,
                    "singular_value": trial.singular_value,
                    "pert_scale": trial.perturbation_scale,
                    **trial.metrics,
                }
                df_list.append(row)

            df = pd.DataFrame(df_list)
            csv_path = output_dir / f"{self.env}_seed{self.seed}_summary.csv"
            df.to_csv(csv_path, index=False)
            print(f"Saved summary CSV to {csv_path}")
        except ImportError:
            print("pandas not available; skipping CSV output")

        # Save summary statistics
        summary_stats = {
            "total_trials": len(self.trials),
            "environment": self.env,
            "seed": self.seed,
            "num_directions_per_family": self.num_directions_per_family,
            "num_scales": self.num_scales,
        }
        stats_path = output_dir / f"{self.env}_seed{self.seed}_stats.json"
        with open(stats_path, "w") as f:
            json.dump(summary_stats, f, indent=2)
        print(f"Saved stats to {stats_path}\n")


if __name__ == "__main__":
    # Example usage
    exp = MechanismExperiment(
        env="HalfCheetah-v5",
        seed=0,
        num_directions_per_family=20,
        num_scales=4,
    )
    trials = exp.run_all_trials()
    exp.save_results()
