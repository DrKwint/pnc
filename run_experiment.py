import os
import sys
import argparse
import time

# Parse --cpu early, BEFORE importing jax, so JAX_PLATFORMS is set before JAX initializes any GPU
if '--cpu' in sys.argv:
    os.environ['JAX_PLATFORMS'] = 'cpu'
    os.environ['CUDA_VISIBLE_DEVICES'] = ''   # Belt-and-suspenders: hide GPU from CUDA too
    sys.argv.remove('--cpu')
else:
    os.environ['JAX_PLATFORMS'] = 'cpu,cuda'

import jax
import jax.numpy as jnp
from flax import nnx
import numpy as np

from pjsvd import find_optimal_perturbation, find_optimal_perturbation_multi_layer, apply_correction
from models import TransitionModel, MCDropoutTransitionModel, ClassificationModel, MCDropoutClassificationModel
from laplace import compute_kfac_factors
from data import collect_data, id_policy_random, ood_policy_run, OODPolicyWrapper, load_mnist
from training import train_model, train_swag_model, train_classification_model, train_swag_classification_model
from ensembles import Ensemble, MultiLayerPJSVDEnsemble, StandardEnsemble, MCDropoutEnsemble, SWAGEnsemble, LaplaceEnsemble
from metrics import compute_nll, compute_calibration, print_metrics, compute_ood_metrics

def train_standard_ensemble(inputs: jax.Array, targets: jax.Array, n_models: int = 5, steps: int = 2000) -> StandardEnsemble:
    print(f"\nTraining Standard Ensemble of {n_models} models...")
    models = []
    
    for i in range(n_models):
        rngs = nnx.Rngs(i) # Different seed for initialization
        model = TransitionModel(inputs.shape[1], targets.shape[1], rngs)
        
        # Train
        print(f"  Training Model {i+1}/{n_models}...")
        model = train_model(model, inputs, targets, steps=steps, batch_size=64)
        models.append(model)
        
    return StandardEnsemble(models)

class PJSVDExperiment:
    def __init__(self, env_name: str, steps: int, subset_size: int):
        self.env_name = env_name
        self.steps = steps
        self.subset_size = subset_size
        self.model = None
        
        self.inputs_id = None
        self.targets_id = None
        self.inputs_id_eval = None
        self.targets_id_eval = None
        self.inputs_ood = None
        self.targets_ood = None
        
        self.ensembles = {}
        self.baseline_ensemble = None

    def setup(self):
        # Data Collection
        self.inputs_id, self.targets_id = collect_data(self.env_name, self.steps, OODPolicyWrapper(), seed=0)
        self.inputs_id_eval, self.targets_id_eval = collect_data(self.env_name, self.steps, OODPolicyWrapper(), seed=42)
        self.inputs_ood, self.targets_ood = collect_data(self.env_name, self.steps, id_policy_random, seed=99)
        
        # Base Model Training
        rngs = nnx.Rngs(0)
        self.model = TransitionModel(self.inputs_id.shape[1], self.targets_id.shape[1], rngs)
        t0 = time.time()
        self.model = train_model(self.model, self.inputs_id, self.targets_id)
        self.base_model_train_time = time.time() - t0
        print(f"Base Model Training Time: {self.base_model_train_time:.4f}s")
    
    def run_pjsvd(self, n_directions: int, n_perturbations: int, perturbation_sizes: list[float]):
        print("\n--- Running PJSVD ---")
        
        actual_subset_size = min(len(self.inputs_id), self.subset_size)
        subset_idx = np.random.choice(len(self.inputs_id), actual_subset_size, replace=False)
        X_subset = self.inputs_id[subset_idx]
        
        if self.model is None:
            raise ValueError("Model must be trained before running PJSVD.")
            
        W1_curr = self.model.l1.kernel.get_value()
        b1_curr = self.model.l1.bias.get_value()
        
        def model_fn_l1(w):
            return nnx.relu(X_subset @ w + b1_curr)
        
        print(f"Finding {n_directions} Orthogonal Null Space Directions...")
        sigmas = []
        v_opts = jnp.zeros((n_directions, *W1_curr.shape))
        
        for k in range(n_directions):
            v_opt, sigma = find_optimal_perturbation(
                model_fn_l1, 
                W1_curr, 
                max_iter=500, 
                orthogonal_directions=v_opts,
            )
            v_opts = v_opts.at[k].set(v_opt)
            sigmas.append(sigma)
            print(f"  Direction {k+1}: Residual Sigma = {sigma:.6f}")
            
        h_old = model_fn_l1(W1_curr)
        mu_old = jnp.mean(h_old, axis=0)
        std_old = jnp.std(h_old, axis=0)
        
        W2_curr = self.model.l2.kernel.get_value()
        b2_curr = self.model.l2.bias.get_value()

        self.ensembles = {}
        all_z = np.random.normal(0, 1, size=(n_perturbations, n_directions))

        for p_size in perturbation_sizes:
            print(f"Generating {n_perturbations} ensemble members for size {p_size}...")
            perturbations = []
            
            for i in range(n_perturbations):
                z = all_z[i]
                safe_sigmas = jnp.array(sigmas) + 1e-6
                coeffs = z / safe_sigmas
                coeffs = coeffs / np.linalg.norm(coeffs) * p_size
                
                weighted_vs = jnp.reshape(coeffs, (-1, 1, 1)) * v_opts
                total_perturbation = jnp.sum(weighted_vs, axis=0)
                
                W1_new = W1_curr + total_perturbation
                h_new = model_fn_l1(W1_new)
                
                W2_new, b2_new = apply_correction((W2_curr, b2_curr), (mu_old, std_old), h_new)
                perturbations.append((W1_new, b1_curr, W2_new, b2_new))
                
            self.ensembles[p_size] = Ensemble(self.model, perturbations)
            
        return self.ensembles

    def run_pjsvd_multi_layer(self, n_directions: int, n_perturbations: int, perturbation_sizes: list[float]):
        print("\n--- Running Multi-Layer PJSVD ---")
        
        actual_subset_size = min(len(self.inputs_id), self.subset_size)
        subset_idx = np.random.choice(len(self.inputs_id), actual_subset_size, replace=False)
        X_subset = self.inputs_id[subset_idx]
        
        if self.model is None:
            raise ValueError("Model must be trained before running PJSVD.")
            
        W1_curr = self.model.l1.kernel.get_value()
        b1_curr = self.model.l1.bias.get_value()
        W2_curr = self.model.l2.kernel.get_value()
        b2_curr = self.model.l2.bias.get_value()
        
        def model_fn_layers(ws):
            w1, w2 = ws
            h1 = nnx.relu(X_subset @ w1 + b1_curr)
            h2 = nnx.relu(h1 @ w2 + b2_curr)
            return h2
        
        print(f"Finding {n_directions} Orthogonal Null Space Directions...")
        sigmas = []
        
        target_params = [W1_curr, W2_curr]
        total_size = W1_curr.size + W2_curr.size
        v_opts = jnp.zeros((n_directions, total_size))
        
        # We need to reshape v_opts list back to flat for the next iteration orthogonalization constraint
        for k in range(n_directions):
            v_opts_list, sigma = find_optimal_perturbation_multi_layer(
                model_fn_layers, 
                target_params, 
                max_iter=500, 
                orthogonal_directions=v_opts,
            )
            v_opt_flat = jnp.concatenate([v.flatten() for v in v_opts_list])
            v_opts = v_opts.at[k].set(v_opt_flat)
            sigmas.append(sigma)
            print(f"  Direction {k+1}: Residual Sigma = {sigma:.6f}")
            
        h_old = model_fn_layers([W1_curr, W2_curr])
        mu_old = jnp.mean(h_old, axis=0)
        std_old = jnp.std(h_old, axis=0)
        
        W3_curr = self.model.l3.kernel.get_value()
        b3_curr = self.model.l3.bias.get_value()

        self.multi_layer_ensembles = {}
        all_z = np.random.normal(0, 1, size=(n_perturbations, n_directions))

        for p_size in perturbation_sizes:
            print(f"Generating {n_perturbations} ensemble members for size {p_size}...")
            perturbations = []
            
            for i in range(n_perturbations):
                z = all_z[i]
                safe_sigmas = jnp.array(sigmas) + 1e-6
                coeffs = z / (safe_sigmas ** 2)
                coeffs = coeffs / np.linalg.norm(coeffs) * p_size
                
                weighted_vs = jnp.reshape(coeffs, (-1, 1)) * v_opts
                total_perturbation = jnp.sum(weighted_vs, axis=0)
                
                v_w1 = total_perturbation[:W1_curr.size].reshape(W1_curr.shape)
                v_w2 = total_perturbation[W1_curr.size:].reshape(W2_curr.shape)
                
                W1_new = W1_curr + v_w1
                W2_new = W2_curr + v_w2
                
                h_new = model_fn_layers([W1_new, W2_new])
                
                W3_new, b3_new = apply_correction((W3_curr, b3_curr), (mu_old, std_old), h_new)
                perturbations.append((W1_new, b1_curr, W2_new, b2_curr, W3_new, b3_new))
                
            self.multi_layer_ensembles[p_size] = MultiLayerPJSVDEnsemble(self.model, perturbations)
            
        return self.multi_layer_ensembles

    def run_baseline(self, n_models: int, steps: int):
        print(f"\n--- Running Standard Ensemble Baseline ({n_models} models) ---")
        self.baseline_ensemble = train_standard_ensemble(self.inputs_id, self.targets_id, n_models, steps)
        return self.baseline_ensemble

    def run_mc_dropout(self, n_samples: int, steps: int):
        print(f"\n--- Running MC Dropout Baseline ({n_samples} samples) ---")
        rngs = nnx.Rngs(123)
        model = MCDropoutTransitionModel(self.inputs_id.shape[1], self.targets_id.shape[1], rngs, dropout_rate=0.1)
        model = train_model(model, self.inputs_id, self.targets_id, steps=steps, batch_size=64)
        self.mc_dropout_ensemble = MCDropoutEnsemble(model, n_samples)
        return self.mc_dropout_ensemble

    def run_swag(self, n_samples: int, steps: int):
        print(f"\n--- Running SWAG Baseline ({n_samples} samples) ---")
        rngs = nnx.Rngs(456)
        model_swag = TransitionModel(self.inputs_id.shape[1], self.targets_id.shape[1], rngs)
        model_swag, swag_mean, swag_var = train_swag_model(
            model_swag, self.inputs_id, self.targets_id, 
            steps=steps, batch_size=64, swag_start=steps // 2
        )
        self.swag_ensemble = SWAGEnsemble(model_swag, swag_mean, swag_var, n_samples)
        return self.swag_ensemble

    def run_laplace(self, n_samples: int, steps: int, prior_precisions: list[float]):
        print(f"\n--- Running Laplace Approximation Baseline ({n_samples} samples) ---")
        rngs = nnx.Rngs(789)
        model_laplace = TransitionModel(self.inputs_id.shape[1], self.targets_id.shape[1], rngs)
        model_laplace = train_model(model_laplace, self.inputs_id, self.targets_id, steps=steps, batch_size=64)
        
        print("Computing KFAC Factors for Laplace Approximation...")
        # Use a subset similar to PJSVD to be fair, or the full ID set. We'll use the subset.
        actual_subset_size = min(len(self.inputs_id), self.subset_size)
        subset_idx = np.random.choice(len(self.inputs_id), actual_subset_size, replace=False)
        X_subset = self.inputs_id[subset_idx]
        Y_subset = self.targets_id[subset_idx]
        
        factors = compute_kfac_factors(model_laplace, X_subset, Y_subset, batch_size=128)
        
        ensembles = {}
        for prior_precision in prior_precisions:
            print(f"\nGenerating Laplace Ensemble with prior_precision={prior_precision}...")
            laplace_ensemble = LaplaceEnsemble(
                model=model_laplace, 
                kfac_factors=factors, 
                prior_precision=prior_precision, 
                n_models=n_samples, 
                data_size=actual_subset_size
            )
            
            # Measure and print average perturbation scale directly
            total_scale = 0.0
            for _ in range(50): # Average over 50 samples
                _, norm = laplace_ensemble._sample_model()
                total_scale += norm
            avg_scale = total_scale / 50.0
            print(f"-> Average Laplace Perturbation Norm ||dw||: {avg_scale:.4f}")
            
            # Reset model to MAP just in case
            w1, b1 = laplace_ensemble.map_params['l1']
            w2, b2 = laplace_ensemble.map_params['l2']
            w3, b3 = laplace_ensemble.map_params['l3']
            nnx.update(model_laplace.l1.kernel, w1)
            nnx.update(model_laplace.l1.bias, b1)
            nnx.update(model_laplace.l2.kernel, w2)
            nnx.update(model_laplace.l2.bias, b2)
            nnx.update(model_laplace.l3.kernel, w3)
            nnx.update(model_laplace.l3.bias, b3)
            
            ensembles[prior_precision] = laplace_ensemble
        
        self.laplace_ensembles = ensembles
        return self.laplace_ensembles

    def evaluate(self, ensemble_name: str, ensemble):
        print(f"\n--- Results: {ensemble_name} ---")
        
        def compute_group_metrics(name, inputs, targets):
            preds = ensemble.predict(inputs)
            mean = jnp.mean(preds, axis=0)
            var = jnp.var(preds, axis=0)
            
            avg_var = float(jnp.mean(var))
            nll = float(compute_nll(mean, var, targets))
            cal_err = float(compute_calibration(mean, var, targets))
            rmse = float(jnp.sqrt(jnp.mean((mean - targets)**2)))
            
            print_metrics(name, rmse, avg_var, nll, cal_err)
            return rmse, avg_var, nll

        rmse_id, var_id, _ = compute_group_metrics("ID", self.inputs_id_eval, self.targets_id_eval)
        rmse_ood, var_ood, _ = compute_group_metrics("OOD", self.inputs_ood, self.targets_ood)
        
        var_id_scores = np.array(jnp.mean(ensemble.predict(self.inputs_id_eval).var(axis=0), axis=1))
        var_ood_scores = np.array(jnp.mean(ensemble.predict(self.inputs_ood).var(axis=0), axis=1))
        
        auroc, aupr = compute_ood_metrics(var_id_scores, var_ood_scores)
        
        print(f"\nOOD Detection Metrics:")
        print(f"AUROC: {auroc:.4f} | AUPR: {aupr:.4f}")

        rmse_ratio = rmse_ood / (rmse_id + 1e-6)
        print(f"RMSE Ratio (OOD / ID): {rmse_ratio:.2f}x")
        
        var_ratio = var_ood / (var_id + 1e-9)
        print(f"Variance Ratio (OOD / ID): {var_ratio:.2f}x")
        
        return auroc

class MNISTExperiment:
    """Classification experiment on MNIST."""
    def __init__(self, n_classes: int = 10):
        self.n_classes = n_classes
        self.x_train, self.y_train, self.x_test, self.y_test = load_mnist()

    def _entropy(self, probs):
        """Predictive entropy: -sum(p * log(p))"""
        return -jnp.sum(probs * jnp.log(probs + 1e-8), axis=-1)

    def _evaluate_classification(self, name: str, ensemble):
        print(f"\n--- Results: {name} ---")

        preds = jax.nn.softmax(ensemble.predict(self.x_test), axis=-1)  # (S, N, C)
        mean  = jnp.mean(preds, axis=0)                                 # (N, C)

        acc    = float(jnp.mean(jnp.argmax(mean, axis=-1) == self.y_test))
        y_oh   = jax.nn.one_hot(self.y_test, self.n_classes)
        brier  = float(jnp.mean(jnp.sum((mean - y_oh) ** 2, axis=-1)))
        ent    = float(jnp.mean(-jnp.sum(mean * jnp.log(mean + 1e-8), axis=-1)))

        print(f"Acc: {acc:.4f} | Brier: {brier:.4f} | Entropy: {ent:.4f}")

    def run_pjsvd(self, model, n_directions: int, n_perturbations: int,
                  perturbation_sizes: list[float], subset_size: int = 4096):
        print(f"\n=== MNIST EXP 0: PJSVD ({n_perturbations} samples, sizes={perturbation_sizes}) ===")

        actual_subset = min(len(self.x_train), subset_size)
        idx = np.random.choice(len(self.x_train), actual_subset, replace=False)
        X_sub = self.x_train[idx]

        W1 = model.l1.kernel.get_value()
        b1 = model.l1.bias.get_value()

        def model_fn_l1(w):
            return nnx.relu(X_sub @ w + b1)

        print(f"Finding {n_directions} orthogonal null-space directions...")
        sigmas = []
        v_opts = jnp.zeros((n_directions, *W1.shape))
        for k in range(n_directions):
            v_opt, sigma = find_optimal_perturbation(
                model_fn_l1, W1, max_iter=500, orthogonal_directions=v_opts
            )
            v_opts = v_opts.at[k].set(v_opt)
            sigmas.append(sigma)
            print(f"  Direction {k+1}: sigma={sigma:.6f}")

        h_old  = model_fn_l1(W1)
        mu_old = jnp.mean(h_old, axis=0)
        std_old = jnp.std(h_old, axis=0)

        W2 = model.l2.kernel.get_value()
        b2 = model.l2.bias.get_value()

        all_z = np.random.normal(0, 1, size=(n_perturbations, n_directions))
        ensembles = {}
        for p_size in perturbation_sizes:
            print(f"Generating {n_perturbations} perturbations for size={p_size}...")
            perturbations = []
            for i in range(n_perturbations):
                z = all_z[i]
                safe_sigmas = jnp.array(sigmas) + 1e-6
                coeffs = z / safe_sigmas
                coeffs = coeffs / np.linalg.norm(coeffs) * p_size
                dW = jnp.sum(jnp.reshape(coeffs, (-1, 1, 1)) * v_opts, axis=0)
                W1_new = W1 + dW
                h_new  = model_fn_l1(W1_new)
                W2_new, b2_new = apply_correction((W2, b2), (mu_old, std_old), h_new)
                perturbations.append((W1_new, b1, W2_new, b2_new))
            ensembles[p_size] = Ensemble(model, perturbations)
        return ensembles

    def run_all(self, n_samples: int = 100, steps: int = 5000, n_baseline: int = 5,
                laplace_priors: list[float] = [1.0, 10.0, 100.0],
                n_directions: int = 40, perturbation_sizes: list[float] = [20.0, 40.0, 60.0, 80.0, 100.0]):
        in_dim, n_cls = self.x_train.shape[1], self.n_classes

        # Train a base model for PJSVD and other methods
        print(f"\n=== MNIST EXP Base Model Training ===")
        t0 = time.time()
        base_model = ClassificationModel(in_dim, n_cls, nnx.Rngs(0)) # Use a fixed rng for reproducibility
        base_model = train_classification_model(base_model, self.x_train, self.y_train, steps=steps)
        print(f"Base Model Training Time: {time.time() - t0:.4f}s")

        # --- 0. PJSVD ---
        t0 = time.time()
        self.pjsvd_ensembles = self.run_pjsvd(
            base_model,
            n_directions=n_directions,
            n_perturbations=n_samples,
            perturbation_sizes=perturbation_sizes
        )
        print(f"PJSVD Null-Space & Generation Time: {time.time() - t0:.4f}s")
        for p_size, ens in self.pjsvd_ensembles.items():
            t0 = time.time()
            self._evaluate_classification(f"PJSVD (size={p_size})", ens)
            print(f"PJSVD (size={p_size}) Evaluation Time: {time.time() - t0:.4f}s")

        # --- 1. Standard Deep Ensemble ---
        print(f"\n=== MNIST EXP 1: Standard Ensemble ({n_baseline} models) ===")
        t0 = time.time()
        std_models = []
        for i in range(n_baseline):
            m = ClassificationModel(in_dim, n_cls, nnx.Rngs(i))
            m = train_classification_model(m, self.x_train, self.y_train, steps=steps)
            std_models.append(m)
        std_ensemble = StandardEnsemble(std_models)
        print(f"Standard Ensemble Training Time: {time.time() - t0:.4f}s")
        t0 = time.time()
        self._evaluate_classification("Standard Ensemble", std_ensemble)
        print(f"Standard Ensemble Evaluation Time: {time.time() - t0:.4f}s")

        # --- 2. MC Dropout ---
        print(f"\n=== MNIST EXP 2: MC Dropout ({n_samples} samples) ===")
        t0 = time.time()
        mc_model = MCDropoutClassificationModel(in_dim, n_cls, nnx.Rngs(42))
        mc_model = train_classification_model(mc_model, self.x_train, self.y_train, steps=steps)
        mc_ensemble = MCDropoutEnsemble(mc_model, n_samples)
        print(f"MC Dropout Training Time: {time.time() - t0:.4f}s")
        t0 = time.time()
        self._evaluate_classification("MC Dropout", mc_ensemble)
        print(f"MC Dropout Evaluation Time: {time.time() - t0:.4f}s")

        # --- 3. SWAG ---
        print(f"\n=== MNIST EXP 3: SWAG ({n_samples} samples) ===")
        t0 = time.time()
        swag_model = ClassificationModel(in_dim, n_cls, nnx.Rngs(99))
        swag_model, swag_mean, swag_var = train_swag_classification_model(
            swag_model, self.x_train, self.y_train, steps=steps, swag_start=steps // 2
        )
        swag_ensemble = SWAGEnsemble(swag_model, swag_mean, swag_var, n_samples)
        print(f"SWAG Training Time: {time.time() - t0:.4f}s")
        t0 = time.time()
        self._evaluate_classification("SWAG", swag_ensemble)
        print(f"SWAG Evaluation Time: {time.time() - t0:.4f}s")

        # --- 4. Laplace ---
        print(f"\n=== MNIST EXP 4: Laplace Approximation ({n_samples} samples) ===")
        t0 = time.time()
        lap_model = ClassificationModel(in_dim, n_cls, nnx.Rngs(7))
        lap_model = train_classification_model(lap_model, self.x_train, self.y_train, steps=steps)
        print(f"Laplace Base Training Time: {time.time() - t0:.4f}s")

        # Use a subset for KFAC
        subset_idx = np.random.choice(len(self.x_train), min(4096, len(self.x_train)), replace=False)
        print("Computing KFAC Factors...")
        t0 = time.time()
        factors = compute_kfac_factors(
            lap_model, self.x_train[subset_idx], self.y_train[subset_idx],
            batch_size=128, is_classification=True
        )
        print(f"Laplace KFAC Time: {time.time() - t0:.4f}s")

        for prior in laplace_priors:
            t0 = time.time()
            lap_ensemble = LaplaceEnsemble(
                model=lap_model,
                kfac_factors=factors,
                prior_precision=prior,
                n_models=n_samples,
                data_size=len(subset_idx)
            )
            print(f"Laplace Ensemble Generation Time (prior={prior}): {time.time() - t0:.4f}s")
            t0 = time.time()
            self._evaluate_classification(f"Laplace (prior={prior})", lap_ensemble)
            print(f"Laplace Evaluation Time (prior={prior}): {time.time() - t0:.4f}s")


def main():
    parser = argparse.ArgumentParser(description="Run PJSVD Expeirment on Gym Environment")
    parser.add_argument("--env", type=str, default="HalfCheetah-v5", help="Gym environment name")
    parser.add_argument("--steps", type=int, default=10000, help="Number of interactions collected per policy")
    parser.add_argument("--subset_size", type=int, default=4096, help="Data subset size for calculating PJSVD null spaces")
    parser.add_argument("--n_directions", type=int, default=40, help="Number of singular directions (K)")
    parser.add_argument("--n_perturbations", type=int, default=1000, help="Number of models to generate per ensemble")
    parser.add_argument("--n_baseline", type=int, default=5, help="Number of deep ensembles for the baseline")
    parser.add_argument("--perturbation_sizes", nargs="+", type=float, default=[20.0, 40.0, 80.0, 160.0], help="List of perturbation norms")
    parser.add_argument("--laplace_priors", nargs="+", type=float, default=[1.0, 5.0, 10.0, 50.0, 100.0], help="List of prior precisions for Laplace Approximation")
    args = parser.parse_args()

    if args.env.upper() == 'MNIST':
        print("=== MNIST Classification Experiment (Deep Ensembles paper) ===")
        mnist_exp = MNISTExperiment()
        mnist_exp.run_all(
            n_samples=args.n_perturbations,
            steps=args.steps,
            n_baseline=args.n_baseline,
            laplace_priors=args.laplace_priors
        )
        return

    print(f"=== EXPERIMENT 1: PJSVD ({args.env}) ===")
    experiment = PJSVDExperiment(env_name=args.env, steps=args.steps, subset_size=args.subset_size)
    
    t0 = time.time()
    experiment.setup()
    print(f"Setup Time (including Base Model Training): {time.time() - t0:.4f}s")
    
    t0 = time.time()
    ensembles = experiment.run_pjsvd(
        n_directions=args.n_directions, 
        n_perturbations=args.n_perturbations, 
        perturbation_sizes=args.perturbation_sizes
    )
    print(f"PJSVD Generation Time: {time.time() - t0:.4f}s")
    
    for p_size, ens in ensembles.items():
        t0 = time.time()
        experiment.evaluate(ensemble_name=f"PJSVD (size={p_size})", ensemble=ens)
        print(f"PJSVD (size={p_size}) Evaluation Time: {time.time() - t0:.4f}s")

    print(f"\n\n=== EXPERIMENT 1.5: Multi-Layer PJSVD ({args.env}) ===")
    t0 = time.time()
    multi_layer_ensembles = experiment.run_pjsvd_multi_layer(
        n_directions=args.n_directions, 
        n_perturbations=args.n_perturbations, 
        perturbation_sizes=[ps / 4 for ps in args.perturbation_sizes]
    )
    print(f"Multi-Layer PJSVD Generation Time: {time.time() - t0:.4f}s")
    
    for p_size, ens in multi_layer_ensembles.items():
        t0 = time.time()
        experiment.evaluate(ensemble_name=f"Multi-Layer PJSVD (size={p_size})", ensemble=ens)
        print(f"Multi-Layer PJSVD (size={p_size}) Evaluation Time: {time.time() - t0:.4f}s")

    print("\n\n=== EXPERIMENT 2: Standard Baseline (Deep Ensemble) ===")
    t0 = time.time()
    experiment.run_baseline(n_models=args.n_baseline, steps=2000)
    print(f"Standard Baseline Training Time: {time.time() - t0:.4f}s")
    t0 = time.time()
    experiment.evaluate(ensemble_name="Standard Baseline", ensemble=experiment.baseline_ensemble)
    print(f"Standard Baseline Evaluation Time: {time.time() - t0:.4f}s")

    print(f"\n\n=== EXPERIMENT 3: MC Dropout Baseline ({args.n_perturbations} samples) ===")
    t0 = time.time()
    experiment.run_mc_dropout(n_samples=args.n_perturbations, steps=2000)
    print(f"MC Dropout Training Time: {time.time() - t0:.4f}s")
    t0 = time.time()
    experiment.evaluate(ensemble_name="MC Dropout", ensemble=experiment.mc_dropout_ensemble)
    print(f"MC Dropout Evaluation Time: {time.time() - t0:.4f}s")

    print(f"\n\n=== EXPERIMENT 4: SWAG Baseline ({args.n_perturbations} samples) ===")
    t0 = time.time()
    experiment.run_swag(n_samples=args.n_perturbations, steps=2000)
    print(f"SWAG Training Time: {time.time() - t0:.4f}s")
    t0 = time.time()
    experiment.evaluate(ensemble_name="SWAG", ensemble=experiment.swag_ensemble)
    print(f"SWAG Evaluation Time: {time.time() - t0:.4f}s")

    print(f"\n\n=== EXPERIMENT 5: Laplace Approximation Baseline ({args.n_perturbations} samples) ===")
    t0 = time.time()
    laplace_ensembles = experiment.run_laplace(n_samples=args.n_perturbations, steps=2000, prior_precisions=args.laplace_priors)
    print(f"Laplace Training & KFAC Time: {time.time() - t0:.4f}s")
    for prior_prec, ens in laplace_ensembles.items():
        t0 = time.time()
        experiment.evaluate(ensemble_name=f"Laplace (prior={prior_prec})", ensemble=ens)
        print(f"Laplace (prior={prior_prec}) Evaluation Time: {time.time() - t0:.4f}s")

if __name__ == "__main__":
    main()