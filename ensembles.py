import jax
import jax.numpy as jnp
from flax import nnx
from typing import List, Tuple
import numpy as np

from models import TransitionModel

class Ensemble:
    def __init__(self, base_model: TransitionModel, perturbations: List[Tuple]):
        self.base_model = base_model
        # List of (w1, b1, w2, b2)
        self.perturbations = perturbations 

    def manual_forward(
        self, 
        x: jax.Array, 
        w1: jax.Array, 
        b1: jax.Array, 
        w2: jax.Array, 
        b2: jax.Array
    ) -> jax.Array:
        """
        Manually performs a forward pass using perturbed weights.
        """
        h1 = nnx.relu(x @ w1 + b1)
        h2 = nnx.relu(h1 @ w2 + b2)
        out = h2 @ self.base_model.l3.kernel.get_value() + self.base_model.l3.bias.get_value()
        return out

    def predict(self, x: jax.Array) -> jax.Array:
        """
        Predicts outputs for all ensemble members.
        Returns array of shape (N_models, Batch, Dim).
        """
        ys = []
        for w1, b1, w2, b2 in self.perturbations:
            y = self.manual_forward(x, w1, b1, w2, b2)
            ys.append(y)
        return jnp.stack(ys, axis=0)
    
    def predict_one(self, x: jax.Array, idx: int) -> jax.Array:
        """Returns the prediction of a single ensemble member."""
        w1, b1, w2, b2 = self.perturbations[idx]
        return self.manual_forward(x, w1, b1, w2, b2)

class MultiLayerPJSVDEnsemble:
    def __init__(self, base_model: TransitionModel, perturbations: List[Tuple]):
        self.base_model = base_model
        # List of (w1, b1, w2, b2, w3, b3)
        self.perturbations = perturbations 

    def manual_forward(
        self, 
        x: jax.Array, 
        w1: jax.Array, 
        b1: jax.Array, 
        w2: jax.Array, 
        b2: jax.Array,
        w3: jax.Array,
        b3: jax.Array
    ) -> jax.Array:
        h1 = nnx.relu(x @ w1 + b1)
        h2 = nnx.relu(h1 @ w2 + b2)
        out = h2 @ w3 + b3
        return out

    def predict(self, x: jax.Array) -> jax.Array:
        ys = []
        for w1, b1, w2, b2, w3, b3 in self.perturbations:
            y = self.manual_forward(x, w1, b1, w2, b2, w3, b3)
            ys.append(y)
        return jnp.stack(ys, axis=0)
    
    def predict_one(self, x: jax.Array, idx: int) -> jax.Array:
        w1, b1, w2, b2, w3, b3 = self.perturbations[idx]
        return self.manual_forward(x, w1, b1, w2, b2, w3, b3)

class StandardEnsemble:
    def __init__(self, models: List[TransitionModel]):
        self.models = models

    def predict(self, x: jax.Array) -> jax.Array:
        """
        Predicts outputs for all ensemble members.
        Returns array of shape (N_models, Batch, Dim).
        """
        ys = []
        for model in self.models:
            ys.append(model(x))
        return jnp.stack(ys, axis=0)
    
    def predict_one(self, x: jax.Array, idx: int) -> jax.Array:
        """Returns the prediction of a single ensemble member."""
        return self.models[idx](x)

class MCDropoutEnsemble:
    def __init__(self, model: TransitionModel, n_models: int):
        self.model = model
        self.n_models = n_models

    def predict(self, x: jax.Array) -> jax.Array:
        """
        Predicts outputs by running multiple forward passes with dropout enabled.
        Returns array of shape (n_models, Batch, Dim).
        """
        ys = []
        for _ in range(self.n_models):
            ys.append(self.model(x, deterministic=False))
        return jnp.stack(ys, axis=0)
    
    def predict_one(self, x: jax.Array, idx: int) -> jax.Array:
        """Returns the prediction of a single forward pass."""
        return self.model(x, deterministic=False)

class SWAGEnsemble:
    def __init__(self, model: TransitionModel, swag_mean: nnx.State, swag_var: nnx.State, n_models: int):
        self.model = model
        self.swag_mean = swag_mean
        self.swag_var = swag_var
        self.n_models = n_models

    def _sample_model(self) -> TransitionModel:
        """Samples a set of weights from the SWAG diagonal posterior and updates the model."""
        sample_params = jax.tree.map(
            lambda m, v: m + jnp.sqrt(v) * np.random.normal(size=m.shape),
            self.swag_mean, self.swag_var
        )
        nnx.update(self.model, sample_params)
        return self.model

    def predict(self, x: jax.Array) -> jax.Array:
        """
        Predicts outputs by sampling models from the SWAG posterior.
        Returns array of shape (n_models, Batch, Dim).
        """
        ys = []
        for _ in range(self.n_models):
            sampled_m = self._sample_model()
            ys.append(sampled_m(x))
        return jnp.stack(ys, axis=0)
    
    def predict_one(self, x: jax.Array, idx: int) -> jax.Array:
        """Returns the prediction of a single sampled model."""
        sampled_m = self._sample_model()
        return sampled_m(x)

class LaplaceEnsemble:
    def __init__(self, model: TransitionModel, kfac_factors: dict, prior_precision: float, n_models: int, data_size: int):
        self.model = model
        self.kfac_factors = kfac_factors
        self.prior_precision = prior_precision
        self.n_models = n_models
        self.data_size = data_size
        
        # Precompute the inverse scale factors for each layer
        self.inv_scales = {}
        for layer_name, (A, S) in self.kfac_factors.items():
            # Add small jitter for numerical stability
            A = A + jnp.eye(A.shape[0]) * 1e-6
            S = S + jnp.eye(S.shape[0]) * 1e-6
            
            # Since A and S are symmetric positive semi-definite, SVD is equivalent to eigendecomposition
            # U_A @ diag(eig_A) @ U_A.T = A
            U_A, eig_A, _ = jnp.linalg.svd(A)
            U_S, eig_S, _ = jnp.linalg.svd(S)
            
            # Ensure eigenvalues are positive
            eig_A = jnp.maximum(eig_A, 0.0)
            eig_S = jnp.maximum(eig_S, 0.0)
            
            self.inv_scales[layer_name] = (U_A, eig_A, U_S, eig_S)
            
        self.map_params = {}
        i = 1
        while hasattr(model, f'l{i}'):
            layer = getattr(model, f'l{i}')
            self.map_params[f'l{i}'] = (layer.kernel.get_value(), layer.bias.get_value())
            i += 1

    def _sample_layer_weights(self, layer_name: str):
        W_map, b_map = self.map_params[layer_name]
        W_full = jnp.concatenate([W_map, jnp.expand_dims(b_map, axis=0)], axis=0) # shape (In+1, Out)
        
        U_A, eig_A, U_S, eig_S = self.inv_scales[layer_name]
        
        # Sample standard normal matrix
        Z = np.random.normal(size=W_full.shape)
        
        # The true variance of vec(W) is (N * A (x) S + lambda I)^-1
        # In the SVD basis, the eigenvalues of the precision matrix are N * (eig_A_i * eig_S_j) + lambda
        # So the std dev we multiply by in the SVD basis is 1 / sqrt(N * eig_A_i * eig_S_j + lambda)
        
        N = self.data_size
        lambda_val = self.prior_precision
        
        # Broadcast eigenvalues to form a matrix of shape (In+1, Out)
        eig_matrix = N * jnp.outer(eig_A, eig_S) + lambda_val
        std_matrix = 1.0 / jnp.sqrt(eig_matrix)
        
        # Scale the standard normal noise
        scaled_Z = Z * std_matrix
        
        # Transform back to original basis
        # W_sampled = U_A @ scaled_Z @ U_S.T
        delta_W = U_A @ scaled_Z @ U_S.T
        
        W_new = W_full + delta_W
        
        # Split back into W and b
        return W_new[:-1, :], W_new[-1, :]

    def _sample_model(self) -> jnp.ndarray:
        w1, b1 = self._sample_layer_weights('l1')
        w2, b2 = self._sample_layer_weights('l2')
        w3, b3 = self._sample_layer_weights('l3')
        
        # Calculate perturbation norm just for tracing out
        # (This calculates norm of difference from MAP)
        norm1 = jnp.linalg.norm((w1 - self.map_params['l1'][0]).flatten())
        norm2 = jnp.linalg.norm((w2 - self.map_params['l2'][0]).flatten())
        norm3 = jnp.linalg.norm((w3 - self.map_params['l3'][0]).flatten())
        total_norm = float(jnp.sqrt(norm1**2 + norm2**2 + norm3**2))
        
        nnx.update(self.model.l1.kernel, w1)
        nnx.update(self.model.l1.bias, b1)
        nnx.update(self.model.l2.kernel, w2)
        nnx.update(self.model.l2.bias, b2)
        nnx.update(self.model.l3.kernel, w3)
        nnx.update(self.model.l3.bias, b3)
        
        return self.model, total_norm

    def predict(self, x: jax.Array) -> jax.Array:
        ys = []
        for _ in range(self.n_models):
            sampled_m, _ = self._sample_model()
            ys.append(sampled_m(x))
        return jnp.stack(ys, axis=0)
    
    def predict_one(self, x: jax.Array, idx: int) -> jax.Array:
        sampled_m, _ = self._sample_model()
        return sampled_m(x)
