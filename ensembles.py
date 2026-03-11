import jax
import jax.numpy as jnp
from flax import nnx
from typing import List, Tuple, Any, Callable
import numpy as np

from models import TransitionModel

def evaluate_tail_from_preact(
    base_model: Any, 
    preact: jax.Array, 
    current_layer_idx: int, 
    activation: Any = nnx.relu
) -> jax.Array:
    """
    Evaluates the remaining layers of a model given a pre-activation from an intermediate layer.
    
    Args:
        base_model: The model to evaluate. Expected to have a 'layers' sequence or 'l1', 'l2', etc.
        preact: The intermediate pre-activation tensor.
        current_layer_idx: The 0-indexed index of the preact's layer (e.g., 0 for 'l1', 1 for 'l2').
        activation: The activation function to apply.
    """
    h = preact
    if hasattr(base_model, 'layers'):
        if current_layer_idx >= len(base_model.layers):
            raise ValueError(f"current_layer_idx {current_layer_idx} is out of bounds for model with {len(base_model.layers)} layers.")
        for idx in range(current_layer_idx + 1, len(base_model.layers)):
            h = activation(h)
            layer = base_model.layers[idx]
            h = h @ layer.kernel.get_value() + layer.bias.get_value()
        return h
    elif hasattr(base_model, f'l{current_layer_idx + 1}'):
        i = current_layer_idx + 2
        while hasattr(base_model, f'l{i}'):
            h = activation(h)
            layer = getattr(base_model, f'l{i}')
            h = h @ layer.kernel.get_value() + layer.bias.get_value()
            i += 1
        return h
    else:
        raise TypeError(
            f"base_model (type {type(base_model).__name__}) does not match the expected interface "
            f"or the current_layer_idx {current_layer_idx} is invalid. The model must have either "
            f"a 'layers' sequence or continuous attributes 'l1', 'l2', etc."
        )

# ==============================================================================
# PJSVD Compact Ensembles
# ==============================================================================
# Instead of storing N copies of full weight matrices, these classes store only:
#   - The K null-space direction vectors  (K, D_flat)  -- shared across members
#   - The N×K latent coefficient matrix  (N, K)        -- tiny
#   - The original base weights           (D_flat)      -- one copy
#
# Perturbed weights are recomputed on-the-fly for each member during predict(),
# so peak memory is O(1 member) rather than O(N members).
# ==============================================================================

class CompactPJSVDEnsemble:
    """
    Memory-efficient PJSVD ensemble for the single-layer (W1) perturbation case.

    Stores the compact representation:
        v_opts     : (K, D_W1) flat null-space directions
        sigmas     : (K,) singular values
        z_coeffs   : (N, K) latent noise drawn at construction time
        base params: W1, b1, W2, b2 (one copy each)

    predict() reconstructs each member's (W1_new, W2_new, b2_new) on-the-fly.
    """

    def __init__(
        self,
        base_model: TransitionModel,
        v_opts: jax.Array,          # (K, D_W1) flat directions
        sigmas: np.ndarray,         # (K,) singular values
        z_coeffs: np.ndarray,       # (N, K) latent coefficients
        perturbation_scale: float,
        W1: jax.Array,
        b1: jax.Array,
        W2: jax.Array,
        b2: jax.Array,
        mu_old: jax.Array,
        std_old: jax.Array,
        sigma_sq_weights: bool = False,  # True for multi-sigma² denominator
        activation=None,             # Activation fn for L1/L2; defaults to nnx.relu
        X_sub: jax.Array = None,     # (N_sub, D_in) training subset
    ):
        self.base_model = base_model
        self.v_opts = v_opts                    # (K, D_W1)
        self.sigmas = jnp.array(sigmas)        # (K,)
        self.z_coeffs = z_coeffs              # (N, K) numpy array
        self.perturbation_scale = perturbation_scale
        self.W1 = W1
        self.b1 = b1
        self.W2 = W2
        self.b2 = b2
        self.mu_old = mu_old
        self.std_old = std_old
        self.sigma_sq_weights = sigma_sq_weights
        self.activation = activation if activation is not None else nnx.relu
        
        if X_sub is None:
            raise ValueError("X_sub must be provided to precompute proper correction factors.")
        self.X_sub = X_sub
        self._precompute_corrections()
        
    def _precompute_corrections(self):
        safe_sigmas = self.sigmas + 1e-6
        if self.sigma_sq_weights:
            coeffs_all = self.z_coeffs / np.array(safe_sigmas ** 2)
        else:
            coeffs_all = self.z_coeffs / np.array(safe_sigmas)

        norms = np.linalg.norm(coeffs_all, axis=1, keepdims=True) + 1e-12
        coeffs_all = (coeffs_all / norms) * self.perturbation_scale  # (N, K)

        dW1s = coeffs_all @ np.array(self.v_opts)

        scale_factors = []
        b2_news = []

        for i in range(len(self.z_coeffs)):
            dW1 = jnp.array(dW1s[i]).reshape(self.W1.shape)
            W1_new = self.W1 + dW1
            h_new = self.activation(self.X_sub @ W1_new + self.b1)

            mu_new = jnp.mean(h_new, axis=0)
            std_new = jnp.std(h_new, axis=0)
            scale_factor = self.std_old / (std_new + 1e-6)
            b2_new = self.b2 + jnp.dot(self.mu_old, self.W2) - jnp.dot(mu_new, self.W2 * scale_factor[:, None])

            scale_factors.append(scale_factor)
            b2_news.append(b2_new)

        self.scale_factors = jnp.stack(scale_factors, axis=0)
        self.b2_news = jnp.stack(b2_news, axis=0)

    def manual_forward(self, x, w1, b1, w2, b2):
        h1 = self.activation(x @ w1 + b1)
        h2_pre = h1 @ w2 + b2
        return evaluate_tail_from_preact(self.base_model, h2_pre, current_layer_idx=1, activation=self.activation)

    def predict(self, x: jax.Array) -> jax.Array:
        ys = []
        safe_sigmas = self.sigmas + 1e-6
        # Precompute all coefficient vectors at once (N, K) -> saves repeated work
        if self.sigma_sq_weights:
            coeffs_all = self.z_coeffs / np.array(safe_sigmas ** 2)
        else:
            coeffs_all = self.z_coeffs / np.array(safe_sigmas)

        norms = np.linalg.norm(coeffs_all, axis=1, keepdims=True) + 1e-12
        coeffs_all = (coeffs_all / norms) * self.perturbation_scale  # (N, K)

        # dW1 for all members: (N, K) @ (K, D_W1) -> (N, D_W1)
        dW1s = coeffs_all @ np.array(self.v_opts)  # on CPU/numpy to avoid devices memory spike

        for i in range(len(self.z_coeffs)):
            dW1 = jnp.array(dW1s[i]).reshape(self.W1.shape)
            W1_new = self.W1 + dW1
            h_new = self.activation(x @ W1_new + self.b1)

            W2_new = self.W2 * self.scale_factors[i][:, None]
            b2_new = self.b2_news[i]

            h2_pre = h_new @ W2_new + b2_new
            out = evaluate_tail_from_preact(self.base_model, h2_pre, current_layer_idx=1, activation=self.activation)
            ys.append(out)

        return jnp.stack(ys, axis=0)

    def predict_one(self, x: jax.Array, idx: int) -> jax.Array:
        safe_sigmas = self.sigmas + 1e-6
        z = self.z_coeffs[idx]
        if self.sigma_sq_weights:
            coeffs = z / np.array(safe_sigmas ** 2)
        else:
            coeffs = z / np.array(safe_sigmas)
        coeffs = coeffs / (np.linalg.norm(coeffs) + 1e-12) * self.perturbation_scale

        dW1 = jnp.dot(jnp.array(coeffs), self.v_opts).reshape(self.W1.shape)
        W1_new = self.W1 + dW1
        h_new = self.activation(x @ W1_new + self.b1)

        W2_new = self.W2 * self.scale_factors[idx][:, None]
        b2_new = self.b2_news[idx]

        h2_pre = h_new @ W2_new + b2_new
        return evaluate_tail_from_preact(self.base_model, h2_pre, current_layer_idx=1, activation=self.activation)




class LeastSquaresCompactPJSVDEnsemble(CompactPJSVDEnsemble):
    """
    PJSVD ensemble for the single-layer (W1) perturbation case,
    but performs the analytic least squares Affine Correction over the entire next layer matrix
    instead of just scalar scaling.
    """
    
    def _precompute_corrections(self):
        safe_sigmas = self.sigmas + 1e-6
        if self.sigma_sq_weights:
            coeffs_all = self.z_coeffs / np.array(safe_sigmas ** 2)
        else:
            coeffs_all = self.z_coeffs / np.array(safe_sigmas)

        norms = np.linalg.norm(coeffs_all, axis=1, keepdims=True) + 1e-12
        coeffs_all = (coeffs_all / norms) * self.perturbation_scale  # (N, K)

        dW1s = coeffs_all @ np.array(self.v_opts)

        W2_news = []
        b2_news = []
        
        # Target is the original unperturbed activations
        h_old = self.activation(self.X_sub @ self.W1 + self.b1)
        # Z is the original pre-activations of the *next* layer
        Z = h_old @ self.W2 + self.b2
        
        N_samples = Z.shape[0]

        for i in range(len(self.z_coeffs)):
            dW1 = jnp.array(dW1s[i]).reshape(self.W1.shape)
            W1_new = self.W1 + dW1
            # Perturbed activations
            h_new = self.activation(self.X_sub @ W1_new + self.b1)

            # We want to find W_next_new, b_next_new such that:
            # h_new @ W_next_new + b_next_new ≈ Z
            # Construct augmented matrix [h_new, 1]
            ones = jnp.ones((N_samples, 1), dtype=h_new.dtype)
            h_new_aug = jnp.concatenate([h_new, ones], axis=-1)
            
            # Solve least squares: h_new_aug @ W_aug ≈ Z
            # W_aug is (D_h + 1, D_out)
            W_aug, _, _, _ = jnp.linalg.lstsq(h_new_aug, Z, rcond=None)
            
            W2_new = W_aug[:-1, :]
            b2_new = W_aug[-1, :]

            W2_news.append(W2_new)
            b2_news.append(b2_new)

        self.W2_news = jnp.stack(W2_news, axis=0)
        self.b2_news = jnp.stack(b2_news, axis=0)

    def predict(self, x: jax.Array) -> jax.Array:
        ys = []
        safe_sigmas = self.sigmas + 1e-6
        if self.sigma_sq_weights:
            coeffs_all = self.z_coeffs / np.array(safe_sigmas ** 2)
        else:
            coeffs_all = self.z_coeffs / np.array(safe_sigmas)

        norms = np.linalg.norm(coeffs_all, axis=1, keepdims=True) + 1e-12
        coeffs_all = (coeffs_all / norms) * self.perturbation_scale

        dW1s = coeffs_all @ np.array(self.v_opts)

        for i in range(len(self.z_coeffs)):
            dW1 = jnp.array(dW1s[i]).reshape(self.W1.shape)
            W1_new = self.W1 + dW1
            h_new = self.activation(x @ W1_new + self.b1)

            W2_new = self.W2_news[i]
            b2_new = self.b2_news[i]

            h2_pre = h_new @ W2_new + b2_new
            out = evaluate_tail_from_preact(self.base_model, h2_pre, current_layer_idx=1, activation=self.activation)
            ys.append(out)

        return jnp.stack(ys, axis=0)

    def predict_one(self, x: jax.Array, idx: int) -> jax.Array:
        safe_sigmas = self.sigmas + 1e-6
        z = self.z_coeffs[idx]
        if self.sigma_sq_weights:
            coeffs = z / np.array(safe_sigmas ** 2)
        else:
            coeffs = z / np.array(safe_sigmas)
        coeffs = coeffs / (np.linalg.norm(coeffs) + 1e-12) * self.perturbation_scale

        dW1 = jnp.dot(jnp.array(coeffs), self.v_opts).reshape(self.W1.shape)
        W1_new = self.W1 + dW1
        h_new = self.activation(x @ W1_new + self.b1)

        W2_new = self.W2_news[idx]
        b2_new = self.b2_news[idx]

        h2_pre = h_new @ W2_new + b2_new
        return evaluate_tail_from_preact(self.base_model, h2_pre, current_layer_idx=1, activation=self.activation)




class CompactMultiLayerPJSVDEnsemble:
    """
    Memory-efficient PJSVD ensemble for the multi-layer (W1, W2) perturbation case.

    Stores:
        v_opts     : (K, D_W1 + D_W2) flat null-space directions
        sigmas     : (K,) singular values
        z_coeffs   : (N, K) latent noise
        base params: W1, b1, W2, b2, W3, b3 (one copy each)
        w1_size    : int (for splitting the flat direction into W1 and W2 parts)
    """

    def __init__(
        self,
        base_model: TransitionModel,
        v_opts: jax.Array,          # (K, D_W1+D_W2)
        sigmas: np.ndarray,         # (K,)
        z_coeffs: np.ndarray,       # (N, K)
        perturbation_scale: float,
        W1: jax.Array,
        b1: jax.Array,
        W2: jax.Array,
        b2: jax.Array,
        W3: jax.Array,
        b3: jax.Array,
        mu_old: jax.Array,
        std_old: jax.Array,
        activation=None,             # Activation fn for L1/L2; defaults to nnx.relu
        X_sub: jax.Array = None,     # (N_sub, D_in) training subset
    ):
        self.base_model = base_model
        self.v_opts = v_opts
        self.sigmas = jnp.array(sigmas)
        self.z_coeffs = z_coeffs
        self.perturbation_scale = perturbation_scale
        self.W1 = W1
        self.b1 = b1
        self.W2 = W2
        self.b2 = b2
        self.W3 = W3
        self.b3 = b3
        self.mu_old = mu_old
        self.std_old = std_old
        self.w1_size = W1.size
        self.activation = activation if activation is not None else nnx.relu
        
        if X_sub is None:
            raise ValueError("X_sub must be provided to precompute proper correction factors.")
        self.X_sub = X_sub
        self._precompute_corrections()
        
    def _precompute_corrections(self):
        safe_sigmas = self.sigmas + 1e-6
        coeffs_all = self.z_coeffs / np.array(safe_sigmas ** 2)
        norms = np.linalg.norm(coeffs_all, axis=1, keepdims=True) + 1e-12
        coeffs_all = (coeffs_all / norms) * self.perturbation_scale

        total_perturbations = coeffs_all @ np.array(self.v_opts)

        scale_factors = []
        b3_news = []

        for i in range(len(self.z_coeffs)):
            p = jnp.array(total_perturbations[i])
            v_w1 = p[:self.w1_size].reshape(self.W1.shape)
            v_w2 = p[self.w1_size:].reshape(self.W2.shape)
            W1_new = self.W1 + v_w1
            W2_new = self.W2 + v_w2

            h1 = self.activation(self.X_sub @ W1_new + self.b1)
            h2 = self.activation(h1 @ W2_new + self.b2)

            mu_new = jnp.mean(h2, axis=0)
            std_new = jnp.std(h2, axis=0)
            scale_factor = self.std_old / (std_new + 1e-6)
            b3_new = self.b3 + jnp.dot(self.mu_old, self.W3) - jnp.dot(mu_new, self.W3 * scale_factor[:, None])

            scale_factors.append(scale_factor)
            b3_news.append(b3_new)

        self.scale_factors = jnp.stack(scale_factors, axis=0)
        self.b3_news = jnp.stack(b3_news, axis=0)

    def _get_perturbed_weights(self, coeffs_vec):
        """Given a (K,) coefficient vector, return perturbed (W1, W2, W3, b3)."""
        total_pert = jnp.dot(jnp.array(coeffs_vec), self.v_opts)  # (D_W1+D_W2,)

        v_w1 = total_pert[:self.w1_size].reshape(self.W1.shape)
        v_w2 = total_pert[self.w1_size:].reshape(self.W2.shape)
        W1_new = self.W1 + v_w1
        W2_new = self.W2 + v_w2
        return W1_new, W2_new

    def _apply_w3_correction(self, x, W1_new, W2_new, idx):
        """Forward pass with correction on W3."""
        h1 = self.activation(x @ W1_new + self.b1)
        h2 = self.activation(h1 @ W2_new + self.b2)

        W3_new = self.W3 * self.scale_factors[idx][:, None]
        b3_new = self.b3_news[idx]

        h3_pre = h2 @ W3_new + b3_new
        return evaluate_tail_from_preact(self.base_model, h3_pre, current_layer_idx=2, activation=self.activation)

    def predict(self, x: jax.Array) -> jax.Array:
        safe_sigmas = self.sigmas + 1e-6
        coeffs_all = self.z_coeffs / np.array(safe_sigmas ** 2)
        norms = np.linalg.norm(coeffs_all, axis=1, keepdims=True) + 1e-12
        coeffs_all = (coeffs_all / norms) * self.perturbation_scale  # (N, K)

        # Compute all perturbation directions upfront on CPU (saves GPU memory)
        total_perturbations = coeffs_all @ np.array(self.v_opts)  # (N, D_W1+D_W2)

        ys = []
        for i in range(len(self.z_coeffs)):
            p = jnp.array(total_perturbations[i])
            v_w1 = p[:self.w1_size].reshape(self.W1.shape)
            v_w2 = p[self.w1_size:].reshape(self.W2.shape)
            W1_new = self.W1 + v_w1
            W2_new = self.W2 + v_w2
            out = self._apply_w3_correction(x, W1_new, W2_new, i)
            ys.append(out)

        return jnp.stack(ys, axis=0)

    def predict_one(self, x: jax.Array, idx: int) -> jax.Array:
        safe_sigmas = self.sigmas + 1e-6
        z = self.z_coeffs[idx]
        coeffs = z / np.array(safe_sigmas ** 2)
        coeffs = coeffs / (np.linalg.norm(coeffs) + 1e-12) * self.perturbation_scale

        p = jnp.dot(jnp.array(coeffs), self.v_opts)
        v_w1 = p[:self.w1_size].reshape(self.W1.shape)
        v_w2 = p[self.w1_size:].reshape(self.W2.shape)
        return self._apply_w3_correction(x, self.W1 + v_w1, self.W2 + v_w2, idx)




class LeastSquaresCompactMultiLayerPJSVDEnsemble(CompactMultiLayerPJSVDEnsemble):
    """
    Memory-efficient PJSVD ensemble for the multi-layer (W1, W2) perturbation case.
    Uses least-squares over W3 to fit to the calibration activations instead of scalar scaling.
    """
    def _precompute_corrections(self):
        safe_sigmas = self.sigmas + 1e-6
        coeffs_all = self.z_coeffs / np.array(safe_sigmas ** 2)
        norms = np.linalg.norm(coeffs_all, axis=1, keepdims=True) + 1e-12
        coeffs_all = (coeffs_all / norms) * self.perturbation_scale

        total_perturbations = coeffs_all @ np.array(self.v_opts)

        W3_news = []
        b3_news = []
        
        # Target is the original unperturbed activations
        h1_old = self.activation(self.X_sub @ self.W1 + self.b1)
        h2_old = self.activation(h1_old @ self.W2 + self.b2)
        # Z is the original pre-activations of the *next* layer
        Z = h2_old @ self.W3 + self.b3
        
        N_samples = Z.shape[0]

        for i in range(len(self.z_coeffs)):
            p = jnp.array(total_perturbations[i])
            v_w1 = p[:self.w1_size].reshape(self.W1.shape)
            v_w2 = p[self.w1_size:].reshape(self.W2.shape)
            W1_new = self.W1 + v_w1
            W2_new = self.W2 + v_w2

            h1 = self.activation(self.X_sub @ W1_new + self.b1)
            h2_new = self.activation(h1 @ W2_new + self.b2)

            ones = jnp.ones((N_samples, 1), dtype=h2_new.dtype)
            h2_new_aug = jnp.concatenate([h2_new, ones], axis=-1)
            
            W_aug, _, _, _ = jnp.linalg.lstsq(h2_new_aug, Z, rcond=None)
            
            W3_new = W_aug[:-1, :]
            b3_new = W_aug[-1, :]

            W3_news.append(W3_new)
            b3_news.append(b3_new)

        self.W3_news = jnp.stack(W3_news, axis=0)
        self.b3_news = jnp.stack(b3_news, axis=0)

    def _apply_w3_correction_lstsq(self, x, W1_new, W2_new, idx):
        """Forward pass with correction on W3."""
        h1 = self.activation(x @ W1_new + self.b1)
        h2 = self.activation(h1 @ W2_new + self.b2)

        W3_new = self.W3_news[idx]
        b3_new = self.b3_news[idx]

        h3_pre = h2 @ W3_new + b3_new
        return evaluate_tail_from_preact(self.base_model, h3_pre, current_layer_idx=2, activation=self.activation)

    def predict(self, x: jax.Array) -> jax.Array:
        safe_sigmas = self.sigmas + 1e-6
        coeffs_all = self.z_coeffs / np.array(safe_sigmas ** 2)
        norms = np.linalg.norm(coeffs_all, axis=1, keepdims=True) + 1e-12
        coeffs_all = (coeffs_all / norms) * self.perturbation_scale

        total_perturbations = coeffs_all @ np.array(self.v_opts)

        ys = []
        for i in range(len(self.z_coeffs)):
            p = jnp.array(total_perturbations[i])
            v_w1 = p[:self.w1_size].reshape(self.W1.shape)
            v_w2 = p[self.w1_size:].reshape(self.W2.shape)
            W1_new = self.W1 + v_w1
            W2_new = self.W2 + v_w2
            out = self._apply_w3_correction_lstsq(x, W1_new, W2_new, i)
            ys.append(out)

        return jnp.stack(ys, axis=0)

    def predict_one(self, x: jax.Array, idx: int) -> jax.Array:
        safe_sigmas = self.sigmas + 1e-6
        z = self.z_coeffs[idx]
        coeffs = z / np.array(safe_sigmas ** 2)
        coeffs = coeffs / (np.linalg.norm(coeffs) + 1e-12) * self.perturbation_scale

        p = jnp.dot(jnp.array(coeffs), self.v_opts)
        v_w1 = p[:self.w1_size].reshape(self.W1.shape)
        v_w2 = p[self.w1_size:].reshape(self.W2.shape)
        return self._apply_w3_correction_lstsq(x, self.W1 + v_w1, self.W2 + v_w2, idx)




# ==============================================================================
# Hybrid Ensemble: Deep Ensemble + PJSVD per member
# ==============================================================================

class EnsemblePJSVDHybrid:
    """
    Hybrid uncertainty estimator: trains M independent base models (deep ensemble)
    and applies PJSVD to each member, yielding M × n_perturbations total samples.

    predict() concatenates predictions from all per-member CompactPJSVDEnsembles
    along axis 0, so the output shape is (M * n_perturbations, N, output_dim).
    """

    def __init__(self, pjsvd_ensembles: List['CompactPJSVDEnsemble']):
        """
        Args:
            pjsvd_ensembles: one CompactPJSVDEnsemble per base model.
        """
        self.pjsvd_ensembles = pjsvd_ensembles

    def predict(self, x: jax.Array) -> jax.Array:
        """Returns (M * S, N, output_dim) stacked predictions."""
        all_preds = [ens.predict(x) for ens in self.pjsvd_ensembles]
        return jnp.concatenate(all_preds, axis=0)

    def predict_one(self, x: jax.Array, idx: int) -> jax.Array:
        """Index into the flattened list of M*S members."""
        n_per = len(self.pjsvd_ensembles[0].z_coeffs)
        member_idx = idx // n_per
        sample_idx = idx % n_per
        return self.pjsvd_ensembles[member_idx].predict_one(x, sample_idx)


# ==============================================================================
# Legacy / Weight-Copy Ensembles (kept for StandardEnsemble compatibility)
# ==============================================================================

class Ensemble:
    """
    Legacy PJSVD ensemble that stores full weight copies.
    Prefer CompactPJSVDEnsemble for large N.
    """
    def __init__(self, base_model: TransitionModel, perturbations: List[Tuple]):
        self.base_model = base_model
        self.perturbations = perturbations

    def manual_forward(self, x, w1, b1, w2, b2):
        act = getattr(self.base_model, 'activation', nnx.relu)
        h1 = act(x @ w1 + b1)
        h2_pre = h1 @ w2 + b2
        return evaluate_tail_from_preact(self.base_model, h2_pre, current_layer_idx=1, activation=act)

    def predict(self, x: jax.Array) -> jax.Array:
        ys = []
        for w1, b1, w2, b2 in self.perturbations:
            ys.append(self.manual_forward(x, w1, b1, w2, b2))
        return jnp.stack(ys, axis=0)

    def predict_one(self, x: jax.Array, idx: int) -> jax.Array:
        w1, b1, w2, b2 = self.perturbations[idx]
        return self.manual_forward(x, w1, b1, w2, b2)


class MultiLayerPJSVDEnsemble:
    """
    Legacy multi-layer PJSVD ensemble that stores full weight copies.
    Prefer CompactMultiLayerPJSVDEnsemble for large N.
    """
    def __init__(self, base_model: TransitionModel, perturbations: List[Tuple]):
        self.base_model = base_model
        self.perturbations = perturbations

    def manual_forward(self, x, w1, b1, w2, b2, w3, b3):
        act = getattr(self.base_model, 'activation', nnx.relu)
        h1 = act(x @ w1 + b1)
        h2 = act(h1 @ w2 + b2)
        h3_pre = h2 @ w3 + b3
        return evaluate_tail_from_preact(self.base_model, h3_pre, current_layer_idx=2, activation=act)

    def predict(self, x: jax.Array) -> jax.Array:
        ys = []
        for w1, b1, w2, b2, w3, b3 in self.perturbations:
            ys.append(self.manual_forward(x, w1, b1, w2, b2, w3, b3))
        return jnp.stack(ys, axis=0)

    def predict_one(self, x: jax.Array, idx: int) -> jax.Array:
        w1, b1, w2, b2, w3, b3 = self.perturbations[idx]
        return self.manual_forward(x, w1, b1, w2, b2, w3, b3)


class StandardEnsemble:
    def __init__(self, models: List[nnx.Module]):
        self.models = models

    def predict(self, x: jax.Array) -> jax.Array:
        ys = []
        for model in self.models:
            out = model(x)
            if isinstance(out, tuple) and len(out) == 2:
                # Probabilistic model: (mean, var)
                mean, var = out
                # We return a "representative sample" for each member
                # By adding Gaussian noise with the predicted variance, 
                # the across-ensemble variance will correctly match the mixture variance:
                # Var_total = E[Var_aleatoric] + Var[E_epistemic]
                rng = jax.random.PRNGKey(np.random.randint(0, 10000)) # Simple seeding
                eps = jax.random.normal(rng, mean.shape)
                ys.append(mean + jnp.sqrt(var) * eps)
            else:
                # Standard point-estimate model
                ys.append(out)
        return jnp.stack(ys, axis=0)

    def predict_one(self, x: jax.Array, idx: int) -> jax.Array:
        out = self.models[idx](x)
        if isinstance(out, tuple) and len(out) == 2:
            mean, var = out
            rng = jax.random.PRNGKey(np.random.randint(0, 10000))
            return mean + jnp.sqrt(var) * jax.random.normal(rng, mean.shape)
        return out


class MCDropoutEnsemble:
    def __init__(self, model: TransitionModel, n_models: int):
        self.model = model
        self.n_models = n_models

    def predict(self, x: jax.Array) -> jax.Array:
        ys = []
        for _ in range(self.n_models):
            ys.append(self.model(x, deterministic=False))
        return jnp.stack(ys, axis=0)

    def predict_one(self, x: jax.Array, idx: int) -> jax.Array:
        return self.model(x, deterministic=False)


class SWAGEnsemble:
    def __init__(self, model: TransitionModel, swag_mean: nnx.State, swag_var: nnx.State, n_models: int):
        self.model = model
        self.swag_mean = swag_mean
        self.swag_var = swag_var
        self.n_models = n_models

    def _sample_model(self) -> TransitionModel:
        sample_params = jax.tree.map(
            # SWAG paper uses 0.5 * Sigma_diag for sampling
            lambda m, v: m + jnp.sqrt(0.5 * v) * np.random.normal(size=m.shape),
            self.swag_mean, self.swag_var
        )
        nnx.update(self.model, sample_params)
        return self.model

    def predict(self, x: jax.Array) -> jax.Array:
        ys = []
        for _ in range(self.n_models):
            sampled_m = self._sample_model()
            out = sampled_m(x)
            if isinstance(out, tuple) and len(out) == 2:
                mean, var = out
                rng = jax.random.PRNGKey(np.random.randint(0, 10000))
                ys.append(mean + jnp.sqrt(var) * jax.random.normal(rng, mean.shape))
            else:
                ys.append(out)
        return jnp.stack(ys, axis=0)

    def predict_one(self, x: jax.Array, idx: int) -> jax.Array:
        out = self._sample_model()(x)
        if isinstance(out, tuple) and len(out) == 2:
            mean, var = out
            rng = jax.random.PRNGKey(np.random.randint(0, 10000))
            return mean + jnp.sqrt(var) * jax.random.normal(rng, mean.shape)
        return out


class LaplaceEnsemble:
    def __init__(self, model: TransitionModel, kfac_factors: dict, prior_precision: float, n_models: int, data_size: int):
        self.model = model
        self.kfac_factors = kfac_factors
        self.prior_precision = prior_precision
        self.n_models = n_models
        self.data_size = data_size

        self.inv_scales = {}
        for layer_name, (A, S) in self.kfac_factors.items():
            A = A + jnp.eye(A.shape[0]) * 1e-6
            S = S + jnp.eye(S.shape[0]) * 1e-6
            U_A, eig_A, _ = jnp.linalg.svd(A)
            U_S, eig_S, _ = jnp.linalg.svd(S)
            eig_A = jnp.maximum(eig_A, 0.0)
            eig_S = jnp.maximum(eig_S, 0.0)
            self.inv_scales[layer_name] = (U_A, eig_A, U_S, eig_S)

        self.map_params = {}
        if hasattr(model, 'layers'):
            for i, layer in enumerate(model.layers):
                self.map_params[f'l{i+1}'] = (layer.kernel.get_value(), layer.bias.get_value())
        else:
            i = 1
            while hasattr(model, f'l{i}'):
                layer = getattr(model, f'l{i}')
                self.map_params[f'l{i}'] = (layer.kernel.get_value(), layer.bias.get_value())
                i += 1

    def _sample_layer_weights(self, layer_name: str):
        W_map, b_map = self.map_params[layer_name]
        W_full = jnp.concatenate([W_map, jnp.expand_dims(b_map, axis=0)], axis=0)
        U_A, eig_A, U_S, eig_S = self.inv_scales[layer_name]
        Z = np.random.normal(size=W_full.shape)
        N, lambda_val = self.data_size, self.prior_precision
        eig_matrix = N * jnp.outer(eig_A, eig_S) + lambda_val
        std_matrix = 1.0 / jnp.sqrt(eig_matrix)
        scaled_Z = Z * std_matrix
        delta_W = U_A @ scaled_Z @ U_S.T
        W_new = W_full + delta_W
        return W_new[:-1, :], W_new[-1, :]

    def _sample_model(self):
        if hasattr(self.model, 'layers'):
            total_norm_sq = 0.0
            for i, layer in enumerate(self.model.layers):
                w, b = self._sample_layer_weights(f'l{i+1}')
                norm = jnp.linalg.norm((w - self.map_params[f'l{i+1}'][0]).flatten())
                total_norm_sq += norm**2
                nnx.update(layer.kernel, w)
                nnx.update(layer.bias, b)
            total_norm = float(jnp.sqrt(total_norm_sq))
            return self.model, total_norm

        w1, b1 = self._sample_layer_weights('l1')
        w2, b2 = self._sample_layer_weights('l2')
        w3, b3 = self._sample_layer_weights('l3')

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

import math

class SubspaceInferenceEnsemble:
    def __init__(
        self, 
        base_model: nnx.Module, 
        swag_mean: nnx.State, 
        pca_components: jax.Array, 
        n_samples: int = 100, 
        temperature: float = 1.0,
        X_train: jax.Array = None,
        Y_train: jax.Array = None,
        use_ess: bool = True,
        is_classification: bool = False
    ):
        self.base_model = base_model
        self.swag_mean = swag_mean
        self.pca_components = pca_components
        self.n_samples = n_samples
        self.temperature = temperature
        self.is_classification = is_classification

        swag_mean_flat, self.unflatten_fn = jax.flatten_util.ravel_pytree(swag_mean)
        self.swag_mean_flat = swag_mean_flat

        self.z_samples = []
        if use_ess and X_train is not None and Y_train is not None:
            self._run_ess(X_train, Y_train)
        else:
            rng = jax.random.PRNGKey(42)
            c = pca_components.shape[1]
            self.z_samples = jax.random.normal(rng, (n_samples, c)) * temperature
            
    def _run_ess(self, X, Y):
        import optax
        def log_likelihood(z):
            w = self.swag_mean_flat + self.pca_components @ z
            nnx.update(self.base_model, self.unflatten_fn(w))
            preds = self.base_model(X)
            if self.is_classification:
                loss = jnp.sum(optax.softmax_cross_entropy_with_integer_labels(logits=preds, labels=Y))
                return -float(loss) / self.temperature
            else:
                if isinstance(preds, tuple) and len(preds) == 2:
                    mean, var = preds
                    nll = jnp.sum(0.5 * (jnp.log(var) + (mean - Y)**2 / var))
                    return -float(nll) / (self.temperature ** 2)
                else:
                    mse = jnp.sum((preds - Y)**2)
                    return -0.5 * float(mse) / (self.temperature ** 2)
            
        def ll_fn(z):
            return log_likelihood(z)

        c = self.pca_components.shape[1]
        z = np.zeros(c)
        ll_z = float(ll_fn(z))
        
        n_iters = self.n_samples + 100
        samples = []
                
        for i in range(n_iters):
            nu = np.random.normal(0, 1, size=c)
            u = np.random.uniform(0, 1)
            log_y = ll_z + math.log(u)
            
            theta = np.random.uniform(0, 2*math.pi)
            theta_min = theta - 2*math.pi
            theta_max = theta
            
            while True:
                z_prime = z * math.cos(theta) + nu * math.sin(theta)
                ll_prime = float(ll_fn(z_prime))
                if ll_prime > log_y:
                    z = z_prime
                    ll_z = ll_prime
                    break
                else:
                    if theta < 0:
                        theta_min = theta
                    else:
                        theta_max = theta
                    theta = np.random.uniform(theta_min, theta_max)
            
            if i >= 100:
                samples.append(z)
                
        self.z_samples = np.array(samples)

    def _sample_model(self, idx: int):
        z = self.z_samples[idx]
        w_diff = self.pca_components @ jnp.array(z)
        w = self.swag_mean_flat + w_diff
        norm = float(jnp.linalg.norm(w_diff))
        nnx.update(self.base_model, self.unflatten_fn(w))
        return self.base_model, norm
        
    def predict(self, x: jax.Array) -> jax.Array:
        ys = []
        for i in range(self.n_samples):
            sampled_m, _ = self._sample_model(i)
            out = sampled_m(x)
            if isinstance(out, tuple) and len(out) == 2:
                # Probabilistic model: (mean, var)
                mean, var = out
                rng = jax.random.PRNGKey(np.random.randint(0, 10000))
                eps = jax.random.normal(rng, mean.shape)
                ys.append(mean + jnp.sqrt(var) * eps)
            else:
                ys.append(out)
        return jnp.stack(ys, axis=0)
        
    def predict_one(self, x: jax.Array, idx: int) -> jax.Array:
        sampled_m, _ = self._sample_model(idx)
        out = sampled_m(x)
        if isinstance(out, tuple) and len(out) == 2:
            mean, var = out
            rng = jax.random.PRNGKey(np.random.randint(0, 10000))
            return mean + jnp.sqrt(var) * jax.random.normal(rng, mean.shape)
        return out


# ==============================================================================
# BatchNorm Refit PJSVD Ensembles  (for ResNet-50 / CIFAR)
# ==============================================================================
# The key idea: instead of refitting the *next layer's full weight matrix*,
# we exploit the Conv→BN→ReLU structure of ResNets.
# For each perturbed conv kernel:
#   1. Run the calibration subset through the perturbed conv.
#   2. Solve, per output channel c, a tiny 1-D linear regression:
#        γ'_c * ỹ_c + β'_c ≈ z_c
#      where ỹ_c = perturbed BN input, z_c = original BN output.
#      Closed form: γ' = cov(ỹ,z)/var(ỹ),  β' = mean(z) - γ'·mean(ỹ)
#   3. Store (γ'_i, β'_i) for each ensemble member i.
# Prediction: apply perturbed conv, override BN with (γ'_i, β'_i), run tail.
# ==============================================================================

def _bn_refit_channel_wise(
    h_perturbed: jax.Array,   # (N, H, W, C) or (N, C) raw conv output (pre-BN)
    z_original: jax.Array,    # (N, H, W, C) or (N, C) original BN output
) -> tuple:
    """
    Per-channel least-squares affine refit.

    Solves  γ'_c * ỹ_c + β'_c ≈ z_c  independently for each channel c,
    where all spatial positions are treated as independent samples.

    Returns:
        gamma_new: (C,) fitted scale factors
        beta_new:  (C,) fitted shifts
    """
    # Flatten spatial dims: (..., C) → (N*H*W, C)
    flat_p = h_perturbed.reshape(-1, h_perturbed.shape[-1])   # (M, C)
    flat_z = z_original.reshape(-1, z_original.shape[-1])     # (M, C)

    mean_p = jnp.mean(flat_p, axis=0)  # (C,)
    mean_z = jnp.mean(flat_z, axis=0)  # (C,)

    # cov(ỹ_c, z_c) / var(ỹ_c)  — per channel
    cov  = jnp.mean((flat_p - mean_p) * (flat_z - mean_z), axis=0)  # (C,)
    var  = jnp.mean((flat_p - mean_p) ** 2, axis=0) + 1e-8           # (C,)

    gamma_new = cov / var                         # (C,)
    beta_new  = mean_z - gamma_new * mean_p       # (C,)
    return gamma_new, beta_new


class BatchNormRefitPJSVDEnsemble:
    """
    Memory-efficient PJSVD ensemble for ResNet-50 (CIFAR) with BatchNorm Refit.

    Perturbation targets the stem conv kernel.  The BN layer immediately
    following the stem is refitted per-channel via closed-form 1-D OLS,
    neutralising the perturbation without touching any other weights.

    Construction precomputes (γ'_i, β'_i) for all N ensemble members.
    Prediction:
        1. Perturb stem kernel  →  raw conv output
        2. Apply refitted BN params  →  corrected activations
        3. Run remainder of ResNet normally

    All images are expected in NHWC float32 format.
    """

    def __init__(
        self,
        base_model,                  # ResNet50 instance (trained)
        v_opts: jax.Array,           # (K, D_stem_kernel) flat directions
        sigmas: jax.Array,           # (K,) singular values
        z_coeffs: np.ndarray,        # (N_ens, K) latent draws
        perturbation_scale: float,
        W_stem: jax.Array,           # stem conv kernel (kH, kW, C_in, C_out)
        X_sub: jax.Array,            # (N_sub, 32, 32, 3) calibration images
        inf_batch_size: int = 64,    # batch size for inference to avoid OOM
    ):
        self.base_model        = base_model
        self.v_opts            = v_opts
        self.sigmas            = jnp.array(sigmas)
        self.z_coeffs          = z_coeffs
        self.perturbation_scale = perturbation_scale
        self.W_stem            = W_stem
        self.X_sub             = X_sub
        self.inf_batch_size    = inf_batch_size
        self._precompute_bn_refit()

    # ------------------------------------------------------------------ internal
    def _perturb_coeffs(self) -> np.ndarray:
        """Return (N_ens, D) perturbation matrix after scaling & normalising."""
        safe_sig   = np.array(self.sigmas) + 1e-6
        coeffs     = self.z_coeffs / safe_sig              # (N_ens, K)
        norms      = np.linalg.norm(coeffs, axis=1, keepdims=True) + 1e-12
        return (coeffs / norms) * self.perturbation_scale  # (N_ens, K)

    def _precompute_bn_refit(self):
        """Compute (γ'_i, β'_i) for all ensemble members on the calibration subset."""
        coeffs_all = self._perturb_coeffs()                     # (N_ens, K)
        dW_all     = coeffs_all @ np.array(self.v_opts)         # (N_ens, D_stem)

        # Original BN output (target for the regression)
        raw_orig = self.base_model.stem_conv_out_raw(self.X_sub)   # (M,32,32,64)
        z_orig   = self.base_model.stem_bn_from_raw(raw_orig,
                                                    use_running_average=True)  # (M,32,32,64)

        gammas, betas = [], []
        for i in range(len(self.z_coeffs)):
            dW   = jnp.array(dW_all[i]).reshape(self.W_stem.shape)
            W_new     = self.W_stem + dW

            # Perturbed raw conv output (patch stem.conv kernel temporarily)
            raw_pert  = self.base_model.stem.conv(self.X_sub,
                                                   kernel=W_new)     # (M,32,32,64)
            # Closed-form BN refit
            g, b = _bn_refit_channel_wise(raw_pert, z_orig)
            gammas.append(g)
            betas.append(b)

        self.gammas = jnp.stack(gammas, axis=0)  # (N_ens, 64)
        self.betas  = jnp.stack(betas,  axis=0)  # (N_ens, 64)

    def _forward_member(self, x: jax.Array, i: int) -> jax.Array:
        """Forward pass for ensemble member i."""
        coeffs_all = self._perturb_coeffs()
        dW = jnp.array((coeffs_all @ np.array(self.v_opts))[i]).reshape(self.W_stem.shape)
        W_new = self.W_stem + dW

        # Perturbed raw conv
        raw_pert = self.base_model.stem.conv(x)   # uses live kernel — we need manual call
        # Manual conv with perturbed kernel using jax.lax.conv_general_dilated
        raw_pert = jax.lax.conv_general_dilated(
            lhs=x,
            rhs=W_new.transpose(3, 2, 0, 1),   # OIHW for lax
            window_strides=(1, 1),
            padding='SAME',
            dimension_numbers=('NHWC', 'OIHW', 'NHWC')
        )                                          # (N,32,32,64)

        # Apply refitted BN: γ'·z + β'  (z = raw_pert here, γ'/β' already encode BN stats)
        h = self.gammas[i] * raw_pert + self.betas[i]  # (N,32,32,64)
        h = jax.nn.relu(h)

        return self.base_model.forward_from_stem_out(h, use_running_average=True)

    def predict(self, x: jax.Array) -> jax.Array:
        """Returns (N_ens, N, n_classes) logits."""
        ys = [self._forward_member(x, i) for i in range(len(self.z_coeffs))]
        return jnp.stack(ys, axis=0)

    def predict_one(self, x: jax.Array, idx: int) -> jax.Array:
        return self._forward_member(x, idx)


class MLBatchNormRefitPJSVDEnsemble:
    """
    Multi-layer BatchNorm Refit PJSVD ensemble for ResNet-50 (CIFAR).

    Perturbations are applied to the stem conv kernel AND the first conv of
    stage1 (conv1 of the first bottleneck, i.e., the 1×1 channel-expansion conv).
    A BN Refit is performed independently after each perturbed conv.

    The flat direction vector layout: [stem_kernel_flat | stage1_b0_conv1_flat]
    """

    def __init__(
        self,
        base_model,
        v_opts: jax.Array,           # (K, D_stem + D_stage1_conv1) flat directions
        sigmas: jax.Array,
        z_coeffs: np.ndarray,        # (N_ens, K)
        perturbation_scale: float,
        W_stem: jax.Array,           # stem conv kernel
        W_s1c1: jax.Array,           # stage1 block0 conv1 kernel (1×1)
        X_sub: jax.Array,            # calibration images
        inf_batch_size: int = 64,
    ):
        self.base_model        = base_model
        self.v_opts            = v_opts
        self.sigmas            = jnp.array(sigmas)
        self.z_coeffs          = z_coeffs
        self.perturbation_scale = perturbation_scale
        self.W_stem            = W_stem
        self.W_s1c1            = W_s1c1
        self.X_sub             = X_sub
        self.inf_batch_size    = inf_batch_size
        self.stem_size         = W_stem.size
        self._precompute_bn_refit()

    def _perturb_coeffs(self) -> np.ndarray:
        safe_sig = np.array(self.sigmas) + 1e-6
        coeffs   = self.z_coeffs / safe_sig
        norms    = np.linalg.norm(coeffs, axis=1, keepdims=True) + 1e-12
        return (coeffs / norms) * self.perturbation_scale

    def _precompute_bn_refit(self):
        coeffs_all = self._perturb_coeffs()
        dW_all     = coeffs_all @ np.array(self.v_opts)  # (N_ens, D_stem+D_s1c1)

        # -- Stem targets --
        raw_stem_orig  = self.base_model.stem.conv(self.X_sub)
        z_stem_orig    = self.base_model.stem.bn(raw_stem_orig, use_running_average=True)
        h_stem_orig    = jax.nn.relu(z_stem_orig)   # input to stage1

        # -- Stage1-block0-conv1 targets --
        blk0  = self.base_model.stage1[0]
        raw_s1c1_orig = blk0.conv1.conv(h_stem_orig)
        z_s1c1_orig   = blk0.conv1.bn(raw_s1c1_orig, use_running_average=True)

        stem_gammas, stem_betas   = [], []
        s1c1_gammas, s1c1_betas   = [], []

        for i in range(len(self.z_coeffs)):
            p = jnp.array(dW_all[i])
            dW_stem = p[:self.stem_size].reshape(self.W_stem.shape)
            dW_s1c1 = p[self.stem_size:].reshape(self.W_s1c1.shape)

            # -- Stem BN refit --
            W_stem_new = self.W_stem + dW_stem
            raw_stem_pert = jax.lax.conv_general_dilated(
                lhs=self.X_sub,
                rhs=W_stem_new.transpose(3, 2, 0, 1),
                window_strides=(1, 1), padding='SAME',
                dimension_numbers=('NHWC', 'OIHW', 'NHWC'))
            g_stem, b_stem = _bn_refit_channel_wise(raw_stem_pert, z_stem_orig)
            stem_gammas.append(g_stem)
            stem_betas.append(b_stem)

            # Corrected stem output (used as input to stage1-b0-conv1)
            h_stem_corrected = jax.nn.relu(g_stem * raw_stem_pert + b_stem)

            # -- Stage1 block0 conv1 BN refit --
            W_s1c1_new = self.W_s1c1 + dW_s1c1
            raw_s1c1_pert = jax.lax.conv_general_dilated(
                lhs=h_stem_corrected,
                rhs=W_s1c1_new.transpose(3, 2, 0, 1),
                window_strides=(1, 1), padding='SAME',
                dimension_numbers=('NHWC', 'OIHW', 'NHWC'))
            g_s1c1, b_s1c1 = _bn_refit_channel_wise(raw_s1c1_pert, z_s1c1_orig)
            s1c1_gammas.append(g_s1c1)
            s1c1_betas.append(b_s1c1)

        self.stem_gammas = jnp.stack(stem_gammas, axis=0)
        self.stem_betas  = jnp.stack(stem_betas,  axis=0)
        self.s1c1_gammas = jnp.stack(s1c1_gammas, axis=0)
        self.s1c1_betas  = jnp.stack(s1c1_betas,  axis=0)

    def _forward_member(self, x: jax.Array, i: int) -> jax.Array:
        coeffs_all = self._perturb_coeffs()
        p          = jnp.array((coeffs_all @ np.array(self.v_opts))[i])
        dW_stem    = p[:self.stem_size].reshape(self.W_stem.shape)
        dW_s1c1    = p[self.stem_size:].reshape(self.W_s1c1.shape)

        W_stem_new  = self.W_stem  + dW_stem
        W_s1c1_new  = self.W_s1c1 + dW_s1c1

        # --- Perturbed stem ---
        raw_stem = jax.lax.conv_general_dilated(
            lhs=x, rhs=W_stem_new.transpose(3, 2, 0, 1),
            window_strides=(1, 1), padding='SAME',
            dimension_numbers=('NHWC', 'OIHW', 'NHWC'))
        h = jax.nn.relu(self.stem_gammas[i] * raw_stem + self.stem_betas[i])

        # --- Stage1 block0: patched conv1 ---
        blk0     = self.base_model.stage1[0]
        identity = h
        # perturbed conv1 (1×1)
        raw_c1 = jax.lax.conv_general_dilated(
            lhs=h, rhs=W_s1c1_new.transpose(3, 2, 0, 1),
            window_strides=(1, 1), padding='SAME',
            dimension_numbers=('NHWC', 'OIHW', 'NHWC'))
        out = jax.nn.relu(self.s1c1_gammas[i] * raw_c1 + self.s1c1_betas[i])
        # remaining convs in block0 (conv2, conv3) with original weights
        out = jax.nn.relu(blk0.conv2(out, use_running_average=True))
        out = blk0.conv3(out, use_running_average=True)
        if blk0.downsample is not None:
            identity = blk0.downsample(h, use_running_average=True)
        out = jax.nn.relu(out + identity)

        # --- Remaining stage1 blocks + stages 2-4 + head ---
        for blk in list(self.base_model.stage1)[1:]:
            out = blk(out, use_running_average=True)
        for blk in self.base_model.stage2:
            out = blk(out, use_running_average=True)
        for blk in self.base_model.stage3:
            out = blk(out, use_running_average=True)
        for blk in self.base_model.stage4:
            out = blk(out, use_running_average=True)
        out = jnp.mean(out, axis=(1, 2))
        return self.base_model.fc(out)

    def predict(self, x: jax.Array) -> jax.Array:
        ys = [self._forward_member(x, i) for i in range(len(self.z_coeffs))]
        return jnp.stack(ys, axis=0)

    def predict_one(self, x: jax.Array, idx: int) -> jax.Array:
        return self._forward_member(x, idx)
