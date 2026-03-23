import jax
import jax.numpy as jnp
from flax import nnx
from typing import List, Tuple, Any
import numpy as np
import math

from jaxtyping_bridge import Array, Float, f32

_BATCH_ANY = "batch ..."
_ENS_BATCH_ANY = "ens batch ..."


from models import TransitionModel

def evaluate_tail_from_preact(
    base_model: Any,
    preact: Float[Array, _BATCH_ANY],
    current_layer_idx: int,
    activation: Any = nnx.relu
) -> Float[Array, _BATCH_ANY]:
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

class PJSVDEnsemble:
    """
    Unified, memory-efficient PJSVD ensemble.

    Handles:
        - Single-layer or Multi-layer perturbations.
        - Affine correction (scaling) or Least-squares correction.
        - Optional BatchNorm refit (for ResNet-style models).
    """

    def __init__(
        self,
        base_model: Any,
        v_opts: jax.Array,
        sigmas: jax.Array,
        z_coeffs: np.ndarray,
        perturbation_scale: float,
        X_sub: jax.Array,
        layers: List[str] = ["l1"],  # Layer names following our convention
        correction_mode: str = "affine",  # "affine", "least_squares", or "bn_refit"
        sigma_sq_weights: bool = False,
        activation: Any = None,
        # Original parameters for the layers being perturbed
        layer_params: dict = None,
        # Targets for correction (e.g. following layer weights/biases or original stats)
        correction_params: dict = None,
        **kwargs
    ):
        self.base_model = base_model
        self.v_opts = v_opts
        self.sigmas = jnp.array(sigmas)
        self.z_coeffs = z_coeffs
        self.perturbation_scale = perturbation_scale
        self.X_sub = X_sub
        self.layers = layers
        self.correction_mode = correction_mode
        self.sigma_sq_weights = sigma_sq_weights
        self.activation = activation if activation is not None else getattr(base_model, 'activation', nnx.relu)
        self.kwargs = kwargs

        # Store layer parameters (W, b) for all perturbed layers
        self.layer_params = layer_params or {}
        # Store correction parameters (e.g., mu_old, std_old or next layer params)
        self.correction_params = correction_params or {}

        self._precompute_corrections()

    def _get_coeffs_all(self):
        safe_sigmas = self.sigmas + 1e-6
        if self.sigma_sq_weights:
            coeffs_all = self.z_coeffs / np.array(safe_sigmas ** 2)
        else:
            coeffs_all = self.z_coeffs / np.array(safe_sigmas)
        norms = np.linalg.norm(coeffs_all, axis=1, keepdims=True) + 1e-12
        return (coeffs_all / norms) * self.perturbation_scale

    def _precompute_corrections(self):
        coeffs_all = self._get_coeffs_all()
        # dW for all members: (N, K) @ (K, D_flat) -> (N, D_flat)
        dp_all = coeffs_all @ np.array(self.v_opts)

        if self.correction_mode == "bn_refit":
            self._precompute_bn_refit(dp_all)
        elif self.correction_mode == "least_squares":
            self._precompute_least_squares(dp_all)
        else: # Default: affine
            self._precompute_affine(dp_all)

    def _precompute_affine(self, dp_all):
        # We assume for affine correction we are matching mu_old, std_old
        # and adjusting the bias/scale of the layer IMMEDIATELY FOLLOWING the perturbed one(s).
        mu_old = self.correction_params.get("mu_old")
        std_old = self.correction_params.get("std_old")
        
        # We need the next layer params to adjust them
        # For multi-layer, we assume the next layer is the one AFTER the last perturbed layer.
        next_w_orig = self.correction_params.get("next_w")
        next_b_orig = self.correction_params.get("next_b")

        scale_factors = []
        next_b_news = []

        for i in range(len(self.z_coeffs)):
            # Apply perturbation(s)
            p = dp_all[i]
            perturbed_vals = self._apply_perturbations(self.X_sub, p)
            h_new = perturbed_vals[-1] # The activation of the last perturbed layer

            mu_new = jnp.mean(h_new, axis=0)
            std_new = jnp.std(h_new, axis=0)
            scale_factor = std_old / (std_new + 1e-6)
            
            # Adjusted bias such that: h_new * W_new + b_new has same mean as h_old * W_old + b_old
            # W_new = W_old * scale_factor
            # b_new = b_old + mu_old @ W_old - mu_new @ W_new
            b_new = next_b_orig + jnp.dot(mu_old, next_w_orig) - jnp.dot(mu_new, next_w_orig * scale_factor[:, None])

            scale_factors.append(scale_factor)
            next_b_news.append(b_new)

        self.scale_factors = jnp.stack(scale_factors, axis=0)
        self.next_b_news = jnp.stack(next_b_news, axis=0)

    def _precompute_least_squares(self, dp_all):
        # Target is the original unperturbed activations/outputs of the next layer
        Z = self.correction_params.get("target_act")
        N_samples = Z.shape[0]

        next_w_news = []
        next_b_news = []

        for i in range(len(self.z_coeffs)):
            p = dp_all[i]
            perturbed_vals = self._apply_perturbations(self.X_sub, p)
            h_new = perturbed_vals[-1]

            # Solve least squares: [h_new, 1] @ W_aug ≈ Z
            ones = jnp.ones((N_samples, 1), dtype=h_new.dtype)
            h_new_aug = jnp.concatenate([h_new, ones], axis=-1)
            W_aug, _, _, _ = jnp.linalg.lstsq(h_new_aug, Z, rcond=None)

            next_w_news.append(W_aug[:-1, :])
            next_b_news.append(W_aug[-1, :])

        self.next_w_news = jnp.stack(next_w_news, axis=0)
        self.next_b_news = jnp.stack(next_b_news, axis=0)

    def _precompute_bn_refit(self, dp_all):
        # Specialized ResNet logic
        # For each perturbed conv, we refit the FOLLOWING BN layer
        # Currently supports stem only or stem + stage1.b0.c1
        # self.correction_params["targets"] = list of original BN outputs
        targets = self.correction_params.get("targets")
        
        bn_params = [] # List of (gamma, beta) per perturbed layer per member

        for i in range(len(self.z_coeffs)):
            p = dp_all[i]
            
            # This is complex because it's model-specific. 
            # We'll delegate to a model-specific helper or implement for ResNet here.
            if hasattr(self.base_model, "stem"):
                member_bn_params = self._resnet_bn_refit_member(p, targets)
                bn_params.append(member_bn_params)
        
        # Transpose to get [perturbed_layer][member]
        self.bn_refits = [jnp.stack([m[l] for m in bn_params]) for l in range(len(self.layers))]

    def _resnet_bn_refit_member(self, p, targets):
        # Helper for ResNet stem + stage1 refit
        results = []
        curr_x = self.X_sub
        
        offset = 0
        for l_idx, l_name in enumerate(self.layers):
            w_orig = self.layer_params[l_name]["W"]
            w_shape = w_orig.shape
            w_size = w_orig.size
            dw = p[offset:offset+w_size].reshape(w_shape)
            w_new = w_orig + dw
            offset += w_size
            
            # Manual conv
            raw_pert = jax.lax.conv_general_dilated(
                lhs=curr_x, rhs=w_new.transpose(3, 2, 0, 1),
                window_strides=(1, 1), padding='SAME',
                dimension_numbers=('NHWC', 'OIHW', 'NHWC'))
            
            g, b = _bn_refit_channel_wise(raw_pert, targets[l_idx])
            results.append({"gamma": g, "beta": b})
            
            # Update curr_x for next layer if any
            if l_idx < len(self.layers) - 1:
                curr_x = jax.nn.relu(g * raw_pert + b)
        return results

    def _apply_perturbations(self, x, p):
        # Applies perturbations to sequential layers
        results = []
        curr_h = x
        offset = 0
        for l_idx, l_name in enumerate(self.layers):
            params = self.layer_params.get(l_name, {})
            w_orig = params.get("W")
            b_orig = params.get("b")
            
            dw = p[offset:offset+w_orig.size].reshape(w_orig.shape)
            offset += w_orig.size
            w_new = w_orig + dw
            
            # Forward pass through this perturbed layer
            z = curr_h @ w_new
            if b_orig is not None:
                z = z + b_orig
            curr_h = self.activation(z)
            results.append(curr_h)
        return results

    def _forward_member(self, x, i, dp_all):
        p = dp_all[i]
        
        if self.correction_mode == "bn_refit":
            return self._forward_bn_refit(x, i, p)
        
        perturbed_vals = self._apply_perturbations(x, p)
        h_last_perturbed = perturbed_vals[-1]
        
        if self.correction_mode == "least_squares":
            w_next = self.next_w_news[i]
            b_next = self.next_b_news[i]
        else: # affine
            w_next = self.correction_params["next_w"] * self.scale_factors[i][:, None]
            b_next = self.next_b_news[i]
            
        z_next = h_last_perturbed @ w_next + b_next
        return evaluate_tail_from_preact(self.base_model, z_next, current_layer_idx=len(self.layers), activation=self.activation)

    def _forward_bn_refit(self, x, i, p):
        # ResNet specific forward
        if len(self.layers) == 1:
            w_stem_orig = self.layer_params["l1"]["W"]
            dw = p.reshape(w_stem_orig.shape)
            w_new = w_stem_orig + dw
            raw_pert = jax.lax.conv_general_dilated(
                lhs=x, rhs=w_new.transpose(3, 2, 0, 1),
                window_strides=(1, 1), padding='SAME',
                dimension_numbers=('NHWC', 'OIHW', 'NHWC'))
            g, b = self.bn_refits[0][i]["gamma"], self.bn_refits[0][i]["beta"]
            h = jax.nn.relu(g * raw_pert + b)
            return self.base_model.forward_from_stem_out(h, use_running_average=True)
        else:
            # Multi-layer BN refit (Stem + Stage1)
            w1_orig = self.layer_params["l1"]["W"]
            w2_orig = self.layer_params["l2"]["W"]
            dw1 = p[:w1_orig.size].reshape(w1_orig.shape)
            dw2 = p[w1_orig.size:].reshape(w2_orig.shape)
            
            # Stem
            raw_st = jax.lax.conv_general_dilated(
                lhs=x, rhs=(w1_orig+dw1).transpose(3, 2, 0, 1),
                window_strides=(1, 1), padding='SAME',
                dimension_numbers=('NHWC', 'OIHW', 'NHWC'))
            h_st = jax.nn.relu(self.bn_refits[0][i]["gamma"] * raw_st + self.bn_refits[0][i]["beta"])
            
            # Stage1 block0 conv1
            blk0 = self.base_model.stage1[0]
            raw_c1 = jax.lax.conv_general_dilated(
                lhs=h_st, rhs=(w2_orig+dw2).transpose(3, 2, 0, 1),
                window_strides=(1, 1), padding='SAME',
                dimension_numbers=('NHWC', 'OIHW', 'NHWC'))
            out = jax.nn.relu(self.bn_refits[1][i]["gamma"] * raw_c1 + self.bn_refits[1][i]["beta"])
            
            # Tail of block0
            out = jax.nn.relu(blk0.conv2(out, use_running_average=True))
            out = blk0.conv3(out, use_running_average=True)
            identity = h_st
            if blk0.downsample is not None:
                identity = blk0.downsample(h_st, use_running_average=True)
            out = jax.nn.relu(out + identity)
            
            # Remainder of ResNet
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
        coeffs_all = self._get_coeffs_all()
        dp_all = coeffs_all @ np.array(self.v_opts)
        
        ys = []
        for i in range(len(self.z_coeffs)):
            ys.append(self._forward_member(x, i, dp_all))
        return jnp.stack(ys, axis=0)

    def predict_one(self, x: jax.Array, idx: int) -> jax.Array:
        coeffs_all = self._get_coeffs_all()
        dp_all = coeffs_all @ np.array(self.v_opts)
        return self._forward_member(x, idx, dp_all)







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

    def predict(self, x: Float[Array, "batch ..."]) -> Float[Array, "ens batch ..."]:
        """Returns (M * S, N, output_dim) stacked predictions."""
        all_preds = [ens.predict(x) for ens in self.pjsvd_ensembles]
        return jnp.concatenate(all_preds, axis=0)

    def predict_one(self, x: Float[Array, "batch ..."], idx: int) -> Float[Array, "batch ..."]:
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

    def predict(self, x: Float[Array, "batch ..."]) -> Float[Array, "ens batch ..."]:
        ys = []
        for w1, b1, w2, b2 in self.perturbations:
            ys.append(self.manual_forward(x, w1, b1, w2, b2))
        return jnp.stack(ys, axis=0)

    def predict_one(self, x: Float[Array, "batch ..."], idx: int) -> Float[Array, "batch ..."]:
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

    def predict(self, x: Float[Array, "batch ..."]) -> Float[Array, "ens batch ..."]:
        ys = []
        for w1, b1, w2, b2, w3, b3 in self.perturbations:
            ys.append(self.manual_forward(x, w1, b1, w2, b2, w3, b3))
        return jnp.stack(ys, axis=0)

    def predict_one(self, x: Float[Array, "batch ..."], idx: int) -> Float[Array, "batch ..."]:
        w1, b1, w2, b2, w3, b3 = self.perturbations[idx]
        return self.manual_forward(x, w1, b1, w2, b2, w3, b3)


class StandardEnsemble:
    def __init__(self, models: List[nnx.Module]):
        self.models = models

    def predict(self, x: Float[Array, "batch ..."]) -> Float[Array, "ens batch ..."]:
        ys = []
        for i, model in enumerate(self.models):
            out = model(x)
            if isinstance(out, tuple) and len(out) == 2:
                # Probabilistic model: (mean, var)
                # We return a "representative sample" for each member
                # By adding Gaussian noise with the predicted variance,
                # the across-ensemble variance will correctly match the mixture variance:
                # Var_total = E[Var_aleatoric] + Var[E_epistemic]
                rng = jax.random.PRNGKey(i) # Use a deterministic key based on the member index
                mean, var = out
                eps = jax.random.normal(rng, mean.shape)
                ys.append(mean + jnp.sqrt(var) * eps)
            else:
                # Standard point-estimate model
                ys.append(out)
        return jnp.stack(ys, axis=0)

    def predict_one(self, x: Float[Array, "batch ..."], idx: int) -> Float[Array, "batch ..."]:
        out = self.models[idx](x)
        if isinstance(out, tuple) and len(out) == 2:
            mean, var = out
            rng = jax.random.PRNGKey(idx)
            return mean + jnp.sqrt(var) * jax.random.normal(rng, mean.shape)
        return out


class MCDropoutEnsemble:
    def __init__(self, model: TransitionModel, n_models: int):
        self.model = model
        self.n_models = n_models

    def predict(self, x: Float[Array, "batch ..."]) -> Float[Array, "ens batch ..."]:
        ys = []
        for _ in range(self.n_models):
            ys.append(self.model(x, deterministic=False))
        return jnp.stack(ys, axis=0)

    def predict_one(self, x: Float[Array, "batch ..."], idx: int) -> Float[Array, "batch ..."]:
        return self.model(x, deterministic=False)


class SWAGEnsemble:
    def __init__(self, model: TransitionModel, swag_mean: nnx.State, swag_var: nnx.State, n_models: int):
        self.model = nnx.clone(model)
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

    def predict(self, x: Float[Array, "batch ..."]) -> Float[Array, "ens batch ..."]:
        ys = []
        for i in range(self.n_models):
            sampled_m = self._sample_model()
            out = sampled_m(x)
            if isinstance(out, tuple) and len(out) == 2:
                mean, var = out
                rng = jax.random.PRNGKey(i)
                ys.append(mean + jnp.sqrt(var) * jax.random.normal(rng, mean.shape))
            else:
                ys.append(out)
        return jnp.stack(ys, axis=0)

    def predict_one(self, x: Float[Array, "batch ..."], idx: int) -> Float[Array, "batch ..."]:
        out = self._sample_model()(x)
        if isinstance(out, tuple) and len(out) == 2:
            mean, var = out
            rng = jax.random.PRNGKey(idx)
            return mean + jnp.sqrt(var) * jax.random.normal(rng, mean.shape)
        return out


class LaplaceEnsemble:
    def __init__(self, model: TransitionModel, kfac_factors: dict, prior_precision: float, n_models: int, data_size: int):
        self.model = nnx.clone(model)
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

    def predict(self, x: Float[Array, "batch ..."]) -> Float[Array, "ens batch ..."]:
        ys = []
        for _ in range(self.n_models):
            sampled_m, _ = self._sample_model()
            ys.append(sampled_m(x))
        return jnp.stack(ys, axis=0)

    def predict_one(self, x: Float[Array, "batch ..."], idx: int) -> Float[Array, "batch ..."]:
        sampled_m, _ = self._sample_model()
        return sampled_m(x)

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
        self.base_model = nnx.clone(base_model)
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

    def predict(self, x: Float[Array, "batch ..."]) -> Float[Array, "ens batch ..."]:
        ys = []
        for i in range(self.n_samples):
            sampled_m, _ = self._sample_model(i)
            out = sampled_m(x)
            if isinstance(out, tuple) and len(out) == 2:
                # Probabilistic model: (mean, var)
                mean, var = out
                rng = jax.random.PRNGKey(i)
                eps = jax.random.normal(rng, mean.shape)
                ys.append(mean + jnp.sqrt(var) * eps)
            else:
                ys.append(out)
        return jnp.stack(ys, axis=0)

    def predict_one(self, x: Float[Array, "batch ..."], idx: int) -> Float[Array, "batch ..."]:
        sampled_m, _ = self._sample_model(idx)
        out = sampled_m(x)
        if isinstance(out, tuple) and len(out) == 2:
            mean, var = out
            rng = jax.random.PRNGKey(idx)
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



