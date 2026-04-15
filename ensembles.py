import jax
import jax.numpy as jnp
import jax.flatten_util
from flax import nnx
from dataclasses import dataclass
from typing import List, Tuple, Any, Callable, Optional
import numpy as np
import math

from jaxtyping import Array, Float
from models import TransitionModel

_BATCH_ANY = "batch ..."
_ENS_BATCH_ANY = "ens batch ..."


def _parse_member_radius_values(values: Optional[Any]) -> Optional[np.ndarray]:
    if values is None:
        return None
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return None
    return arr.reshape(-1)


def _sample_member_radius_multipliers(
    n_members: int,
    distribution: str = "fixed",
    *,
    std: float = 0.0,
    values: Optional[Any] = None,
    seed: int = 0,
) -> np.ndarray:
    distribution = str(distribution).lower()
    if n_members <= 0:
        return np.zeros((0,), dtype=np.float64)

    if distribution == "fixed":
        return np.ones((n_members,), dtype=np.float64)

    rng = np.random.RandomState(seed)
    if distribution == "lognormal":
        sigma = float(std)
        if sigma <= 0.0:
            return np.ones((n_members,), dtype=np.float64)
        mu = -0.5 * sigma * sigma
        multipliers = rng.lognormal(mean=mu, sigma=sigma, size=n_members)
    elif distribution == "two_point":
        parsed_values = _parse_member_radius_values(values)
        if parsed_values is not None:
            if parsed_values.size != 2:
                raise ValueError("member_radius_values must contain exactly two positive values for two_point")
            multipliers = rng.choice(parsed_values, size=n_members, replace=True)
        else:
            delta = float(std)
            if delta <= 0.0:
                return np.ones((n_members,), dtype=np.float64)
            if delta >= 1.0:
                raise ValueError("member_radius_std must be < 1.0 for two_point when member_radius_values is not provided")
            multipliers = rng.choice(np.array([1.0 - delta, 1.0 + delta], dtype=np.float64), size=n_members, replace=True)
    else:
        raise ValueError(f"Unsupported member_radius_distribution: {distribution}")

    if np.any(multipliers <= 0.0):
        raise ValueError("All member radius multipliers must be positive")

    return multipliers / np.mean(multipliers)


def _scale_coefficients_with_member_radii(
    z_coeffs: np.ndarray,
    sigmas: np.ndarray,
    perturbation_scale: float,
    sigma_sq_weights: bool,
    member_radius_multipliers: np.ndarray,
) -> np.ndarray:
    safe_sigmas = np.asarray(sigmas, dtype=np.float64) + 1e-6
    z_coeffs = np.asarray(z_coeffs, dtype=np.float64)
    if sigma_sq_weights:
        coeffs_all = z_coeffs / (safe_sigmas ** 2)
    else:
        coeffs_all = z_coeffs / safe_sigmas
    norms = np.linalg.norm(coeffs_all, axis=1, keepdims=True) + 1e-12
    target_radii = float(perturbation_scale) * np.asarray(member_radius_multipliers, dtype=np.float64).reshape(-1, 1)
    return (coeffs_all / norms) * target_radii


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
        # Handle ProbabilisticRegressionModel dual-head output
        if hasattr(base_model, 'mean_layer'):
            h = activation(h)
            mean = h @ base_model.mean_layer.kernel.get_value() + base_model.mean_layer.bias.get_value()
            var_logits = h @ base_model.var_layer.kernel.get_value() + base_model.var_layer.bias.get_value()
            var = jax.nn.softplus(var_logits) + 1e-6
            return mean, var
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

def _ls_or_ridge_solve(h_aug: jax.Array, target: jax.Array, lambda_reg: float) -> jax.Array:
    """Solve h_aug @ W_aug ≈ target.

    λ=0: pseudoinverse via jnp.linalg.lstsq (handles rank-deficient h_aug with a
    minimum-norm solution; exactly the legacy behavior).
    λ>0: ridge via solve((h_augᵀ h_aug + λI), h_augᵀ target). Regularizes all
    rows including the bias column, matching pnc._ridge_regression_solve.
    """
    if lambda_reg > 0.0:
        D = h_aug.shape[1]
        H = h_aug.T @ h_aug + lambda_reg * jnp.eye(D, dtype=h_aug.dtype)
        rhs = h_aug.T @ target
        return jnp.linalg.solve(H, rhs)
    W_aug, _, _, _ = jnp.linalg.lstsq(h_aug, target, rcond=None)
    return W_aug


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
        member_radius_distribution: str = "fixed",
        member_radius_std: float = 0.0,
        member_radius_values: Optional[Any] = None,
        member_radius_seed: int = 0,
        activation: Any = None,
        # Original parameters for the layers being perturbed
        layer_params: dict = None,
        # Targets for correction (e.g. following layer weights/biases or original stats)
        correction_params: dict = None,
        # Tikhonov regularization for the least-squares correction (MLP path only).
        # 0.0 preserves legacy behavior (pseudoinverse via jnp.linalg.lstsq), which
        # is well-defined for rank-deficient h_aug but amplifies corrections when
        # the perturbed hidden activations lie in a near-singular subspace (the
        # Low family's ill-conditioned-on-purpose case).
        lambda_reg: float = 0.0,
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
        self.member_radius_distribution = member_radius_distribution
        self.member_radius_std = float(member_radius_std)
        self.member_radius_values = _parse_member_radius_values(member_radius_values)
        self.member_radius_seed = int(member_radius_seed)
        self.member_radius_multipliers = _sample_member_radius_multipliers(
            len(self.z_coeffs),
            distribution=self.member_radius_distribution,
            std=self.member_radius_std,
            values=self.member_radius_values,
            seed=self.member_radius_seed,
        )
        self.activation = activation if activation is not None else getattr(base_model, 'activation', nnx.relu)
        self.kwargs = kwargs
        self.tail_is_hidden = bool(kwargs.get("tail_is_hidden", False))

        # Store layer parameters (W, b) for all perturbed layers
        self.layer_params = layer_params or {}
        # Targets for correction (e.g. following layer weights/biases or original stats)
        self.correction_params = correction_params or {}
        self.lambda_reg = float(lambda_reg)
        # Per-layer independent specs for multi-layer PnC (like CIFAR MultiBlock).
        # Each entry: {"v_opts": array(K, D), "sigmas": array(K,), "W_shape": tuple}
        # When set, z_coeffs shape must be (n_members, n_layers, K).
        self.layer_specs = kwargs.get("layer_specs", None)

        self._n_members = self.z_coeffs.shape[0] if hasattr(self.z_coeffs, 'shape') else len(self.z_coeffs)
        self._precompute_corrections()

    def set_perturbation_size(self, size: float):
        self.perturbation_scale = size
        self._precompute_corrections()

    def _get_coeffs_all(self):
        return _scale_coefficients_with_member_radii(
            self.z_coeffs,
            np.array(self.sigmas),
            self.perturbation_scale,
            self.sigma_sq_weights,
            self.member_radius_multipliers,
        )

    @property
    def member_radii(self) -> np.ndarray:
        return self.perturbation_scale * self.member_radius_multipliers

    def _precompute_corrections(self):
        # Per-layer independent multi-layer: skip joint coefficient scaling.
        # Both "least_squares" and "none" enter the sequential path; "none" just
        # uses the original next-layer W/b as the (non-)correction so the
        # perturbation propagates through without compensation.
        if self.layer_specs is not None and self.correction_mode in ("least_squares", "none"):
            self._precompute_sequential_ls(dp_all=None)
            return

        coeffs_all = self._get_coeffs_all()
        # dW for all members: (N, K) @ (K, D_flat) -> (N, D_flat)
        dp_all = coeffs_all @ np.array(self.v_opts)

        if self.correction_mode == "bn_refit":
            self._precompute_bn_refit(dp_all)
        elif self.correction_mode == "least_squares":
            self._precompute_least_squares(dp_all)
        elif self.correction_mode == "none":
            self._precompute_none(dp_all)
        else: # Default: affine
            self._precompute_affine(dp_all)

    def _precompute_none(self, dp_all):
        """No-correction mode: perturb but use the ORIGINAL next-layer W/b
        instead of any compensation. Stores per-member original weights so the
        forward path can reuse the same scaffolding as the LS path.
        """
        n_members = len(self.z_coeffs)
        # The perturbed weights are stored as next_w_news/next_b_news for the
        # forward path; here we just broadcast the ORIGINAL next-layer params.
        w_next_orig = self.correction_params["next_w"]
        b_next_orig = self.correction_params.get("next_b")
        if b_next_orig is None:
            b_next_orig = jnp.zeros(w_next_orig.shape[-1], dtype=w_next_orig.dtype)
        self.next_w_news = [w_next_orig for _ in range(n_members)]
        self.next_b_news = [b_next_orig for _ in range(n_members)]
        # dp_all is needed by _forward_member to apply the perturbation
        self.dp_all = dp_all
        self.scale_factors = [jnp.ones(w_next_orig.shape[0], dtype=w_next_orig.dtype) for _ in range(n_members)]

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
        if len(self.layers) > 1:
            # Multi-layer: sequential correction at each layer interface
            self._precompute_sequential_ls(dp_all)
            return

        # Single layer: one correction at the next-layer interface
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
            W_aug = _ls_or_ridge_solve(h_new_aug, Z, self.lambda_reg)

            next_w_news.append(W_aug[:-1, :])
            next_b_news.append(W_aug[-1, :])

        self.next_w_news = jnp.stack(next_w_news, axis=0)
        self.next_b_news = jnp.stack(next_b_news, axis=0)

    def _precompute_sequential_ls(self, dp_all):
        """Sequential layerwise correction for multi-layer PnC.

        Two modes depending on whether per-layer independent specs are provided:

        1. **Per-layer independent** (``self.layer_specs`` is set): each layer has
           its own direction set, sigmas, and z_coeffs.  Perturbation at layer j
           is computed from ``layer_specs[j]`` independently, then corrected at the
           j→(j+1) interface via LS.  This is the intended algorithm behaviour and
           matches the CIFAR MultiBlockPnCEnsemble approach.

        2. **Joint fallback** (``self.layer_specs`` is None): perturbations come
           from a single joint direction vector split across layers.  Each layer's
           segment is applied and corrected sequentially, but the segments are
           correlated because they come from the same direction.
        """
        n_layers = len(self.layers)
        n_members = len(self.z_coeffs) if self.layer_specs is None else self.z_coeffs.shape[0]

        # 1. Compute original pre-activations at each interface (correction targets)
        h = self.X_sub
        orig_preacts = []
        for l_name in self.layers:
            W = self.layer_params[l_name]["W"]
            b = self.layer_params[l_name]["b"]
            z = h @ W + b
            orig_preacts.append(z)
            h = self.activation(z)
        final_target = self.correction_params.get("target_act")

        # 2. Compute per-layer perturbation dWs for every member
        if self.layer_specs is not None:
            # Per-layer independent: each layer has own v_opts/sigmas/z_coeffs
            all_dWs = []  # list of length n_members, each a list of n_layers dW arrays
            for i in range(n_members):
                member_dWs = []
                for j, spec in enumerate(self.layer_specs):
                    z_row = self.z_coeffs[i, j]  # (K,)
                    coeffs = _scale_coefficients_with_member_radii(
                        z_row[None, :],
                        np.asarray(spec["sigmas"]),
                        self.perturbation_scale,
                        self.sigma_sq_weights,
                        np.asarray([self.member_radius_multipliers[i]]),
                    )[0]
                    dp = coeffs @ np.asarray(spec["v_opts"])
                    member_dWs.append(jnp.array(dp.reshape(spec["W_shape"])))
                all_dWs.append(member_dWs)
        else:
            # Joint fallback: split single dp vector per layer
            all_dWs = []
            for i in range(n_members):
                p = dp_all[i]
                dWs = []
                offset = 0
                for l_name in self.layers:
                    W = self.layer_params[l_name]["W"]
                    dw = p[offset:offset + W.size].reshape(W.shape)
                    dWs.append(dw)
                    offset += W.size
                all_dWs.append(dWs)

        # 3. Compute original hidden states needed for correction targets.
        #    For each perturbed layer j, the correction at the NEXT layer targets
        #    the original pre-activation there: orig_h[j] @ W_next + b_next.
        #    We need the original post-activation after each perturbed layer's
        #    correction pair to feed into the next block.
        all_orig_h = []  # post-activation through entire original model at each layer
        h_orig = self.X_sub
        n_model_layers = max(
            int(l_name.replace("l", "")) for l_name in self.layer_params
        )  # highest perturbed layer index
        # Walk through the full model to collect all intermediate activations
        base_model = self.base_model
        h_walk = self.X_sub
        if hasattr(base_model, 'layers'):
            for idx in range(len(base_model.layers)):
                z = h_walk @ base_model.layers[idx].kernel.get_value() + base_model.layers[idx].bias.get_value()
                all_orig_h.append(h_walk)  # input to this layer
                h_walk = self.activation(z)
            all_orig_h.append(h_walk)  # final post-activation

        # 4. Sequential perturb-correct per member
        #    Each perturbed layer j is paired with correction layer j+1.
        #    The correction target is: all_orig_h[j+1] @ W_{j+1} + b_{j+1}
        #    (the original pre-activation at the correction layer).
        per_block_w = [[] for _ in range(n_layers)]
        per_block_b = [[] for _ in range(n_layers)]

        for i in range(n_members):
            dWs = all_dWs[i]
            h = self.X_sub

            for j in range(n_layers):
                l_name = self.layers[j]
                layer_idx = int(l_name.replace("l", "")) - 1  # 0-based model layer index
                corr_idx = layer_idx + 1  # correction layer is the next one
                W_pert = self.layer_params[l_name]["W"]
                b_pert = self.layer_params[l_name]["b"]

                # Perturb this layer
                h_pert = self.activation(h @ (W_pert + dWs[j]) + b_pert)

                # Correction target: original pre-activation at the correction layer
                if corr_idx < len(base_model.layers):
                    W_corr_orig = base_model.layers[corr_idx].kernel.get_value()
                    b_corr_orig = base_model.layers[corr_idx].bias.get_value()
                    target = all_orig_h[corr_idx] @ W_corr_orig + b_corr_orig
                else:
                    target = final_target

                if self.correction_mode == "none":
                    # No correction: use the ORIGINAL next-layer weights so
                    # the perturbation propagates through unchanged.
                    if corr_idx < len(base_model.layers):
                        W_corr = W_corr_orig
                        b_corr = b_corr_orig
                    else:
                        W_corr = jnp.eye(h_pert.shape[-1], dtype=h_pert.dtype)
                        b_corr = jnp.zeros(h_pert.shape[-1], dtype=h_pert.dtype)
                else:
                    # Solve LS: [h_pert, 1] @ W_aug ≈ target
                    ones = jnp.ones((h_pert.shape[0], 1), dtype=h_pert.dtype)
                    h_aug = jnp.concatenate([h_pert, ones], axis=-1)
                    W_aug = _ls_or_ridge_solve(h_aug, target, self.lambda_reg)
                    W_corr = W_aug[:-1, :]
                    b_corr = W_aug[-1, :]

                per_block_w[j].append(W_corr)
                per_block_b[j].append(b_corr)

                # Apply correction and move to next block's input
                h = self.activation(h_pert @ W_corr + b_corr)

        self.seq_w_effs = [jnp.stack(ws, axis=0) for ws in per_block_w]
        self.seq_b_effs = [jnp.stack(bs, axis=0) for bs in per_block_b]
        # Store per-member per-layer perturbation dWs for the forward pass
        self.seq_dWs = [[dWs[j] for dWs in all_dWs] for j in range(n_layers)]
        self.seq_dWs = [jnp.stack(dws, axis=0) for dws in self.seq_dWs]

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
        self.bn_refits = [[m[idx] for m in bn_params] for idx in range(len(self.layers))]

    def _resnet_bn_refit_member(self, p, targets):
        # Helper for ResNet stem + stage1 refit
        results = []
        curr_x = self.X_sub
        
        offset = 0
        for l_idx, l_name in enumerate(self.layers):
            try:
                w_orig = self.layer_params[l_name]["W"]
            except KeyError:
                print(f"Layer {l_name} not found in layer_params. Available keys: {list(self.layer_params.keys())}")
                print(type(self.layer_params))
                raise
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
        # Sequential correction for multi-layer least_squares (or no-correction).
        if self.correction_mode in ("least_squares", "none") and len(self.layers) > 1:
            return self._forward_member_sequential(x, i, dp_all)

        p = dp_all[i]

        if self.correction_mode == "bn_refit":
            return self._forward_bn_refit(x, i, p)

        perturbed_vals = self._apply_perturbations(x, p)
        h_last_perturbed = perturbed_vals[-1]

        if self.correction_mode in ("least_squares", "none"):
            w_next = self.next_w_news[i]
            b_next = self.next_b_news[i]
        else: # affine
            w_next = self.correction_params["next_w"] * self.scale_factors[i][:, None]
            b_next = self.next_b_news[i]

        z_next = h_last_perturbed @ w_next + b_next
        if self.tail_is_hidden:
            mean = z_next @ self.base_model.mean_layer.kernel.get_value() + self.base_model.mean_layer.bias.get_value()
            var_logits = z_next @ self.base_model.var_layer.kernel.get_value() + self.base_model.var_layer.bias.get_value()
            var = jax.nn.softplus(var_logits) + 1e-6
            return mean, var
        return evaluate_tail_from_preact(self.base_model, z_next, current_layer_idx=len(self.layers), activation=self.activation)

    def _forward_member_sequential(self, x, i, dp_all):
        """Forward pass with alternating perturb/correct blocks.

        Each block j:
          1. Perturb layer j: h = act(h @ (W_j + dW_j) + b_j)
          2. Correct next layer: h = act(h @ W_corr + b_corr)
        After all blocks, evaluate remaining tail layers.
        """
        n_blocks = len(self.layers)
        h = x

        for j in range(n_blocks):
            l_name = self.layers[j]
            W_pert = self.layer_params[l_name]["W"]
            b_pert = self.layer_params[l_name]["b"]
            # Perturb
            h = self.activation(h @ (W_pert + self.seq_dWs[j][i]) + b_pert)
            # Correct
            h = self.activation(h @ self.seq_w_effs[j][i] + self.seq_b_effs[j][i])

        # h is post-activation past 2*n_blocks layers. Evaluate remaining tail.
        last_perturb_idx = int(self.layers[-1].replace("l", "")) - 1
        tail_start_idx = last_perturb_idx + 2  # 0-based index of first remaining layer

        if hasattr(self.base_model, 'layers'):
            for idx in range(tail_start_idx, len(self.base_model.layers)):
                layer = self.base_model.layers[idx]
                z = h @ layer.kernel.get_value() + layer.bias.get_value()
                if idx < len(self.base_model.layers) - 1:
                    h = self.activation(z)
                else:
                    h = z  # last layer: no activation
            # Handle probabilistic dual-head
            if hasattr(self.base_model, 'mean_layer'):
                h = self.activation(h)  # activate last hidden pre-act
                mean = h @ self.base_model.mean_layer.kernel.get_value() + self.base_model.mean_layer.bias.get_value()
                var_logits = h @ self.base_model.var_layer.kernel.get_value() + self.base_model.var_layer.bias.get_value()
                var = jax.nn.softplus(var_logits) + 1e-6
                return mean, var
            return h
        raise ValueError("Sequential forward requires model.layers interface")

    def _predict_member_intermediate_and_corrected(self, x, i, dp_all):
        # Sequential correction for multi-layer least_squares (or no-correction).
        if self.correction_mode in ("least_squares", "none") and len(self.layers) > 1:
            return self._predict_member_intermediate_and_corrected_sequential(x, i, dp_all)

        p = dp_all[i]
        perturbed_vals = self._apply_perturbations(x, p)
        h_last_perturbed = perturbed_vals[-1]

        if self.correction_mode in ("least_squares", "none"):
            w_next = self.next_w_news[i]
            b_next = self.next_b_news[i]
        else:
            w_next = self.correction_params["next_w"] * self.scale_factors[i][:, None]
            b_next = self.next_b_news[i]

        z_next = h_last_perturbed @ w_next + b_next
        return h_last_perturbed, z_next

    def _predict_member_intermediate_and_corrected_sequential(self, x, i, dp_all):
        """Intermediate + corrected output using alternating perturb/correct blocks."""
        n_blocks = len(self.layers)
        h = x

        for j in range(n_blocks):
            l_name = self.layers[j]
            W_pert = self.layer_params[l_name]["W"]
            b_pert = self.layer_params[l_name]["b"]
            h = self.activation(h @ (W_pert + self.seq_dWs[j][i]) + b_pert)
            z_corr = h @ self.seq_w_effs[j][i] + self.seq_b_effs[j][i]
            h = self.activation(z_corr)

        # h_last_perturbed: post-activation of last perturbed layer (before correction)
        # z_next: pre-activation after last correction
        return h, z_corr

    def _forward_bn_refit(self, x, i, p):
        # PreActResNet-18 specific forward
        if len(self.layers) == 1:
            w_stem_orig = self.layer_params["l1"]["W"]
            dw = p.reshape(w_stem_orig.shape)
            w_new = w_stem_orig + dw
            raw_pert = jax.lax.conv_general_dilated(
                lhs=x, rhs=w_new.transpose(3, 2, 0, 1),
                window_strides=(1, 1), padding='SAME',
                dimension_numbers=('NHWC', 'OIHW', 'NHWC'))
            g, b = self.bn_refits[0][i]["gamma"], self.bn_refits[0][i]["beta"]
            
            out_bn1 = g * raw_pert + b
            out_relu1 = jax.nn.relu(out_bn1)
            
            blk0 = self.base_model.stage1[0]
            y = blk0.conv1(out_relu1)
            y = blk0.bn2(y, use_running_average=True)
            y = jax.nn.relu(y)
            y = blk0.conv2(y)
            
            identity = raw_pert
            if blk0.downsample is not None:
                identity = blk0.downsample(out_relu1)
                
            out = y + identity
            
        else:
            # Multi-layer BN refit (Stem + Stage1 b0 conv1)
            w1_orig = self.layer_params["l1"]["W"]
            w2_orig = self.layer_params["l2"]["W"]
            dw1 = p[:w1_orig.size].reshape(w1_orig.shape)
            dw2 = p[w1_orig.size:].reshape(w2_orig.shape)
            
            # Stem
            raw_st = jax.lax.conv_general_dilated(
                lhs=x, rhs=(w1_orig+dw1).transpose(3, 2, 0, 1),
                window_strides=(1, 1), padding='SAME',
                dimension_numbers=('NHWC', 'OIHW', 'NHWC'))
            g1, b1 = self.bn_refits[0][i]["gamma"], self.bn_refits[0][i]["beta"]
            out_bn1 = g1 * raw_st + b1
            out_relu1 = jax.nn.relu(out_bn1)
            
            # Stage1 block0 conv1
            blk0 = self.base_model.stage1[0]
            raw_c1 = jax.lax.conv_general_dilated(
                lhs=out_relu1, rhs=(w2_orig+dw2).transpose(3, 2, 0, 1),
                window_strides=(1, 1), padding='SAME',
                dimension_numbers=('NHWC', 'OIHW', 'NHWC'))
            g2, b2 = self.bn_refits[1][i]["gamma"], self.bn_refits[1][i]["beta"]
            out_bn2 = g2 * raw_c1 + b2
            out_relu2 = jax.nn.relu(out_bn2)
            
            y = blk0.conv2(out_relu2)
            identity = raw_st
            if blk0.downsample is not None:
                identity = blk0.downsample(out_relu1)
            out = y + identity
            
        # Remainder of PreActResNet
        for blk in list(self.base_model.stage1)[1:]:
            out = blk(out, use_running_average=True)
        for blk in self.base_model.stage2:
            out = blk(out, use_running_average=True)
        for blk in self.base_model.stage3:
            out = blk(out, use_running_average=True)
        for blk in self.base_model.stage4:
            out = blk(out, use_running_average=True)
        
        out = self.base_model.final_bn(out, use_running_average=True)
        out = jax.nn.relu(out)
        out = jnp.mean(out, axis=(1, 2))
        return self.base_model.fc(out)

    def _compute_dp_all(self):
        """Compute per-member perturbation vectors. Not needed for per-layer independent mode."""
        if self.layer_specs is not None:
            return None  # sequential methods use self.seq_dWs instead
        coeffs_all = self._get_coeffs_all()
        return coeffs_all @ np.array(self.v_opts)

    def predict(self, x: jax.Array) -> jax.Array:
        dp_all = self._compute_dp_all()

        ys = []
        for i in range(self._n_members):
            ys.append(self._forward_member(x, i, dp_all))
        if ys and isinstance(ys[0], tuple) and len(ys[0]) == 2:
            means = jnp.stack([y[0] for y in ys], axis=0)
            vars = jnp.stack([y[1] for y in ys], axis=0)
            return means, vars
        return jnp.stack(ys, axis=0)

    def predict_one(self, x: jax.Array, idx: int) -> jax.Array:
        dp_all = self._compute_dp_all()
        return self._forward_member(x, idx, dp_all)

    def predict_intermediate_and_corrected(self, x: jax.Array) -> tuple[jax.Array, jax.Array]:
        dp_all = self._compute_dp_all()
        h_vals = []
        z_vals = []
        for i in range(self._n_members):
            h_last_perturbed, z_next = self._predict_member_intermediate_and_corrected(x, i, dp_all)
            h_vals.append(h_last_perturbed)
            z_vals.append(z_next)
        return jnp.stack(h_vals, axis=0), jnp.stack(z_vals, axis=0)







# ==============================================================================
# Hybrid Ensemble: Deep Ensemble + PJSVD per member
# ==============================================================================

class EnsemblePJSVDHybrid:
    """
    Hybrid uncertainty estimator: M independent base models (deep ensemble)
    with PJSVD applied to each member, yielding M × K total samples.

    If the per-member ensembles are probabilistic (each ``predict`` returns a
    ``(means, vars)`` tuple of shape ``((K, B, D), (K, B, D))``), the hybrid
    concatenates both the means and the variances along the member axis and
    returns a tuple of shape ``((M*K, B, D), (M*K, B, D))``. Downstream
    ``_predictive_mean_var`` then applies the law of total variance correctly.

    If the per-member ensembles are deterministic (each ``predict`` returns a
    single ``(K, B, D)`` tensor), the hybrid concatenates along axis 0 and
    returns a single ``(M*K, B, D)`` tensor.
    """

    def __init__(self, pjsvd_ensembles: List['PJSVDEnsemble']):
        """
        Args:
            pjsvd_ensembles: one PJSVDEnsemble per base model.
        """
        self.pjsvd_ensembles = pjsvd_ensembles

    def predict(self, x):
        """Returns concatenated predictions along the ensemble axis."""
        all_preds = [ens.predict(x) for ens in self.pjsvd_ensembles]
        if all_preds and isinstance(all_preds[0], tuple) and len(all_preds[0]) == 2:
            means = jnp.concatenate([p[0] for p in all_preds], axis=0)
            vars_ = jnp.concatenate([p[1] for p in all_preds], axis=0)
            return means, vars_
        return jnp.concatenate(all_preds, axis=0)

    def predict_one(self, x: Float[Array, "batch ..."], idx: int) -> Float[Array, "batch ..."]:
        """Index into the flattened list of M*K members."""
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
            ys.append(_sample_probabilistic(model(x), i))
        return jnp.stack(ys, axis=0)

    def predict_one(self, x: Float[Array, "batch ..."], idx: int) -> Float[Array, "batch ..."]:
        return _sample_probabilistic(self.models[idx](x), idx)


class MCDropoutEnsemble:
    def __init__(self, model: TransitionModel, n_models: int):
        self.model = model
        self.n_models = n_models

    def predict(self, x: Float[Array, "batch ..."]) -> Float[Array, "ens batch ..."]:
        ys = []
        for _ in range(self.n_models):
            ys.append(self.model(x, deterministic=False))
        if ys and isinstance(ys[0], tuple) and len(ys[0]) == 2:
            means = jnp.stack([y[0] for y in ys], axis=0)
            vars_ = jnp.stack([y[1] for y in ys], axis=0)
            return means, vars_
        return jnp.stack(ys, axis=0)

    def predict_one(self, x: Float[Array, "batch ..."], idx: int) -> Float[Array, "batch ..."]:
        return self.model(x, deterministic=False)


class SWAGEnsemble:
    """Full SWAG ensemble with low-rank-plus-diagonal sampling and BN refresh.

    Legacy callers can omit `swag_cov_mat_sqrt`, in which case sampling falls back to
    the diagonal term only.
    """

    def __init__(
        self,
        model: Any,
        swag_mean: nnx.State,
        swag_var: nnx.State,
        n_models: int,
        swag_cov_mat_sqrt: Optional[jax.Array] = None,
        *,
        bn_refresh_inputs: Optional[np.ndarray] = None,
        bn_refresh_batch_size: int = 128,
        use_bn_refresh: bool = True,
        seed: int = 0,
        cache_samples: bool = False,
    ):
        self.base_model = nnx.clone(model)
        self.swag_mean = swag_mean
        self.swag_var = swag_var
        self.swag_cov_mat_sqrt = None
        self.n_models = n_models
        self.bn_refresh_inputs = None if bn_refresh_inputs is None else np.asarray(bn_refresh_inputs)
        self.bn_refresh_batch_size = max(1, int(bn_refresh_batch_size))
        self.use_bn_refresh = bool(use_bn_refresh and self.bn_refresh_inputs is not None)
        self.rng = np.random.RandomState(seed)
        self.cache_samples = bool(cache_samples)
        self._cached_models: Optional[list] = None
        self.swag_mean_flat, self.unflatten_fn = jax.flatten_util.ravel_pytree(swag_mean)
        self.swag_var_flat, _ = jax.flatten_util.ravel_pytree(swag_var)
        if swag_cov_mat_sqrt is None:
            self.swag_cov_mat_sqrt = jnp.zeros((self.swag_mean_flat.shape[0], 0), dtype=self.swag_mean_flat.dtype)
        else:
            self.swag_cov_mat_sqrt = jnp.asarray(swag_cov_mat_sqrt, dtype=self.swag_mean_flat.dtype)
        if self.swag_cov_mat_sqrt.ndim != 2:
            raise ValueError("swag_cov_mat_sqrt must have shape (n_params, rank)")
        if self.swag_cov_mat_sqrt.shape[0] != self.swag_mean_flat.shape[0]:
            raise ValueError("swag_cov_mat_sqrt first dimension must match flattened parameters")

    def _sample_params(self) -> nnx.State:
        flat_dtype = np.asarray(self.swag_mean_flat).dtype
        diag_noise = self.rng.normal(size=self.swag_mean_flat.shape).astype(flat_dtype)
        sampled_flat = self.swag_mean_flat + jnp.sqrt(0.5 * self.swag_var_flat) * jnp.asarray(diag_noise)

        low_rank_rank = self.swag_cov_mat_sqrt.shape[1]
        if low_rank_rank >= 2:
            low_rank_noise = self.rng.normal(size=(low_rank_rank,)).astype(flat_dtype)
            # Standard SWAG samples from a covariance made of a diagonal term plus a
            # low-rank empirical covariance D D^T / (K - 1), with each term scaled by 1/2.
            sampled_flat = sampled_flat + (
                jnp.sqrt(0.5 / float(low_rank_rank - 1))
                * (self.swag_cov_mat_sqrt @ jnp.asarray(low_rank_noise))
            )

        return self.unflatten_fn(sampled_flat)

    def _resolve_path(self, root: Any, path: tuple[Any, ...]) -> Any:
        node = root
        for key in path:
            if isinstance(key, int):
                node = node[key]
            else:
                node = getattr(node, key)
        return node

    def _reset_batch_norm_stats(self, model: Any) -> None:
        for path, variable in nnx.to_flat_state(nnx.state(model, nnx.BatchStat)):
            parent = self._resolve_path(model, path[:-1])
            leaf_name = path[-1]
            target = getattr(parent, leaf_name)
            current = variable.get_value()
            if leaf_name == "mean":
                nnx.update(target, jnp.zeros_like(current))
            elif leaf_name == "var":
                nnx.update(target, jnp.ones_like(current))

    def _set_bn_momentum(self, model: Any, momentum: float) -> None:
        """Set momentum on all BatchNorm modules in the model."""
        for _, mod in nnx.iter_modules(model):
            if isinstance(mod, nnx.BatchNorm):
                mod.momentum = momentum

    def _refresh_batch_norm_stats(self, model: Any) -> None:
        if not self.use_bn_refresh or self.bn_refresh_inputs is None or len(self.bn_refresh_inputs) == 0:
            return

        self._reset_batch_norm_stats(model)
        # Use cumulative average (momentum = 1/k for batch k) instead of EMA.
        # With default EMA momentum (~0.9), 16 batches from zero only reaches
        # ~82% of true running stats, producing random-chance predictions.
        # Cumulative averaging converges in O(n_batches) regardless of momentum.
        n_batches = max(1, (len(self.bn_refresh_inputs) + self.bn_refresh_batch_size - 1) // self.bn_refresh_batch_size)
        original_momentum = None
        for _, mod in nnx.iter_modules(model):
            if isinstance(mod, nnx.BatchNorm):
                original_momentum = mod.momentum
                break

        for batch_idx, start in enumerate(range(0, len(self.bn_refresh_inputs), self.bn_refresh_batch_size)):
            self._set_bn_momentum(model, 1.0 / (batch_idx + 1))
            xb = jnp.array(self.bn_refresh_inputs[start:start + self.bn_refresh_batch_size])
            logits = model(xb, use_running_average=False)
            if hasattr(logits, "block_until_ready"):
                logits.block_until_ready()

        if original_momentum is not None:
            self._set_bn_momentum(model, original_momentum)

    def _sample_model(self) -> Any:
        sampled_model = nnx.clone(self.base_model)
        nnx.update(sampled_model, self._sample_params())
        self._refresh_batch_norm_stats(sampled_model)
        return sampled_model

    def _get_cached_models(self) -> list:
        if self._cached_models is None:
            self._cached_models = [self._sample_model() for _ in range(self.n_models)]
        return self._cached_models

    def _call_model(self, model, x):
        """Call model, passing use_running_average only if it accepts it (BN models)."""
        try:
            return model(x, use_running_average=True)
        except TypeError:
            return model(x)

    def predict(self, x: Float[Array, "batch ..."]) -> Float[Array, "ens batch ..."]:
        if self.cache_samples:
            models = self._get_cached_models()
            ys = [self._call_model(m, x) for m in models]
        else:
            ys = []
            for _ in range(self.n_models):
                sampled_m = self._sample_model()
                ys.append(self._call_model(sampled_m, x))
        if ys and isinstance(ys[0], tuple) and len(ys[0]) == 2:
            means = jnp.stack([y[0] for y in ys], axis=0)
            vars_ = jnp.stack([y[1] for y in ys], axis=0)
            return means, vars_
        return jnp.stack(ys, axis=0)

    def predict_one(self, x: Float[Array, "batch ..."], idx: int) -> Float[Array, "batch ..."]:
        if self.cache_samples:
            models = self._get_cached_models()
            return self._call_model(models[idx], x)
        del idx
        return self._call_model(self._sample_model(), x)

    def predict_intermediate(self, x: jax.Array, layer_idx: int = 1) -> jax.Array:
        from util import get_intermediate_state
        hs = []
        for _ in range(self.n_models):
            sampled_m = self._sample_model()
            hs.append(get_intermediate_state(sampled_m, x, layer_idx))
        return jnp.stack(hs, axis=0)


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

        # Detect dual-head probabilistic model
        self.is_probabilistic = hasattr(model, 'mean_layer') and hasattr(model, 'var_layer')
        if self.is_probabilistic:
            self.map_params['mean_layer'] = (model.mean_layer.kernel.get_value(), model.mean_layer.bias.get_value())
            self.map_params['var_layer'] = (model.var_layer.kernel.get_value(), model.var_layer.bias.get_value())

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
            # Also perturb mean/var heads for probabilistic models
            if self.is_probabilistic:
                for head_name, head_attr in [('mean_layer', self.model.mean_layer), ('var_layer', self.model.var_layer)]:
                    w, b = self._sample_layer_weights(head_name)
                    norm = jnp.linalg.norm((w - self.map_params[head_name][0]).flatten())
                    total_norm_sq += norm**2
                    nnx.update(head_attr.kernel, w)
                    nnx.update(head_attr.bias, b)
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
        if ys and isinstance(ys[0], tuple) and len(ys[0]) == 2:
            means = jnp.stack([y[0] for y in ys], axis=0)
            vars_ = jnp.stack([y[1] for y in ys], axis=0)
            return means, vars_
        return jnp.stack(ys, axis=0)

    def predict_one(self, x: Float[Array, "batch ..."], idx: int) -> Float[Array, "batch ..."]:
        sampled_m, _ = self._sample_model()
        return sampled_m(x)

    def predict_intermediate(self, x: jax.Array, layer_idx: int = 1) -> jax.Array:
        from util import get_intermediate_state
        hs = []
        for _ in range(self.n_models):
            sampled_m, _ = self._sample_model()
            hs.append(get_intermediate_state(sampled_m, x, layer_idx))
        return jnp.stack(hs, axis=0)


class Epinet(nnx.Module):
    """Epistemic network (Osband et al., 2023).

    Takes base-network features and an epistemic index z, and outputs a
    logit correction of shape ``[batch, n_classes]``.  The final linear
    layer has shape ``[hidden, n_classes * index_dim]``; its output is
    reshaped to ``[batch, n_classes, index_dim]`` and contracted with z.
    """

    def __init__(
        self,
        feature_dim: int,
        n_classes: int,
        index_dim: int = 8,
        hiddens: tuple[int, ...] = (50, 50),
        *,
        rngs: nnx.Rngs,
    ):
        self.n_classes = n_classes
        self.index_dim = index_dim

        in_dim = feature_dim + index_dim
        layers = []
        for h in hiddens:
            layers.append(nnx.Linear(in_dim, h, rngs=rngs))
            in_dim = h
        layers.append(nnx.Linear(in_dim, n_classes * index_dim, rngs=rngs))
        self.layers = nnx.List(layers)

    def __call__(
        self,
        features: Float[Array, "batch feature_dim"],
        z: Float[Array, "index_dim"],
    ) -> Float[Array, "batch n_classes"]:
        z_broadcast = jnp.broadcast_to(z, (features.shape[0], self.index_dim))
        x = jnp.concatenate([features, z_broadcast], axis=-1)
        for layer in self.layers[:-1]:
            x = jax.nn.relu(layer(x))
        x = self.layers[-1](x)  # [batch, n_classes * index_dim]
        x = x.reshape(-1, self.n_classes, self.index_dim)  # [batch, n_classes, index_dim]
        return jnp.sum(x * z_broadcast[:, None, :], axis=-1)  # [batch, n_classes]


class EpinetWithPrior(nnx.Module):
    """Trainable epinet plus a frozen random prior network (Osband et al.)."""

    def __init__(
        self,
        feature_dim: int,
        n_classes: int,
        index_dim: int = 8,
        hiddens: tuple[int, ...] = (50, 50),
        prior_scale: float = 1.0,
        *,
        rngs: nnx.Rngs,
    ):
        self.prior_scale = prior_scale
        self.learnable = Epinet(feature_dim, n_classes, index_dim, hiddens, rngs=rngs)
        self.prior = Epinet(feature_dim, n_classes, index_dim, hiddens, rngs=rngs)

    def __call__(
        self,
        features: Float[Array, "batch feature_dim"],
        z: Float[Array, "index_dim"],
    ) -> Float[Array, "batch n_classes"]:
        learned = self.learnable(features, z)
        prior = jax.lax.stop_gradient(self.prior(features, z))
        return learned + self.prior_scale * prior


class EpinetEnsemble:
    """Wraps a frozen base model + trained EpinetWithPrior for ensemble prediction."""

    def __init__(
        self,
        base_model: Any,
        epinet: EpinetWithPrior,
        n_models: int,
        index_dim: int = 8,
        seed: int = 0,
    ):
        self.base_model = base_model
        self.epinet = epinet
        self.n_models = n_models
        self.index_dim = index_dim
        self.rng = np.random.RandomState(seed)

    def predict(self, x: Float[Array, "batch ..."]) -> Float[Array, "ens batch ..."]:
        features = jax.lax.stop_gradient(self.base_model.features(x, use_running_average=True))
        base_logits = jax.lax.stop_gradient(self.base_model(x, use_running_average=True))
        ys = []
        for _ in range(self.n_models):
            z = jnp.array(self.rng.normal(size=(self.index_dim,)).astype(np.float32))
            correction = self.epinet(features, z)
            ys.append(base_logits + correction)
        return jnp.stack(ys, axis=0)

    def predict_one(self, x: Float[Array, "batch ..."], idx: int) -> Float[Array, "batch ..."]:
        del idx
        features = jax.lax.stop_gradient(self.base_model.features(x, use_running_average=True))
        base_logits = jax.lax.stop_gradient(self.base_model(x, use_running_average=True))
        z = jnp.array(self.rng.normal(size=(self.index_dim,)).astype(np.float32))
        return base_logits + self.epinet(features, z)


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
            ys.append(_sample_probabilistic(sampled_m(x), i))
        return jnp.stack(ys, axis=0)

    def predict_one(self, x: Float[Array, "batch ..."], idx: int) -> Float[Array, "batch ..."]:
        sampled_m, _ = self._sample_model(idx)
        return _sample_probabilistic(sampled_m(x), idx)

    def predict_intermediate(self, x: jax.Array, layer_idx: int = 1) -> jax.Array:
        from util import get_intermediate_state
        hs = []
        for i in range(self.n_samples):
            self._sample_model(i)
            hs.append(get_intermediate_state(self.base_model, x, layer_idx))
        return jnp.stack(hs, axis=0)


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


def _sample_probabilistic(out: Any, seed: int) -> jax.Array:
    if isinstance(out, tuple) and len(out) == 2:
        mean, var = out
        rng = jax.random.PRNGKey(seed)
        return mean + jnp.sqrt(var) * jax.random.normal(rng, mean.shape)
    return out

# ==============================================================================
# Perturb-and-Correct (PnC) Ensemble
# ==============================================================================
from pnc import flatten_conv_kernel_to_patches, solve_chunked_conv2_correction

class PnCEnsemble:
    """
    Evaluates inference for a Perturb-and-Correct ensemble.
    For each member, perturbs Conv1 weights and dynamically refits Conv2
    using Ridge Regression to preserve the block output on calibration data.
    """
    def __init__(
        self,
        base_model: Any,
        v_opts: jax.Array,
        sigmas: jax.Array,
        z_coeffs: np.ndarray,
        perturbation_scale: float,
        get_Y_fn: Callable,
        w1_orig: jax.Array,
        w2_orig: jax.Array,
        chunks: List[jax.Array],
        T_orig_chunks: List[jax.Array],
        target_stage_idx: int,
        target_block_idx: int,
        lambda_reg: float = 1e-3,
        sigma_sq_weights: bool = False,
        member_radius_distribution: str = "fixed",
        member_radius_std: float = 0.0,
        member_radius_values: Optional[Any] = None,
        member_radius_seed: int = 0,
    ):
        self.base_model = base_model
        self.v_opts = v_opts
        self.sigmas = jnp.array(sigmas)
        self.z_coeffs = z_coeffs
        self.perturbation_scale = perturbation_scale
        self.get_Y_fn = get_Y_fn
        self.w1_orig = w1_orig
        self.w2_orig = w2_orig
        self.chunks = chunks
        self.T_orig_chunks = T_orig_chunks
        self.target_stage_idx = target_stage_idx
        self.target_block_idx = target_block_idx
        self.lambda_reg = lambda_reg
        self.sigma_sq_weights = sigma_sq_weights
        self.member_radius_distribution = member_radius_distribution
        self.member_radius_std = float(member_radius_std)
        self.member_radius_values = _parse_member_radius_values(member_radius_values)
        self.member_radius_seed = int(member_radius_seed)
        self.member_radius_multipliers = _sample_member_radius_multipliers(
            len(self.z_coeffs),
            distribution=self.member_radius_distribution,
            std=self.member_radius_std,
            values=self.member_radius_values,
            seed=self.member_radius_seed,
        )
        
        self.members_w1 = []
        self.members_w2 = []
        
        self._precompute_corrections()

    def _get_coeffs_all(self):
        return _scale_coefficients_with_member_radii(
            self.z_coeffs,
            np.array(self.sigmas),
            self.perturbation_scale,
            self.sigma_sq_weights,
            self.member_radius_multipliers,
        )

    def _precompute_corrections(self):
        coeffs_all = self._get_coeffs_all()
        # dp_all: (N, D_flat)
        dp_all = coeffs_all @ np.array(self.v_opts)

        for i in range(len(self.z_coeffs)):
            p = dp_all[i]
            w1_pert = self.w1_orig + p.reshape(self.w1_orig.shape)
            # solve_chunked_conv2_correction returns W2_new shaped (kh,kw,Cin,Cout)
            w2_pert, b2_pert = solve_chunked_conv2_correction(
                self.get_Y_fn, w1_pert, self.w2_orig, self.chunks, self.T_orig_chunks,
                lambda_reg=self.lambda_reg
            )
            self.members_w1.append(w1_pert)
            self.members_w2.append((w2_pert, b2_pert))

        # Report calibration-set diagnostics once all members are solved.
        self.calib_raw_arr, self.calib_corr_arr = self.compute_shift_diagnostics(
            self.chunks, self.T_orig_chunks, label="calib")

    def compute_shift_diagnostics(
        self,
        chunks: List[jax.Array],
        T_orig_chunks: List[jax.Array],
        label: str = "calib",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        For every ensemble member, compute the RMS per-sample L2 shift in block
        output both *before* and *after* the Conv2 correction, evaluated on the
        supplied chunked data.

        Args:
            chunks:        List of pre-activation input chunks to the target block,
                           each shaped (n, H, W, C_in_block).
            T_orig_chunks: Corresponding list of original block outputs (Conv2 output)
                           from the unperturbed model, each shaped (n, H', W', C_out).
            label:         String prefix used in console output (e.g. "calib", "test").

        Returns:
            raw_arr  : (N,) RMS shift before correction, one value per member.
            corr_arr : (N,) RMS shift after  correction, one value per member.
        """
        kh, kw, C_in, C_out = self.w2_orig.shape
        w2_flat_orig = flatten_conv_kernel_to_patches(self.w2_orig)

        raw_norms  = []
        corr_norms = []
        for w1_pert, (w2_pert, b2_pert) in zip(self.members_w1, self.members_w2):
            w2_pert_flat = flatten_conv_kernel_to_patches(w2_pert)
            total_samples   = 0
            raw_norm_sq_sum  = 0.0
            corr_norm_sq_sum = 0.0
            for chunk, T_orig_chunk in zip(chunks, T_orig_chunks):
                n = chunk.shape[0]
                _, H_t, W_t, C_t = T_orig_chunk.shape
                Y_pert = self.get_Y_fn(w1_pert, chunk)  # (n*H_t*W_t, C_in*kh*kw)

                # Uncorrected: perturbed w1, original w2 (no bias)
                T_raw_flat    = Y_pert @ w2_flat_orig
                T_raw_spatial = T_raw_flat.reshape(n, H_t, W_t, C_t)
                diff_raw = (T_raw_spatial - T_orig_chunk).reshape(n, -1)
                raw_norm_sq_sum += float(jnp.sum(jnp.sum(diff_raw ** 2, axis=-1)))

                # Corrected: perturbed w1, refitted w2 + bias
                T_corr_flat    = Y_pert @ w2_pert_flat + b2_pert
                T_corr_spatial = T_corr_flat.reshape(n, H_t, W_t, C_t)
                diff_corr = (T_corr_spatial - T_orig_chunk).reshape(n, -1)
                corr_norm_sq_sum += float(jnp.sum(jnp.sum(diff_corr ** 2, axis=-1)))

                total_samples += n

            raw_norms.append(float(np.sqrt(raw_norm_sq_sum  / total_samples)))
            corr_norms.append(float(np.sqrt(corr_norm_sq_sum / total_samples)))

        raw_arr  = np.array(raw_norms)
        corr_arr = np.array(corr_norms)

        N = len(self.z_coeffs)
        print(f"[PnC diagnostic | {label}] perturbation_scale={self.perturbation_scale}")
        print(f"  Uncorrected shift (mean±std, {N} members): "
              f"{raw_arr.mean():.4f} ± {raw_arr.std():.4f}")
        print(f"  Corrected   shift (mean±std, {N} members): "
              f"{corr_arr.mean():.4f} ± {corr_arr.std():.4f}")
        reduction = 1.0 - corr_arr.mean() / (raw_arr.mean() + 1e-12)
        print(f"  Correction reduction: {reduction * 100:.1f}%")

        return raw_arr, corr_arr

    def _forward_member(self, x: jax.Array, idx: int) -> jax.Array:
        # 1. Stem
        h = self.base_model.stem(x)
        
        # 2. Stages
        stages = [self.base_model.stage1, self.base_model.stage2, 
                  self.base_model.stage3, self.base_model.stage4]
                  
        w1_pert = self.members_w1[idx]
        w2_pert, b2_pert = self.members_w2[idx]
        
        for s_idx, stage in enumerate(stages):
            for b_idx, blk in enumerate(stage):
                if s_idx == self.target_stage_idx and b_idx == self.target_block_idx:
                    # Custom forward for the perturbed block.
                    # get_Y_fn returns post-BN2-ReLU Conv1 patches, so apply BN2/ReLU
                    # before applying W2_new to match the regression setup.
                    out_bn1 = blk.bn1(h, use_running_average=True)
                    out_relu1 = jax.nn.relu(out_bn1)
                    
                    # Perturbed Conv1
                    strides = tuple(blk.conv1.strides)
                    y_raw = jax.lax.conv_general_dilated(
                        lhs=out_relu1, rhs=w1_pert.transpose(3, 2, 0, 1),
                        window_strides=strides, padding='SAME',
                        dimension_numbers=('NHWC', 'OIHW', 'NHWC'))
                    
                    # Apply BN2 and ReLU before perturbed Conv2
                    y_bn2 = blk.bn2(y_raw, use_running_average=True)
                    y_relu2 = jax.nn.relu(y_bn2)
                    
                    # Perturbed Conv2 applied to post-BN2-ReLU Conv1 output
                    t = jax.lax.conv_general_dilated(
                        lhs=y_relu2, rhs=w2_pert.transpose(3, 2, 0, 1),
                        window_strides=(1, 1), padding='SAME',
                        dimension_numbers=('NHWC', 'OIHW', 'NHWC'))
                    t = t + b2_pert
                    
                    identity = h
                    if blk.downsample is not None:
                        identity = blk.downsample(out_relu1)
                    
                    h = t + identity
                else:
                    h = blk(h, use_running_average=True)
                    
        # 3. Head
        h = self.base_model.final_bn(h, use_running_average=True)
        h = jax.nn.relu(h)
        h = jnp.mean(h, axis=(1, 2))
        return self.base_model.fc(h)

    def predict(self, x: jax.Array) -> jax.Array:
        ys = []
        for i in range(len(self.z_coeffs)):
            ys.append(self._forward_member(x, i))
        return jnp.stack(ys, axis=0)

    def predict_one(self, x: jax.Array, idx: int) -> jax.Array:
        return self._forward_member(x, idx)


@dataclass
class MultiBlockMemberWeights:
    w1: jax.Array
    w2: jax.Array
    b2: jax.Array


class MultiBlockPnCEnsemble:
    """
    Perturb-and-Correct on multiple residual blocks.

    Directions are cached per block, but the actual perturbed `w1` kernels and
    ridge-solved `w2`/`b2` corrections are recomputed for every
    `perturbation_scale`. That is the key semantic guarantee: two different
    scales produce genuinely different solved members.
    """

    def __init__(
        self,
        base_model: Any,
        block_specs: List[dict],
        z_coeffs: np.ndarray,
        perturbation_scale: float,
        lambda_reg: float = 1e-3,
        sigma_sq_weights: bool = False,
        member_radius_distribution: str = "fixed",
        member_radius_std: float = 0.0,
        member_radius_values: Optional[Any] = None,
        member_radius_seed: int = 0,
        members: Optional[List[List[Any]]] = None,
        raw_calib_arr: Optional[np.ndarray] = None,
        corr_calib_arr: Optional[np.ndarray] = None,
        progress_desc: str = "Multi-block PnC: ridge solves",
    ):
        self.base_model = base_model
        self.block_specs = block_specs
        self.z_coeffs = np.asarray(z_coeffs, dtype=np.float64)
        self.perturbation_scale = float(perturbation_scale)
        self.lambda_reg = lambda_reg
        self.sigma_sq_weights = sigma_sq_weights
        self.member_radius_distribution = member_radius_distribution
        self.member_radius_std = float(member_radius_std)
        self.member_radius_values = _parse_member_radius_values(member_radius_values)
        self.member_radius_seed = int(member_radius_seed)
        self.member_radius_multipliers = _sample_member_radius_multipliers(
            len(self.z_coeffs),
            distribution=self.member_radius_distribution,
            std=self.member_radius_std,
            values=self.member_radius_values,
            seed=self.member_radius_seed,
        )

        if members is not None:
            self.members = self._normalize_members(members)
            self.calib_raw_arr = raw_calib_arr
            self.calib_corr_arr = corr_calib_arr
            return

        n_mem, n_blk, _ = self.z_coeffs.shape
        assert n_blk == len(block_specs), "z_coeffs middle dim must match len(block_specs)"
        assert all(
            int(self.z_coeffs.shape[2]) == int(np.asarray(spec["v_opts"]).shape[0])
            for spec in block_specs
        )

        self.members = []

        try:
            from tqdm import tqdm
        except ImportError:
            class tqdm:  # type: ignore[no-redef]
                def __init__(self, *args, **kwargs):
                    pass
                def update(self, n: int = 1):
                    return None
                def close(self):
                    return None

        pbar = tqdm(total=n_mem * n_blk, desc=progress_desc, unit="solve")
        for i in range(n_mem):
            row = []
            for b, spec in enumerate(block_specs):
                z_row = self.z_coeffs[i, b]
                sigmas = np.asarray(spec["sigmas"])
                v_opts = np.asarray(spec["v_opts"])
                w1_orig = spec["w1_orig"]
                coeffs = self._coeffs_from_z(z_row, sigmas, member_index=i)
                dp = coeffs @ v_opts
                w1_pert = w1_orig + dp.reshape(w1_orig.shape)
                w2_pert, b2_pert = solve_chunked_conv2_correction(
                    spec["get_Y_fn"],
                    w1_pert,
                    spec["w2_orig"],
                    spec["chunks"],
                    spec["T_orig_chunks"],
                    lambda_reg=self.lambda_reg,
                )
                row.append(MultiBlockMemberWeights(w1=w1_pert, w2=w2_pert, b2=b2_pert))
                pbar.update(1)
            self.members.append(row)
        pbar.close()

        raw_stack = []
        corr_stack = []
        for b, spec in enumerate(block_specs):
            raw_arr, corr_arr = self._shift_diagnostics_one_block(
                b,
                spec["chunks"],
                spec["T_orig_chunks"],
                label=f"calib block {b}",
            )
            raw_stack.append(raw_arr)
            corr_stack.append(corr_arr)

        self.calib_raw_arr = np.mean(np.stack(raw_stack, axis=0), axis=0)
        self.calib_corr_arr = np.mean(np.stack(corr_stack, axis=0), axis=0)
        print(
            f"[MultiBlock PnC | calib aggregated over {n_blk} blocks] "
            f"perturbation_scale={self.perturbation_scale} | raw {self.calib_raw_arr.mean():.4f} "
            f"| corr {self.calib_corr_arr.mean():.4f}"
        )

    def _normalize_members(self, members: List[List[Any]]) -> List[List[MultiBlockMemberWeights]]:
        normalized = []
        for row in members:
            norm_row = []
            for item in row:
                if isinstance(item, MultiBlockMemberWeights):
                    norm_row.append(item)
                else:
                    w1, w2, b2 = item
                    norm_row.append(MultiBlockMemberWeights(w1=w1, w2=w2, b2=b2))
            normalized.append(norm_row)
        return normalized

    def _coeffs_from_z(self, z_row: np.ndarray, sigmas: np.ndarray, member_index: int) -> np.ndarray:
        coeffs = _scale_coefficients_with_member_radii(
            np.asarray(z_row, dtype=np.float64)[None, :],
            np.asarray(sigmas, dtype=np.float64),
            self.perturbation_scale,
            self.sigma_sq_weights,
            np.asarray([self.member_radius_multipliers[member_index]], dtype=np.float64),
        )
        return coeffs[0]

    def _shift_diagnostics_one_block(
        self,
        block_index: int,
        chunks: List[jax.Array],
        T_orig_chunks: List[jax.Array],
        label: str = "calib",
    ) -> Tuple[np.ndarray, np.ndarray]:
        spec = self.block_specs[block_index]
        kh, kw, c_in, c_out = spec["w2_orig"].shape
        w2_flat_orig = flatten_conv_kernel_to_patches(spec["w2_orig"])

        raw_norms = []
        corr_norms = []
        for member_weights in self.members:
            weights = member_weights[block_index]
            w2_pert_flat = flatten_conv_kernel_to_patches(weights.w2)
            total_samples = 0
            raw_norm_sq_sum = 0.0
            corr_norm_sq_sum = 0.0

            for chunk, T_orig_chunk in zip(chunks, T_orig_chunks):
                n = chunk.shape[0]
                _, h_t, w_t, c_t = T_orig_chunk.shape
                Y_pert = spec["get_Y_fn"](weights.w1, chunk)

                T_raw_flat = Y_pert @ w2_flat_orig
                T_raw_spatial = T_raw_flat.reshape(n, h_t, w_t, c_t)
                diff_raw = (T_raw_spatial - T_orig_chunk).reshape(n, -1)
                raw_norm_sq_sum += float(jnp.sum(jnp.sum(diff_raw ** 2, axis=-1)))

                T_corr_flat = Y_pert @ w2_pert_flat + weights.b2
                T_corr_spatial = T_corr_flat.reshape(n, h_t, w_t, c_t)
                diff_corr = (T_corr_spatial - T_orig_chunk).reshape(n, -1)
                corr_norm_sq_sum += float(jnp.sum(jnp.sum(diff_corr ** 2, axis=-1)))

                total_samples += n

            raw_norms.append(float(np.sqrt(raw_norm_sq_sum / total_samples)))
            corr_norms.append(float(np.sqrt(corr_norm_sq_sum / total_samples)))

        return np.array(raw_norms), np.array(corr_norms)

    def compute_shift_diagnostics(
        self,
        block_chunks: List[List[jax.Array]],
        block_T_orig: List[List[jax.Array]],
        label: str = "test",
    ) -> Tuple[np.ndarray, np.ndarray]:
        raw_stack = []
        corr_stack = []
        for b in range(len(self.block_specs)):
            raw_arr, corr_arr = self._shift_diagnostics_one_block(
                b,
                block_chunks[b],
                block_T_orig[b],
                label=f"{label} block {b}",
            )
            raw_stack.append(raw_arr)
            corr_stack.append(corr_arr)

        raw_arr = np.mean(np.stack(raw_stack, axis=0), axis=0)
        corr_arr = np.mean(np.stack(corr_stack, axis=0), axis=0)
        print(
            f"[MultiBlock PnC diagnostic | {label}] perturbation_scale={self.perturbation_scale} "
            f"| raw mean {raw_arr.mean():.4f} | corr mean {corr_arr.mean():.4f}"
        )
        return raw_arr, corr_arr

    def _forward_member(self, x: jax.Array, idx: int) -> jax.Array:
        h = self.base_model.stem(x)
        stages = [
            self.base_model.stage1,
            self.base_model.stage2,
            self.base_model.stage3,
            self.base_model.stage4,
        ]
        member_lookup = {
            (spec["stage_idx"], spec["block_idx"]): self.members[idx][b]
            for b, spec in enumerate(self.block_specs)
        }

        for s_idx, stage in enumerate(stages):
            for b_idx, blk in enumerate(stage):
                weights = member_lookup.get((s_idx, b_idx))
                if weights is None:
                    h = blk(h, use_running_average=True)
                    continue

                out_bn1 = blk.bn1(h, use_running_average=True)
                out_relu1 = jax.nn.relu(out_bn1)
                y_raw = jax.lax.conv_general_dilated(
                    lhs=out_relu1,
                    rhs=weights.w1.transpose(3, 2, 0, 1),
                    window_strides=tuple(blk.conv1.strides),
                    padding="SAME",
                    dimension_numbers=("NHWC", "OIHW", "NHWC"),
                )
                y_bn2 = blk.bn2(y_raw, use_running_average=True)
                y_relu2 = jax.nn.relu(y_bn2)
                t = jax.lax.conv_general_dilated(
                    lhs=y_relu2,
                    rhs=weights.w2.transpose(3, 2, 0, 1),
                    window_strides=(1, 1),
                    padding="SAME",
                    dimension_numbers=("NHWC", "OIHW", "NHWC"),
                )
                t = t + weights.b2

                identity = h
                if blk.downsample is not None:
                    identity = blk.downsample(out_relu1)
                h = t + identity

        h = self.base_model.final_bn(h, use_running_average=True)
        h = jax.nn.relu(h)
        h = jnp.mean(h, axis=(1, 2))
        return self.base_model.fc(h)

    def predict(self, x: jax.Array) -> jax.Array:
        ys = []
        for i in range(len(self.z_coeffs)):
            ys.append(self._forward_member(x, i))
        return jnp.stack(ys, axis=0)

    def predict_one(self, x: jax.Array, idx: int) -> jax.Array:
        return self._forward_member(x, idx)


# ==============================================================================
# Last-Layer Laplace Approximation (LLLA) Ensemble
# ==============================================================================

class LLLAEnsemble:
    """
    Last-Layer Laplace Approximation Ensemble.
    Samples weights for the final dense layer from a Gaussian posterior
    N(W_map, (G + lambda I)^-1) where G is the Generalized Gauss-Newton matrix.
    """
    def __init__(
        self,
        model: nnx.Module,
        mean_params: nnx.State,
        cov: jax.Array,
        n_models: int,
        seed: int = 0
    ):
        """
        Args:
            model: The base model (e.g. PreActResNet18).
            mean_params: nnx.State containing the MAP parameters of the 'fc' layer.
            cov: The posterior covariance matrix (D*K + K, D*K + K).
            n_models: Number of samples in the ensemble.
            seed: RNG seed for sampling.
        """
        # We store a clone to avoid mutating the original model in the caller
        self.model = nnx.clone(model)
        self.mean_params = mean_params
        self.cov = cov
        self.n_models = n_models
        self.seed = seed
        
        # Pre-compute Cholesky for efficient sampling: W ~ N(mu, L L^T)
        # We add a small epsilon to the diagonal for numerical stability.
        self.L = jnp.linalg.cholesky(cov + jnp.eye(cov.shape[0]) * 1e-6)
        
        # Helper to flatten/unflatten the fc layer parameters
        self.flat_mean, self.unflatten_fn = jax.flatten_util.ravel_pytree(mean_params)

    def _sample_model(self, step: int) -> nnx.Module:
        rng = jax.random.PRNGKey(self.seed + step)
        eps = jax.random.normal(rng, self.flat_mean.shape)
        
        # Sample: theta = mu + L @ epsilon
        flat_sample = self.flat_mean + self.L @ eps
        sampled_fc_state = self.unflatten_fn(flat_sample)
        
        # Update ONLY the final layer (fc) with the sampled weights
        nnx.update(self.model.fc, sampled_fc_state)
        return self.model

    def predict(self, x: jax.Array) -> jax.Array:
        """Returns (n_models, batch, n_classes) stacked predictions."""
        ys = []
        for i in range(self.n_models):
            m = self._sample_model(i)
            # Standard eval-time forward pass
            ys.append(m(x, use_running_average=True))
        return jnp.stack(ys, axis=0)

    def predict_one(self, x: jax.Array, idx: int) -> jax.Array:
        """Returns (batch, n_classes) prediction for a single ensemble member."""
        m = self._sample_model(idx)
        return m(x, use_running_average=True)
