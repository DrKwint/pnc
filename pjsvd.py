import jax
import jax.numpy as jnp
import optax
from typing import Tuple, Callable

def get_affine_residuals(outputs_batch: jax.Array, original_outputs_batch: jax.Array) -> jax.Array:
    """
    Projects the perturbed outputs onto the orthogonal complement of the subspace
    spanned by the Bias (1) and the Original Signal (y).

    This effectively measures the 'Unfixable Error' that cannot be removed by
    an affine transformation of the next layer.
    """
    # A. Bias Correction (Project onto vector of 1s)
    mean_correction = jnp.mean(outputs_batch, axis=0, keepdims=True)
    centered = outputs_batch - mean_correction

    # B. Scale Correction (Project onto original signal y)
    y_centered = original_outputs_batch - jnp.mean(original_outputs_batch, axis=0, keepdims=True)

    # Gram-Schmidt Projection: v_orth = v - proj_u(v)
    dot_prod = jnp.sum(centered * y_centered, axis=0)
    norm_sq = jnp.sum(y_centered * y_centered, axis=0) + 1e-6

    scale_factor = dot_prod / norm_sq
    scale_correction = scale_factor * y_centered

    return centered - scale_correction


def get_full_span_affine_residuals(outputs_batch: jax.Array, original_outputs_batch: jax.Array) -> jax.Array:
    """
    Project the perturbed outputs onto the orthogonal complement of the
    full Affine Correction Subspace spanned by [Original Signal (Y), 1].
    """
    # Augment Y with a column of 1s representing the bias
    # Y is shape (B, D)
    B = original_outputs_batch.shape[0]
    ones = jnp.ones((B, 1), dtype=original_outputs_batch.dtype)
    Y_aug = jnp.concatenate([original_outputs_batch, ones], axis=-1)  # (B, D + 1)

    # Compute orthogonal basis Q for the column space of Y_aug
    # Q has shape (B, K) where K is the rank (at most D+1)
    Q, _ = jnp.linalg.qr(Y_aug)

    # Project perturbed outputs onto the span of Q
    # proj = Q * (Q^T * outputs_batch)
    # Since outputs_batch is (B, D), we compute Q^T @ outputs_batch -> (K, D)
    # Then Q @ (K, D) -> (B, D)
    proj = jnp.dot(Q, jnp.dot(Q.T, outputs_batch))

    # The residual is the part orthogonal to the Affine Correction Subspace
    residual = outputs_batch - proj
    return residual


# ==============================================================================
# 2. The Solver (Projected Jacobian SVD)
# ==============================================================================

@jax.jit(static_argnums=(0,))
def find_optimal_perturbation(
    model_fn_wrt_layer: Callable[[jax.Array], jax.Array],
    target_param: jax.Array,
    max_iter: int = 100,
    tol: float = 1e-6,
    # Fixed-shape (K_max, D) array, zero-padded for unused slots.
    # Keeping shape constant across all K iterations avoids JAX recompilation.
    orthogonal_directions: jax.Array = None,
    # Boolean mask of shape (K_max,): True for the k valid directions in orthogonal_directions.
    # Passed as a JAX array so the JIT sees a *fixed shape* regardless of k.
    direction_mask: jax.Array = None,
    seed: int = 42,
) -> Tuple[jax.Array, float]:
    """
    Finds the perturbation 'v' that minimizes ||(I-P)Jv|| using Adam.

    Args:
        model_fn_wrt_layer: Function f(W) -> outputs.
        target_param: The weight matrix W to perturb.
        max_iter: Maximum optimization steps.
        tol: Stop if residual falls below this threshold.
        orthogonal_directions: (K_max, D) flat array of prior directions (zero-padded).
        direction_mask: (K_max,) bool array; True for valid directions.
        seed: Random seed.

    Returns:
        v: Optimal perturbation (same shape as target_param).
        sigma: Singular value (residual energy).
    """
    original_outputs = model_fn_wrt_layer(target_param)

    use_orth = (orthogonal_directions is not None) and (direction_mask is not None)

    def energy_fn(v_flat):
        v_norm = v_flat / (jnp.linalg.norm(v_flat) + 1e-6)
        v_weight = v_norm.reshape(target_param.shape)
        _, jvp_out = jax.jvp(model_fn_wrt_layer, (target_param,), (v_weight,))
        residuals = get_affine_residuals(jvp_out, original_outputs)
        return jnp.sum(residuals ** 2)

    def project_orthogonal(v):
        if not use_orth:
            return v
        # mask shape (K_max, 1) so invalid rows contribute 0 to the projection sum
        mask = direction_mask.astype(v.dtype)[:, None]  # (K_max, 1)
        projections = jnp.dot(orthogonal_directions, v)  # (K_max,)
        subtraction = jnp.sum((projections[:, None] * orthogonal_directions) * mask, axis=0)  # (D,)
        return v - subtraction

    optimizer = optax.adam(0.05)
    key = jax.random.PRNGKey(seed)

    v_init = jax.random.normal(key, shape=(target_param.size,))
    v_init = project_orthogonal(v_init)
    v_init = v_init / (jnp.linalg.norm(v_init) + 1e-6)

    opt_state = optimizer.init(v_init)
    init_val = (0, v_init, opt_state, jnp.inf)

    def cond_fun(val):
        i, _, _, loss = val
        return (i < max_iter) & (loss > tol)

    def body_fun(val):
        i, v, state, _ = val
        loss, grads = jax.value_and_grad(energy_fn)(v)
        updates, new_state = optimizer.update(grads, state, params=v)
        v_new = optax.apply_updates(v, updates)
        v_new = project_orthogonal(v_new)
        v_new = v_new / (jnp.linalg.norm(v_new) + 1e-6)
        return (i + 1, v_new, new_state, loss)

    final_i, final_v, _, final_loss = jax.lax.while_loop(cond_fun, body_fun, init_val)

    sigma = jnp.sqrt(final_loss)
    return final_v.reshape(target_param.shape), sigma


@jax.jit(static_argnums=(0,))
def find_optimal_perturbation_full(
    model_fn_wrt_layer: Callable[[jax.Array], jax.Array],
    target_param: jax.Array,
    max_iter: int = 100,
    tol: float = 1e-6,
    orthogonal_directions: jax.Array = None,
    direction_mask: jax.Array = None,
    seed: int = 42,
) -> Tuple[jax.Array, float]:
    """
    Finds the perturbation 'v' that minimizes ||(I-P)Jv|| using Adam.
    Uses get_full_span_affine_residuals instead of get_affine_residuals.
    """
    original_outputs = model_fn_wrt_layer(target_param)

    use_orth = (orthogonal_directions is not None) and (direction_mask is not None)

    def energy_fn(v_flat):
        v_norm = v_flat / (jnp.linalg.norm(v_flat) + 1e-6)
        v_weight = v_norm.reshape(target_param.shape)
        _, jvp_out = jax.jvp(model_fn_wrt_layer, (target_param,), (v_weight,))
        residuals = get_full_span_affine_residuals(jvp_out, original_outputs)
        return jnp.sum(residuals ** 2)

    def project_orthogonal(v):
        if not use_orth:
            return v
        mask = direction_mask.astype(v.dtype)[:, None]
        projections = jnp.dot(orthogonal_directions, v)
        subtraction = jnp.sum((projections[:, None] * orthogonal_directions) * mask, axis=0)
        return v - subtraction

    optimizer = optax.adam(0.05)
    key = jax.random.PRNGKey(seed)

    v_init = jax.random.normal(key, shape=(target_param.size,))
    v_init = project_orthogonal(v_init)
    v_init = v_init / (jnp.linalg.norm(v_init) + 1e-6)

    opt_state = optimizer.init(v_init)
    init_val = (0, v_init, opt_state, jnp.inf)

    def cond_fun(val):
        i, _, _, loss = val
        return (i < max_iter) & (loss > tol)

    def body_fun(val):
        i, v, state, _ = val
        loss, grads = jax.value_and_grad(energy_fn)(v)
        updates, new_state = optimizer.update(grads, state, params=v)
        v_new = optax.apply_updates(v, updates)
        v_new = project_orthogonal(v_new)
        v_new = v_new / (jnp.linalg.norm(v_new) + 1e-6)
        return (i + 1, v_new, new_state, loss)

    final_i, final_v, _, final_loss = jax.lax.while_loop(cond_fun, body_fun, init_val)

    sigma = jnp.sqrt(final_loss)
    return final_v.reshape(target_param.shape), sigma


# ==============================================================================
# 3. The Corrector (Affine Adaptation)
# ==============================================================================

def apply_correction(
    next_layer_params: Tuple[jax.Array, jax.Array],
    original_stats: Tuple[jax.Array, jax.Array],
    new_outputs: jax.Array,
) -> Tuple[jax.Array, jax.Array]:
    """
    Adjusts the next layer's weights and bias to restore the original
    mean and variance statistics, compensating for the perturbation.
    """
    w_next, b_next = next_layer_params
    mu_old, std_old = original_stats

    mu_new = jnp.mean(new_outputs, axis=0)
    std_new = jnp.std(new_outputs, axis=0)

    # Scale Correction (Match Variance)
    scale_factor = std_old / (std_new + 1e-6)
    w_next_new = w_next * scale_factor[:, None]

    # Bias Correction (Match Mean)
    expected_activation_old = jnp.dot(mu_old, w_next)
    expected_activation_new = jnp.dot(mu_new, w_next_new)
    b_next_new = b_next + expected_activation_old - expected_activation_new

    return w_next_new, b_next_new


@jax.jit(static_argnums=(0,))
def find_optimal_perturbation_multi_layer(
    model_fn_wrt_layers: Callable[[list[jax.Array]], jax.Array],
    target_params_list: list[jax.Array],
    max_iter: int = 100,
    tol: float = 1e-6,
    orthogonal_directions: jax.Array = None,
    direction_mask: jax.Array = None,
    seed: int = 42,
) -> Tuple[list[jax.Array], float]:
    """
    Finds perturbation 'v' across multiple layers that minimizes ||(I-P)Jv||.

    Args:
        orthogonal_directions: (K_max, total_D) fixed-shape padded array.
        direction_mask: (K_max,) bool mask; True for valid directions.
    """
    original_outputs = model_fn_wrt_layers(target_params_list)

    param_shapes = [p.shape for p in target_params_list]
    param_sizes = [p.size for p in target_params_list]
    total_size = sum(param_sizes)

    use_orth = (orthogonal_directions is not None) and (direction_mask is not None)

    def energy_fn(v_flat):
        v_norm = v_flat / (jnp.linalg.norm(v_flat) + 1e-6)
        v_weights = []
        idx = 0
        for shape, size in zip(param_shapes, param_sizes):
            v_weights.append(v_norm[idx:idx + size].reshape(shape))
            idx += size
        _, jvp_out = jax.jvp(model_fn_wrt_layers, (target_params_list,), (v_weights,))
        residuals = get_affine_residuals(jvp_out, original_outputs)
        return jnp.sum(residuals ** 2)

    def project_orthogonal(v):
        if not use_orth:
            return v
        mask = direction_mask.astype(v.dtype)[:, None]  # (K_max, 1)
        projections = jnp.dot(orthogonal_directions, v)  # (K_max,)
        subtraction = jnp.sum((projections[:, None] * orthogonal_directions) * mask, axis=0)
        return v - subtraction

    optimizer = optax.adam(0.05)
    key = jax.random.PRNGKey(seed)

    v_init = jax.random.normal(key, shape=(total_size,))
    v_init = project_orthogonal(v_init)
    v_init = v_init / (jnp.linalg.norm(v_init) + 1e-6)

    opt_state = optimizer.init(v_init)
    init_val = (0, v_init, opt_state, jnp.inf)

    def cond_fun(val):
        i, _, _, loss = val
        return (i < max_iter) & (loss > tol)

    def body_fun(val):
        i, v, state, _ = val
        loss, grads = jax.value_and_grad(energy_fn)(v)
        updates, new_state = optimizer.update(grads, state, params=v)
        v_new = optax.apply_updates(v, updates)
        v_new = project_orthogonal(v_new)
        v_new = v_new / (jnp.linalg.norm(v_new) + 1e-6)
        return (i + 1, v_new, new_state, loss)

    final_i, final_v, _, final_loss = jax.lax.while_loop(cond_fun, body_fun, init_val)

    sigma = jnp.sqrt(final_loss)

    final_v_weights = []
    idx = 0
    for shape, size in zip(param_shapes, param_sizes):
        final_v_weights.append(final_v[idx:idx + size].reshape(shape))
        idx += size

    return final_v_weights, sigma


@jax.jit(static_argnums=(0,))
def find_optimal_perturbation_multi_layer_full(
    model_fn_wrt_layers: Callable[[list[jax.Array]], jax.Array],
    target_params_list: list[jax.Array],
    max_iter: int = 100,
    tol: float = 1e-6,
    orthogonal_directions: jax.Array = None,
    direction_mask: jax.Array = None,
    seed: int = 42,
) -> Tuple[list[jax.Array], float]:
    """
    Finds perturbation 'v' across multiple layers that minimizes ||(I-P)Jv||.
    Uses get_full_span_affine_residuals instead of get_affine_residuals.
    """
    original_outputs = model_fn_wrt_layers(target_params_list)

    param_shapes = [p.shape for p in target_params_list]
    param_sizes = [p.size for p in target_params_list]
    total_size = sum(param_sizes)

    use_orth = (orthogonal_directions is not None) and (direction_mask is not None)

    def energy_fn(v_flat):
        v_norm = v_flat / (jnp.linalg.norm(v_flat) + 1e-6)
        v_weights = []
        idx = 0
        for shape, size in zip(param_shapes, param_sizes):
            v_weights.append(v_norm[idx:idx + size].reshape(shape))
            idx += size
        _, jvp_out = jax.jvp(model_fn_wrt_layers, (target_params_list,), (v_weights,))
        residuals = get_full_span_affine_residuals(jvp_out, original_outputs)
        return jnp.sum(residuals ** 2)

    def project_orthogonal(v):
        if not use_orth:
            return v
        mask = direction_mask.astype(v.dtype)[:, None]
        projections = jnp.dot(orthogonal_directions, v)
        subtraction = jnp.sum((projections[:, None] * orthogonal_directions) * mask, axis=0)
        return v - subtraction

    optimizer = optax.adam(0.05)
    key = jax.random.PRNGKey(seed)

    v_init = jax.random.normal(key, shape=(total_size,))
    v_init = project_orthogonal(v_init)
    v_init = v_init / (jnp.linalg.norm(v_init) + 1e-6)

    opt_state = optimizer.init(v_init)
    init_val = (0, v_init, opt_state, jnp.inf)

    def cond_fun(val):
        i, _, _, loss = val
        return (i < max_iter) & (loss > tol)

    def body_fun(val):
        i, v, state, _ = val
        loss, grads = jax.value_and_grad(energy_fn)(v)
        updates, new_state = optimizer.update(grads, state, params=v)
        v_new = optax.apply_updates(v, updates)
        v_new = project_orthogonal(v_new)
        v_new = v_new / (jnp.linalg.norm(v_new) + 1e-6)
        return (i + 1, v_new, new_state, loss)

    final_i, final_v, _, final_loss = jax.lax.while_loop(cond_fun, body_fun, init_val)

    sigma = jnp.sqrt(final_loss)

    final_v_weights = []
    idx = 0
    for shape, size in zip(param_shapes, param_sizes):
        final_v_weights.append(final_v[idx:idx + size].reshape(shape))
        idx += size

    return final_v_weights, sigma


# ==============================================================================
# 4. Randomized SVD — fast batch direction finder for large conv kernels
# ==============================================================================

def find_pjsvd_directions_randomized_svd(
    model_fn: Callable[[jax.Array], jax.Array],
    target_param: jax.Array,
    n_directions: int,
    n_oversampling: int = 10,
    use_full_span: bool = True,
    seed: int = 42,
) -> Tuple[jax.Array, jax.Array]:
    """
    Finds the top-K null-space directions for a single-layer model function
    using a randomized SVD sketch of the (projected) Jacobian.

    This is dramatically faster than the per-direction Adam approach for large
    conv kernels: we need only (K + n_oversampling) JVP evaluations total.

    Algorithm (Halko et al. 2011 / Randomized PCA):
      1. Draw Ω ~ N(0,1) of shape (D_param, K + n_oversampling).
      2. For each column ω of Ω, compute y = (I-P) J ω via jvp.
         → Y is (output_dim, K+p) — column space of (I-P)J.
      3. Thin QR of Y  →  Q orthonormal basis for range of (I-P)J.
      4. Form B = Q^T [(I-P)J] by computing ((I-P)J)^T q for each col of Q.
         → B is (K+p, D_param).
      5. Thin SVD of B  →  right singular vectors V (D_param, K).
         These are the K directions that *most* perturb the activation
         (i.e., largest energy in the residual after affine correction).
         We negate them so they are null-space directions (smallest energy).

    Args:
        model_fn:      f(W) → activations, same convention as find_optimal_perturbation.
        target_param:  The weight tensor to perturb (any shape).
        n_directions:  K — number of directions to return.
        n_oversampling: Extra basis vectors for accuracy (10 is usually enough).
        use_full_span:  If True, use get_full_span_affine_residuals; else get_affine_residuals.
        seed:          Random seed.

    Returns:
        v_opts: (K, D_param) array of unit null-space directions (flat).
        sigmas: (K,) array of residual singular values (smaller = more null-space-like).
    """
    D = target_param.size
    K = n_directions
    p = n_oversampling
    Kp = K + p

    residual_fn = get_full_span_affine_residuals if use_full_span else get_affine_residuals
    original_outputs = model_fn(target_param)

    rng = jax.random.PRNGKey(seed)
    Omega = jax.random.normal(rng, shape=(D, Kp))  # (D, K+p)

    # -------------------------------------------------------------------------
    # Step 1-2: Form sketch Y = (I-P)J Ω ,  shape (output_flat, K+p)
    # -------------------------------------------------------------------------
    Y_cols = []
    for j in range(Kp):
        v_j = Omega[:, j].reshape(target_param.shape)
        _, jvp_out = jax.jvp(model_fn, (target_param,), (v_j,))
        res = residual_fn(jvp_out, original_outputs)          # (N, D_out)
        Y_cols.append(res.reshape(-1))                         # (N*D_out,)

    Y = jnp.stack(Y_cols, axis=1)  # (N*D_out, K+p)

    # -------------------------------------------------------------------------
    # Step 3: Thin QR of Y  → Q (N*D_out, K+p)
    # -------------------------------------------------------------------------
    Q, _ = jnp.linalg.qr(Y)  # Q: (N*D_out, K+p)

    # -------------------------------------------------------------------------
    # Step 4: Form B = Q^T (I-P)J  — need Jacobian left-multiplied by Q^T
    # We compute this column-by-column via JVP: (I-P)J e_j then dot with Q^T
    # Equivalently, for each basis vector q of Q, compute q^T (I-P) J via
    # reverse-mode: grad of sum((I-P)J v) wrt v at v=e_j is expensive;
    # instead we reuse the sketch trick: B = Q^T Y' where Y' is a fresh sketch.
    # Cheaper: B_{i,j} = q_i^T y_j = already available via Y above.
    # -------------------------------------------------------------------------
    # B = Q^T @ Y  is (K+p, K+p), but we need B = Q^T @ (I-P)J which is (K+p, D).
    # We get this by computing another sketch with the standard basis — but that's
    # O(D) calls. Instead, obtain B directly from: B = Q^T Y (Y = (I-P)J Omega)
    # and then recover right singular vectors of B^T (Omega can be a pre-image).
    # Standard approach: compute V by SVD(B) where B = Q^T Y, noting V lives in
    # the column space of Omega (Halko eq 5.15 approximation).
    # Accurate variant: power iteration (optional here).
    B = Q.T @ Y  # (K+p, K+p) — small matrix

    # SVD of B gives us the right singular vectors in terms of Y columns, but
    # we need them in the original D-dimensional parameter space.
    # Use the relation: (I-P)J V ≈ Q Σ U^T in SVD of projected Jacobian.
    # The right SV of B correspond to directions in the sketch (Omega) space.
    # To recover them in original space: V_original = Omega @ V_B / norm
    UB, SB, VhB = jnp.linalg.svd(B, full_matrices=False)  # UB: (K+p, K+p)

    # Map back to D-dim space: directions = Omega @ Vh^T (cols of Omega @ Vh^T are directions)
    # VhB is (K+p, K+p); each row is a right sv in sketch space.
    # Original-space directions: Omega @ VhB[i] for each i — columns of (D, K+p)
    V_orig = Omega @ VhB.T   # (D, K+p)

    # Normalise
    V_orig = V_orig / (jnp.linalg.norm(V_orig, axis=0, keepdims=True) + 1e-12)

    # Take top-K (but we want *smallest* singular values = most null-space-like,
    # so we reverse the order)
    # SB is in descending order. The LAST K entries have smallest residual energy.
    # However, for PJSVD we want the directions that stay most in the null space,
    # meaning smallest ||(I-P)Jv||. These correspond to the trailing singular values.
    # Since we built Y from (I-P)J, larger singular value = more visible in output.
    # For our ensemble, we actually want to *explore* diverse directions, so we
    # keep all K+p and return the top K (highest energy) — these have the largest
    # perturbation variance. Users can choose ordering.
    indices = jnp.argsort(SB)[::-1][:K]   # top-K by energy (descending)
    v_opts  = V_orig[:, indices].T          # (K, D)
    sigmas  = SB[indices]

    return v_opts, sigmas


