import jax
import jax.numpy as jnp
import numpy as np
from scipy.sparse.linalg import eigsh, LinearOperator
from typing import Callable, Tuple, List
import time


def extract_patches(x: jax.Array, k: int = 3, strides: int = 1) -> jax.Array:
    """
    Given x of shape (N, H, W, C), returns patches of shape (N*H*W, C*k*k).
    """
    patches = jax.lax.conv_general_dilated_patches(
        lhs=x,
        filter_shape=(k, k),
        window_strides=(strides, strides),
        padding='SAME',
        dimension_numbers=('NHWC', 'OIHW', 'NHWC')
    )
    return patches.reshape(-1, patches.shape[-1])


def _pad_and_stack(chunks: List[jax.Array]) -> Tuple[jax.Array, jax.Array]:
    """
    Stack a list of chunks into a single (n_chunks, chunk_size, ...) array.
    The last chunk is zero-padded to match the leading chunk size if needed.
    Returns padded chunks and a boolean mask (n_chunks, chunk_size) where True indicates valid (non-padded) rows.
    """
    if not chunks:
        raise ValueError("chunks list is empty")
    chunk_size = chunks[0].shape[0]
    n_chunks = len(chunks)
    padded_chunks = []
    mask = jnp.zeros((n_chunks, chunk_size), dtype=bool)
    
    for i, chunk in enumerate(chunks):
        if chunk.shape[0] < chunk_size:
            pad = jnp.zeros(
                (chunk_size - chunk.shape[0],) + chunk.shape[1:], dtype=chunk.dtype
            )
            padded_chunk = jnp.concatenate([chunk, pad], axis=0)
            mask = mask.at[i, :chunk.shape[0]].set(True)
        else:
            padded_chunk = chunk
            mask = mask.at[i, :].set(True)
        padded_chunks.append(padded_chunk)
    
    return jnp.stack(padded_chunks), mask



# ── JIT Kernels for Chunked Operations ────────────────────────────────────

@jax.jit(static_argnums=(0,))
def _geo_body_jit(get_Y_fn, w1_orig, G_acc, x_i, mask_i):
    Y_i = get_Y_fn(w1_orig, x_i)
    ones = jnp.ones((Y_i.shape[0], 1), dtype=jnp.float32)
    M_i = jnp.concatenate([ones, Y_i], axis=1)
    patches_per_sample = Y_i.shape[0] // x_i.shape[0]
    mask_expanded = jnp.repeat(mask_i, patches_per_sample)
    M_i = M_i * mask_expanded[:, None]
    return G_acc + (M_i.T @ M_i), None

@jax.jit(static_argnums=(0,))
def _pass1_body_jit(get_Y_fn, w1_orig, v, c_acc, x_i, mask_i):
    def f(w):
        return get_Y_fn(w, x_i)
    Y_i, Jv_i = jax.jvp(f, (w1_orig,), (v,))
    ones = jnp.ones((Y_i.shape[0], 1), dtype=jnp.float32)
    M_i = jnp.concatenate([ones, Y_i], axis=1)
    patches_per_sample = Y_i.shape[0] // x_i.shape[0]
    mask_expanded = jnp.repeat(mask_i, patches_per_sample)
    M_i = M_i * mask_expanded[:, None]
    Jv_i = Jv_i * mask_expanded[:, None]
    return c_acc + M_i.T @ Jv_i, None

@jax.jit(static_argnums=(0,))
def _pass2_body_jit(get_Y_fn, w1_orig, v, alpha, r_acc, x_i, mask_i):
    def f(w):
        return get_Y_fn(w, x_i)
    Y_i, Jv_i = jax.jvp(f, (w1_orig,), (v,))
    _, vjp_fn = jax.vjp(f, w1_orig)
    ones = jnp.ones((Y_i.shape[0], 1), dtype=jnp.float32)
    M_i = jnp.concatenate([ones, Y_i], axis=1)
    r_i = Jv_i - M_i @ alpha
    patches_per_sample = Y_i.shape[0] // x_i.shape[0]
    mask_expanded = jnp.repeat(mask_i, patches_per_sample)
    r_i = r_i * mask_expanded[:, None]
    (JT_ri,) = vjp_fn(r_i)
    return r_acc + JT_ri.reshape(-1), None

@jax.jit(static_argnums=(0,))
def _ridge_body_jit(get_Y_fn, w1_pert, w2_flat_orig, carry, x_i, T_i, mask_i):
    """
    Accumulates H = sum Y^T Y and b = sum Y^T (T - Y*w_orig).
    Solving (H + lambda I) delta_w = b gives the correction to add to w_orig.
    """
    Y_v = get_Y_fn(w1_pert, x_i)
    ones = jnp.ones((Y_v.shape[0], 1), dtype=jnp.float32)
    # T_tilde includes bias (ones) as the first feature.
    # We solve for delta_w (features) and delta_b (ones).
    M_i = jnp.concatenate([ones, Y_v], axis=1)
    
    # Residual error of the perturbed model relative to original target
    T_i_flat = T_i.reshape(-1, T_i.shape[-1])
    R_i = T_i_flat - Y_v @ w2_flat_orig
    
    patches_per_sample = Y_v.shape[0] // x_i.shape[0]
    mask_expanded = jnp.repeat(mask_i, patches_per_sample)
    M_i = M_i * mask_expanded[:, None]
    R_i = R_i * mask_expanded[:, None]
    
    H_acc, b_acc = carry
    return (H_acc + M_i.T @ M_i,
            b_acc + M_i.T @ R_i), None


def build_chunked_geometry(
    get_Y_fn: Callable[[jax.Array, jax.Array], jax.Array],
    w1_orig: jax.Array,
    chunks: List[jax.Array],
) -> jax.Array:
    """
    Accumulates G = sum_i M_i^T M_i where M_i = [1, Y_i].
    """
    chunks_stacked, mask = _pad_and_stack(chunks)
    Y0 = get_Y_fn(w1_orig, chunks[0][:1])
    D_M = 1 + Y0.shape[-1]

    def body(G_acc, args):
        x_i, mask_i = args
        return _geo_body_jit(get_Y_fn, w1_orig, G_acc, x_i, mask_i)

    print("Building chunked geometry (scan)...")
    G, _ = jax.lax.scan(
        body, jnp.zeros((D_M, D_M), jnp.float32), (chunks_stacked, mask)
    )
    G_inv = jnp.linalg.inv(G + 1e-6 * jnp.eye(D_M))
    return G_inv


def find_random_directions(
    flat_dim: int,
    K: int,
    seed: int = 99,
) -> Tuple[jax.Array, jax.Array]:
    """
    Generates K random orthonormal directions as a baseline.
    """
    rng = np.random.RandomState(seed)
    V = rng.standard_normal((K, flat_dim)).astype(np.float32)
    V, _ = np.linalg.qr(V.T)
    V = V.T.astype(np.float32)
    sigmas = jnp.ones(K, dtype=jnp.float32)
    return jnp.array(V), sigmas


def find_pnc_subspace_lanczos(
    get_Y_fn: Callable[[jax.Array, jax.Array], jax.Array],
    w1_orig: jax.Array,
    chunks: List[jax.Array],
    K: int,
    backend: str = "projected_residual",
    seed: int = 42,
) -> Tuple[jax.Array, jax.Array]:
    """
    Finds the bottom K eigenvectors of A^T A via ARPACK eigsh.
    """
    if backend not in {"projected_residual", "activation_covariance"}:
        raise ValueError(f"Unsupported backend: {backend}")
    if backend == "activation_covariance":
        print(
            "find_pnc_subspace_lanczos: activation_covariance requested, "
            "falling back to projected_residual implementation."
        )
    G_inv = build_chunked_geometry(get_Y_fn, w1_orig, chunks)
    chunks_stacked, mask = _pad_and_stack(chunks)
    D = int(w1_orig.size)
    Y0 = get_Y_fn(w1_orig, chunks[0][:1])
    D_Y = Y0.shape[-1]
    D_M = 1 + D_Y

    def jax_matvec(v_flat):
        v = v_flat.reshape(w1_orig.shape)
        
        def p1_body(c_acc, args):
            x_i, mask_i = args
            return _pass1_body_jit(get_Y_fn, w1_orig, v, c_acc, x_i, mask_i)
            
        c, _ = jax.lax.scan(
            p1_body, jnp.zeros((D_M, D_Y), jnp.float32), (chunks_stacked, mask)
        )
        alpha = G_inv @ c

        def p2_body(r_acc, args):
            x_i, mask_i = args
            return _pass2_body_jit(get_Y_fn, w1_orig, v, alpha, r_acc, x_i, mask_i)

        result, _ = jax.lax.scan(
            p2_body, jnp.zeros(D, jnp.float32), (chunks_stacked, mask)
        )
        return result

    def matvec_np(v):
        return np.array(jax_matvec(jnp.array(v, dtype=jnp.float32)))

    A_op_neg = LinearOperator((D, D), matvec=lambda v: -matvec_np(v), dtype=np.float32)
    np.random.seed(seed)
    v0 = np.random.randn(D).astype(np.float32)

    print("Warming up JIT kernel...")
    _ = jax_matvec(jnp.array(v0))
    print("JIT kernel compiled.")

    print(f"Running ARPACK eigsh for bottom {K} eigenvectors...")
    t0 = time.time()
    eigvals, eigvecs = eigsh(A_op_neg, k=K, which='LM', v0=v0, tol=1e-4)
    eigvals = -eigvals
    print(f"ARPACK done in {time.time() - t0:.1f}s")

    idx = np.argsort(eigvals)
    eigvecs = eigvecs[:, idx]
    eigvals = eigvals[idx]

    v_opts = jnp.array(eigvecs.T)
    return v_opts, jnp.array(eigvals)


def _ridge_regression_solve(H: jax.Array, b: jax.Array, lambda_reg: float, w2_orig: jax.Array) -> Tuple[jax.Array, jax.Array]:
    """Shared linear algebra for ridge solution."""
    D_M = H.shape[0]
    C_out = b.shape[1]
    reg = H + lambda_reg * jnp.eye(D_M)
    Theta_delta = jnp.linalg.solve(reg, b)  # (D_M, C_out)
    
    kh, kw, C_in, _ = w2_orig.shape
    # Delta-Bias (index 0)
    b2_new = Theta_delta[0:1, :]
    # Delta-Weights (index 1 onwards)
    w2_delta = Theta_delta[1:, :].reshape(kh, kw, C_in, C_out)
    
    # Corrected weights = Original + Delta
    w2_new = w2_orig + w2_delta
    return w2_new, b2_new


def solve_chunked_conv2_correction(
    get_Y_fn: Callable[[jax.Array, jax.Array], jax.Array],
    w1_pert: jax.Array,
    w2_orig: jax.Array,
    chunks: List[jax.Array],
    T_orig_chunks: List[jax.Array],
    lambda_reg: float = 1e-3,
) -> Tuple[jax.Array, jax.Array]:
    """
    Ridge regression for new Conv2 weights, iterating over chunks to save memory.
    """
    kh, kw, C_in, C_out = w2_orig.shape
    # Determine D_M from first chunk
    Y0 = get_Y_fn(w1_pert, chunks[0][:1])
    D_M = Y0.shape[-1] + 1
    w2_flat_orig = w2_orig.reshape(-1, C_out)

    H = jnp.zeros((D_M, D_M), jnp.float32)
    b = jnp.zeros((D_M, C_out), jnp.float32)

    print(f"Solving ridge regression (iterative, delta-reg={lambda_reg})...")
    for x_i, T_i in zip(chunks, T_orig_chunks):
        # We assume chunks are already correct size or not needing masking 
        # (the Luigi tasks produce clean chunks).
        mask_i = jnp.ones(x_i.shape[0], dtype=bool)
        (H, b), _ = _ridge_body_jit(get_Y_fn, w1_pert, w2_flat_orig, (H, b), x_i, T_i, mask_i)
        
    return _ridge_regression_solve(H, b, lambda_reg, w2_orig)
