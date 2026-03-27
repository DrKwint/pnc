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


def _pad_and_stack(chunks: List[jax.Array]) -> jax.Array:
    """
    Stack a list of chunks into a single (n_chunks, chunk_size, ...) array.
    The last chunk is zero-padded to match the leading chunk size if needed.
    """
    if not chunks:
        raise ValueError("chunks list is empty")
    chunk_size = chunks[0].shape[0]
    last = chunks[-1]
    if last.shape[0] < chunk_size:
        pad = jnp.zeros(
            (chunk_size - last.shape[0],) + last.shape[1:], dtype=last.dtype
        )
        chunks = list(chunks[:-1]) + [jnp.concatenate([last, pad], axis=0)]
    return jnp.stack(chunks)


def build_chunked_geometry(
    get_Y_fn: Callable[[jax.Array, jax.Array], jax.Array],
    w1_orig: jax.Array,
    chunks: List[jax.Array],
) -> jax.Array:
    """
    Accumulates G = sum_i M_i^T M_i where M_i = [1, Y_i].
    """
    # Determine D_M from one sample forward pass.
    Y0 = get_Y_fn(w1_orig, chunks[0][:1])
    D_M = 1 + Y0.shape[-1]  # [bias, Y]

    @jax.jit
    def geo_step(G_acc, x_i):
        Y_i = get_Y_fn(w1_orig, x_i)
        ones = jnp.ones((Y_i.shape[0], 1), dtype=jnp.float32)
        M_i = jnp.concatenate([ones, Y_i], axis=1)
        return G_acc + (M_i.T @ M_i)

    G = jnp.zeros((D_M, D_M), jnp.float32)
    for x_i in chunks:
        G = geo_step(G, x_i)
        
    G = G + 1e-6 * jnp.eye(D_M)
    G_inv = jnp.linalg.inv(G)
    return G_inv


def find_pnc_subspace_lanczos(
    get_Y_fn: Callable[[jax.Array, jax.Array], jax.Array],
    w1_orig: jax.Array,
    chunks: List[jax.Array],
    K: int,
    seed: int = 42,
) -> Tuple[jax.Array, jax.Array]:
    """
    Finds the bottom K eigenvectors of A^T A via ARPACK eigsh.
    """
    G_inv = build_chunked_geometry(get_Y_fn, w1_orig, chunks)

    D = int(w1_orig.size)
    Y0 = get_Y_fn(w1_orig, chunks[0][:1])
    D_Y = Y0.shape[-1]
    D_M = 1 + D_Y

    @jax.jit
    def jax_matvec_pass1(v, x_i):
        def f(w):
            return get_Y_fn(w, x_i)
        Y_i, Jv_i = jax.jvp(f, (w1_orig,), (v,))
        ones = jnp.ones((Y_i.shape[0], 1), dtype=jnp.float32)
        M_i = jnp.concatenate([ones, Y_i], axis=1)
        return M_i.T @ Jv_i

    @jax.jit
    def jax_matvec_pass2(v, x_i, alpha):
        def f(w):
            return get_Y_fn(w, x_i)
        Y_i, Jv_i = jax.jvp(f, (w1_orig,), (v,))
        _, vjp_fn = jax.vjp(f, w1_orig)
        ones = jnp.ones((Y_i.shape[0], 1), dtype=jnp.float32)
        M_i = jnp.concatenate([ones, Y_i], axis=1)
        r_i = Jv_i - M_i @ alpha
        (JT_ri,) = vjp_fn(r_i)
        return JT_ri.reshape(-1)

    def jax_matvec(v_flat: jax.Array) -> jax.Array:
        v = v_flat.reshape(w1_orig.shape)
        
        c = jnp.zeros((D_M, D_Y), jnp.float32)
        for x_i in chunks:
            c = c + jax_matvec_pass1(v, x_i)
            
        alpha = G_inv @ c
        
        result = jnp.zeros(D, jnp.float32)
        for x_i in chunks:
            result = result + jax_matvec_pass2(v, x_i, alpha)
            
        return result

    def matvec_np(v: np.ndarray) -> np.ndarray:
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


def solve_chunked_conv2_correction(
    get_Y_fn: Callable[[jax.Array, jax.Array], jax.Array],
    w1_pert: jax.Array,
    chunks: List[jax.Array],
    T_orig_chunks: List[jax.Array],
    w2_shape: tuple,
    lambda_reg: float = 1e-3,
) -> Tuple[jax.Array, jax.Array]:
    """
    Ridge regression for new Conv2 weights, avoiding full array stacking.
    """
    kh, kw, C_in, C_out = w2_shape
    Y0 = get_Y_fn(w1_pert, chunks[0][:1])
    D_Y = Y0.shape[-1]
    D_M = D_Y + 1

    @jax.jit
    def ridge_step(H_acc, b_acc, x_i, T_i, w1_arg):
        Y_v = get_Y_fn(w1_arg, x_i)
        ones = jnp.ones((Y_v.shape[0], 1), dtype=jnp.float32)
        Y_tilde = jnp.concatenate([Y_v, ones], axis=1)
        T_i_flat = T_i.reshape(-1, T_i.shape[-1])
        return H_acc + Y_tilde.T @ Y_tilde, b_acc + Y_tilde.T @ T_i_flat

    H = jnp.zeros((D_M, D_M), jnp.float32)
    b = jnp.zeros((D_M, C_out), jnp.float32)
    
    for x_i, T_i in zip(chunks, T_orig_chunks):
        H, b = ridge_step(H, b, x_i, T_i, w1_pert)

    reg = H + lambda_reg * jnp.eye(D_M)
    Theta = jnp.linalg.solve(reg, b)   # (D_M, C_out)

    W2_flat = Theta[:-1, :]             # (D_Y, C_out) = (C_in*kh*kw, C_out)
    b2_new  = Theta[-1, :]              # (C_out,)

    W2_new = W2_flat.reshape(C_in, kh, kw, C_out).transpose(1, 2, 0, 3)
    return W2_new, b2_new
