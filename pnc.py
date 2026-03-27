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


def build_chunked_geometry(
    get_Y_fn: Callable[[jax.Array, jax.Array], jax.Array],
    w1_orig: jax.Array,
    chunks: List[jax.Array],
) -> jax.Array:
    """
    Accumulates G = sum_i M_i^T M_i where M_i = [1, Y_i], via a single
    JIT-compiled lax.scan pass.  Returns G_inv = G^{-1}.
    """
    chunks_stacked, mask = _pad_and_stack(chunks)

    # Determine D_M from one sample forward pass (outside JIT).
    Y0 = get_Y_fn(w1_orig, chunks[0][:1])
    D_M = 1 + Y0.shape[-1]  # [bias, Y]

    def geo_body(G_acc, args):
        x_i, mask_i = args
        Y_i = get_Y_fn(w1_orig, x_i)
        ones = jnp.ones((Y_i.shape[0], 1), dtype=jnp.float32)
        M_i = jnp.concatenate([ones, Y_i], axis=1)
        # Mask out padded rows
        M_i = M_i * mask_i[:, None]
        return G_acc + (M_i.T @ M_i), None

    @jax.jit
    def run_geo(chunks_arr, mask_arr):
        G, _ = jax.lax.scan(geo_body, jnp.zeros((D_M, D_M), jnp.float32), (chunks_arr, mask_arr))
        return G

    print("Building chunked geometry (scan)...")
    G = run_geo(chunks_stacked, mask)
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

    The matvec A^T A v is evaluated in two lax.scan passes (compiled once):
      Pass 1  – c      = sum_i  M_i^T J_i v           (JVP per chunk)
      Pass 2  – result = sum_i  J_i^T (J_i v - M_i alpha) (JVP+VJP per chunk;
                XLA CSE fuses the two forward passes within each scan body)
    A JIT warmup call before ARPACK ensures all compilation is done upfront.
    """
    print("Building chunked geometry...")
    G_inv = build_chunked_geometry(get_Y_fn, w1_orig, chunks)

    chunks_stacked, mask = _pad_and_stack(chunks)
    D = int(w1_orig.size)

    Y0 = get_Y_fn(w1_orig, chunks[0][:1])
    D_Y = Y0.shape[-1]
    D_M = 1 + D_Y

    @jax.jit
    def jax_matvec(v_flat: jax.Array) -> jax.Array:
        v = v_flat.reshape(w1_orig.shape)

        # ── Pass 1: c = Σ M_i^T Jv_i ──────────────────────────────────────
        def pass1_body(c_acc, args):
            x_i, mask_i = args
            def f(w):
                return get_Y_fn(w, x_i)
            Y_i, Jv_i = jax.jvp(f, (w1_orig,), (v,))
            ones = jnp.ones((Y_i.shape[0], 1), dtype=jnp.float32)
            M_i = jnp.concatenate([ones, Y_i], axis=1)
            # Mask out padded rows
            M_i = M_i * mask_i[:, None]
            Jv_i = Jv_i * mask_i[:, None]
            return c_acc + M_i.T @ Jv_i, None

        c, _ = jax.lax.scan(
            pass1_body, jnp.zeros((D_M, D_Y), jnp.float32), (chunks_stacked, mask)
        )
        alpha = G_inv @ c  # (D_M, D_Y)

        # ── Pass 2: result = Σ J_i^T r_i, r_i = Jv_i - M_i alpha ─────────
        # jax.vjp is fully compatible with jax.jit + lax.scan.
        # XLA CSE will fuse the forward passes from jvp and vjp in each body.
        def pass2_body(r_acc, args):
            x_i, mask_i = args
            def f(w):
                return get_Y_fn(w, x_i)

            Y_i, Jv_i = jax.jvp(f, (w1_orig,), (v,))
            _, vjp_fn = jax.vjp(f, w1_orig)
            ones = jnp.ones((Y_i.shape[0], 1), dtype=jnp.float32)
            M_i = jnp.concatenate([ones, Y_i], axis=1)
            r_i = Jv_i - M_i @ alpha  # (S, D_Y)
            # Mask out padded rows
            r_i = r_i * mask_i[:, None]
            (JT_ri,) = vjp_fn(r_i)
            return r_acc + JT_ri.reshape(-1), None

        result, _ = jax.lax.scan(
            pass2_body, jnp.zeros(D, jnp.float32), (chunks_stacked, mask)
        )
        return result

    def matvec_np(v: np.ndarray) -> np.ndarray:
        return np.array(jax_matvec(jnp.array(v, dtype=jnp.float32)))

    A_op = LinearOperator((D, D), matvec=matvec_np, dtype=np.float32)
    # Negate so that ARPACK's 'LM' (fastest mode) finds the K smallest
    # eigenvalues of A^T A.  ARPACK's 'SM' requires O(D) Arnoldi steps;
    # 'LM' on −A^T A converges in O(K) steps.  Eigenvectors are identical.
    neg_matvec_np = lambda v: -matvec_np(v)
    A_op_neg = LinearOperator((D, D), matvec=neg_matvec_np, dtype=np.float32)

    np.random.seed(seed)
    v0 = np.random.randn(D).astype(np.float32)

    # Warmup: compile the JIT kernel before handing control to ARPACK.
    print("Warming up JIT kernel...")
    _ = jax_matvec(jnp.array(v0))
    jax.effects_barrier()
    print("JIT kernel compiled.")

    print(f"Running ARPACK eigsh for bottom {K} eigenvectors...")
    t0 = time.time()
    eigvals, eigvecs = eigsh(A_op_neg, k=K, which='LM', v0=v0, tol=1e-4)
    eigvals = -eigvals  # flip sign back to A^T A eigenvalues
    print(f"ARPACK done in {time.time() - t0:.1f}s")

    idx = np.argsort(eigvals)
    eigvecs = eigvecs[:, idx]
    eigvals = eigvals[idx]

    v_opts = jnp.array(eigvecs.T)  # (K, D)
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
    Recomputes Y with perturbed Conv1 and solves ridge regression for new Conv2
    weights, using a single JIT-compiled lax.scan pass over chunk pairs.

    Returns:
        W2_new : kernel of shape w2_shape = (kh, kw, C_in, C_out)
        b2_new : bias of shape (C_out,)

    Patch ordering note: conv_general_dilated_patches with
    ('NHWC','OIHW','NHWC') yields last dim ordered as (C_in, kh, kw), so
    W2_flat (D, C_out) is reshaped as (C_in, kh, kw, C_out) then transposed
    to (kh, kw, C_in, C_out).
    """
    chunks_stacked, mask = _pad_and_stack(chunks)
    T_stacked, _ = _pad_and_stack(T_orig_chunks)

    kh, kw, C_in, C_out = w2_shape
    Y0 = get_Y_fn(w1_pert, chunks[0][:1])
    D_Y = Y0.shape[-1]
    D_M = D_Y + 1  # Y then bias (convention of original)

    @jax.jit
    def run_ridge(chunks_arr, T_arr, mask_arr, w1_arg):
        def ridge_body(carry, args):
            x_i, T_i, mask_i = args
            Y_v = get_Y_fn(w1_arg, x_i)
            ones = jnp.ones((Y_v.shape[0], 1), dtype=jnp.float32)
            Y_tilde = jnp.concatenate([Y_v, ones], axis=1)       # (S, D_M)
            T_i_flat = T_i.reshape(-1, T_i.shape[-1])            # (S, C_out)
            # Mask out padded rows
            Y_tilde = Y_tilde * mask_i[:, None]
            T_i_flat = T_i_flat * mask_i[:, None]
            H_acc, b_acc = carry
            return (H_acc + Y_tilde.T @ Y_tilde,
                    b_acc + Y_tilde.T @ T_i_flat), None

        (H, b), _ = jax.lax.scan(
            ridge_body,
            (jnp.zeros((D_M, D_M), jnp.float32),
             jnp.zeros((D_M, C_out), jnp.float32)),
            (chunks_arr, T_arr, mask_arr),
        )
        return H, b

    print("Solving ridge regression (scan)...")
    H, b = run_ridge(chunks_stacked, T_stacked, mask, w1_pert)

    reg = H + lambda_reg * jnp.eye(D_M)
    Theta = jnp.linalg.solve(reg, b)   # (D_M, C_out)

    W2_flat = Theta[:-1, :]             # (D_Y, C_out) = (C_in*kh*kw, C_out)
    b2_new  = Theta[-1, :]              # (C_out,)

    W2_new = W2_flat.reshape(C_in, kh, kw, C_out).transpose(1, 2, 0, 3)
    return W2_new, b2_new
