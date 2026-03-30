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
        padding="SAME",
        dimension_numbers=("NHWC", "OIHW", "NHWC"),
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
            mask = mask.at[i, : chunk.shape[0]].set(True)
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
        G, _ = jax.lax.scan(
            geo_body, jnp.zeros((D_M, D_M), jnp.float32), (chunks_arr, mask_arr)
        )
        return G

    print("Building chunked geometry (scan)...")
    G = run_geo(chunks_stacked, mask)
    G = G + 1e-6 * jnp.eye(D_M)
    G_inv = jnp.linalg.inv(G)
    return G_inv


import abc


class SafeSubspaceOperator(abc.ABC):
    @abc.abstractmethod
    def apply_R(self, v_flat: jax.Array) -> jax.Array:
        pass

    @abc.abstractmethod
    def apply_RT(self, u: jax.Array) -> jax.Array:
        pass

    def matvec(self, v_flat: jax.Array) -> jax.Array:
        return self.apply_RT(self.apply_R(v_flat))

    def find_singular_directions(
        self, K: int, seed: int = 42
    ) -> Tuple[jax.Array, jax.Array]:
        """Find the bottom K right-singular vectors."""
        D = self.D

        # We can jit the matvec
        @jax.jit
        def jax_matvec(v_flat: jax.Array) -> jax.Array:
            return self.matvec(v_flat)

        def matvec_np(v: np.ndarray) -> np.ndarray:
            return np.array(jax_matvec(jnp.array(v, dtype=jnp.float32)))

        A_op = LinearOperator((D, D), matvec=matvec_np, dtype=np.float32)
        # ARPACK 'LM' on -A^T A finds the K smallest eigenvalues
        neg_matvec_np = lambda v: -matvec_np(v)
        A_op_neg = LinearOperator((D, D), matvec=neg_matvec_np, dtype=np.float32)

        np.random.seed(seed)
        v0 = np.random.randn(D).astype(np.float32)

        print("Warming up JIT kernel...")
        _ = jax_matvec(jnp.array(v0))
        jax.effects_barrier()
        print("JIT kernel compiled.")

        print(f"Running ARPACK eigsh for bottom {K} eigenvectors...")
        t0 = time.time()
        eigvals, eigvecs = eigsh(A_op_neg, k=K, which="LM", v0=v0, tol=1e-4)
        eigvals = -eigvals
        print(f"ARPACK done in {time.time() - t0:.1f}s")

        idx = np.argsort(eigvals)
        eigvecs = eigvecs[:, idx]
        eigvals = eigvals[idx]

        v_opts = jnp.array(eigvecs.T)
        return v_opts, jnp.array(eigvals)


class ActivationCovarianceOperator(SafeSubspaceOperator):
    def __init__(self, get_Y_fn, w1_orig, chunks_stacked, mask):
        self.get_Y_fn = get_Y_fn
        self.w1_orig = w1_orig
        self.chunks_stacked = chunks_stacked
        self.mask = mask
        Y0 = get_Y_fn(w1_orig, chunks_stacked[0][:1])
        self.D_Y = Y0.shape[-1]
        self.D = w1_orig.size

        @jax.jit
        def _apply_R(v_flat):
            v = v_flat.reshape(self.w1_orig.shape)

            def body(acc, args):
                x_i, mask_i = args

                def f(w):
                    return self.get_Y_fn(w, x_i)

                _, Jv_i = jax.jvp(f, (self.w1_orig,), (v,))
                Jv_i = Jv_i * mask_i[:, None]
                return None, Jv_i

            _, all_Jv = jax.lax.scan(body, None, (self.chunks_stacked, self.mask))
            return all_Jv

        @jax.jit
        def _apply_RT(u):
            def body(acc, args):
                x_i, mask_i, u_i = args

                def f(w):
                    return self.get_Y_fn(w, x_i)

                _, vjp_fn = jax.vjp(f, self.w1_orig)
                u_i = u_i * mask_i[:, None]
                (JT_u_i,) = vjp_fn(u_i)
                return acc + JT_u_i.reshape(-1), None

            result, _ = jax.lax.scan(
                body,
                jnp.zeros(self.D, jnp.float32),
                (self.chunks_stacked, self.mask, u),
            )
            return result

        # Fast fused matvec for eigsh
        @jax.jit
        def _fused_matvec(v_flat):
            v = v_flat.reshape(self.w1_orig.shape)

            def body(acc, args):
                x_i, mask_i = args

                def f(w):
                    return self.get_Y_fn(w, x_i)

                _, Jv_i = jax.jvp(f, (self.w1_orig,), (v,))
                _, vjp_fn = jax.vjp(f, self.w1_orig)
                Jv_i = Jv_i * mask_i[:, None]
                (JT_u_i,) = vjp_fn(Jv_i)
                return acc + JT_u_i.reshape(-1), None

            result, _ = jax.lax.scan(
                body, jnp.zeros(self.D, jnp.float32), (self.chunks_stacked, self.mask)
            )
            return result

        self._apply_R_jit = _apply_R
        self._apply_RT_jit = _apply_RT
        self._fused_matvec_jit = _fused_matvec

    def apply_R(self, v_flat: jax.Array) -> jax.Array:
        return self._apply_R_jit(v_flat)

    def apply_RT(self, u: jax.Array) -> jax.Array:
        return self._apply_RT_jit(u)

    def matvec(self, v_flat: jax.Array) -> jax.Array:
        return self._fused_matvec_jit(v_flat)


class ProjectedResidualOperator(SafeSubspaceOperator):
    def __init__(self, get_Y_fn, w1_orig, chunks_stacked, mask, G_inv):
        self.get_Y_fn = get_Y_fn
        self.w1_orig = w1_orig
        self.chunks_stacked = chunks_stacked
        self.mask = mask
        self.G_inv = G_inv

        Y0 = get_Y_fn(w1_orig, chunks_stacked[0][:1])
        self.D_Y = Y0.shape[-1]
        self.D_M = 1 + self.D_Y
        self.D = w1_orig.size

        @jax.jit
        def _apply_R(v_flat):
            v = v_flat.reshape(self.w1_orig.shape)

            def pass1_body(c_acc, args):
                x_i, mask_i = args

                def f(w):
                    return self.get_Y_fn(w, x_i)

                Y_i, Jv_i = jax.jvp(f, (self.w1_orig,), (v,))
                ones = jnp.ones((Y_i.shape[0], 1), dtype=jnp.float32)
                M_i = jnp.concatenate([ones, Y_i], axis=1)
                M_i = M_i * mask_i[:, None]
                Jv_i = Jv_i * mask_i[:, None]
                return c_acc + M_i.T @ Jv_i, Jv_i

            c, all_Jv = jax.lax.scan(
                pass1_body,
                jnp.zeros((self.D_M, self.D_Y), jnp.float32),
                (self.chunks_stacked, self.mask),
            )
            alpha = self.G_inv @ c

            def pass2_body(acc, args):
                x_i, mask_i, Jv_i = args
                Y_i = self.get_Y_fn(self.w1_orig, x_i)
                ones = jnp.ones((Y_i.shape[0], 1), dtype=jnp.float32)
                M_i = jnp.concatenate([ones, Y_i], axis=1)
                r_i = Jv_i - M_i @ alpha
                r_i = r_i * mask_i[:, None]
                return acc, r_i

            _, all_r = jax.lax.scan(
                pass2_body, None, (self.chunks_stacked, self.mask, all_Jv)
            )
            return all_r

        @jax.jit
        def _apply_RT(u):
            def proj_body(c_acc, args):
                x_i, mask_i, u_i = args
                Y_i = self.get_Y_fn(self.w1_orig, x_i)
                ones = jnp.ones((Y_i.shape[0], 1), dtype=jnp.float32)
                M_i = jnp.concatenate([ones, Y_i], axis=1)
                M_i = M_i * mask_i[:, None]
                u_i = u_i * mask_i[:, None]
                return c_acc + M_i.T @ u_i, None

            c, _ = jax.lax.scan(
                proj_body,
                jnp.zeros((self.D_M, self.D_Y), jnp.float32),
                (self.chunks_stacked, self.mask, u),
            )
            alpha = self.G_inv @ c

            def jt_body(acc, args):
                x_i, mask_i, u_i = args

                def f(w):
                    return self.get_Y_fn(w, x_i)

                Y_i = f(self.w1_orig)
                ones = jnp.ones((Y_i.shape[0], 1), dtype=jnp.float32)
                M_i = jnp.concatenate([ones, Y_i], axis=1)

                r_i = u_i - M_i @ alpha
                r_i = r_i * mask_i[:, None]

                _, vjp_fn = jax.vjp(f, self.w1_orig)
                (JT_ri,) = vjp_fn(r_i)
                return acc + JT_ri.reshape(-1), None

            result, _ = jax.lax.scan(
                jt_body,
                jnp.zeros(self.D, jnp.float32),
                (self.chunks_stacked, self.mask, u),
            )
            return result

        @jax.jit
        def _fused_matvec(v_flat):
            # This is the exact original fused jax_matvec
            v = v_flat.reshape(self.w1_orig.shape)

            def pass1_body(c_acc, args):
                x_i, mask_i = args

                def f(w):
                    return self.get_Y_fn(w, x_i)

                Y_i, Jv_i = jax.jvp(f, (self.w1_orig,), (v,))
                ones = jnp.ones((Y_i.shape[0], 1), dtype=jnp.float32)
                M_i = jnp.concatenate([ones, Y_i], axis=1)
                M_i = M_i * mask_i[:, None]
                Jv_i = Jv_i * mask_i[:, None]
                return c_acc + M_i.T @ Jv_i, None

            c, _ = jax.lax.scan(
                pass1_body,
                jnp.zeros((self.D_M, self.D_Y), jnp.float32),
                (self.chunks_stacked, self.mask),
            )
            alpha = self.G_inv @ c

            def pass2_body(r_acc, args):
                x_i, mask_i = args

                def f(w):
                    return self.get_Y_fn(w, x_i)

                Y_i, Jv_i = jax.jvp(f, (self.w1_orig,), (v,))
                _, vjp_fn = jax.vjp(f, self.w1_orig)
                ones = jnp.ones((Y_i.shape[0], 1), dtype=jnp.float32)
                M_i = jnp.concatenate([ones, Y_i], axis=1)
                r_i = Jv_i - M_i @ alpha
                r_i = r_i * mask_i[:, None]
                (JT_ri,) = vjp_fn(r_i)
                return r_acc + JT_ri.reshape(-1), None

            result, _ = jax.lax.scan(
                pass2_body,
                jnp.zeros(self.D, jnp.float32),
                (self.chunks_stacked, self.mask),
            )
            return result

        self._apply_R_jit = _apply_R
        self._apply_RT_jit = _apply_RT
        self._fused_matvec_jit = _fused_matvec

    def apply_R(self, v_flat: jax.Array) -> jax.Array:
        return self._apply_R_jit(v_flat)

    def apply_RT(self, u: jax.Array) -> jax.Array:
        return self._apply_RT_jit(u)

    def matvec(self, v_flat: jax.Array) -> jax.Array:
        return self._fused_matvec_jit(v_flat)


def find_pnc_subspace_lanczos(
    get_Y_fn: Callable[[jax.Array, jax.Array], jax.Array],
    w1_orig: jax.Array,
    chunks: List[jax.Array],
    K: int,
    backend: str = "projected_residual",
    seed: int = 42,
) -> Tuple[jax.Array, jax.Array]:
    """
    Finds the bottom K eigenvectors using ARPACK for the specified backend operator.
    """
    chunks_stacked, mask = _pad_and_stack(chunks)

    if backend == "projected_residual":
        print("Building chunked geometry for projected residual...")
        # To avoid duplicating chunks_stacked allocation inside build_chunked_geometry,
        # we can just use the build_chunked_geometry directly, but let's just call it.
        # Wait, build_chunked_geometry calls _pad_and_stack internally. That's fine.
        G_inv = build_chunked_geometry(get_Y_fn, w1_orig, chunks)
        operator = ProjectedResidualOperator(
            get_Y_fn, w1_orig, chunks_stacked, mask, G_inv
        )
    elif backend == "activation_covariance":
        print("Using empirical activation covariance backend...")
        operator = ActivationCovarianceOperator(get_Y_fn, w1_orig, chunks_stacked, mask)
    else:
        raise ValueError(f"Unknown safe subspace backend: {backend}")

    return operator.find_singular_directions(K, seed=seed)


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
            Y_tilde = jnp.concatenate([Y_v, ones], axis=1)  # (S, D_M)
            T_i_flat = T_i.reshape(-1, T_i.shape[-1])  # (S, C_out)
            # Mask out padded rows
            Y_tilde = Y_tilde * mask_i[:, None]
            T_i_flat = T_i_flat * mask_i[:, None]
            H_acc, b_acc = carry
            return (H_acc + Y_tilde.T @ Y_tilde, b_acc + Y_tilde.T @ T_i_flat), None

        (H, b), _ = jax.lax.scan(
            ridge_body,
            (jnp.zeros((D_M, D_M), jnp.float32), jnp.zeros((D_M, C_out), jnp.float32)),
            (chunks_arr, T_arr, mask_arr),
        )
        return H, b

    print("Solving ridge regression (scan)...")
    H, b = run_ridge(chunks_stacked, T_stacked, mask, w1_pert)

    reg = H + lambda_reg * jnp.eye(D_M)
    Theta = jnp.linalg.solve(reg, b)  # (D_M, C_out)

    W2_flat = Theta[:-1, :]  # (D_Y, C_out) = (C_in*kh*kw, C_out)
    b2_new = Theta[-1, :]  # (C_out,)

    W2_new = W2_flat.reshape(C_in, kh, kw, C_out).transpose(1, 2, 0, 3)
    return W2_new, b2_new
