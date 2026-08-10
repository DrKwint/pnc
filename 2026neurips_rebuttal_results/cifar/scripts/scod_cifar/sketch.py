"""Randomized low-rank Fisher sketch for SCOD (JAX, memory-safe for an 8GB GPU).

The empirical (per-example, output-Fisher-weighted) Gauss-Newton / Fisher matrix is

    A = (1/N) sum_i  L_i^T L_i ,   L_i = d ztilde_i / d w   (shape C x P),

with P ~ 1.1e7. We never form A (P x P) or a full P x T test matrix on the host. Instead we
use the column-separability of Y = A Omega:

    Y[:, t] = (1/N) sum_i L_i^T ( L_i Omega[:, t] ),

so Omega is processed in GPU column-blocks (Omega_b: P x Tb, small), every per-example op stays
on the GPU (jacrev is ~0.017 s), and Y is streamed to the host block by block. Recovery uses the
Nystrom method, which needs only host contractions of Y touched O(1) times:

    B0 = Omega^T Y = (1/N) sum_i M_i^T M_i          (M_i = L_i Omega, tiny: C x T)
    A  ~= Y B0^{-1} Y^T
       => eigvecs U (P x k), eigvals lambda via a T x T generalized problem, U = Y Z.

This recovers the same leading eigenspace as Tropp's shifted sketch; the SCOD shrinkage
(scorer.py) is applied separately at scoring time. Validated to machine agreement against an
exact eigendecomposition on a tiny network (tests.py, Section 6.2).
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp


@dataclass
class SketchResult:
    eigvals: np.ndarray          # (k,) descending, eigenvalues of the averaged Fisher A
    basis: np.ndarray            # (P, k) float32, orthonormal-ish parameter-space eigenvectors
    T: int                       # sketch size (num_samples)
    k: int                       # retained rank (k_max)
    n_examples: int
    sketch_seed: int
    diagnostics: dict


def _omega_block(base_key, block_id: int, P: int, tb: int) -> jnp.ndarray:
    """Deterministic Gaussian test-matrix block Omega_b (P x tb) on device."""
    key = jax.random.fold_in(base_key, block_id)
    return jax.random.normal(key, (P, tb), dtype=jnp.float32)


def _rowchunked_gram(Y: np.ndarray, chunk: int = 1_000_000) -> np.ndarray:
    """Y^T Y (T x T) accumulated over P-row chunks -- avoids a 5.5GB BLAS transpose copy."""
    T = Y.shape[1]
    G = np.zeros((T, T), dtype=np.float64)
    for s in range(0, Y.shape[0], chunk):
        blk = Y[s:s + chunk]
        G += (blk.T @ blk).astype(np.float64)
    return G


def _rowchunked_matmul(Y: np.ndarray, Z: np.ndarray, chunk: int = 1_000_000) -> np.ndarray:
    """Y @ Z (P x k) computed in P-row chunks (Z is T x k). Result stays float32."""
    out = np.empty((Y.shape[0], Z.shape[1]), dtype=np.float32)
    Zf = Z.astype(np.float32)
    for s in range(0, Y.shape[0], chunk):
        out[s:s + chunk] = Y[s:s + chunk] @ Zf
    return out


def build_fisher_sketch(ztilde_fn, flat_w, X, *, C: int, T: int, k: int,
                        sketch_seed: int, block_cols: int = 31, jitter: float = 1e-10,
                        log_every: int = 256) -> SketchResult:
    """Build the rank-k Fisher sketch.

    ztilde_fn(flat_w, x1) -> (C,)   Fisher-weighted temperature-scaled logits for one image.
    X: (N, H, W, C_img) float32 calibration images (clean, deterministic transforms).
    """
    P = int(flat_w.shape[0])
    N = int(X.shape[0])
    base_key = jax.random.PRNGKey(sketch_seed)
    Xd = jnp.asarray(X)

    # Fused per-example kernel: materialise L_i on-GPU (transient), update Y_b and emit M_ib.
    @jax.jit
    def kernel(fw, x1, omega_b, yb):
        Li = jax.jacrev(ztilde_fn)(fw, x1)      # (C, P)
        mib = Li @ omega_b                       # (C, tb)
        yb = yb + Li.T @ mib                     # (P, tb)
        return yb, mib

    Y_host = np.zeros((P, T), dtype=np.float32)
    M = np.zeros((N, C, T), dtype=np.float32)     # tiny: N*C*T floats

    col_blocks = [(s, min(s + block_cols, T)) for s in range(0, T, block_cols)]
    t0 = time.time()
    for b, (c0, c1) in enumerate(col_blocks):
        tb = c1 - c0
        omega_b = _omega_block(base_key, b, P, tb)
        yb = jnp.zeros((P, tb), dtype=jnp.float32)
        for i in range(N):
            yb, mib = kernel(flat_w, Xd[i], omega_b, yb)
            M[i, :, c0:c1] = np.asarray(mib)
            if log_every and (i % log_every == 0) and b == 0:
                print(f"  [sketch] block {b+1}/{len(col_blocks)} ex {i}/{N} "
                      f"({time.time()-t0:.0f}s)", flush=True)
        Y_host[:, c0:c1] = np.asarray(yb) / N
        del omega_b, yb
        print(f"  [sketch] column-block {b+1}/{len(col_blocks)} done ({time.time()-t0:.0f}s)",
              flush=True)

    # ---- Nystrom recovery via eigh-pinv square root (robust to singular B0) ----
    # A ~= Y B0^+ Y^T. With B0 = Q diag(d) Q^T, R = Q_r diag(d_r^{-1/2}) (drop tiny d),
    # W = Y R, SVD(W) => eigvecs U = Y (R V sigma^-1), eigvals lambda = sigma^2.
    B0 = np.einsum("ncs,nct->st", M, M).astype(np.float64) / N   # Omega^T Y  (T x T)
    B0 = 0.5 * (B0 + B0.T)
    YtY = _rowchunked_gram(Y_host)                               # (T x T), row-chunked, no blowup
    YtY = 0.5 * (YtY + YtY.T)

    d, Q = np.linalg.eigh(B0)                                     # ascending
    d = d[::-1]; Q = Q[:, ::-1]
    d = np.clip(d, 0.0, None)
    tol = max(d[0], 1.0) * (T * np.finfo(np.float64).eps) + jitter * max(d[0], 1.0)
    r = int(np.sum(d > tol))
    r = max(r, 1)
    R = Q[:, :r] / np.sqrt(d[:r])[None, :]                        # (T x r), B0^+ = R R^T
    Gr = R.T @ YtY @ R                                            # (r x r) = W^T W
    Gr = 0.5 * (Gr + Gr.T)
    sig2, V = np.linalg.eigh(Gr)
    order = np.argsort(sig2)[::-1]
    sig2 = np.clip(sig2[order], 0.0, None); V = V[:, order]
    kk = min(k, r)
    lam = sig2[:kk]                                               # eigenvalues of A
    sig = np.sqrt(np.clip(lam, 1e-300, None))
    Z = (R @ V[:, :kk]) / sig[None, :]                           # (T x k)
    U = _rowchunked_matmul(Y_host, Z)                            # (P x k), row-chunked, f32
    tr = float(np.trace(B0))

    # Orthonormality diagnostic (U should be ~orthonormal columns)
    UtU = U.T @ U
    orth_err = float(np.max(np.abs(UtU - np.eye(kk))))
    diagnostics = dict(
        P=P, N=N, T=T, k=kk, build_seconds=round(time.time() - t0, 1),
        eig_top=float(lam[0]) if kk else 0.0, eig_min_retained=float(lam[-1]) if kk else 0.0,
        stable_rank=float(np.sum(lam) / lam[0]) if kk and lam[0] > 0 else 0.0,
        basis_orthonormality_max_err=orth_err, B0_trace=tr, sketch_numerical_rank=r,
        num_column_blocks=len(col_blocks), block_cols=block_cols,
    )
    return SketchResult(eigvals=lam.astype(np.float64), basis=U, T=T, k=kk, n_examples=N,
                        sketch_seed=sketch_seed, diagnostics=diagnostics)


def save_sketch(res: SketchResult, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, eigvals=res.eigvals, basis=res.basis, T=res.T, k=res.k,
             n_examples=res.n_examples, sketch_seed=res.sketch_seed,
             diagnostics=np.array(str(res.diagnostics)))


def load_sketch(path: Path):
    d = np.load(path, allow_pickle=True)
    return dict(eigvals=d["eigvals"], basis=d["basis"], T=int(d["T"]), k=int(d["k"]),
                n_examples=int(d["n_examples"]), sketch_seed=int(d["sketch_seed"]))
