"""P&C construction pieces for the ViT preflight.

Reuse policy (see REUSE_MAP.md): the ridge solve is the existing
``pnc_theory.linalg.ridge_solve``. Because a 197-token ImageNet design matrix must
not be materialised, :func:`ridge_solve_from_stats` performs the *same* solve from
streamed sufficient statistics ``G = X^T W X`` and ``C = X^T W Y``. That is a
reformulation, not a second solver: ``ridge_solve`` itself builds ``Xv.T @ Xv`` and
``Xv.T @ target`` internally, and ``validate.py`` gates the two to bitwise equality.

The basis / coefficient / scale conventions are the Banking77 ones
(``experiments/banking77_pnc/construct.py``), re-expressed in pure numpy so this
module imports in a torch-only environment; ``validate.py`` gates them against the
Banking77 originals element-for-element.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from pnc_theory.linalg import ridge_solve  # noqa: E402,F401  (REUSED solver; parity gate)

DIN, DHID = 768, 3072
D_FLAT = DIN * DHID


# --------------------------------------------------------------------------
# Perturbation conventions (Banking77 construct.py, verbatim semantics)
# --------------------------------------------------------------------------
def perturbation_basis(seed: int, K: int = 20) -> np.ndarray:
    """Shared random low-rank basis on the flattened W1, unit-normalized rows."""
    rng = np.random.RandomState(seed)
    U = rng.normal(size=(K, D_FLAT)).astype(np.float32)
    U /= np.linalg.norm(U, axis=1, keepdims=True) + 1e-12
    return U


def member_coefficients(seed: int, M: int, K: int) -> np.ndarray:
    return np.random.RandomState(seed + 1).normal(size=(M, K)).astype(np.float32)


def base_scale(U, coeffs, W1_norm, target_rel: float = 0.5) -> float:
    """Numerical scale so the median ||dW1||/||W1|| equals target_rel at multiplier 1.0."""
    rels = [np.linalg.norm(coeffs[m] @ U) / W1_norm for m in range(len(coeffs))]
    return float(target_rel / (np.median(rels) + 1e-12))


def member_dW1(U: np.ndarray, coeff: np.ndarray, scale: float) -> np.ndarray:
    """Materialise one member's (768, 3072) perturbation on demand -- never stored."""
    return (scale * (coeff @ U)).reshape(DIN, DHID)


# --------------------------------------------------------------------------
# Streaming sufficient statistics (spec section 10)
# --------------------------------------------------------------------------
class SufficientStats:
    """Accumulate G = X^T W X and C = X^T W Y in float32 on GPU; solve in float64 on CPU.

    X rows are the bias-augmented post-GELU activations ``[1, y]`` (bias FIRST, the
    Banking77 layout), so p = 3073 and the solution is ``Theta = [b2; W2]``.
    """

    def __init__(self, p: int = DHID + 1, d: int = DIN, device="cuda",
                 dtype=torch.float32):
        self.p, self.d = p, d
        self.G = torch.zeros(p, p, device=device, dtype=dtype)
        self.C = torch.zeros(p, d, device=device, dtype=dtype)
        self.yty = 0.0          # sum ||Y_i||^2, lets residuals be computed without the design
        self.n_rows = 0

    @staticmethod
    def augment(y: torch.Tensor) -> torch.Tensor:
        """(n, 3072) post-GELU -> (n, 3073) with a leading ones column."""
        return torch.cat([torch.ones(y.shape[0], 1, device=y.device, dtype=y.dtype), y], 1)

    @torch.inference_mode()
    def update(self, X: torch.Tensor, Y: torch.Tensor, w: torch.Tensor | None = None):
        """X (n, p) bias-augmented, Y (n, d) targets, optional non-negative row weights."""
        if w is None:
            self.G += X.T @ X
            self.C += X.T @ Y
            self.yty += float((Y.double() ** 2).sum())
            self.n_rows += X.shape[0]
        else:
            Xw = X * w[:, None]
            self.G += Xw.T @ X
            self.C += Xw.T @ Y
            self.yty += float((w.double()[:, None] * Y.double() ** 2).sum())
            self.n_rows += int(w.sum().item())
        return self

    def to_cpu_f64(self):
        return (self.G.double().cpu().numpy(), self.C.double().cpu().numpy())

    def snapshot(self) -> dict:
        """Detached float64 CPU copy of (G, C, yty, n_rows) for checkpointed sweeps."""
        G, C = self.to_cpu_f64()
        return {"G": G, "C": C, "yty": self.yty, "n_rows": self.n_rows}

    def nbytes_gpu(self) -> int:
        return self.G.numel() * self.G.element_size() + self.C.numel() * self.C.element_size()


def relative_residual(stats: dict, Theta) -> float:
    """||X Theta - Y||_F / ||Y||_F computed from sufficient statistics alone.

    ||X Th - Y||^2 = tr(Th^T G Th) - 2 tr(Th^T C) + ||Y||^2, so no design matrix is
    ever materialised -- the same trick that makes all-token accumulation affordable.
    """
    Th = np.asarray(Theta, dtype=np.float64)
    G, C, yty = stats["G"], stats["C"], stats["yty"]
    sq = float(np.trace(Th.T @ G @ Th) - 2.0 * np.trace(Th.T @ C) + yty)
    return float(np.sqrt(max(sq, 0.0)) / (np.sqrt(yty) + 1e-30))


def ridge_solve_from_stats(G, C, lam: float, w_prior=None) -> np.ndarray:
    """Streaming-statistics form of ``pnc_theory.linalg.ridge_solve`` (sum_legacy mode).

    ``ridge_solve`` computes ``(X^T X + lam I)^-1 (X^T target + lam w_prior)``; given
    ``G = X^T X`` and ``C = X^T target`` this is the identical linear system, solved in
    float64. Bitwise parity with the original is gated in ``validate.py``.
    """
    G = np.asarray(G, dtype=np.float64)
    C = np.asarray(C, dtype=np.float64)
    p = G.shape[0]
    A = G + lam * np.eye(p)
    rhs = C if w_prior is None else C + lam * np.asarray(w_prior, dtype=np.float64)
    return np.linalg.solve(A, rhs)


def cho_solve_shared(G, C, lam: float, w_prior=None):
    """Same solve via one Cholesky factorisation shared across all 768 outputs.

    Returns ``(Theta, factor_time_s, solve_time_s)``. Spec section 13 requires that the
    768 output columns share a single factorisation rather than 768 independent solves.
    """
    import time

    from scipy.linalg import cho_factor, cho_solve
    G = np.asarray(G, dtype=np.float64)
    C = np.asarray(C, dtype=np.float64)
    rhs = C if w_prior is None else C + lam * np.asarray(w_prior, dtype=np.float64)
    A = G + lam * np.eye(G.shape[0])
    t0 = time.perf_counter()
    cf = cho_factor(A, lower=True, check_finite=False)
    t1 = time.perf_counter()
    Theta = cho_solve(cf, rhs, check_finite=False)     # one factor, all 768 columns
    t2 = time.perf_counter()
    return Theta, t1 - t0, t2 - t1


# --------------------------------------------------------------------------
# Deterministic token sampling (spec section 11)
# --------------------------------------------------------------------------
TOKEN_MODES = {"cls": 0, "cls+4": 4, "cls+16": 16, "all": None}


def token_index_matrix(mode: str, image_ids, n_tokens: int, seed: int) -> np.ndarray:
    """Per-image token positions to keep. Deterministic in (seed, global image id).

    Position 0 is the CLS token; 1..n_tokens-1 are patch tokens. ``all`` returns every
    position. Returns an (n_images, rows_per_image) int array.
    """
    if mode not in TOKEN_MODES:
        raise ValueError(f"unknown token mode {mode!r}; expected one of {sorted(TOKEN_MODES)}")
    if mode == "all":
        return np.tile(np.arange(n_tokens), (len(image_ids), 1))
    n_patch = TOKEN_MODES[mode]
    out = np.empty((len(image_ids), 1 + n_patch), dtype=np.int64)
    for i, img_id in enumerate(image_ids):
        out[i, 0] = 0
        if n_patch:
            rng = np.random.RandomState((int(seed) * 100003 + int(img_id)) % (2 ** 31 - 1))
            out[i, 1:] = rng.choice(np.arange(1, n_tokens), size=n_patch, replace=False)
    return out


def gather_tokens(t: torch.Tensor, idx: np.ndarray) -> torch.Tensor:
    """t (B, T, D) + idx (B, R) -> (B*R, D), rows ordered image-major."""
    gi = torch.as_tensor(idx, device=t.device, dtype=torch.long)
    picked = torch.gather(t, 1, gi[:, :, None].expand(-1, -1, t.shape[-1]))
    return picked.reshape(-1, t.shape[-1])


# --------------------------------------------------------------------------
# Compact member representation (spec section 8/16)
# --------------------------------------------------------------------------
def member_nbytes(K: int = 20, dtype_bytes: int = 4) -> dict:
    """Per-member and shared storage of the compact P&C state, in bytes."""
    return {
        "shared_basis": K * D_FLAT * dtype_bytes,
        "coeff_per_member": K * dtype_bytes,
        "W2_per_member": DHID * DIN * dtype_bytes,
        "b2_per_member": DIN * dtype_bytes,
    }


def condition_estimate(G, lam: float) -> dict:
    """2-norm condition number of the regularised Gram matrix, plus its spectrum ends."""
    G = np.asarray(G, dtype=np.float64)
    ev = np.linalg.eigvalsh(0.5 * (G + G.T))
    lo, hi = float(ev[0]), float(ev[-1])
    return {"gram_eig_min": lo, "gram_eig_max": hi,
            "cond_gram": hi / lo if lo > 0 else float("inf"),
            "cond_regularised": (hi + lam) / (lo + lam)}
