"""Memory-bounded categorical SCOD at restricted parameter scopes (spec Part E).

Mathematical reference: StanfordASL/SCOD @ 6a569734d3e246e25c53c0dff97e4e83690087d4.
The sketch (`RandomSymSketch`), the sketch operator and the `posterior_pred` projection are
vendored from `nn_ood/sketching.py` essentially verbatim; the SCOD uncertainty definition is
unchanged. Four adaptations are made, all permitted by the spec:

1. **Categorical MC Fisher.** Upstream's exact `CategoricalLogit.apply_sqrt_F` produces a
   k-column factor per example, so the weight Jacobian has k rows. At k=1000 that is 1000
   backward passes per image. Instead we sample y ~ p(.|x) and use
   g = grad_w log p(y|x); since E_y[g g^T] = F(x), the q columns g_j/sqrt(q) form an
   unbiased factor. The SCOD supplement explicitly permits this.
2. **Restricted parameter scopes.** Only the named parameter block enters the sketch.
3. **Analytic per-example gradients.** For the `linear` and `ffn` scopes every gradient is
   available in closed form from the cached CLS residual, so no autograd and no
   per-example backward loop is needed. Gate B below checks them against autograd.
4. **Blocked updates.** Columns are accumulated in blocks and passed to a single
   `low_rank_update`, which is algebraically identical to feeding them one at a time.

The Gaussian sketch operator (upstream's *default* `sketch_type='random'`) is used rather
than the CIFAR config's SRFT. SRFT's forward is an inverse DCT of length N per call, whose
cost is O(N log N) independent of how many columns are passed, so at N ~ 10^6 with ~10^4
blocks it dominates everything else. Gaussian sketching is pure GEMM. This trades memory
(a dense Omega and Psi, 2*T*N floats instead of T*N) for tractable runtime, and the memory
is measured and reported.
"""
from __future__ import annotations

import argparse
import gc
import json
import time

import numpy as np
import torch
import torch.nn as nn

from . import full_cache as fc
from . import fu_common as F
from .memprobe import GIB, cpu_peak_rss_gib, cpu_rss_gib, reset_cuda
from .vit_adapter import ViTPnCAdapter

K_EIGS = 30
T_SKETCH = 184                      # 6k+4, the paper's CIFAR SCOD-LL setting
K_FALLBACK = 12
T_FALLBACK = 76
RAM_CEILING_GIB = 18.0
SKETCH_SEED = 20260815
MEPS = 5000.0
SCOPES = ["linear", "ffn"]
N_CAL = 32768


# ============================================================ vendored sketch
class GaussianSketchOp(nn.Module):
    """Upstream nn_ood/sketching.py::GaussianSketchOp."""

    def __init__(self, d, N, device=torch.device("cpu"), gen=None):
        super().__init__()
        self.d, self.N = d, N
        self.test_matrix = torch.randn(d, N, dtype=torch.float, device=device,
                                       generator=gen)

    @torch.no_grad()
    def forward(self, M, transpose=False):
        if transpose:
            return M @ self.test_matrix.t()
        return self.test_matrix @ M


class RandomSymSketch:
    """Upstream nn_ood/sketching.py::RandomSymSketch + LinearSketch.

    Sketches A = sum_i (1/M) v_i v_i^T from columns presented sequentially, then recovers a
    rank-2k eigenbasis. Verbatim algebra; only the update is allowed to take a block of
    columns at once, which is identical to presenting them one at a time.
    """

    def __init__(self, N, M, r, T=None, device="cpu", seed=SKETCH_SEED):
        self.N, self.M, self.r = N, M, r
        self.T = T if T is not None else 6 * r + 4
        self.device = torch.device(device)
        self.k = max(self.r + 2, (self.T - 1) // 3)
        self.l = self.T - self.k
        gen = torch.Generator(device=self.device).manual_seed(seed)
        self.Om = GaussianSketchOp(self.k, self.N, device=self.device, gen=gen)
        self.Psi = GaussianSketchOp(self.l, self.N, device=self.device, gen=gen)
        self.Y = torch.zeros(self.N, self.k, dtype=torch.float, device=self.device)
        self.W = torch.zeros(self.l, self.N, dtype=torch.float, device=self.device)

    def Om_fn(self, M):
        return self.Om(M, transpose=True)

    def Psi_fn(self, M):
        return self.Psi(M)

    @torch.no_grad()
    def low_rank_update(self, v, weight=1.0):
        """v is (N, d): a block of columns of A^(1/2)."""
        v = v.to(self.device)
        torch.addmm(self.Y, weight * v, self.Om_fn(v.t()), alpha=1.0 / self.M,
                    out=self.Y)
        torch.addmm(self.W, weight * self.Psi_fn(v), v.t(), alpha=1.0 / self.M,
                    out=self.W)

    @torch.no_grad()
    def low_rank_approx(self):
        Q, _ = torch.linalg.qr(self.Y, mode="reduced")
        U, T = torch.linalg.qr(self.Psi_fn(Q), mode="reduced")
        X = torch.linalg.solve_triangular(T, U.t() @ self.W, upper=True)
        return Q, X

    @torch.no_grad()
    def sym_low_rank_approx(self):
        Q, X = self.low_rank_approx()
        # W and Psi are dead once X exists; freeing them here keeps the peak below the
        # RAM ceiling during the (N, 2k) QR that follows. Algebra is unchanged.
        del self.W, self.Psi
        gc.collect()
        U, T = torch.linalg.qr(torch.cat([Q, X.t()], dim=1), mode="reduced")
        del Q, X
        T1, T2 = T[:, :self.k], T[:, self.k:2 * self.k]
        return U, (T1 @ T2.t() + T2 @ T1.t()) / 2

    @torch.no_grad()
    def get_range_basis(self):
        # Omega is not needed once accumulation is done; free it before the QR workspace
        del self.Om
        gc.collect()
        U, S = self.sym_low_rank_approx()
        D, V = torch.linalg.eigh(S)
        r = 2 * self.k
        return D[-r:], U @ V[:, -r:]


class Projector:
    """Upstream nn_ood/sketching.py::Projector, `posterior_pred` metric."""

    def __init__(self, eigs, basis):
        self.eigs, self.basis = eigs, basis

    @torch.no_grad()
    def posterior_pred(self, L, n_eigs, Meps=MEPS):
        basis = self.basis[:, -n_eigs:]
        eigs = torch.clamp(self.eigs[-n_eigs:], min=0.0)
        scaling = torch.sqrt(eigs / (eigs + 1.0 / (2 * Meps)))
        proj = scaling[:, None] * (basis.t() @ L)
        return torch.sqrt(torch.clamp(torch.sum(L ** 2) - torch.sum(proj ** 2), min=0.0))


# ==================================================== analytic per-example grads
def gelu_grad(a: torch.Tensor) -> torch.Tensor:
    """d/da [a * Phi(a)] for the exact erf GELU used by torchvision's ViT."""
    Phi = 0.5 * (1.0 + torch.erf(a / np.sqrt(2.0)))
    phi = torch.exp(-0.5 * a ** 2) / np.sqrt(2.0 * np.pi)
    return Phi + a * phi


def ln_vjp(ln: nn.LayerNorm, b: torch.Tensor, dphi: torch.Tensor) -> torch.Tensor:
    """J_LN^T dphi for an affine LayerNorm over the last dim."""
    D = b.shape[-1]
    mu = b.mean(-1, keepdim=True)
    var = b.var(-1, unbiased=False, keepdim=True)
    sigma = torch.sqrt(var + ln.eps)
    chat = (b - mu) / sigma
    u = dphi * ln.weight / sigma
    return u - u.mean(-1, keepdim=True) - chat * (u * chat).mean(-1, keepdim=True)


class ScopeGrad:
    """Forward quantities and analytic grad_w log p(y|x) for a restricted scope."""

    SIZES = {
        "linear": [("head_W", (1000, 768)), ("head_b", (1000,))],
        "ffn": [("W1", (768, 3072)), ("b1", (3072,)), ("W2", (3072, 768)),
                ("b2", (768,)), ("head_W", (1000, 768)), ("head_b", (1000,))],
    }

    def __init__(self, ad, scope: str):
        assert scope in self.SIZES, f"unsupported scope {scope}"
        self.ad, self.scope = ad, scope
        self.blocks = self.SIZES[scope]
        self.n_params = sum(int(np.prod(s)) for _, s in self.blocks)
        self.head = ad.model.heads[-1] if hasattr(ad.model.heads, "__getitem__") \
            else ad.model.heads

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> dict:
        ad = self.ad
        h = ad.block.ln_2(x[:, None, :])[:, 0]
        a = h @ ad.W1 + ad.b1
        y = torch.nn.functional.gelu(a)
        z = y @ ad.W2 + ad.b2
        b = x + z
        phi = ad.enc.ln(b[:, None, :])[:, 0]
        logits = self.head(phi)
        return {"x": x, "h": h, "a": a, "y": y, "b": b, "phi": phi, "logits": logits,
                "logp": torch.log_softmax(logits.double(), -1)}

    @torch.no_grad()
    def grad_columns(self, fw: dict, ys: torch.Tensor, out: torch.Tensor):
        """Write grad_w log p(y_j | x_i) into `out` (n_params, B*q), column-major by (i, j).

        Column ordering is (example, sample): column i*q + j.
        """
        ad = self.ad
        B, q = ys.shape
        p = torch.softmax(fw["logits"], -1)                       # (B, 1000)
        for j in range(q):
            s = -p.clone()
            s[torch.arange(B, device=s.device), ys[:, j]] += 1.0   # e_y - p
            cols = torch.arange(j, B * q, q, device=s.device)
            off = 0
            # head
            gW = s[:, :, None] * fw["phi"][:, None, :]             # (B, 1000, 768)
            if self.scope == "linear":
                out[off:off + 768000, cols] = gW.reshape(B, -1).T
                off += 768000
                out[off:off + 1000, cols] = s.T
                off += 1000
                del gW, s
                continue
            dphi = s @ self.head.weight                            # (B, 768)
            db = ln_vjp(ad.enc.ln, fw["b"], dphi)                  # (B, 768)
            g_b2 = db
            g_W2 = fw["y"][:, :, None] * db[:, None, :]            # (B, 3072, 768)
            dy = db @ ad.W2.T                                      # (B, 3072)
            da = dy * gelu_grad(fw["a"])
            g_W1 = fw["h"][:, :, None] * da[:, None, :]            # (B, 768, 3072)
            for name, g in (("W1", g_W1), ("b1", da), ("W2", g_W2), ("b2", g_b2),
                            ("head_W", gW), ("head_b", s)):
                n = int(np.prod(dict(self.blocks)[name]))
                out[off:off + n, cols] = g.reshape(B, -1).T
                off += n
            del gW, s, dphi, db, g_b2, g_W2, dy, da, g_W1
        assert off == self.n_params, (off, self.n_params)


def sample_labels(logp: torch.Tensor, q: int, gen: torch.Generator) -> torch.Tensor:
    """y_j ~ Categorical(p(.|x)) — the MC Fisher draw."""
    return torch.multinomial(logp.exp().float(), q, replacement=True, generator=gen)


class LastBlockScope:
    """SCOD-last-block: all of encoder_layer_11, the final encoder LayerNorm and the head.

    Self-attention mixes tokens, so the CLS-only residual cache is not sufficient here and
    the analytic route used for `linear`/`ffn` does not apply. Gradients come from ordinary
    autograd on the block, driven by the full 197-token input to block 11. Because SCOD
    needs *per-example* Jacobians, the block is run one example at a time; the expensive
    backbone prefix that produces the tokens is still batched.
    """

    def __init__(self, ad):
        self.ad = ad
        self.head = ad.model.heads[-1] if hasattr(ad.model.heads, "__getitem__") \
            else ad.model.heads
        b = ad.block
        self.params = [b.ln_1.weight, b.ln_1.bias,
                       b.self_attention.in_proj_weight, b.self_attention.in_proj_bias,
                       b.self_attention.out_proj.weight, b.self_attention.out_proj.bias,
                       b.ln_2.weight, b.ln_2.bias,
                       b.mlp[0].weight, b.mlp[0].bias, b.mlp[3].weight, b.mlp[3].bias,
                       ad.enc.ln.weight, ad.enc.ln.bias,
                       self.head.weight, self.head.bias]
        self.n_params = int(sum(p.numel() for p in self.params))
        self._prev = [p.requires_grad for p in self.params]
        for p in self.params:
            p.requires_grad_(True)

    def release(self):
        for p, w in zip(self.params, self._prev):
            p.requires_grad_(w)

    @torch.no_grad()
    def prefix_tokens(self, images: torch.Tensor) -> torch.Tensor:
        """All 197 token states entering block 11, batched."""
        ad, m = self.ad, self.ad.model
        x = m._process_input(images)
        x = torch.cat([m.class_token.expand(x.shape[0], -1, -1), x], dim=1)
        x = ad.enc.dropout(x + ad.enc.pos_embedding)
        for i in range(ad.block_index):
            x = ad.enc.layers[i](x)
        return x

    def logp(self, tok1: torch.Tensor) -> torch.Tensor:
        out = self.ad.block(tok1)
        phi = self.ad.enc.ln(out)[:, 0]
        return torch.log_softmax(self.head(phi), -1)[0]

    def grad_columns(self, tok: torch.Tensor, ys: torch.Tensor, out: torch.Tensor):
        """Per-example autograd columns; `out` is (n_params, B*q), column i*q + j."""
        B, q = ys.shape
        for i in range(B):
            lp = self.logp(tok[i:i + 1])
            for j in range(q):
                gs = torch.autograd.grad(lp[ys[i, j]], self.params,
                                         retain_graph=(j < q - 1))
                # out[:, c] is a strided view, so cat(out=) is not usable here
                out[:, i * q + j] = torch.cat([g.reshape(-1) for g in gs])
                del gs
            del lp


# ============================================================ fit / score
def fit_scod(ad, scope: str, phi_src: torch.Tensor, n_cal: int, k: int, T: int, q: int,
             block: int = 16, seed: int = SKETCH_SEED, log_every: int = 4096):
    sg = ScopeGrad(ad, scope)
    N = sg.n_params
    sk = RandomSymSketch(N, n_cal, r=k, T=T, device="cpu", seed=seed)
    print(f"  scope={scope} n_params={N:,}  k={k} T={T} (k_sk={sk.k}, l={sk.l})  q={q}")
    print(f"  persistent sketch {(sk.Y.numel()+sk.W.numel())*4/GIB:.2f} GiB + "
          f"operators {(sk.Om.test_matrix.numel()+sk.Psi.test_matrix.numel())*4/GIB:.2f}"
          f" GiB", flush=True)
    gen = torch.Generator(device=ad.device).manual_seed(seed + 7)
    buf = torch.zeros(N, block * q, dtype=torch.float32, device=ad.device)
    t0, peak = time.perf_counter(), 0.0
    for s in range(0, n_cal, block):
        x = phi_src[s:s + block]
        B = x.shape[0]
        v = buf[:, :B * q]
        v.zero_()
        fw = sg.forward(x)
        ys = sample_labels(fw["logp"], q, gen)
        sg.grad_columns(fw, ys, v)
        v.div_(np.sqrt(q))                       # in place: v is a 1-4 GiB view
        sk.low_rank_update(v)
        peak = max(peak, cpu_rss_gib())
        if peak > RAM_CEILING_GIB:
            raise MemoryError(f"RSS {peak:.1f} GiB exceeded ceiling {RAM_CEILING_GIB}")
        if log_every and (s // block) % (log_every // block) == 0 and s:
            el = time.perf_counter() - t0
            print(f"    {s:>6,}/{n_cal:,}  {s/el:.0f} ex/s  RSS {peak:.1f} GiB  "
                  f"eta {(n_cal-s)/(s/el)/60:.1f} min", flush=True)
        del fw, x
    del buf
    reset_cuda()
    gc.collect()
    fit_s = time.perf_counter() - t0
    print(f"  accumulation done in {fit_s/60:.1f} min; recovering range basis...",
          flush=True)
    t1 = time.perf_counter()
    k_sketch, l_sketch = sk.k, sk.l
    eigs, basis = sk.get_range_basis()
    basis_s = time.perf_counter() - t1
    peak = max(peak, cpu_peak_rss_gib())
    del sk
    gc.collect()
    print(f"  basis in {basis_s:.0f}s; top eigs "
          f"{[f'{v:.4g}' for v in eigs[-5:].tolist()]}  peak RSS {peak:.1f} GiB")
    return {"eigs": eigs, "basis": basis, "n_params": N, "k": k, "T": T, "q": q,
            "n_cal": n_cal, "fit_seconds": fit_s, "basis_seconds": basis_s,
            "peak_rss_gib": peak, "k_sketch": k_sketch, "l_sketch": l_sketch,
            "recovered_rank": int(eigs.numel()),
            "storage_mib": (eigs.numel() + basis.numel()) * 4 / 1024 ** 2}


def fit_scod_lastblock(ad, img_iter, n_cal: int, k: int, T: int, q: int,
                       seed: int = SKETCH_SEED, log_every: int = 2048):
    """SCOD-last-block fit. `img_iter()` yields (images, n) batches in any order."""
    sg = LastBlockScope(ad)
    N = sg.n_params
    sk = RandomSymSketch(N, n_cal, r=k, T=T, device="cpu", seed=seed)
    print(f"  scope=last_block n_params={N:,}  k={k} T={T} (k_sk={sk.k}, l={sk.l})  q={q}")
    print(f"  persistent sketch {(sk.Y.numel()+sk.W.numel())*4/GIB:.2f} GiB + "
          f"operators {(sk.Om.test_matrix.numel()+sk.Psi.test_matrix.numel())*4/GIB:.2f}"
          f" GiB", flush=True)
    gen = torch.Generator(device=ad.device).manual_seed(seed + 7)
    t0, peak, done = time.perf_counter(), 0.0, 0
    for images in img_iter():
        B = images.shape[0]
        tok = sg.prefix_tokens(images.to(ad.device, ad.dtype))
        with torch.no_grad():
            lp = torch.stack([sg.logp(tok[i:i + 1]) for i in range(B)])
        ys = sample_labels(lp, q, gen)
        v = torch.zeros(N, B * q, dtype=torch.float32, device=ad.device)
        sg.grad_columns(tok, ys, v)
        v.div_(np.sqrt(q))
        sk.low_rank_update(v)
        done += B
        peak = max(peak, cpu_rss_gib())
        if peak > RAM_CEILING_GIB:
            sg.release()
            raise MemoryError(f"RSS {peak:.1f} GiB exceeded ceiling {RAM_CEILING_GIB}")
        if log_every and done % log_every < B:
            el = time.perf_counter() - t0
            print(f"    {done:>6,}/{n_cal:,}  {done/el:.1f} ex/s  RSS {peak:.1f} GiB  "
                  f"eta {(n_cal-done)/(done/el)/60:.1f} min", flush=True)
        del tok, v, lp
        reset_cuda()
    sg.release()
    fit_s = time.perf_counter() - t0
    print(f"  accumulation done in {fit_s/60:.1f} min; recovering range basis...",
          flush=True)
    t1 = time.perf_counter()
    k_sketch, l_sketch = sk.k, sk.l
    eigs, basis = sk.get_range_basis()
    peak = max(peak, cpu_peak_rss_gib())
    del sk
    gc.collect()
    return {"eigs": eigs, "basis": basis, "n_params": N, "k": k, "T": T, "q": q,
            "n_cal": done, "fit_seconds": fit_s,
            "basis_seconds": time.perf_counter() - t1, "peak_rss_gib": peak,
            "k_sketch": k_sketch, "l_sketch": l_sketch,
            "recovered_rank": int(eigs.numel()),
            "storage_mib": (eigs.numel() + basis.numel()) * 4 / 1024 ** 2}


def score_scod_lastblock(ad, proj: Projector, img_iter, n: int, q: int, n_eigs: int,
                         seed: int = SKETCH_SEED + 99) -> np.ndarray:
    sg = LastBlockScope(ad)
    N = sg.n_params
    gen = torch.Generator(device=ad.device).manual_seed(seed)
    basis = proj.basis[:, -n_eigs:].to(ad.device)
    eigs = torch.clamp(proj.eigs[-n_eigs:], min=0.0).to(ad.device)
    scaling = torch.sqrt(eigs / (eigs + 1.0 / (2 * MEPS)))
    out, at = np.empty(n, dtype=np.float64), 0
    for images in img_iter():
        B = images.shape[0]
        tok = sg.prefix_tokens(images.to(ad.device, ad.dtype))
        with torch.no_grad():
            lp = torch.stack([sg.logp(tok[i:i + 1]) for i in range(B)])
        ys = sample_labels(lp, q, gen)
        v = torch.zeros(N, B * q, dtype=torch.float32, device=ad.device)
        sg.grad_columns(tok, ys, v)
        v.div_(np.sqrt(q))
        with torch.no_grad():
            pr = scaling[:, None] * (basis.t() @ v)
            tot = (v ** 2).reshape(N, B, q).sum(0).sum(-1)
            prj = (pr ** 2).reshape(-1, B, q).sum(0).sum(-1)
            out[at:at + B] = torch.sqrt(torch.clamp(tot - prj, min=0.0)) \
                .double().cpu().numpy()
        at += B
        del tok, v, pr, lp
        reset_cuda()
    sg.release()
    del basis
    reset_cuda()
    return out[:at]


@torch.no_grad()
def score_scod(ad, scope: str, proj: Projector, X: torch.Tensor, q: int, n_eigs: int,
               block: int = 16, seed: int = SKETCH_SEED + 99) -> np.ndarray:
    sg = ScopeGrad(ad, scope)
    N = sg.n_params
    gen = torch.Generator(device=ad.device).manual_seed(seed)
    buf = torch.zeros(N, block * q, dtype=torch.float32, device=ad.device)
    basis = proj.basis[:, -n_eigs:].to(ad.device)
    eigs = torch.clamp(proj.eigs[-n_eigs:], min=0.0).to(ad.device)
    scaling = torch.sqrt(eigs / (eigs + 1.0 / (2 * MEPS)))
    out = np.empty(X.shape[0], dtype=np.float64)
    for s in range(0, X.shape[0], block):
        x = X[s:s + block]
        B = x.shape[0]
        v = buf[:, :B * q]
        v.zero_()
        fw = sg.forward(x)
        ys = sample_labels(fw["logp"], q, gen)
        sg.grad_columns(fw, ys, v)
        v /= np.sqrt(q)
        pr = scaling[:, None] * (basis.t() @ v)                   # (n_eigs, B*q)
        tot = (v ** 2).reshape(N, B, q).sum(0).sum(-1)
        prj = (pr ** 2).reshape(-1, B, q).sum(0).sum(-1)
        out[s:s + B] = torch.sqrt(torch.clamp(tot - prj, min=0.0)).double().cpu().numpy()
        del fw, v, pr
    del buf, basis
    reset_cuda()
    return out
