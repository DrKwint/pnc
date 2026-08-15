"""Exact categorical SCOD — no Monte-Carlo Fisher.

The MC route failed its own §14 convergence check on this head (genuine across-seed Spearman
~0.46 and score CV ~0.39 even at q=8; see `metrics/scod_qsweep.json`), because one sample is
a poor estimate of a 1000-class expectation. Rather than weaken SCOD to make it run, this
module uses upstream's *exact* factorisation

    F = diag(p) - p p^T = L_F L_F^T,     L_F = (I - p 1^T) diag(p^{1/2})

which is `CategoricalLogit.apply_sqrt_F(exact=True)` in `nn_ood/distributions.py`.

Upstream cannot use it at this scale: the exact factor needs C=1000 backward passes per
image and yields an (n_params x 1000) matrix L — 3 GB per example for the head alone. Both
costs vanish here because for these scopes the per-example gradient map is **linear in the
output residual**,

    g(s) = A_x s,        s = d log p(y|x) / d logits,

so L = A L_F and every sketch quantity factors through A without forming L:

    Om L  = (Om A) L_F                                   (T applications of A^T)
    Y    += (1/M) L (Om L)^T = (1/M) A [ L_F (Om L)^T ]  (k applications of A)
    W    += (1/M) (Psi L) L^T = (1/M) [ A ( L_F (Psi L)^T ) ]^T

Per example this is 2*T applications of A or A^T — the same order as the q=1 MC update, but
exact. Right/left multiplication by L_F is O(T*C):

    X L_F  = (X - (X p) 1^T) * sqrt(p)
    L_F Z  = sqrt(p) * Z - p * sum(sqrt(p) * Z)

Squared Frobenius norms use tr(A F A^T) = sum_c p_c ||A e_c||^2 - ||A p||^2, which for the
head collapses to the closed form (||phi||^2 + 1)(1 - ||p||^2) and for the FFN decomposes
block by block. Norms need no sketch, so they are computed on the GPU; only A/A^T touch the
CPU-resident Omega and Psi.
"""
from __future__ import annotations

import gc
import time

import numpy as np
import torch

from .memprobe import GIB, cpu_peak_rss_gib, cpu_rss_gib, reset_cuda
from .scod_ll import MEPS, Projector, RandomSymSketch, gelu_grad

C_OUT = 1000


def _ln_parts(b: torch.Tensor, eps: float):
    mu = b.mean(-1, keepdim=True)
    sigma = torch.sqrt(b.var(-1, unbiased=False, keepdim=True) + eps)
    return sigma, (b - mu) / sigma


def _S(d: torch.Tensor, chat: torch.Tensor):
    """(I - 11^T/D - chat chat^T/D) d — the symmetric core of the LayerNorm Jacobian."""
    return d - d.mean(-1, keepdim=True) - chat * (d * chat).mean(-1, keepdim=True)


def ln_vjp_fn(weight, eps, b, d):
    """J_LN^T d, with J_LN = diag(gamma/sigma) S, so J_LN^T = S diag(gamma/sigma)."""
    sigma, chat = _ln_parts(b, eps)
    return _S(d * (weight / sigma), chat)


def ln_jvp_fn(weight, eps, b, d):
    """J_LN d. NOT the same as ln_vjp_fn: the diagonal sits on the other side."""
    sigma, chat = _ln_parts(b, eps)
    return (weight / sigma) * _S(d, chat)


def right_LF(X: torch.Tensor, p: torch.Tensor, sp: torch.Tensor) -> torch.Tensor:
    """X L_F, X of shape (..., C); p, sp broadcast on the last axis."""
    return (X - (X * p).sum(-1, keepdim=True)) * sp


def left_LF(Z: torch.Tensor, p: torch.Tensor, sp: torch.Tensor) -> torch.Tensor:
    """L_F Z, Z of shape (C, ...); p, sp shaped (C, 1...)."""
    u = sp * Z
    return u - p * u.sum(0, keepdim=True)


# ----------------------------------------------------------------- scopes
class ExactLinear:
    """`heads.head`. A s = [s phi^T ; s]."""

    name = "linear"
    param_blocks = [("head_W", C_OUT * 768), ("head_b", C_OUT)]

    def __init__(self, ad):
        self.ad = ad
        self.head = ad.model.heads[-1] if hasattr(ad.model.heads, "__getitem__") \
            else ad.model.heads
        self.n_params = sum(n for _, n in self.param_blocks)
        self.off, o = {}, 0
        for k, n in self.param_blocks:
            self.off[k] = (o, o + n)
            o += n
        self.ln_w = ad.enc.ln.weight.detach()
        self.ln_eps = ad.enc.ln.eps

    @torch.no_grad()
    def _forward(self, x):
        ad = self.ad
        h = ad.block.ln_2(x[:, None, :])[:, 0]
        a = h @ ad.W1 + ad.b1
        y = torch.nn.functional.gelu(a)
        b = x + y @ ad.W2 + ad.b2
        phi = ad.enc.ln(b[:, None, :])[:, 0]
        return h, a, y, b, phi

    @torch.no_grad()
    def prepare(self, x: torch.Tensor) -> dict:
        _, _, _, _, phi = self._forward(x)
        p = torch.softmax(self.head(phi), -1)
        return {"phi_g": phi, "phi": phi.cpu(), "p_g": p,
                "p": p.double().cpu(), "sp": p.double().sqrt().cpu(),
                "B": x.shape[0]}

    def split(self, V: torch.Tensor) -> dict:
        """Per-block contiguous copies of the sketch rows, made ONCE.

        `V` is a row-slice of a (T, n_params) operator, so every block view is
        non-contiguous; reshaping it inside the per-batch loop would copy hundreds of MB
        (gigabytes for the FFN scope) on every batch.
        """
        R = V.shape[0]
        o0, o1 = self.off["head_W"]
        return {"R": R,
                "head_W": V[:, o0:o1].contiguous().reshape(R * C_OUT, 768),
                "head_b": V[:, self.off["head_b"][0]:].contiguous()}

    @torch.no_grad()
    def AT(self, ctx, VS: dict) -> torch.Tensor:
        """A^T applied to the split rows -> (R, C, B)."""
        out = (VS["head_W"] @ ctx["phi"].T).reshape(VS["R"], C_OUT, -1)
        return out + VS["head_b"][:, :, None]

    @torch.no_grad()
    def A_apply(self, ctx, Q: torch.Tensor) -> torch.Tensor:
        """sum_b A_b Q[:, :, b] for Q (C, R, B) -> (n_params, R)."""
        R = Q.shape[1]
        out = torch.empty(self.n_params, R, dtype=torch.float32)
        out[slice(*self.off["head_W"])] = torch.einsum(
            "crb,bd->cdr", Q, ctx["phi"]).reshape(-1, R)
        out[slice(*self.off["head_b"])] = Q.sum(-1)
        return out

    @torch.no_grad()
    def normsq(self, ctx) -> torch.Tensor:
        p = ctx["p_g"].double()
        return (((ctx["phi_g"].double() ** 2).sum(-1) + 1.0)
                * (1.0 - (p ** 2).sum(-1))).cpu()


class ExactFFN(ExactLinear):
    """W1, b1, W2, b2 and the head — still linear in s, so the same factorisation holds."""

    name = "ffn"
    param_blocks = [("W1", 768 * 3072), ("b1", 3072), ("W2", 3072 * 768),
                    ("b2", 768), ("head_W", C_OUT * 768), ("head_b", C_OUT)]

    @torch.no_grad()
    def prepare(self, x: torch.Tensor) -> dict:
        h, a, y, b, phi = self._forward(x)
        p = torch.softmax(self.head(phi), -1)
        gp = gelu_grad(a)
        return {"h": h.cpu(), "y": y.cpu(), "b": b.cpu(), "gp": gp.cpu(),
                "phi": phi.cpu(), "h_g": h, "y_g": y, "b_g": b, "gp_g": gp,
                "phi_g": phi, "p_g": p, "p": p.double().cpu(),
                "sp": p.double().sqrt().cpu(), "B": x.shape[0],
                "W2": self.ad.W2.cpu(), "Wh": self.head.weight.cpu()}

    def _db_da(self, ctx, S: torch.Tensor, gpu: bool):
        """db, da for output residuals S (R, C, B). Returns (R, B, 768), (R, B, 3072)."""
        Wh = self.head.weight if gpu else ctx["Wh"]
        W2 = self.ad.W2 if gpu else ctx["W2"]
        b = ctx["b_g"] if gpu else ctx["b"]
        gp = ctx["gp_g"] if gpu else ctx["gp"]
        w = self.ln_w if gpu else self.ln_w.cpu()
        dphi = torch.einsum("rcb,cd->rbd", S, Wh)
        db = ln_vjp_fn(w, self.ln_eps, b[None], dphi)
        return db, (db @ W2.T) * gp[None]

    def split(self, V: torch.Tensor) -> dict:
        """See ExactLinear.split — hoisted because these copies are ~1.7 GiB each."""
        R = V.shape[0]
        g = lambda k: V[:, slice(*self.off[k])].contiguous()
        return {"R": R, "W1": g("W1").reshape(R, 768, 3072), "b1": g("b1"),
                "W2": g("W2").reshape(R, 3072, 768), "b2": g("b2"),
                "head_W": g("head_W").reshape(R, C_OUT, 768), "head_b": g("head_b")}

    @torch.no_grad()
    def AT(self, ctx, VS: dict) -> torch.Tensor:
        u1 = torch.einsum("rij,bi->rbj", VS["W1"], ctx["h"]) + VS["b1"][:, None, :]
        u2 = torch.einsum("rij,bi->rbj", VS["W2"], ctx["y"]) + VS["b2"][:, None, :]
        w = u2 + (ctx["gp"][None] * u1) @ ctx["W2"]
        # the forward chain applies J_LN^T, so its adjoint applies J_LN
        t = ln_jvp_fn(self.ln_w.cpu(), self.ln_eps, ctx["b"][None], w)
        head = torch.einsum("rcd,bd->rcb", VS["head_W"], ctx["phi"]) \
            + VS["head_b"][:, :, None]
        return head + torch.einsum("cd,rbd->rcb", ctx["Wh"], t)

    @torch.no_grad()
    def A_apply(self, ctx, Q: torch.Tensor) -> torch.Tensor:
        R = Q.shape[1]
        db, da = self._db_da(ctx, Q.permute(1, 0, 2).contiguous(), gpu=False)
        out = torch.empty(self.n_params, R, dtype=torch.float32)
        out[slice(*self.off["W1"])] = torch.einsum(
            "bi,rbj->ijr", ctx["h"], da).reshape(-1, R)
        out[slice(*self.off["b1"])] = da.sum(1).T
        out[slice(*self.off["W2"])] = torch.einsum(
            "bi,rbj->ijr", ctx["y"], db).reshape(-1, R)
        out[slice(*self.off["b2"])] = db.sum(1).T
        out[slice(*self.off["head_W"])] = torch.einsum(
            "crb,bd->cdr", Q, ctx["phi"]).reshape(-1, R)
        out[slice(*self.off["head_b"])] = Q.sum(-1)
        return out

    @torch.no_grad()
    def normsq(self, ctx, chunk: int = 100) -> torch.Tensor:
        """tr(A F A^T) = sum_c p_c ||A e_c||^2 - ||A p||^2, on the GPU."""
        B = ctx["B"]
        hn = (ctx["h_g"].double() ** 2).sum(-1) + 1.0
        yn = (ctx["y_g"].double() ** 2).sum(-1) + 1.0
        pn = (ctx["phi_g"].double() ** 2).sum(-1) + 1.0
        p = ctx["p_g"].double()
        tot = pn.clone()                       # head block: sum_c p_c ||e_c||^2 (|phi|^2+1)
        eye = torch.eye(C_OUT, device=ctx["phi_g"].device, dtype=ctx["phi_g"].dtype)
        for s in range(0, C_OUT, chunk):
            E = eye[s:s + chunk][:, :, None].expand(-1, C_OUT, B)
            db, da = self._db_da(ctx, E, gpu=True)
            w = p[:, s:s + chunk].T
            tot += (w * ((db.double() ** 2).sum(-1) * yn[None]
                         + (da.double() ** 2).sum(-1) * hn[None])).sum(0)
            del E, db, da
        dbp, dap = self._db_da(ctx, p.to(ctx["phi_g"].dtype).T[None], gpu=True)
        ap2 = (pn * (p ** 2).sum(-1) + yn * (dbp[0].double() ** 2).sum(-1)
               + hn * (dap[0].double() ** 2).sum(-1))
        return (tot - ap2).cpu()


SCOPES = {"linear": ExactLinear, "ffn": ExactFFN}


# ----------------------------------------------------------------- fit / score
def fit(ad, scope: str, X: torch.Tensor, k: int, T: int, block: int = 128,
        seed: int = 20260815, log_every: int = 4096):
    sc = SCOPES[scope](ad)
    N, n_cal = sc.n_params, X.shape[0]
    sk = RandomSymSketch(N, n_cal, r=k, T=T, device="cpu", seed=seed)
    print(f"  scope={scope} n_params={N:,}  k={k} T={T} (k_sk={sk.k}, l={sk.l})  "
          f"EXACT categorical Fisher (no MC)")
    print(f"  persistent sketch {(sk.Y.numel()+sk.W.numel())*4/GIB:.2f} GiB + "
          f"operators {(sk.Om.test_matrix.numel()+sk.Psi.test_matrix.numel())*4/GIB:.2f}"
          f" GiB", flush=True)
    OmS, PsS = sc.split(sk.Om.test_matrix), sc.split(sk.Psi.test_matrix)
    kk, ll = sk.k, sk.l
    t0, peak = time.perf_counter(), 0.0
    for s in range(0, n_cal, block):
        ctx = sc.prepare(X[s:s + block])
        p = ctx["p"].T[:, None, :]                       # (C, 1, B)
        spT = ctx["sp"].T[:, None, :]
        pr, spr = ctx["p"][None], ctx["sp"][None]        # (1, B, C)
        OmL = right_LF(sc.AT(ctx, OmS).permute(0, 2, 1), pr, spr).permute(0, 2, 1)
        PsL = right_LF(sc.AT(ctx, PsS).permute(0, 2, 1), pr, spr).permute(0, 2, 1)
        sk.Y += sc.A_apply(ctx, left_LF(OmL.permute(1, 0, 2), p, spT).float()) / n_cal
        sk.W += sc.A_apply(ctx, left_LF(PsL.permute(1, 0, 2), p, spT).float()).T / n_cal
        peak = max(peak, cpu_rss_gib())
        if log_every and (s % log_every) < block and s:
            el = time.perf_counter() - t0
            print(f"    {s:>6,}/{n_cal:,}  {s/el:.0f} ex/s  RSS {peak:.1f} GiB  "
                  f"eta {(n_cal-s)/(s/el)/60:.1f} min", flush=True)
        del ctx, OmL, PsL
        reset_cuda()
    del OmS, PsS
    gc.collect()
    fit_s = time.perf_counter() - t0
    print(f"  accumulation done in {fit_s/60:.1f} min; recovering range basis...",
          flush=True)
    t1 = time.perf_counter()
    eigs, basis = sk.get_range_basis()
    peak = max(peak, cpu_peak_rss_gib())
    del sk
    gc.collect()
    print(f"  basis in {time.perf_counter()-t1:.0f}s; top eigs "
          f"{[f'{v:.4g}' for v in eigs[-5:].tolist()]}  peak RSS {peak:.1f} GiB")
    return {"eigs": eigs, "basis": basis, "n_params": N, "k": k, "T": T,
            "k_sketch": kk, "l_sketch": ll, "recovered_rank": int(eigs.numel()),
            "n_cal": n_cal, "fit_seconds": fit_s,
            "basis_seconds": time.perf_counter() - t1, "peak_rss_gib": peak,
            "storage_mib": (eigs.numel() + basis.numel()) * 4 / 1024 ** 2}


@torch.no_grad()
def score(ad, scope: str, proj: Projector, X: torch.Tensor, n_eigs: int,
          block: int = 128) -> np.ndarray:
    sc = SCOPES[scope](ad)
    basisS = sc.split(proj.basis[:, -n_eigs:].T.contiguous())         # (n_eigs, N)
    eigs = torch.clamp(proj.eigs[-n_eigs:], min=0.0)
    scaling = torch.sqrt(eigs / (eigs + 1.0 / (2 * MEPS))).double()
    out = np.empty(X.shape[0], dtype=np.float64)
    for s in range(0, X.shape[0], block):
        ctx = sc.prepare(X[s:s + block])
        pr, spr = ctx["p"][None], ctx["sp"][None]
        BL = right_LF(sc.AT(ctx, basisS).permute(0, 2, 1).double(), pr, spr)
        tot = sc.normsq(ctx)                                          # (B,)
        prj = ((scaling[:, None, None] * BL) ** 2).sum(-1).sum(0)     # (B,)
        out[s:s + ctx["B"]] = torch.sqrt(torch.clamp(tot - prj, min=0.0)).numpy()
        del ctx, BL
        reset_cuda()
    return out


# ----------------------------------------------------------------- validation
@torch.no_grad()
def _rand(shape, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(*shape, generator=g)


def validate(ad, scope: str, n: int = 2, R: int = 5, seed: int = 0) -> dict:
    """Three gates on the exact factorisation, on the real ViT tail.

    1. adjointness   <A q, v> == <q, A^T v>
    2. columns of A  A e_c == grad_w logits_c   (autograd)
    3. norms         tr(A F A^T) == sum_c p_c ||A e_c||^2 - ||A p||^2  (explicit)
    """
    from . import full_cache as fc
    from . import fu_common as F
    sc = SCOPES[scope](ad)
    corr = fc.load_cache(F.SRC / "raw" / "cache_correction.npz")
    X = torch.as_tensor(corr["x_resid_cls"][:n], device=ad.device, dtype=ad.dtype)

    # ---- 1. adjointness (one example at a time: A_apply sums over the batch) ----
    worst_adj = 0.0
    for i in range(n):
        ctx = sc.prepare(X[i:i + 1])
        Q = _rand((C_OUT, R, 1), seed + i)
        V = _rand((R, sc.n_params), seed + 100 + i)
        lhs = sc.A_apply(ctx, Q).T @ V.T                       # (R_Q, R_V)
        rhs = torch.einsum("crb,vcb->rv", Q, sc.AT(ctx, sc.split(V)))
        worst_adj = max(worst_adj, float((lhs - rhs).abs().max()
                                         / (lhs.abs().max() + 1e-30)))

    # ---- 2. A e_c against autograd on the logits ----
    head = sc.head
    leaves = ([(head.weight, False), (head.bias, False)] if scope == "linear"
              else [(ad.mlp1.weight, True), (ad.mlp1.bias, False),
                    (ad.mlp2.weight, True), (ad.mlp2.bias, False),
                    (head.weight, False), (head.bias, False)])
    params = [p for p, _ in leaves]
    prev = [p.requires_grad for p in params]
    for p in params:
        p.requires_grad_(True)
    worst_col = 0.0
    classes = [0, 7, 499, 999]
    for i in range(n):
        ctx = sc.prepare(X[i:i + 1])
        E = torch.zeros(C_OUT, len(classes), 1)
        for j, c in enumerate(classes):
            E[c, j, 0] = 1.0
        got = sc.A_apply(ctx, E)                                # (N, len(classes))
        x = X[i:i + 1]
        h = ad.block.ln_2(x[:, None, :])[:, 0]
        y = torch.nn.functional.gelu(h @ ad.W1 + ad.b1)
        phi = ad.enc.ln((x + y @ ad.W2 + ad.b2)[:, None, :])[:, 0]
        lg = head(phi)[0]
        for j, c in enumerate(classes):
            gs = torch.autograd.grad(lg[c], params, retain_graph=True)
            ref = torch.cat([(g.T if t else g).reshape(-1)
                             for g, (_, t) in zip(gs, leaves)]).cpu()
            worst_col = max(worst_col,
                            float((ref - got[:, j]).norm() / (ref.norm() + 1e-30)))
    for p, w in zip(params, prev):
        p.requires_grad_(w)

    # ---- 3. tr(A F A^T) against an explicit class sum ----
    worst_norm = 0.0
    for i in range(n):
        ctx = sc.prepare(X[i:i + 1])
        p = ctx["p"][0]
        acc = 0.0
        for s in range(0, C_OUT, 250):
            E = torch.zeros(C_OUT, min(250, C_OUT - s), 1)
            for j in range(E.shape[1]):
                E[s + j, j, 0] = 1.0
            col = sc.A_apply(ctx, E).double()
            acc += float((p[s:s + E.shape[1]] * (col ** 2).sum(0)).sum())
            del E, col
        Pq = p.float()[:, None, None]
        ap = sc.A_apply(ctx, Pq).double()
        explicit = acc - float((ap ** 2).sum())
        ours = float(sc.normsq(ctx)[0])
        worst_norm = max(worst_norm, abs(ours - explicit) / abs(explicit))

    ok = worst_adj < 1e-4 and worst_col < 1e-4 and worst_norm < 1e-3
    return {"scope": scope, "n_examples": n, "adjoint_max_rel_error": worst_adj,
            "A_column_vs_autograd_max_rel_error": worst_col,
            "normsq_max_rel_error": worst_norm, "ok": bool(ok)}
