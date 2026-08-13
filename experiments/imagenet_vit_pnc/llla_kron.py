"""LLLA-Kron — scalable last-layer Laplace via the official `laplace-torch` (spec Part C).

The dense last-layer Laplace of the CIFAR code is impossible here (769,000² covariance =
2.37 TB), so this is a **Kronecker-factored last-layer Laplace** and is labelled
`LLLA-Kron` throughout. It is not claimed to be numerically identical to the dense variant.

Because the ViT backbone is frozen, last-layer Laplace over the full network is exactly
Laplace over the classifier head applied to the frozen features. The fit therefore runs on
a head-only `nn.Linear(768, 1000)` carrying the checkpoint's own weights, fed the cached
CLS features — `subset_of_weights="all"` on that module is the same object as
`subset_of_weights="last_layer"` on the full ViT, and it avoids backpropagating through the
backbone (spec C3 permits exactly this frozen-feature wrapper).

Predictive: the GLM/linearised Laplace predictive with the probit link approximation
(`pred_type="glm"`, `link_approx="probit"`), the package's standard multiclass
recommendation, with an MC-sampling parity check on an ID subset (C5).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from experiments.banking77_pnc.pnc_metrics import clf_metrics

from . import full_cache as fc
from .memprobe import GIB, cpu_peak_rss_gib, reset_cuda
from .vit_adapter import ViTPnCAdapter

SRC = Path("results/neurips_2026_rebuttal/imagenet_vit")
OUT = Path("results/neurips_2026_rebuttal/imagenet_vit_geometry_scod_llla")
PRIOR_GRID = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8]
EXTEND = [1e9, 1e10, 1e11]


def head_module(ad) -> nn.Linear:
    src = ad.model.heads[-1] if hasattr(ad.model.heads, "__getitem__") else ad.model.heads
    h = nn.Linear(src.in_features, src.out_features).to(ad.device, ad.dtype)
    h.weight.data.copy_(src.weight.data)
    h.bias.data.copy_(src.bias.data)
    h.eval()
    return h


def features_and_labels(ad, name: str):
    c = fc.load_cache(SRC / "raw" / f"cache_{name}.npz")
    X = fc.cache_to_gpu(c, ad)
    phi = fc.features_from_cache(ad, X)
    # features_from_cache runs under inference_mode, so its output is an inference tensor
    # and laplace-torch's GLM predictive cannot backprop through it. Round-trip to numpy
    # to get an ordinary autograd-capable tensor.
    phi = torch.from_numpy(phi.cpu().numpy()).to(ad.device, ad.dtype)
    del X
    reset_cuda()
    return phi, torch.from_numpy(c["labels"]).long()


def fit(structure: str = "kron", n_cal: int | None = None):
    from laplace import Laplace
    import importlib.metadata as md

    ad = ViTPnCAdapter(device="cuda", dtype=torch.float32)
    head = head_module(ad)
    phi_c, y_c = features_and_labels(ad, "correction")
    if n_cal:
        phi_c, y_c = phi_c[:n_cal], y_c[:n_cal]
    phi_s, y_s = features_and_labels(ad, "selection")

    loader = DataLoader(TensorDataset(phi_c.cpu(), y_c), batch_size=512, shuffle=False)
    reset_cuda()
    t0 = time.perf_counter()
    # subset_of_weights="last_layer" is essential, not cosmetic: the generic path builds
    # the Jacobian with torch.func.jacrev, which for a 1000-class 769k-parameter head asks
    # for a 62,500 GiB tensor. LLLaplace uses the analytic linear-layer Jacobian instead.
    import collections
    wrapped = nn.Sequential(collections.OrderedDict(
        [("feat", nn.Identity()), ("head", head)])).to(ad.device, ad.dtype).eval()
    la = Laplace(wrapped, likelihood="classification", subset_of_weights="last_layer",
                 hessian_structure=structure, last_layer_name="head")
    la.fit(loader)
    t_fit = time.perf_counter() - t0
    peak_gpu = torch.cuda.max_memory_allocated() / GIB
    print(f"  {structure}: fitted on {len(y_c):,} in {t_fit:.1f}s "
          f"(peak GPU {peak_gpu:.2f} GiB, peak CPU {cpu_peak_rss_gib():.2f} GiB)")

    rows, grid = [], list(PRIOR_GRID)
    i = 0
    while i < len(grid):
        lam = grid[i]
        i += 1
        la.prior_precision = torch.tensor(float(lam), device=ad.device)
        p = glm_probit_probs(la, head, phi_s)     # closed form; see note below
        m = clf_metrics(p, y_s.numpy())
        rows.append({"prior_precision": lam, "top1": m["accuracy"], "nll": m["nll"],
                     "ece": m["ece"]})
        print(f"    lam={lam:<9g} top1 {m['accuracy']*100:7.3f}  NLL {m['nll']:.4f}  "
              f"ECE {m['ece']:.4f}")
        if i == len(grid):                       # extend if the optimum is on the edge
            best = min(rows, key=lambda r: r["nll"])
            if best["prior_precision"] == grid[-1]:
                nxt = next((v for v in EXTEND if v not in grid), None)
                if nxt:
                    grid.append(nxt)
                    print(f"      optimum on grid edge -> extending to {nxt:g}")
    best = min(rows, key=lambda r: r["nll"])
    print(f"  -> selected prior_precision={best['prior_precision']:g} "
          f"(ID selection NLL {best['nll']:.4f})")
    la.prior_precision = torch.tensor(float(best["prior_precision"]), device=ad.device)
    return ad, la, rows, best, {"fit_seconds": t_fit, "peak_gpu_gib": peak_gpu,
                                "peak_cpu_gib": cpu_peak_rss_gib(),
                                "n_cal": int(len(y_c)),
                                "laplace_version": md.version("laplace-torch"),
                                "curvlinops_version": md.version("curvlinops-for-pytorch"),
                                "structure": structure}


@torch.no_grad()
def _predict(la, phi: torch.Tensor, chunk: int = 4096, pred_type="glm",
             link_approx="probit") -> np.ndarray:
    out = []
    for s in range(0, phi.shape[0], chunk):
        out.append(la(phi[s:s + chunk], pred_type=pred_type,
                      link_approx=link_approx).cpu().numpy())
    return np.concatenate(out)


@torch.no_grad()
def _mc_link(la, head, phi: torch.Tensor, n_samples: int = 100, chunk: int = 2048):
    """MC link approximation over the same closed-form logit posterior (C5 parity check)."""
    p = kron_predictive_parts(la)
    bias_var = p["Qb2"] @ p["vb"]
    g = torch.Generator(device="cpu").manual_seed(0)
    out = []
    for s in range(0, phi.shape[0], chunk):
        x = phi[s:s + chunk]
        f_mu = head(x)
        var = ((x @ p["QA"]) ** 2) @ p["M"].T @ p["QS2"].T + bias_var[None, :]
        sd = var.clamp(min=0).sqrt()
        acc = torch.zeros_like(f_mu, dtype=torch.float64)
        for _ in range(n_samples):
            eps = torch.randn(f_mu.shape, generator=g).to(f_mu.device, f_mu.dtype)
            acc += torch.softmax(f_mu + sd * eps, -1).double()
        out.append((acc / n_samples).cpu().numpy())
    return np.concatenate(out)


def mc_parity(la, head, phi: torch.Tensor, n: int = 4096, n_samples: int = 100) -> dict:
    """C5: check the GLM/probit predictive against MC posterior sampling on an ID subset."""
    from scipy.stats import spearmanr
    sub = phi[:n]
    p_glm = glm_probit_probs(la, head, sub)
    # MC over the same closed-form logit posterior: f ~ N(f_mu, var), averaged softmax
    p_mc = _mc_link(la, head, sub, n_samples=n_samples)
    ent = lambda p: -np.sum(p * np.log(p + 1e-12), -1)
    return {"n": int(n), "mc_link_samples": n_samples,
            "entropy_spearman": float(spearmanr(ent(p_glm), ent(p_mc)).statistic),
            "max_abs_prob_diff": float(np.abs(p_glm - p_mc).max()),
            "top1_agreement": float((p_glm.argmax(-1) == p_mc.argmax(-1)).mean())}


# ---------------------------------------------------------------------------
# Closed-form last-layer GLM predictive
#
# laplace-torch computes the functional variance by materialising the last-layer
# Jacobian, which for a 1000-class 769k-parameter head is an 11,718 GiB tensor. For a
# linear head J = [phi (x) I_K, I_K], so with the fitted Kronecker posterior
#   P_W ~ (Q_S L_S Q_S^T) (x) (Q_A L_A Q_A^T),  Sigma_W = (Q_S (x) Q_A) diag(1/(l_S l_A + d)) (...)^T
# the diagonal of J Sigma J^T is available in closed form and never needs J:
#   a      = Q_A^T phi
#   s_k    = sum_j a_j^2 / (l_S,k * l_A,j + d)
#   var_c  = sum_k Q_S[c,k]^2 s_k        (+ the bias group, same shape with L_b)
# The package's fitted factors are used unchanged; only the predictive is reformulated.
# `validate_closed_form` checks it against the library's own functional_variance on a
# small head where the explicit Jacobian does fit.
# ---------------------------------------------------------------------------
def kron_predictive_parts(la):
    pp = la.posterior_precision
    ev, Q = pp.eigenvalues, pp.eigenvectors
    d = pp.deltas.detach()
    lS, lA = ev[0][0].detach(), ev[0][1].detach()          # output 1000, input 768
    QS, QA = Q[0][0].detach(), Q[0][1].detach()
    lb, Qb = ev[1][0].detach(), Q[1][0].detach()
    dW = float(d[0]) ** 0.5 if pp.damping else float(d[0])
    db = float(d[1]) ** 0.5 if pp.damping else float(d[1])
    if pp.damping:
        M = 1.0 / (torch.outer(lS + dW, lA + dW))
        vb = 1.0 / (lb + db) ** 2
    else:
        M = 1.0 / (torch.outer(lS, lA) + float(d[0]))
        vb = 1.0 / (lb + float(d[1]))
    return {"M": M, "QA": QA, "QS2": QS ** 2, "Qb2": Qb ** 2, "vb": vb}


@torch.no_grad()
def glm_probit_probs(la, head, phi: torch.Tensor, chunk: int = 4096) -> np.ndarray:
    """GLM predictive with the probit link, computed without forming any Jacobian."""
    p = kron_predictive_parts(la)
    bias_var = (p["Qb2"] @ p["vb"])                                    # (K,)
    out = []
    for s in range(0, phi.shape[0], chunk):
        x = phi[s:s + chunk]
        f_mu = head(x)
        a2 = (x @ p["QA"]) ** 2                                        # (n, 768)
        sk = a2 @ p["M"].T                                             # (n, 1000)
        var = sk @ p["QS2"].T + bias_var[None, :]                      # (n, 1000)
        kappa = 1.0 / torch.sqrt(1.0 + (np.pi / 8.0) * var)
        out.append(torch.softmax(f_mu * kappa, -1).double().cpu().numpy())
    return np.concatenate(out)


def validate_closed_form(device="cuda", d_in=16, k_out=10, n=512, seed=0) -> dict:
    """Gate the closed form against laplace-torch's own functional_variance."""
    import collections
    from laplace import Laplace
    from torch.utils.data import DataLoader, TensorDataset
    torch.manual_seed(seed)
    h = nn.Linear(d_in, k_out).to(device).eval()
    w = nn.Sequential(collections.OrderedDict(
        [("feat", nn.Identity()), ("head", h)])).to(device).eval()
    X = torch.randn(n, d_in, device=device)
    y = torch.randint(0, k_out, (n,))
    la = Laplace(w, likelihood="classification", subset_of_weights="last_layer",
                 hessian_structure="kron", last_layer_name="head")
    la.fit(DataLoader(TensorDataset(X.cpu(), y), batch_size=128))
    la.prior_precision = torch.tensor(1.0, device=device)
    Js, _ = la.backend.last_layer_jacobians(X[:64])
    ref = la.functional_variance(Js)
    ref_diag = torch.diagonal(ref, dim1=-2, dim2=-1)
    p = kron_predictive_parts(la)
    a2 = (X[:64] @ p["QA"]) ** 2
    mine = a2 @ p["M"].T @ p["QS2"].T + (p["Qb2"] @ p["vb"])[None, :]
    rel = float((mine - ref_diag).abs().max() / ref_diag.abs().max())
    return {"max_rel_error": rel, "ok": rel < 1e-4,
            "n_checked": 64, "d_in": d_in, "k_out": k_out}
