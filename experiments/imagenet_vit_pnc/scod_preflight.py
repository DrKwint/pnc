"""Measured SCOD feasibility preflight on ViT-B/16 (spec §13).

SCOD sketches the dataset Fisher F = Σ_n J̃_nᵀ J̃_n over the parameters it differentiates
through, using the single-pass Nyström sketch of Tropp et al. with
``num_samples = 6·num_eigs + 4``. Only F's action on a vector is needed — a JVP followed by
a VJP over the calibration set — but the sketch itself stores two P × num_samples matrices.

This module measures the two things that decide feasibility rather than asserting them:

  1. wall-clock of one Fisher matvec (JVP + VJP over N images) at N = 256 and N = 1,024,
     extrapolated to the repo's calibration size;
  2. peak memory of a single matvec, and the storage the Nyström sketch would require.

The repo's SCOD implementation has **no categorical likelihood** — it covers Case A/B
diagonal Gaussians for MuJoCo regression only. To measure anything at all, the softmax
Fisher factor is constructed here: for logits f with p = softmax(f), the Fisher in logit
space is F_θ = diag(p) − ppᵀ = AᵀA with A = diag(√p)(I − 1pᵀ), verified numerically below.
That construction exists solely to make the *cost* measurable; it is **not** offered as a
ported SCOD baseline (§13 forbids substituting a new SCOD-like score).
"""
from __future__ import annotations

import json
import time

import numpy as np
import torch

from . import full_cache as fc
from .baselines_vit import OUT, SRC, write_json
from .memprobe import GIB, cpu_peak_rss_gib, reset_cuda
from .vit_adapter import ViTPnCAdapter

try:                                    # torch >= 2.0 attention backend control
    from torch.nn.attention import SDPBackend, sdpa_kernel
except ImportError:                     # pragma: no cover
    SDPBackend = sdpa_kernel = None

NUM_EIGS_MAX = 100          # configs/mujoco_posthoc.yaml
NUM_SAMPLES = 604           # 6*100 + 4
CAL_N = 32768               # the P&C calibration pool


def _fisher_factor(logits: torch.Tensor) -> torch.Tensor:
    """A with AᵀA = diag(p) − ppᵀ, detached (the Fisher factor is held constant)."""
    p = torch.softmax(logits, -1).detach()
    sp = p.sqrt()
    return sp[..., :, None] * (torch.eye(p.shape[-1], device=p.device, dtype=p.dtype)
                               - p[..., None, :])


def verify_factor(device="cuda") -> dict:
    g = torch.Generator(device="cpu").manual_seed(0)
    f = torch.randn(4, 50, generator=g).to(device).double()
    p = torch.softmax(f, -1)
    A = _fisher_factor(f)
    F_direct = torch.diag_embed(p) - p[..., None] * p[..., None, :]
    F_fac = A.transpose(-1, -2) @ A
    err = float((F_direct - F_fac).abs().max())
    return {"max_abs_error": err, "ok": err < 1e-10}


def run(device: str = "cuda"):
    ad = ViTPnCAdapter(device=device, dtype=torch.float32)
    P = sum(p.numel() for p in ad.model.parameters())
    ver = verify_factor(device)
    print(f"softmax Fisher factor check: max|AᵀA − (diag(p)−ppᵀ)| = "
          f"{ver['max_abs_error']:.2e} -> {'OK' if ver['ok'] else 'FAIL'}")
    print(f"ViT-B/16 parameters P = {P:,}")

    params = {k: v.detach() for k, v in ad.model.named_parameters()}
    names = list(params)
    buffers = {k: v.detach() for k, v in ad.model.named_buffers()}
    omega = {k: torch.randn_like(v) for k, v in params.items()}

    def t_of_params(pd, x):
        """Fisher-weighted output t(θ) = A_detached · f(θ), flattened."""
        out = torch.func.functional_call(ad.model, ({**pd, **buffers}), (x,))
        A = _fisher_factor(out)
        return torch.einsum("bij,bj->bi", A, out).reshape(-1)

    val = fc.load_cache(SRC / "raw" / "cache_val50k.npz")
    from .full_data import ImageNetShards, shard_paths
    ds = ImageNetShards(shard_paths("val"))

    timings = {}
    for n in (256, 1024):
        idx = np.arange(n)
        imgs, _ = ds.load(idx)
        reset_cuda()
        t0 = time.perf_counter()
        acc = {k: torch.zeros_like(v) for k, v in params.items()}
        bs = 8
        # the fused SDPA kernels implement neither forward-AD nor a second derivative;
        # the MATH backend is the only one that supports the double-VJP below
        mk_ctx = ((lambda: sdpa_kernel(SDPBackend.MATH)) if sdpa_kernel
                  else torch.enable_grad)
        for s in range(0, n, bs):
          with mk_ctx():
            xb = imgs[s:s + bs].to(ad.device, ad.dtype)
            f = lambda pd: t_of_params(pd, xb)                       # noqa: E731
            # torchvision ViT calls the fused aten::_native_multi_head_attention, which
            # has no forward-AD rule, so torch.func.jvp fails outright. Obtain J̃ω by the
            # double-VJP identity instead (reverse mode only):
            #     w(u) = J̃ᵀu   ;   J̃ω = d/du <w(u), ω>
            t_out, vjp_fn = torch.func.vjp(f, params)
            u = torch.zeros_like(t_out, requires_grad=True)
            with torch.enable_grad():
                _, vjp_u = torch.func.vjp(lambda uu: vjp_fn(uu)[0], u)
                jvp_out = vjp_u(omega)[0]                            # J̃ ω
            g = vjp_fn(jvp_out)[0]                                   # J̃ᵀ (J̃ ω)
            for k in acc:
                acc[k] += g[k]
            del xb, jvp_out, vjp_fn, g
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        peak = torch.cuda.max_memory_allocated() / GIB
        timings[n] = {"matvec_seconds": dt, "seconds_per_image": dt / n,
                      "peak_gpu_gib": peak, "batch_size": bs}
        print(f"  one Fisher matvec over N={n:>5}: {dt:6.1f} s "
              f"({dt/n*1000:.1f} ms/image), peak GPU {peak:.2f} GiB")
        del acc, imgs
        reset_cuda()

    per_img = timings[1024]["seconds_per_image"]
    sketch_bytes = P * NUM_SAMPLES * 4
    one_vec_gib = P * 4 / GIB
    est = {
        "parameters_P": int(P),
        "num_eigs_max": NUM_EIGS_MAX, "num_samples": NUM_SAMPLES,
        "source_config": "configs/mujoco_posthoc.yaml",
        "seconds_per_image_per_matvec": per_img,
        "matvec_seconds_at_32768": per_img * CAL_N,
        "total_sketch_seconds_at_32768": per_img * CAL_N * NUM_SAMPLES,
        "total_sketch_hours_at_32768": per_img * CAL_N * NUM_SAMPLES / 3600,
        "sketch_test_matrix_gib": sketch_bytes / GIB,
        "sketch_output_gib": sketch_bytes / GIB,
        "sketch_total_gib": 2 * sketch_bytes / GIB,
        "one_parameter_vector_gib": one_vec_gib,
        "gpu_total_gib": 12.0, "system_ram_gib": 24.6,
        "rank10_sketch_total_gib": 2 * P * 64 * 4 / GIB,
    }
    verdict = ("SCOD_NOT_TRACTABLE_AT_VIT_SCALE"
               if est["sketch_total_gib"] > 24.6 else "TRACTABLE")
    out = {"verdict": verdict, "fisher_factor_check": ver,
           "timings": timings, "extrapolation": est,
           "blockers": [
               "the repository's SCOD implements only Gaussian Case A/B (MuJoCo "
               "regression); there is no categorical likelihood to port",
               f"the Nystrom sketch at the repo's own num_samples={NUM_SAMPLES} needs "
               f"{2*sketch_bytes/GIB:.0f} GiB against 12 GiB VRAM and 24.6 GiB RAM",
           ]}
    write_json(OUT / "metrics" / "scod_preflight.json", out)
    print(f"\n  extrapolated to N={CAL_N}, num_samples={NUM_SAMPLES}:")
    print(f"    one matvec over the pool : {est['matvec_seconds_at_32768']/60:.1f} min")
    print(f"    full sketch              : {est['total_sketch_hours_at_32768']:.0f} h")
    print(f"    sketch storage (Omega+Y) : {est['sketch_total_gib']:.0f} GiB "
          f"(one parameter vector alone is {one_vec_gib:.2f} GiB)")
    print(f"    even at num_eigs=10      : {est['rank10_sketch_total_gib']:.0f} GiB")
    print(f"\n  VERDICT: {verdict}")
    return out


if __name__ == "__main__":
    run()
