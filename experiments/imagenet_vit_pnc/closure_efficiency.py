"""Revision closure Part J: ImageNet P&C construction, storage and inference cost.

Measured on the frozen primary configuration (ViT-B/16, r=2, lambda=1000, K=20, M=20,
32,768 correction images, final-block CLS correction, ridge centred on the ORIGINAL W2).

Three things this deliberately does NOT do:
  * it does not time the one-time dataset download;
  * it uses the cached implementation that is actually recommended for practical use
    (one backbone pass over the correction pool, then sufficient statistics reused across
    all 20 members), not the naive preflight that re-ran the backbone per member;
  * it does not claim any inference-time advantage. P&C keeps O(M) member inference and
    the measured throughput says so.

Stages timed, from "base checkpoint loaded and manifests prepared" to "all 20 members'
perturbations and corrections materialised":

  feature_cache     one backbone pass over the 32,768 correction images -> CLS residual
                    cache, plus the derived h and z0 the ridge consumes
  basis             the shared K=20 perturbation basis and per-member coefficients
  sufficient_stats  per member: perturbed post-GELU design, G = X^T X, C = X^T z0
  ridge_solve       per member: Cholesky of (G + lambda I) and the 768-column solve

    .venv_vit/bin/python -m experiments.imagenet_vit_pnc.closure_efficiency
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from . import full_cache as fc
from . import fu_common as F
from . import pnc_core as pc
from .memprobe import GIB, cpu_peak_rss_gib, cpu_rss_gib, reset_cuda
from .vit_adapter import ViTPnCAdapter

OUT = Path("2026neurips_rebuttal_results/revision_experiment_closure")
K, M, LAM, R_TARGET, N_CAL, SEED = 20, 20, 1000.0, 2.0, 32768, 0
BATCH_INFER = 64
WARMUP, TIMED = 5, 30


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


# ------------------------------------------------------------------ construction
def construction(ad, rebuild_cache: bool) -> dict:
    reset_cuda()
    rss0 = cpu_rss_gib()
    t = {}

    # --- stage 1: feature cache over the correction pool -----------------------
    t0 = time.perf_counter()
    if rebuild_cache:
        from .full_data import ImageNetShards, shard_paths
        rows = np.load(F.SRC / "splits" / "correction_rows.npy")
        ds = ImageNetShards(shard_paths("train"))
        xs = []
        for imgs, _, _ in ds.iter_batches(rows, 16):
            xs.append(ad.prefix(imgs.to(ad.device, ad.dtype))[:, 0].cpu())
        X = torch.cat(xs).to(ad.device, ad.dtype)
        del xs
    else:
        corr = fc.load_cache(F.SRC / "raw" / "cache_correction.npz")
        X = torch.as_tensor(corr["x_resid_cls"][:N_CAL], device=ad.device, dtype=ad.dtype)
    with torch.inference_mode():
        h = ad.block.ln_2(X[:, None, :])[:, 0]
        z0 = torch.nn.functional.gelu(h @ ad.W1 + ad.b1) @ ad.W2 + ad.b2
    _sync()
    t["feature_cache_s"] = time.perf_counter() - t0

    # --- stage 2: shared basis + member coefficients ---------------------------
    t0 = time.perf_counter()
    U = pc.perturbation_basis(SEED, K)
    co = pc.member_coefficients(SEED, M, K)
    W1n = float(ad.W1.norm())
    scale = pc.base_scale(U, co, W1n, target_rel=R_TARGET)
    _sync()
    t["basis_s"] = time.perf_counter() - t0

    Theta0 = ad.theta().detach().double().cpu().numpy()
    t_stats = t_solve = 0.0
    members = []
    for m in range(M):
        t0 = time.perf_counter()
        dW1 = torch.as_tensor(pc.member_dW1(U, co[m], scale), device=ad.device,
                              dtype=ad.dtype)
        with torch.inference_mode():
            y = torch.nn.functional.gelu(h @ (ad.W1 + dW1) + ad.b1)
            Xd = pc.SufficientStats.augment(y)
            G = (Xd.T @ Xd).double().cpu().numpy()
            C = (Xd.T @ z0).double().cpu().numpy()
        _sync()
        t_stats += time.perf_counter() - t0

        t0 = time.perf_counter()
        Theta, _, _ = pc.cho_solve_shared(G, C, LAM, w_prior=Theta0)   # ORIGINAL centre
        t_solve += time.perf_counter() - t0
        members.append({"coeff": co[m].astype(np.float32),
                        "W2": Theta[1:].astype(np.float32),
                        "b2": Theta[0].astype(np.float32)})
        del dW1, y, Xd, G, C
    t["sufficient_stats_s"] = t_stats
    t["ridge_solve_s"] = t_solve
    t["total_construction_s"] = sum(t.values())
    t["peak_gpu_gib"] = torch.cuda.max_memory_allocated() / GIB
    t["peak_cpu_rss_gib"] = cpu_peak_rss_gib()
    t["cpu_rss_before_gib"] = rss0
    t["n_cal"] = N_CAL
    t["cache_rebuilt_from_images"] = bool(rebuild_cache)
    del X, h, z0
    reset_cuda()
    return t, members, U, scale


# ------------------------------------------------------------------ storage
def storage(ad, members, U, scale) -> dict:
    tmp = OUT / "_tmp_storage"
    tmp.mkdir(parents=True, exist_ok=True)

    def sz(path):
        return Path(path).stat().st_size

    ck = tmp / "base_checkpoint.pt"
    torch.save(ad.model.state_dict(), ck)
    base_bytes = sz(ck)

    compact = tmp / "pnc_compact.npz"
    np.savez(compact,
             coefficients=np.stack([m["coeff"] for m in members]),
             W2=np.stack([m["W2"] for m in members]),
             b2=np.stack([m["b2"] for m in members]),
             scale=np.float32(scale), temperature=np.float32(F.temperature()))
    compact_bytes = sz(compact)

    basis = tmp / "shared_basis.npz"
    np.savez(basis, U=U.astype(np.float32))
    basis_bytes = sz(basis)

    coeff_only = tmp / "coeffs.npz"
    np.savez(coeff_only, coefficients=np.stack([m["coeff"] for m in members]))
    coeff_bytes = sz(coeff_only)

    w2_only = tmp / "w2.npz"
    np.savez(w2_only, W2=np.stack([m["W2"] for m in members]),
             b2=np.stack([m["b2"] for m in members]))
    w2_bytes = sz(w2_only)

    out = {
        "base_checkpoint_mib": base_bytes / 1024 ** 2,
        "pnc_compact_total_mib": compact_bytes / 1024 ** 2,
        "shared_basis_mib": basis_bytes / 1024 ** 2,
        "per_member_coefficients_mib": coeff_bytes / 1024 ** 2,
        "per_member_corrected_W2_b2_mib": w2_bytes / 1024 ** 2,
        "temperature_metadata_bytes": 8,
        "pnc_total_with_base_mib": (base_bytes + compact_bytes) / 1024 ** 2,
        "full_ensemble_20_checkpoints_mib_DERIVED": base_bytes * M / 1024 ** 2,
        "full_ensemble_note": "DERIVED as 20 x one checkpoint; the 20 full checkpoints "
                              "were never physically written.",
        "ratio_pnc_over_full_ensemble": (base_bytes + compact_bytes) / (base_bytes * M),
        "basis_stored": "the shared basis is regenerable from (seed, K) and is NOT part "
                        "of the shipped compact representation; measured here for "
                        "completeness",
    }
    for p in (ck, compact, basis, coeff_only, w2_only):
        p.unlink()
    tmp.rmdir()
    return out


# ------------------------------------------------------------------ inference
@torch.no_grad()
def inference(ad, members, scale: float) -> dict:
    val = fc.load_cache(F.cache_path("val50k"))
    Xb = torch.as_tensor(val["x_resid_cls"][:BATCH_INFER], device=ad.device,
                         dtype=ad.dtype)
    from .full_data import ImageNetShards, shard_paths
    ds = ImageNetShards(shard_paths("val"))
    imgs, _ = ds.load(np.arange(BATCH_INFER))
    imgs = imgs.to(ad.device, ad.dtype)

    def timeit(fn):
        for _ in range(WARMUP):
            fn()
        _sync()
        t0 = time.perf_counter()
        for _ in range(TIMED):
            fn()
        _sync()
        return (time.perf_counter() - t0) / TIMED

    reset_cuda()
    t_base = timeit(lambda: ad.model(imgs))
    vram_base = torch.cuda.max_memory_allocated() / GIB

    U = pc.perturbation_basis(SEED, K)
    mems = [{"W1v": ad.W1 + torch.as_tensor(pc.member_dW1(U, m["coeff"], scale),
                                            device=ad.device, dtype=ad.dtype),
             "W2": torch.as_tensor(m["W2"], device=ad.device, dtype=ad.dtype),
             "b2": torch.as_tensor(m["b2"], device=ad.device, dtype=ad.dtype)}
            for m in members]

    reset_cuda()

    def pnc_forward():
        x = ad.prefix(imgs)[:, 0]
        for mem in mems:
            hh = ad.block.ln_2(x[:, None, :])[:, 0]
            yv = torch.nn.functional.gelu(hh @ mem["W1v"] + ad.b1)
            ad.head(ad.enc.ln((x + yv @ mem["W2"] + mem["b2"])[:, None, :])[:, 0])

    t_pnc = timeit(pnc_forward)
    vram_pnc = torch.cuda.max_memory_allocated() / GIB
    del Xb, imgs, mems
    reset_cuda()
    return {
        "batch_size": BATCH_INFER, "warmup_iters": WARMUP, "timed_iters": TIMED,
        "base_s_per_batch": t_base,
        "base_images_per_s": BATCH_INFER / t_base,
        "base_ms_per_image": 1000 * t_base / BATCH_INFER,
        "base_peak_vram_gib": vram_base,
        "pnc_M20_s_per_batch": t_pnc,
        "pnc_M20_images_per_s": BATCH_INFER / t_pnc,
        "pnc_M20_ms_per_image": 1000 * t_pnc / BATCH_INFER,
        "pnc_M20_peak_vram_gib": vram_pnc,
        "slowdown_vs_base": t_pnc / t_base,
        "note": "P&C retains O(M) member inference: the backbone prefix is shared, but "
                "each of the 20 members needs its own final-block FFN and head pass. No "
                "inference-time advantage is claimed.",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild-cache", action="store_true",
                    help="re-run the backbone over the 32,768 correction images instead "
                         "of loading the persisted CLS cache")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    ad = ViTPnCAdapter(device="cuda", dtype=torch.float32)

    print("=== construction ===")
    con, members, U, scale = construction(ad, a.rebuild_cache)
    for k, v in con.items():
        print(f"  {k:<32} {v}")

    print("\n=== storage ===")
    sto = storage(ad, members, U, scale)
    for k, v in sto.items():
        print(f"  {k:<40} {v}")

    print("\n=== inference ===")
    inf = inference(ad, members, scale)
    for k, v in inf.items():
        print(f"  {k:<28} {v}")

    res = {"config": {"model": "ViT-B/16 IMAGENET1K_V1", "r": R_TARGET, "lambda": LAM,
                      "K": K, "M": M, "n_correction_images": N_CAL,
                      "ridge_center": "original", "target": "final block FFN, CLS rows"},
           "construction": con, "storage": sto, "inference": inf,
           "gpu": torch.cuda.get_device_name(0)}
    name = ("imagenet_efficiency_full_construction.json" if a.rebuild_cache
            else "imagenet_efficiency.json")
    (OUT / name).write_text(json.dumps(res, indent=2))
    print(f"\nwrote {OUT}/{name}")


if __name__ == "__main__":
    main()
