"""Memory / throughput stages of the preflight (spec sections 6-9, 15-17).

Every stage records into ``memory_preflight.csv`` through :mod:`memprobe`, so an OOM
is captured as a row rather than killing the sweep.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from . import pnc_core as pc
from .memprobe import GIB, Recorder, cpu_peak_rss_gib, cpu_rss_gib, probe, reset_cuda, timed

# The spec lists 1..32; memory turned out to be nowhere near binding on this card, so
# the sweep continues past 32 until the OOM / 11 GiB stop rule actually fires.
BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64, 128, 256]
RESERVED_CEILING_GIB = 11.0      # stop the sweep past this (spec section 6)
HEADROOM_GIB = 1.5               # required free VRAM at the chosen batch size
TOTAL_VRAM_GIB = 12.0


def _synthetic(batch: int, device, dtype):
    return torch.randn(batch, 3, 224, 224, device=device, dtype=dtype)


# ---------------------------------------------------------------- section 6
def stage_batch_sweep(ad, rec: Recorder, thr: Recorder, dtype_name="float32",
                      dtype=torch.float32, baseline_other_gib=0.0):
    """Batch-size sweep on synthetic B x 3 x 224 x 224 tensors; returns the safe batch."""
    rows = []
    for b in BATCH_SIZES:
        with probe(rec, "base_inference", batch_size=b, dtype=dtype_name,
                   target_block="n/a", member_count=1, token_mode="n/a") as r:
            x = _synthetic(b, ad.device, dtype)
            t = timed(lambda: ad.model(x), warmup=3, iters=20)
            r["notes"] = (f"median {t['median_s']*1e3:.1f} ms, mean {t['mean_s']*1e3:.1f} ms, "
                          f"{b / t['median_s']:.1f} img/s")
            r["_timing"] = t
            del x
        row = r["row"]
        if r["status"] == "ok":
            t = r["_timing"]
            thr.write(test="base_inference", batch_size=b, dtype=dtype_name,
                      token_mode="n/a", member_count=1,
                      rows_per_image="", rows_per_s="",
                      images_per_s=round(b / t["median_s"], 2),
                      median_s=round(t["median_s"], 6), mean_s=round(t["mean_s"], 6),
                      peak_allocated_gib=row["peak_allocated_gib"],
                      peak_reserved_gib=row["peak_reserved_gib"],
                      notes=f"{t['iters']} timed iters after 3 warmup")
        rows.append({"batch": b, **{k: row[k] for k in
                                    ("peak_allocated_gib", "peak_reserved_gib", "status")},
                     "median_s": r.get("_timing", {}).get("median_s")})
        print(f"  batch {b:>3}: status={r['status']:<5} peak_alloc="
              f"{row['peak_allocated_gib']:.3f} GiB peak_res={row['peak_reserved_gib']:.3f} GiB "
              f"{r['notes']}", flush=True)
        if r["status"] != "ok" or row["peak_reserved_gib"] > RESERVED_CEILING_GIB:
            print(f"  -> stopping sweep ({r['status']}, "
                  f"peak reserved {row['peak_reserved_gib']:.2f} GiB)", flush=True)
            break

    ok = [r for r in rows if r["status"] == "ok"]
    budget = TOTAL_VRAM_GIB - baseline_other_gib - HEADROOM_GIB
    viable = [r for r in ok if r["peak_reserved_gib"] <= budget]
    # Throughput-driven, not "largest that fits": this card saturates compute long before
    # memory, so every candidate clears the 1.5 GiB headroom rule by ~9 GiB. With the
    # headroom criterion non-binding, pick the throughput plateau and, among batches
    # within 5% of peak (run-to-run clock noise is ~3%), the largest -- it amortises the
    # per-member Python/launch overhead of the M-member loop best.
    safe = None
    if viable:
        for r in viable:
            r["images_per_s"] = r["batch"] / r["median_s"]
        best = max(r["images_per_s"] for r in viable)
        safe = max((r for r in viable if r["images_per_s"] >= 0.95 * best),
                   key=lambda r: r["batch"])
    return {"rows": rows, "safe_batch": safe["batch"] if safe else None,
            "safe_batch_images_per_s": safe["images_per_s"] if safe else None,
            "selection_rule": "throughput plateau (within 5% of peak), largest such batch; "
                              "headroom criterion non-binding (~9 GiB spare at every candidate)",
            "vram_budget_gib": budget, "baseline_other_gib": baseline_other_gib}


def stage_sustained(ad, rec: Recorder, thr: Recorder, batch: int, seconds: float = 60.0,
                    dtype_name="float32"):
    """Settled throughput under sustained load -- the number runtime projections must use.

    The cold batch sweep is measured on a cool card; this TITAN X drops its SM clock from
    ~1683 to ~1417 MHz and raises SwThermalSlowdown once it heats up, so short timings
    overstate what a multi-hour run achieves.
    """
    import subprocess

    def gpu_state():
        try:
            o = subprocess.run(
                ["nvidia-smi", "--query-gpu=clocks.sm,temperature.gpu,power.draw,"
                 "clocks_throttle_reasons.active", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10).stdout.strip()
            sm_clk, temp, power, throttle = [s.strip() for s in o.split(",")]
            return {"sm_clock_mhz": float(sm_clk), "temp_c": float(temp),
                    "power_w": float(power), "throttle_reasons": throttle}
        except Exception:
            return {}

    x = _synthetic(batch, ad.device, ad.dtype)
    start_state = gpu_state()
    with torch.inference_mode():
        for _ in range(3):
            ad.model(x)
    torch.cuda.synchronize()

    windows, n_img, t_start = [], 0, time.perf_counter()
    while time.perf_counter() - t_start < seconds:
        w0 = time.perf_counter()
        with torch.inference_mode():
            for _ in range(5):
                ad.model(x)
        torch.cuda.synchronize()
        dt = time.perf_counter() - w0
        windows.append(5 * batch / dt)
        n_img += 5 * batch
    end_state = gpu_state()

    first, last = windows[0], windows[-1]
    settled = float(np.median(windows[len(windows) // 2:]))     # second half = thermally settled
    out = {"batch": batch, "seconds": seconds, "images": n_img,
           "first_window_images_per_s": first, "last_window_images_per_s": last,
           "settled_images_per_s": settled,
           "overall_images_per_s": n_img / (time.perf_counter() - t_start),
           "decay_pct": 100.0 * (1 - last / first),
           "gpu_state_start": start_state, "gpu_state_end": end_state}
    with probe(rec, "sustained_throughput", batch_size=batch, dtype=dtype_name,
               target_block="n/a", member_count=1, token_mode="n/a") as r:
        r["notes"] = (f"{seconds:.0f}s sustained: {first:.1f} -> {last:.1f} img/s "
                      f"({out['decay_pct']:.1f}% decay), settled {settled:.1f} img/s; "
                      f"clock {start_state.get('sm_clock_mhz')} -> {end_state.get('sm_clock_mhz')} MHz, "
                      f"temp {start_state.get('temp_c')} -> {end_state.get('temp_c')} C")
    thr.write(test="sustained_throughput", batch_size=batch, dtype=dtype_name,
              token_mode="n/a", member_count=1, rows_per_image="", rows_per_s="",
              images_per_s=round(settled, 2), median_s=round(batch / settled, 6), mean_s="",
              peak_allocated_gib=r["row"]["peak_allocated_gib"],
              peak_reserved_gib=r["row"]["peak_reserved_gib"],
              notes=f"settled over {seconds:.0f}s; {out['decay_pct']:.1f}% thermal decay")
    del x
    reset_cuda()
    return out


# ---------------------------------------------------------------- section 7
def stage_activations(ad, rec: Recorder, batch: int, dtype=torch.float32,
                      dtype_name="float32"):
    """Capture FFN input / post-GELU / original output and report each one's cost."""
    out = {}
    x = _synthetic(batch, ad.device, dtype)
    reset_cuda()
    with torch.inference_mode():
        x_resid = ad.prefix(x)
    base_alloc = torch.cuda.memory_allocated()
    with torch.inference_mode():
        h, y, z = ad.ffn_triplet(x_resid)
    torch.cuda.synchronize()
    sizes = {"x_resid": x_resid, "h_ffn_input": h, "y_post_gelu": y, "z_ffn_output": z}
    for name, t in sizes.items():
        out[name] = {"shape": tuple(t.shape), "dtype": str(t.dtype),
                     "mib": t.numel() * t.element_size() / 1024**2,
                     "mib_per_image": t.numel() * t.element_size() / 1024**2 / batch}
    out["retained_increment_gib"] = (torch.cuda.memory_allocated() - base_alloc) / GIB
    out["n_tokens"] = int(x_resid.shape[1])
    with probe(rec, "activation_capture", batch_size=batch, dtype=dtype_name,
               target_block=ad.block_index, member_count=1, token_mode="all") as r:
        r["notes"] = (f"T={out['n_tokens']}; h {out['h_ffn_input']['mib']:.1f} MiB, "
                      f"y {out['y_post_gelu']['mib']:.1f} MiB, z {out['z_ffn_output']['mib']:.1f} MiB; "
                      f"retaining all three costs {out['retained_increment_gib']*1024:.1f} MiB")
    del x, x_resid, h, y, z
    reset_cuda()
    return out


# ---------------------------------------------------------------- section 8
def stage_basis(ad, rec: Recorder, seed=0, K=20, M=20):
    """Cost of the rank-K basis and of one on-demand member perturbation."""
    rss0 = cpu_rss_gib()
    t0 = time.perf_counter()
    U = pc.perturbation_basis(seed, K)
    t_basis = time.perf_counter() - t0
    rss1 = cpu_rss_gib()
    coeffs = pc.member_coefficients(seed, M, K)
    W1n = float(ad.W1.norm())
    scale = pc.base_scale(U, coeffs, W1n)

    t0 = time.perf_counter()
    dW1 = pc.member_dW1(U, coeffs[0], scale)
    t_member = time.perf_counter() - t0

    with probe(rec, "perturbation_basis", batch_size="", dtype="float32",
               target_block=ad.block_index, member_count=M, token_mode="n/a") as r:
        r["notes"] = (f"U {U.shape} = {U.nbytes/1024**2:.1f} MiB CPU built in {t_basis:.2f}s "
                      f"(no GPU); one dW1 {dW1.shape} = {dW1.nbytes/1024**2:.1f} MiB in "
                      f"{t_member*1e3:.1f} ms")
    info = {
        "K": K, "M": M, "D_flat": pc.D_FLAT,
        "basis_shape": list(U.shape),
        "basis_mib": U.nbytes / 1024**2,
        "basis_build_s": t_basis,
        "basis_gpu_mib": 0.0,
        "cpu_rss_before_gib": rss0, "cpu_rss_after_gib": rss1,
        "cpu_rss_delta_mib": (rss1 - rss0) * 1024,
        "member_dW1_mib": dW1.nbytes / 1024**2,
        "member_dW1_build_s": t_member,
        "numerical_scale": scale, "W1_frobenius": W1n,
        "rel_dW1_median": float(np.median(
            [np.linalg.norm(scale * (coeffs[m] @ U)) / W1n for m in range(M)])),
        "twenty_full_models_gib": M * 86_567_656 * 4 / GIB,
        "compact_state_mib": (U.nbytes + M * (pc.DHID * pc.DIN + pc.DIN + K) * 4) / 1024**2,
    }
    del dW1
    return info, U, coeffs, scale


# ---------------------------------------------------------------- section 9
def stage_single_member(ad, rec: Recorder, U, coeffs, scale, batch: int,
                        dtype=torch.float32, dtype_name="float32"):
    """Base vs uncorrected-perturbed forward, then exact restoration."""
    x = _synthetic(batch, ad.device, dtype)
    with torch.inference_mode():
        base = ad.model(x).clone()

    dW1 = torch.as_tensor(pc.member_dW1(U, coeffs[0], scale), device=ad.device, dtype=dtype)
    with probe(rec, "perturbed_member_forward", batch_size=batch, dtype=dtype_name,
               target_block=ad.block_index, member_count=1, token_mode="n/a") as r:
        W1v = ad.W1 + dW1
        with torch.inference_mode():
            x_resid = ad.prefix(x)
            pert = ad.tail(x_resid, W1=W1v).clone()
        t = timed(lambda: ad.tail(ad.prefix(x), W1=W1v), warmup=2, iters=10)
        r["notes"] = f"median {t['median_s']*1e3:.1f} ms"
    peak_perturbed = r["row"]["peak_allocated_gib"]

    # in-place mutation path + exact restoration
    ad.set_W1_code(ad.W1 + dW1)
    with torch.inference_mode():
        pert_full = ad.model(x).clone()
    ad.restore()
    with torch.inference_mode():
        restored = ad.model(x).clone()

    d_logit = float((pert - base).abs().max())
    out = {
        "batch": batch,
        "finite_base": bool(torch.isfinite(base).all()),
        "finite_perturbed": bool(torch.isfinite(pert).all()),
        "max_abs_logit_change": d_logit,
        "mean_abs_logit_change": float((pert - base).abs().mean()),
        "top1_agreement_base_vs_uncorrected": float(
            (pert.argmax(-1) == base.argmax(-1)).float().mean()),
        "functional_vs_inplace_max_diff": float((pert - pert_full).abs().max()),
        "restore_bitwise_equal": bool(torch.equal(restored, base)),
        "weights_pristine_after_restore": ad.weights_are_pristine(),
        "peak_allocated_gib": peak_perturbed,
        "perturbation_changes_logits": d_logit > 1e-4,
    }
    del x, dW1, base, pert, pert_full, restored
    reset_cuda()
    return out


# ------------------------------------------------------------- sections 15/16
@torch.inference_mode()
def _ensemble_probs(ad, x_resid, members, U, scale, stream: bool, W1v_cache=None):
    """Sequential shared-prefix member evaluation; accumulates mean softmax.

    ``W1v_cache`` holds the M perturbed W1 matrices precomputed once (9 MiB each). Without
    it every batch re-expands ``coeff @ U`` over the 2.36M-dim basis on the CPU, which for
    M=20 costs ~150 ms of pure basis expansion per batch and would dominate a 50k-image
    pass. The real experiment should precompute; both paths are measured.
    """
    acc = None
    for i, mem in enumerate(members):
        if W1v_cache is not None:
            W1v = W1v_cache[i]
            dW1 = None
        else:
            dW1 = torch.as_tensor(pc.member_dW1(U, mem["coeff"], scale),
                                  device=ad.device, dtype=ad.dtype)
            W1v = ad.W1 + dW1
        W2 = mem["W2"].to(ad.device, non_blocking=True) if stream else mem["W2"]
        b2 = mem["b2"].to(ad.device, non_blocking=True) if stream else mem["b2"]
        logits = ad.tail(x_resid, W1=W1v, W2=W2, b2=b2)
        p = torch.softmax(logits, -1)
        acc = p if acc is None else acc + p
        del logits, p
        if dW1 is not None:
            del dW1, W1v
        if stream:
            del W2, b2
    return acc / len(members)


def stage_ensemble(ad, rec: Recorder, thr: Recorder, members, U, scale, batch: int,
                   m_list=(1, 5, 20), stream: bool = False, dtype_name="float32",
                   precompute: bool = False):
    """Shared-prefix sequential inference at several ensemble sizes (spec section 15)."""
    out = []
    x = _synthetic(batch, ad.device, ad.dtype)
    mode = ("streaming" if stream else "resident") + ("_precomputed_W1" if precompute else "")
    cache_all = None
    if precompute:
        cache_all = [ad.W1 + torch.as_tensor(pc.member_dW1(U, m["coeff"], scale),
                                             device=ad.device, dtype=ad.dtype)
                     for m in members]
    for M in m_list:
        subset = members[:M]
        with probe(rec, f"ensemble_{mode}", batch_size=batch, dtype=dtype_name,
                   target_block=ad.block_index, member_count=M, token_mode="cls") as r:
            with torch.inference_mode():
                x_resid = ad.prefix(x)
            cache = cache_all[:M] if cache_all is not None else None
            t = timed(lambda: _ensemble_probs(ad, x_resid, subset, U, scale, stream, cache),
                      warmup=1, iters=5)
            r["notes"] = (f"{mode}; prefix once + {M} sequential tails; "
                          f"median {t['median_s']*1e3:.1f} ms")
            r["_t"] = t
        row = r["row"]
        rec_out = {"M": M, "mode": mode, "median_s": r["_t"]["median_s"],
                   "images_per_s": batch / r["_t"]["median_s"],
                   "peak_allocated_gib": row["peak_allocated_gib"],
                   "peak_reserved_gib": row["peak_reserved_gib"], "status": r["status"]}
        out.append(rec_out)
        thr.write(test=f"ensemble_{mode}", batch_size=batch, dtype=dtype_name,
                  token_mode="cls", member_count=M, rows_per_image="", rows_per_s="",
                  images_per_s=round(rec_out["images_per_s"], 2),
                  median_s=round(r["_t"]["median_s"], 6), mean_s=round(r["_t"]["mean_s"], 6),
                  peak_allocated_gib=row["peak_allocated_gib"],
                  peak_reserved_gib=row["peak_reserved_gib"],
                  notes=f"shared prefix, {mode} corrections"
                        + ("" if cache_all is None else
                           f"; {len(cache_all)} perturbed W1 precomputed "
                           f"({len(cache_all)*9.0:.0f} MiB GPU)"))
        print(f"  M={M:<3} {mode:<9} peak_alloc={row['peak_allocated_gib']:.3f} GiB  "
              f"{rec_out['images_per_s']:.1f} img/s", flush=True)
        del x_resid
        reset_cuda()
    del x
    if cache_all is not None:
        del cache_all
    reset_cuda()
    return out


def stage_storage(members, U, out_dir: Path, K=20):
    """Actual compact-state footprint on CPU, GPU and disk (spec section 16)."""
    M = len(members)
    per = pc.member_nbytes(K)
    npz = out_dir / "raw" / "compact_members_M20.npz"
    npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        npz, basis=U,
        coefficients=np.stack([m["coeff"] for m in members]),
        W2=np.stack([m["W2"].detach().cpu().numpy() for m in members]),
        b2=np.stack([m["b2"].detach().cpu().numpy() for m in members]))
    gpu_res = sum(m["W2"].numel() * m["W2"].element_size()
                  + m["b2"].numel() * m["b2"].element_size()
                  for m in members if m["W2"].is_cuda)
    return {
        "M": M,
        "shared_basis_mib": per["shared_basis"] / 1024**2,
        "coefficients_total_kib": M * per["coeff_per_member"] / 1024,
        "corrected_W2_total_mib": M * per["W2_per_member"] / 1024**2,
        "corrected_b2_total_kib": M * per["b2_per_member"] / 1024,
        "cpu_total_mib": (per["shared_basis"]
                          + M * (per["coeff_per_member"] + per["W2_per_member"]
                                 + per["b2_per_member"])) / 1024**2,
        "gpu_resident_corrections_mib": gpu_res / 1024**2,
        "disk_npz_mib": npz.stat().st_size / 1024**2,
        "disk_path": str(npz),
        "twenty_full_models_gib": M * 86_567_656 * 4 / GIB,
        "cpu_peak_rss_gib": cpu_peak_rss_gib(),
    }


def write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str))
