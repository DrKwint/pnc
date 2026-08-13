"""Measured runtime and memory for the final experiment (spec §31).

Assembles the costs already recorded by each stage and measures the few primitives no
stage records on its own (checkpoint load, basis construction, sustained throughput).

Throughput is reported **settled**, not cold: this TITAN X drops its SM clock from ~1683 to
~1417 MHz once it heats up, so a short cold timing overstates what a long run achieves.

The efficiency claim this supports is narrow and is stated as such in the report: P&C
builds its ensemble from one pretrained checkpoint without training independent ViTs. It is
**not** a claim of faster inference — M=20 members cost M tail evaluations.
"""
from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from . import full_cache as fc
from . import pnc_core as pc
from .memprobe import GIB, cpu_peak_rss_gib, reset_cuda
from .vit_adapter import ViTPnCAdapter


def _gpu_state() -> dict:
    try:
        o = subprocess.run(
            ["nvidia-smi", "--query-gpu=clocks.sm,temperature.gpu,power.draw,"
             "clocks_throttle_reasons.active", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10).stdout.strip()
        sm, t, p, thr = [x.strip() for x in o.split(",")]
        return {"sm_clock_mhz": float(sm), "temp_c": float(t), "power_w": float(p),
                "throttle": thr}
    except Exception:
        return {}


def collect(out: Path, sustain_s: float = 60.0) -> dict:
    res = {}

    t0 = time.perf_counter()
    ad = ViTPnCAdapter(device="cuda", dtype=torch.float32)
    res["checkpoint_load_s"] = time.perf_counter() - t0
    res["model_resident_gib"] = torch.cuda.memory_allocated() / GIB

    t0 = time.perf_counter()
    U = pc.perturbation_basis(0, 20)
    res["basis_construction_s"] = time.perf_counter() - t0
    res["basis_mib"] = U.nbytes / 1024**2
    res["basis_gpu_mib"] = 0.0

    # sustained ImageNet throughput at the experiment's batch size
    x = torch.randn(16, 3, 224, 224, device=ad.device)
    with torch.inference_mode():
        for _ in range(3):
            ad.model(x)
    torch.cuda.synchronize()
    start_state = _gpu_state()
    windows, n, t_start = [], 0, time.perf_counter()
    while time.perf_counter() - t_start < sustain_s:
        w0 = time.perf_counter()
        with torch.inference_mode():
            for _ in range(5):
                ad.model(x)
        torch.cuda.synchronize()
        windows.append(80 / (time.perf_counter() - w0))
        n += 80
    res["sustained"] = {
        "batch": 16, "images": n,
        "first_window_images_per_s": windows[0],
        "settled_images_per_s": float(np.median(windows[len(windows) // 2:])),
        "decay_pct": 100.0 * (1 - windows[-1] / windows[0]),
        "gpu_state_start": start_state, "gpu_state_end": _gpu_state(),
    }
    del x
    reset_cuda()

    # stage-recorded costs
    for name, path in (("cache_val50k", "timing/cache_val50k.json"),
                       ("cache_train_pools", "timing/cache_train_pools.json"),
                       ("cache_ood", "timing/cache_ood.json")):
        p = out / path
        if p.exists():
            res[name] = json.loads(p.read_text())

    idf = out / "metrics" / "id_final.json"
    if idf.exists():
        d = json.loads(idf.read_text())
        seeds = d["seeds"]
        res["ensemble_construction"] = {
            "M": d["config"]["M_final"], "n_cal": d["config"]["n_cal"],
            "seconds_per_ensemble_mean": float(np.mean(
                [s["construct_s"] for s in seeds.values()])),
            "seconds_per_member_mean": float(np.mean(
                [s["construct_s"] for s in seeds.values()])) / d["config"]["M_final"],
            "peak_gpu_build_gib": float(np.max(
                [s["peak_gpu_build_gib"] for s in seeds.values()])),
            "peak_gpu_eval_gib": float(np.max(
                [s["peak_gpu_eval_gib"] for s in seeds.values()])),
            "id_eval_seconds_per_ensemble_mean": float(np.mean(
                [s["eval_s"] for s in seeds.values()])),
            "note": "construction excludes the one-off (h, z0) cache, reported separately",
        }
        n_val = 50000
        res["ensemble_construction"]["id_val_images_per_s_M20_both_variants"] = (
            2 * n_val / float(np.mean([s["eval_s"] for s in seeds.values()])))

    cdir = out / "construction"
    if cdir.exists():
        files = sorted(cdir.glob("members_seed*.npz"))
        res["compact_ensemble_disk"] = {
            "files": len(files),
            "mib_per_seed": [f.stat().st_size / 1024**2 for f in files],
            "total_mib": sum(f.stat().st_size for f in files) / 1024**2,
            "twenty_full_vit_copies_gib": 20 * 86_567_656 * 4 / GIB,
        }
    res["cpu_peak_rss_gib"] = cpu_peak_rss_gib()
    fc.write_json(out / "timing" / "summary.json", res)
    return res


def report(res: dict) -> str:
    s = res["sustained"]
    lines = [
        "| quantity | measured |", "|---|---|",
        f"| base checkpoint load | {res['checkpoint_load_s']:.1f} s |",
        f"| model resident (fp32) | {res['model_resident_gib']:.3f} GiB |",
        f"| perturbation basis (K=20) | {res['basis_construction_s']:.2f} s, "
        f"{res['basis_mib']:.0f} MiB CPU, 0 GiB GPU |",
    ]
    if "cache_train_pools" in res and "correction" in res["cache_train_pools"]:
        c = res["cache_train_pools"]["correction"]
        lines.append(f"| (h, z0) cache, 32,768 images | {c['cache_seconds']/60:.1f} min at "
                     f"{c['images_per_s']:.0f} img/s, {c['disk_mib']:.0f} MiB |")
    if "cache_val50k" in res:
        v = res["cache_val50k"]
        lines.append(f"| ImageNet val cache, 50,000 images | {v['cache_seconds']/60:.1f} min "
                     f"at {v['images_per_s']:.0f} img/s |")
    if "cache_ood" in res:
        tot_n = sum(d["n"] for d in res["cache_ood"].values())
        tot_s = sum(d["seconds"] for d in res["cache_ood"].values())
        lines.append(f"| OOD cache, {tot_n:,} images | {tot_s/60:.1f} min at "
                     f"{tot_n/tot_s:.0f} img/s |")
    if "ensemble_construction" in res:
        e = res["ensemble_construction"]
        lines += [
            f"| per-member correction | {e['seconds_per_member_mean']:.2f} s |",
            f"| M=20 ensemble construction | {e['seconds_per_ensemble_mean']:.1f} s |",
            f"| peak GPU, construction | {e['peak_gpu_build_gib']:.3f} GiB |",
            f"| peak GPU, inference | {e['peak_gpu_eval_gib']:.3f} GiB |",
        ]
    if "compact_ensemble_disk" in res:
        d = res["compact_ensemble_disk"]
        lines.append(f"| compact ensemble on disk | {d['total_mib']/d['files']:.0f} MiB per "
                     f"seed ({d['files']} seeds) vs {d['twenty_full_vit_copies_gib']:.2f} "
                     f"GiB for 20 ViT copies |")
    lines += [
        f"| ImageNet throughput (cold / settled) | {s['first_window_images_per_s']:.0f} / "
        f"**{s['settled_images_per_s']:.0f}** img/s |",
        f"| thermal decay over {s['images']:,} images | {s['decay_pct']:.1f}% "
        f"({s['gpu_state_start'].get('sm_clock_mhz')} -> "
        f"{s['gpu_state_end'].get('sm_clock_mhz')} MHz, "
        f"{s['gpu_state_end'].get('temp_c')} °C) |",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    out = Path("results/neurips_2026_rebuttal/imagenet_vit")
    print(report(collect(out)))
