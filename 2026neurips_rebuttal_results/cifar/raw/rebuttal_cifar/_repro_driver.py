#!/usr/bin/env python3
"""Phase 1 reproduction driver (NeurIPS 2026 rebuttal, CIFAR).

Recomputes a single-block PnC OpenOOD eval FROM SCRATCH (no Luigi cache) for one
candidate config + one seed, writing to a FRESH path under
results/neurips_2026_rebuttal/cifar/ so no existing checkpoint or result file is
overwritten. Then diffs every headline metric against the cached submitted JSON.

Usage:
  python _repro_driver.py A   # Candidate A: s3b0 / ps25 / bf0.05
  python _repro_driver.py B   # Candidate B: s3b1 / ps50 / no bootstrap
"""
from __future__ import annotations
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import sys, json, glob, time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import luigi

# repo root is two levels up from this file (results/neurips_2026_rebuttal/cifar/)
REPO = Path(__file__).resolve().parents[3]
os.chdir(REPO)
sys.path.insert(0, str(REPO))

from cifar_tasks import CIFAROpenOODPnC  # noqa: E402

OUT_DIR = REPO / "results" / "neurips_2026_rebuttal" / "cifar"

CANDIDATES = {
    "A": dict(
        label="A_s3b0_ps25_bf0.05",
        params=dict(dataset="cifar10", epochs=300, n_directions=20, n_perturbations=50,
                    perturbation_sizes=[25.0], subset_size=1024, chunk_size=1024,
                    target_stage_idx=3, target_block_idx=0, random_directions=True,
                    seed=0, lambda_reg=1e-3, posthoc_calibrate=True, bootstrap_frac=0.05),
        cached_glob="results/cifar10/openood_v1p5_pnc_single_block_vcal_s3b0_k20_n50_ps25.0_lr0.001_bf0.05_*_seed0_random.json",
        cached_key="25.0",
    ),
    "B": dict(
        label="B_s3b1_ps50_nobf",
        params=dict(dataset="cifar10", epochs=300, n_directions=20, n_perturbations=50,
                    perturbation_sizes=[50.0], subset_size=1024, chunk_size=1024,
                    target_stage_idx=3, target_block_idx=1, random_directions=True,
                    seed=0, lambda_reg=1e-3, posthoc_calibrate=True, bootstrap_frac=0.0),
        cached_glob="results/cifar10/openood_v1p5_pnc_single_block_vcal_s3b1_k20_n50_ps50.0_lr0.001_e300_*_seed0_random.json",
        cached_key="50.0",
    ),
}


def headline(entry: dict) -> dict:
    idm = entry["id_metrics"]
    def fpr95(kind):
        return entry[kind]["aggregate"]["predictive_entropy"]["mean_fpr95"]
    return {
        "acc": idm["accuracy"] * 100,
        "nll": idm["nll"],
        "ece": idm["ece"],
        "posthoc_temperature": idm.get("posthoc_temperature"),
        "near_auroc": entry["near_ood_auroc"] * 100,
        "far_auroc": entry["far_ood_auroc"] * 100,
        "near_fpr95": fpr95("near_ood") * 100,
        "far_fpr95": fpr95("far_ood") * 100,
    }


def main():
    which = sys.argv[1].upper() if len(sys.argv) > 1 else "A"
    c = CANDIDATES[which]
    out_json = OUT_DIR / f"repro_{c['label']}_seed0.json"

    task = CIFAROpenOODPnC(**c["params"])
    # redirect output so we never touch the cached submitted file
    task.output = lambda p=str(out_json): luigi.LocalTarget(p)  # type: ignore

    ckpt = task.input().path
    assert Path(ckpt).exists(), f"base checkpoint missing: {ckpt}"
    print(f"[repro {which}] checkpoint = {ckpt}", flush=True)
    print(f"[repro {which}] params = {c['params']}", flush=True)

    t0 = time.time()
    start_iso = datetime.now(timezone.utc).isoformat()
    task.run()   # recomputes from scratch; writes out_json
    runtime = time.time() - t0
    end_iso = datetime.now(timezone.utc).isoformat()

    repro = json.load(open(out_json))[c["cached_key"]]
    cached_path = sorted(glob.glob(c["cached_glob"]))[0]
    cached = json.load(open(cached_path))[c["cached_key"]]

    hr, hc = headline(repro), headline(cached)
    diff = {k: (hr[k] - hc[k]) if (hr[k] is not None and hc[k] is not None) else None
            for k in hr}

    summary = {
        "candidate": which,
        "label": c["label"],
        "seed": 0,
        "params": c["params"],
        "checkpoint": ckpt,
        "cached_reference": cached_path,
        "repro_output": str(out_json),
        "start_utc": start_iso, "end_utc": end_iso, "runtime_sec": runtime,
        "hardware": "NVIDIA RTX 5060 8GB, WSL2; JAX 0.9.1 Flax 0.12.3 py3.12.3",
        "reproduced": hr, "cached": hc, "abs_diff": diff,
    }
    cmp_json = OUT_DIR / f"reproduction_cifar_seed0_{which}.json"
    json.dump(summary, open(cmp_json, "w"), indent=2)

    print(f"\n===== REPRO {which} vs CACHED ({c['label']}) =====", flush=True)
    hdr = f"{'metric':16} {'reproduced':>12} {'cached':>12} {'abs_diff':>12}"
    print(hdr); print("-" * len(hdr))
    for k in hr:
        r = hr[k]; ca = hc[k]; d = diff[k]
        rs = f"{r:.4f}" if r is not None else "--"
        cs = f"{ca:.4f}" if ca is not None else "--"
        ds = f"{d:+.4f}" if d is not None else "--"
        print(f"{k:16} {rs:>12} {cs:>12} {ds:>12}")
    print(f"\nruntime: {runtime:.1f}s -> wrote {cmp_json}", flush=True)


if __name__ == "__main__":
    main()
