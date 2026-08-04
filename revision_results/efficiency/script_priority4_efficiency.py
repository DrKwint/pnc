#!/usr/bin/env python3
"""Priority 4 — efficiency accounting (storage + construction; CPU-only parts).

Separates construction cost, persistent storage, and inference cost, honestly distinguishing
marginal (given a pretrained base) from total. Inference micro-benchmark for MuJoCo is a
separate GPU job (priority4_inference_bench.py); CIFAR inference is read from the cached
results/cifar10/inference_cost.json.

This script computes:
  * STORAGE (exact param counts) for the MuJoCo MLP config, for every method, including the
    three P&C storage regimes (naive full-checkpoint / implemented shared-base+corrected-block /
    theoretical minimal shared-base+coeffs).
  * CONSTRUCTION time pulled from cached MuJoCo result JSON train_time fields.
  * CIFAR inference cost copied from inference_cost.json.

Outputs (results/neurips_2026_rebuttal/):
  efficiency_storage.csv, efficiency_construction_mujoco.csv, efficiency_cifar_inference.csv
CPU-only.
"""
import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))
os.chdir(_REPO)

import csv
import glob
import json

OUT = Path("results/neurips_2026_rebuttal")

# ---- MuJoCo MLP param counts (hidden [200,200,200,200], prob head = 2*out) -------
HIDDEN = [200, 200, 200, 200]
ENV_DIMS = {"Ant-v5": (113, 105), "HalfCheetah-v5": (23, 17), "Hopper-v5": (14, 11)}
M = 50               # ensemble size (submitted)
K = 20               # n_directions
N_PERT_LAYERS = 2    # multi-block [0,2]
FLOAT_BYTES = 4


def mlp_param_count(d_in, d_out, prob=True):
    dims = [d_in] + HIDDEN + [2 * d_out if prob else d_out]
    return sum(dims[i] * dims[i + 1] + dims[i + 1] for i in range(len(dims) - 1))


def corrected_block_params():
    # each corrected layer is a hidden->hidden affine: 200x200 + 200 bias
    return N_PERT_LAYERS * (200 * 200 + 200)


def storage_rows():
    rows = []
    for env, (d_in, d_out) in ENV_DIMS.items():
        base = mlp_param_count(d_in, d_out)
        corr = corrected_block_params()
        directions = N_PERT_LAYERS * K * (200 * 200)      # shared across members
        zcoeffs = M * N_PERT_LAYERS * K                   # tiny per-member latent
        variants = {
            # Deep Ensemble: M independently trained full models
            "DeepEnsemble(M=50)": M * base,
            # P&C naive: M full corrected checkpoints (if stored naively)
            "PnC_naive_full(M=50)": M * base,
            # P&C implemented: 1 base + M x corrected-block weights (ensembles.py stores seq_w/b)
            "PnC_impl_sharedbase+corrblocks(M=50)": base + M * corr,
            # P&C theoretical minimal: 1 base + shared directions + M x tiny coeffs
            "PnC_min_sharedbase+coeffs(M=50)": base + directions + zcoeffs,
            # SWAG: base + low-rank(20) + diagonal over all params
            "SWAG(rank=20)": base + 20 * base + base,
            # NB: MuJoCo Laplace uses KFAC (Kronecker-factored) covariance (laplace.py),
            # not a dense GGN — its storage is a small per-layer factor pair; omitted here
            # to avoid a misleading dense estimate. It is not the headline comparison.
        }
        for name, params in variants.items():
            rows.append({"env": env, "method": name, "params": params,
                         "megabytes_fp32": round(params * FLOAT_BYTES / 1e6, 3),
                         "base_params": base,
                         "per_member_marginal_params": (
                             base if name in ("DeepEnsemble(M=50)", "PnC_naive_full(M=50)")
                             else corr if "corrblocks" in name
                             else (N_PERT_LAYERS * K) if "coeffs" in name else "")})
    return rows


# ---- Construction time from cached MuJoCo JSONs ----------------------------------
def construction_rows():
    rows = []
    patterns = {
        "PnC(bf0.1)": "pjsvd_multi_least_squares_random_projected_residual_prob_bf0.1_k20_n50_*_seed{s}.json",
        "DeepEnsemble": "standard_ensemble_vcal_n*_h200-200-200-200_act-relu_seed{s}.json",
        "SWAG": "swag_vcal_n*_h200-200-200-200_act-relu_seed{s}.json",
        "Laplace": "laplace_*_n*_h200-200-200-200_act-relu_seed{s}.json",
        "MCDropout": "mc_dropout_vcal_n*_h200-200-200-200_act-relu_seed{s}.json",
    }
    for env in ENV_DIMS:
        for method, pat in patterns.items():
            for seed in (0, 10, 42):
                for p in sorted(glob.glob(f"results/{env}/{pat.format(s=seed)}")):
                    try:
                        d = json.loads(Path(p).read_text())
                    except Exception:
                        continue
                    first = next(iter(d.values())) if isinstance(d, dict) and d else None
                    tt = None
                    if isinstance(first, dict):
                        tt = first.get("train_time")
                        et = first.get("eval_time")
                    elif isinstance(d, dict):
                        tt = d.get("train_time"); et = d.get("eval_time")
                    if tt is not None:
                        rows.append({"env": env, "method": method, "seed": seed,
                                     "train_time_s": tt, "eval_time_s": locals().get("et"),
                                     "file": Path(p).name})
                    break  # one config per method per seed is enough for the time estimate
    return rows


def cifar_inference_rows():
    p = Path("results/cifar10/inference_cost.json")
    if not p.exists():
        return []
    d = json.loads(p.read_text())
    rows = []
    def walk(obj, prefix=""):
        if isinstance(obj, dict):
            if any(k in obj for k in ("ms_per_sample", "warm_total_s", "latency_ms")):
                rows.append({"method": prefix, **{k: obj.get(k) for k in
                             ("cold_warmup_s", "warm_total_s", "ms_per_sample",
                              "n_forward_passes", "batch_size", "latency_ms")}})
            else:
                for k, v in obj.items():
                    walk(v, k if not prefix else f"{prefix}.{k}")
    walk(d)
    return rows


def write(path, rows, cols):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main():
    s = storage_rows()
    write(OUT / "efficiency_storage.csv", s,
          ["env", "method", "params", "megabytes_fp32", "base_params", "per_member_marginal_params"])
    print("=== STORAGE (Ant-v5, MB fp32) ===")
    for r in s:
        if r["env"] == "Ant-v5":
            print(f"  {r['method']:42s} {r['megabytes_fp32']:8.3f} MB  ({r['params']:,} params)")

    c = construction_rows()
    write(OUT / "efficiency_construction_mujoco.csv", c,
          ["env", "method", "seed", "train_time_s", "eval_time_s", "file"])
    print(f"\n=== CONSTRUCTION times pulled: {len(c)} rows -> efficiency_construction_mujoco.csv ===")

    ci = cifar_inference_rows()
    write(OUT / "efficiency_cifar_inference.csv", ci,
          ["method", "cold_warmup_s", "warm_total_s", "ms_per_sample", "n_forward_passes", "batch_size", "latency_ms"])
    print(f"=== CIFAR inference rows: {len(ci)} -> efficiency_cifar_inference.csv ===")


if __name__ == "__main__":
    main()
