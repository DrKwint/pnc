"""Stage 1 -- build the SCOD Fisher sketch for each submitted checkpoint (SCOD-1024).

For every model seed: write the parameter layout, run a memory preflight, fit the base-classifier
temperature on ID-val, build the rank-k_max Fisher sketch (tempered = primary, plus an untempered
diagnostic), verify the model weights are bit-unchanged (Section 6.3 / gate 7-8), and save the
sketch + spectrum + per-seed metadata. Resumable: an existing sketch is skipped unless --force.

Usage:
  python -m experiments.scod_cifar.run_build_sketch --config configs/scod_cifar.yaml [--force]
         [--sketch-seed-bases 100000]        # extra bases for sketch-seed sensitivity
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import yaml

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
# cuBLAS instead of Triton for the tall-skinny (P x Tb) GEMMs -- Triton fails to autotune them.
os.environ["XLA_FLAGS"] = os.environ.get("XLA_FLAGS", "") + " --xla_gpu_enable_triton_gemm=false"

REPO = Path(__file__).resolve().parents[2]
import sys
os.chdir(REPO); sys.path.insert(0, str(REPO))

from experiments.scod_cifar.parameter_layout import (
    write_parameter_layout, make_ztilde_fn, split_params)
from experiments.scod_cifar import protocol
from experiments.scod_cifar.sketch import build_fisher_sketch, save_sketch


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def mem_available_gb() -> float:
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / (1024 ** 2)
    return float("nan")


def gpu_total_mb() -> float:
    try:
        import subprocess
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"])
        return float(out.decode().splitlines()[0])
    except Exception:
        return float("nan")


def memory_preflight(P: int, T: int, k: int, C: int, block_cols: int) -> dict:
    """Section 8: print and return the memory estimate before allocating anything."""
    f32 = 4
    sketch_bytes = P * T * f32                      # Y_host (host)
    basis_bytes = P * k * f32                       # U (host)
    omega_block = P * block_cols * f32              # Omega_b (GPU)
    yb_block = P * block_cols * f32                 # Y_b (GPU)
    jac_bytes = C * P * f32                         # L_i materialised (GPU, transient)
    peak_gpu = omega_block + yb_block + jac_bytes + yb_block  # + one temp of Y_b size
    peak_cpu = sketch_bytes + basis_bytes + 0.7e9   # + CIFAR-10 + model + runtime slack
    pf = dict(
        P=P, k_max=k, T=T, C=C, block_cols=block_cols,
        sketch_bytes=sketch_bytes, basis_bytes=basis_bytes,
        est_peak_gpu_bytes=peak_gpu, est_peak_cpu_bytes=peak_cpu,
        available_cpu_gb=round(mem_available_gb(), 2),
        gpu_total_mb=gpu_total_mb(),
        est_peak_gpu_gb=round(peak_gpu / 1e9, 2), est_peak_cpu_gb=round(peak_cpu / 1e9, 2),
        est_sketch_gb=round(sketch_bytes / 1e9, 2),
    )
    print("[preflight] " + json.dumps({k_: pf[k_] for k_ in
          ["P", "k_max", "T", "est_sketch_gb", "est_peak_cpu_gb", "est_peak_gpu_gb",
           "available_cpu_gb", "gpu_total_mb"]}), flush=True)
    return pf


def build_one_sketch(S, seed, temperature, sketch_seed, T, k, block_cols, xsub, tag):
    ztilde = make_ztilde_fn(S["graphdef"], S["rest"], S["unravel"], temperature)
    print(f"[build] seed{seed} {tag} sketch_seed={sketch_seed} T_temp={temperature:.4f} "
          f"N={len(xsub)} T={T} k={k}", flush=True)
    res = build_fisher_sketch(ztilde, S["flat_w"], xsub, C=10, T=T, k=k,
                              sketch_seed=sketch_seed, block_cols=block_cols, log_every=256)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--sketch-seed-bases", type=int, nargs="*", default=None,
                    help="override sketch-seed bases (default: primary base only)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    root = REPO / cfg["root"]
    (root / "sketches").mkdir(parents=True, exist_ok=True)
    (root / "spectra").mkdir(parents=True, exist_ok=True)
    (root / "configs").mkdir(parents=True, exist_ok=True)
    (root / "metrics").mkdir(parents=True, exist_ok=True)
    (root / "timing").mkdir(parents=True, exist_ok=True)

    T = int(cfg["num_samples"]); k = int(cfg["num_eigs_max"]); block_cols = int(cfg["block_cols"])
    bases = args.sketch_seed_bases if args.sketch_seed_bases is not None else [int(cfg["sketch_seed_base_primary"])]
    primary_base = int(cfg["sketch_seed_base_primary"])

    build_log = {}
    for seed in cfg["model_seeds"]:
        t0 = time.time()
        # (7) parameter layout
        li = write_parameter_layout(seed, root / "configs")
        P = li["total_params"]
        # (8) memory preflight
        pf = memory_preflight(P, T, k, 10, block_cols)
        json.dump(pf, open(root / "timing" / f"seed{seed}_memory_preflight.json", "w"), indent=2)

        # temperature + model
        S = protocol.prepare_seed(seed)
        flat_snapshot = np.array(S["flat_w"])   # immutability snapshot

        xsub, ysub, idx = protocol.get_calibration_subset(seed)
        seed_meta = dict(seed=seed, P=P, temperature=S["temperature"], temp_info=S["temp_info"],
                         calib_subset_size=int(len(xsub)), calib_idx_sha256=protocol.sha256_array(idx),
                         calib_idx_first10=idx[:10].tolist())

        for base in bases:
            sketch_seed = base + seed
            is_primary = (base == primary_base)
            # tempered (primary) sketch
            out = root / "sketches" / f"scod1024_seed{seed}_sketchseed{sketch_seed}_tempered.npz"
            if out.exists() and not args.force:
                print(f"[build] {out.name} exists, skip", flush=True)
            else:
                res = build_one_sketch(S, seed, S["temperature"], sketch_seed, T, k, block_cols, xsub,
                                       f"tempered/base{base}")
                save_sketch(res, out)
                json.dump(dict(kind="tempered", seed=seed, sketch_seed=sketch_seed,
                               temperature=S["temperature"], eigvals=res.eigvals.tolist(),
                               diagnostics=res.diagnostics),
                          open(root / "spectra" / f"{out.stem}.json", "w"), indent=2)

            # untempered diagnostic sketch (T=1) -- primary base only
            if is_primary:
                outu = root / "sketches" / f"scod1024_seed{seed}_sketchseed{sketch_seed}_untempered.npz"
                if outu.exists() and not args.force:
                    print(f"[build] {outu.name} exists, skip", flush=True)
                else:
                    resu = build_one_sketch(S, seed, 1.0, sketch_seed, T, k, block_cols, xsub,
                                            "untempered")
                    save_sketch(resu, outu)
                    json.dump(dict(kind="untempered", seed=seed, sketch_seed=sketch_seed,
                                   temperature=1.0, eigvals=resu.eigvals.tolist(),
                                   diagnostics=resu.diagnostics),
                              open(root / "spectra" / f"{outu.stem}.json", "w"), indent=2)

        # (6.3) model immutability: weights bit-unchanged after all sketching
        _, _, _, flat_after, _, _ = split_params(S["model"])
        max_abs = float(np.max(np.abs(np.array(flat_after) - flat_snapshot)))
        seed_meta["immutability_max_abs_weight_change"] = max_abs
        seed_meta["immutable"] = bool(max_abs == 0.0)

        # base ID metrics (bookkeeping, Section 14)
        idm = protocol.base_id_metrics(seed, S["flat_w"], S["graphdef"], S["rest"], S["unravel"],
                                       S["temperature"])
        seed_meta["base_id_metrics"] = idm
        seed_meta["build_seconds"] = round(time.time() - t0, 1)
        json.dump(seed_meta, open(root / "metrics" / f"seed{seed}_build_meta.json", "w"), indent=2)
        build_log[str(seed)] = dict(P=P, temperature=S["temperature"], immutable=seed_meta["immutable"],
                                    id_test_acc=idm["id_test_acc"], build_seconds=seed_meta["build_seconds"])
        print(f"[build] seed{seed} DONE acc={idm['id_test_acc']:.2f} T={S['temperature']:.4f} "
              f"immutable={seed_meta['immutable']} ({seed_meta['build_seconds']:.0f}s)", flush=True)

    json.dump(build_log, open(root / "metrics" / "build_summary.json", "w"), indent=2)
    write_manifest(cfg, root, build_log)
    print("[build] ALL SEEDS DONE:", json.dumps(build_log), flush=True)


def write_manifest(cfg, root, build_log):
    """Section 2 -- freeze the submitted protocol provenance (JAX/Flax adaptation)."""
    import subprocess, jax, flax
    from experiments.scod_cifar.parameter_layout import checkpoint_path
    x_tr, y_tr, x_va, y_va = protocol.get_splits()
    ckpt_paths, ckpt_hashes, temps, calib_hashes = {}, {}, {}, {}
    for seed in cfg["model_seeds"]:
        cp = checkpoint_path(seed); ckpt_paths[str(seed)] = str(cp)
        ckpt_hashes[str(seed)] = protocol.sha256_file(cp)
        bm = json.load(open(root / "metrics" / f"seed{seed}_build_meta.json"))
        temps[str(seed)] = bm["temperature"]
        calib_hashes[str(seed)] = bm["calib_idx_sha256"]
    try:
        git = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    except Exception:
        git = "unknown"
    manifest = dict(
        framework_note="Spec is PyTorch; this repo/impl is JAX/Flax nnx (faithful analogue).",
        method="SCOD", variant="SCOD-1024",
        checkpoint_paths=ckpt_paths, checkpoint_hashes=ckpt_hashes,
        model_seeds=list(cfg["model_seeds"]), model_architecture="PreActResNet18 (CIFAR-10, 10 classes)",
        train_split_hash=protocol.sha256_array(x_tr), id_validation_split_hash=protocol.sha256_array(x_va),
        id_test_split_hash=protocol.sha256_array(protocol.get_id_test()[0]),
        pnc_calibration_subset_hashes=calib_hashes,
        temperature_by_seed=temps, temperature_fit_split="id_validation_only (base classifier)",
        transform_description="deterministic eval, no crop/flip/cutout",
        normalization=dict(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616],
                           formula="(x/255 - mean)/std"),
        near_ood_datasets=["CIFAR-100", "TinyImageNet"],
        far_ood_datasets=["MNIST", "SVHN", "Textures", "Places365"],
        metric_code_paths=dict(evaluator="openood_eval.py:_binary_ood_metrics",
                               orientation="higher score = more OOD; ID=neg class"),
        aggregation_convention="macro-mean over datasets; +/- = sample std (ddof=1) over 3 seeds",
        git_commit=git, jax_version=jax.__version__, flax_version=flax.__version__,
        cuda_note="JAX GPU (CUDA); torch_version/cuda_version N/A (JAX impl)",
        build_summary=build_log,
        scod_hyperparameters=dict(num_eigs_max=cfg["num_eigs_max"], num_samples=cfg["num_samples"],
                                  reported_num_eigs=cfg["reported_num_eigs"], Meps=cfg["Meps"],
                                  sketch_type=cfg["sketch_type"], weighted=cfg["weighted"],
                                  projection_type=cfg["projection_type"]),
    )
    json.dump(manifest, open(root / "MANIFEST.json", "w"), indent=2)
    print(f"[build] wrote MANIFEST.json", flush=True)


if __name__ == "__main__":
    main()
