"""Full CIFAR-10 P&C grid: 6 blocks x 3 scales x 3 bootstrap fractions x 3 checkpoints = 162 runs.
Val-ONLY (no OOD). Checkpoint-seed major; per-candidate resumable. Common random numbers are
inherent (basis/coeffs/calib all derive from the checkpoint seed).

Usage:
  python -m experiments.pnc_grid.run_grid --mode smoke        # 8-candidate gated smoke (seed 0)
  python -m experiments.pnc_grid.run_grid --mode full [--seeds 0 1 2] [--force]
"""
from __future__ import annotations
import argparse, json, os, time, traceback, fcntl
from pathlib import Path
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ["XLA_FLAGS"] = os.environ.get("XLA_FLAGS", "") + " --xla_gpu_enable_triton_gemm=false"
import numpy as np
REPO = Path(__file__).resolve().parents[2]
import sys; os.chdir(REPO); sys.path.insert(0, str(REPO))
import subprocess

from experiments.pnc_grid import candidate

ROOT = REPO / "results" / "neurips_2026_rebuttal" / "cifar" / "pnc_full_grid"
TARGET_BLOCKS = [(1, 0), (1, 1), (2, 0), (2, 1), (3, 0), (3, 1)]
SCALES = [25.0, 50.0, 100.0]
FRACS = [0.05, 0.10, 0.20]
SMOKE = dict(blocks=[(1, 0), (3, 1)], scales=[25.0, 100.0], fracs=[0.05, 0.20])


def chunk_for(stage_idx):  # early (larger spatial) blocks need smaller chunks on 8GB
    return 64 if stage_idx >= 3 else 16


def cand_dir(seed, s, b, scale, frac):
    return ROOT / "candidates" / f"seed_{seed}" / f"s{s}b{b}_ps{scale:g}_bf{frac:g}"


class GpuLock:
    def __init__(self, path=ROOT / ".gpu.lock", poll=20):
        self.path = Path(path); self.poll = poll; self.fd = None
    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True); self.fd = open(self.path, "w")
        while True:
            try:
                fcntl.flock(self.fd, fcntl.LOCK_EX | fcntl.LOCK_NB); break
            except BlockingIOError:
                print("[lock] GPU busy, waiting...", flush=True); time.sleep(self.poll)
        return self
    def __exit__(self, *a):
        try: fcntl.flock(self.fd, fcntl.LOCK_UN); self.fd.close()
        except Exception: pass


def git_commit():
    try: return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    except Exception: return "unknown"


def run_candidate(seed, s, b, scale, frac, cache, force):
    d = cand_dir(seed, s, b, scale, frac); d.mkdir(parents=True, exist_ok=True)
    if (d / "COMPLETE").exists() and not force:
        return "cached"
    chunk = chunk_for(s)
    cfg = dict(seed=seed, stage_idx=s, block_idx=b, label=f"s{s}b{b}", scale=scale,
               bootstrap_frac=frac, K=20, M=50, lambda_reg=1e-3, subset_size=1024, chunk_size=chunk,
               basis_seed=seed, coeff_seed=seed + 17, bootstrap_seed=seed, git_commit=git_commit())
    json.dump(cfg, open(d / "config.json", "w"), indent=2)
    try:
        m, mix = candidate.build_and_eval(seed, s, b, scale, frac, chunk, cache=cache)
        np.savez(d / "val_predictions.npz", **mix)
        # gate: metrics reproduce from saved predictions
        y = mix["y_va"].astype(int); p = mix["mix_cal"]; eps = 1e-12
        nll_rep = float(-np.log(p[np.arange(len(y)), y] + eps).mean())
        m["repro_nll_from_predictions"] = nll_rep
        m["repro_ok"] = bool(abs(nll_rep - m["val_nll_calibrated"]) < 1e-6)
        m["status"] = "ok"
        json.dump(m, open(d / "metrics.json", "w"), indent=2)
        json.dump(dict(build_secs=m["build_secs"], total_secs=m["total_secs"], chunk_size=chunk),
                  open(d / "timing.json", "w"), indent=2)
        if m["repro_ok"]:
            (d / "COMPLETE").write_text("ok\n")
        return f"ok nll={m['val_nll_calibrated']:.4f} ({m['total_secs']:.0f}s)"
    except Exception as e:
        tb = traceback.format_exc()
        oom = "RESOURCE_EXHAUSTED" in tb or "Out of memory" in tb
        json.dump(dict(**cfg, status="oom" if oom else "error", error=str(e)[:300]),
                  open(d / "metrics.json", "w"), indent=2)
        print(f"[grid] FAIL s{s}b{b}/ps{scale:g}/bf{frac:g} seed{seed}: {'oom' if oom else 'error'}: "
              f"{str(e)[:120]}", flush=True)
        return "fail"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["smoke", "full"], required=True)
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    ROOT.mkdir(parents=True, exist_ok=True)
    if args.mode == "smoke":
        seeds = [0]; blocks = SMOKE["blocks"]; scales = SMOKE["scales"]; fracs = SMOKE["fracs"]
    else:
        seeds = args.seeds; blocks = TARGET_BLOCKS; scales = SCALES; fracs = FRACS
    n_total = len(seeds) * len(blocks) * len(scales) * len(fracs)
    print(f"[grid] mode={args.mode} seeds={seeds} -> {n_total} candidates", flush=True)
    done = 0; t0 = time.time()
    with GpuLock():
        for seed in seeds:
            cache = candidate.load_seed(seed)
            for (s, b) in blocks:
                for scale in scales:
                    for frac in fracs:
                        r = run_candidate(seed, s, b, scale, frac, cache, args.force)
                        done += 1
                        print(f"[grid] ({done}/{n_total}) seed{seed} s{s}b{b} ps{scale:g} bf{frac:g}: "
                              f"{r} [elapsed {time.time()-t0:.0f}s]", flush=True)
    print(f"[grid] DONE {done}/{n_total} ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
