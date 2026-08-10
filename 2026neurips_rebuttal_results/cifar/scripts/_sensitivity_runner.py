#!/usr/bin/env python3
"""Phase 3 -- one-factor sensitivity sweeps around the frozen anchor.

Anchor: single-block PnC, s3b0 (stage_idx=3,block_idx=0), K=20, M=50, ps=25,
lambda=1e-3, bootstrap_frac=0.05, subset=1024, random dirs, posthoc temp on ID-val.
Each sweep varies EXACTLY ONE factor; all else fixed at the anchor.

Runs the real CIFAROpenOODPnC task (recomputes from scratch), redirecting output to
results/neurips_2026_rebuttal/cifar/sweeps/ so no existing result file is touched.
Resumable: a cell already present in sensitivity_cifar_raw.csv is skipped. Failures
(OOM / singular solve / NaN) are recorded with status!=ok instead of aborting the sweep.

Usage:
  python _sensitivity_runner.py --seed 0 --factors scale,rank,calib,ridge,block
  python _sensitivity_runner.py --seed 1 --factors scale,rank
"""
from __future__ import annotations
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import sys, json, time, csv, argparse, traceback, gc
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import luigi

REPO = Path(__file__).resolve().parents[3]
os.chdir(REPO); sys.path.insert(0, str(REPO))
from cifar_tasks import CIFAROpenOODPnC  # noqa

OUT = REPO / "results" / "neurips_2026_rebuttal" / "cifar"
SWEEP_DIR = OUT / "sweeps"; SWEEP_DIR.mkdir(exist_ok=True)
CSV = OUT / "sensitivity_cifar_raw.csv"
# DONE_CSV: read-only, consulted for skip/reuse detection. OUT_CSV: where new rows are appended.
# Defaults keep single-process behaviour (both == the main raw CSV); parallel seeds override OUT_CSV
# with a per-seed shard while still reading the shared main CSV for cached cells.
DONE_CSV = CSV
OUT_CSV = CSV

# anchor defaults (every task field not exposed per-cell)
ANCHOR = dict(dataset="cifar10", epochs=300, n_directions=20, n_perturbations=50,
              perturbation_sizes=[25.0], subset_size=1024, chunk_size=1024,
              target_stage_idx=3, target_block_idx=0, random_directions=True,
              lambda_reg=1e-3, posthoc_calibrate=True, bootstrap_frac=0.05)


def cells_for(factor):
    """Return list of (cell_key, overrides) for a factor. cell_key is stable across seeds."""
    out = []
    if factor == "scale":
        for ps in [6.25, 12.5, 25.0, 50.0, 100.0]:
            out.append((f"scale_ps{ps}", dict(perturbation_sizes=[ps])))
    elif factor == "scalex":
        # perturbation-scale multiplier grid {0.5,0.75,1,1.25,1.5}x anchor ps=25.
        # 12.5 and 25.0 reuse the existing `scale_ps*` cells (same key -> cached).
        for ps in [12.5, 18.75, 25.0, 31.25, 37.5]:
            out.append((f"scale_ps{ps}", dict(perturbation_sizes=[ps])))
    elif factor == "boot":
        # bootstrap-fraction grid. bf=0.05 is the anchor (== scale_ps25.0); we only
        # run the 4 missing fractions and reuse the anchor cell for 0.05 at report time.
        for bf in [0.01, 0.03, 0.07, 0.1]:
            out.append((f"boot_bf{bf:g}", dict(bootstrap_frac=bf)))
    elif factor == "rank":
        for k in [1, 2, 5, 20, 40]:
            out.append((f"rank_k{k}", dict(n_directions=k)))
    elif factor == "calib":
        for ss in [256, 512, 1024, 2048, 4096]:
            out.append((f"calib_ss{ss}", dict(subset_size=ss)))
    elif factor == "ridge":
        for lam in [0.0, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]:
            out.append((f"ridge_lam{lam:g}", dict(lambda_reg=lam)))
    elif factor == "block":
        # early / middle / late(anchor). Stage/block are 0-indexed; stage3=stage4(deepest).
        # early stages have large spatial maps -> smaller chunk for 8GB.
        out.append(("block_s1b0", dict(target_stage_idx=1, target_block_idx=0, chunk_size=16)))
        out.append(("block_s2b1", dict(target_stage_idx=2, target_block_idx=1, chunk_size=16)))
        out.append(("block_s3b0", dict(target_stage_idx=3, target_block_idx=0)))  # anchor
    else:
        raise ValueError(factor)
    return out


def headline_from_json(path, key):
    d = json.load(open(path))
    e = d[key]
    idm = e["id_metrics"]
    def fpr(k): return e[k]["aggregate"]["predictive_entropy"]["mean_fpr95"] * 100
    def vnll():
        v = e.get("val_metrics") or {}
        return v.get("nll")
    return dict(
        id_acc=idm["accuracy"] * 100, id_nll=idm["nll"], id_ece=idm["ece"],
        val_nll=vnll(), posthoc_temperature=idm.get("posthoc_temperature"),
        near_auroc=e["near_ood_auroc"] * 100, far_auroc=e["far_ood_auroc"] * 100,
        near_fpr95=fpr("near_ood"), far_fpr95=fpr("far_ood"),
    )


def load_done():
    done = set()
    for path in {DONE_CSV, OUT_CSV}:  # union of shared-main and this shard
        if not Path(path).exists():
            continue
        with open(path) as f:
            for r in csv.DictReader(f):
                done.add((r["cell_key"], int(r["seed"])))
    return done


def append_row(row):
    fields = ["factor", "cell_key", "seed", "status", "id_acc", "id_nll", "id_ece", "val_nll",
              "posthoc_temperature", "near_auroc", "near_fpr95", "far_auroc", "far_fpr95",
              "ps", "K", "M", "subset_size", "chunk_size", "lambda_reg", "bootstrap_frac",
              "stage_idx", "block_idx", "runtime_sec", "start_utc", "end_utc", "output_path", "note"]
    new = not Path(OUT_CSV).exists()
    with open(OUT_CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if new: w.writeheader()
        w.writerow({k: row.get(k, "") for k in fields})


def run_cell(factor, cell_key, overrides, seed):
    params = dict(ANCHOR); params.update(overrides); params["seed"] = seed
    ps = params["perturbation_sizes"]
    ps_tag = "-".join(f"{v:g}" for v in ps)
    out_path = SWEEP_DIR / (f"sweep_{cell_key}_seed{seed}.json")
    task = CIFAROpenOODPnC(**params)
    task.output = lambda p=str(out_path): luigi.LocalTarget(p)  # type: ignore
    ckpt = task.input().path
    row = dict(factor=factor, cell_key=cell_key, seed=seed,
               ps=ps_tag, K=params["n_directions"], M=params["n_perturbations"],
               subset_size=params["subset_size"], chunk_size=params["chunk_size"],
               lambda_reg=params["lambda_reg"], bootstrap_frac=params["bootstrap_frac"],
               stage_idx=params["target_stage_idx"], block_idx=params["target_block_idx"],
               output_path=str(out_path))
    if not Path(ckpt).exists():
        row.update(status="missing_checkpoint", note=ckpt); append_row(row); return
    t0 = time.time(); row["start_utc"] = datetime.now(timezone.utc).isoformat()
    try:
        task.run()
        m = headline_from_json(out_path, str(ps[0]) if len(ps) == 1 else list(json.load(open(out_path)).keys())[0])
        # NaN check
        bad = any(v is not None and isinstance(v, float) and (np.isnan(v) or np.isinf(v))
                  for v in m.values())
        row.update(m); row["status"] = "nan_or_inf" if bad else "ok"
    except Exception as e:
        tb = traceback.format_exc()
        oom = "RESOURCE_EXHAUSTED" in tb or "Out of memory" in tb or "RESOURCE_EXHAUSTED" in str(e)
        row["status"] = "oom" if oom else "error"
        row["note"] = (str(e)[:300]).replace("\n", " ")
        print(f"[sweep] FAIL {cell_key} seed{seed}: {row['status']}: {row['note'][:120]}", flush=True)
    row["end_utc"] = datetime.now(timezone.utc).isoformat()
    row["runtime_sec"] = round(time.time() - t0, 1)
    append_row(row)
    s = row.get("status")
    print(f"[sweep] {cell_key} seed{seed} [{s}] "
          + (f"idacc={row.get('id_acc'):.2f} nearAUROC={row.get('near_auroc'):.2f} "
             f"nearFPR95={row.get('near_fpr95'):.1f} ({row['runtime_sec']:.0f}s)" if s == "ok" else f"({row['runtime_sec']:.0f}s)"),
          flush=True)
    gc.collect()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--factors", default="scale,rank,calib,ridge,block")
    ap.add_argument("--out-csv", default=None, help="append new rows here (default: main raw CSV)")
    args = ap.parse_args()
    global OUT_CSV
    if args.out_csv:
        OUT_CSV = Path(args.out_csv)
    done = load_done()
    factors = [f.strip() for f in args.factors.split(",") if f.strip()]
    print(f"[sweep] seed={args.seed} factors={factors}; {len(done)} cells already done", flush=True)
    for factor in factors:
        for cell_key, overrides in cells_for(factor):
            if (cell_key, args.seed) in done:
                print(f"[sweep] skip {cell_key} seed{args.seed} (cached)", flush=True); continue
            run_cell(factor, cell_key, overrides, args.seed)


if __name__ == "__main__":
    main()
