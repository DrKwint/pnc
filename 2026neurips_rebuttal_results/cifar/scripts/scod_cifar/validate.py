"""Validation gates (Section 20). Fails (exit 1) unless the stage's invariants all hold.

  --stage sketches     : checkpoints+hashes, param layout, memory preflight, sketch files present,
                         finite eigenvalues, model immutability, k_max=20 (=> k in {5,10,20} share
                         ONE sketch), base temperature recorded, no OOD used in build.
  --stage predictions  : all 7 datasets scored, SCOD scores finite & nonnegative, ID acc/NLL match
                         base classifier, macro Near/Far reproduce per-dataset aggregates, the main
                         row uses the predeclared k=10 (not the best-OOD cell), per-example retained.
"""
from __future__ import annotations

import argparse, json
from pathlib import Path
import numpy as np, yaml

REPO = Path(__file__).resolve().parents[2]
from experiments.scod_cifar.protocol import sha256_file
from experiments.scod_cifar.parameter_layout import checkpoint_path


def _cfg(p):
    with open(p) as f:
        return yaml.safe_load(f)


def _fail(msg, errs): errs.append(msg); print("  [FAIL]", msg)
def _ok(msg): print("  [ok]", msg)


def validate_sketches(cfg, root):
    errs = []
    k_max = int(cfg["num_eigs_max"]); T = int(cfg["num_samples"])
    if k_max != 20:
        _fail(f"num_eigs_max={k_max} != 20 (gate 14: k in 5,10,20 from one k_max=20 sketch)", errs)
    if T != 6 * k_max + 4:
        _fail(f"num_samples={T} != 6*k_max+4={6*k_max+4}", errs)
    base = int(cfg["sketch_seed_base_primary"])
    for seed in cfg["model_seeds"]:
        ck = checkpoint_path(seed)
        if not ck.exists():
            _fail(f"checkpoint missing seed{seed}: {ck}", errs); continue
        lay = root / "configs" / f"{seed}_parameter_layout.json"
        if not lay.exists(): _fail(f"parameter layout missing seed{seed}", errs)
        pf = root / "timing" / f"seed{seed}_memory_preflight.json"
        if not pf.exists(): _fail(f"memory preflight missing seed{seed}", errs)
        bm = root / "metrics" / f"seed{seed}_build_meta.json"
        if not bm.exists():
            _fail(f"build meta missing seed{seed}", errs); continue
        meta = json.load(open(bm))
        if not meta.get("immutable", False):
            _fail(f"seed{seed} model NOT immutable (max_abs="
                  f"{meta.get('immutability_max_abs_weight_change')})", errs)
        else:
            _ok(f"seed{seed} model immutable (weights bit-unchanged)")
        if meta.get("temperature") is None: _fail(f"seed{seed} temperature missing", errs)
        sk = root / "sketches" / f"scod1024_seed{seed}_sketchseed{base+seed}_tempered.npz"
        if not sk.exists():
            _fail(f"tempered sketch missing seed{seed}", errs); continue
        d = np.load(sk, allow_pickle=True)
        ev, U = d["eigvals"], d["basis"]
        if int(d["k"]) != k_max: _fail(f"seed{seed} sketch k={int(d['k'])} != {k_max}", errs)
        if not np.all(np.isfinite(ev)): _fail(f"seed{seed} non-finite eigenvalues", errs)
        elif not np.all(ev >= -1e-9): _fail(f"seed{seed} negative eigenvalues", errs)
        else: _ok(f"seed{seed} eigenvalues finite & nonneg (top={ev[0]:.3g}, k={len(ev)})")
        if not np.all(np.isfinite(U)): _fail(f"seed{seed} non-finite basis", errs)
    return errs


def validate_predictions(cfg, root):
    errs = []
    need = ["cifar10"] + list(cfg["near_ood"]) + list(cfg["far_ood"])
    base = int(cfg["sketch_seed_base_primary"])
    k_prim = int(cfg["reported_num_eigs"])
    for seed in cfg["model_seeds"]:
        seed_dir = root / "predictions" / f"seed_{seed}" / f"sketch_{base+seed}"
        for name in need:
            pq = seed_dir / f"{name}.parquet"
            if not pq.exists(): _fail(f"seed{seed} predictions missing: {name}", errs); continue
            import pandas as pd
            df = pd.read_parquet(pq)
            for col in ("scod_score_k5", "scod_score_k10", "scod_score_k20"):
                s = df[col].to_numpy()
                if not np.all(np.isfinite(s)): _fail(f"seed{seed} {name} {col} non-finite", errs)
                if np.any(s < -1e-6): _fail(f"seed{seed} {name} {col} negative", errs)
        mfile = root / "metrics" / f"seed{seed}_metrics.json"
        if not mfile.exists(): _fail(f"seed{seed} metrics.json missing", errs); continue
        M = json.load(open(mfile))
        # main row uses predeclared k=10
        prim_key = f"tempered_k{k_prim}_Meps{int(float(cfg['Meps']))}"
        if prim_key not in M["variants"]:
            _fail(f"seed{seed} primary variant {prim_key} absent", errs)
        else:
            _ok(f"seed{seed} primary variant present ({prim_key})")
        # ID acc/NLL == base
        bid = M.get("base_id", {})
        bmeta = json.load(open(root / "metrics" / f"seed{seed}_build_meta.json"))["base_id_metrics"]
        if abs(bid.get("id_test_acc", -1) - bmeta["id_test_acc"]) > 1e-6:
            _fail(f"seed{seed} eval ID acc != build ID acc", errs)
        else:
            _ok(f"seed{seed} ID acc/NLL == base classifier (acc={bmeta['id_test_acc']:.2f})")
        # macro reproduces per-dataset
        v = M["variants"].get(prim_key, {})
        for fam, keys in [("near", cfg["near_ood"]), ("far", cfg["far_ood"])]:
            recomputed = np.mean([v["per_dataset"][d]["auroc"] for d in keys])
            if abs(recomputed - v[fam]["auroc"]) > 1e-6:
                _fail(f"seed{seed} {fam} macro AUROC mismatch", errs)
        _ok(f"seed{seed} macro Near/Far reproduce per-dataset aggregates")
        if all(v):
            print(f"    seed{seed} {prim_key}: Near {v['near']['auroc']:.2f}/{v['near']['fpr95']:.2f}"
                  f"  Far {v['far']['auroc']:.2f}/{v['far']['fpr95']:.2f}")
    return errs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--stage", required=True, choices=["sketches", "predictions"])
    args = ap.parse_args()
    cfg = _cfg(args.config); root = REPO / cfg["root"]
    print(f"=== validate stage={args.stage} ===")
    errs = validate_sketches(cfg, root) if args.stage == "sketches" else validate_predictions(cfg, root)
    if errs:
        print(f"\nVALIDATION FAILED ({len(errs)} issue(s)).")
        raise SystemExit(1)
    print(f"\nVALIDATION PASSED (stage={args.stage}).")


if __name__ == "__main__":
    main()
