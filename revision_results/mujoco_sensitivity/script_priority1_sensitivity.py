#!/usr/bin/env python3
"""Priority 1 — MuJoCo hyperparameter robustness (one-factor-at-a-time).

Anchor = the reproduced canonical P&C config (Phase 0):
  env=HalfCheetah-v5, layer_scope=multi, correction=least_squares,
  safe_subspace_backend=projected_residual, pjsvd_family=random, prob base,
  n_directions(k)=20, n_perturbations(M)=50, subset_size=10000 (full),
  lambda_reg=0.0, bootstrap_frac=0.1, hidden=[200,200,200,200], relu,
  perturbation size grid {5,10,20,50} with the OPERATIVE size selected per (env,seed)
  by best validation NLL (the submitted ID-only protocol).

Sweeps (each varies ONE factor, all else at anchor):
  scale     : perturbation size (report ALL grid points; wide relative grid)
  rank      : n_directions k in {1,2,5,20*,40}
  bootstrap : bootstrap_frac in {0.0, 0.25, 0.5, 0.75, 0.99}  (+anchor 0.1*)
  ridge     : lambda_reg in {0.0*, 1e-4, 1e-2, 1.0, 100.0}
  (layer sweep handled separately: priority1_layer.py)

For non-scale sweeps we report the val-NLL-selected size row (scale held at its default).
'*' marks the anchor/default value.

Outputs (under results/neurips_2026_rebuttal/priority1/):
  - sensitivity_mujoco.csv           (append; machine-readable)
  - per-run redirected P&C JSONs under priority1/repro/
Run:  .venv/bin/python scripts/neurips_2026_rebuttal/priority1_sensitivity.py \
        --env HalfCheetah-v5 --seeds 0 --sweeps scale rank bootstrap ridge
"""
import os
import sys
os.environ.setdefault("JAX_PLATFORMS", "cuda,cpu")
import argparse
import csv
import json
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))
os.chdir(_REPO_ROOT)

import luigi
from pnc_core.gym_tasks import GymPJSVD

OUT_DIR = Path("results/neurips_2026_rebuttal/priority1")
REPRO_DIR = OUT_DIR / "repro"
CSV_PATH = OUT_DIR / "sensitivity_mujoco.csv"

# ---- Anchor (submitted/canonical default) --------------------------------------
ANCHOR = dict(
    steps=10000,
    subset_size=10000,
    n_directions=20,
    n_perturbations=50,
    perturbation_sizes=[5.0, 10.0, 20.0, 50.0],
    layer_scope="multi",
    pjsvd_family="random",
    correction_mode="least_squares",
    safe_subspace_backend="projected_residual",
    probabilistic_base_model=True,
    hidden_dims=[200, 200, 200, 200],
    activation="relu",
    lambda_reg=0.0,
    bootstrap_frac=0.1,
    policy_preset="neurips_minari",
    compute_geometry=False,
    compute_l2=False,
)

METRIC_FIELDS = [
    "rmse_val", "nll_val", "rmse_id", "nll_id",
    "nll_ood_near", "nll_ood_mid", "nll_ood_far",
    "auroc_ood_near", "auroc_ood_mid", "auroc_ood_far",
    "train_time", "eval_time",
]
CSV_COLS = [
    "sweep", "value", "is_default", "env", "seed",
    "family", "k", "M", "subset", "lambda_reg", "bootstrap_frac", "layer_scope",
    "perturbation_size", "size_selected_by_valnll",
] + METRIC_FIELDS


class _RedirectPJSVD(GymPJSVD):
    def output(self) -> luigi.LocalTarget:
        orig = Path(super().output().path)
        name = orig.name
        if len(self.perturb_layers) > 0:
            tok = "PL" + "-".join(str(int(i)) for i in self.perturb_layers)
            name = name.replace(".json", f"_{tok}.json")
        return luigi.LocalTarget(str(REPRO_DIR / orig.parent.name / name))


def run_config(env: str, seed: int, overrides: dict) -> dict:
    cfg = dict(ANCHOR)
    cfg.update(overrides)
    task = _RedirectPJSVD(env=env, seed=seed, **cfg)
    out = Path(task.output().path)
    out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    ok = luigi.build([task], local_scheduler=True, workers=1, log_level="WARNING")
    wall = time.time() - t0
    if not ok or not out.exists():
        raise SystemExit(f"[priority1] FAILED: {overrides} seed={seed} -> {out}")
    metrics = json.loads(out.read_text())
    return {"metrics": metrics, "wall": wall}


def select_size(metrics: dict) -> str:
    return min(metrics.keys(), key=lambda k: metrics[k]["nll_val"])


def emit_rows(sweep: str, value, is_default: bool, env: str, seed: int,
              overrides: dict, metrics: dict, all_sizes: bool) -> list[dict]:
    cfg = dict(ANCHOR); cfg.update(overrides)
    sel = select_size(metrics)
    sizes = sorted(metrics.keys(), key=float) if all_sizes else [sel]
    rows = []
    for ps in sizes:
        m = metrics[ps]
        row = {
            "sweep": sweep, "value": value, "is_default": is_default,
            "env": env, "seed": seed,
            "family": cfg["pjsvd_family"], "k": cfg["n_directions"], "M": cfg["n_perturbations"],
            "subset": cfg["subset_size"], "lambda_reg": cfg["lambda_reg"],
            "bootstrap_frac": cfg["bootstrap_frac"], "layer_scope": cfg["layer_scope"],
            "perturbation_size": float(ps), "size_selected_by_valnll": (ps == sel),
        }
        for f in METRIC_FIELDS:
            row[f] = m.get(f)
        rows.append(row)
    return rows


# ---- Sweep definitions ---------------------------------------------------------
def sweep_scale(env, seed):
    # Step 1: default operative size = val-NLL argmin over the standard grid {5,10,20,50}.
    std_grid = [5.0, 10.0, 20.0, 50.0]
    r0 = run_config(env, seed, {"perturbation_sizes": std_grid})
    default_size = float(min(r0["metrics"].keys(), key=lambda k: r0["metrics"][k]["nll_val"]))
    # Step 2: relative grid 0.25/0.5/1/2/4 x default (plan 1A).
    rel = [0.25, 0.5, 1.0, 2.0, 4.0]
    grid = [round(default_size * m, 4) for m in rel]
    r = run_config(env, seed, {"perturbation_sizes": grid})
    cfg = dict(ANCHOR)
    rows = []
    for mult, ps in zip(rel, grid):
        key = next((k for k in r["metrics"] if abs(float(k) - ps) < 1e-6), None)
        if key is None:
            continue
        m = r["metrics"][key]
        row = {
            "sweep": "scale", "value": f"{mult}x(={ps})", "is_default": (mult == 1.0),
            "env": env, "seed": seed, "family": cfg["pjsvd_family"], "k": cfg["n_directions"],
            "M": cfg["n_perturbations"], "subset": cfg["subset_size"], "lambda_reg": cfg["lambda_reg"],
            "bootstrap_frac": cfg["bootstrap_frac"], "layer_scope": cfg["layer_scope"],
            "perturbation_size": ps, "size_selected_by_valnll": (mult == 1.0),
        }
        for f in METRIC_FIELDS:
            row[f] = m.get(f)
        rows.append(row)
    return rows


def sweep_rank(env, seed):
    rows = []
    for k in [1, 2, 5, 20, 40]:
        r = run_config(env, seed, {"n_directions": k})
        rows += emit_rows("rank", k, k == 20, env, seed, {"n_directions": k}, r["metrics"], all_sizes=False)
    return rows


def sweep_bootstrap(env, seed):
    rows = []
    for bf in [0.0, 0.1, 0.25, 0.5, 0.75, 0.99]:
        r = run_config(env, seed, {"bootstrap_frac": bf})
        rows += emit_rows("bootstrap", bf, bf == 0.1, env, seed, {"bootstrap_frac": bf}, r["metrics"], all_sizes=False)
    return rows


def sweep_ridge(env, seed):
    rows = []
    for lam in [0.0, 1e-4, 1e-2, 1.0, 100.0]:
        r = run_config(env, seed, {"lambda_reg": lam})
        rows += emit_rows("ridge", lam, lam == 0.0, env, seed, {"lambda_reg": lam}, r["metrics"], all_sizes=False)
    return rows


def sweep_layer(env, seed):
    # The sequential-correction architecture (ensembles.py: h=X_sub start) requires
    # the perturbed set to begin at layer 0. Natively supported target-layer
    # configs are therefore single-block [0] and multi-block [0,2] (submitted
    # default). Both use the identical LS multi path, so this is an apples-to-apples
    # single-vs-multi-block comparison. Deeper single layers ([1],[2]) are not
    # supported without running the unperturbed prefix (documented limitation).
    rows = []
    configs = [([0], "single-block-L0"), ([0, 2], "multi-block-L0,2*")]
    for layers, label in configs:
        is_def = (layers == [0, 2])
        r = run_config(env, seed, {"perturb_layers": layers, "layer_scope": "multi"})
        rows += emit_rows("layer", label, is_def, env, seed,
                          {"perturb_layers": layers, "layer_scope": "multi"},
                          r["metrics"], all_sizes=False)
    return rows


SWEEPS = {"scale": sweep_scale, "rank": sweep_rank, "bootstrap": sweep_bootstrap,
          "ridge": sweep_ridge, "layer": sweep_layer}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="HalfCheetah-v5")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--sweeps", nargs="+", default=list(SWEEPS), choices=list(SWEEPS))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    # Run matrix
    counts = {"scale": 2, "rank": 5, "bootstrap": 6, "ridge": 5, "layer": 2}
    n_builds = sum(counts[s] for s in args.sweeps) * len(args.seeds)
    print("=" * 70)
    print(f"Priority 1 sensitivity | env={args.env} seeds={args.seeds} sweeps={args.sweeps}")
    print(f"Builds per seed: " + ", ".join(f"{s}={counts[s]}" for s in args.sweeps))
    print(f"TOTAL BUILDS: {n_builds}  (~40s each => ~{n_builds*40//60} min)")
    print(f"Anchor: {ANCHOR}")
    print("=" * 70)
    if args.dry_run:
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    new_file = not CSV_PATH.exists()
    with open(CSV_PATH, "a", newline="") as fcsv:
        w = csv.DictWriter(fcsv, fieldnames=CSV_COLS)
        if new_file:
            w.writeheader()
        for seed in args.seeds:
            for s in args.sweeps:
                t0 = time.time()
                rows = SWEEPS[s](args.env, seed)
                for row in rows:
                    w.writerow(row)
                fcsv.flush()
                print(f"[{args.env} seed{seed}] sweep '{s}': {len(rows)} rows in {time.time()-t0:.0f}s")
    print(f"\nWrote {CSV_PATH}")


if __name__ == "__main__":
    main()
