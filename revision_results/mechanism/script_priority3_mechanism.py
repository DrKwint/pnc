#!/usr/bin/env python3
"""Priority 3 — distance–disagreement mechanism beyond Ant-v5 (MuJoCo second domain).

Replicates the submitted Ant-v5 diagnostic (plot_ant_bridge_q123 Panel B) on a second
environment: hidden-space regularized Mahalanobis distance to the calibration distribution
vs the P&C ensemble's per-point predictive disagreement (sqrt of total predictive variance,
the paper's uncertainty summary).

Pipeline per (env, seed):
  1. Build base prob MLP (seed-matched, deterministic).
  2. Hidden Mahalanobis at the PnC calibration layer (layer_idx=2), mu/Sigma from id_train
     hidden activations, Sigma regularized by 1e-4*mean(diag).
  3. Run the canonical P&C (redirected) to write per-point pred_var sidecars per regime.
  4. Correlate distance vs sqrt(pred_var): pooled + within-regime Spearman, regime-controlled
     log-linear OLS slope, binned means +/- SEM.

Outputs (results/neurips_2026_rebuttal/priority3/):
  mechanism_second_domain.csv           (per-sample distance + disagreement, all regimes)
  mechanism_second_domain_summary.md
Single GPU job. Run: .venv/bin/python scripts/neurips_2026_rebuttal/priority3_mechanism.py --env HalfCheetah-v5
"""
import os
import sys
os.environ.setdefault("JAX_PLATFORMS", "cuda,cpu")
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))
os.chdir(_REPO)

import argparse
import csv
import json

import numpy as np
import jax.numpy as jnp
import luigi
from flax import nnx
from scipy.stats import spearmanr

from pnc_core.models import ProbabilisticRegressionModel
from pnc_core.training import train_probabilistic_model
from pnc_core.util import _split_data, get_intermediate_state, seed_everything
from pnc_core.gym_tasks import GymPJSVD

PNC = dict(steps=10000, subset_size=10000, n_directions=20, n_perturbations=50,
           perturbation_sizes=[5.0, 10.0, 20.0, 50.0], layer_scope="multi",
           pjsvd_family="random", correction_mode="least_squares",
           safe_subspace_backend="projected_residual", probabilistic_base_model=True,
           hidden_dims=[200, 200, 200, 200], activation="relu", lambda_reg=0.0,
           bootstrap_frac=0.1, policy_preset="neurips_minari",
           compute_geometry=False, compute_l2=False)
OUT = Path("results/neurips_2026_rebuttal/priority3")
REG_FILES = {"id": "data_id_eval", "near": "data_ood_near", "mid": "data_ood_mid", "far": "data_ood_far"}
REG_STORE = {"id": "id", "near": "ood_near", "mid": "ood_mid", "far": "ood_far"}


class _RedirectPJSVD(GymPJSVD):
    def output(self):
        orig = Path(super().output().path)
        return luigi.LocalTarget(str(OUT / "pnc" / orig.parent.name / orig.name))


def hidden_mahal(env, seed):
    seed_everything(seed)
    tr = np.load(f"results/{env}/data_id_train_seed{seed}_steps10000.npz")
    xin, yin = jnp.array(tr["inputs"], jnp.float32), jnp.array(tr["targets"], jnp.float32)
    x_tr, y_tr, x_va, y_va = _split_data(xin, yin)
    model = ProbabilisticRegressionModel(xin.shape[1], yin.shape[1], rngs=nnx.Rngs(params=seed),
                                         hidden_dims=[200, 200, 200, 200], activation=nnx.relu)
    model = train_probabilistic_model(model, x_tr, y_tr, x_va, y_va)

    def h(x_np):
        return np.asarray(get_intermediate_state(model, jnp.array(x_np, jnp.float32), layer_idx=2))
    h_train = h(tr["inputs"])
    mu = h_train.mean(0).astype(np.float64)
    C = np.cov(h_train.astype(np.float64), rowvar=False)
    lam = 1e-4 * np.trace(C) / C.shape[0]
    Cinv = np.linalg.inv(C + lam * np.eye(C.shape[0]))

    def mahal(x_np):
        dv = h(x_np).astype(np.float64) - mu
        return np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", dv, Cinv, dv), 0.0))
    return {r: mahal(np.load(f"results/{env}/{REG_FILES[r]}_seed{seed}_steps10000.npz")["inputs"])
            for r in REG_FILES}


def run_pnc_pred_var(env, seed):
    task = _RedirectPJSVD(env=env, seed=seed, compute_geometry=False, **{k: v for k, v in PNC.items() if k not in ("compute_geometry", "compute_l2")}, compute_l2=False)
    Path(task.output().path).parent.mkdir(parents=True, exist_ok=True)
    ok = luigi.build([task], local_scheduler=True, workers=1, log_level="WARNING")
    if not ok:
        raise SystemExit("PnC build failed")
    metrics = json.loads(Path(task.output().path).read_text())
    sel = min(metrics.keys(), key=lambda k: metrics[k]["nll_val"])
    base_npz = task.output().path.replace(".json", "")
    side = np.load(f"{base_npz}_ps{sel}.npz")
    return sel, {r: np.asarray(side[f"pred_var_{REG_STORE[r]}"]) for r in REG_STORE}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="HalfCheetah-v5")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    dist = hidden_mahal(args.env, args.seed)
    sel, pv = run_pnc_pred_var(args.env, args.seed)
    print(f"{args.env}: selected size {sel}")

    regimes = ["id", "near", "mid", "far"]
    rows, d_all, y_all, reg_all = [], [], [], []
    for i, r in enumerate(regimes):
        d = np.asarray(dist[r], np.float64)
        y = np.sqrt(np.maximum(np.asarray(pv[r], np.float64), 0.0))
        n = min(d.size, y.size)
        d, y = d[:n], y[:n]
        d_all.append(d); y_all.append(y); reg_all.append(np.full(n, i))
        for j in range(n):
            rows.append({"env": args.env, "seed": args.seed, "regime": r,
                         "distance_mahal": float(d[j]), "disagreement_sqrt_predvar": float(y[j])})
    d_all = np.concatenate(d_all); y_all = np.concatenate(y_all); reg_all = np.concatenate(reg_all)

    # stats
    pooled_rho, pooled_p = spearmanr(d_all, y_all)
    within = {r: spearmanr(d_all[reg_all == i], y_all[reg_all == i])[0] for i, r in enumerate(regimes)}
    # regime-controlled OLS: y ~ 1 + log10(d) + regime dummies
    n = d_all.size
    cols = [np.ones(n), np.log10(np.maximum(d_all, 1e-9))]
    for i in range(1, len(regimes)):
        cols.append((reg_all == i).astype(float))
    X = np.column_stack(cols)
    beta, *_ = np.linalg.lstsq(X, y_all, rcond=None)
    resid = y_all - X @ beta
    dof = n - X.shape[1]
    se = np.sqrt(np.diag((resid @ resid / dof) * np.linalg.inv(X.T @ X)))

    with open(OUT / f"mechanism_{args.env}_seed{args.seed}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["env", "seed", "regime", "distance_mahal", "disagreement_sqrt_predvar"])
        w.writeheader()
        for r in rows:
            w.writerow(r)

    summary = {
        "env": args.env, "seed": args.seed, "selected_size": sel,
        "pooled_spearman": {"rho": float(pooled_rho), "p": float(pooled_p), "n": int(n)},
        "within_regime_spearman": {r: float(v) for r, v in within.items()},
        "regime_controlled_ols_slope_log10d": float(beta[1]), "slope_se": float(se[1]),
    }
    (OUT / f"mechanism_{args.env}_seed{args.seed}.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
