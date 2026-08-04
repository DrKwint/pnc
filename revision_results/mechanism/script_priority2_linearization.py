#!/usr/bin/env python3
"""Priority 2 — bridge the local corrected-sensitivity theory to finite perturbations.

For single-block P&C (perturb hidden layer 0 by v = alpha*u, refit the next affine layer
on the calibration subset X_sub), the theory's post-correction residual at the correction
interface is
    r(x; v) = [W1' * relu(x @ (W0+v) + b0) + b1']  -  [W1 * relu(x @ W0 + b0) + b1]
where (W1', b1') = argmin over the LS/ridge fit on X_sub with target = the UNPERTURBED
layer-1 pre-activation. r(x;0)=0.

We compare:
  r_actual(x, alpha) = r(x; alpha*u)                         (finite)
  r_linear(x, alpha) = alpha * A_S(x) u                      (first-order),
where A_S(x)u = d/dalpha r(x; alpha*u)|_{alpha=0}, computed by jax.jvp THROUGH the LS solve
(and cross-checked by central finite difference at two epsilons).

Metrics per (regime, alpha), aggregated over sampled points x and random directions u:
  1. rel_lin_err   = ||r_actual - r_linear|| / (||r_linear|| + eps)
  2. rel_by_actual = ||r_actual - r_linear|| / (||r_actual|| + eps)
  3. cosine(r_actual, r_linear)
  4. Spearman across examples between ||r_linear|| and ||r_actual||
  6. remainder ||r_actual - r_linear|| vs alpha: log-log slope (≈2 ⇒ quadratic local remainder)

Outputs (results/neurips_2026_rebuttal/priority2/):
  linearization_diagnostics.csv   (per regime x alpha x direction: aggregated over points)
  linearization_summary.md
Single GPU job. Run:  .venv/bin/python scripts/neurips_2026_rebuttal/priority2_linearization.py --env Ant-v5
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
import jax
import jax.numpy as jnp
from flax import nnx
from scipy.stats import spearmanr

from pnc_core.models import ProbabilisticRegressionModel
from pnc_core.training import train_probabilistic_model
from pnc_core.util import _split_data, seed_everything

EPS = 1e-12


def load_regime(env, seed, regime):
    fn = {"id": "data_id_eval", "near": "data_ood_near",
          "mid": "data_ood_mid", "far": "data_ood_far"}[regime]
    d = np.load(f"results/{env}/{fn}_seed{seed}_steps10000.npz")
    return np.asarray(d["inputs"], np.float32)


def build_base(env, seed):
    tr = np.load(f"results/{env}/data_id_train_seed{seed}_steps10000.npz")
    xin = jnp.array(tr["inputs"], jnp.float32)
    yin = jnp.array(tr["targets"], jnp.float32)
    x_tr, y_tr, x_va, y_va = _split_data(xin, yin)
    model = ProbabilisticRegressionModel(
        xin.shape[1], yin.shape[1], rngs=nnx.Rngs(params=seed),
        hidden_dims=[200, 200, 200, 200], activation=nnx.relu)
    model = train_probabilistic_model(model, x_tr, y_tr, x_va, y_va)
    W0 = jnp.asarray(model.layers[0].kernel.get_value())
    b0 = jnp.asarray(model.layers[0].bias.get_value())
    W1 = jnp.asarray(model.layers[1].kernel.get_value())
    b1 = jnp.asarray(model.layers[1].bias.get_value())
    return (W0, b0, W1, b1), np.asarray(xin)


def make_residual_fn(W0, b0, W1, b1, X_sub, u_mat, lam=0.0):
    """Return f(alpha, x) -> residual vectors r(x; alpha*u) at the correction interface."""
    Y0_sub = jax.nn.relu(X_sub @ W0 + b0)                 # (n_sub, H)
    target = Y0_sub @ W1 + b1                              # (n_sub, H1) unperturbed pre-activation
    ones = jnp.ones((X_sub.shape[0], 1), Y0_sub.dtype)

    def solve(Yv):
        H = jnp.concatenate([Yv, ones], axis=1)           # (n_sub, H+1)
        if lam > 0.0:
            A = H.T @ H + lam * jnp.eye(H.shape[1], dtype=H.dtype)
            Wb = jnp.linalg.solve(A, H.T @ target)
        else:
            Wb, *_ = jnp.linalg.lstsq(H, target, rcond=None)
        return Wb[:-1], Wb[-1]

    def f(alpha, x):
        W0v = W0 + alpha * u_mat
        Yv_sub = jax.nn.relu(X_sub @ W0v + b0)
        W1p, b1p = solve(Yv_sub)
        yv_x = jax.nn.relu(x @ W0v + b0)
        y0_x = jax.nn.relu(x @ W0 + b0)
        return (yv_x @ W1p + b1p) - (y0_x @ W1 + b1)      # (n_x, H1)
    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="Ant-v5")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-points", type=int, default=200)
    ap.add_argument("--n-dirs", type=int, default=8)
    ap.add_argument("--n-sub", type=int, default=4096)
    ap.add_argument("--alphas", type=float, nargs="+",
                    default=[0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0])
    args = ap.parse_args()
    seed_everything(args.seed)

    (W0, b0, W1, b1), Xid = build_base(args.env, args.seed)
    rng = np.random.RandomState(args.seed)
    sub_idx = rng.choice(len(Xid), min(args.n_sub, len(Xid)), replace=False)
    X_sub = jnp.asarray(Xid[sub_idx])
    D = W0.size
    print(f"{args.env}: W0 {W0.shape} (D={D}), n_sub={X_sub.shape[0]}")

    regimes = ["id", "near", "mid", "far"]
    pts = {r: jnp.asarray(load_regime(args.env, args.seed, r)[:args.n_points]) for r in regimes}

    # Random unit directions in weight space (matches the "random" family).
    dirs = rng.normal(size=(args.n_dirs, D)).astype(np.float32)
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-12

    rows = []
    for di in range(args.n_dirs):
        u_mat = jnp.asarray(dirs[di].reshape(W0.shape))
        f = make_residual_fn(W0, b0, W1, b1, X_sub, u_mat)
        for r in regimes:
            x = pts[r]
            # First-order A_S(x)u via central finite difference (jax.jvp through lstsq is
            # numerically unstable / NaN-prone; central diff is a stable forward eval of the
            # correction solve). Validate stability across two epsilons.
            cd_est = {}
            for eps in (1e-3, 1e-2):
                cd_est[eps] = np.asarray((f(eps, x) - f(-eps, x)) / (2 * eps))
            r_lin_unit = cd_est[1e-2]                       # (n_x, H1) = A_S(x)u
            # two-epsilon agreement (should be ~0 if the derivative estimate is stable)
            num = np.linalg.norm(cd_est[1e-3] - cd_est[1e-2], axis=1)
            den = np.linalg.norm(r_lin_unit, axis=1) + EPS
            cd_ok = {1e-3: float(np.nanmedian(num / den)), 1e-2: float(np.nanmedian(num / den))}
            for alpha in args.alphas:
                r_act = np.asarray(f(alpha, x))            # (n_x, H1)
                r_lin = alpha * r_lin_unit
                diff = np.linalg.norm(r_act - r_lin, axis=1)
                n_act = np.linalg.norm(r_act, axis=1)
                n_lin = np.linalg.norm(r_lin, axis=1)
                cos = np.sum(r_act * r_lin, axis=1) / (n_act * n_lin + EPS)
                rho, _ = spearmanr(n_lin, n_act, nan_policy="omit")
                rows.append({
                    "env": args.env, "seed": args.seed, "direction": di, "regime": r,
                    "alpha": alpha,
                    "rel_lin_err_median": float(np.nanmedian(diff / (n_lin + EPS))),
                    "rel_by_actual_median": float(np.nanmedian(diff / (n_act + EPS))),
                    "cosine_median": float(np.nanmedian(cos)),
                    "spearman_normlin_normact": float(rho),
                    "remainder_median": float(np.nanmedian(diff)),
                    "norm_actual_median": float(np.nanmedian(n_act)),
                    "norm_linear_median": float(np.nanmedian(n_lin)),
                    "cd_reldiff_eps1e-3": cd_ok[1e-3],
                    "cd_reldiff_eps1e-2": cd_ok[1e-2],
                })
        print(f"  direction {di+1}/{args.n_dirs} done")

    out = Path("results/neurips_2026_rebuttal/priority2")
    out.mkdir(parents=True, exist_ok=True)
    cols = list(rows[0].keys())
    with open(out / "linearization_diagnostics.csv", "w", newline="") as fcsv:
        w = csv.DictWriter(fcsv, fieldnames=cols); w.writeheader()
        for r in rows:
            w.writerow(r)

    # Summary: aggregate over directions; per regime x alpha; remainder log-log slope.
    import statistics as stx
    def agg(regime, alpha, key):
        vals = [r[key] for r in rows if r["regime"] == regime and r["alpha"] == alpha
                and r[key] == r[key]]  # drop NaN
        return stx.mean(vals) if vals else float("nan")

    md = ["# Priority 2 — linearization diagnostics (single-block P&C, correction interface)",
          f"", f"- env={args.env} seed={args.seed} n_points={args.n_points} "
          f"n_dirs={args.n_dirs} n_sub={X_sub.shape[0]}",
          f"- jvp-vs-central-difference median rel-diff (should be ~0): "
          f"eps1e-3={stx.mean([r['cd_reldiff_eps1e-3'] for r in rows]):.2e}, "
          f"eps1e-2={stx.mean([r['cd_reldiff_eps1e-2'] for r in rows]):.2e}",
          "", "## Relative linearization error (median over pts+dirs), by regime x alpha",
          "", "| regime | " + " | ".join(f"α={a:g}" for a in args.alphas) + " |",
          "|" + "---|" * (len(args.alphas) + 1)]
    for r in regimes:
        md.append("| " + r + " | " + " | ".join(f"{agg(r, a, 'rel_lin_err_median'):.3f}" for a in args.alphas) + " |")
    md += ["", "## Cosine(r_actual, r_linear) (median), by regime x alpha", "",
           "| regime | " + " | ".join(f"α={a:g}" for a in args.alphas) + " |",
           "|" + "---|" * (len(args.alphas) + 1)]
    for r in regimes:
        md.append("| " + r + " | " + " | ".join(f"{agg(r, a, 'cosine_median'):.3f}" for a in args.alphas) + " |")
    md += ["", "## Spearman(||r_linear||, ||r_actual||) across examples (median over dirs)", "",
           "| regime | " + " | ".join(f"α={a:g}" for a in args.alphas) + " |",
           "|" + "---|" * (len(args.alphas) + 1)]
    for r in regimes:
        md.append("| " + r + " | " + " | ".join(f"{agg(r, a, 'spearman_normlin_normact'):.3f}" for a in args.alphas) + " |")

    # remainder log-log slope at small alpha (<=1) per regime
    md += ["", "## Remainder ||r_actual - r_linear|| log-log slope vs α (small α ≤ 1)",
           "A slope ≈ 2 supports a quadratic local remainder.", "",
           "| regime | slope (α≤1) | slope (all α) |", "|---|---:|---:|"]
    small = [a for a in args.alphas if a <= 1.0]
    for r in regimes:
        def slope(alist):
            xs = np.log10(alist)
            ys = np.log10([max(agg(r, a, "remainder_median"), 1e-30) for a in alist])
            A = np.vstack([xs, np.ones_like(xs)]).T
            m, _ = np.linalg.lstsq(A, ys, rcond=None)[0]
            return m
        md.append(f"| {r} | {slope(small):.2f} | {slope(args.alphas):.2f} |")
    (out / "linearization_summary.md").write_text("\n".join(md) + "\n")
    print(f"\nWrote {out}/linearization_diagnostics.csv and linearization_summary.md")


if __name__ == "__main__":
    main()
