#!/usr/bin/env python3
"""Empirical validation of Theorem T4 (expected variance ∝ squared distance).

For a fixed env and seed, rebuilds a PJSVD ensemble and a Deep Ensemble from
scratch (no new luigi runs required beyond these ad-hoc builds), computes
per-sample predictive variance V(x) and per-sample distance-from-calibration-
hull d(x), and fits a linear model ``V(x) = β₀ + β₁·d(x)²``. Reports R² and
slope per (method, env).

The test of T4 is whether PJSVD's V(x) is well-explained by d(x)² (high R²
and positive slope), indicating that its variance truly tracks geometric
distance from the calibration data.

Usage:
    .venv/bin/python experiments/scripts/variance_distance_decomposition.py \
        --env Ant-v5 --seed 0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make the repo root importable when running this script directly.
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import jax
import jax.numpy as jnp
import numpy as np


def build_pjsvd_ensemble(env: str, seed: int, scale: float = 20.0, n_perturbations: int = 50, n_directions: int = 20):
    """Train a probabilistic base model and build a multi-layer LS PJSVD ensemble.

    Returns (ensemble, base_model, X_sub, h_old, geom).
    """
    from flax import nnx

    from models import ProbabilisticRegressionModel
    from training import train_probabilistic_model
    from data import load_minari_transitions
    from gym_tasks import _split_data
    from ensembles import PJSVDEnsemble
    from geometry import CalibrationGeometry

    inputs_id, targets_id, _ = load_minari_transitions(env, "id_train", 10000, seed)
    x_tr, y_tr, x_va, y_va = _split_data(inputs_id, targets_id)

    print(f"[vdd] Training probabilistic base model on {env} seed {seed}...")
    model = ProbabilisticRegressionModel(
        inputs_id.shape[1], targets_id.shape[1],
        nnx.Rngs(params=seed),
        hidden_dims=[200, 200, 200, 200],
        activation=nnx.relu,
    )
    model = train_probabilistic_model(model, x_tr, y_tr, x_va, y_va, steps=5000, batch_size=64)
    for p in jax.tree_util.tree_leaves(nnx.state(model)):
        if hasattr(p, "block_until_ready"):
            p.block_until_ready()

    Ws = [model.layers[i].kernel.get_value() for i in range(len(model.layers))]
    bs = [model.layers[i].bias.get_value() for i in range(len(model.layers))]
    n_hidden = len(Ws)
    perturb_indices = list(range(0, n_hidden, 2))
    layer_params = {f"l{i+1}": {"W": Ws[i], "b": bs[i]} for i in perturb_indices}

    X_sub = jnp.array(inputs_id[np.random.RandomState(seed).choice(len(inputs_id), 4096, replace=False)])
    act_fn = nnx.relu

    # Per-layer random direction specs (matches GymHybridPnCDE pattern)
    layer_specs_list = []
    for li, pi in enumerate(perturb_indices):
        W_li = Ws[pi]
        D = W_li.size
        rng_li = np.random.RandomState(seed + li)
        rand_dirs = rng_li.normal(size=(n_directions, D)).astype(np.float32)
        rand_dirs /= np.linalg.norm(rand_dirs, axis=1, keepdims=True) + 1e-12
        layer_specs_list.append({
            "v_opts": rand_dirs,
            "sigmas": np.ones(n_directions, dtype=np.float32),
            "W_shape": W_li.shape,
        })

    h_old = X_sub
    for pi in perturb_indices:
        h_old = act_fn(h_old @ Ws[pi] + bs[pi])

    geom = CalibrationGeometry(h_old, retain_rank=None)

    all_z = np.stack([
        np.random.RandomState(seed + li + 7).normal(size=(n_perturbations, n_directions))
        for li in range(len(perturb_indices))
    ], axis=1)

    ens = PJSVDEnsemble(
        base_model=model,
        v_opts=np.zeros((1, 1)),
        sigmas=np.ones(1),
        z_coeffs=all_z,
        perturbation_scale=scale,
        X_sub=X_sub,
        layers=[f"l{pi+1}" for pi in perturb_indices],
        correction_mode="least_squares",
        activation=act_fn,
        layer_params=layer_params,
        correction_params={"target_act": h_old},
        tail_is_hidden=True,
        layer_specs=layer_specs_list,
    )
    return ens, model, X_sub, h_old, geom, perturb_indices


def build_de_ensemble(env: str, seed: int, n_members: int = 5):
    from flax import nnx

    from models import ProbabilisticRegressionModel
    from training import train_probabilistic_model
    from data import load_minari_transitions
    from gym_tasks import _split_data
    from ensembles import StandardEnsemble

    inputs_id, targets_id, _ = load_minari_transitions(env, "id_train", 10000, seed)
    x_tr, y_tr, x_va, y_va = _split_data(inputs_id, targets_id)

    models = []
    for i in range(n_members):
        m = ProbabilisticRegressionModel(
            inputs_id.shape[1], targets_id.shape[1],
            nnx.Rngs(params=seed + i * 1000),
            hidden_dims=[200, 200, 200, 200],
            activation=nnx.relu,
        )
        m = train_probabilistic_model(m, x_tr, y_tr, x_va, y_va, steps=5000, batch_size=64)
        models.append(m)
    return StandardEnsemble(models)


def per_sample_variance(ens, inputs):
    from util import _predictive_mean_var
    preds = ens.predict(inputs)
    _, var = _predictive_mean_var(preds)
    return np.array(jnp.mean(var, axis=-1))


def intermediate_activation(model, inputs, layer_idx):
    from util import get_intermediate_state
    return get_intermediate_state(model, inputs, layer_idx=layer_idx)


def mahalanobis_distance(reference: jax.Array, query: jax.Array, shrinkage: float = 1e-3) -> np.ndarray:
    """Mahalanobis distance from every row of ``query`` to the distribution of ``reference``.

    Uses the centered reference covariance with a small shrinkage on the diagonal
    for numerical stability. When ``reference`` has more rows than columns (B > N),
    the covariance is full rank and Mahalanobis distance is well-defined for all query
    points; by contrast the affine-hull L2 distance collapses to zero because the hull
    spans the ambient space. Mahalanobis distance is the right geometric quantity for
    T4's empirical validation in the regime where calibration is plentiful.
    """
    ref = np.asarray(reference)
    q = np.asarray(query)
    mu = ref.mean(axis=0)
    X = ref - mu
    cov = (X.T @ X) / max(1, X.shape[0] - 1)
    cov_reg = cov + shrinkage * np.eye(cov.shape[0], dtype=cov.dtype)
    try:
        cov_inv = np.linalg.inv(cov_reg)
    except np.linalg.LinAlgError:
        cov_inv = np.linalg.pinv(cov_reg)
    d = q - mu
    return np.sqrt(np.clip(np.einsum("ij,jk,ik->i", d, cov_inv, d), a_min=0.0, a_max=None))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="Ant-v5")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", default="experiments/figures")
    parser.add_argument("--n-de", type=int, default=5)
    parser.add_argument("--scale", type=float, default=20.0)
    args = parser.parse_args()

    from data import load_minari_transitions

    regimes = ["id_eval", "ood_near", "ood_mid", "ood_far"]
    regime_inputs = {}
    regime_targets = {}
    for reg in regimes:
        ins, tgts, _ = load_minari_transitions(args.env, reg, 10000, args.seed)
        regime_inputs[reg] = jnp.array(ins)
        regime_targets[reg] = jnp.array(tgts)

    # Build PJSVD ensemble + helpers
    pnc_ens, base_model, X_sub, h_old, geom, perturb_indices = build_pjsvd_ensemble(
        args.env, args.seed, scale=args.scale,
    )
    n_pert_layers = len(perturb_indices)

    # Build DE ensemble
    print(f"[vdd] Training Deep Ensemble ({args.n_de} members)...")
    de_ens = build_de_ensemble(args.env, args.seed, n_members=args.n_de)

    # Per-sample variance and per-sample distance from calibration hull
    methods = {"PJSVD": pnc_ens, "DE": de_ens}
    per_method = {m: {"d": [], "v": [], "regime": []} for m in methods}

    # Use calibration activations h_old from the PnC setup as the reference.
    # Mahalanobis distance is meaningful in the B >> N regime (which is our case),
    # where the affine-hull distance from CalibrationGeometry collapses to zero.
    h_old_np = np.asarray(h_old)
    for reg in regimes:
        x = regime_inputs[reg]
        # Distance from calibration distribution at the post-perturb layer's activation
        h_reg = intermediate_activation(base_model, x, layer_idx=n_pert_layers)
        d_reg = mahalanobis_distance(h_old_np, np.asarray(h_reg))
        for name, ens in methods.items():
            v_reg = per_sample_variance(ens, x)
            per_method[name]["d"].append(d_reg)
            per_method[name]["v"].append(v_reg)
            per_method[name]["regime"].append([reg] * len(d_reg))

    # Aggregate + fit linear V ~ β0 + β1·d²
    print(f"\n=== Variance ~ distance² fits ({args.env} seed {args.seed}) ===")
    fit_results = {}
    for name, buf in per_method.items():
        d = np.concatenate(buf["d"])
        v = np.concatenate(buf["v"])
        d2 = d ** 2
        A = np.stack([np.ones_like(d2), d2], axis=1)
        coef, *_ = np.linalg.lstsq(A, v, rcond=None)
        v_pred = A @ coef
        ss_res = float(np.sum((v - v_pred) ** 2))
        ss_tot = float(np.sum((v - v.mean()) ** 2))
        r2 = 1.0 - ss_res / (ss_tot + 1e-12)
        fit_results[name] = {"beta0": float(coef[0]), "beta1": float(coef[1]), "r2": r2}
        print(f"  {name}: V = {coef[0]:.4e} + {coef[1]:.4e}·d²   R² = {r2:.4f}")

    # Plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        colors = {"id_eval": "#1f77b4", "ood_near": "#2ca02c", "ood_mid": "#ff7f0e", "ood_far": "#d62728"}

        fig, axes = plt.subplots(1, len(methods), figsize=(6 * len(methods), 4), sharey=False)
        for ax, (name, buf) in zip(axes, per_method.items()):
            for d_reg, v_reg, reg_labels in zip(buf["d"], buf["v"], buf["regime"]):
                reg = reg_labels[0]
                ax.scatter(
                    d_reg ** 2, v_reg, s=4, alpha=0.3,
                    color=colors[reg], label=reg,
                )
            # Fit line
            d_all = np.concatenate(buf["d"])
            d2_all = d_all ** 2
            xs = np.linspace(0, float(d2_all.max()), 200)
            f = fit_results[name]
            ax.plot(xs, f["beta0"] + f["beta1"] * xs, "k-", linewidth=1.5,
                    label=f"fit: R²={f['r2']:.2f}")
            ax.set_xlabel("dist(y(x), G(Y))²")
            ax.set_title(f"{name} — {args.env} s{args.seed}")
            ax.set_ylabel("per-sample variance")
            ax.grid(alpha=0.3)
            # Dedupe legend
            handles, labels = ax.get_legend_handles_labels()
            seen = {}
            for h, l in zip(handles, labels):
                if l not in seen:
                    seen[l] = h
            ax.legend(list(seen.values()), list(seen.keys()), loc="upper left", fontsize=8)

        fig.suptitle(
            f"Empirical test of Theorem T4: variance linear in dist² — {args.env}",
            fontsize=12,
        )
        fig.tight_layout()
        Path(args.out_dir).mkdir(parents=True, exist_ok=True)
        out_path = Path(args.out_dir) / f"variance_distance_{args.env}_seed{args.seed}.png"
        fig.savefig(out_path, dpi=140, bbox_inches="tight")
        print(f"\nSaved {out_path}")
    except ImportError:
        print("matplotlib not available, skipping plot.")


if __name__ == "__main__":
    main()
