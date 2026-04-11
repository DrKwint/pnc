#!/usr/bin/env python3
"""Empirical validation of Theorem T6 (exponential coverage rate).

For a fixed PnC base model on one env, load its PJSVD ensemble at
n_perturbations = 200 (or the largest available) and measure Far AUROC when
we subsample to M' ∈ {5, 10, 20, 50, 100, 200} members. Plot the empirical
AUROC curve against the theoretical coverage lower bound
    1 − exp(−M' · I_k(θ))
for a fixed half-angle θ.

We don't have M=200 saved JSONs directly; the largest existing is n=50. For
E-T6 we re-load a trained base model, recompute a large PJSVD ensemble at
M=200, and do the subsampling analysis. This script does NOT require new GPU
runs beyond that one-shot rebuild.

Usage:
    .venv/bin/python experiments/scripts/coverage_rate.py \
        --env Ant-v5 --seed 0
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import jax
import jax.numpy as jnp
import numpy as np


def spherical_cap_mass(k: int, theta_rad: float) -> float:
    """Normalized spherical cap mass I_k(θ) of half-angle θ in R^k.

    I_k(θ) = (1/2) · I_{sin²θ}(k/2, 1/2) where I is the regularized incomplete
    beta function (Li, 2011; Ball, 1997). Returned value is in [0, 1/2].
    """
    from scipy.special import betainc  # noqa: WPS433

    if theta_rad <= 0:
        return 0.0
    if theta_rad >= math.pi / 2:
        return 0.5
    sin_sq = math.sin(theta_rad) ** 2
    return 0.5 * float(betainc(k / 2.0, 0.5, sin_sq))


def theoretical_coverage(m: int, k: int, theta_rad: float) -> float:
    """Theoretical lower bound on coverage prob from T6."""
    p_cap = 2.0 * spherical_cap_mass(k, theta_rad)  # both hemispheres
    if p_cap <= 0:
        return 0.0
    return 1.0 - math.exp(-m * p_cap)


def sorted_subsample_auroc(
    means_np: np.ndarray,
    vars_np: np.ndarray,
    split_idx: int,
    n_sub: int,
    rng: np.random.RandomState,
    n_trials: int = 16,
) -> float:
    """Randomly subsample n_sub members, compute ensemble predictive var via
    law of total variance, then AUROC. All computations in numpy to avoid
    GPU OOM when n_members × B × D is large.
    """
    from metrics import compute_ood_metrics

    total_members = means_np.shape[0]
    aurocs = []
    for _ in range(n_trials):
        idx = rng.choice(total_members, size=n_sub, replace=False)
        sm = means_np[idx]  # (n_sub, B, D)
        sv = vars_np[idx]   # (n_sub, B, D)
        mean = sm.mean(axis=0)  # (B, D)
        total_var = sv.mean(axis=0) + sm.var(axis=0)  # (B, D)
        # Collapse dim-axis to get per-sample scalar variance
        pv = total_var.mean(axis=-1)  # (B,)
        pvi = pv[:split_idx]
        pvo = pv[split_idx:]
        if len(pvi) == 0 or len(pvo) == 0:
            continue
        auroc, _ = compute_ood_metrics(pvi, pvo)
        aurocs.append(auroc)
    return float(np.mean(aurocs)) if aurocs else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="Ant-v5")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-members", type=int, default=200)
    parser.add_argument("--n-directions", type=int, default=20)
    parser.add_argument("--scale", type=float, default=20.0)
    parser.add_argument("--out-dir", default="experiments/figures")
    parser.add_argument("--theta-deg", type=float, default=30.0)
    args = parser.parse_args()

    # Build the PJSVD ensemble with n_members (e.g. 200) on the fly so we can
    # subsample without re-training for each M'.
    import numpy as np
    from flax import nnx

    from models import ProbabilisticRegressionModel
    from training import train_probabilistic_model
    from data import load_minari_transitions
    from gym_tasks import _split_data
    from ensembles import PJSVDEnsemble
    from util import _evaluate_gym, _predictive_mean_var

    print(f"[coverage_rate] env={args.env} seed={args.seed} n_members={args.n_members}")

    # Load data via the CollectGymData-equivalent path
    inputs_id, targets_id, _ = load_minari_transitions(args.env, "id_train", 10000, args.seed)
    inputs_id_eval, targets_id_eval, _ = load_minari_transitions(args.env, "id_eval", 10000, args.seed)
    inputs_ood, targets_ood, _ = load_minari_transitions(args.env, "ood_far", 10000, args.seed)
    x_tr, y_tr, x_va, y_va = _split_data(inputs_id, targets_id)

    # Train base model
    print("Training probabilistic base model...")
    model = ProbabilisticRegressionModel(
        inputs_id.shape[1], targets_id.shape[1],
        nnx.Rngs(params=args.seed),
        hidden_dims=[200, 200, 200, 200],
        activation=nnx.relu,
    )
    model = train_probabilistic_model(model, x_tr, y_tr, x_va, y_va, steps=5000, batch_size=64)

    # Build large PJSVD ensemble
    Ws = [model.layers[i].kernel.get_value() for i in range(len(model.layers))]
    bs = [model.layers[i].bias.get_value() for i in range(len(model.layers))]
    n_hidden = len(Ws)
    perturb_indices = list(range(0, n_hidden, 2))
    layer_params = {f"l{i+1}": {"W": Ws[i], "b": bs[i]} for i in perturb_indices}

    actual_subset = min(len(inputs_id), 4096)
    subset_idx = np.random.choice(len(inputs_id), actual_subset, replace=False)
    X_sub = jnp.array(inputs_id[subset_idx])

    # Per-layer random direction specs
    layer_specs_list = []
    for li, pi in enumerate(perturb_indices):
        W_li = Ws[pi]
        D = W_li.size
        rng_li = np.random.RandomState(args.seed + li)
        rand_dirs = rng_li.normal(size=(args.n_directions, D)).astype(np.float32)
        rand_dirs /= np.linalg.norm(rand_dirs, axis=1, keepdims=True) + 1e-12
        layer_specs_list.append({
            "v_opts": rand_dirs,
            "sigmas": np.ones(args.n_directions, dtype=np.float32),
            "W_shape": W_li.shape,
        })

    # Original activations through perturbed layers
    act_fn = nnx.relu
    h_old = X_sub
    for pi in perturb_indices:
        h_old = act_fn(h_old @ Ws[pi] + bs[pi])

    # Sample z_coeffs for all M=args.n_members members
    all_z = np.stack([
        np.random.RandomState(args.seed + li).normal(size=(args.n_members, args.n_directions))
        for li in range(len(perturb_indices))
    ], axis=1)  # (n_members, n_layers, n_directions)

    perturbed_layers = [f"l{pi+1}" for pi in perturb_indices]
    ens = PJSVDEnsemble(
        base_model=model,
        v_opts=np.zeros((1, 1)),
        sigmas=np.ones(1),
        z_coeffs=all_z,
        perturbation_scale=args.scale,
        X_sub=X_sub,
        layers=perturbed_layers,
        correction_mode="least_squares",
        activation=act_fn,
        layer_params=layer_params,
        correction_params={"target_act": h_old},
        tail_is_hidden=True,
        layer_specs=layer_specs_list,
    )

    # Predict on stacked ID+OOD — do it in chunks to avoid a single massive GPU allocation,
    # and pull to numpy so subsampling does not hit GPU memory.
    inputs_id_eval_j = jnp.array(inputs_id_eval)
    inputs_ood_j = jnp.array(inputs_ood)
    print("Predicting on stacked ID+OOD (chunked to save GPU memory)...")

    def _predict_to_numpy(ens_obj, x_j, chunk_size=1000):
        means_chunks = []
        vars_chunks = []
        for start in range(0, x_j.shape[0], chunk_size):
            p = ens_obj.predict(x_j[start:start + chunk_size])
            if isinstance(p, tuple):
                means_chunks.append(np.array(p[0]))
                vars_chunks.append(np.array(p[1]))
            else:
                means_chunks.append(np.array(p))
                vars_chunks.append(np.zeros_like(np.array(p)))
        return np.concatenate(means_chunks, axis=1), np.concatenate(vars_chunks, axis=1)

    m_id, v_id = _predict_to_numpy(ens, inputs_id_eval_j)
    m_ood, v_ood = _predict_to_numpy(ens, inputs_ood_j)
    all_means_np = np.concatenate([m_id, m_ood], axis=1)  # (n_members, B_all, D)
    all_vars_np = np.concatenate([v_id, v_ood], axis=1)
    print(f"Ensemble predictions (numpy): means shape={all_means_np.shape}")
    split_idx = len(inputs_id_eval)

    # Sweep M'
    m_values = [5, 10, 20, 50, 100, args.n_members]
    m_values = [m for m in m_values if m <= args.n_members]

    # Effective k used in T6: safe subspace dimension per layer × n_layers
    k_eff = args.n_directions * len(perturb_indices)
    theta_rad = math.radians(args.theta_deg)
    print(f"k_eff = {k_eff}, theta = {args.theta_deg} deg")

    rows = []
    for m in m_values:
        rng = np.random.RandomState(args.seed + m * 131)
        auroc = sorted_subsample_auroc(
            all_means_np, all_vars_np, split_idx, m, rng, n_trials=16,
        )
        cov = theoretical_coverage(m, k_eff, theta_rad)
        rows.append((m, auroc, cov))
        print(f"  M'={m:>4}: empirical AUROC={auroc:.4f}  theoretical coverage={cov:.4f}")

    # Plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        ms = [r[0] for r in rows]
        aurocs = [r[1] for r in rows]
        covs = [r[2] for r in rows]

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(ms, aurocs, "o-", label="Empirical Far AUROC", color="#1f77b4", linewidth=2, markersize=7)
        ax.plot(ms, covs, "s--", label=f"Theoretical coverage, θ={args.theta_deg}°", color="#d62728", linewidth=1.5, markersize=6)
        ax.set_xscale("log")
        ax.set_xlabel("Ensemble size M'")
        ax.set_ylabel("Value")
        ax.set_title(f"Coverage rate — {args.env} seed {args.seed} (k_eff={k_eff})")
        ax.grid(alpha=0.3)
        ax.legend(loc="lower right")
        Path(args.out_dir).mkdir(parents=True, exist_ok=True)
        out_path = Path(args.out_dir) / f"coverage_rate_{args.env}_seed{args.seed}.png"
        fig.tight_layout()
        fig.savefig(out_path, dpi=140, bbox_inches="tight")
        print(f"Saved {out_path}")
    except ImportError:
        print("matplotlib not available, skipping plot.")


if __name__ == "__main__":
    main()
