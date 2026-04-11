#!/usr/bin/env python3
"""Empirical validation of Theorem T7 (hybrid variance decomposition).

Retrains an M=5 K=10 hybrid ensemble on one env/seed and computes, per test
point, the exact decomposition of the hybrid predictive variance into:

    Var_hyb(x) = V_within(x)   +   V_between(x)
               = mean_i[Var_k(F_{i,j}(x))]  +  Var_i[Mean_k(F_{i,j}(x))]

The two terms come from the discrete law of total variance applied to the
joint (i,j) index where i ranges over DE members and j over PnC perturbations
of each member.

Produces:
- Per-regime (id_eval, ood_near, ood_mid, ood_far) bar chart decomposing
  V_hyb into V_within + V_between.
- Scatter of V_within vs V_between colored by regime.

Usage:
    .venv/bin/python experiments/scripts/hybrid_variance_decomposition.py \
        --env Hopper-v5 --seed 0

Expected: V_within (the PnC side) should be large on OOD and small on ID;
V_between (the DE side) should be more uniform across regimes. Ratio
V_within/V_between per regime tells which mechanism dominates per env.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import jax
import jax.numpy as jnp
import numpy as np


def train_hybrid(env: str, seed: int, n_de: int, k_pnc: int, n_directions: int, scale: float):
    """Train M=n_de probabilistic bases + build per-member PJSVD ensembles."""
    from flax import nnx

    from models import ProbabilisticRegressionModel
    from training import train_probabilistic_model
    from data import load_minari_transitions
    from gym_tasks import _split_data
    from ensembles import PJSVDEnsemble

    inputs_id, targets_id, _ = load_minari_transitions(env, "id_train", 10000, seed)
    x_tr, y_tr, x_va, y_va = _split_data(inputs_id, targets_id)

    base_models = []
    for m_idx in range(n_de):
        print(f"[hvd] Training DE member {m_idx+1}/{n_de} on {env} seed {seed}...")
        m = ProbabilisticRegressionModel(
            inputs_id.shape[1], targets_id.shape[1],
            nnx.Rngs(params=seed + m_idx * 1000),
            hidden_dims=[200, 200, 200, 200],
            activation=nnx.relu,
        )
        m = train_probabilistic_model(m, x_tr, y_tr, x_va, y_va, steps=5000, batch_size=64)
        for p in jax.tree_util.tree_leaves(nnx.state(m)):
            if hasattr(p, "block_until_ready"):
                p.block_until_ready()
        base_models.append(m)

    X_sub = jnp.array(inputs_id[np.random.RandomState(seed).choice(len(inputs_id), 4096, replace=False)])
    act_fn = nnx.relu

    def _build_pjsvd_for_member(m_idx, base_m):
        Ws = [base_m.layers[i].kernel.get_value() for i in range(len(base_m.layers))]
        bs = [base_m.layers[i].bias.get_value() for i in range(len(base_m.layers))]
        n_hidden = len(Ws)
        perturb_indices = list(range(0, n_hidden, 2))
        layer_params = {f"l{i+1}": {"W": Ws[i], "b": bs[i]} for i in perturb_indices}

        layer_specs_list = []
        for li, pi in enumerate(perturb_indices):
            W_li = Ws[pi]
            D = W_li.size
            rng_li = np.random.RandomState(seed + m_idx * 1000 + li)
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

        z_coeffs = np.stack([
            np.random.RandomState(seed + m_idx * 1000 + li + 7).normal(size=(k_pnc, n_directions))
            for li in range(len(perturb_indices))
        ], axis=1)

        return PJSVDEnsemble(
            base_model=base_m,
            v_opts=np.zeros((1, 1)),
            sigmas=np.ones(1),
            z_coeffs=z_coeffs,
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

    per_member = [_build_pjsvd_for_member(i, bm) for i, bm in enumerate(base_models)]
    return per_member


def per_member_predict_np(per_member, inputs_np, chunk_size=1000):
    """Call each per-member PJSVDEnsemble on inputs in chunks, collecting
    (M, K, B, D) means and vars as numpy arrays."""
    x_j = jnp.array(inputs_np)
    per_member_means = []
    per_member_vars = []
    for ens in per_member:
        means_chunks = []
        vars_chunks = []
        for start in range(0, x_j.shape[0], chunk_size):
            p = ens.predict(x_j[start:start + chunk_size])
            if not isinstance(p, tuple):
                raise SystemExit("Expected probabilistic (means, vars) tuple")
            means_chunks.append(np.array(p[0]))  # (K, chunk, D)
            vars_chunks.append(np.array(p[1]))
        per_member_means.append(np.concatenate(means_chunks, axis=1))  # (K, B, D)
        per_member_vars.append(np.concatenate(vars_chunks, axis=1))
    return np.stack(per_member_means, axis=0), np.stack(per_member_vars, axis=0)  # (M, K, B, D)


def decompose_variance(means_mk_b_d: np.ndarray, vars_mk_b_d: np.ndarray) -> dict:
    """Given per-member per-perturbation means and vars of shape (M, K, B, D),
    compute V_within, V_between, V_hyb per test point.

    V_within(x) = E_i[Var_k(mean) + Mean_k(var)]  (average within-DE PnC variance)
    V_between(x) = Var_i[Mean_k(mean_{i,j}(x))]   (variance across DE members of per-member means)
    V_hyb(x) = V_within + V_between              (law of total variance)
    """
    # Mean over PnC (axis=1) gives (M, B, D): per-DE-member mean prediction
    per_i_mean = means_mk_b_d.mean(axis=1)  # (M, B, D)
    # Variance over PnC (axis=1) plus mean aleatoric var: within-member var
    per_i_within = means_mk_b_d.var(axis=1) + vars_mk_b_d.mean(axis=1)  # (M, B, D)
    V_within = per_i_within.mean(axis=0).mean(axis=-1)  # (B,)
    V_between = per_i_mean.var(axis=0).mean(axis=-1)    # (B,)
    V_hyb = V_within + V_between
    return {"within": V_within, "between": V_between, "hyb": V_hyb}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="Hopper-v5")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-de", type=int, default=5)
    parser.add_argument("--k-pnc", type=int, default=10)
    parser.add_argument("--n-directions", type=int, default=20)
    parser.add_argument("--scale", type=float, default=5.0)
    parser.add_argument("--out-dir", default="experiments/figures")
    args = parser.parse_args()

    from data import load_minari_transitions

    print(f"[hvd] {args.env} seed {args.seed}  M={args.n_de}  K={args.k_pnc}  scale={args.scale}")

    per_member = train_hybrid(
        args.env, args.seed, args.n_de, args.k_pnc, args.n_directions, args.scale
    )

    regimes = ["id_eval", "ood_near", "ood_mid", "ood_far"]
    regime_data = {}
    for reg in regimes:
        ins, _, _ = load_minari_transitions(args.env, reg, 10000, args.seed)
        means_mk_b_d, vars_mk_b_d = per_member_predict_np(per_member, ins)
        regime_data[reg] = decompose_variance(means_mk_b_d, vars_mk_b_d)

    print(f"\n=== Variance decomposition on {args.env} seed {args.seed} ===")
    print(f"{'regime':<12} {'V_within':>12} {'V_between':>12} {'V_hyb':>12} {'within/hyb %':>14} {'between/hyb %':>15}")
    for reg in regimes:
        d = regime_data[reg]
        vw = float(d["within"].mean())
        vb = float(d["between"].mean())
        vh = float(d["hyb"].mean())
        print(f"{reg:<12} {vw:>12.6f} {vb:>12.6f} {vh:>12.6f} {100*vw/vh:>13.2f}% {100*vb/vh:>14.2f}%")

    # Plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7, 4))
        regime_labels = ["ID eval", "Near OOD", "Mid OOD", "Far OOD"]
        within_means = [float(regime_data[r]["within"].mean()) for r in regimes]
        between_means = [float(regime_data[r]["between"].mean()) for r in regimes]

        x = np.arange(len(regimes))
        ax.bar(x, within_means, label="V_within (PnC per member)", color="#1f77b4")
        ax.bar(x, between_means, bottom=within_means, label="V_between (DE)", color="#d62728")
        ax.set_xticks(x)
        ax.set_xticklabels(regime_labels)
        ax.set_ylabel("Mean predictive variance per sample")
        ax.set_title(
            f"Hybrid variance decomposition (T7) — {args.env} s{args.seed}  M={args.n_de} K={args.k_pnc}"
        )
        ax.grid(alpha=0.3, axis="y")
        ax.legend(loc="upper left")
        fig.tight_layout()
        Path(args.out_dir).mkdir(parents=True, exist_ok=True)
        out_path = Path(args.out_dir) / f"hybrid_variance_decomposition_{args.env}_seed{args.seed}.png"
        fig.savefig(out_path, dpi=140, bbox_inches="tight")
        print(f"\nSaved {out_path}")
    except ImportError:
        print("matplotlib not available, skipping plot.")


if __name__ == "__main__":
    main()
