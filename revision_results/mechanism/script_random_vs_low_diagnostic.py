#!/usr/bin/env python3
"""Mechanistic diagnostic: Random-Proj vs Low-Proj PnC.

For a single trained MLP base model (matching the gym pipeline), build both
the Low (Lanczos bottom-eigenvector) and Random (i.i.d. Gaussian unit vector)
multi-layer PJSVD ensembles with LS correction, then compute per-ensemble:

  Layer-level (per perturbed layer):
    - sigma spectrum        -- H3
    - participation ratio of z_coeffs / sigma  -- H3
    - Delta-h (post-activation perturbation) ID / OOD  -- H2
    - Delta-z (correction-layer pre-activation after LS) ID / OOD -- H2
    - Uncorrected Delta-z (pre-activation using ORIGINAL W_next) ID / OOD -- H2
    - ||W_corr - W_corr_orig|| (correction magnitude) -- H4

  Ensemble-level:
    - predictive std ID / OOD (per-sample, averaged)  -- H1, H4
    - var_ratio (ood_far / id)  -- H1
    - effective rank of Delta-Y across members (participation ratio)  -- H5
    - median member-pair cosine similarity of Delta-Y  -- H5

Outputs ``experiments/logs/random_vs_low_diag_{env}_seed{S}_ps{P}.json`` plus
``experiments/figures/random_vs_low_diag_{env}.png`` after collating seeds.

Usage:
    .venv/bin/python experiments/scripts/random_vs_low_diagnostic.py \\
        --env Ant-v5 --seeds 0 10 42 100 200 --pert-size 50.0 --ood-suffix far
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
from flax import nnx

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pnc_core.ensembles import PJSVDEnsemble
from pnc_core.gym_tasks import _sample_member_latents
from pnc_core.models import ProbabilisticRegressionModel
from pnc_core.training import train_probabilistic_model
from pnc_core.util import _split_data

HIDDEN_DIMS = [200, 200, 200, 200]
N_PNC_MEMBERS = 50
N_PNC_DIRECTIONS = 20
TRAIN_STEPS = 5000
BATCH_SIZE = 64
STEPS_DATA = 10000
SUBSET_SIZE = 4096
FAMILIES = ("random", "low")


def _load_npz(path: Path):
    d = np.load(path)
    return jnp.array(d["inputs"]), jnp.array(d["targets"])


def load_dataset(env: str, seed: int) -> dict:
    root = Path("results") / env
    ds = {}
    for k in ["id_train", "id_eval", "ood_near", "ood_mid", "ood_far"]:
        p = root / f"data_{k}_seed{seed}_steps{STEPS_DATA}.npz"
        if p.exists():
            ds[k] = _load_npz(p)
    return ds


def train_base_model(dataset, seed: int):
    inputs_id, targets_id = dataset["id_train"]
    x_tr, y_tr, x_va, y_va = _split_data(inputs_id, targets_id)
    model = ProbabilisticRegressionModel(
        inputs_id.shape[1], targets_id.shape[1],
        nnx.Rngs(params=seed), hidden_dims=HIDDEN_DIMS, activation=nnx.relu,
    )
    model = train_probabilistic_model(
        model, x_tr, y_tr, x_va, y_va,
        steps=TRAIN_STEPS, batch_size=BATCH_SIZE,
    )
    for p in jax.tree_util.tree_leaves(nnx.state(model)):
        if hasattr(p, "block_until_ready"):
            p.block_until_ready()
    return model, inputs_id


def build_layer_specs(family: str, model, X_sub, seed: int, perturb_indices):
    """Return (layer_specs_list, perturbed_layer_names).

    family = 'random' mirrors gym_tasks.MultiLayerPJSVDExperiment for the
    ``random`` branch. family = 'low' uses pnc.find_pnc_subspace_lanczos.
    """
    from pnc_core.pnc import find_pnc_subspace_lanczos  # local to avoid import at top

    act_fn = nnx.relu
    Ws = [model.layers[i].kernel.get_value() for i in range(len(HIDDEN_DIMS))]
    bs = [model.layers[i].bias.get_value() for i in range(len(HIDDEN_DIMS))]
    W_pert_list = [Ws[pi] for pi in perturb_indices]
    b_pert_list = [bs[pi] for pi in perturb_indices]

    layer_specs_list = []
    for li, pi in enumerate(perturb_indices):
        W_li = W_pert_list[li]
        b_li = b_pert_list[li]
        if family == "random":
            D = W_li.size
            rng_li = np.random.RandomState(seed + li)
            rand_dirs = rng_li.normal(size=(N_PNC_DIRECTIONS, D)).astype(np.float32)
            rand_dirs /= np.linalg.norm(rand_dirs, axis=1, keepdims=True) + 1e-12
            v_li = np.asarray(rand_dirs)
            s_li = np.ones(N_PNC_DIRECTIONS, dtype=np.float32)
        elif family in ("low", "low_flat_sigma"):
            def _make_get_Y(layer_idx, w_list, b_list, activation):
                def get_Y_fn(w, x):
                    h = x
                    for k in range(layer_idx):
                        h = activation(h @ w_list[k] + b_list[k])
                    return activation(h @ w + b_list[layer_idx])
                return get_Y_fn
            get_Y_fn = _make_get_Y(li, W_pert_list, b_pert_list, act_fn)
            v_jax, s_jax = find_pnc_subspace_lanczos(
                get_Y_fn, W_li, [X_sub], N_PNC_DIRECTIONS,
                backend="projected_residual", seed=seed + li,
            )
            v_li = np.asarray(v_jax)
            if family == "low_flat_sigma":
                s_li = np.ones(N_PNC_DIRECTIONS, dtype=np.float32)
            else:
                s_li = np.asarray(s_jax)
        else:
            raise ValueError(family)
        layer_specs_list.append({
            "v_opts": v_li,
            "sigmas": s_li,
            "W_shape": W_li.shape,
        })
    perturbed_layers = [f"l{pi+1}" for pi in perturb_indices]
    layer_params = {f"l{pi+1}": {"W": Ws[pi], "b": bs[pi]} for pi in perturb_indices}
    return layer_specs_list, perturbed_layers, layer_params


def build_pnc_ensemble(model, dataset, family, seed, pert_size):
    inputs_id, _ = dataset["id_train"]
    perturb_indices = list(range(0, len(HIDDEN_DIMS), 2))  # [0, 2]
    np.random.seed(seed)
    subset_idx = np.random.choice(len(inputs_id), min(len(inputs_id), SUBSET_SIZE), replace=False)
    X_sub = inputs_id[subset_idx]

    layer_specs_list, perturbed_layers, layer_params = build_layer_specs(
        family, model, X_sub, seed, perturb_indices
    )

    # Per-layer latents, mirroring gym_tasks
    all_z = np.stack([
        _sample_member_latents(
            np.random.RandomState(seed + li),
            N_PNC_MEMBERS, N_PNC_DIRECTIONS, antithetic_pairing=False,
        ) for li in range(len(perturb_indices))
    ], axis=1)

    Ws = [model.layers[i].kernel.get_value() for i in range(len(HIDDEN_DIMS))]
    bs = [model.layers[i].bias.get_value() for i in range(len(HIDDEN_DIMS))]
    h_old = X_sub
    for pi in perturb_indices:
        h_old = nnx.relu(h_old @ Ws[pi] + bs[pi])
    correction_params = {"target_act": h_old}

    ens = PJSVDEnsemble(
        base_model=model,
        v_opts=np.zeros((1, 1)),
        sigmas=np.ones(1),
        z_coeffs=all_z,
        perturbation_scale=pert_size,
        X_sub=X_sub,
        layers=perturbed_layers,
        correction_mode="least_squares",
        activation=nnx.relu,
        layer_params=layer_params,
        correction_params=correction_params,
        tail_is_hidden=True,
        layer_specs=layer_specs_list,
    )
    return ens, layer_specs_list, perturb_indices, X_sub


def participation_ratio(values: np.ndarray, axis=None) -> float:
    """(Σ v²)² / Σ v⁴  — effective number of non-zero components."""
    v2 = np.square(np.asarray(values, dtype=np.float64))
    s1 = v2.sum(axis=axis)
    s2 = (v2 ** 2).sum(axis=axis)
    return float(s1 ** 2 / (s2 + 1e-30))


def compute_ensemble_metrics(ens: PJSVDEnsemble, x: jax.Array):
    """Predictive mean/var and ensemble std for a PJSVD ensemble.

    Returns dict with keys:
        pred_std_mean — average (over samples) L2 norm of per-output ensemble std
        var_mean      — average (over samples) of mean-variance output
        deltaY        — (n_members, batch, dim) difference from per-member mean
        y_base        — the base model output mean
    """
    # Base model mean
    out = ens.base_model(x)
    y_base = out[0] if isinstance(out, tuple) else out
    y_members = []
    for i in range(ens._n_members):
        yi, vi = ens.predict_one(x, i)
        y_members.append(np.asarray(yi))
    y_members = np.stack(y_members, axis=0)  # (N, B, D)
    # Ensemble std across members
    per_sample_std = y_members.std(axis=0)  # (B, D)
    per_sample_l2 = np.linalg.norm(per_sample_std, axis=-1)  # (B,)
    return {
        "pred_std_mean": float(per_sample_l2.mean()),
        "pred_std_median": float(np.median(per_sample_l2)),
        "y_base": np.asarray(y_base),
        "y_members": y_members,
    }


def output_rank_diagnostics(y_members: np.ndarray, y_base: np.ndarray):
    """Return (effective_rank, median_cosine) on Delta-Y across members.

    effective_rank = participation ratio of singular values of the stacked
    member-difference matrix.
    """
    N = y_members.shape[0]
    dy = (y_members - y_base[None]).reshape(N, -1)  # (N, B*D)
    # SVD on dy; singular values give per-mode variance.
    try:
        s = np.linalg.svd(dy, compute_uv=False)
    except Exception:
        return float("nan"), float("nan")
    eff = participation_ratio(s)

    # Median pairwise cosine sim
    norms = np.linalg.norm(dy, axis=1, keepdims=True) + 1e-12
    dy_hat = dy / norms
    C = dy_hat @ dy_hat.T
    iu = np.triu_indices(N, k=1)
    cos = C[iu]
    median_cos = float(np.median(cos))
    return eff, median_cos


def per_layer_perturbation_diagnostics(
    ens: PJSVDEnsemble,
    layer_specs,
    perturb_indices,
    x_id: jax.Array,
    x_ood: jax.Array,
) -> dict:
    """Compute Delta-h, Delta-z (uncorrected and corrected) at each perturbed layer."""
    Ws = [ens.base_model.layers[i].kernel.get_value() for i in range(len(HIDDEN_DIMS))]
    bs = [ens.base_model.layers[i].bias.get_value() for i in range(len(HIDDEN_DIMS))]
    act = nnx.relu
    diag = {"per_layer": []}
    for li, pi in enumerate(perturb_indices):
        # Recover dW per member at this layer from ens.seq_dWs[li]
        dWs = np.asarray(ens.seq_dWs[li])  # (N, *W_shape)
        # Compute base h prior to this layer, on ID and OOD
        for tag, x in (("id", x_id), ("ood", x_ood)):
            h_in = x
            for k in range(pi):
                h_in = act(h_in @ Ws[k] + bs[k])
            h_base = act(h_in @ Ws[pi] + bs[pi])  # (B, H)

            # Per-member perturbed h
            h_perts = []
            for i in range(ens._n_members):
                h_pert = act(h_in @ (Ws[pi] + dWs[i]) + bs[pi])  # (B, H)
                h_perts.append(np.asarray(h_pert))
            h_perts = np.stack(h_perts, axis=0)  # (N, B, H)
            # Delta-h norms
            dh = np.linalg.norm(h_perts - np.asarray(h_base)[None], axis=-1)  # (N, B)

            # Delta-z uncorrected (use original W_next)
            corr_idx = pi + 1
            W_corr_orig = np.asarray(Ws[corr_idx])
            b_corr_orig = np.asarray(bs[corr_idx])
            z_perts_uncorr = h_perts @ W_corr_orig + b_corr_orig
            z_base = np.asarray(h_base) @ W_corr_orig + b_corr_orig
            dz_uncorr = np.linalg.norm(z_perts_uncorr - z_base[None], axis=-1)  # (N, B)

            # Delta-z corrected (use per-member W_corr from LS)
            W_corrs = np.asarray(ens.seq_w_effs[li])  # (N, H, H)
            b_corrs = np.asarray(ens.seq_b_effs[li])  # (N, H)
            z_perts_corr = np.einsum("nbh,nhk->nbk", h_perts, W_corrs) + b_corrs[:, None, :]
            dz_corr = np.linalg.norm(z_perts_corr - z_base[None], axis=-1)  # (N, B)

            entry = {
                "layer_idx": int(pi),
                "tag": tag,
                "dh_mean": float(dh.mean()),
                "dh_median": float(np.median(dh)),
                "dz_uncorr_mean": float(dz_uncorr.mean()),
                "dz_uncorr_median": float(np.median(dz_uncorr)),
                "dz_corr_mean": float(dz_corr.mean()),
                "dz_corr_median": float(np.median(dz_corr)),
            }

            # Only record σ / W-correction magnitude once (independent of tag)
            if tag == "id":
                sigmas = np.asarray(layer_specs[li]["sigmas"])
                entry["sigma_min"] = float(sigmas.min())
                entry["sigma_max"] = float(sigmas.max())
                entry["sigma_ratio"] = float(sigmas.max() / (sigmas.min() + 1e-30))
                entry["sigma_participation"] = float(participation_ratio(sigmas))

                # Participation ratio of z_coeffs / sigma across 20 dims, averaged over members
                # ens.z_coeffs has shape (N, n_layers, K) when layer_specs is set
                z = np.asarray(ens.z_coeffs[:, li, :])  # (N, K)
                safe_s = sigmas + 1e-6
                c = z / safe_s
                # Normalise per-member
                c_norm = c / (np.linalg.norm(c, axis=1, keepdims=True) + 1e-12)
                # Per-member participation ratio of |c_norm|^2 coefficients
                prs = [participation_ratio(row) for row in c_norm]
                entry["coeff_participation_mean"] = float(np.mean(prs))
                entry["coeff_participation_std"] = float(np.std(prs))

                # Correction magnitude: ||W_corr - W_corr_orig|| per member
                dW_corr = W_corrs - W_corr_orig[None]
                dW_norms = np.linalg.norm(dW_corr.reshape(dW_corr.shape[0], -1), axis=1)
                entry["corr_dW_mean"] = float(dW_norms.mean())
                entry["corr_dW_median"] = float(np.median(dW_norms))
                db_corrs = b_corrs - b_corr_orig[None]
                entry["corr_db_mean"] = float(np.linalg.norm(db_corrs, axis=1).mean())

            diag["per_layer"].append(entry)
    return diag


def run_one(env: str, seed: int, pert_size: float, ood_suffix: str, out_dir: Path):
    print(f"\n[{env} seed {seed} ps {pert_size}] ==== loading ====")
    dataset = load_dataset(env, seed)
    if "id_train" not in dataset or f"ood_{ood_suffix}" not in dataset:
        print(f"!! missing data, skipping seed {seed}")
        return None

    print(f"[{env} seed {seed}] training base model")
    t0 = time.time()
    model, inputs_id = train_base_model(dataset, seed)
    print(f"  train time: {time.time() - t0:.1f}s")

    x_id_all, _ = dataset["id_eval"]
    x_ood_all, _ = dataset[f"ood_{ood_suffix}"]
    # Use up to 2000 points each for diagnostics (speed)
    x_id = x_id_all[:2000]
    x_ood = x_ood_all[:2000]

    out = {
        "env": env, "seed": seed, "pert_size": pert_size,
        "ood_suffix": ood_suffix, "hidden_dims": HIDDEN_DIMS,
        "n_members": N_PNC_MEMBERS, "n_directions": N_PNC_DIRECTIONS,
    }
    for family in FAMILIES:
        t0 = time.time()
        print(f"\n[{env} seed {seed}] building {family} ensemble")
        ens, layer_specs, perturb_indices, _ = build_pnc_ensemble(
            model, dataset, family, seed, pert_size
        )
        print(f"  build time: {time.time() - t0:.1f}s")
        per_layer = per_layer_perturbation_diagnostics(
            ens, layer_specs, perturb_indices, x_id, x_ood,
        )
        ens_id = compute_ensemble_metrics(ens, x_id)
        ens_ood = compute_ensemble_metrics(ens, x_ood)
        eff_rank_id, med_cos_id = output_rank_diagnostics(ens_id["y_members"], ens_id["y_base"])
        eff_rank_ood, med_cos_ood = output_rank_diagnostics(ens_ood["y_members"], ens_ood["y_base"])
        out[family] = {
            "per_layer": per_layer["per_layer"],
            "pred_std_id": ens_id["pred_std_mean"],
            "pred_std_ood": ens_ood["pred_std_mean"],
            "pred_std_ratio": ens_ood["pred_std_mean"] / (ens_id["pred_std_mean"] + 1e-12),
            "eff_rank_id": eff_rank_id,
            "eff_rank_ood": eff_rank_ood,
            "median_cos_id": med_cos_id,
            "median_cos_ood": med_cos_ood,
        }
        # Release jax buffers
        del ens
    fam_tag = "+".join(FAMILIES) if FAMILIES != ("random", "low") else ""
    suffix = f"_{fam_tag}" if fam_tag else ""
    out_path = out_dir / f"random_vs_low_diag_{env}_seed{seed}_ps{pert_size}{suffix}.json"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"  wrote {out_path}")
    return out


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--env", default="Ant-v5")
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--pert-size", type=float, default=50.0)
    p.add_argument("--ood-suffix", default="far", choices=["near", "mid", "far"])
    p.add_argument("--out-dir", type=Path,
                   default=Path("experiments/logs/random_vs_low_diag"))
    p.add_argument("--families", nargs="+",
                   default=["random", "low"],
                   choices=["random", "low", "low_flat_sigma"])
    args = p.parse_args(argv)
    global FAMILIES
    FAMILIES = tuple(args.families)
    for seed in args.seeds:
        run_one(args.env, seed, args.pert_size, args.ood_suffix, args.out_dir)


if __name__ == "__main__":
    main()
