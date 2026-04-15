#!/usr/bin/env python3
"""
Analyze why PnC can have lower ID RMSE than SWAG/Subspace despite much larger
hidden perturbations.  Produces paper-usable figures and diagnostics.

Diagnostics:
  A. Effective projected-residual score  s_eff = sqrt(1 - alpha)
  B. Alignment with PnC perturbation subspace  alpha = ||proj||^2 / ||delta||^2
  C. Output sensitivity per unit hidden perturbation  gamma = ||dy|| / ||dh||

Usage:
    .venv/bin/python experiments/scripts/analyze_perturbation_subspaces.py \
        --env Ant-v5 --seeds 0,10,42,100,200 --out-dir experiments/figures
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from flax import nnx

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ensembles import (
    PJSVDEnsemble,
    SWAGEnsemble,
    SubspaceInferenceEnsemble,
    LaplaceEnsemble,
    _sample_member_radius_multipliers,
    _scale_coefficients_with_member_radii,
)
from gym_tasks import _sample_member_latents
from laplace import compute_kfac_factors
from models import ProbabilisticRegressionModel
from training import train_probabilistic_model, train_swag_model, train_subspace_model
from util import seed_everything, _split_data, get_intermediate_state

# ---------------------------------------------------------------------------
HIDDEN_DIMS = [200, 200, 200, 200]
N_PNC_MEMBERS = 50
N_BASELINE_MEMBERS = 100
N_PNC_DIRECTIONS = 20
PNC_PERT_SIZES = [5.0, 10.0, 20.0, 50.0]
STEPS = 10000


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--env", default="Ant-v5")
    p.add_argument("--seeds", default="0,10,42,100,200")
    p.add_argument("--pnc-size", type=float, default=None,
                   help="PnC perturbation size to analyse (default: best by nll_val)")
    p.add_argument("--out-dir", default="experiments/figures")
    return p.parse_args()


# ── data loading ──────────────────────────────────────────────────────────

def _load_npz(path: str):
    d = np.load(path)
    return jnp.array(d["inputs"]), jnp.array(d["targets"])


def load_dataset(env: str, seed: int) -> dict:
    root = Path("results") / env
    ds = {}
    for k in ["id_train", "id_eval", "ood_far"]:
        p = root / f"data_{k}_seed{seed}_steps{STEPS}.npz"
        if p.exists():
            ds[k] = _load_npz(str(p))
    return ds


# ── model / ensemble builders ─────────────────────────────────────────────

def _build_base_model(dataset, seed):
    inputs_id, targets_id = dataset["id_train"]
    x_tr, y_tr, x_va, y_va = _split_data(inputs_id, targets_id)
    model = ProbabilisticRegressionModel(
        inputs_id.shape[1], targets_id.shape[1],
        nnx.Rngs(params=seed), hidden_dims=HIDDEN_DIMS, activation=nnx.relu,
    )
    model = train_probabilistic_model(model, x_tr, y_tr, x_va, y_va,
                                      steps=5000, batch_size=64)
    for p in jax.tree_util.tree_leaves(nnx.state(model)):
        if hasattr(p, "block_until_ready"):
            p.block_until_ready()
    return model, x_tr, y_tr, x_va, y_va


def _build_pnc_ensemble(model, dataset, seed, correction_mode, pert_size):
    """Build a multi-layer PnC ensemble matching the experiment config."""
    inputs_id, _ = dataset["id_train"]
    n_hidden = len(HIDDEN_DIMS)
    perturb_indices = list(range(0, n_hidden, 2))       # [0, 2]
    corr_layer_idx = perturb_indices[-1] + 1             # 3

    Ws = [model.layers[i].kernel.get_value() for i in range(n_hidden)]
    bs = [model.layers[i].bias.get_value() for i in range(n_hidden)]

    actual_subset = min(len(inputs_id), 4096)
    np.random.seed(seed)  # match gym_tasks RNG state
    subset_idx = np.random.choice(len(inputs_id), actual_subset, replace=False)
    X_sub = inputs_id[subset_idx]

    # Build per-layer specs with random directions (same RNG as gym_tasks)
    layer_specs_list = []
    perturbed_layers = [f"l{i+1}" for i in perturb_indices]
    layer_params = {f"l{i+1}": {"W": Ws[i], "b": bs[i]} for i in perturb_indices}

    for li, pi in enumerate(perturb_indices):
        D = Ws[pi].size
        rng_li = np.random.RandomState(seed + li)
        rand_dirs = rng_li.normal(size=(N_PNC_DIRECTIONS, D)).astype(np.float32)
        rand_dirs /= np.linalg.norm(rand_dirs, axis=1, keepdims=True) + 1e-12
        layer_specs_list.append({
            "v_opts": rand_dirs,
            "sigmas": np.ones(N_PNC_DIRECTIONS, dtype=np.float32),
            "W_shape": Ws[pi].shape,
        })

    all_z = np.stack([
        _sample_member_latents(
            np.random.RandomState(seed + li),
            N_PNC_MEMBERS, N_PNC_DIRECTIONS, antithetic_pairing=False,
        ) for li in range(len(perturb_indices))
    ], axis=1)

    h_old = X_sub
    for idx, pi in enumerate(perturb_indices):
        h_old = nnx.relu(h_old @ Ws[pi] + bs[pi])

    if correction_mode == "least_squares":
        correction_params = {"target_act": h_old}
    else:
        correction_params = {"target_act": h_old}

    ens = PJSVDEnsemble(
        base_model=model,
        v_opts=np.zeros((1, 1)),
        sigmas=np.ones(1),
        z_coeffs=all_z,
        perturbation_scale=pert_size,
        X_sub=X_sub,
        layers=perturbed_layers,
        correction_mode=correction_mode,
        activation=nnx.relu,
        layer_params=layer_params,
        correction_params=correction_params,
        tail_is_hidden=True,
        layer_specs=layer_specs_list,
    )
    return ens, layer_specs_list, perturb_indices, Ws, bs


def _build_swag_ensemble(dataset, seed):
    inputs_id, targets_id = dataset["id_train"]
    x_tr, y_tr, x_va, y_va = _split_data(inputs_id, targets_id)
    model = ProbabilisticRegressionModel(
        inputs_id.shape[1], targets_id.shape[1],
        nnx.Rngs(params=seed), hidden_dims=HIDDEN_DIMS, activation=nnx.relu,
    )
    model, swag_mean, swag_var = train_swag_model(
        model, x_tr, y_tr, x_va, y_va, steps=5000, batch_size=64, swag_start=1000,
    )
    jax.block_until_ready(swag_mean)
    jax.block_until_ready(swag_var)
    ens = SWAGEnsemble(model, swag_mean, swag_var, N_BASELINE_MEMBERS)
    return model, ens


def _build_subspace_ensemble(dataset, seed):
    inputs_id, targets_id = dataset["id_train"]
    x_tr, y_tr, x_va, y_va = _split_data(inputs_id, targets_id)
    model = ProbabilisticRegressionModel(
        inputs_id.shape[1], targets_id.shape[1],
        nnx.Rngs(params=seed), hidden_dims=HIDDEN_DIMS, activation=nnx.relu,
    )
    model, swag_mean, pca_components = train_subspace_model(
        model, x_tr, y_tr, x_va, y_va,
        steps=2000, batch_size=64, swag_start=1000, max_rank=20,
    )
    jax.block_until_ready(swag_mean)
    T = 2.0 * float(np.sqrt(len(inputs_id)))
    ens = SubspaceInferenceEnsemble(
        model, swag_mean, pca_components, N_BASELINE_MEMBERS,
        temperature=T, X_train=inputs_id, Y_train=targets_id,
        use_ess=True, is_classification=False,
    )
    return model, ens


def _build_laplace_ensemble(dataset, seed, prior=1000.0):
    inputs_id, targets_id = dataset["id_train"]
    x_tr, y_tr, x_va, y_va = _split_data(inputs_id, targets_id)
    model = ProbabilisticRegressionModel(
        inputs_id.shape[1], targets_id.shape[1],
        nnx.Rngs(params=seed), hidden_dims=HIDDEN_DIMS, activation=nnx.relu,
    )
    model = train_probabilistic_model(model, x_tr, y_tr, x_va, y_va,
                                      steps=5000, batch_size=64)
    for p in jax.tree_util.tree_leaves(nnx.state(model)):
        if hasattr(p, "block_until_ready"):
            p.block_until_ready()
    actual_subset = min(len(inputs_id), 4096)
    np.random.seed(seed)
    subset_idx = np.random.choice(len(inputs_id), actual_subset, replace=False)
    X_sub = inputs_id[subset_idx]
    Y_sub = targets_id[subset_idx]
    factors = compute_kfac_factors(model, X_sub, Y_sub, batch_size=128)
    for f in jax.tree_util.tree_leaves(factors):
        if hasattr(f, "block_until_ready"):
            f.block_until_ready()
    ens = LaplaceEnsemble(
        model=model, kfac_factors=factors, prior_precision=prior,
        n_models=N_BASELINE_MEMBERS, data_size=actual_subset,
    )
    return model, ens


# ── Diagnostic helpers ────────────────────────────────────────────────────

def _compute_pnc_h_at_layer1(base_model, pnc_ens, x):
    """Compute per-member hidden state at layer 1 for PnC by replaying perturbation."""
    W1 = base_model.layers[0].kernel.get_value()
    b1 = base_model.layers[0].bias.get_value()
    h_base = nnx.relu(x @ W1 + b1)
    hs = []
    for i in range(pnc_ens._n_members):
        dW = pnc_ens.seq_dWs[0][i]  # layer 0 (l1) perturbation for member i
        h_pert = nnx.relu(x @ (W1 + dW) + b1)
        hs.append(h_pert)
    return h_base, jnp.stack(hs, axis=0)


def _extract_diagnostics(base_model, ensemble, x, layer_idx, method):
    """Extract per-member hidden states at layer_idx, predictions, and raw dy/dh."""
    h_base = get_intermediate_state(base_model, x, layer_idx)
    y_base = base_model(x)
    if isinstance(y_base, tuple):
        y_base = y_base[0]

    h_members = ensemble.predict_intermediate(x, layer_idx=layer_idx)
    y_members = ensemble.predict(x)
    if isinstance(y_members, tuple):
        y_members = y_members[0]

    return h_base, y_base, h_members, y_members


def compute_gamma_and_dy(h_base, y_base, h_members, y_members):
    """Output sensitivity gamma_i and absolute prediction change dy_i per member."""
    n_members = h_members.shape[0]
    gammas = []
    dys = []
    dhs = []
    for i in range(n_members):
        dh = float(jnp.mean(jnp.sqrt(jnp.sum((h_members[i] - h_base) ** 2, axis=-1))))
        dy = float(jnp.mean(jnp.sqrt(jnp.sum((y_members[i] - y_base) ** 2, axis=-1))))
        dhs.append(dh)
        dys.append(dy)
        if dh > 1e-12:
            gammas.append(dy / dh)
    return np.array(gammas), np.array(dys), np.array(dhs)


def _extract_weight_deltas(base_model, ensemble, perturb_indices, method):
    """Extract per-member weight deltas at the perturbed layers.

    Returns list of arrays, one per member. Each array is the flattened
    concatenation of ΔW at the perturbed layer indices.
    """
    # Store original weights
    orig_Ws = [np.array(base_model.layers[pi].kernel.get_value()) for pi in perturb_indices]

    deltas = []
    if method == "swag":
        for _ in range(ensemble.n_models):
            sampled_m = ensemble._sample_model()
            delta_flat = []
            for j, pi in enumerate(perturb_indices):
                W_sampled = np.array(sampled_m.layers[pi].kernel.get_value())
                delta_flat.append((W_sampled - orig_Ws[j]).flatten())
            deltas.append(np.concatenate(delta_flat))
    elif method == "subspace":
        for idx in range(ensemble.n_samples):
            ensemble._sample_model(idx)
            delta_flat = []
            for j, pi in enumerate(perturb_indices):
                W_sampled = np.array(ensemble.base_model.layers[pi].kernel.get_value())
                delta_flat.append((W_sampled - orig_Ws[j]).flatten())
            deltas.append(np.concatenate(delta_flat))
    elif method == "laplace":
        for _ in range(ensemble.n_models):
            ensemble._sample_model()
            delta_flat = []
            for j, pi in enumerate(perturb_indices):
                W_sampled = np.array(ensemble.model.layers[pi].kernel.get_value())
                delta_flat.append((W_sampled - orig_Ws[j]).flatten())
            deltas.append(np.concatenate(delta_flat))
    return np.array(deltas)  # (n_members, D_total)


def _pnc_weight_deltas(pnc_ens, layer_specs, perturb_indices):
    """Extract per-member weight deltas for PnC from pre-computed dWs."""
    deltas = []
    for i in range(pnc_ens._n_members):
        delta_flat = []
        for j in range(len(perturb_indices)):
            dW = np.array(pnc_ens.seq_dWs[j][i]).flatten()
            delta_flat.append(dW)
        deltas.append(np.concatenate(delta_flat))
    return np.array(deltas)


def diagnostic_b(pnc_directions, weight_deltas):
    """Subspace alignment: fraction of delta that lies in PnC direction span.

    pnc_directions: (k_total, D_total) where rows are (possibly non-orthogonal)
                    unit directions for all perturbed layers concatenated.
    weight_deltas:  (n_members, D_total)
    """
    # Orthogonalise via QR
    Q, _ = np.linalg.qr(pnc_directions.T)  # (D_total, k_total)
    alphas = []
    for i in range(weight_deltas.shape[0]):
        d = weight_deltas[i]
        norm_d = np.linalg.norm(d)
        if norm_d < 1e-12:
            continue
        proj = Q @ (Q.T @ d)
        alphas.append(float(np.linalg.norm(proj) ** 2 / norm_d ** 2))
    return np.array(alphas)


# ── Best perturbation size selection ──────────────────────────────────────

def _select_pnc_size(env: str, seed: int, correction_mode: str,
                      forced_size: float | None) -> float:
    """Pick the best perturbation size the same way the report does (nll_val)."""
    if forced_size is not None:
        return forced_size
    stem = f"pjsvd_multi_{correction_mode}_random_projected_residual"
    ps_str = "-".join(str(s) for s in PNC_PERT_SIZES)
    fname = (f"{stem}_prob_k{N_PNC_DIRECTIONS}_n{N_PNC_MEMBERS}_ps{ps_str}"
             f"_h200-200-200-200_act-relu_seed{seed}.json")
    p = Path("results") / env / fname
    if not p.exists():
        return PNC_PERT_SIZES[0]  # fallback
    data = json.loads(p.read_text())
    best_key, best_nll = None, float("inf")
    for k, v in data.items():
        nll = v.get("nll_val", v.get("nll_id", float("inf")))
        if nll < best_nll:
            best_nll = nll
            best_key = k
    return float(best_key) if best_key else PNC_PERT_SIZES[0]


# ── Per-seed analysis ─────────────────────────────────────────────────────

def analyze_seed(env: str, seed: int, pnc_size_override: float | None):
    print(f"\n{'='*60}")
    print(f"  {env}  seed={seed}")
    print(f"{'='*60}")

    seed_everything(seed)
    dataset = load_dataset(env, seed)
    if "id_train" not in dataset or "id_eval" not in dataset:
        print(f"  Skipping seed {seed}: data not found")
        return None

    x_eval, _ = dataset["id_eval"]

    # Use a manageable eval subset
    n_eval = min(2000, x_eval.shape[0])
    x_eval = x_eval[:n_eval]

    # Measure hidden perturbation at layer 1 (matching report's Unc-L2-h)
    layer_idx = 1

    # ── 1. Base model + PnC ensembles ────────────────────────────────────
    t0 = time.time()
    base_model, x_tr, y_tr, x_va, y_va = _build_base_model(dataset, seed)
    print(f"  Base model trained ({time.time()-t0:.1f}s)")

    pnc_ls_size = _select_pnc_size(env, seed, "least_squares", pnc_size_override)
    pnc_none_size = _select_pnc_size(env, seed, "none", pnc_size_override)
    print(f"  PnC sizes: LS={pnc_ls_size}, none={pnc_none_size}")

    pnc_ls_ens, layer_specs, perturb_indices, Ws, bs = _build_pnc_ensemble(
        base_model, dataset, seed, "least_squares", pnc_ls_size)
    pnc_none_ens, _, _, _, _ = _build_pnc_ensemble(
        base_model, dataset, seed, "none", pnc_none_size)
    print(f"  PnC ensembles built ({time.time()-t0:.1f}s)")

    # ── 2. SWAG ──────────────────────────────────────────────────────────
    t1 = time.time()
    swag_base, swag_ens = _build_swag_ensemble(dataset, seed)
    print(f"  SWAG trained ({time.time()-t1:.1f}s)")

    # ── 3. Subspace ──────────────────────────────────────────────────────
    t2 = time.time()
    sub_base, sub_ens = _build_subspace_ensemble(dataset, seed)
    print(f"  Subspace trained ({time.time()-t2:.1f}s)")

    # ── 4. Laplace ───────────────────────────────────────────────────────
    t3 = time.time()
    lap_base, lap_ens = _build_laplace_ensemble(dataset, seed)
    print(f"  Laplace trained ({time.time()-t3:.1f}s)")

    # ── Diagnostic C on ID and OOD-far ─────────────────────────────────────

    def _run_diagnostic_c(x_data, label):
        """Compute gamma/dy/dh for all methods on a given data split."""
        res = {}

        y_b = base_model(x_data)
        if isinstance(y_b, tuple):
            y_b = y_b[0]

        # PnC LS
        h_b_pnc, h_m_ls = _compute_pnc_h_at_layer1(base_model, pnc_ls_ens, x_data)
        y_m_ls = pnc_ls_ens.predict(x_data)
        if isinstance(y_m_ls, tuple):
            y_m_ls = y_m_ls[0]
        g, dy, dh = compute_gamma_and_dy(h_b_pnc, y_b, h_m_ls, y_m_ls)
        res["PnC (LS corr.)"] = {"gamma": g, "dy": dy, "dh": dh}

        # PnC none
        _, h_m_none = _compute_pnc_h_at_layer1(base_model, pnc_none_ens, x_data)
        y_m_none = pnc_none_ens.predict(x_data)
        if isinstance(y_m_none, tuple):
            y_m_none = y_m_none[0]
        g, dy, dh = compute_gamma_and_dy(h_b_pnc, y_b, h_m_none, y_m_none)
        res["PnC (no corr.)"] = {"gamma": g, "dy": dy, "dh": dh}

        # SWAG
        h_b3, _, h_m3, _ = _extract_diagnostics(swag_base, swag_ens, x_data, layer_idx, "swag")
        y_swag_b = swag_base(x_data)
        if isinstance(y_swag_b, tuple):
            y_swag_b = y_swag_b[0]
        y_m_swag = swag_ens.predict(x_data)
        if isinstance(y_m_swag, tuple):
            y_m_swag = y_m_swag[0]
        g, dy, dh = compute_gamma_and_dy(h_b3, y_swag_b, h_m3, y_m_swag)
        res["SWAG"] = {"gamma": g, "dy": dy, "dh": dh}

        # Subspace
        h_b4, _, h_m4, _ = _extract_diagnostics(sub_base, sub_ens, x_data, layer_idx, "subspace")
        y_sub_b = sub_base(x_data)
        if isinstance(y_sub_b, tuple):
            y_sub_b = y_sub_b[0]
        y_m_sub = sub_ens.predict(x_data)
        if isinstance(y_m_sub, tuple):
            y_m_sub = y_m_sub[0]
        g, dy, dh = compute_gamma_and_dy(h_b4, y_sub_b, h_m4, y_m_sub)
        res["Subspace"] = {"gamma": g, "dy": dy, "dh": dh}

        # Laplace
        h_b5, _, h_m5, _ = _extract_diagnostics(lap_base, lap_ens, x_data, layer_idx, "laplace")
        y_lap_b = lap_base(x_data)
        if isinstance(y_lap_b, tuple):
            y_lap_b = y_lap_b[0]
        y_m_lap = lap_ens.predict(x_data)
        if isinstance(y_m_lap, tuple):
            y_m_lap = y_m_lap[0]
        g, dy, dh = compute_gamma_and_dy(h_b5, y_lap_b, h_m5, y_m_lap)
        res["Laplace"] = {"gamma": g, "dy": dy, "dh": dh}

        print(f"  Diagnostic C ({label}) done")
        return res

    results_id = _run_diagnostic_c(x_eval, "ID")

    # OOD-far
    results_ood = None
    if "ood_far" in dataset:
        x_ood, _ = dataset["ood_far"]
        n_ood = min(2000, x_ood.shape[0])
        x_ood = x_ood[:n_ood]
        results_ood = _run_diagnostic_c(x_ood, "OOD-far")

    # Merge into a single results dict with region prefixes
    results = {}
    for m in results_id:
        results[m] = {}
        for k, v in results_id[m].items():
            results[m][k] = v                 # ID (default)
            results[m][f"{k}_id"] = v
        if results_ood and m in results_ood:
            for k, v in results_ood[m].items():
                results[m][f"{k}_ood"] = v

    print("  Diagnostic C done (ID + OOD-far)")

    # ── Diagnostics A & B: weight-space alignment ────────────────────────
    # Build the full direction matrix for PnC (all perturbed layers concatenated)
    all_dirs = []
    for j, pi in enumerate(perturb_indices):
        v = layer_specs[j]["v_opts"]  # (k, D_layer)
        # Embed in the concatenated space
        offsets = [Ws[perturb_indices[jj]].size for jj in range(j)]
        total_offset = sum(offsets)
        total_D = sum(Ws[perturb_indices[jj]].size for jj in range(len(perturb_indices)))
        for row in range(v.shape[0]):
            full_row = np.zeros(total_D, dtype=np.float32)
            full_row[total_offset:total_offset + v.shape[1]] = v[row]
            all_dirs.append(full_row)
    pnc_dirs = np.array(all_dirs)  # (k_total, D_total)

    # PnC weight deltas
    pnc_ls_deltas = _pnc_weight_deltas(pnc_ls_ens, layer_specs, perturb_indices)
    pnc_none_deltas = _pnc_weight_deltas(pnc_none_ens, layer_specs, perturb_indices)

    results["PnC (LS corr.)"]["alpha"] = diagnostic_b(pnc_dirs, pnc_ls_deltas)
    results["PnC (no corr.)"]["alpha"] = diagnostic_b(pnc_dirs, pnc_none_deltas)

    # SWAG weight deltas at the same layers
    swag_deltas = _extract_weight_deltas(swag_base, swag_ens, perturb_indices, "swag")
    results["SWAG"]["alpha"] = diagnostic_b(pnc_dirs, swag_deltas)

    # Subspace weight deltas
    sub_deltas = _extract_weight_deltas(sub_base, sub_ens, perturb_indices, "subspace")
    results["Subspace"]["alpha"] = diagnostic_b(pnc_dirs, sub_deltas)

    # Laplace weight deltas
    lap_deltas = _extract_weight_deltas(lap_base, lap_ens, perturb_indices, "laplace")
    results["Laplace"]["alpha"] = diagnostic_b(pnc_dirs, lap_deltas)

    print("  Diagnostics A & B done")

    # Derived scalars for aggregation
    for m in results:
        results[m]["h_l2"] = float(np.median(results[m]["dh"]))
        results[m]["dy_median"] = float(np.median(results[m]["dy"]))
        if "dh_ood" in results[m]:
            results[m]["h_l2_ood"] = float(np.median(results[m]["dh_ood"]))
            results[m]["dy_median_ood"] = float(np.median(results[m]["dy_ood"]))
            results[m]["gamma_ood"] = results[m].get("gamma_ood", np.array([]))

    return results


# ── Aggregation ───────────────────────────────────────────────────────────

def aggregate_seeds(all_results):
    """Aggregate per-seed diagnostics into median + IQR."""
    methods = list(all_results[0].keys())
    agg = {}
    for m in methods:
        agg[m] = {}
        scalar_keys = ["gamma", "alpha", "h_l2", "dy_median",
                        "gamma_ood", "h_l2_ood", "dy_median_ood"]
        for key in scalar_keys:
            vals = []
            for r in all_results:
                if m in r and key in r[m]:
                    v = r[m][key]
                    if isinstance(v, np.ndarray):
                        vals.append(float(np.median(v)))
                    else:
                        vals.append(float(v))
            if vals:
                agg[m][key] = {
                    "median": float(np.median(vals)),
                    "q25": float(np.percentile(vals, 25)),
                    "q75": float(np.percentile(vals, 75)),
                    "values": vals,
                }
        # Collect per-member distributions from all seeds
        array_keys = ["gamma", "alpha", "dy", "dh",
                      "gamma_ood", "dy_ood", "dh_ood"]
        for key in array_keys:
            all_members = []
            for r in all_results:
                if m in r and key in r[m] and isinstance(r[m][key], np.ndarray):
                    all_members.extend(r[m][key].tolist())
            if all_members:
                agg[m][f"{key}_all"] = np.array(all_members)
    return agg


# ── Plotting ──────────────────────────────────────────────────────────────

METHOD_ORDER = ["Subspace", "SWAG", "Laplace", "PnC (no corr.)", "PnC (LS corr.)"]
METHOD_COLORS = {
    "Subspace": "#2ca02c",
    "SWAG": "#1f77b4",
    "Laplace": "#9467bd",
    "PnC (no corr.)": "#ff7f0e",
    "PnC (LS corr.)": "#d62728",
}


def _plot_bar_and_boxplot(axes_row, agg, methods, colors, region_tag, region_label, env):
    """Plot dh/dy bars and gamma boxplot on a pair of axes."""
    x_pos = np.arange(len(methods))
    h_key = "h_l2" if region_tag == "" else f"h_l2_{region_tag}"
    dy_key = "dy_median" if region_tag == "" else f"dy_median_{region_tag}"
    gamma_key = "gamma_all" if region_tag == "" else f"gamma_{region_tag}_all"

    # bars
    ax = axes_row[0]
    h_l2s = [agg[m].get(h_key, {}).get("median", 0) for m in methods]
    dys = [agg[m].get(dy_key, {}).get("median", 0) for m in methods]
    width = 0.35
    ax.bar(x_pos - width/2, h_l2s, width, color=colors, alpha=0.45,
           label=r"$\|\Delta h\|$ at layer 1")
    ax.bar(x_pos + width/2, dys, width, color=colors,
           label=r"$\|\Delta y\|$ (pred. change)")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(methods, fontsize=8, rotation=15, ha="right")
    ax.set_ylabel("Mean L2 distance")
    ax.set_title(f"{env} ({region_label}): dh vs dy")
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(axis="y", alpha=0.3)

    # boxplot
    ax2 = axes_row[1]
    data, labels, cols = [], [], []
    for m, c in zip(methods, colors):
        if gamma_key in agg[m]:
            data.append(agg[m][gamma_key])
            labels.append(m)
            cols.append(c)
    if data:
        bp = ax2.boxplot(data, tick_labels=labels, patch_artist=True,
                         showfliers=False, widths=0.5)
        for patch, c in zip(bp["boxes"], cols):
            patch.set_facecolor(c)
            patch.set_alpha(0.6)
    ax2.set_ylabel(r"$\gamma = \|\Delta y\| / \|\Delta h\|$")
    ax2.set_title(f"{env} ({region_label}): output sensitivity")
    ax2.set_xticklabels(labels, fontsize=8, rotation=15, ha="right")
    ax2.grid(axis="y", alpha=0.3)


def plot_two_panel(agg, env, out_path):
    """Figure: ID vs OOD-far comparison (2x2 if OOD data present, else 1x2).

    ID and OOD rows share y-axes for easy comparison.
    """
    methods = [m for m in METHOD_ORDER if m in agg]
    colors = [METHOD_COLORS[m] for m in methods]

    has_ood = any("gamma_ood_all" in agg[m] for m in methods)
    nrows = 2 if has_ood else 1
    fig, axes = plt.subplots(nrows, 2, figsize=(13, 4.5 * nrows),
                             sharey="col" if has_ood else False)
    if nrows == 1:
        axes = [axes]

    _plot_bar_and_boxplot(axes[0], agg, methods, colors, "", "ID", env)
    if has_ood:
        _plot_bar_and_boxplot(axes[1], agg, methods, colors, "ood", "OOD-far", env)

    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")


def plot_gamma_boxplot(agg, env, out_path):
    """Standalone output-sensitivity boxplot."""
    fig, ax = plt.subplots(figsize=(5, 4))
    methods = [m for m in METHOD_ORDER if m in agg and "gamma_all" in agg[m]]
    data = [agg[m]["gamma_all"] for m in methods]
    colors = [METHOD_COLORS[m] for m in methods]

    bp = ax.boxplot(data, tick_labels=methods, patch_artist=True,
                    showfliers=False, widths=0.5)
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.6)
    ax.set_ylabel(r"Output sensitivity $\gamma = \||\Delta y\|| / \||\Delta h\||$")
    ax.set_title(f"{env}: output sensitivity per unit hidden perturbation")
    ax.set_xticklabels(methods, fontsize=8, rotation=15, ha="right")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    out_dir = Path(args.out_dir)
    env = args.env
    env_tag = env.lower().replace("-", "_").replace("v5", "")

    all_results = []
    for seed in seeds:
        r = analyze_seed(env, seed, args.pnc_size)
        if r is not None:
            all_results.append(r)

    if not all_results:
        print("No results collected.")
        return

    agg = aggregate_seeds(all_results)

    # Print summary table
    print(f"\n{'='*60}")
    print(f"  SUMMARY: {env}  ({len(all_results)} seeds)")
    print(f"{'='*60}")
    for region, h_key, d_key, g_key in [
        ("ID",      "h_l2",     "dy_median",     "gamma"),
        ("OOD-far", "h_l2_ood", "dy_median_ood", "gamma_ood"),
    ]:
        has_data = any(g_key in agg.get(m, {}) for m in METHOD_ORDER)
        if not has_data:
            continue
        print(f"\n  ── {region} ──")
        print(f"  {'Method':<22} {'dh (L1)':>10} {'dy':>10} {'gamma':>10}")
        print("  " + "-" * 52)
        for m in METHOD_ORDER:
            if m not in agg:
                continue
            h = agg[m].get(h_key, {}).get("median", float("nan"))
            d = agg[m].get(d_key, {}).get("median", float("nan"))
            g = agg[m].get(g_key, {}).get("median", float("nan"))
            print(f"  {m:<22} {h:10.3f} {d:10.3f} {g:10.4f}")

    # Alpha (data-independent, print once)
    print(f"\n  ── Subspace alignment ──")
    print(f"  {'Method':<22} {'alpha':>10} {'s_eff':>10}")
    print("  " + "-" * 42)
    for m in METHOD_ORDER:
        if m not in agg:
            continue
        a = agg[m].get("alpha", {}).get("median", float("nan"))
        s = np.sqrt(1 - a) if np.isfinite(a) else float("nan")
        print(f"  {m:<22} {a:10.4f} {s:10.4f}")

    # Save raw aggregated data
    json_path = out_dir / f"pnc_subspace_analysis_{env_tag}data.json"
    json_out = {}
    for m, d in agg.items():
        json_out[m] = {k: v for k, v in d.items()
                       if not isinstance(v, np.ndarray)}
        for k in list(json_out[m].keys()):
            if k.endswith("_all"):
                del json_out[m][k]
    with open(json_path, "w") as f:
        json.dump(json_out, f, indent=2)
    print(f"Saved {json_path}")

    # Figures
    plot_two_panel(agg, env, out_dir / f"pnc_subspace_analysis_{env_tag}.png")
    plot_gamma_boxplot(agg, env, out_dir / f"pnc_gamma_boxplot_{env_tag}.png")


if __name__ == "__main__":
    main()
