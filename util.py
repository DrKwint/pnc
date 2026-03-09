import os
import sys
import argparse
import time
import json
import random
import pickle
from pathlib import Path
import jax
import jax.numpy as jnp
from flax import nnx
import numpy as np
from metrics import compute_nll, compute_calibration, print_metrics, compute_ood_metrics
from pjsvd import find_optimal_perturbation, find_optimal_perturbation_full

# ---------------------------------------------------------------------------
# Activation function registry
# ---------------------------------------------------------------------------

ACTIVATIONS: dict[str, callable] = {
    "relu":  nnx.relu,
    "gelu":  nnx.gelu,
    "tanh":  jnp.tanh,
    "silu":  jax.nn.silu,
    "elu":   jax.nn.elu,
}


def _get_activation(name: str) -> callable:
    if name not in ACTIVATIONS:
        raise ValueError(f"Unknown activation '{name}'. Choose from: {list(ACTIVATIONS)}")
    return ACTIVATIONS[name]


# ---------------------------------------------------------------------------
# Seeding
# ---------------------------------------------------------------------------

def seed_everything(seed: int) -> None:
    """Seed all random number generators for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"[seed] Seeded everything with seed={seed}")


# ---------------------------------------------------------------------------
# Shared evaluation helpers
# ---------------------------------------------------------------------------

def _evaluate_gym(ensemble_name: str, ensemble,
                  inputs_id_eval, targets_id_eval,
                  inputs_ood, targets_ood,
                  sidecar_path: str | None = None) -> dict:
    """Evaluate a gym ensemble, return a scalar metrics dict.

    If sidecar_path is given, also writes a .npz file with per-point
    sq_error and predictive_variance arrays (ID and OOD) so that
    error-variance plots can be reconstructed post-hoc.
    """
    print(f"\n--- Results: {ensemble_name} ---")

    def _group(name, inputs, targets):
        preds    = ensemble.predict(inputs)          # (S, N, D)
        mean     = jnp.mean(preds, axis=0)           # (N, D)
        var      = jnp.var(preds, axis=0)            # (N, D)
        avg_var  = float(jnp.mean(var))
        nll      = float(compute_nll(mean, var, targets))
        ece      = float(compute_calibration(mean, var, targets))
        rmse     = float(jnp.sqrt(jnp.mean((mean - targets) ** 2)))
        print_metrics(name, rmse, avg_var, nll, ece)
        # Per-point scalars for error-variance plot:
        # sq_error: mean squared error per sample (averaged over output dims)
        sq_err_per_pt = np.array(jnp.mean((mean - targets) ** 2, axis=-1))  # (N,)
        # predictive_variance: mean predictive variance per sample
        pred_var_per_pt = np.array(jnp.mean(var, axis=-1))                  # (N,)
        return rmse, avg_var, nll, ece, sq_err_per_pt, pred_var_per_pt

    rmse_id,  var_id,  nll_id,  ece_id,  sq_err_id,  pred_var_id  = \
        _group("ID",  inputs_id_eval,  targets_id_eval)
    rmse_ood, var_ood, nll_ood, ece_ood, sq_err_ood, pred_var_ood = \
        _group("OOD", inputs_ood, targets_ood)

    # Re-use the per-point predictive variance we already computed!
    auroc, aupr = compute_ood_metrics(pred_var_id, pred_var_ood)

    rmse_ratio = rmse_ood / (rmse_id + 1e-6)
    var_ratio  = var_ood  / (var_id  + 1e-9)
    print("\nOOD Detection Metrics:")
    print(f"AUROC: {auroc:.4f} | AUPR: {aupr:.4f}")
    print(f"RMSE Ratio (OOD / ID): {rmse_ratio:.2f}x | Variance Ratio: {var_ratio:.2f}x")

    if sidecar_path is not None:
        np.savez(
            sidecar_path,
            sq_error_id=sq_err_id,   pred_var_id=pred_var_id,
            sq_error_ood=sq_err_ood, pred_var_ood=pred_var_ood,
        )

    return {
        "rmse_id":  rmse_id,  "nll_id":  nll_id,  "ece_id":  ece_id,
        "rmse_ood": rmse_ood, "nll_ood": nll_ood, "ece_ood": ece_ood,
        "var_id": var_id, "var_ood": var_ood,
        "auroc": auroc, "aupr": aupr,
        "rmse_ratio": rmse_ratio, "var_ratio": var_ratio,
    }


def _evaluate_mnist(ensemble_name: str, ensemble, x_test, y_test,
                    n_classes: int = 10,
                    sidecar_path: str | None = None) -> dict:
    """Evaluate a classification ensemble and return a metrics dict.

    ECE is computed via reliability diagram binning (confidence-accuracy).
    If sidecar_path is given, writes a .npz with per-sample confidence and
    correctness arrays for calibration / error-variance plot reconstruction.
    """
    print(f"\n--- Results: {ensemble_name} ---")

    preds = jax.nn.softmax(ensemble.predict(x_test), axis=-1)  # (S, N, C)
    mean  = jnp.mean(preds, axis=0)                            # (N, C)

    pred_class  = jnp.argmax(mean, axis=-1)                    # (N,)
    correct     = (pred_class == y_test).astype(jnp.float32)   # (N,)
    confidence  = jnp.max(mean, axis=-1)                       # (N,)

    acc   = float(jnp.mean(correct))
    y_oh  = jax.nn.one_hot(y_test, n_classes)
    brier = float(jnp.mean(jnp.sum((mean - y_oh) ** 2, axis=-1)))
    # Per-sample entropy for sidecar
    ent_per_sample = np.array(-jnp.sum(mean * jnp.log(mean + 1e-8), axis=-1))
    ent = float(np.mean(ent_per_sample))

    # ECE: reliability-diagram binning (confidence → accuracy)
    n_bins   = 15
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    conf_np   = np.array(confidence)
    corr_np   = np.array(correct)
    ece_sum   = 0.0
    n_total   = len(conf_np)
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (conf_np >= lo) & (conf_np < hi)
        if mask.sum() == 0:
            continue
        bin_acc  = corr_np[mask].mean()
        bin_conf = conf_np[mask].mean()
        ece_sum += mask.sum() * abs(bin_acc - bin_conf)
    ece = float(ece_sum / n_total)

    print(f"Acc: {acc:.4f} | Brier: {brier:.4f} | Entropy: {ent:.4f} | ECE: {ece:.4f}")

    if sidecar_path is not None:
        np.savez(
            sidecar_path,
            confidence=conf_np,
            correct=corr_np,
            pred_entropy=ent_per_sample,
        )

    return {"accuracy": acc, "brier": brier, "entropy": ent, "ece": ece}


# ---------------------------------------------------------------------------
# Data I/O helpers
# ---------------------------------------------------------------------------

def _load_gym_data(paths: dict):
    """Load arrays written by CollectGymData."""
    def _npz(p):
        d = np.load(p)
        return jnp.array(d["inputs"]), jnp.array(d["targets"])

    inputs_id,      targets_id      = _npz(paths["id_train"].path)
    inputs_id_eval, targets_id_eval = _npz(paths["id_eval"].path)
    inputs_ood,     targets_ood     = _npz(paths["ood"].path)
    return inputs_id, targets_id, inputs_id_eval, targets_id_eval, inputs_ood, targets_ood


def _ps_str(sizes) -> str:
    """Compact string for a list of perturbation sizes used in filenames."""
    return "-".join(str(s) for s in sizes)

# ---------------------------------------------------------------------------
# PJSVD null-space search (shared logic, called from multiple tasks)
# ---------------------------------------------------------------------------

def _find_pjsvd_directions(model_fn, W_curr, n_directions: int, use_full_span=False):
    """Find n_directions orthogonal null-space directions for a single-layer model fn."""
    D = W_curr.size
    v_opts_buf   = np.zeros((n_directions, D), dtype=np.float32)
    direction_mask = np.zeros(n_directions, dtype=bool)
    sigmas = []

    for k in range(n_directions):
        v_opts_jax = jnp.array(v_opts_buf)
        mask_jax   = jnp.array(direction_mask)
        
        solver_fn = find_optimal_perturbation_full if use_full_span else find_optimal_perturbation
        
        v_opt, sigma = solver_fn(
            model_fn, W_curr, max_iter=500,
            orthogonal_directions=v_opts_jax,
            direction_mask=mask_jax,
        )
        v_opts_buf[k]    = np.array(v_opt.reshape(-1))
        direction_mask[k] = True
        sigmas.append(float(sigma))
        print(f"  Direction {k+1}: sigma={sigma:.6f}")

    return jnp.array(v_opts_buf), np.array(sigmas)


def _evaluate_cifar(ensemble_name: str, ensemble, x_test, y_test,
                    n_classes: int = 10,
                    batch_size: int = 256,
                    sidecar_path: str | None = None) -> dict:
    """
    Evaluate a classification ensemble on CIFAR images.

    Runs inference in mini-batches to avoid OOM on large ResNet-50 ensembles.
    Returns the same metrics dict as _evaluate_mnist for easy reporting.
    """
    print(f"\n--- Results: {ensemble_name} ---")

    # Collect per-batch softmax predictions
    all_preds = []
    for start in range(0, len(x_test), batch_size):
        x_batch = x_test[start:start + batch_size]
        # ensemble.predict returns (S, N_batch, C) logits
        logits_batch = ensemble.predict(x_batch)  # (S, N_batch, C)
        probs_batch  = jax.nn.softmax(logits_batch, axis=-1)  # (S, N_batch, C)
        all_preds.append(probs_batch)
    preds = jnp.concatenate(all_preds, axis=1)  # (S, N, C)

    mean         = jnp.mean(preds, axis=0)                             # (N, C)
    pred_class   = jnp.argmax(mean, axis=-1)                           # (N,)
    correct      = (pred_class == y_test).astype(jnp.float32)          # (N,)
    confidence   = jnp.max(mean, axis=-1)                              # (N,)

    acc    = float(jnp.mean(correct))
    y_oh   = jax.nn.one_hot(y_test, n_classes)
    brier  = float(jnp.mean(jnp.sum((mean - y_oh) ** 2, axis=-1)))
    ent_ps = np.array(-jnp.sum(mean * jnp.log(mean + 1e-8), axis=-1))  # (N,)
    ent    = float(np.mean(ent_ps))

    # ECE (reliability diagram)
    n_bins    = 15
    conf_np   = np.array(confidence)
    corr_np   = np.array(correct)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece_sum   = 0.0
    n_total   = len(conf_np)
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (conf_np >= lo) & (conf_np < hi)
        if mask.sum() == 0:
            continue
        ece_sum += mask.sum() * abs(corr_np[mask].mean() - conf_np[mask].mean())
    ece = float(ece_sum / n_total)

    print(f"Acc: {acc:.4f} | Brier: {brier:.4f} | Entropy: {ent:.4f} | ECE: {ece:.4f}")

    if sidecar_path is not None:
        np.savez(sidecar_path,
                 confidence=conf_np, correct=corr_np, pred_entropy=ent_ps)

    return {"accuracy": acc, "brier": brier, "entropy": ent, "ece": ece}

