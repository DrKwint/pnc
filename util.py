import os
import random
import time
from typing import Callable, Any

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jaxtyping import Array, Float

from metrics import compute_calibration, compute_nll, compute_ood_metrics, print_metrics
from pjsvd import find_optimal_perturbation, find_optimal_perturbation_full

# ---------------------------------------------------------------------------
# Activation function registry
# ---------------------------------------------------------------------------

ACTIVATIONS: dict[str, callable] = {
    "relu": nnx.relu,
    "gelu": nnx.gelu,
    "tanh": jnp.tanh,
    "silu": jax.nn.silu,
    "elu": jax.nn.elu,
}


def _get_activation(name: str) -> callable:
    if name not in ACTIVATIONS:
        raise ValueError(
            f"Unknown activation '{name}'. Choose from: {list(ACTIVATIONS)}"
        )
    return ACTIVATIONS[name]


# ---------------------------------------------------------------------------
# Seeding
# ---------------------------------------------------------------------------


def seed_everything(seed: int) -> None:
    """Seed all random number generators for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    # Note: JAX is stateless, so we seed nnx.Rngs and use explicit keys elsewhere.
    print(f"[seed] Seeded everything with seed={seed}")


# ---------------------------------------------------------------------------
# Shared evaluation helpers
# ---------------------------------------------------------------------------


def get_intermediate_state(model: Any, x: jax.Array, layer_idx: int) -> jax.Array:
    """
    Extracts the intermediate post-activation state from a model.
    layer_idx=1 corresponds to the output of the first hidden layer (e.g. l1),
    layer_idx=2 corresponds to the second (e.g. l2).
    """
    act = getattr(model, "activation", nnx.relu)
    if hasattr(model, "layers"):
        h = x
        for i in range(layer_idx):
            h = act(model.layers[i](h))
        return h
    else:
        h = act(model.l1(x))
        if layer_idx == 1:
            return h
        h = act(model.l2(h))
        if layer_idx == 2:
            return h
        raise ValueError(f"layer_idx {layer_idx} not supported")


def compute_l2_distance(perturbed_states: jax.Array, orig_state: jax.Array) -> float:
    """
    perturbed_states is (N_members, Batch, Dim)
    orig_state is (Batch, Dim)
    returns mean distance over members and batch
    """
    return float(
        jnp.mean(jnp.sqrt(jnp.sum((perturbed_states - orig_state) ** 2, axis=-1)))
    )


def _block_until_ready_tree(tree: Any) -> Any:
    return jax.tree_util.tree_map(
        lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x,
        tree,
    )


def _predictive_mean_var(preds: Any) -> tuple[jax.Array, jax.Array]:
    if isinstance(preds, tuple) and len(preds) == 2:
        means, vars = preds
        mean = jnp.mean(means, axis=0)
        var = jnp.mean(vars, axis=0) + jnp.var(means, axis=0)
    else:
        mean = jnp.mean(preds, axis=0)
        var = jnp.var(preds, axis=0)
    return mean, var


def _fit_posthoc_variance_scale(
    ensemble: Any,
    inputs: jax.Array,
    targets: jax.Array,
) -> float:
    """Fit a single multiplicative variance scale by minimizing Gaussian NLL."""
    if len(inputs) == 0:
        return 1.0

    _ = ensemble.predict(inputs[:1])
    preds = ensemble.predict(inputs)
    _block_until_ready_tree(preds)

    mean, var = _predictive_mean_var(preds)
    sq_err = (targets - mean) ** 2
    raw_scale = jnp.mean(sq_err / (var + 1e-6))
    scale = float(jnp.clip(raw_scale, 1e-6, 1e6))

    if not np.isfinite(scale):
        return 1.0
    return scale


def _predict_cifar_logits(
    ensemble: Any,
    inputs: Float[np.ndarray, "batch ..."],
    batch_size: int = 256,
) -> jax.Array:
    """Run CIFAR ensemble inference in mini-batches and concatenate logits."""
    logits_batches = []
    for start in range(0, len(inputs), batch_size):
        x_batch = inputs[start : start + batch_size]
        logits_batches.append(ensemble.predict(x_batch))
    return jnp.concatenate(logits_batches, axis=1)


def _fit_posthoc_temperature(
    logits: jax.Array,
    targets: Array,
    max_iters: int = 40,
) -> float:
    """Fit a positive temperature by minimizing validation cross-entropy."""
    if logits.shape[1] == 0:
        return 1.0

    targets = jnp.asarray(targets)
    idx = jnp.arange(targets.shape[0])

    def objective(log_temperature: float) -> float:
        temperature = jnp.exp(log_temperature)
        probs = jax.nn.softmax(logits / temperature, axis=-1)
        mean_probs = jnp.mean(probs, axis=0)
        nll = -jnp.mean(jnp.log(mean_probs[idx, targets] + 1e-8))
        return float(nll)

    # Golden-section search over log-temperature for positivity and stability.
    left = float(np.log(1e-2))
    right = float(np.log(1e2))
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    inv_phi = 1.0 / phi

    c = right - (right - left) * inv_phi
    d = left + (right - left) * inv_phi
    fc = objective(c)
    fd = objective(d)

    for _ in range(max_iters):
        if fc <= fd:
            right = d
            d = c
            fd = fc
            c = right - (right - left) * inv_phi
            fc = objective(c)
        else:
            left = c
            c = d
            fc = fd
            d = left + (right - left) * inv_phi
            fd = objective(d)

    best_log_temperature = c if fc <= fd else d
    temperature = float(np.exp(best_log_temperature))
    if not np.isfinite(temperature):
        return 1.0
    return temperature


def _evaluate_gym(
    ensemble_name: str,
    ensemble: Any,
    dataset: dict,
    sidecar_path: str | None = None,
    calibration_data: tuple[jax.Array, jax.Array] | None = None,
    posthoc_calibrate: bool = False,
    validation_data: tuple[jax.Array, jax.Array] | None = None,
) -> dict[str, float]:
    """Evaluate a gym ensemble, return a scalar metrics dict over multiple regimes.

    `validation_data` is the held-out *training* split slice (e.g., `x_va, y_va`)
    used purely for hyperparameter selection. Metrics computed on it land under
    keys with the `_val` suffix and MUST NOT be reported as final test metrics.
    Reports based on `id_eval` (`*_id` keys) remain the test metrics.
    """
    print(f"\\n--- Results: {ensemble_name} ---")

    variance_scale = 1.0
    if posthoc_calibrate and calibration_data is not None:
        cal_inputs, cal_targets = calibration_data
        variance_scale = _fit_posthoc_variance_scale(ensemble, cal_inputs, cal_targets)
        print(
            f"[posthoc] Learned variance scale on validation split: {variance_scale:.6f}"
        )

    def _group(name, inputs, targets):
        if len(inputs) == 0:
            return float("nan"), float("nan"), float("nan"), float("nan"), np.array([]), np.array([]), 0.0
        # Warm-up
        _ = ensemble.predict(inputs[:1])

        t0 = time.time()
        preds = ensemble.predict(inputs)
        _block_until_ready_tree(preds)
        eval_time = time.time() - t0

        mean, raw_var = _predictive_mean_var(preds)
        var = raw_var * variance_scale

        avg_var = float(jnp.mean(var))
        nll = float(compute_nll(mean, var, targets))
        ece = float(compute_calibration(mean, var, targets))
        rmse = float(jnp.sqrt(jnp.mean((mean - targets) ** 2)))
        print_metrics(name, rmse, avg_var, nll, ece)
        sq_err_per_pt = np.array(jnp.mean((mean - targets) ** 2, axis=-1))
        pred_var_per_pt = np.array(jnp.mean(var, axis=-1))
        return rmse, avg_var, nll, ece, sq_err_per_pt, pred_var_per_pt, eval_time

    results = {}
    sidecar_data = {}

    # Validation split (held out from training data, used ONLY for hyperparameter
    # selection — not for final test reporting)
    if validation_data is not None:
        val_inputs, val_targets = validation_data
        if len(val_inputs) > 0:
            rmse_val, var_val, nll_val, ece_val, _, _, _ = _group("VAL", val_inputs, val_targets)
            results["rmse_val"] = rmse_val
            results["nll_val"] = nll_val
            results["ece_val"] = ece_val
            results["var_val"] = var_val

    # ID (this is the test split, reported as the final test metrics)
    inputs_id, targets_id = dataset["id_eval"]
    rmse_id, var_id, nll_id, ece_id, sq_err_id, pred_var_id, t_id = _group("ID", inputs_id, targets_id)

    results["rmse_id"] = rmse_id
    results["nll_id"] = nll_id
    results["ece_id"] = ece_id
    results["var_id"] = var_id
    results["eval_time"] = t_id
    if posthoc_calibrate:
        results["posthoc_variance_scale"] = variance_scale

    sidecar_data["sq_error_id"] = sq_err_id
    sidecar_data["pred_var_id"] = pred_var_id

    print("\\nOOD Metrics:")
    ood_keys = ["ood_near", "ood_mid", "ood_far", "ood"]
    
    for reg in ood_keys:
        if reg not in dataset: continue
        inputs_ood, targets_ood = dataset[reg]
        r_ood, v_ood, n_ood, e_ood, sq_ood, pv_ood, t_ood = _group(f"OOD ({reg})", inputs_ood, targets_ood)
        
        results[f"rmse_{reg}"] = r_ood
        results[f"nll_{reg}"] = n_ood
        results[f"ece_{reg}"] = e_ood
        results[f"var_{reg}"] = v_ood
        
        sidecar_data[f"sq_error_{reg}"] = sq_ood
        sidecar_data[f"pred_var_{reg}"] = pv_ood
        
        if len(pred_var_id) > 0 and len(pv_ood) > 0:
            auroc, aupr = compute_ood_metrics(pred_var_id, pv_ood)
            results[f"auroc_{reg}"] = auroc
            results[f"aupr_{reg}"] = aupr
            rmse_rat = r_ood / (rmse_id + 1e-6) if rmse_id > 0 else float('inf')
            var_rat = v_ood / (var_id + 1e-9) if var_id > 0 else float('inf')
            print(f"[{reg}] AUROC: {auroc:.4f} | AUPR: {aupr:.4f} | RMSE Rat: {rmse_rat:.2f}x")
            
            # Populate un-suffixed keys from ood_far to not break backward compatibility completely
            if reg == "ood_far":
                results["auroc"] = auroc
                results["aupr"] = aupr
                results["rmse_ratio"] = rmse_rat
                results["var_ratio"] = var_rat
                
    if sidecar_path is not None:
        np.savez(sidecar_path, **sidecar_data)

    return results


def _evaluate_mnist(
    ensemble_name: str,
    ensemble: Any,
    x_test: Float[Array, "batch ..."],
    y_test: Array,
    n_classes: int = 10,
    sidecar_path: str | None = None,
) -> dict[str, float]:
    """Evaluate a classification ensemble and return a metrics dict.

    ECE is computed via reliability diagram binning (confidence-accuracy).
    If sidecar_path is given, writes a .npz with per-sample confidence and
    correctness arrays for calibration / error-variance plot reconstruction.
    """
    print(f"\n--- Results: {ensemble_name} ---")

    # Warm-up
    _ = ensemble.predict(x_test[:1])

    t0 = time.time()
    preds = jax.nn.softmax(ensemble.predict(x_test), axis=-1)  # (S, N, C)
    preds.block_until_ready()
    eval_time = time.time() - t0
    mean = jnp.mean(preds, axis=0)  # (N, C)

    pred_class = jnp.argmax(mean, axis=-1)  # (N,)
    correct = (pred_class == y_test).astype(jnp.float32)  # (N,)
    confidence = jnp.max(mean, axis=-1)  # (N,)

    acc = float(jnp.mean(correct))
    y_oh = jax.nn.one_hot(y_test, n_classes)
    brier = float(jnp.mean(jnp.sum((mean - y_oh) ** 2, axis=-1)))
    # Per-sample entropy for sidecar
    ent_per_sample = np.array(-jnp.sum(mean * jnp.log(mean + 1e-8), axis=-1))
    ent = float(np.mean(ent_per_sample))
    nll = float(-jnp.mean(jnp.log(mean[jnp.arange(len(y_test)), y_test])))

    # ECE: reliability-diagram binning (confidence → accuracy)
    n_bins = 15
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    conf_np = np.array(confidence)
    corr_np = np.array(correct)
    ece_sum = 0.0
    n_total = len(conf_np)
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:], strict=True):
        mask = (conf_np >= lo) & (conf_np < hi)
        if mask.sum() == 0:
            continue
        bin_acc = corr_np[mask].mean()
        bin_conf = conf_np[mask].mean()
        ece_sum += mask.sum() * abs(bin_acc - bin_conf)
    ece = float(ece_sum / n_total)

    print(
        f"Acc: {acc:.4f} | Brier: {brier:.4f} | Entropy: {ent:.4f} | NLL: {nll:.4f} | ECE: {ece:.4f}"
    )

    if sidecar_path is not None:
        np.savez(
            sidecar_path,
            confidence=conf_np,
            correct=corr_np,
            pred_entropy=ent_per_sample,
        )

    return {
        "accuracy": acc,
        "brier": brier,
        "entropy": ent,
        "nll": nll,
        "ece": ece,
        "eval_time": eval_time,
    }


# ---------------------------------------------------------------------------
# Data I/O helpers
# ---------------------------------------------------------------------------


def _load_gym_data(
    paths: dict,
) -> dict:
    """Load arrays written by CollectGymData."""

    def _npz(p):
        d = np.load(p)
        return jnp.array(d["inputs"]), jnp.array(d["targets"])

    ds = {}
    for k in ["id_train", "id_eval", "ood_near", "ood_mid", "ood_far", "ood"]:
        if k in paths:
            ds[k] = _npz(paths[k].path)
    return ds


def _split_data(
    inputs: Any, targets: Any, val_split: float = 0.1, seed: int = 99
) -> tuple[Any, Any, Any, Any]:
    """Split ID data into train and validation sets."""
    n_val = max(1, int(len(inputs) * val_split)) if val_split > 0 else 0
    if n_val > 0:
        rng = np.random.RandomState(seed)
        idx = rng.permutation(len(inputs))
        train_idx, val_idx = idx[n_val:], idx[:n_val]
        return inputs[train_idx], targets[train_idx], inputs[val_idx], targets[val_idx]
    else:
        return inputs, targets, inputs, targets


def _make_cifar_protocol_splits(
    train_inputs: Any,
    train_targets: Any,
    test_inputs: Any,
    test_targets: Any,
    *,
    val_model_select_split: float = 0.1,
    train_bn_refresh_split: float = 0.1,
    seed: int = 99,
) -> dict[str, Any]:
    """Build deterministic paper-style CIFAR splits with disjoint train/val/test roles."""
    if not 0.0 <= val_model_select_split < 1.0:
        raise ValueError("val_model_select_split must be in [0, 1)")
    if not 0.0 <= train_bn_refresh_split < 1.0:
        raise ValueError("train_bn_refresh_split must be in [0, 1)")

    n_total = len(train_inputs)
    if n_total < 2:
        raise ValueError("Need at least 2 training examples for protocol splits")

    rng = np.random.RandomState(seed)
    perm = rng.permutation(n_total)

    n_val = int(np.floor(n_total * val_model_select_split))
    if val_model_select_split > 0.0:
        n_val = max(1, n_val)
    n_val = min(n_val, n_total - 1)

    remaining_after_val = n_total - n_val
    n_bn = int(np.floor(remaining_after_val * train_bn_refresh_split))
    if train_bn_refresh_split > 0.0 and remaining_after_val >= 2:
        n_bn = max(1, n_bn)
    n_bn = min(n_bn, max(0, remaining_after_val - 1))

    val_idx = perm[:n_val]
    bn_idx = perm[n_val:n_val + n_bn]
    fit_idx = perm[n_val + n_bn:]
    if len(fit_idx) == 0:
        raise ValueError("Protocol split left no examples for train_fit")

    metadata = {
        "seed": int(seed),
        "val_model_select_split": float(val_model_select_split),
        "train_bn_refresh_split": float(train_bn_refresh_split),
        "n_train_total": int(n_total),
        "n_train_fit": int(len(fit_idx)),
        "n_train_bn_refresh": int(len(bn_idx)),
        "n_val_model_select": int(len(val_idx)),
        "n_test_report": int(len(test_inputs)),
    }
    return {
        "train_fit": (train_inputs[fit_idx], train_targets[fit_idx]),
        "train_bn_refresh": (train_inputs[bn_idx], train_targets[bn_idx]),
        "val_model_select": (train_inputs[val_idx], train_targets[val_idx]),
        "test_report": (test_inputs, test_targets),
        "metadata": metadata,
    }


def _ps_str(sizes) -> str:
    """Compact string for a list of perturbation sizes used in filenames."""
    return "-".join(str(s) for s in sizes)


# ---------------------------------------------------------------------------
# PJSVD null-space search (shared logic, called from multiple tasks)
# ---------------------------------------------------------------------------


def _find_pjsvd_directions(
    model_fn: Callable,
    W_curr: Float[Array, "..."],
    n_directions: int,
    use_full_span: bool = False,
    seed: int = 99,
) -> tuple[Float[Array, "k D_flat"], np.ndarray]:
    """Find n_directions orthogonal null-space directions for a single-layer model fn."""
    D = W_curr.size
    v_opts_buf = np.zeros((n_directions, D), dtype=np.float32)
    direction_mask = np.zeros(n_directions, dtype=bool)
    sigmas = []

    for k in range(n_directions):
        v_opts_jax = jnp.array(v_opts_buf)
        mask_jax = jnp.array(direction_mask)

        if use_full_span:
            v_opt, sigma = find_optimal_perturbation_full(
                model_fn,
                W_curr,
                max_iter=500,
                orthogonal_directions=v_opts_jax,
                direction_mask=mask_jax,
                seed=seed + k,
            )
        else:
            v_opt, sigma = find_optimal_perturbation(
                model_fn,
                W_curr,
                max_iter=500,
                orthogonal_directions=v_opts_jax,
                direction_mask=mask_jax,
                seed=seed + k,
            )

        v_opts_buf[k] = np.array(v_opt.reshape(-1))
        direction_mask[k] = True
        sigmas.append(float(sigma))
        print(f"  Direction {k + 1}: sigma={sigma:.6f}")

    return jnp.array(v_opts_buf), np.array(sigmas)


def _evaluate_cifar_logits(
    ensemble_name: str,
    logits: jax.Array,
    y_true: Array,
    *,
    n_classes: int = 10,
    temperature: float = 1.0,
    eval_time: float = 0.0,
    sidecar_path: str | None = None,
) -> dict[str, float]:
    """Evaluate CIFAR classification metrics from frozen ensemble logits."""
    print(f"\n--- Results: {ensemble_name} ---")
    preds = jax.nn.softmax(logits / temperature, axis=-1)  # (S, N, C)
    preds.block_until_ready()
    mean = jnp.mean(preds, axis=0)  # (N, C)
    pred_class = jnp.argmax(mean, axis=-1)
    correct = (pred_class == y_true).astype(jnp.float32)
    confidence = jnp.max(mean, axis=-1)  # (N,)

    acc = float(jnp.mean(correct))
    y_oh = jax.nn.one_hot(y_true, n_classes)
    brier = float(jnp.mean(jnp.sum((mean - y_oh) ** 2, axis=-1)))
    ent_ps = np.array(-jnp.sum(mean * jnp.log(mean + 1e-8), axis=-1))  # (N,)
    ent = float(np.mean(ent_ps))
    idx = jnp.arange(len(y_true))
    nll = float(-jnp.mean(jnp.log(mean[idx, y_true] + 1e-8)))

    # ECE (reliability diagram)
    n_bins = 15
    conf_np = np.array(confidence)
    corr_np = np.array(correct)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece_sum = 0.0
    n_total = len(conf_np)
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:], strict=True):
        mask = (conf_np >= lo) & (conf_np < hi)
        if mask.sum() == 0:
            continue
        ece_sum += mask.sum() * abs(corr_np[mask].mean() - conf_np[mask].mean())
    ece = float(ece_sum / n_total)

    print(
        f"Acc: {acc:.4f} | Brier: {brier:.4f} | Entropy: {ent:.4f} | NLL: {nll:.4f} | ECE: {ece:.4f}"
    )

    if sidecar_path is not None:
        np.savez(sidecar_path, confidence=conf_np, correct=corr_np, pred_entropy=ent_ps)

    return {
        "accuracy": acc,
        "brier": brier,
        "entropy": ent,
        "nll": nll,
        "ece": ece,
        "eval_time": eval_time,
        "temperature": float(temperature),
    }


def _evaluate_cifar(
    ensemble_name: str,
    ensemble: Any,
    x_test: Float[np.ndarray, "batch ..."],
    y_test: Array,
    n_classes: int = 10,
    batch_size: int = 256,
    sidecar_path: str | None = None,
    calibration_data: tuple[np.ndarray, Array] | None = None,
    posthoc_calibrate: bool = False,
) -> dict[str, float]:
    """
    Evaluate a classification ensemble on CIFAR images.

    Runs inference in mini-batches to avoid OOM on large ResNet-50 ensembles.
    Returns the same metrics dict as _evaluate_mnist for easy reporting.
    """
    # Warm-up
    _ = ensemble.predict(x_test[:1])

    temperature = 1.0
    if posthoc_calibrate and calibration_data is not None:
        cal_inputs, cal_targets = calibration_data
        cal_logits = _predict_cifar_logits(ensemble, cal_inputs, batch_size=batch_size)
        _block_until_ready_tree(cal_logits)
        temperature = _fit_posthoc_temperature(cal_logits, cal_targets)
        print(f"[posthoc] Learned temperature on validation split: {temperature:.6f}")

    t0 = time.time()
    logits = _predict_cifar_logits(ensemble, x_test, batch_size=batch_size)
    _block_until_ready_tree(logits)
    eval_time = time.time() - t0

    metrics = _evaluate_cifar_logits(
        ensemble_name,
        logits,
        y_test,
        n_classes=n_classes,
        temperature=temperature,
        eval_time=eval_time,
        sidecar_path=sidecar_path,
    )
    metrics["posthoc_temperature"] = metrics.pop("temperature")
    return metrics
