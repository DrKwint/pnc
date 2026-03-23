import jax
import jax.numpy as jnp
from sklearn.metrics import roc_auc_score, average_precision_score
import numpy as np

def compute_nll(mean: jax.Array, variance: jax.Array, targets: jax.Array) -> jax.Array:
    """
    Computes Negative Log Likelihood (Gaussian).
    """
    variance = variance + 1e-6 # Stability
    nll = 0.5 * (jnp.log(variance) + (targets - mean)**2 / variance + jnp.log(2 * jnp.pi))
    return jnp.mean(nll)

def compute_calibration(mean: jax.Array, variance: jax.Array, targets: jax.Array, n_bins: int = 10) -> jax.Array:
    """
    Computes Expected Calibration Error (ECE) for regression by checking quantiles.
    """
    std = jnp.sqrt(variance + 1e-6)
    # Calculate CDF values for targets under predicted Gaussian
    # Using error function for standard normal CDF: 0.5 * (1 + erf((x - mu) / (sigma * sqrt(2))))
    z = (targets - mean) / (std * jnp.sqrt(2))
    cdfs = 0.5 * (1 + jax.scipy.special.erf(z))

    # ECE: Check if CDFs are uniform [0, 1]
    expected = jnp.linspace(0, 1, n_bins + 1)
    observed = []

    # Needs to be a scalar list for jnp.array
    for p in expected.tolist():
        observed.append(jnp.mean(cdfs <= p))

    # Mean absolute difference between observed proportion and expected proportion
    return jnp.mean(jnp.abs(jnp.array(observed) - expected))

def print_metrics(name: str, rmse: float, var: float, nll: float, calib_err: float):
    print(f"[{name}] RMSE: {rmse:.5f} | Var: {var:.5f} | NLL: {nll:.5f} | CalibErr: {calib_err:.5f}")

def compute_ood_metrics(
    var_id_scores: np.ndarray,
    var_ood_scores: np.ndarray
) -> tuple[float, float]:
    """
    Calculates AUROC and AUPR for Out-of-Distribution detection based on uncertainty scores.
    """
    labels = np.concatenate([np.zeros(len(var_id_scores)), np.ones(len(var_ood_scores))])
    scores = np.concatenate([var_id_scores, var_ood_scores])

    auroc = roc_auc_score(labels, scores)
    aupr = average_precision_score(labels, scores)
    return float(auroc), float(aupr)
