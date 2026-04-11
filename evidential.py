"""Evidential Deep Regression baseline (Amini et al. 2020, NeurIPS).

Replaces the Gaussian NLL head of a probabilistic regressor with a
Normal-Inverse-Gamma (NIG) head parameterized by (γ, ν, α, β). Predictive mean
is γ; total predictive variance is β·(1 + 1/ν)/(α−1). We train with the
evidential NIG loss + an L1 regularizer on mismatched evidence (Eq. 10 in
Amini et al.).

Usage:
    from evidential import (
        EvidentialRegressionModel,
        train_evidential_model,
        EvidentialPredictor,
    )

    model = EvidentialRegressionModel(in_features=D, out_features=K, rngs=rngs,
                                      hidden_dims=[200, 200, 200, 200])
    model = train_evidential_model(model, x_tr, y_tr, x_va, y_va, lam=0.01)
    predictor = EvidentialPredictor(model)
    mean, var = predictor.predict(x_test)  # both (B, K)
"""

from __future__ import annotations

from typing import Callable, List

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from training import train_generic


class EvidentialRegressionModel(nnx.Module):
    """Deep evidential regressor with Normal-Inverse-Gamma output.

    The output head produces 4*K channels from the last hidden layer, which
    are split into (γ, ν, α, β). Positivity constraints:
        ν = softplus(raw_ν)
        α = softplus(raw_α) + 1       (must be > 1 for finite variance)
        β = softplus(raw_β)

    Layer structure mirrors ProbabilisticRegressionModel (self.layers,
    self.activation) so that PJSVD / per-layer operations keep working if ever
    needed.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rngs: nnx.Rngs,
        hidden_dims: List[int] = [50],
        activation: Callable = nnx.relu,
    ):
        layers = []
        dims = [in_features] + hidden_dims
        for i in range(len(dims) - 1):
            layers.append(nnx.Linear(dims[i], dims[i + 1], rngs=rngs))
        self.layers = nnx.List(layers)
        self.out_features = out_features

        # One head that produces 4*K logits
        self.evidential_head = nnx.Linear(hidden_dims[-1], 4 * out_features, rngs=rngs)
        self.activation = activation

    def __call__(
        self, x: jax.Array
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        for layer in self.layers:
            x = self.activation(layer(x))

        logits = self.evidential_head(x)  # (B, 4*K)
        gamma, raw_nu, raw_alpha, raw_beta = jnp.split(logits, 4, axis=-1)
        nu = jax.nn.softplus(raw_nu) + 1e-6
        alpha = jax.nn.softplus(raw_alpha) + 1.0 + 1e-6
        beta = jax.nn.softplus(raw_beta) + 1e-6
        return gamma, nu, alpha, beta


def evidential_nig_nll(
    y: jax.Array,
    gamma: jax.Array,
    nu: jax.Array,
    alpha: jax.Array,
    beta: jax.Array,
) -> jax.Array:
    """Negative log-likelihood of y under the NIG predictive (Eq. 8, Amini 2020)."""
    twobeta_1_nu = 2.0 * beta * (1.0 + nu)
    nll = (
        0.5 * jnp.log(jnp.pi / nu)
        - alpha * jnp.log(twobeta_1_nu)
        + (alpha + 0.5) * jnp.log((y - gamma) ** 2 * nu + twobeta_1_nu)
        + jax.lax.lgamma(alpha)
        - jax.lax.lgamma(alpha + 0.5)
    )
    return nll


def evidential_reg(
    y: jax.Array,
    gamma: jax.Array,
    nu: jax.Array,
    alpha: jax.Array,
) -> jax.Array:
    """Evidence regularizer (Eq. 10, Amini 2020): penalizes evidence on errors."""
    return jnp.abs(y - gamma) * (2.0 * nu + alpha)


def evidential_loss(m: nnx.Module, x: jax.Array, y: jax.Array, lam: float = 0.01):
    gamma, nu, alpha, beta = m(x)
    nll = evidential_nig_nll(y, gamma, nu, alpha, beta)
    reg = evidential_reg(y, gamma, nu, alpha)
    return jnp.mean(nll + lam * reg)


def train_evidential_model(
    model: nnx.Module,
    train_inputs: jax.Array,
    train_targets: jax.Array,
    val_inputs: jax.Array,
    val_targets: jax.Array,
    lam: float = 0.01,
    steps: int = 5000,
    batch_size: int = 64,
    patience: int = 10,
    eval_freq: int = 100,
) -> nnx.Module:
    """Train an EvidentialRegressionModel with the NIG loss + reg."""

    def loss_fn(m, x, y):
        return evidential_loss(m, x, y, lam=lam)

    return train_generic(
        model,
        train_inputs,
        train_targets,
        val_inputs,
        val_targets,
        loss_fn=loss_fn,
        steps=steps,
        batch_size=batch_size,
        patience=patience,
        eval_freq=eval_freq,
        log_prefix=f"Training Evidential (lam={lam})",
    )


class EvidentialPredictor:
    """Adapter that exposes an ``ensemble``-like interface over a trained
    EvidentialRegressionModel.

    ``predict(x)`` returns ``(means, vars)`` where both are shape ``(1, B, K)``
    so that the downstream ``_predictive_mean_var`` (which averages over the
    leading ensemble axis) yields the NIG total predictive variance.
    """

    def __init__(self, model: EvidentialRegressionModel):
        self.model = model

    def predict(self, x: jax.Array) -> tuple[jax.Array, jax.Array]:
        gamma, nu, alpha, beta = self.model(x)
        # Total predictive variance from the NIG posterior predictive (Student-t):
        #   Var[y] = β · (1 + 1/ν) / (α − 1)   for α > 1
        var = beta * (1.0 + 1.0 / nu) / (alpha - 1.0)
        # Insert a leading size-1 ensemble axis so downstream averaging is a no-op.
        mean = gamma[None, ...]
        var = var[None, ...]
        return mean, var

    def predict_one(self, x: jax.Array, idx: int) -> jax.Array:
        gamma, nu, alpha, beta = self.model(x)
        return gamma
