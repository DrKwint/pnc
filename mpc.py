"""Cross-Entropy Method (CEM) planner over a learned dynamics model.

The planner treats the learned dynamics as a simulator and searches over
fixed-length action sequences, optionally penalizing trajectories through
high-predictive-variance regions of the model. This lets us evaluate whether
a given uncertainty estimator (DE, PnC, or Hybrid) produces variance signals
that actually help decision making under distribution shift.

Interface assumption: the `dynamics` object has a `.predict_mean_var(x)`
method that takes a batch of (state, action) inputs of shape `(B, S+A)` and
returns `(means, vars)` each of shape `(B, S)`. (This is what `_predictive_mean_var`
gives us when we call `ensemble.predict(x)` in the existing Gym pipeline.)

Usage:
    planner = CEMPlanner(dynamics, action_dim, horizon=10, n_candidates=32,
                        n_elites=8, n_iters=3, variance_penalty=1.0)
    action = planner.plan(state, reward_fn)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

import jax
import jax.numpy as jnp
import numpy as np


class DynamicsModel(Protocol):
    """Duck-typed dynamics-model interface.

    Anything that exposes ``predict_mean_var(xa_batch)`` returning
    ``(means, vars)`` with shape ``((B, S), (B, S))`` works.
    """

    def predict_mean_var(self, xa_batch: jax.Array) -> tuple[jax.Array, jax.Array]:
        ...


class EnsembleAsDynamics:
    """Adapter from the existing ensemble interface to the DynamicsModel protocol.

    The Gym ensembles in ``ensembles.py`` expose ``predict(x)`` that returns either
    a single tensor or a ``(means, vars)`` tuple. We wrap one so that
    ``predict_mean_var`` collapses the ensemble axis via the law of total variance.
    """

    def __init__(self, ensemble):
        self.ensemble = ensemble

    def predict_mean_var(self, xa_batch):
        preds = self.ensemble.predict(xa_batch)
        if isinstance(preds, tuple) and len(preds) == 2:
            mean_per_member, var_per_member = preds
            mean = jnp.mean(mean_per_member, axis=0)
            var = jnp.mean(var_per_member, axis=0) + jnp.var(mean_per_member, axis=0)
            return mean, var
        mean = jnp.mean(preds, axis=0)
        var = jnp.var(preds, axis=0)
        return mean, var


@dataclass
class CEMConfig:
    horizon: int = 10
    n_candidates: int = 64
    n_elites: int = 8
    n_iters: int = 3
    action_low: float = -1.0
    action_high: float = 1.0
    variance_penalty: float = 0.0
    init_std: float = 1.0
    min_std: float = 0.05
    elite_smoothing: float = 0.1  # exponential smoothing of mean/std across iterations


class CEMPlanner:
    """Batched CEM planner. Plans a horizon-H action sequence at each call
    and returns the first action. Supports uncertainty-aware planning via
    a ``variance_penalty`` scalar.
    """

    def __init__(
        self,
        dynamics: DynamicsModel,
        action_dim: int,
        config: CEMConfig | None = None,
    ):
        self.dynamics = dynamics
        self.action_dim = action_dim
        self.config = config or CEMConfig()
        # Persistent mean sequence across calls (warm-starting the planner)
        self._mean_seq: np.ndarray | None = None

    def _reset_warm_start(self) -> None:
        self._mean_seq = np.zeros((self.config.horizon, self.action_dim), dtype=np.float32)

    def plan(
        self,
        state: np.ndarray,
        reward_fn: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
    ) -> np.ndarray:
        """Plan and return the first action.

        Args:
            state: shape (S,) current state.
            reward_fn: callable(state, action, next_state) -> reward, vectorized
                so that a call with batched `(B, S)`, `(B, A)`, `(B, S)` returns
                shape `(B,)`.

        Returns:
            action: shape (A,) the first action of the best planned sequence.
        """
        cfg = self.config
        if self._mean_seq is None:
            self._reset_warm_start()

        mean_seq = np.array(self._mean_seq, dtype=np.float32)
        std_seq = cfg.init_std * np.ones_like(mean_seq)
        rng = np.random.default_rng()

        for _ in range(cfg.n_iters):
            # Sample n_candidates action sequences from the current Gaussian
            samples = rng.normal(
                size=(cfg.n_candidates, cfg.horizon, self.action_dim)
            ).astype(np.float32)
            samples = mean_seq[None] + std_seq[None] * samples
            samples = np.clip(samples, cfg.action_low, cfg.action_high)

            # Roll out each candidate through the learned dynamics, compute total reward
            total_returns = self._rollout_and_score(state, samples, reward_fn)

            # Pick elites
            elite_idx = np.argsort(-total_returns)[: cfg.n_elites]
            elites = samples[elite_idx]
            new_mean = elites.mean(axis=0)
            new_std = elites.std(axis=0) + cfg.min_std
            mean_seq = (1.0 - cfg.elite_smoothing) * new_mean + cfg.elite_smoothing * mean_seq
            std_seq = (1.0 - cfg.elite_smoothing) * new_std + cfg.elite_smoothing * std_seq

        # Warm-start next call by shifting the mean sequence forward one step
        first_action = mean_seq[0].copy()
        self._mean_seq = np.concatenate([mean_seq[1:], mean_seq[-1:]], axis=0)
        return first_action

    def _rollout_and_score(
        self,
        state: np.ndarray,
        candidate_actions: np.ndarray,  # (n_candidates, H, A)
        reward_fn: Callable,
    ) -> np.ndarray:
        """Run each candidate sequence through the learned dynamics and return
        per-candidate cumulative reward (minus variance penalty).
        """
        cfg = self.config
        n = candidate_actions.shape[0]
        H = cfg.horizon

        cur_state = np.tile(state[None], (n, 1))  # (n, S)
        total_reward = np.zeros(n, dtype=np.float32)
        total_var_penalty = np.zeros(n, dtype=np.float32)

        for t in range(H):
            action = candidate_actions[:, t]  # (n, A)
            xa = jnp.asarray(np.concatenate([cur_state, action], axis=-1))  # (n, S+A)
            mean, var = self.dynamics.predict_mean_var(xa)
            mean_np = np.asarray(mean)
            var_np = np.asarray(var)
            next_state = mean_np  # use the mean prediction as the next-state estimate
            reward = reward_fn(cur_state, action, next_state)
            total_reward += reward
            # Variance penalty: per-step scalar = mean over dims
            if cfg.variance_penalty > 0:
                total_var_penalty += cfg.variance_penalty * var_np.mean(axis=-1)
            cur_state = next_state

        return total_reward - total_var_penalty
