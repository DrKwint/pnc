from dataclasses import dataclass
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np

from geometry import CalibrationGeometry


@dataclass(frozen=True)
class LocalSensitivityMetrics:
    """
    Per-point local response metrics for a perturbation direction at the perturbed layer.

    `local_sens_raw` measures how strongly the point responds to the perturbation.
    `local_sens_proj` measures how much of that response lies outside the calibration
    transfer geometry's linear span when such a geometry is available.
    """

    response: jax.Array
    local_sens_raw: jax.Array
    local_sens_proj: jax.Array | None


def _reshape_direction_like_params(direction: jax.Array, params: jax.Array) -> jax.Array:
    if direction.shape == params.shape:
        return direction
    if direction.size != params.size:
        raise ValueError(
            f"Direction shape {direction.shape} is incompatible with parameter shape {params.shape}."
        )
    return jnp.reshape(direction, params.shape)


def compute_local_sensitivity(
    get_Y_fn: Callable[[jax.Array, jax.Array], jax.Array],
    params: jax.Array,
    direction: jax.Array,
    x: jax.Array,
    geometry: CalibrationGeometry | None = None,
    projection: str = "none",
) -> LocalSensitivityMetrics:
    tangent = _reshape_direction_like_params(direction, params)

    def activation_fn(w):
        return get_Y_fn(w, x)

    _, response = jax.jvp(activation_fn, (params,), (tangent,))
    local_sens_raw = jnp.linalg.norm(response, axis=-1)

    local_sens_proj = None
    if projection == "calibration_span" and geometry is not None:
        residual = geometry.linear_residual(response)
        local_sens_proj = jnp.linalg.norm(residual, axis=-1)

    return LocalSensitivityMetrics(
        response=response,
        local_sens_raw=local_sens_raw,
        local_sens_proj=local_sens_proj,
    )


def build_local_sensitivity_evaluator(
    get_Y_fn: Callable[[jax.Array, jax.Array], jax.Array],
    params: jax.Array,
    geometry: CalibrationGeometry | None = None,
    projection: str = "none",
) -> Callable[[jax.Array, jax.Array], tuple[jax.Array, jax.Array | None]]:
    basis = None
    if projection == "calibration_span" and geometry is not None:
        basis = geometry.basis

    @jax.jit
    def evaluate(direction: jax.Array, x: jax.Array):
        tangent = jnp.reshape(direction, params.shape)

        def activation_fn(w):
            return get_Y_fn(w, x)

        _, response = jax.jvp(activation_fn, (params,), (tangent,))
        local_sens_raw = jnp.linalg.norm(response, axis=-1)

        if basis is None:
            return local_sens_raw, None

        coeffs = jnp.dot(response, basis)
        projected = jnp.dot(coeffs, basis.T)
        residual = response - projected
        local_sens_proj = jnp.linalg.norm(residual, axis=-1)
        return local_sens_raw, local_sens_proj

    return evaluate


def evaluate_local_sensitivity_batch(
    evaluator: Callable[[jax.Array, jax.Array], tuple[jax.Array, jax.Array | None]],
    directions: jax.Array,
    x: jax.Array,
) -> tuple[np.ndarray, np.ndarray | None]:
    raw_values = []
    proj_values = []
    has_proj = None

    for direction in directions:
        local_sens_raw, local_sens_proj = evaluator(direction, x)
        raw_values.append(np.array(local_sens_raw))
        if local_sens_proj is not None:
            proj_values.append(np.array(local_sens_proj))
            has_proj = True
        elif has_proj is None:
            has_proj = False

    raw_array = np.stack(raw_values, axis=0)
    proj_array = np.stack(proj_values, axis=0) if has_proj else None
    return raw_array, proj_array


def add_local_sensitivity_metadata(
    bundle: dict[str, np.ndarray],
    directions: jax.Array,
    perturbation_scale: float,
    projection: str,
) -> None:
    directions_np = np.asarray(directions)
    bundle["member_ids"] = np.arange(directions_np.shape[0], dtype=np.int32)
    bundle["perturbation_direction_l2"] = np.linalg.norm(directions_np, axis=1).astype(
        np.float32
    )
    bundle["perturbation_scale"] = np.array(perturbation_scale, dtype=np.float32)
    bundle["local_sensitivity_projection"] = np.array(projection)


def add_local_sensitivity_fields(
    bundle: dict[str, np.ndarray],
    regime: str,
    local_sens_raw: np.ndarray,
    local_sens_proj: np.ndarray | None,
) -> None:
    bundle[f"local_sens_raw_{regime}"] = np.asarray(local_sens_raw, dtype=np.float32)
    if local_sens_proj is not None:
        bundle[f"local_sens_proj_{regime}"] = np.asarray(local_sens_proj, dtype=np.float32)
