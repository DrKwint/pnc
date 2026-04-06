import numpy as np
import jax.numpy as jnp
from numpy.testing import assert_allclose

from geometry import CalibrationGeometry
from geometry_analysis import aggregate_reports, analyze_geometry_metrics
from local_sensitivity import (
    add_local_sensitivity_fields,
    add_local_sensitivity_metadata,
    compute_local_sensitivity,
)


def test_local_sensitivity_jvp_matches_finite_difference():
    params = jnp.array([[0.2, -0.3], [0.4, 0.1]], dtype=jnp.float32)
    direction = jnp.array([[0.5, -0.2], [0.1, 0.3]], dtype=jnp.float32)
    bias = jnp.array([0.05, -0.1], dtype=jnp.float32)
    x = jnp.array([[1.0, 2.0], [-0.5, 1.5]], dtype=jnp.float32)

    def get_Y_fn(w, x_batch):
        return jnp.tanh(x_batch @ w + bias)

    metrics = compute_local_sensitivity(get_Y_fn, params, direction.reshape(-1), x)

    eps = 1e-3
    fd = (
        get_Y_fn(params + eps * direction, x) - get_Y_fn(params - eps * direction, x)
    ) / (2.0 * eps)

    assert_allclose(np.array(metrics.response), np.array(fd), atol=2e-3, rtol=2e-2)
    assert_allclose(
        np.array(metrics.local_sens_raw),
        np.linalg.norm(np.array(fd), axis=-1),
        atol=2e-3,
        rtol=2e-2,
    )


def test_projected_local_sensitivity_tracks_orthogonal_component():
    geometry = CalibrationGeometry(
        jnp.array([[0.0, 0.0], [1.0, 0.0], [-1.0, 0.0]], dtype=jnp.float32)
    )
    params = jnp.array([[0.0, 0.0]], dtype=jnp.float32)
    x = jnp.array([[1.0], [1.0]], dtype=jnp.float32)

    def get_Y_fn(w, x_batch):
        return x_batch @ w

    in_span = compute_local_sensitivity(
        get_Y_fn,
        params,
        jnp.array([[2.0, 0.0]], dtype=jnp.float32).reshape(-1),
        x,
        geometry=geometry,
        projection="calibration_span",
    )
    assert_allclose(np.array(in_span.local_sens_proj), np.zeros(2), atol=1e-6)

    with_orthogonal = compute_local_sensitivity(
        get_Y_fn,
        params,
        jnp.array([[2.0, 3.0]], dtype=jnp.float32).reshape(-1),
        x,
        geometry=geometry,
        projection="calibration_span",
    )
    assert_allclose(np.array(with_orthogonal.local_sens_proj), np.full(2, 3.0), atol=1e-6)


def test_local_sensitivity_fields_are_logged_in_npz_bundle(tmp_path):
    bundle = {}
    directions = jnp.array([[1.0, 0.0], [0.0, 1.0]], dtype=jnp.float32)
    add_local_sensitivity_metadata(bundle, directions, perturbation_scale=4.0, projection="calibration_span")
    add_local_sensitivity_fields(
        bundle,
        "id",
        np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32),
        np.array([[0.01, 0.02], [0.03, 0.04]], dtype=np.float32),
    )

    out_path = tmp_path / "toy_geometry.npz"
    np.savez(out_path, **bundle)

    with np.load(out_path) as data:
        assert "member_ids" in data
        assert "perturbation_direction_l2" in data
        assert "local_sens_raw_id" in data
        assert "local_sens_proj_id" in data
        assert data["local_sens_raw_id"].shape == (2, 2)


def test_geometry_analysis_emits_regression_outputs(tmp_path):
    geom = np.array([[0.0, 1.0, 2.0], [1.5, 2.5, 3.5]], dtype=np.float32)
    local_proj = np.array([[1.0, 0.5, 2.0], [0.0, 1.0, 1.5]], dtype=np.float32)
    local_raw = local_proj + 0.25
    corr = 1.0 + 2.0 * geom + 3.0 * local_proj

    out_path = tmp_path / "toy_ps1_geometry.npz"
    np.savez(
        out_path,
        geom_dist_id=geom,
        corr_l2_id=corr,
        local_sens_raw_id=local_raw,
        local_sens_proj_id=local_proj,
        member_ids=np.array([0, 1], dtype=np.int32),
    )

    results = analyze_geometry_metrics(
        str(out_path),
        p_size=1.0,
        env_name="ToyEnv",
        out_dir=str(tmp_path),
        generate_individual_plots=False,
    )

    combined = results["Combined"]
    assert "geom_dist__corr_l2" in combined["pairwise"]
    assert "local_sens_proj__corr_l2" in combined["pairwise"]
    assert "geom_plus_proj" in combined["regressions"]

    fit = combined["regressions"]["geom_plus_proj"]
    assert np.isfinite(fit["coefficients"]["geom_dist"])
    assert np.isfinite(fit["coefficients"]["local_sens_proj"])
    assert fit["r_squared"] > 0.99


def test_aggregate_reports_separates_environments_with_matching_experiment_names():
    shared_results = {
        "Combined": {
            "pairwise": {
                "geom_dist__corr_l2": {
                    "pearson": 0.1,
                    "spearman": 0.2,
                    "slope": 0.3,
                    "intercept": 0.4,
                    "r_squared": 0.5,
                }
            },
            "regressions": {},
        }
    }

    aggregated = aggregate_reports(
        [
            (
                "results/Ant-v5/experiment_seed0_ps1_geometry.npz",
                "Ant-v5",
                1.0,
                shared_results,
            ),
            (
                "results/HalfCheetah-v5/experiment_seed0_ps1_geometry.npz",
                "HalfCheetah-v5",
                1.0,
                shared_results,
            ),
        ]
    )

    assert len(aggregated) == 2
    env_names = {group["env_name"] for group in aggregated.values()}
    assert env_names == {"Ant-v5", "HalfCheetah-v5"}

    ant_group = next(group for group in aggregated.values() if group["env_name"] == "Ant-v5")
    cheetah_group = next(
        group for group in aggregated.values() if group["env_name"] == "HalfCheetah-v5"
    )
    assert ant_group["canonical"] == "experiment_ps1"
    assert cheetah_group["canonical"] == "experiment_ps1"
