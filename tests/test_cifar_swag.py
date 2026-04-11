import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from cifar_tasks import CIFARPreActSWAG, CIFARTrainSWAGPreActResNet18
from ensembles import SWAGEnsemble
from models import PreActResNet18
from training import train_resnet_swag


def _batch_stat_arrays(state, leaf_name: str):
    vals = []
    for path, variable in nnx.to_flat_state(state):
        if path[-1] == leaf_name:
            vals.append(np.array(variable.get_value()))
    return vals


def test_swag_task_output_paths_include_swag_hyperparameters():
    train_task = CIFARTrainSWAGPreActResNet18(
        swag_start_epoch=123,
        swag_collect_freq=2,
        swag_max_rank=7,
    )
    eval_task = CIFARPreActSWAG(
        swag_start_epoch=123,
        swag_collect_freq=2,
        swag_max_rank=7,
        swag_use_bn_refresh=True,
        bn_refresh_subset_size=512,
    )

    assert "_sws123_swf2_swr7_" in train_task.output().path
    assert "_sws123_swf2_swr7_bnr1_bns512_" in eval_task.output().path


def test_swag_eval_task_requires_matching_train_configuration():
    eval_task = CIFARPreActSWAG(
        swag_start_epoch=123,
        swag_collect_freq=2,
        swag_max_rank=7,
        swag_use_bn_refresh=False,
        bn_refresh_subset_size=256,
    )
    train_task = eval_task.requires()

    assert isinstance(train_task, CIFARTrainSWAGPreActResNet18)
    assert train_task.swag_start_epoch == 123
    assert train_task.swag_collect_freq == 2
    assert train_task.swag_max_rank == 7


def test_swag_ensemble_refreshes_batchnorm_and_samples_distinct_logits():
    model = PreActResNet18(n_classes=10, rngs=nnx.Rngs(0))
    params = nnx.state(model, nnx.Param)
    swag_var = jax.tree.map(lambda p: jnp.full_like(p, 1e-4), params)
    flat_params, _ = jax.flatten_util.ravel_pytree(params)
    cov_rng = np.random.RandomState(2)
    swag_cov_mat_sqrt = jnp.asarray(
        np.stack(
            [
                cov_rng.normal(size=flat_params.shape).astype(np.asarray(flat_params).dtype) * 1e-2,
                cov_rng.normal(size=flat_params.shape).astype(np.asarray(flat_params).dtype) * 1e-2,
            ],
            axis=1,
        )
    )
    bn_refresh_inputs = np.random.RandomState(0).normal(size=(8, 32, 32, 3)).astype(np.float32)
    x_eval = jnp.array(np.random.RandomState(1).normal(size=(4, 32, 32, 3)).astype(np.float32))

    ensemble = SWAGEnsemble(
        model,
        params,
        swag_var,
        n_models=2,
        swag_cov_mat_sqrt=swag_cov_mat_sqrt,
        bn_refresh_inputs=bn_refresh_inputs,
        bn_refresh_batch_size=4,
        use_bn_refresh=True,
        seed=0,
    )

    sampled_1 = ensemble._sample_model()
    sampled_2 = ensemble._sample_model()
    logits_1 = sampled_1(x_eval, use_running_average=True)
    logits_2 = sampled_2(x_eval, use_running_average=True)
    bn_state = nnx.state(sampled_1, nnx.BatchStat)

    mean_leaves = _batch_stat_arrays(bn_state, "mean")
    var_leaves = _batch_stat_arrays(bn_state, "var")

    assert any(not np.allclose(mean, 0.0) for mean in mean_leaves)
    assert any(not np.allclose(var, 1.0) for var in var_leaves)
    assert not np.allclose(np.array(logits_1), np.array(logits_2))


def test_swag_ensemble_low_rank_term_changes_samples_relative_to_diagonal_only():
    model = PreActResNet18(n_classes=10, rngs=nnx.Rngs(0))
    params = nnx.state(model, nnx.Param)
    swag_var = jax.tree.map(lambda p: jnp.full_like(p, 1e-6), params)
    flat_params, _ = jax.flatten_util.ravel_pytree(params)
    cov_rng = np.random.RandomState(3)
    swag_cov_mat_sqrt = jnp.asarray(
        np.stack(
            [
                cov_rng.normal(size=flat_params.shape).astype(np.asarray(flat_params).dtype) * 2e-2,
                cov_rng.normal(size=flat_params.shape).astype(np.asarray(flat_params).dtype) * 2e-2,
            ],
            axis=1,
        )
    )
    x_eval = jnp.array(np.random.RandomState(4).normal(size=(2, 32, 32, 3)).astype(np.float32))

    diag_only = SWAGEnsemble(
        model,
        params,
        swag_var,
        n_models=1,
        swag_cov_mat_sqrt=jnp.zeros((flat_params.shape[0], 0), dtype=flat_params.dtype),
        use_bn_refresh=False,
        seed=5,
    )
    full_swag = SWAGEnsemble(
        model,
        params,
        swag_var,
        n_models=1,
        swag_cov_mat_sqrt=swag_cov_mat_sqrt,
        use_bn_refresh=False,
        seed=5,
    )

    diag_logits = diag_only._sample_model()(x_eval, use_running_average=True)
    full_logits = full_swag._sample_model()(x_eval, use_running_average=True)

    assert not np.allclose(np.array(diag_logits), np.array(full_logits))


def test_train_resnet_swag_collects_snapshots():
    rng = np.random.RandomState(0)
    x_train = rng.normal(size=(8, 32, 32, 3)).astype(np.float32)
    y_train = rng.randint(0, 10, size=(8,), dtype=np.int32)
    x_val = rng.normal(size=(4, 32, 32, 3)).astype(np.float32)
    y_val = rng.randint(0, 10, size=(4,), dtype=np.int32)
    model = PreActResNet18(n_classes=10, rngs=nnx.Rngs(0))

    trained_model, swag_mean, swag_var, swag_cov_mat_sqrt, metadata = train_resnet_swag(
        model,
        x_train,
        y_train,
        x_val,
        y_val,
        epochs=2,
        batch_size=4,
        lr=0.01,
        warmup_epochs=0,
        cutout_size=0,
        swag_start_epoch=1,
        swag_collect_freq=1,
        swag_max_rank=3,
        seed=0,
    )

    assert metadata["variant"] == "full_swag"
    assert metadata["n_snapshots_collected"] == 2
    assert metadata["n_snapshots_retained"] == 2
    assert metadata["low_rank_rank"] == 2
    assert metadata["snapshot_epochs"] == [1, 2]
    assert len(jax.tree_util.tree_leaves(nnx.state(trained_model, nnx.Param))) == len(
        jax.tree_util.tree_leaves(swag_mean)
    )
    assert len(jax.tree_util.tree_leaves(swag_mean)) == len(jax.tree_util.tree_leaves(swag_var))
    assert swag_cov_mat_sqrt.shape[1] == 2
