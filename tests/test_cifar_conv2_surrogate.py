import numpy as np
import jax
import jax.numpy as jnp
from flax import nnx
from numpy.testing import assert_allclose

from cifar_tasks import make_cifar_block_get_Y_fn
from models import PreActResNet18
from pnc import (
    extract_patches,
    flatten_conv_kernel_to_patches,
    solve_chunked_conv2_correction,
)


def _make_cifar_block_context():
    model = PreActResNet18(n_classes=10, rngs=nnx.Rngs(0))
    blk = model.stage1[0]
    x = jax.random.normal(jax.random.key(1), (2, 32, 32, 3), dtype=jnp.float32)
    h = model.stem(x)
    return model, blk, h


def _conv2_preact_features(blk, h, w1):
    out_bn1 = blk.bn1(h, use_running_average=True)
    out_relu1 = jax.nn.relu(out_bn1)
    y_raw = jax.lax.conv_general_dilated(
        lhs=out_relu1,
        rhs=w1.transpose(3, 2, 0, 1),
        window_strides=tuple(blk.conv1.strides),
        padding="SAME",
        dimension_numbers=("NHWC", "OIHW", "NHWC"),
    )
    y_bn2 = blk.bn2(y_raw, use_running_average=True)
    return jax.nn.relu(y_bn2)


def _conv2_true_output(blk, y_relu2, w2, bias=None):
    out = jax.lax.conv_general_dilated(
        lhs=y_relu2,
        rhs=w2.transpose(3, 2, 0, 1),
        window_strides=tuple(blk.conv2.strides),
        padding="SAME",
        dimension_numbers=("NHWC", "OIHW", "NHWC"),
    )
    if bias is not None:
        out = out + bias
    return out


def _surrogate_output(blk, y_relu2, w2, bias=None, flatten_fn=flatten_conv_kernel_to_patches):
    Y = extract_patches(y_relu2, k=blk.conv2.kernel_size[0], strides=blk.conv2.strides[0])
    out = Y @ flatten_fn(w2)
    if bias is not None:
        out = out + bias
    return out.reshape(y_relu2.shape[0], y_relu2.shape[1], y_relu2.shape[2], w2.shape[-1])


def _metrics(pred, target):
    diff = np.asarray(pred - target)
    return {
        "max_abs": float(np.max(np.abs(diff))),
        "mean_abs": float(np.mean(np.abs(diff))),
        "rmse": float(np.sqrt(np.mean(diff**2))),
    }


def _legacy_flatten(w2):
    return w2.reshape(-1, w2.shape[-1])


def test_original_kernel_surrogate_matches_real_conv2_for_cifar_block():
    _, blk, h = _make_cifar_block_context()
    w1_orig = blk.conv1.kernel[...]
    w2_orig = blk.conv2.kernel[...]

    y_relu2 = _conv2_preact_features(blk, h, w1_orig)
    true_out = _conv2_true_output(blk, y_relu2, w2_orig)
    surrogate_out = _surrogate_output(blk, y_relu2, w2_orig)

    metrics = _metrics(surrogate_out, true_out)
    assert_allclose(np.asarray(surrogate_out), np.asarray(true_out), atol=1e-5, rtol=1e-5)
    assert metrics["max_abs"] <= 1e-5


def test_perturbed_kernel_and_bias_surrogate_match_real_conv2():
    _, blk, h = _make_cifar_block_context()
    w1_orig = blk.conv1.kernel[...]
    w2_pert = 0.1 * jax.random.normal(jax.random.key(2), blk.conv2.kernel[...].shape, dtype=jnp.float32)
    bias = 0.1 * jax.random.normal(jax.random.key(3), (1, w2_pert.shape[-1]), dtype=jnp.float32)

    y_relu2 = _conv2_preact_features(blk, h, w1_orig)
    true_out = _conv2_true_output(blk, y_relu2, w2_pert, bias=bias)
    surrogate_out = _surrogate_output(blk, y_relu2, w2_pert, bias=bias)

    metrics = _metrics(surrogate_out, true_out)
    assert_allclose(np.asarray(surrogate_out), np.asarray(true_out), atol=1e-5, rtol=1e-5)
    assert metrics["max_abs"] <= 1e-5


def test_convention_search_identifies_correct_patch_order_and_rejects_legacy_reshape():
    _, blk, h = _make_cifar_block_context()
    w1_orig = blk.conv1.kernel[...]
    w2 = 0.1 * jax.random.normal(jax.random.key(4), blk.conv2.kernel[...].shape, dtype=jnp.float32)
    y_relu2 = _conv2_preact_features(blk, h, w1_orig)
    true_out = _conv2_true_output(blk, y_relu2, w2)

    candidates = {
        "legacy_hwio_reshape": lambda kernel: kernel.reshape(-1, kernel.shape[-1]),
        "channel_first_patch_order": flatten_conv_kernel_to_patches,
        "transpose_0213": lambda kernel: kernel.transpose(0, 2, 1, 3).reshape(-1, kernel.shape[-1]),
        "transpose_1023": lambda kernel: kernel.transpose(1, 0, 2, 3).reshape(-1, kernel.shape[-1]),
    }
    errors = {
        name: _metrics(_surrogate_output(blk, y_relu2, w2, flatten_fn=fn), true_out)
        for name, fn in candidates.items()
    }

    best_name = min(errors, key=lambda name: errors[name]["max_abs"])
    assert best_name == "channel_first_patch_order"
    assert errors["channel_first_patch_order"]["max_abs"] <= 1e-5
    assert errors["legacy_hwio_reshape"]["max_abs"] > 1.0


def test_ridge_corrected_kernel_surrogate_matches_actual_conv2_after_solve():
    _, blk, h = _make_cifar_block_context()
    w1_orig = blk.conv1.kernel[...]
    w2_orig = blk.conv2.kernel[...]
    get_Y_fn = make_cifar_block_get_Y_fn(blk)

    y_relu2_orig = _conv2_preact_features(blk, h, w1_orig)
    T_orig = _conv2_true_output(blk, y_relu2_orig, w2_orig)

    w1_pert = w1_orig + 0.05 * jax.random.normal(jax.random.key(5), w1_orig.shape, dtype=jnp.float32)
    w2_new, b2_new = solve_chunked_conv2_correction(
        get_Y_fn,
        w1_pert,
        w2_orig,
        [h],
        [T_orig],
        lambda_reg=1e-3,
    )

    y_relu2_pert = _conv2_preact_features(blk, h, w1_pert)
    actual_corrected = _conv2_true_output(blk, y_relu2_pert, w2_new, bias=b2_new)
    surrogate_corrected = _surrogate_output(blk, y_relu2_pert, w2_new, bias=b2_new)

    metrics = _metrics(surrogate_corrected, actual_corrected)
    assert_allclose(
        np.asarray(surrogate_corrected),
        np.asarray(actual_corrected),
        atol=1e-5,
        rtol=1e-5,
    )
    assert metrics["max_abs"] <= 1e-5

    legacy_metrics = _metrics(
        _surrogate_output(blk, y_relu2_pert, w2_new, bias=b2_new, flatten_fn=_legacy_flatten),
        actual_corrected,
    )
    assert legacy_metrics["max_abs"] > 1.0
