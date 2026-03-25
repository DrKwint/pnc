from __future__ import annotations
from flax import nnx
from typing import Callable
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

class TransitionModel(nnx.Module):
    def __init__(self, in_features: int, out_features: int, rngs: nnx.Rngs,
                 activation: Callable = nnx.relu):
        self.l1 = nnx.Linear(in_features, 64, rngs=rngs)
        self.l2 = nnx.Linear(64, 64, rngs=rngs)
        self.l3 = nnx.Linear(64, out_features, rngs=rngs)
        self.activation = activation

    def __call__(self, x: jax.Array) -> jax.Array:
        h1 = self.activation(self.l1(x))
        h2 = self.activation(self.l2(h1))
        out = self.l3(h2)
        return out


class MCDropoutTransitionModel(nnx.Module):
    def __init__(self, in_features: int, out_features: int, rngs: nnx.Rngs,
                 dropout_rate: float = 0.1, activation: Callable = nnx.relu):
        self.l1 = nnx.Linear(in_features, 64, rngs=rngs)
        self.dropout1 = nnx.Dropout(dropout_rate, rngs=rngs)
        self.l2 = nnx.Linear(64, 64, rngs=rngs)
        self.dropout2 = nnx.Dropout(dropout_rate, rngs=rngs)
        self.l3 = nnx.Linear(64, out_features, rngs=rngs)
        self.activation = activation

    def __call__(self, x: jax.Array, deterministic: bool = False) -> jax.Array:
        h1 = self.activation(self.l1(x))
        h1 = self.dropout1(h1, deterministic=deterministic)
        h2 = self.activation(self.l2(h1))
        h2 = self.dropout2(h2, deterministic=deterministic)
        out = self.l3(h2)
        return out


class ClassificationModel(nnx.Module):
    def __init__(self, in_features: int, out_features: int, rngs: nnx.Rngs,
                 activation: Callable = nnx.relu):
        self.l1 = nnx.Linear(in_features, 200, rngs=rngs)
        self.l2 = nnx.Linear(200, 200, rngs=rngs)
        self.l3 = nnx.Linear(200, 200, rngs=rngs)
        self.l4 = nnx.Linear(200, out_features, rngs=rngs)
        self.activation = activation

    def __call__(self, x: jax.Array) -> jax.Array:
        h1 = self.activation(self.l1(x))
        h2 = self.activation(self.l2(h1))
        h3 = self.activation(self.l3(h2))
        out = self.l4(h3)
        return out


class MCDropoutClassificationModel(nnx.Module):
    def __init__(self, in_features: int, out_features: int, rngs: nnx.Rngs,
                 dropout_rate: float = 0.5, activation: Callable = nnx.relu):
        self.l1 = nnx.Linear(in_features, 200, rngs=rngs)
        self.dropout1 = nnx.Dropout(dropout_rate, rngs=rngs)
        self.l2 = nnx.Linear(200, 200, rngs=rngs)
        self.dropout2 = nnx.Dropout(dropout_rate, rngs=rngs)
        self.l3 = nnx.Linear(200, 200, rngs=rngs)
        self.dropout3 = nnx.Dropout(dropout_rate, rngs=rngs)
        self.l4 = nnx.Linear(200, out_features, rngs=rngs)
        self.activation = activation

    def __call__(self, x: jax.Array, deterministic: bool = False) -> jax.Array:
        h1 = self.activation(self.l1(x))
        h1 = self.dropout1(h1, deterministic=deterministic)
        h2 = self.activation(self.l2(h1))
        h2 = self.dropout2(h2, deterministic=deterministic)
        h3 = self.activation(self.l3(h2))
        h3 = self.dropout3(h3, deterministic=deterministic)
        out = self.l4(h3)
        return out

class RegressionModel(nnx.Module):
    def __init__(self, in_features: int, out_features: int, rngs: nnx.Rngs,
                 hidden_dims: list[int] = [50],
                 activation: Callable = nnx.relu):
        layers = []
        dims = [in_features] + hidden_dims + [out_features]
        for i in range(len(dims) - 1):
            layers.append(nnx.Linear(dims[i], dims[i+1], rngs=rngs))
        self.layers = nnx.List(layers)
        self.activation = activation

    def __call__(self, x: jax.Array) -> jax.Array:
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        out = self.layers[-1](x)
        return out

class MCDropoutRegressionModel(nnx.Module):
    def __init__(self, in_features: int, out_features: int, rngs: nnx.Rngs,
                 hidden_dims: list[int] = [50],
                 dropout_rate: float = 0.05,
                 activation: Callable = nnx.relu):
        layers = []
        dropouts = []
        dims = [in_features] + hidden_dims + [out_features]
        for i in range(len(dims) - 1):
            layers.append(nnx.Linear(dims[i], dims[i+1], rngs=rngs))
            if i < len(dims) - 2: # No dropout before last layer
                dropouts.append(nnx.Dropout(dropout_rate, rngs=rngs))
        self.layers = nnx.List(layers)
        self.dropouts = nnx.List(dropouts)
        self.activation = activation

    def __call__(self, x: jax.Array, deterministic: bool = False) -> jax.Array:
        for i, layer in enumerate(self.layers[:-1]):
            x = self.activation(layer(x))
            x = self.dropouts[i](x, deterministic=deterministic)
        out = self.layers[-1](x)
        return out


class ProbabilisticRegressionModel(nnx.Module):
    def __init__(self, in_features: int, out_features: int, rngs: nnx.Rngs,
                 hidden_dims: list[int] = [50],
                 activation: Callable = nnx.relu):
        layers = []
        dims = [in_features] + hidden_dims
        for i in range(len(dims) - 1):
            layers.append(nnx.Linear(dims[i], dims[i+1], rngs=rngs))
        self.layers = nnx.List(layers)

        # Two output heads: one for mean, one for variance (pre-softplus)
        self.mean_layer = nnx.Linear(hidden_dims[-1], out_features, rngs=rngs)
        self.var_layer = nnx.Linear(hidden_dims[-1], out_features, rngs=rngs)
        self.activation = activation

    def __call__(self, x: jax.Array) -> tuple[jax.Array, jax.Array]:
        for layer in self.layers:
            x = self.activation(layer(x))

        mean = self.mean_layer(x)
        # Softplus to ensure strictly positive variance, plus a small epsilon for stability
        var = jax.nn.softplus(self.var_layer(x)) + 1e-6
        return mean, var


# ===========================================================================
# ResNet-50 for CIFAR (2D Conv, BatchNorm, NHWC input)
# ===========================================================================

class _BNConvFixed(nnx.Module):
    """Conv2D (explicit in_features) + BatchNorm, no activation."""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 strides: int, use_bias: bool = False, rngs: nnx.Rngs = None):
        self.conv = nnx.Conv(
            in_features=in_channels,
            out_features=out_channels,
            kernel_size=(kernel_size, kernel_size),
            strides=(strides, strides),
            padding='SAME',
            use_bias=use_bias,
            rngs=rngs,
        )
        self.bn = nnx.BatchNorm(num_features=out_channels, rngs=rngs)

    def __call__(self, x: jax.Array, use_running_average: bool = False) -> jax.Array:
        return self.bn(self.conv(x), use_running_average=use_running_average)


class _Bottleneck(nnx.Module):
    """ResNet-50 bottleneck: 1×1 → 3×3 → 1×1, each with Conv+BN+ReLU."""
    expansion = 4

    def __init__(self, in_channels: int, planes: int, strides: int = 1,
                 rngs: nnx.Rngs = None):
        out_channels = planes * self.expansion
        self.conv1 = _BNConvFixed(in_channels, planes,       kernel_size=1, strides=1, rngs=rngs)
        self.conv2 = _BNConvFixed(planes,      planes,       kernel_size=3, strides=strides, rngs=rngs)
        self.conv3 = _BNConvFixed(planes,      out_channels, kernel_size=1, strides=1, rngs=rngs)

        self.downsample = (
            _BNConvFixed(in_channels, out_channels, kernel_size=1, strides=strides, rngs=rngs)
            if strides != 1 or in_channels != out_channels
            else None
        )

    _SHAPE_IN = "batch H W in_channels"
    _SHAPE_OUT = "batch H_out W_out out_channels"

    def __call__(self, x: Float[Array, _SHAPE_IN], use_running_average: bool = False) -> Float[Array, _SHAPE_OUT]:
        identity = x
        out = jax.nn.relu(self.conv1(x,   use_running_average=use_running_average))
        out = jax.nn.relu(self.conv2(out, use_running_average=use_running_average))
        out = self.conv3(out, use_running_average=use_running_average)
        if self.downsample is not None:
            identity = self.downsample(x, use_running_average=use_running_average)
        return jax.nn.relu(out + identity)


def _make_resnet_stage(in_channels: int, planes: int, n_blocks: int,
                       strides: int, rngs: nnx.Rngs):
    """Build one ResNet stage, return (nnx.List of blocks, out_channels)."""
    blocks = []
    for i in range(n_blocks):
        s   = strides if i == 0 else 1
        inp = in_channels if i == 0 else planes * _Bottleneck.expansion
        blocks.append(_Bottleneck(inp, planes, strides=s, rngs=rngs))
    return nnx.List(blocks), planes * _Bottleneck.expansion


class ResNet50(nnx.Module):
    """
    ResNet-50 adapted for CIFAR-10/100 (32×32 input, NHWC).

    Differences from ImageNet ResNet-50:
      - Stem: 3×3 conv / stride 1 (instead of 7×7 / stride 2 + max-pool)
      - Stages: 3-4-6-3 bottleneck blocks (same as standard)
      - Output: raw logits of shape (N, n_classes)

    Every Conv is followed immediately by BatchNorm (no bias), making the
    BatchNorm Refit trick directly applicable after the stem conv.
    """

    def __init__(self, n_classes: int = 10, rngs: nnx.Rngs = None):
        # CIFAR stem: 3×3, 64-channel, stride 1, then BN
        self.stem = _BNConvFixed(3, 64, kernel_size=3, strides=1, rngs=rngs)

        self.stage1, c1 = _make_resnet_stage(64,  64,  n_blocks=3, strides=1, rngs=rngs)
        self.stage2, c2 = _make_resnet_stage(c1,  128, n_blocks=4, strides=2, rngs=rngs)
        self.stage3, c3 = _make_resnet_stage(c2,  256, n_blocks=6, strides=2, rngs=rngs)
        self.stage4, c4 = _make_resnet_stage(c3,  512, n_blocks=3, strides=2, rngs=rngs)

        self.fc = nnx.Linear(c4, n_classes, rngs=rngs)

    def _run_stages(self, x, use_running_average: bool):
        for blk in self.stage1:
            x = blk(x, use_running_average=use_running_average)
        for blk in self.stage2:
            x = blk(x, use_running_average=use_running_average)
        for blk in self.stage3:
            x = blk(x, use_running_average=use_running_average)
        for blk in self.stage4:
            x = blk(x, use_running_average=use_running_average)
        return x

    _SHAPE_X = "batch H W C"
    _SHAPE_OUT = "batch n_classes"
    def __call__(self, x: Float[Array, _SHAPE_X], use_running_average: bool = False) -> Float[Array, _SHAPE_OUT]:
        # x: (N, 32, 32, 3)
        x = jax.nn.relu(self.stem(x, use_running_average=use_running_average))
        x = self._run_stages(x, use_running_average=use_running_average)
        x = jnp.mean(x, axis=(1, 2))   # global average pool → (N, 2048)
        return self.fc(x)

    _SHAPE_STEM_IN = "batch H W C"
    _SHAPE_STEM_OUT = "batch H W C_out"
    def stem_out(self, x: Float[Array, _SHAPE_STEM_IN], use_running_average: bool = True) -> Float[Array, _SHAPE_STEM_OUT]:
        """Post-stem-BN-ReLU activations, shape (N, 32, 32, 64)."""
        return jax.nn.relu(self.stem(x, use_running_average=use_running_average))

    _SHAPE_FORWARD_FROM_STEM_IN = "batch H W C_in"
    _SHAPE_FORWARD_FROM_STEM_OUT = "batch n_classes"
    def forward_from_stem_out(self, h: Float[Array, _SHAPE_FORWARD_FROM_STEM_IN], use_running_average: bool = True) -> Float[Array, _SHAPE_FORWARD_FROM_STEM_OUT]:
        """Complete forward from stem activations through stages + head."""
        h = self._run_stages(h, use_running_average=use_running_average)
        h = jnp.mean(h, axis=(1, 2))
        return self.fc(h)

    _SHAPE_STEM_CONV_RAW_IN = "batch H W C"
    _SHAPE_STEM_CONV_RAW_OUT = "batch H W C_out"
    def stem_conv_out_raw(self, x: Float[Array, _SHAPE_STEM_CONV_RAW_IN]) -> Float[Array, _SHAPE_STEM_CONV_RAW_OUT]:
        """Raw post-conv (before BN) activations, shape (N, 32, 32, 64)."""
        return self.stem.conv(x)

    _SHAPE_STEM_BN_FROM_RAW_IN = "batch H W C_in"
    _SHAPE_STEM_BN_FROM_RAW_OUT = "batch H W C_in"
    def stem_bn_from_raw(self, raw_conv_out: Float[Array, _SHAPE_STEM_BN_FROM_RAW_IN], use_running_average: bool = True) -> Float[Array, _SHAPE_STEM_BN_FROM_RAW_OUT]:
        """Apply stem BN to raw conv output (used by BN Refit)."""
        return self.stem.bn(raw_conv_out, use_running_average=use_running_average)


class MCDropoutResNet50(ResNet50):
    """
    ResNet-50 with MC Dropout before the classification head.
    All conv layers are unchanged; uncertainty comes from head dropout.
    """
    def __init__(self, n_classes: int = 10, dropout_rate: float = 0.1,
                 rngs: nnx.Rngs = None):
        super().__init__(n_classes=n_classes, rngs=rngs)
        self.dropout = nnx.Dropout(rate=dropout_rate, rngs=rngs)

    def __call__(self, x: Float[Array, "batch H W C"], use_running_average: bool = False,
                 deterministic: bool = False) -> Float[Array, "batch n_classes"]:
        x = jax.nn.relu(self.stem(x, use_running_average=use_running_average))
        x = self._run_stages(x, use_running_average=use_running_average)
        x = jnp.mean(x, axis=(1, 2))
        x = self.dropout(x, deterministic=deterministic)
        return self.fc(x)


# ===========================================================================
# PreAct ResNet-18 for CIFAR
# ===========================================================================

class _PreActBasicBlock(nnx.Module):
    """Standard Pre-Activation ResNet basic block (BN -> ReLU -> Conv -> BN -> ReLU -> Conv)."""
    def __init__(self, in_channels: int, out_channels: int, strides: int = 1,
                 rngs: nnx.Rngs = None):
        self.bn1 = nnx.BatchNorm(num_features=in_channels, rngs=rngs, momentum=0.9)
        self.conv1 = nnx.Conv(in_features=in_channels, out_features=out_channels, 
                              kernel_size=(3, 3), strides=(strides, strides), padding='SAME', use_bias=False, rngs=rngs)
        self.bn2 = nnx.BatchNorm(num_features=out_channels, rngs=rngs, momentum=0.9)
        self.conv2 = nnx.Conv(in_features=out_channels, out_features=out_channels, 
                              kernel_size=(3, 3), strides=(1, 1), padding='SAME', use_bias=False, rngs=rngs)

        self.downsample = (
            nnx.Conv(in_features=in_channels, out_features=out_channels, 
                     kernel_size=(1, 1), strides=(strides, strides), padding='SAME', use_bias=False, rngs=rngs)
            if strides != 1 or in_channels != out_channels
            else None
        )

    def __call__(self, x: jax.Array, use_running_average: bool = False) -> jax.Array:
        # 1. First Pre-Activation
        out_bn1 = self.bn1(x, use_running_average=use_running_average)
        out_relu1 = jax.nn.relu(out_bn1)
        
        # 2. Main Branch
        y = self.conv1(out_relu1)
        y = self.bn2(y, use_running_average=use_running_average)
        y = jax.nn.relu(y)
        y = self.conv2(y)
        
        # 3. Shortcut Branch
        if self.downsample is not None:
            # Apply the 1x1 projection to the normalized, activated tensor
            identity = self.downsample(out_relu1) 
        else:
            # If dimensions match, keep the true identity mapping clean
            identity = x
            
        return y + identity


def _make_preact_resnet_stage(in_channels: int, out_channels: int, n_blocks: int,
                              strides: int, rngs: nnx.Rngs):
    blocks = []
    for i in range(n_blocks):
        s = strides if i == 0 else 1
        inp = in_channels if i == 0 else out_channels
        blocks.append(_PreActBasicBlock(inp, out_channels, strides=s, rngs=rngs))
    return nnx.List(blocks)


class PreActResNet18(nnx.Module):
    """
    PreAct ResNet-18 adapted for CIFAR.
    Features a 3x3 stride-1 stem (no maxpool) followed by 4 stages of [2, 2, 2, 2] blocks.
    Channels: 64, 128, 256, 512.
    """
    def __init__(self, n_classes: int = 10, rngs: nnx.Rngs = None):
        self.stem = nnx.Conv(in_features=3, out_features=64, kernel_size=(3, 3), strides=(1, 1), padding='SAME', use_bias=False, rngs=rngs)
        
        self.stage1 = _make_preact_resnet_stage(64, 64, n_blocks=2, strides=1, rngs=rngs)
        self.stage2 = _make_preact_resnet_stage(64, 128, n_blocks=2, strides=2, rngs=rngs)
        self.stage3 = _make_preact_resnet_stage(128, 256, n_blocks=2, strides=2, rngs=rngs)
        self.stage4 = _make_preact_resnet_stage(256, 512, n_blocks=2, strides=2, rngs=rngs)
        
        self.final_bn = nnx.BatchNorm(num_features=512, rngs=rngs, momentum=0.9)
        self.fc = nnx.Linear(512, n_classes, rngs=rngs)

    def _run_stages(self, x, use_running_average: bool):
        for blk in self.stage1:
            x = blk(x, use_running_average=use_running_average)
        for blk in self.stage2:
            x = blk(x, use_running_average=use_running_average)
        for blk in self.stage3:
            x = blk(x, use_running_average=use_running_average)
        for blk in self.stage4:
            x = blk(x, use_running_average=use_running_average)
        return x

    def __call__(self, x: Float[Array, "batch H W C"], use_running_average: bool = False) -> Float[Array, "batch n_classes"]:
        x = self.stem(x)
        x = self._run_stages(x, use_running_average=use_running_average)
        x = self.final_bn(x, use_running_average=use_running_average)
        x = jax.nn.relu(x)
        x = jnp.mean(x, axis=(1, 2))
        return self.fc(x)

    def forward_from_stem_out(self, h: Float[Array, "batch H W C_in"], use_running_average: bool = True) -> Float[Array, "batch n_classes"]:
        h = self._run_stages(h, use_running_average=use_running_average)
        h = self.final_bn(h, use_running_average=use_running_average)
        h = jax.nn.relu(h)
        h = jnp.mean(h, axis=(1, 2))
        return self.fc(h)


class MCDropoutPreActResNet18(PreActResNet18):
    """
    PreAct ResNet-18 with MC Dropout before the classification head.
    """
    def __init__(self, n_classes: int = 10, dropout_rate: float = 0.1,
                 rngs: nnx.Rngs = None):
        super().__init__(n_classes=n_classes, rngs=rngs)
        self.dropout = nnx.Dropout(rate=dropout_rate, rngs=rngs)

    def __call__(self, x: Float[Array, "batch H W C"], use_running_average: bool = False,
                 deterministic: bool = False) -> Float[Array, "batch n_classes"]:
        x = self.stem(x)
        x = self._run_stages(x, use_running_average=use_running_average)
        x = self.final_bn(x, use_running_average=use_running_average)
        x = jax.nn.relu(x)
        x = jnp.mean(x, axis=(1, 2))
        x = self.dropout(x, deterministic=deterministic)
        return self.fc(x)
