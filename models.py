from flax import nnx
from typing import Callable
import jax
import jax.numpy as jnp

class TransitionModel(nnx.Module):
    def __init__(self, in_features, out_features, rngs: nnx.Rngs,
                 activation: Callable = nnx.relu):
        self.l1 = nnx.Linear(in_features, 64, rngs=rngs)
        self.l2 = nnx.Linear(64, 64, rngs=rngs)
        self.l3 = nnx.Linear(64, out_features, rngs=rngs)
        self.activation = activation

    def __call__(self, x):
        h1 = self.activation(self.l1(x))
        h2 = self.activation(self.l2(h1))
        out = self.l3(h2)
        return out


class MCDropoutTransitionModel(nnx.Module):
    def __init__(self, in_features, out_features, rngs: nnx.Rngs,
                 dropout_rate: float = 0.1, activation: Callable = nnx.relu):
        self.l1 = nnx.Linear(in_features, 64, rngs=rngs)
        self.dropout1 = nnx.Dropout(dropout_rate, rngs=rngs)
        self.l2 = nnx.Linear(64, 64, rngs=rngs)
        self.dropout2 = nnx.Dropout(dropout_rate, rngs=rngs)
        self.l3 = nnx.Linear(64, out_features, rngs=rngs)
        self.activation = activation

    def __call__(self, x, deterministic: bool = False):
        h1 = self.activation(self.l1(x))
        h1 = self.dropout1(h1, deterministic=deterministic)
        h2 = self.activation(self.l2(h1))
        h2 = self.dropout2(h2, deterministic=deterministic)
        out = self.l3(h2)
        return out


class ClassificationModel(nnx.Module):
    def __init__(self, in_features, out_features, rngs: nnx.Rngs,
                 activation: Callable = nnx.relu):
        self.l1 = nnx.Linear(in_features, 200, rngs=rngs)
        self.l2 = nnx.Linear(200, 200, rngs=rngs)
        self.l3 = nnx.Linear(200, 200, rngs=rngs)
        self.l4 = nnx.Linear(200, out_features, rngs=rngs)
        self.activation = activation

    def __call__(self, x):
        h1 = self.activation(self.l1(x))
        h2 = self.activation(self.l2(h1))
        h3 = self.activation(self.l3(h2))
        out = self.l4(h3)
        return out


class MCDropoutClassificationModel(nnx.Module):
    def __init__(self, in_features, out_features, rngs: nnx.Rngs,
                 dropout_rate: float = 0.5, activation: Callable = nnx.relu):
        self.l1 = nnx.Linear(in_features, 200, rngs=rngs)
        self.dropout1 = nnx.Dropout(dropout_rate, rngs=rngs)
        self.l2 = nnx.Linear(200, 200, rngs=rngs)
        self.dropout2 = nnx.Dropout(dropout_rate, rngs=rngs)
        self.l3 = nnx.Linear(200, 200, rngs=rngs)
        self.dropout3 = nnx.Dropout(dropout_rate, rngs=rngs)
        self.l4 = nnx.Linear(200, out_features, rngs=rngs)
        self.activation = activation

    def __call__(self, x, deterministic: bool = False):
        h1 = self.activation(self.l1(x))
        h1 = self.dropout1(h1, deterministic=deterministic)
        h2 = self.activation(self.l2(h1))
        h2 = self.dropout2(h2, deterministic=deterministic)
        h3 = self.activation(self.l3(h2))
        h3 = self.dropout3(h3, deterministic=deterministic)
        out = self.l4(h3)
        return out

class RegressionModel(nnx.Module):
    def __init__(self, in_features, out_features, rngs: nnx.Rngs,
                 hidden_dims: list[int] = [50],
                 activation: Callable = nnx.relu):
        layers = []
        dims = [in_features] + hidden_dims + [out_features]
        for i in range(len(dims) - 1):
            layers.append(nnx.Linear(dims[i], dims[i+1], rngs=rngs))
        self.layers = nnx.List(layers)
        self.activation = activation

    def __call__(self, x):
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        out = self.layers[-1](x)
        return out

class MCDropoutRegressionModel(nnx.Module):
    def __init__(self, in_features, out_features, rngs: nnx.Rngs,
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

    def __call__(self, x, deterministic: bool = False):
        for i, layer in enumerate(self.layers[:-1]):
            x = self.activation(layer(x))
            x = self.dropouts[i](x, deterministic=deterministic)
        out = self.layers[-1](x)
        return out


class ProbabilisticRegressionModel(nnx.Module):
    def __init__(self, in_features, out_features, rngs: nnx.Rngs,
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

    def __call__(self, x):
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

    def __call__(self, x, use_running_average: bool = False):
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

    def __call__(self, x, use_running_average: bool = False):
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
        for blk in self.stage1: x = blk(x, use_running_average=use_running_average)
        for blk in self.stage2: x = blk(x, use_running_average=use_running_average)
        for blk in self.stage3: x = blk(x, use_running_average=use_running_average)
        for blk in self.stage4: x = blk(x, use_running_average=use_running_average)
        return x

    def __call__(self, x, use_running_average: bool = False):
        # x: (N, 32, 32, 3)
        x = jax.nn.relu(self.stem(x, use_running_average=use_running_average))
        x = self._run_stages(x, use_running_average=use_running_average)
        x = jnp.mean(x, axis=(1, 2))   # global average pool → (N, 2048)
        return self.fc(x)

    def stem_out(self, x, use_running_average: bool = True):
        """Post-stem-BN-ReLU activations, shape (N, 32, 32, 64)."""
        return jax.nn.relu(self.stem(x, use_running_average=use_running_average))

    def forward_from_stem_out(self, h, use_running_average: bool = True):
        """Complete forward from stem activations through stages + head."""
        h = self._run_stages(h, use_running_average=use_running_average)
        h = jnp.mean(h, axis=(1, 2))
        return self.fc(h)

    def stem_conv_out_raw(self, x):
        """Raw post-conv (before BN) activations, shape (N, 32, 32, 64)."""
        return self.stem.conv(x)

    def stem_bn_from_raw(self, raw_conv_out, use_running_average: bool = True):
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

    def __call__(self, x, use_running_average: bool = False,
                 deterministic: bool = False):
        x = jax.nn.relu(self.stem(x, use_running_average=use_running_average))
        x = self._run_stages(x, use_running_average=use_running_average)
        x = jnp.mean(x, axis=(1, 2))
        x = self.dropout(x, deterministic=deterministic)
        return self.fc(x)

