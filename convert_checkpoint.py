"""
Checkpoint conversion utility: TensorFlow → JAX/Flax

Loads a TensorFlow checkpoint from uncertainty_baselines, converts weights to NumPy,
initializes matching Flax model, copies arrays, and saves as JAX checkpoint using Orbax.
"""

import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf
from flax import nnx
import orbax.checkpoint as ocp
from pathlib import Path
from models import WideResNet


def convert_tf_checkpoint_to_jax(
    tf_checkpoint_path: str,
    jax_checkpoint_path: str,
    depth: int = 28,
    width_multiplier: int = 10,
    num_classes: int = 10,
    input_shape: tuple = (32, 32, 3)
):
    """
    Convert TF checkpoint to JAX/Flax checkpoint.

    Args:
        tf_checkpoint_path: Path to TF checkpoint directory (containing .index, .data files)
        jax_checkpoint_path: Path to save JAX checkpoint
        depth, width_multiplier, num_classes: Model config
        input_shape: Input shape for model
    """
    print(f"Converting TF checkpoint from {tf_checkpoint_path} to JAX at {jax_checkpoint_path}")

    # 1. Instantiate and load TF model
    # Use manual TF Wide ResNet instead of ub.models.wide_resnet to avoid dependency issues
    class BasicBlock(tf.keras.layers.Layer):
        def __init__(self, in_channels, out_channels, strides=1):
            super().__init__()
            self.conv1 = tf.keras.layers.Conv2D(out_channels, 3, strides=strides, padding='same', use_bias=False)
            self.bn1 = tf.keras.layers.BatchNormalization()
            self.conv2 = tf.keras.layers.Conv2D(out_channels, 3, strides=1, padding='same', use_bias=False)
            self.bn2 = tf.keras.layers.BatchNormalization()
            self.downsample = None
            if strides != 1 or in_channels != out_channels:
                self.downsample = tf.keras.Sequential([
                    tf.keras.layers.Conv2D(out_channels, 1, strides=strides, use_bias=False),
                    tf.keras.layers.BatchNormalization()
                ])

        def call(self, x, training=False):
            identity = x
            out = tf.nn.relu(self.bn1(self.conv1(x), training=training))
            out = self.bn2(self.conv2(out), training=training)
            if self.downsample:
                identity = self.downsample(x, training=training)
            return tf.nn.relu(out + identity)

    # Build TF model matching JAX WideResNet
    inputs = tf.keras.Input(shape=input_shape)
    k = width_multiplier
    x = tf.keras.layers.Conv2D(16 * k, 3, padding='same', use_bias=False)(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    # Stages
    n = (depth - 4) // 6
    for i in range(n):
        x = BasicBlock(16 * k, 16 * k, strides=1 if i > 0 else 1)(x)
    for i in range(n):
        x = BasicBlock(16 * k, 32 * k, strides=2 if i == 0 else 1)(x)
    for i in range(n):
        x = BasicBlock(32 * k, 64 * k, strides=2 if i == 0 else 1)(x)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    outputs = tf.keras.layers.Dense(num_classes)(x)
    tf_model = tf.keras.Model(inputs, outputs)

    # Load TF checkpoint
    checkpoint = tf.train.Checkpoint(model=tf_model)
    checkpoint.restore(tf_checkpoint_path).expect_partial()
    print("TF checkpoint loaded")

    # 2. Initialize JAX model
    rngs = nnx.Rngs(params=0)  # Dummy RNG for init
    jax_model = WideResNet(depth=depth, width_multiplier=width_multiplier, n_classes=num_classes, rngs=rngs)

    # 3. Convert and copy weights
    # This requires mapping TF variable names to Flax variable tree
    # For simplicity, assume we can access tf_model variables and match by structure

    # Get TF variables
    tf_vars = tf_model.variables
    tf_var_dict = {v.name: v.numpy() for v in tf_vars}

    # Get JAX variable tree
    jax_state = nnx.state(jax_model)

    # Manual mapping (this needs to be adjusted based on actual names)
    # Example mappings - need to inspect actual names
    mappings = {
        # Stem
        'wide_resnet/conv2d/kernel:0': ('stem', 'conv', 'kernel'),
        'wide_resnet/conv2d/bias:0': ('stem', 'conv', 'bias'),  # if bias
        'wide_resnet/batch_normalization/gamma:0': ('stem', 'bn', 'scale'),
        'wide_resnet/batch_normalization/beta:0': ('stem', 'bn', 'bias'),
        'wide_resnet/batch_normalization/moving_mean:0': ('stem', 'bn', 'mean'),
        'wide_resnet/batch_normalization/moving_variance:0': ('stem', 'bn', 'var'),
        # And so on for stages and fc
    }

    # Apply mappings
    for tf_name, jax_path in mappings.items():
        if tf_name in tf_var_dict:
            np_array = tf_var_dict[tf_name]
            # Set in jax_state
            # This is pseudo-code; actual implementation needs careful path matching
            set_nested(jax_state, jax_path, jnp.array(np_array))

    # Update JAX model
    nnx.update(jax_model, jax_state)

    # 4. Save JAX checkpoint as pickle (easier to load)
    import pickle
    Path(jax_checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
    with open(jax_checkpoint_path, 'wb') as f:
        pickle.dump({'state': jax_state}, f)
    print(f"JAX checkpoint saved to {jax_checkpoint_path}")


def set_nested(dct, keys, value):
    """Set nested dict value by key path."""
    for key in keys[:-1]:
        dct = dct[key]
    dct[keys[-1]] = value


def create_test_tf_checkpoint(checkpoint_path: str, depth: int = 28, width_multiplier: int = 10, num_classes: int = 10):
    """Create a test TF Wide ResNet checkpoint by training briefly."""
    print(f"Creating test TF Wide ResNet checkpoint at {checkpoint_path}")

    # Simple TF implementation of Wide ResNet basic block
    class BasicBlock(tf.keras.layers.Layer):
        def __init__(self, in_channels, out_channels, strides=1):
            super().__init__()
            self.conv1 = tf.keras.layers.Conv2D(out_channels, 3, strides=strides, padding='same', use_bias=False)
            self.bn1 = tf.keras.layers.BatchNormalization()
            self.conv2 = tf.keras.layers.Conv2D(out_channels, 3, strides=1, padding='same', use_bias=False)
            self.bn2 = tf.keras.layers.BatchNormalization()
            self.downsample = None
            if strides != 1 or in_channels != out_channels:
                self.downsample = tf.keras.Sequential([
                    tf.keras.layers.Conv2D(out_channels, 1, strides=strides, use_bias=False),
                    tf.keras.layers.BatchNormalization()
                ])

        def call(self, x):
            identity = x
            out = tf.keras.layers.ReLU()(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            if self.downsample:
                identity = self.downsample(x)
            return tf.keras.layers.ReLU()(out + identity)

    # Build model
    inputs = tf.keras.Input(shape=(32, 32, 3))
    k = width_multiplier
    x = tf.keras.layers.Conv2D(16 * k, 3, padding='same', use_bias=False)(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.nn.relu(x)

    # Stages
    n = (depth - 4) // 6
    for i in range(n):
        x = BasicBlock(16 * k, 16 * k, strides=1 if i > 0 else 1)(x)
    for i in range(n):
        x = BasicBlock(16 * k, 32 * k, strides=2 if i == 0 else 1)(x)
    for i in range(n):
        x = BasicBlock(32 * k, 64 * k, strides=2 if i == 0 else 1)(x)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    outputs = tf.keras.layers.Dense(num_classes)(x)
    model = tf.keras.Model(inputs, outputs)

    # Quick training
    x_dummy = tf.random.normal((10, 32, 32, 3))
    y_dummy = tf.random.uniform((10,), 0, num_classes, dtype=tf.int32)
    model.compile(optimizer='sgd', loss='sparse_categorical_crossentropy')
    model.fit(x_dummy, y_dummy, epochs=1, verbose=0)

    # Save
    Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
    checkpoint = tf.train.Checkpoint(model=model)
    checkpoint.save(checkpoint_path)
    print(f"Test TF checkpoint saved")


if __name__ == "__main__":
    # Test with downloaded hyper_ensemble checkpoint
    convert_tf_checkpoint_to_jax(
        tf_checkpoint_path="/home/equint/github/pnc/checkpoints/hyper_ensemble/model_1/checkpoint-1",
        jax_checkpoint_path="/home/equint/github/pnc/checkpoints/hyper_ensemble/model_1_jax.pkl",
        depth=28,
        width_multiplier=10,
        num_classes=10
    )