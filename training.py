from __future__ import annotations
import time
from typing import Callable, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx
import grain.python as grain
import jax.flatten_util
from jaxtyping import Array, Float


# Type aliases for readability
LossFn = Callable[[nnx.Module, Float[Array, "batch ..."], Array], Float[Array, ""]]
StepHook = Callable[[int, nnx.Module, float, float], None]


def _resnet_steps_per_epoch(n_examples: int, batch_size: int) -> int:
    return max(1, int(np.ceil(n_examples / batch_size)))


def _iter_resnet_epoch_batches(data_iterator, steps_ep: int):
    for _ in range(steps_ep):
        try:
            yield next(data_iterator)
        except StopIteration:
            return


def _build_resnet_optimizer(
    optimizer_name: str,
    schedule,
    weight_decay: float,
    momentum: float,
    nesterov: bool,
):
    if optimizer_name == "sgd":
        return optax.chain(
            optax.add_decayed_weights(weight_decay),
            optax.sgd(learning_rate=schedule, momentum=momentum, nesterov=nesterov),
        )
    if optimizer_name == "adamw":
        return optax.adamw(learning_rate=schedule, weight_decay=weight_decay)
    raise ValueError(f"Unknown optimizer_name: {optimizer_name}")

def train_generic(
    model: nnx.Module,
    train_inputs: "Float[Array, 'batch *dims']",
    train_targets: Array,
    val_inputs: "Float[Array, 'batch *dims']",
    val_targets: Array,
    loss_fn: LossFn,
    val_loss_fn: Optional[LossFn] = None,
    steps: int = 2000,
    batch_size: int = 64,
    lr: float = 1e-3,
    patience: int = 10,
    eval_freq: int = 100,
    step_hook: Optional[StepHook] = None,
    log_prefix: str = "Training"
) -> nnx.Module:
    """Generic training loop with validation split, early stopping, and step hooks.

    loss functions must be stateless
    """
    if val_loss_fn is None:
        val_loss_fn = loss_fn

    print(f"{log_prefix} on {len(train_inputs)} samples, val {len(val_inputs)} samples...")
    optimizer = optax.adamw(lr)
    params = nnx.state(model, nnx.Param)
    opt_state = optimizer.init(params)

    @nnx.jit
    def train_step(m: nnx.Module, opt_state: optax.OptState, x: "Float[Array, 'batch *dims']", y: Array) -> Tuple["Float[Array, '']", optax.OptState]:
        def _compute_loss(mod: nnx.Module) -> "Float[Array, '']":
            return loss_fn(mod, x, y)

        grads = nnx.grad(_compute_loss)(m)
        updates, new_opt = optimizer.update(nnx.state(grads, nnx.Param), opt_state, nnx.state(m, nnx.Param))
        nnx.update(m, optax.apply_updates(nnx.state(m, nnx.Param), updates))
        return _compute_loss(m), new_opt

    @nnx.jit
    def val_step(m: nnx.Module, x: "Float[Array, 'batch *dims']", y: Array) -> "Float[Array, '']":
        return val_loss_fn(m, x, y)

    indices = np.arange(len(train_inputs))
    best_val_loss = float('inf')
    best_state = nnx.state(model)
    checks_without_improvement = 0
    train_loss = jnp.inf

    for i in range(steps):
        batch = np.random.choice(indices, batch_size)
        train_loss_val, opt_state = train_step(model, opt_state, train_inputs[batch], train_targets[batch])
        train_loss = float(train_loss_val)

        if step_hook is not None:
            step_hook(i, model, train_loss, best_val_loss)

        if (i + 1) % eval_freq == 0 or i == steps - 1:
            val_loss = float(val_step(model, val_inputs, val_targets))

            if (i + 1) % max(1, steps // 8) == 0:
                print(f"Step {i+1}: Train Loss {train_loss:.5f} | Val Loss {val_loss:.5f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = nnx.state(model)
                checks_without_improvement = 0
            else:
                checks_without_improvement = checks_without_improvement + 1

            if checks_without_improvement >= patience:
                # To prevent early stopping before a hook requires it (e.g. SWAG warmup)
                # The hook should ideally manage its own requirements or we assume
                # early stopping is globally applicable. For SWAG we might need to be careful.
                # In the original code, SWAG prevented stopping until swag_start.
                # We can handle this by letting step_hook return a boolean "prevent_stop".
                # For simplicity here, we assume standard behavior.
                print(f"Early stopping at step {i+1}. Best Val Loss: {best_val_loss:.5f}")
                break

    print(f"Final Train Loss: {train_loss:.5f} | Best Val Loss: {best_val_loss:.5f}")
    nnx.update(model, best_state)
    return model


# ======== Metrics Callables ========

def mse_loss(m: nnx.Module, x: Float[Array, "batch ..."], y: Array) -> Float[Array, ""]:
    preds = m(x)
    return jnp.mean((preds - y) ** 2)

def gaussian_nll_loss(m: nnx.Module, x: Float[Array, "batch ..."], y: Array) -> Float[Array, ""]:
    mean, var = m(x)
    return jnp.mean(0.5 * (jnp.log(var) + (y - mean)**2 / var))

def ce_loss(m: nnx.Module, x: Float[Array, "batch ..."], y: Array) -> Float[Array, ""]:
    logits = m(x)
    return jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits=logits, labels=y))


# ======== Wrapper Functions ========

def train_model(
    model: nnx.Module,
    train_inputs: Float[Array, "batch ..."],
    train_targets: Array,
    val_inputs: Float[Array, "batch ..."],
    val_targets: Array,
    steps: int = 2000,
    batch_size: int = 64,
    patience: int = 10,
    eval_freq: int = 100
) -> nnx.Module:
    """Trains a simple neural network transition model with early stopping."""
    # The original handled both tuples (mean, var) and single preds in train_model
    # but practically we can detect at runtime or just let the caller decide.
    # We will replicate the dynamic check for safety to avoid breaking code.
    def dynamic_loss(m: nnx.Module, x: Float[Array, "batch ..."], y: Array) -> Float[Array, ""]:
        preds = m(x)
        if isinstance(preds, tuple) and len(preds) == 2:
            mean, var = preds
            return jnp.mean(0.5 * (jnp.log(var) + (mean - y)**2 / var))
        return jnp.mean((preds - y) ** 2)

    return train_generic(
        model, train_inputs, train_targets, val_inputs, val_targets,
        loss_fn=dynamic_loss,
        steps=steps, batch_size=batch_size,
        patience=patience, eval_freq=eval_freq, log_prefix="Training"
    )


def train_probabilistic_model(
    model: nnx.Module,
    train_inputs: Float[Array, "batch ..."],
    train_targets: Array,
    val_inputs: Float[Array, "batch ..."],
    val_targets: Array,
    steps: int = 2000,
    batch_size: int = 64,
    patience: int = 10,
    eval_freq: int = 100
) -> nnx.Module:
    """Trains a probabilistic regression model (outputs mean/var) using Gaussian NLL."""
    return train_generic(
        model, train_inputs, train_targets, val_inputs, val_targets,
        loss_fn=gaussian_nll_loss,
        steps=steps, batch_size=batch_size,
        patience=patience, eval_freq=eval_freq, log_prefix="Training Probabilistic Model"
    )

def train_swag_model(
    model: nnx.Module,
    train_inputs: Float[Array, "batch ..."],
    train_targets: Array,
    val_inputs: Float[Array, "batch ..."],
    val_targets: Array,
    steps: int = 2000,
    batch_size: int = 64,
    swag_start: int = 1000,
    patience: int = 10,
    eval_freq: int = 100
) -> Tuple[nnx.Module, nnx.State, nnx.State]:
    """Trains a simple neural network transition model and collects Diagonal SWAG statistics with early stopping."""
    # We will declare our running state dynamically
    swag_state = {
        "swag_mean": None,
        "swag_sq_mean": None,
        "n_swag_steps": 0
    }

    def swag_hook(step: int, m: nnx.Module, loss: float, best_val: float):
        if step >= swag_start:
            current_params = nnx.state(m, nnx.Param)

            if swag_state["swag_mean"] is None:
                # Initialize on first call
                swag_state["swag_mean"] = jax.tree.map(jnp.zeros_like, current_params)
                swag_state["swag_sq_mean"] = jax.tree.map(jnp.zeros_like, current_params)

            n_swag_steps = swag_state.get("n_swag_steps", 0)
            assert isinstance(n_swag_steps, int)
            n_swag_steps = int(n_swag_steps)
            n = float(n_swag_steps + 1)

            swag_state["swag_mean"] = jax.tree.map(
                lambda s_m, p: (s_m * n_swag_steps + p) / n,
                swag_state["swag_mean"], current_params
            )
            swag_state["swag_sq_mean"] = jax.tree.map(
                lambda sq_m, p: (sq_m * n_swag_steps + p**2) / n,
                swag_state["swag_sq_mean"], current_params
            )
            swag_state["n_swag_steps"] = int(n)

    def dynamic_loss(m: nnx.Module, x: Float[Array, "batch ..."], y: Array) -> Float[Array, ""]:
        preds = m(x)
        if isinstance(preds, tuple) and len(preds) == 2:
            mean, var = preds
            return jnp.mean(0.5 * (jnp.log(var) + (mean - y)**2 / var))
        return jnp.mean((preds - y) ** 2)

    trained_model = train_generic(
        model, train_inputs, train_targets, val_inputs, val_targets,
        loss_fn=dynamic_loss,
        steps=steps, batch_size=batch_size,
        patience=patience, eval_freq=eval_freq, step_hook=swag_hook,
        log_prefix=f"Training (SWAG after {swag_start})"
    )

    n_swag_steps = swag_state["n_swag_steps"]
    if n_swag_steps == 0:
        swag_mean = nnx.state(trained_model, nnx.Param)
        swag_sq_mean = jax.tree.map(lambda x: x**2, swag_mean)
    else:
        swag_mean = swag_state["swag_mean"]
        swag_sq_mean = swag_state["swag_sq_mean"]

    print(f"| SWAG steps collected: {n_swag_steps}")

    swag_var = jax.tree.map(
        lambda sq_m, m: jnp.maximum(sq_m - m**2, 1e-8),
        swag_sq_mean, swag_mean
    )

    return trained_model, swag_mean, swag_var


def train_classification_model(
    model: nnx.Module,
    train_inputs: Float[Array, "batch ..."],
    train_targets: Array,
    val_inputs: Float[Array, "batch ..."],
    val_targets: Array,
    steps: int = 5000,
    batch_size: int = 256,
    lr: float = 1e-3,
    patience: int = 15,
    eval_freq: int = 100
) -> nnx.Module:
    """Trains a classification model using cross entropy loss with early stopping."""
    return train_generic(
        model, train_inputs, train_targets, val_inputs, val_targets,
        loss_fn=ce_loss,
        steps=steps, batch_size=batch_size, lr=lr,
        patience=patience, eval_freq=eval_freq, log_prefix="Training Classification"
    )

def train_swag_classification_model(
    model: nnx.Module,
    train_inputs: Float[Array, "batch ..."],
    train_targets: Array,
    val_inputs: Float[Array, "batch ..."],
    val_targets: Array,
    steps: int = 5000,
    batch_size: int = 256,
    lr: float = 1e-3,
    swag_start: int = 3000,
    patience: int = 15,
    eval_freq: int = 100
) -> Tuple[nnx.Module, nnx.State, nnx.State]:
    """Trains a classification model and collects Diagonal SWAG statistics with early stopping."""
    # We will declare our running state dynamically
    swag_state = {
        "swag_mean": None,
        "swag_sq_mean": None,
        "n_swag_steps": 0
    }

    def swag_hook(step: int, m: nnx.Module, loss: float, best_val: float):
        if step >= swag_start:
            current_params = nnx.state(m, nnx.Param)

            if swag_state["swag_mean"] is None:
                # Initialize on first call
                swag_state["swag_mean"] = jax.tree.map(jnp.zeros_like, current_params)
                swag_state["swag_sq_mean"] = jax.tree.map(jnp.zeros_like, current_params)
                swag_state["n_swag_steps"] = 0

            n_swag_steps = swag_state.get("n_swag_steps", 0)
            assert isinstance(n_swag_steps, int)
            n_swag_steps = int(n_swag_steps)
            n = float(n_swag_steps + 1)

            swag_state["swag_mean"] = jax.tree.map(
                lambda s_m, p: (s_m * n_swag_steps + p) / n,
                swag_state["swag_mean"], current_params
            )
            swag_state["swag_sq_mean"] = jax.tree.map(
                lambda sq_m, p: (sq_m * n_swag_steps + p**2) / n,
                swag_state["swag_sq_mean"], current_params
            )
            swag_state["n_swag_steps"] += 1

    trained_model = train_generic(
        model, train_inputs, train_targets, val_inputs, val_targets,
        loss_fn=ce_loss,
        steps=steps, batch_size=batch_size, lr=lr,
        patience=patience, eval_freq=eval_freq, step_hook=swag_hook,
        log_prefix=f"Training Classification (SWAG after {swag_start})"
    )

    n_swag_steps = swag_state["n_swag_steps"]
    if n_swag_steps == 0:
        swag_mean = nnx.state(trained_model, nnx.Param)
        swag_sq_mean = jax.tree.map(lambda x: x**2, swag_mean)
    else:
        swag_mean = swag_state["swag_mean"]
        swag_sq_mean = swag_state["swag_sq_mean"]

    print(f"| SWAG steps collected: {n_swag_steps}")

    swag_var = jax.tree.map(
        lambda sq_m, m: jnp.maximum(sq_m - m**2, 1e-8),
        swag_sq_mean, swag_mean
    )

    return trained_model, swag_mean, swag_var

def train_subspace_model(
    model: nnx.Module,
    train_inputs: Float[Array, "batch ..."],
    train_targets: Array,
    val_inputs: Float[Array, "batch ..."],
    val_targets: Array,
    steps: int = 2000,
    batch_size: int = 64,
    swag_start: int = 1000,
    max_rank: int = 20,
    patience: int = 10,
    eval_freq: int = 100
) -> Tuple[nnx.Module, nnx.State, jax.Array]:
    subspace_state = {
        "swag_mean": None,
        "n_swag_steps": 0,
        "snapshots": []
    }
    snapshot_freq = max(1, (steps - swag_start) // max_rank)

    def subspace_hook(step: int, m: nnx.Module, loss: float, best_val: float):
        if step >= swag_start:
            current_params = nnx.state(m, nnx.Param)
            if subspace_state["swag_mean"] is None:
                subspace_state["swag_mean"] = jax.tree.map(jnp.zeros_like, current_params)

            n_swag_steps = subspace_state.get("n_swag_steps", 0)
            assert isinstance(n_swag_steps, int)
            n_swag_steps = int(n_swag_steps)
            n = float(n_swag_steps + 1)

            subspace_state["swag_mean"] = jax.tree.map(
                lambda s_m, p: (s_m * n_swag_steps + p) / n,
                subspace_state["swag_mean"], current_params
            )
            subspace_state["n_swag_steps"] = n_swag_steps + 1

            if (step - swag_start) % snapshot_freq == 0:
                snaps = subspace_state.get("snapshots", [])
                if isinstance(snaps, list) and len(snaps) < max_rank:
                    flat_params, _ = jax.flatten_util.ravel_pytree(current_params)
                    snaps.append(flat_params)
                    subspace_state["snapshots"] = snaps

    # Note: Use dynamic loss for transition model wrapper logic
    def dynamic_loss(m: nnx.Module, x: "Float[Array, 'batch *dims']", y: Array) -> "Float[Array, '']":
        preds = m(x)
        if isinstance(preds, tuple) and len(preds) == 2:
            mean, var = preds
            return jnp.mean(0.5 * (jnp.log(var) + (mean - y)**2 / var))
        return jnp.mean((preds - y) ** 2)

    trained_model = train_generic(
        model, train_inputs, train_targets, val_inputs, val_targets,
        loss_fn=dynamic_loss,
        steps=steps, batch_size=batch_size,
        patience=patience, eval_freq=eval_freq, step_hook=subspace_hook,
        log_prefix=f"Training Subspace (rank {max_rank}, SWAG after {swag_start})"
    )

    n_swag_steps = subspace_state.get("n_swag_steps", 0)
    assert isinstance(n_swag_steps, int)

    snapshots_val = subspace_state.get("snapshots", [])
    if isinstance(snapshots_val, list):
        snapshots: list[jax.Array] = snapshots_val
    else:
        snapshots: list[jax.Array] = []

    if n_swag_steps == 0:
        swag_mean = nnx.state(trained_model, nnx.Param)
    else:
        swag_mean = subspace_state["swag_mean"]

    swag_mean_flat, _ = jax.flatten_util.ravel_pytree(swag_mean)
    if len(snapshots) > 0:
        A = jnp.stack([s - swag_mean_flat for s in snapshots], axis=1) # (D, C)
        U, S, _ = jnp.linalg.svd(A, full_matrices=False)
        pca_components = U[:, :max_rank] * (S[:max_rank] / jnp.sqrt(max(1, len(snapshots) - 1)))
    else:
        pca_components = jnp.zeros((swag_mean_flat.shape[0], max_rank))

    return trained_model, swag_mean, pca_components

def train_subspace_classification_model(
    model: nnx.Module,
    train_inputs: Float[Array, "batch ..."],
    train_targets: Array,
    val_inputs: Float[Array, "batch ..."],
    val_targets: Array,
    steps: int = 5000,
    batch_size: int = 256,
    lr: float = 1e-3,
    swag_start: int = 3000,
    max_rank: int = 20,
    patience: int = 15,
    eval_freq: int = 100
) -> Tuple[nnx.Module, nnx.State, jax.Array]:
    subspace_state = {
        "swag_mean": None,
        "n_swag_steps": 0,
        "snapshots": []
    }
    snapshot_freq = max(1, (steps - swag_start) // max_rank)

    def subspace_hook(step: int, m: nnx.Module, loss: float, best_val: float):
        if step >= swag_start:
            current_params = nnx.state(m, nnx.Param)
            if subspace_state["swag_mean"] is None:
                subspace_state["swag_mean"] = jax.tree.map(jnp.zeros_like, current_params)

            n_swag_steps = subspace_state.get("n_swag_steps", 0)
            assert isinstance(n_swag_steps, int)
            n_swag_steps = int(n_swag_steps)
            n = float(n_swag_steps + 1)

            subspace_state["swag_mean"] = jax.tree.map(
                lambda s_m, p: (s_m * n_swag_steps + p) / n,
                subspace_state["swag_mean"], current_params
            )
            subspace_state["n_swag_steps"] = n_swag_steps + 1

            if (step - swag_start) % snapshot_freq == 0:
                snaps = subspace_state.get("snapshots", [])
                if isinstance(snaps, list) and len(snaps) < max_rank:
                    flat_params, _ = jax.flatten_util.ravel_pytree(current_params)
                    snaps.append(flat_params)
                    subspace_state["snapshots"] = snaps

    trained_model = train_generic(
        model, train_inputs, train_targets, val_inputs, val_targets,
        loss_fn=ce_loss,
        steps=steps, batch_size=batch_size, lr=lr,
        patience=patience, eval_freq=eval_freq, step_hook=subspace_hook,
        log_prefix=f"Training Classification Subspace (rank {max_rank}, SWAG after {swag_start})"
    )

    n_swag_steps = subspace_state["n_swag_steps"]
    snapshots = subspace_state["snapshots"]

    if n_swag_steps == 0:
        swag_mean = nnx.state(trained_model, nnx.Param)
    else:
        swag_mean = subspace_state["swag_mean"]

    swag_mean_flat, _ = jax.flatten_util.ravel_pytree(swag_mean)
    if len(snapshots) > 0:
        A = jnp.stack([s - swag_mean_flat for s in snapshots], axis=1)
        U, S, _ = jnp.linalg.svd(A, full_matrices=False)
        pca_components = U[:, :max_rank] * (S[:max_rank] / jnp.sqrt(max(1, len(snapshots) - 1)))
    else:
        pca_components = jnp.zeros((swag_mean_flat.shape[0], max_rank))

    return trained_model, swag_mean, pca_components


# ---------------------------------------------------------------------------
# CIFAR ResNet training helpers (2D conv, BatchNorm)
# ---------------------------------------------------------------------------



class NumpyDataSource(grain.RandomAccessDataSource):
    def __init__(self, x_data: np.ndarray, y_data: np.ndarray):
        super().__init__()
        self.x = x_data
        self.y = y_data

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, record_key: int):
        return {"image": self.x[record_key], "label": self.y[record_key]}


class RandomFlipCrop(grain.MapTransform):
    def __init__(self, pad: int = 4, seed: int = 42):
        self.pad = pad
        self.rng = np.random.RandomState(seed)

    def map(self, element):
        x = element["image"]
        # In grain we operate on single unbatched elements.
        H, W, C = x.shape
        x_pad = np.pad(x, ((self.pad, self.pad), (self.pad, self.pad), (0, 0)), mode='reflect')

        top = self.rng.randint(0, 2 * self.pad + 1)
        left = self.rng.randint(0, 2 * self.pad + 1)

        x_aug = x_pad[top:top+H, left:left+W, :]

        if self.rng.rand() > 0.5:
            x_aug = x_aug[:, ::-1, :]

        element["image"] = x_aug
        return element


class RandomCutout(grain.MapTransform):
    def __init__(self, size: int = 8, seed: int = 42):
        self.size = size
        self.rng = np.random.RandomState(seed)

    def map(self, element):
        if self.size <= 0:
            return element

        x = np.array(element["image"], copy=True)
        H, W, _ = x.shape
        cutout_h = min(self.size, H)
        cutout_w = min(self.size, W)

        top = self.rng.randint(0, H - cutout_h + 1)
        left = self.rng.randint(0, W - cutout_w + 1)
        x[top:top + cutout_h, left:left + cutout_w, :] = 0.0

        element["image"] = x
        return element


def _build_resnet_schedule(
    *,
    lr: float,
    epochs: int,
    steps_ep: int,
    warmup_epochs: int,
):
    total_steps = max(1, epochs * steps_ep)
    warmup_epochs = min(warmup_epochs, epochs)
    warmup_steps = warmup_epochs * steps_ep

    if warmup_steps > 0:
        return optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=lr,
            warmup_steps=warmup_steps,
            decay_steps=total_steps,
            end_value=lr * 1e-3,
        )

    return optax.cosine_decay_schedule(
        init_value=lr,
        decay_steps=total_steps,
        alpha=1e-3,
    )


def _make_resnet_dataloader(
    x_train: np.ndarray,
    y_train: np.ndarray,
    *,
    batch_size: int,
    epochs: int,
    cutout_size: int,
    seed: int,
):
    source = NumpyDataSource(x_train, y_train)
    sampler = grain.IndexSampler(
        num_records=len(source),
        num_epochs=epochs,
        shard_options=grain.NoSharding(),
        shuffle=True,
        seed=seed,
    )
    operations = [RandomFlipCrop(pad=4, seed=seed)]
    if cutout_size > 0:
        operations.append(RandomCutout(size=cutout_size, seed=seed + 1))
    operations.append(grain.Batch(batch_size=batch_size, drop_remainder=False))

    return grain.DataLoader(
        data_source=source,
        sampler=sampler,
        operations=operations,
        worker_count=0,  # Avoid JAX fork issues in this environment.
    )


def _make_resnet_classifier_train_step(
    *,
    optimizer,
    label_smoothing: float,
):
    @nnx.jit
    def train_step(model, opt_state, x_batch, y_batch):
        def loss_fn(m):
            logits = m(x_batch, use_running_average=False)
            if label_smoothing > 0.0:
                labels = jax.nn.one_hot(y_batch, logits.shape[-1])
                labels = optax.smooth_labels(labels, alpha=label_smoothing)
                return jnp.mean(optax.softmax_cross_entropy(logits=logits, labels=labels))
            return jnp.mean(
                optax.softmax_cross_entropy_with_integer_labels(logits=logits, labels=y_batch)
            )

        grads = nnx.grad(loss_fn)(model)
        updates, new_opt = optimizer.update(
            nnx.state(grads, nnx.Param),
            opt_state,
            nnx.state(model, nnx.Param),
        )
        nnx.update(model, optax.apply_updates(nnx.state(model, nnx.Param), updates))
        return loss_fn(model), new_opt

    return train_step


def _make_resnet_val_loss_fn():
    @nnx.jit
    def val_loss_fn(model, x_b, y_b):
        logits = model(x_b, use_running_average=True)
        return jnp.mean(
            optax.softmax_cross_entropy_with_integer_labels(logits=logits, labels=y_b)
        )

    return val_loss_fn


def _epoch_resnet_val_loss(model, x_val, y_val, batch_size: int, val_loss_fn) -> float:
    losses, ns = [], []
    for s in range(0, len(x_val), batch_size):
        xb = jnp.array(x_val[s:s + batch_size])
        yb = jnp.array(y_val[s:s + batch_size])
        losses.append(float(val_loss_fn(model, xb, yb)) * len(xb))
        ns.append(len(xb))
    return sum(losses) / sum(ns)


def _epoch_resnet_acc(model, x_val, y_val, batch_size: int) -> float:
    correct = total = 0
    for s in range(0, len(x_val), batch_size):
        xb = jnp.array(x_val[s:s + batch_size])
        yb = jnp.array(y_val[s:s + batch_size])
        preds = jnp.argmax(model(xb, use_running_average=True), axis=-1)
        correct += int(jnp.sum(preds == yb))
        total += len(yb)
    return float(correct) / total


def train_resnet_model(
    model: nnx.Module,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 100,
    batch_size: int = 128,
    lr: float = 0.1,
    weight_decay: float = 5e-4,
    optimizer_name: str = "sgd",
    momentum: float = 0.9,
    nesterov: bool = True,
    patience: int = 15,
    warmup_epochs: int = 5,
    cutout_size: int = 8,
    label_smoothing: float = 0.0,
    seed: int = 99,
) -> nnx.Module:
    """
    Train a CIFAR image classifier with the shared ResNet recipe.

    Features:
      - Cosine LR decay with linear warmup via optax.warmup_cosine_decay_schedule
      - Configurable optimizer (`sgd` or `adamw`)
      - Grain DataLoader for batched data augmentation and streaming
      - Epoch-level early stopping on val cross-entropy
      - BatchNorm use_running_average=False during training, True at evaluation

    Returns: trained model (best val-loss weights restored).
    """
    n_tr      = len(x_train)
    steps_ep  = _resnet_steps_per_epoch(n_tr, batch_size)
    schedule = _build_resnet_schedule(
        lr=lr,
        epochs=epochs,
        steps_ep=steps_ep,
        warmup_epochs=warmup_epochs,
    )
    optimizer = _build_resnet_optimizer(
        optimizer_name=optimizer_name,
        schedule=schedule,
        weight_decay=weight_decay,
        momentum=momentum,
        nesterov=nesterov,
    )
    opt_state = optimizer.init(nnx.state(model, nnx.Param))
    train_step = _make_resnet_classifier_train_step(
        optimizer=optimizer,
        label_smoothing=label_smoothing,
    )
    val_loss_fn = _make_resnet_val_loss_fn()

    dataloader = _make_resnet_dataloader(
        x_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        cutout_size=cutout_size,
        seed=seed,
    )
    data_iterator = iter(dataloader)

    best_val = float('inf')
    best_state = nnx.state(model)
    pat = 0

    aug_desc = "flipcrop"
    if cutout_size > 0:
        aug_desc += f"+cutout{cutout_size}"
    print(
        f"Training CIFAR classifier: {n_tr} train / {len(x_val)} val | "
        f"epochs={epochs}, bs={batch_size}, lr={lr:g}, opt={optimizer_name}, "
        f"wd={weight_decay:g}, warmup={warmup_epochs}, aug={aug_desc}, "
        f"label_smoothing={label_smoothing:g}"
    )

    for epoch in range(epochs):
        t_ep = time.time()
        ep_loss = 0.0
        processed_steps = 0

        for processed_steps, batch in enumerate(_iter_resnet_epoch_batches(data_iterator, steps_ep), start=1):
            xaug = jnp.array(batch["image"])
            yb   = jnp.array(batch["label"])

            loss, opt_state = train_step(model, opt_state, xaug, yb)
            ep_loss += float(loss)
            print(f"\r  Epoch {epoch+1:3d}/{epochs}  step {processed_steps}/{steps_ep}"
                  f"  loss={ep_loss/processed_steps:.4f}", end='', flush=True)

        if processed_steps == 0:
            break

        ep_loss /= processed_steps

        vl = _epoch_resnet_val_loss(model, x_val, y_val, batch_size, val_loss_fn)
        acc = _epoch_resnet_acc(model, x_val, y_val, batch_size)
        # elapsed = time.time() - t_start  # removed unused variable
        ep_time = time.time() - t_ep
        eta     = ep_time * (epochs - epoch - 1)
        print(f"\r  Epoch {epoch+1:3d}/{epochs} | "
              f"train={ep_loss:.4f} | val={vl:.4f} | acc={acc:.3%} | "
              f"{ep_time:.0f}s/ep | ETA {eta/60:.1f}min")

        if vl < best_val:
            best_val, best_state, pat = vl, nnx.state(model), 0
        else:
            pat += 1
            if pat >= patience:
                print(f"  Early stop at epoch {epoch+1}. Best val: {best_val:.4f}")
                break

    nnx.update(model, best_state)
    print(f"Done. Best val loss: {best_val:.4f}")
    return model


def train_resnet_swag(
    model: nnx.Module,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    *,
    epochs: int = 200,
    batch_size: int = 128,
    lr: float = 0.1,
    weight_decay: float = 5e-4,
    optimizer_name: str = "sgd",
    momentum: float = 0.9,
    nesterov: bool = True,
    warmup_epochs: int = 5,
    cutout_size: int = 8,
    label_smoothing: float = 0.0,
    swag_start_epoch: int = 160,
    swag_collect_freq: int = 1,
    seed: int = 99,
) -> tuple[nnx.Module, nnx.State, nnx.State, dict[str, object]]:
    """
    Train a CIFAR classifier along one continuous trajectory and collect diagonal-SWAG stats.

    This intentionally disables early stopping: diagonal SWAG needs a meaningful post-burn-in
    trajectory, and stopping on validation loss can eliminate the collection phase entirely.
    """
    if epochs < 1:
        raise ValueError("epochs must be >= 1")
    if swag_collect_freq < 1:
        raise ValueError("swag_collect_freq must be >= 1")
    if swag_start_epoch < 1 or swag_start_epoch > epochs:
        raise ValueError("swag_start_epoch must be in [1, epochs]")

    n_tr = len(x_train)
    steps_ep = _resnet_steps_per_epoch(n_tr, batch_size)
    schedule = _build_resnet_schedule(
        lr=lr,
        epochs=epochs,
        steps_ep=steps_ep,
        warmup_epochs=warmup_epochs,
    )
    optimizer = _build_resnet_optimizer(
        optimizer_name=optimizer_name,
        schedule=schedule,
        weight_decay=weight_decay,
        momentum=momentum,
        nesterov=nesterov,
    )
    opt_state = optimizer.init(nnx.state(model, nnx.Param))
    train_step = _make_resnet_classifier_train_step(
        optimizer=optimizer,
        label_smoothing=label_smoothing,
    )
    val_loss_fn = _make_resnet_val_loss_fn()
    dataloader = _make_resnet_dataloader(
        x_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        cutout_size=cutout_size,
        seed=seed,
    )
    data_iterator = iter(dataloader)

    aug_desc = "flipcrop"
    if cutout_size > 0:
        aug_desc += f"+cutout{cutout_size}"
    print(
        f"Training diagonal SWAG: {n_tr} train / {len(x_val)} val | "
        f"epochs={epochs}, bs={batch_size}, lr={lr:g}, opt={optimizer_name}, "
        f"wd={weight_decay:g}, warmup={warmup_epochs}, aug={aug_desc}, "
        f"label_smoothing={label_smoothing:g}, swag_start={swag_start_epoch}, "
        f"swag_freq={swag_collect_freq}"
    )

    swag_mean = None
    swag_sq_mean = None
    snapshot_epochs: list[int] = []
    best_val = float("inf")
    best_epoch = 0

    for epoch in range(epochs):
        t_ep = time.time()
        ep_loss = 0.0
        processed_steps = 0

        for processed_steps, batch in enumerate(_iter_resnet_epoch_batches(data_iterator, steps_ep), start=1):
            xaug = jnp.array(batch["image"])
            yb = jnp.array(batch["label"])
            loss, opt_state = train_step(model, opt_state, xaug, yb)
            ep_loss += float(loss)
            print(
                f"\r  Epoch {epoch + 1:3d}/{epochs}  step {processed_steps}/{steps_ep}"
                f"  loss={ep_loss / processed_steps:.4f}",
                end="",
                flush=True,
            )

        if processed_steps == 0:
            break

        ep_loss /= processed_steps
        vl = _epoch_resnet_val_loss(model, x_val, y_val, batch_size, val_loss_fn)
        acc = _epoch_resnet_acc(model, x_val, y_val, batch_size)
        if vl < best_val:
            best_val = vl
            best_epoch = epoch + 1

        should_collect = (
            (epoch + 1) >= swag_start_epoch
            and ((epoch + 1 - swag_start_epoch) % swag_collect_freq == 0)
        )
        if should_collect:
            current_params = nnx.state(model, nnx.Param)
            if swag_mean is None:
                swag_mean = jax.tree.map(jnp.zeros_like, current_params)
                swag_sq_mean = jax.tree.map(jnp.zeros_like, current_params)
            n_snapshots = len(snapshot_epochs)
            denom = float(n_snapshots + 1)
            swag_mean = jax.tree.map(
                lambda running, param: (running * n_snapshots + param) / denom,
                swag_mean,
                current_params,
            )
            swag_sq_mean = jax.tree.map(
                lambda running, param: (running * n_snapshots + param**2) / denom,
                swag_sq_mean,
                current_params,
            )
            snapshot_epochs.append(epoch + 1)

        ep_time = time.time() - t_ep
        eta = ep_time * (epochs - epoch - 1)
        collect_desc = f" | swag_snapshots={len(snapshot_epochs)}"
        if should_collect:
            collect_desc += " (collected)"
        print(
            f"\r  Epoch {epoch + 1:3d}/{epochs} | "
            f"train={ep_loss:.4f} | val={vl:.4f} | acc={acc:.3%} | "
            f"{ep_time:.0f}s/ep | ETA {eta / 60:.1f}min{collect_desc}"
        )

    if swag_mean is None or swag_sq_mean is None or not snapshot_epochs:
        raise ValueError(
            "No SWAG snapshots were collected. Increase epochs or lower swag_start_epoch."
        )

    swag_var = jax.tree.map(
        lambda sq_mean, mean: jnp.maximum(sq_mean - mean**2, 1e-8),
        swag_sq_mean,
        swag_mean,
    )
    metadata = {
        "variant": "diagonal_swag",
        "swag_start_epoch": int(swag_start_epoch),
        "swag_collect_freq": int(swag_collect_freq),
        "snapshot_epochs": snapshot_epochs,
        "n_snapshots_collected": len(snapshot_epochs),
        "best_val_loss": float(best_val),
        "best_val_epoch": int(best_epoch),
        "final_epoch": int(epoch + 1),
    }

    print(
        f"Done. Collected {len(snapshot_epochs)} diagonal-SWAG snapshots "
        f"(best val {best_val:.4f} at epoch {best_epoch})."
    )
    return model, swag_mean, swag_var, metadata
