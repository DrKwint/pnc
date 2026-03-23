import time
import jax
import jax.numpy as jnp
import optax
import numpy as np
from flax import nnx


def train_model(
    model: nnx.Module, 
    inputs: jax.Array, 
    targets: jax.Array, 
    steps: int = 2000, 
    batch_size: int = 64,
    val_split: float = 0.1,
    patience: int = 10,
    eval_freq: int = 100
) -> nnx.Module:
    """Trains a simple neural network transition model with early stopping."""
    n_val = max(1, int(len(inputs) * val_split)) if val_split > 0 else 0
    if n_val > 0:
        idx = np.random.permutation(len(inputs))
        train_idx, val_idx = idx[n_val:], idx[:n_val]
        train_inputs, train_targets = inputs[train_idx], targets[train_idx]
        val_inputs, val_targets = inputs[val_idx], targets[val_idx]
    else:
        train_inputs, train_targets = inputs, targets
        val_inputs, val_targets = inputs, targets 
        
    print(f"Training on {len(train_inputs)} samples, val {len(val_inputs)} samples...")
    optimizer = optax.adamw(1e-3)
    params = nnx.state(model, nnx.Param)
    opt_state = optimizer.init(params)
    
    @nnx.jit
    def train_step(model, opt_state, x, y):
        def loss_fn(m):
            preds = m(x)
            if isinstance(preds, tuple) and len(preds) == 2:
                mean, var = preds
                return jnp.mean(0.5 * (jnp.log(var) + (mean - y)**2 / var))
            return jnp.mean((preds - y) ** 2)
        grads = nnx.grad(loss_fn)(model)
        updates, new_opt = optimizer.update(nnx.state(grads, nnx.Param), opt_state, nnx.state(model, nnx.Param))
        nnx.update(model, optax.apply_updates(nnx.state(model, nnx.Param), updates))
        return loss_fn(model), new_opt

    @nnx.jit
    def val_loss_fn(model, x, y):
        preds = model(x)
        if isinstance(preds, tuple) and len(preds) == 2:
            mean, var = preds
            return jnp.mean(0.5 * (jnp.log(var) + (mean - y)**2 / var))
        return jnp.mean((preds - y) ** 2)

    indices = np.arange(len(train_inputs))
    loss = jnp.inf
    
    best_val_loss = jnp.inf
    best_state = nnx.state(model)
    checks_without_improvement = 0
    
    for i in range(steps):
        batch = np.random.choice(indices, batch_size)
        loss, opt_state = train_step(model, opt_state, train_inputs[batch], train_targets[batch])
        
        if (i + 1) % eval_freq == 0 or i == steps - 1:
            if n_val > 0:
                v_loss = val_loss_fn(model, val_inputs, val_targets)
            else:
                v_loss = loss
                
            if (i+1) % 500 == 0: 
                print(f"Step {i+1}: Train Loss {loss:.5f} | Val Loss {v_loss:.5f}")
                
            if float(v_loss) < best_val_loss:
                best_val_loss = float(v_loss)
                best_state = nnx.state(model)
                checks_without_improvement = 0
            else:
                checks_without_improvement += 1
                
            if checks_without_improvement >= patience:
                print(f"Early stopping at step {i+1}. Best Val Loss: {best_val_loss:.5f}")
                break
                
    print(f"Final Train Loss: {loss:.5f} | Best Val Loss: {best_val_loss:.5f}")
    nnx.update(model, best_state)
    return model


def train_probabilistic_model(
    model: nnx.Module, 
    inputs: jax.Array, 
    targets: jax.Array, 
    steps: int = 2000, 
    batch_size: int = 64,
    val_split: float = 0.1,
    patience: int = 10,
    eval_freq: int = 100
) -> nnx.Module:
    """Trains a probabilistic regression model (outputs mean/var) using Gaussian NLL."""
    n_val = max(1, int(len(inputs) * val_split)) if val_split > 0 else 0
    if n_val > 0:
        idx = np.random.permutation(len(inputs))
        train_idx, val_idx = idx[n_val:], idx[:n_val]
        train_inputs, train_targets = inputs[train_idx], targets[train_idx]
        val_inputs, val_targets = inputs[val_idx], targets[val_idx]
    else:
        train_inputs, train_targets = inputs, targets
        val_inputs, val_targets = inputs, targets 
        
    print(f"Training Probabilistic Model on {len(train_inputs)} samples, val {len(val_inputs)} samples...")
    optimizer = optax.adamw(1e-3)
    params = nnx.state(model, nnx.Param)
    opt_state = optimizer.init(params)
    
    @nnx.jit
    def train_step(model, opt_state, x, y):
        def loss_fn(m):
            mean, var = m(x)
            # Gaussian NLL loss: 0.5 * (log(var) + (y - mean)^2 / var)
            # We ignore the constant (log(2pi)/2) for optimization
            return jnp.mean(0.5 * (jnp.log(var) + (y - mean)**2 / var))
            
        grads = nnx.grad(loss_fn)(model)
        updates, new_opt = optimizer.update(nnx.state(grads, nnx.Param), opt_state, nnx.state(model, nnx.Param))
        nnx.update(model, optax.apply_updates(nnx.state(model, nnx.Param), updates))
        return loss_fn(model), new_opt

    @nnx.jit
    def val_loss_fn(model, x, y):
        mean, var = model(x)
        return jnp.mean(0.5 * (jnp.log(var) + (y - mean)**2 / var))

    indices = np.arange(len(train_inputs))
    loss = jnp.inf
    
    best_val_loss = jnp.inf
    best_state = nnx.state(model)
    checks_without_improvement = 0
    
    for i in range(steps):
        batch = np.random.choice(indices, batch_size)
        loss, opt_state = train_step(model, opt_state, train_inputs[batch], train_targets[batch])
        
        if (i + 1) % eval_freq == 0 or i == steps - 1:
            if n_val > 0:
                v_loss = val_loss_fn(model, val_inputs, val_targets)
            else:
                v_loss = loss
                
            if (i+1) % 500 == 0: 
                print(f"Step {i+1}: Train NLL {loss:.5f} | Val NLL {v_loss:.5f}")
                
            if float(v_loss) < best_val_loss:
                best_val_loss = float(v_loss)
                best_state = nnx.state(model)
                checks_without_improvement = 0
            else:
                checks_without_improvement += 1
                
            if checks_without_improvement >= patience:
                print(f"Early stopping at step {i+1}. Best Val NLL: {best_val_loss:.5f}")
                break
                
    print(f"Final Train NLL: {loss:.5f} | Best Val NLL: {best_val_loss:.5f}")
    nnx.update(model, best_state)
    return model

def train_swag_model(
    model: nnx.Module, 
    inputs: jax.Array, 
    targets: jax.Array, 
    steps: int = 2000, 
    batch_size: int = 64,
    swag_start: int = 1000,
    val_split: float = 0.1,
    patience: int = 10,
    eval_freq: int = 100
):
    """Trains a simple neural network transition model and collects Diagonal SWAG statistics with early stopping."""
    n_val = max(1, int(len(inputs) * val_split)) if val_split > 0 else 0
    if n_val > 0:
        idx = np.random.permutation(len(inputs))
        train_idx, val_idx = idx[n_val:], idx[:n_val]
        train_inputs, train_targets = inputs[train_idx], targets[train_idx]
        val_inputs, val_targets = inputs[val_idx], targets[val_idx]
    else:
        train_inputs, train_targets = inputs, targets
        val_inputs, val_targets = inputs, targets 
        
    print(f"Training on {len(train_inputs)} samples (SWAG enabled after {swag_start} steps, val {len(val_inputs)})...")
    optimizer = optax.adamw(1e-3)
    params = nnx.state(model, nnx.Param)
    opt_state = optimizer.init(params)
    
    @nnx.jit
    def train_step(model, opt_state, x, y):
        def loss_fn(m):
            preds = m(x)
            if isinstance(preds, tuple) and len(preds) == 2:
                mean, var = preds
                return jnp.mean(0.5 * (jnp.log(var) + (mean - y)**2 / var))
            return jnp.mean((preds - y) ** 2)
        grads = nnx.grad(loss_fn)(model)
        updates, new_opt = optimizer.update(nnx.state(grads, nnx.Param), opt_state, nnx.state(model, nnx.Param))
        nnx.update(model, optax.apply_updates(nnx.state(model, nnx.Param), updates))
        return loss_fn(model), new_opt

    @nnx.jit
    def val_loss_fn(model, x, y):
        preds = model(x)
        if isinstance(preds, tuple) and len(preds) == 2:
            mean, var = preds
            return jnp.mean(0.5 * (jnp.log(var) + (mean - y)**2 / var))
        return jnp.mean((preds - y) ** 2)

    indices = np.arange(len(train_inputs))
    loss = jnp.inf
    
    swag_mean = jax.tree.map(lambda x: jnp.zeros_like(x), params)
    swag_sq_mean = jax.tree.map(lambda x: jnp.zeros_like(x), params)
    n_swag_steps = 0
    
    best_val_loss = jnp.inf
    best_state = nnx.state(model)
    checks_without_improvement = 0
    
    best_swag_mean = None
    best_swag_sq_mean = None
    
    for i in range(steps):
        batch = np.random.choice(indices, batch_size)
        loss, opt_state = train_step(model, opt_state, train_inputs[batch], train_targets[batch])
        
        if i >= swag_start:
            current_params = nnx.state(model, nnx.Param)
            n = float(n_swag_steps + 1)
            swag_mean = jax.tree.map(
                lambda m, p: (m * n_swag_steps + p) / n, 
                swag_mean, current_params
            )
            swag_sq_mean = jax.tree.map(
                lambda sq_m, p: (sq_m * n_swag_steps + p**2) / n, 
                swag_sq_mean, current_params
            )
            n_swag_steps += 1

        if (i + 1) % eval_freq == 0 or i == steps - 1:
            if n_val > 0:
                v_loss = val_loss_fn(model, val_inputs, val_targets)
            else:
                v_loss = loss
                
            if (i+1) % 500 == 0: 
                print(f"Step {i+1}: Train Loss {loss:.5f} | Val Loss {v_loss:.5f}")
                
            if float(v_loss) < best_val_loss:
                best_val_loss = float(v_loss)
                best_state = nnx.state(model)
                checks_without_improvement = 0
            else:
                checks_without_improvement += 1
                
            if checks_without_improvement >= patience and i < swag_start:
                print(f"Early stopping at step {i+1}. Best Val Loss: {best_val_loss:.5f}")
                break
                
    nnx.update(model, best_state)
    
    if n_swag_steps == 0:
        swag_mean = nnx.state(model, nnx.Param)
        swag_sq_mean = jax.tree.map(lambda x: x**2, swag_mean)
        
    print(f"Final Train Loss: {loss:.5f} | Best Val Loss: {best_val_loss:.5f} | SWAG steps collected: {n_swag_steps}")
    
    swag_var = jax.tree.map(
        lambda sq_m, m: jnp.maximum(sq_m - m**2, 1e-8), 
        swag_sq_mean, swag_mean
    )
    
    return model, swag_mean, swag_var


def train_classification_model(
    model: nnx.Module, 
    inputs: jax.Array, 
    targets: jax.Array, 
    steps: int = 5000, 
    batch_size: int = 256,
    lr: float = 1e-3,
    val_split: float = 0.1,
    patience: int = 15,
    eval_freq: int = 100
) -> nnx.Module:
    """Trains a classification model using cross entropy loss with early stopping."""
    n_val = max(1, int(len(inputs) * val_split)) if val_split > 0 else 0
    if n_val > 0:
        idx = np.random.permutation(len(inputs))
        train_idx, val_idx = idx[n_val:], idx[:n_val]
        train_inputs, train_targets = inputs[train_idx], targets[train_idx]
        val_inputs, val_targets = inputs[val_idx], targets[val_idx]
    else:
        train_inputs, train_targets = inputs, targets
        val_inputs, val_targets = inputs, targets 
        
    print(f"Training Classification on {len(train_inputs)} samples, val {len(val_inputs)} samples...")
    optimizer = optax.adamw(lr)
    params = nnx.state(model, nnx.Param)
    opt_state = optimizer.init(params)
    
    @nnx.jit
    def train_step(model, opt_state, x, y):
        def loss_fn(m):
            logits = m(x)
            return jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits=logits, labels=y))
        grads = nnx.grad(loss_fn)(model)
        updates, new_opt = optimizer.update(nnx.state(grads, nnx.Param), opt_state, nnx.state(model, nnx.Param))
        nnx.update(model, optax.apply_updates(nnx.state(model, nnx.Param), updates))
        return loss_fn(model), new_opt

    @nnx.jit
    def val_loss_fn(model, x, y):
        logits = model(x)
        return jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits=logits, labels=y))

    indices = np.arange(len(train_inputs))
    loss = jnp.inf
    
    best_val_loss = jnp.inf
    best_state = nnx.state(model)
    checks_without_improvement = 0
    
    for i in range(steps):
        batch = np.random.choice(indices, batch_size)
        loss, opt_state = train_step(model, opt_state, train_inputs[batch], train_targets[batch])
        
        if (i + 1) % eval_freq == 0 or i == steps - 1:
            if n_val > 0:
                v_loss = val_loss_fn(model, val_inputs, val_targets)
            else:
                v_loss = loss
            
            if (i+1) % 1000 == 0: 
                print(f"Step {i+1}: Train Loss {loss:.5f} | Val Loss {v_loss:.5f}")
                
            if float(v_loss) < best_val_loss:
                best_val_loss = float(v_loss)
                best_state = nnx.state(model)
                checks_without_improvement = 0
            else:
                checks_without_improvement += 1
                
            if checks_without_improvement >= patience:
                print(f"Early stopping at step {i+1}. Best Val Loss: {best_val_loss:.5f}")
                break
                
    print(f"Final Train Loss: {loss:.5f} | Best Val Loss: {best_val_loss:.5f}")
    nnx.update(model, best_state)
    return model

def train_swag_classification_model(
    model: nnx.Module, 
    inputs: jax.Array, 
    targets: jax.Array, 
    steps: int = 5000, 
    batch_size: int = 256,
    lr: float = 1e-3,
    swag_start: int = 3000,
    val_split: float = 0.1,
    patience: int = 15,
    eval_freq: int = 100
):
    """Trains a classification model and collects Diagonal SWAG statistics with early stopping."""
    n_val = max(1, int(len(inputs) * val_split)) if val_split > 0 else 0
    if n_val > 0:
        idx = np.random.permutation(len(inputs))
        train_idx, val_idx = idx[n_val:], idx[:n_val]
        train_inputs, train_targets = inputs[train_idx], targets[train_idx]
        val_inputs, val_targets = inputs[val_idx], targets[val_idx]
    else:
        train_inputs, train_targets = inputs, targets
        val_inputs, val_targets = inputs, targets 
        
    print(f"Training Classification on {len(train_inputs)} samples (SWAG enabled after {swag_start} steps, val {len(val_inputs)})...")
    optimizer = optax.adamw(lr)
    params = nnx.state(model, nnx.Param)
    opt_state = optimizer.init(params)
    
    @nnx.jit
    def train_step(model, opt_state, x, y):
        def loss_fn(m):
            logits = m(x)
            return jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits=logits, labels=y))
        grads = nnx.grad(loss_fn)(model)
        updates, new_opt = optimizer.update(nnx.state(grads, nnx.Param), opt_state, nnx.state(model, nnx.Param))
        nnx.update(model, optax.apply_updates(nnx.state(model, nnx.Param), updates))
        return loss_fn(model), new_opt

    @nnx.jit
    def val_loss_fn(model, x, y):
        logits = model(x)
        return jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits=logits, labels=y))

    indices = np.arange(len(train_inputs))
    loss = jnp.inf
    
    swag_mean = jax.tree.map(lambda x: jnp.zeros_like(x), params)
    swag_sq_mean = jax.tree.map(lambda x: jnp.zeros_like(x), params)
    n_swag_steps = 0
    
    best_val_loss = jnp.inf
    best_state = nnx.state(model)
    checks_without_improvement = 0
    
    best_swag_mean = None
    best_swag_sq_mean = None
    
    for i in range(steps):
        batch = np.random.choice(indices, batch_size)
        loss, opt_state = train_step(model, opt_state, train_inputs[batch], train_targets[batch])
        
        if i >= swag_start:
            current_params = nnx.state(model, nnx.Param)
            n = float(n_swag_steps + 1)
            swag_mean = jax.tree.map(
                lambda m, p: (m * n_swag_steps + p) / n, 
                swag_mean, current_params
            )
            swag_sq_mean = jax.tree.map(
                lambda sq_m, p: (sq_m * n_swag_steps + p**2) / n, 
                swag_sq_mean, current_params
            )
            n_swag_steps += 1

        if (i + 1) % eval_freq == 0 or i == steps - 1:
            if n_val > 0:
                v_loss = val_loss_fn(model, val_inputs, val_targets)
            else:
                v_loss = loss
            
            if (i+1) % 1000 == 0: 
                print(f"Step {i+1}: Train Loss {loss:.5f} | Val Loss {v_loss:.5f}")
                
            if float(v_loss) < best_val_loss:
                best_val_loss = float(v_loss)
                best_state = nnx.state(model)
                checks_without_improvement = 0
            else:
                checks_without_improvement += 1
                
            if checks_without_improvement >= patience and i < swag_start:
                print(f"Early stopping at step {i+1}. Best Val Loss: {best_val_loss:.5f}")
                break
                
    nnx.update(model, best_state)
    
    if n_swag_steps == 0:
        swag_mean = nnx.state(model, nnx.Param)
        swag_sq_mean = jax.tree.map(lambda x: x**2, swag_mean)
        
    print(f"Final Train Loss: {loss:.5f} | Best Val Loss: {best_val_loss:.5f} | SWAG steps collected: {n_swag_steps}")
    
    swag_var = jax.tree.map(
        lambda sq_m, m: jnp.maximum(sq_m - m**2, 1e-8), 
        swag_sq_mean, swag_mean
    )
    
    return model, swag_mean, swag_var

import jax.flatten_util

def train_subspace_model(
    model: nnx.Module, 
    inputs: jax.Array, 
    targets: jax.Array, 
    steps: int = 2000, 
    batch_size: int = 64,
    swag_start: int = 1000,
    max_rank: int = 20,
    val_split: float = 0.1,
    patience: int = 10,
    eval_freq: int = 100
):
    n_val = max(1, int(len(inputs) * val_split)) if val_split > 0 else 0
    if n_val > 0:
        idx = np.random.permutation(len(inputs))
        train_idx, val_idx = idx[n_val:], idx[:n_val]
        train_inputs, train_targets = inputs[train_idx], targets[train_idx]
        val_inputs, val_targets = inputs[val_idx], targets[val_idx]
    else:
        train_inputs, train_targets = inputs, targets
        val_inputs, val_targets = inputs, targets 
        
    print(f"Training on {len(train_inputs)} samples (Subspace enabled after {swag_start} steps, rank {max_rank})...")
    optimizer = optax.adamw(1e-3)
    params = nnx.state(model, nnx.Param)
    opt_state = optimizer.init(params)
    
    @nnx.jit
    def train_step(model, opt_state, x, y):
        def loss_fn(m):
            preds = m(x)
            if isinstance(preds, tuple) and len(preds) == 2:
                mean, var = preds
                return jnp.mean(0.5 * (jnp.log(var) + (mean - y)**2 / var))
            return jnp.mean((preds - y) ** 2)
        grads = nnx.grad(loss_fn)(model)
        updates, new_opt = optimizer.update(nnx.state(grads, nnx.Param), opt_state, nnx.state(model, nnx.Param))
        nnx.update(model, optax.apply_updates(nnx.state(model, nnx.Param), updates))
        return loss_fn(model), new_opt

    @nnx.jit
    def val_loss_fn(model, x, y):
        preds = model(x)
        if isinstance(preds, tuple) and len(preds) == 2:
            mean, var = preds
            return jnp.mean(0.5 * (jnp.log(var) + (mean - y)**2 / var))
        return jnp.mean((preds - y) ** 2)

    indices = np.arange(len(train_inputs))
    loss = jnp.inf
    
    swag_mean = jax.tree.map(lambda x: jnp.zeros_like(x), params)
    n_swag_steps = 0
    snapshots = []
    snapshot_freq = max(1, (steps - swag_start) // max_rank)
    
    best_val_loss = jnp.inf
    best_state = nnx.state(model)
    checks_without_improvement = 0
    
    for i in range(steps):
        batch = np.random.choice(indices, batch_size)
        loss, opt_state = train_step(model, opt_state, train_inputs[batch], train_targets[batch])
        
        if i >= swag_start:
            current_params = nnx.state(model, nnx.Param)
            n = float(n_swag_steps + 1)
            swag_mean = jax.tree.map(
                lambda m, p: (m * n_swag_steps + p) / n, 
                swag_mean, current_params
            )
            n_swag_steps += 1
            if (i - swag_start) % snapshot_freq == 0 and len(snapshots) < max_rank:
                flat_params, _ = jax.flatten_util.ravel_pytree(current_params)
                snapshots.append(flat_params)

        if (i + 1) % eval_freq == 0 or i == steps - 1:
            v_loss = val_loss_fn(model, val_inputs, val_targets) if n_val > 0 else loss
            if float(v_loss) < best_val_loss:
                best_val_loss = float(v_loss)
                best_state = nnx.state(model)
                checks_without_improvement = 0
            else:
                checks_without_improvement += 1
            if checks_without_improvement >= patience and i < swag_start:
                break
                
    nnx.update(model, best_state)
    
    if n_swag_steps == 0:
        swag_mean = nnx.state(model, nnx.Param)
    
    swag_mean_flat, _ = jax.flatten_util.ravel_pytree(swag_mean)
    if len(snapshots) > 0:
        A = jnp.stack([s - swag_mean_flat for s in snapshots], axis=1) # (D, C)
        U, S, _ = jnp.linalg.svd(A, full_matrices=False)
        pca_components = U[:, :max_rank] * (S[:max_rank] / jnp.sqrt(max(1, len(snapshots) - 1)))
    else:
        pca_components = jnp.zeros((swag_mean_flat.shape[0], max_rank))
    
    return model, swag_mean, pca_components

def train_subspace_classification_model(
    model: nnx.Module, 
    inputs: jax.Array, 
    targets: jax.Array, 
    steps: int = 5000, 
    batch_size: int = 256,
    lr: float = 1e-3,
    swag_start: int = 3000,
    max_rank: int = 20,
    val_split: float = 0.1,
    patience: int = 15,
    eval_freq: int = 100
):
    n_val = max(1, int(len(inputs) * val_split)) if val_split > 0 else 0
    if n_val > 0:
        idx = np.random.permutation(len(inputs))
        train_idx, val_idx = idx[n_val:], idx[:n_val]
        train_inputs, train_targets = inputs[train_idx], targets[train_idx]
        val_inputs, val_targets = inputs[val_idx], targets[val_idx]
    else:
        train_inputs, train_targets = inputs, targets
        val_inputs, val_targets = inputs, targets 
        
    print(f"Training Classification on {len(train_inputs)} samples (Subspace enabled after {swag_start} steps)...")
    optimizer = optax.adamw(lr)
    params = nnx.state(model, nnx.Param)
    opt_state = optimizer.init(params)
    
    @nnx.jit
    def train_step(model, opt_state, x, y):
        def loss_fn(m):
            logits = m(x)
            return jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits=logits, labels=y))
        grads = nnx.grad(loss_fn)(model)
        updates, new_opt = optimizer.update(nnx.state(grads, nnx.Param), opt_state, nnx.state(model, nnx.Param))
        nnx.update(model, optax.apply_updates(nnx.state(model, nnx.Param), updates))
        return loss_fn(model), new_opt

    @nnx.jit
    def val_loss_fn(model, x, y):
        return jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits=model(x), labels=y))

    indices = np.arange(len(train_inputs))
    loss = jnp.inf
    
    swag_mean = jax.tree.map(lambda x: jnp.zeros_like(x), params)
    n_swag_steps = 0
    snapshots = []
    snapshot_freq = max(1, (steps - swag_start) // max_rank)
    
    best_val_loss = jnp.inf
    best_state = nnx.state(model)
    checks_without_improvement = 0
    
    for i in range(steps):
        batch = np.random.choice(indices, batch_size)
        loss, opt_state = train_step(model, opt_state, train_inputs[batch], train_targets[batch])
        
        if i >= swag_start:
            current_params = nnx.state(model, nnx.Param)
            n = float(n_swag_steps + 1)
            swag_mean = jax.tree.map(
                lambda m, p: (m * n_swag_steps + p) / n, 
                swag_mean, current_params
            )
            n_swag_steps += 1
            if (i - swag_start) % snapshot_freq == 0 and len(snapshots) < max_rank:
                flat_params, _ = jax.flatten_util.ravel_pytree(current_params)
                snapshots.append(flat_params)

        if (i + 1) % eval_freq == 0 or i == steps - 1:
            v_loss = val_loss_fn(model, val_inputs, val_targets) if n_val > 0 else loss
            if float(v_loss) < best_val_loss:
                best_val_loss = float(v_loss)
                best_state = nnx.state(model)
                checks_without_improvement = 0
            else:
                checks_without_improvement += 1
            if checks_without_improvement >= patience and i < swag_start:
                break
                
    nnx.update(model, best_state)
    
    if n_swag_steps == 0:
        swag_mean = nnx.state(model, nnx.Param)
        
    swag_mean_flat, _ = jax.flatten_util.ravel_pytree(swag_mean)
    if len(snapshots) > 0:
        A = jnp.stack([s - swag_mean_flat for s in snapshots], axis=1)
        U, S, _ = jnp.linalg.svd(A, full_matrices=False)
        pca_components = U[:, :max_rank] * (S[:max_rank] / jnp.sqrt(max(1, len(snapshots) - 1)))
    else:
        pca_components = jnp.zeros((swag_mean_flat.shape[0], max_rank))
    
    return model, swag_mean, pca_components


# ---------------------------------------------------------------------------
# ResNet-50 training (CIFAR images, 2D conv, BatchNorm)
# ---------------------------------------------------------------------------

def _random_flip_crop(x: np.ndarray, pad: int = 4) -> np.ndarray:
    """Random horizontal flip + crop for a batch of NHWC images."""
    N, H, W, C = x.shape
    x_pad = np.pad(x, ((0,0),(pad,pad),(pad,pad),(0,0)), mode='reflect')
    tops  = np.random.randint(0, 2 * pad, size=N)
    lefts = np.random.randint(0, 2 * pad, size=N)
    out   = np.stack([x_pad[i, tops[i]:tops[i]+H, lefts[i]:lefts[i]+W, :] for i in range(N)])
    flip  = np.random.rand(N) > 0.5
    out[flip] = out[flip, :, ::-1, :]
    return out


def train_resnet_model(
    model,
    x_train: jax.Array,
    y_train: jax.Array,
    epochs: int = 100,
    batch_size: int = 128,
    lr: float = 0.001,
    weight_decay: float = 1e-4,
    val_split: float = 0.1,
    patience: int = 15,
    warmup_epochs: int = 5,
):
    """
    Trains a ResNet-50 (or MCDropoutResNet50) on CIFAR images.

    Features:
      - Cosine LR decay with linear warmup via optax.warmup_cosine_decay_schedule
      - AdamW optimizer
      - Random crop + horizontal flip augmentation per batch
      - Epoch-level early stopping on val cross-entropy
      - BatchNorm use_running_average=False during training, True at evaluation

    Returns: trained model (best val-loss weights restored).
    """
    N = x_train.shape[0]
    n_val = max(1, int(N * val_split)) if val_split > 0 else 0
    if n_val > 0:
        perm   = np.random.permutation(N)
        tr_idx, va_idx = perm[n_val:], perm[:n_val]
        x_tr, y_tr = np.array(x_train[tr_idx]), np.array(y_train[tr_idx])
        x_va, y_va = np.array(x_train[va_idx]), np.array(y_train[va_idx])
    else:
        x_tr, y_tr = np.array(x_train), np.array(y_train)
        x_va, y_va = x_tr, y_tr

    n_tr      = len(x_tr)
    steps_ep  = max(1, n_tr // batch_size)
    total_steps = epochs * steps_ep

    warmup_epochs = min(warmup_epochs, epochs)
    warmup_steps = warmup_epochs * steps_ep
    if warmup_steps > 0:
        schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0, peak_value=lr,
            warmup_steps=warmup_steps,
            decay_steps=total_steps, end_value=lr * 1e-3)
    else:
        schedule = optax.cosine_decay_schedule(
            init_value=lr,
            decay_steps=total_steps, alpha=1e-3)
    optimizer = optax.adamw(learning_rate=schedule, weight_decay=weight_decay)
    opt_state = optimizer.init(nnx.state(model, nnx.Param))

    @nnx.jit
    def train_step(model, opt_state, x_batch, y_batch):
        def loss_fn(m):
            logits = m(x_batch, use_running_average=False)
            return jnp.mean(
                optax.softmax_cross_entropy_with_integer_labels(logits=logits, labels=y_batch))
        grads = nnx.grad(loss_fn)(model)
        updates, new_opt = optimizer.update(
            nnx.state(grads, nnx.Param), opt_state, nnx.state(model, nnx.Param))
        nnx.update(model, optax.apply_updates(nnx.state(model, nnx.Param), updates))
        return loss_fn(model), new_opt

    @nnx.jit
    def val_loss_fn(model, x_b, y_b):
        logits = model(x_b, use_running_average=True)
        return jnp.mean(
            optax.softmax_cross_entropy_with_integer_labels(logits=logits, labels=y_b))

    def _epoch_val():
        losses, ns = [], []
        for s in range(0, len(x_va), batch_size):
            xb = jnp.array(x_va[s:s+batch_size])
            yb = jnp.array(y_va[s:s+batch_size])
            losses.append(float(val_loss_fn(model, xb, yb)) * len(xb))
            ns.append(len(xb))
        return sum(losses) / sum(ns)

    def _epoch_acc():
        correct = total = 0
        for s in range(0, len(x_va), batch_size):
            xb  = jnp.array(x_va[s:s+batch_size])
            yb  = np.array(y_va[s:s+batch_size])
            preds = np.array(jnp.argmax(model(xb, use_running_average=True), axis=-1))
            correct += (preds == yb).sum()
            total   += len(yb)
        return correct / total

    best_val = float('inf')
    best_state = nnx.state(model)
    pat = 0
    t_start = time.time()

    print(f"Training ResNet-50: {n_tr} train / {len(x_va)} val | "
          f"epochs={epochs}, bs={batch_size}, lr={lr:g}")

    for epoch in range(epochs):
        t_ep = time.time()
        perm_ep = np.random.permutation(n_tr)
        ep_loss = 0.0
        for step in range(steps_ep):
            idx  = perm_ep[step * batch_size:(step + 1) * batch_size]
            xaug = jnp.array(_random_flip_crop(x_tr[idx]))
            yb   = jnp.array(y_tr[idx])
            loss, opt_state = train_step(model, opt_state, xaug, yb)
            ep_loss += float(loss)
            print(f"\r  Epoch {epoch+1:3d}/{epochs}  step {step+1}/{steps_ep}"
                  f"  loss={ep_loss/(step+1):.4f}", end='', flush=True)
        ep_loss /= steps_ep

        vl  = _epoch_val()
        acc = _epoch_acc()
        elapsed = time.time() - t_start
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
