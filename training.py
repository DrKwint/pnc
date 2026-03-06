import jax
import jax.numpy as jnp
import optax
import numpy as np
from flax import nnx

from models import TransitionModel

def train_model(
    model: TransitionModel, 
    inputs: jax.Array, 
    targets: jax.Array, 
    steps: int = 2000, 
    batch_size: int = 64
) -> TransitionModel:
    """Trains a simple neural network transition model."""
    print(f"Training on {len(inputs)} samples...")
    optimizer = optax.adamw(1e-3)
    params = nnx.state(model, nnx.Param)
    opt_state = optimizer.init(params)
    
    @nnx.jit
    def train_step(model, opt_state, x, y):
        def loss_fn(model):
            return jnp.mean((model(x) - y) ** 2)
        
        grads = nnx.grad(loss_fn)(model)
        updates, new_opt = optimizer.update(nnx.state(grads, nnx.Param), opt_state, nnx.state(model, nnx.Param))
        nnx.update(model, optax.apply_updates(nnx.state(model, nnx.Param), updates))
        return loss_fn(model), new_opt

    indices = np.arange(len(inputs))
    loss = jnp.inf
    
    for i in range(steps):
        batch = np.random.choice(indices, batch_size)
        loss, opt_state = train_step(model, opt_state, inputs[batch], targets[batch])
        if i % 500 == 0: 
            print(f"Step {i}: Loss {loss:.5f}")
            
    print(f"Final Loss: {loss:.5f}")
    return model

def train_swag_model(
    model: TransitionModel, 
    inputs: jax.Array, 
    targets: jax.Array, 
    steps: int = 2000, 
    batch_size: int = 64,
    swag_start: int = 1000
):
    """Trains a simple neural network transition model and collects Diagonal SWAG statistics."""
    print(f"Training on {len(inputs)} samples (SWAG enabled after {swag_start} steps)...")
    optimizer = optax.adamw(1e-3)
    params = nnx.state(model, nnx.Param)
    opt_state = optimizer.init(params)
    
    @nnx.jit
    def train_step(model, opt_state, x, y):
        def loss_fn(model):
            return jnp.mean((model(x) - y) ** 2)
        
        grads = nnx.grad(loss_fn)(model)
        updates, new_opt = optimizer.update(nnx.state(grads, nnx.Param), opt_state, nnx.state(model, nnx.Param))
        nnx.update(model, optax.apply_updates(nnx.state(model, nnx.Param), updates))
        return loss_fn(model), new_opt

    indices = np.arange(len(inputs))
    loss = jnp.inf
    
    # Initialize SWAG statistics
    swag_mean = jax.tree.map(lambda x: jnp.zeros_like(x), params)
    swag_sq_mean = jax.tree.map(lambda x: jnp.zeros_like(x), params)
    n_swag_steps = 0
    
    for i in range(steps):
        batch = np.random.choice(indices, batch_size)
        loss, opt_state = train_step(model, opt_state, inputs[batch], targets[batch])
        
        if i >= swag_start:
            current_params = nnx.state(model, nnx.Param)
            
            # Update running means
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

        if i % 500 == 0: 
            print(f"Step {i}: Loss {loss:.5f}")
            
    print(f"Final Loss: {loss:.5f} | SWAG steps collected: {n_swag_steps}")
    
    # Calculate variance: E[X^2] - E[X]^2
    # Add small epsilon for numerical stability
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
    lr: float = 1e-3
) -> nnx.Module:
    """Trains a classification model using cross entropy loss."""
    print(f"Training Classification Model on {len(inputs)} samples...")
    optimizer = optax.adamw(lr)
    params = nnx.state(model, nnx.Param)
    opt_state = optimizer.init(params)
    
    @nnx.jit
    def train_step(model, opt_state, x, y):
        def loss_fn(model):
            logits = model(x)
            return jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits=logits, labels=y))
        
        grads = nnx.grad(loss_fn)(model)
        updates, new_opt = optimizer.update(nnx.state(grads, nnx.Param), opt_state, nnx.state(model, nnx.Param))
        nnx.update(model, optax.apply_updates(nnx.state(model, nnx.Param), updates))
        return loss_fn(model), new_opt

    indices = np.arange(len(inputs))
    loss = jnp.inf
    
    for i in range(steps):
        batch = np.random.choice(indices, batch_size)
        loss, opt_state = train_step(model, opt_state, inputs[batch], targets[batch])
        if i % 1000 == 0: 
            print(f"Step {i}: Loss {loss:.5f}")
            
    print(f"Final Loss: {loss:.5f}")
    return model

def train_swag_classification_model(
    model: nnx.Module, 
    inputs: jax.Array, 
    targets: jax.Array, 
    steps: int = 5000, 
    batch_size: int = 256,
    lr: float = 1e-3,
    swag_start: int = 3000
):
    """Trains a classification model and collects Diagonal SWAG statistics."""
    print(f"Training Classification on {len(inputs)} samples (SWAG enabled after {swag_start} steps)...")
    optimizer = optax.adamw(lr)
    params = nnx.state(model, nnx.Param)
    opt_state = optimizer.init(params)
    
    @nnx.jit
    def train_step(model, opt_state, x, y):
        def loss_fn(model):
            logits = model(x)
            return jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits=logits, labels=y))
        
        grads = nnx.grad(loss_fn)(model)
        updates, new_opt = optimizer.update(nnx.state(grads, nnx.Param), opt_state, nnx.state(model, nnx.Param))
        nnx.update(model, optax.apply_updates(nnx.state(model, nnx.Param), updates))
        return loss_fn(model), new_opt

    indices = np.arange(len(inputs))
    loss = jnp.inf
    
    # Initialize SWAG statistics
    swag_mean = jax.tree.map(lambda x: jnp.zeros_like(x), params)
    swag_sq_mean = jax.tree.map(lambda x: jnp.zeros_like(x), params)
    n_swag_steps = 0
    
    for i in range(steps):
        batch = np.random.choice(indices, batch_size)
        loss, opt_state = train_step(model, opt_state, inputs[batch], targets[batch])
        
        if i >= swag_start:
            current_params = nnx.state(model, nnx.Param)
            
            # Update running means
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

        if i % 1000 == 0: 
            print(f"Step {i}: Loss {loss:.5f}")
            
    print(f"Final Loss: {loss:.5f} | SWAG steps collected: {n_swag_steps}")
    
    swag_var = jax.tree.map(
        lambda sq_m, m: jnp.maximum(sq_m - m**2, 1e-8), 
        swag_sq_mean, swag_mean
    )
    
    return model, swag_mean, swag_var
