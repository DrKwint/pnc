import jax
import jax.numpy as jnp
from flax import nnx
import numpy as np

def compute_kfac_factors(model, inputs, targets, batch_size=128, is_classification=False):
    """
    Computes KFAC (Kronecker-Factored Approximate Curvature) components for a linear model.
    Returns a dictionary of tuple factors: { 'l1': (A1, S1), ... }
    """
    # Extract linear layers dynamically
    layers = []
    i = 1
    while hasattr(model, f'l{i}'):
        layer = getattr(model, f'l{i}')
        # We need the kernel and bias
        layers.append({
            'name': f'l{i}',
            'w': layer.kernel.get_value(),
            'b': layer.bias.get_value()
        })
        i += 1
        
    def forward_acts(x):
        acts = [jnp.concatenate([x, jnp.ones((*x.shape[:-1], 1))], axis=-1)]
        pre_acts = []
        
        curr_x = x
        for idx, layer in enumerate(layers):
            pre = curr_x @ layer['w'] + layer['b']
            pre_acts.append(pre)
            
            # Apply ReLU for all but the last layer
            if idx < len(layers) - 1:
                curr_x = nnx.relu(pre)
                acts.append(jnp.concatenate([curr_x, jnp.ones((*curr_x.shape[:-1], 1))], axis=-1))
            else:
                curr_x = pre
                
        return acts, pre_acts

    # Initialize accumulators
    accumulators = {}
    for layer in layers:
        accumulators[layer['name']] = {
            'A': jnp.zeros((layer['w'].shape[0] + 1, layer['w'].shape[0] + 1)),
            'S': jnp.zeros((layer['w'].shape[1], layer['w'].shape[1]))
        }
        
    n_samples = len(inputs)
    out_dim = targets.shape[-1] if len(targets.shape) > 1 else -1

    @jax.jit
    def process_batch(x_batch, y_batch):
        acts, pre_acts = forward_acts(x_batch)
        
        # Calculate gradients with respect to pre-activations of the last layer
        pre_last = pre_acts[-1]
        if is_classification:
            # For Cross Entropy with Softmax: dL/dz = softmax(z) - one_hot(y)
            # The loss is mean(CrossEntropy). The per-example gradient is the above
            # Note: optax mean reduction divides by batch size, but we are computing empirical Fisher
            # which expects the un-averaged per-example gradient covariance.
            probs = jax.nn.softmax(pre_last)
            y_one_hot = jax.nn.one_hot(y_batch, probs.shape[-1])
            d_pre = probs - y_one_hot
        else:
            # For MSE, we averaged over out_dim: jnp.mean((preds - y)**2)
            d_pre = (2.0 / out_dim) * (pre_last - y_batch)
            
        d_pres = [d_pre]
        
        # Backprop through hidden layers
        for idx in range(len(layers) - 2, -1, -1):
            w_next = layers[idx + 1]['w']
            pre_curr = pre_acts[idx]
            
            d_h = d_pres[0] @ w_next.T
            d_pre_curr = d_h * (pre_curr > 0).astype(jnp.float32)
            d_pres.insert(0, d_pre_curr)
            
        # Compute covariances
        batch_factors = {}
        for idx, layer in enumerate(layers):
            a = acts[idx]
            s = d_pres[idx]
            
            batch_A = jnp.einsum('bi,bj->ij', a, a)
            batch_S = jnp.einsum('bi,bj->ij', s, s)
            
            batch_factors[layer['name']] = {'A': batch_A, 'S': batch_S}
            
        return batch_factors

    for i in range(0, n_samples, batch_size):
        x_batch = inputs[i:i+batch_size]
        y_batch = targets[i:i+batch_size]
        
        batch_factors = process_batch(x_batch, y_batch)
        
        for name in accumulators:
            accumulators[name]['A'] += batch_factors[name]['A']
            accumulators[name]['S'] += batch_factors[name]['S']
            
    # Average over total samples
    factors = {}
    for name, accs in accumulators.items():
        factors[name] = (accs['A'] / n_samples, accs['S'] / n_samples)
        
    return factors
