import jax
import jax.numpy as jnp
from flax import nnx

def compute_kfac_factors(model, inputs, targets, batch_size=128, is_classification=False, seed=0):
    """
    Computes KFAC (Kronecker-Factored Approximate Curvature) components for a linear model.
    Returns a dictionary of tuple factors: { 'l1': (A1, S1), ... }
    """
    # Extract linear layers dynamically
    layers = []
    if hasattr(model, 'layers'):
        for i, layer in enumerate(model.layers):
            layers.append({
                'name': f'l{i+1}',
                'w': layer.kernel.get_value(),
                'b': layer.bias.get_value()
            })
    else:
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

    # Resolve the model's activation function (added with the activation param refactor).
    # Fall back to nnx.relu for models that predate the field.
    act_fn = getattr(model, 'activation', nnx.relu)

    def forward_acts(x):
        acts = [jnp.concatenate([x, jnp.ones((*x.shape[:-1], 1))], axis=-1)]
        pre_acts = []

        curr_x = x
        for idx, layer in enumerate(layers):
            pre = curr_x @ layer['w'] + layer['b']
            pre_acts.append(pre)

            # Apply the model's activation for all but the last layer
            if idx < len(layers) - 1:
                curr_x = act_fn(pre)
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
    def process_batch(x_batch, rng_key):
        acts, pre_acts = forward_acts(x_batch)
        
        # Calculate gradients with respect to pre-activations of the last layer
        pre_last = pre_acts[-1]
        
        # Sample targets from the model's predictive distribution to compute the True Fisher (Gauss-Newton)
        if is_classification:
            probs = jax.nn.softmax(pre_last)
            y_sampled = jax.random.categorical(rng_key, jnp.log(probs + 1e-10))
            y_one_hot = jax.nn.one_hot(y_sampled, probs.shape[-1])
            d_pre = probs - y_one_hot
        else:
            out_dim_dynamic = pre_last.shape[-1]
            y_sampled = pre_last + jnp.sqrt(out_dim_dynamic / 2.0) * jax.random.normal(rng_key, pre_last.shape)
            d_pre = (2.0 / out_dim_dynamic) * (pre_last - y_sampled)
            
        d_pres = [d_pre]
        
        # Backprop through hidden layers
        for idx in range(len(layers) - 2, -1, -1):
            w_next = layers[idx + 1]['w']
            pre_curr = pre_acts[idx]
            
            d_h = d_pres[0] @ w_next.T
            _, vjp_fn = jax.vjp(act_fn, pre_curr)
            d_pre_curr = vjp_fn(d_h)[0]
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

    rng = jax.random.PRNGKey(seed)
    for i in range(0, n_samples, batch_size):
        x_batch = inputs[i:i+batch_size]
        rng, key = jax.random.split(rng)
        
        batch_factors = process_batch(x_batch, key)
        
        for name in accumulators:
            accumulators[name]['A'] += batch_factors[name]['A']
            accumulators[name]['S'] += batch_factors[name]['S']
            
    # Average over total samples
    factors = {}
    for name, accs in accumulators.items():
        factors[name] = (accs['A'] / n_samples, accs['S'] / n_samples)
        
    return factors
