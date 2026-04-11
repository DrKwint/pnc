import jax
import jax.numpy as jnp
from flax import nnx


def compute_kfac_factors(
    model, inputs, targets, batch_size=128, is_classification=False, seed=0
):
    """
    Computes KFAC (Kronecker-Factored Approximate Curvature) components for a linear model.
    Returns a dictionary of tuple factors: { 'l1': (A1, S1), ... }

    Supports both single-head models (RegressionModel) and dual-head probabilistic
    models (ProbabilisticRegressionModel with mean_layer + var_layer).
    """
    # Detect dual-head probabilistic model
    is_probabilistic = hasattr(model, "mean_layer") and hasattr(model, "var_layer")

    # Extract linear layers dynamically
    layers = []
    if hasattr(model, "layers"):
        for i, layer in enumerate(model.layers):
            layers.append(
                {
                    "name": f"l{i + 1}",
                    "w": layer.kernel.get_value(),
                    "b": layer.bias.get_value(),
                }
            )
    else:
        i = 1
        while hasattr(model, f"l{i}"):
            layer = getattr(model, f"l{i}")
            layers.append(
                {
                    "name": f"l{i}",
                    "w": layer.kernel.get_value(),
                    "b": layer.bias.get_value(),
                }
            )
            i += 1

    # For probabilistic models, add the mean and var output heads
    mean_head = None
    var_head = None
    if is_probabilistic:
        mean_head = {
            "name": "mean_layer",
            "w": model.mean_layer.kernel.get_value(),
            "b": model.mean_layer.bias.get_value(),
        }
        var_head = {
            "name": "var_layer",
            "w": model.var_layer.kernel.get_value(),
            "b": model.var_layer.bias.get_value(),
        }

    # Resolve the model's activation function
    act_fn = getattr(model, "activation", nnx.relu)

    if is_probabilistic:
        # For probabilistic models: hidden layers are ALL layers in model.layers
        # (no output layer in the list), plus separate mean/var heads.
        def forward_acts(x):
            acts = [jnp.concatenate([x, jnp.ones((*x.shape[:-1], 1))], axis=-1)]
            pre_acts = []

            curr_x = x
            for idx, layer in enumerate(layers):
                pre = curr_x @ layer["w"] + layer["b"]
                pre_acts.append(pre)
                curr_x = act_fn(pre)
                acts.append(
                    jnp.concatenate(
                        [curr_x, jnp.ones((*curr_x.shape[:-1], 1))], axis=-1
                    )
                )

            # Dual-head output: both heads take the last hidden activation
            pre_mean = curr_x @ mean_head["w"] + mean_head["b"]
            pre_var = curr_x @ var_head["w"] + var_head["b"]

            return acts, pre_acts, pre_mean, pre_var
    else:
        def forward_acts(x):
            acts = [jnp.concatenate([x, jnp.ones((*x.shape[:-1], 1))], axis=-1)]
            pre_acts = []

            curr_x = x
            for idx, layer in enumerate(layers):
                pre = curr_x @ layer["w"] + layer["b"]
                pre_acts.append(pre)

                if idx < len(layers) - 1:
                    curr_x = act_fn(pre)
                    acts.append(
                        jnp.concatenate(
                            [curr_x, jnp.ones((*curr_x.shape[:-1], 1))], axis=-1
                        )
                    )
                else:
                    curr_x = pre

            return acts, pre_acts, None, None

    # Initialize accumulators
    all_layer_dicts = list(layers)
    if is_probabilistic:
        all_layer_dicts.append(mean_head)
        all_layer_dicts.append(var_head)

    accumulators = {}
    for ld in all_layer_dicts:
        accumulators[ld["name"]] = {
            "A": jnp.zeros((ld["w"].shape[0] + 1, ld["w"].shape[0] + 1)),
            "S": jnp.zeros((ld["w"].shape[1], ld["w"].shape[1])),
        }

    n_samples = len(inputs)

    if is_probabilistic:
        @jax.jit
        def process_batch(x_batch, rng_key):
            acts, pre_acts, pre_mean, pre_var = forward_acts(x_batch)

            # Probabilistic model: compute Gaussian NLL Fisher gradients
            mean = pre_mean  # linear output head
            var = jax.nn.softplus(pre_var) + 1e-6
            out_dim = mean.shape[-1]

            # Sample from predictive distribution for Fisher
            y_sampled = mean + jnp.sqrt(var) * jax.random.normal(rng_key, mean.shape)

            # Gradient of Gaussian NLL w.r.t. pre-activations of each head
            # L = (1/D) * sum_d 0.5 * (log(var_d) + (y_d - mean_d)^2 / var_d)
            d_pre_mean = (1.0 / out_dim) * (mean - y_sampled) / var
            sigmoid_pre_var = jax.nn.sigmoid(pre_var)
            d_pre_var = (0.5 / out_dim) * (1.0 - (y_sampled - mean) ** 2 / var) * sigmoid_pre_var

            # Input activation for both heads is the last hidden activation (same)
            act_last_hidden = acts[-1]

            # Backprop combined gradient through hidden layers
            d_h = d_pre_mean @ mean_head["w"].T + d_pre_var @ var_head["w"].T

            d_pres = []
            for idx in range(len(layers) - 1, -1, -1):
                pre_curr = pre_acts[idx]
                _, vjp_fn = jax.vjp(act_fn, pre_curr)
                d_pre_curr = vjp_fn(d_h)[0]
                d_pres.insert(0, d_pre_curr)
                if idx > 0:
                    d_h = d_pre_curr @ layers[idx]["w"].T

            # Compute KFAC covariances
            batch_factors = {}
            for idx, layer in enumerate(layers):
                a = acts[idx]
                s = d_pres[idx]
                batch_factors[layer["name"]] = {
                    "A": jnp.einsum("bi,bj->ij", a, a),
                    "S": jnp.einsum("bi,bj->ij", s, s),
                }

            # KFAC factors for mean and var heads
            batch_factors["mean_layer"] = {
                "A": jnp.einsum("bi,bj->ij", act_last_hidden, act_last_hidden),
                "S": jnp.einsum("bi,bj->ij", d_pre_mean, d_pre_mean),
            }
            batch_factors["var_layer"] = {
                "A": jnp.einsum("bi,bj->ij", act_last_hidden, act_last_hidden),
                "S": jnp.einsum("bi,bj->ij", d_pre_var, d_pre_var),
            }

            return batch_factors
    else:
        @jax.jit
        def process_batch(x_batch, rng_key):
            acts, pre_acts, _, _ = forward_acts(x_batch)

            pre_last = pre_acts[-1]

            if is_classification:
                probs = jax.nn.softmax(pre_last)
                y_sampled = jax.random.categorical(rng_key, jnp.log(probs + 1e-10))
                y_one_hot = jax.nn.one_hot(y_sampled, probs.shape[-1])
                d_pre = probs - y_one_hot
            else:
                out_dim_dynamic = pre_last.shape[-1]
                y_sampled = pre_last + jnp.sqrt(out_dim_dynamic / 2.0) * jax.random.normal(
                    rng_key, pre_last.shape
                )
                d_pre = (2.0 / out_dim_dynamic) * (pre_last - y_sampled)

            d_pres = [d_pre]

            for idx in range(len(layers) - 2, -1, -1):
                w_next = layers[idx + 1]["w"]
                pre_curr = pre_acts[idx]

                d_h = d_pres[0] @ w_next.T
                _, vjp_fn = jax.vjp(act_fn, pre_curr)
                d_pre_curr = vjp_fn(d_h)[0]
                d_pres.insert(0, d_pre_curr)

            batch_factors = {}
            for idx, layer in enumerate(layers):
                a = acts[idx]
                s = d_pres[idx]

                batch_A = jnp.einsum("bi,bj->ij", a, a)
                batch_S = jnp.einsum("bi,bj->ij", s, s)

                batch_factors[layer["name"]] = {"A": batch_A, "S": batch_S}

            return batch_factors

    rng = jax.random.PRNGKey(seed)
    for i in range(0, n_samples, batch_size):
        x_batch = inputs[i : i + batch_size]
        rng, key = jax.random.split(rng)

        batch_factors = process_batch(x_batch, key)

        for name in accumulators:
            accumulators[name]["A"] += batch_factors[name]["A"]
            accumulators[name]["S"] += batch_factors[name]["S"]

    # Average over total samples
    factors = {}
    for name, accs in accumulators.items():
        factors[name] = (accs["A"] / n_samples, accs["S"] / n_samples)

    return factors
