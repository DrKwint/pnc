import sys
import os
import pickle
import jax
import jax.numpy as jnp
from flax import nnx
import numpy as np

# Add current dir to sys.path
sys.path.append(os.getcwd())

from models import PreActResNet18
from ensembles import LLLAEnsemble
from data import load_cifar10

def test_llla():
    # 1. Load model
    ckpt_path = "results/cifar10/preact_resnet18_e100_lr1e-03_wd1e-04_seed0.pkl"
    if not os.path.exists(ckpt_path):
        print(f"Skipping test: {ckpt_path} not found.")
        return

    print("Loading model...")
    with open(ckpt_path, "rb") as f:
        ckpt = pickle.load(f)
    
    n_cls = 10
    model = PreActResNet18(n_classes=n_cls, rngs=nnx.Rngs(0))
    nnx.update(model, ckpt['state'])
    
    # 2. Load small subset of data
    # We use a mocked data loading if it's too slow or fails
    try:
        x_train, y_train, x_test, y_test = load_cifar10()
        x_sub = x_train[:100]
        x_batch_test = x_test[:10]
    except Exception as e:
        print(f"Data loading failed: {e}. Using random data.")
        x_sub = np.random.normal(size=(100, 32, 32, 3)).astype(np.float32)
        x_batch_test = np.random.normal(size=(10, 32, 32, 3)).astype(np.float32)
    
    # 3. Compute GGN (Simplified for test)
    @jax.jit
    def get_features(x):
        h = model.stem(x)
        h = model._run_stages(h, use_running_average=True)
        h = model.final_bn(h, use_running_average=True)
        h = jax.nn.relu(h)
        h = jnp.mean(h, axis=(1, 2))
        return h

    print("Computing features...")
    feats = get_features(x_sub)
    logits = model.fc(feats)
    probs = jax.nn.softmax(logits)
    
    print("Computing GGN...")
    @jax.jit
    def compute_ggn(f, p):
        H = jax.vmap(lambda pi: jnp.diag(pi) - jnp.outer(pi, pi))(p)
        X_hat = jnp.concatenate([f, jnp.ones((f.shape[0], 1))], axis=-1)
        return jnp.einsum('si,sj,skm->ikjm', X_hat, X_hat, H)

    G = compute_ggn(feats, probs)
    
    D, K = 512, 10
    G_flat = G.reshape((D + 1) * K, (D + 1) * K)
    
    prior_prec = 1.0
    precision = G_flat + prior_prec * jnp.eye(G_flat.shape[0])
    
    print("Inverting GGN...")
    covariance = jnp.linalg.inv(precision)
    
    # 4. Create Ensemble
    print("Creating Ensemble...")
    fc_state = nnx.state(model.fc)
    ens = LLLAEnsemble(model, fc_state, covariance, n_models=5, seed=0)
    
    # 5. Predict
    print("Sampling predictions...")
    preds = ens.predict(x_batch_test)
    print(f"Ensemble predictions shape: {preds.shape}")
    assert preds.shape == (5, 10, 10)
    
    # Check diversity
    std = jnp.std(preds, axis=0)
    mean_std = float(jnp.mean(std))
    print(f"Mean prediction std across ensemble members: {mean_std:.6f}")
    assert mean_std > 0, "Ensemble members should have diverse predictions"

    print("LLLA test passed!")

if __name__ == "__main__":
    test_llla()
