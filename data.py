import gymnasium as gym
import jax.numpy as jnp
import numpy as np
from typing import Callable, Tuple
import jax
import os

def positive_policy(env: gym.Env, obs: np.ndarray) -> np.ndarray:
    """A placeholder policy that always takes positive actions."""
    action_dim = env.action_space.shape[0] # type: ignore
    return np.random.uniform(0.5, 1.0, size=(action_dim,)).astype(np.float32)

def negative_policy(env: gym.Env, obs: np.ndarray) -> np.ndarray:
    """A placeholder policy that always takes negative actions."""
    action_dim = env.action_space.shape[0] # type: ignore
    return np.random.uniform(-1.0, -0.5, size=(action_dim,)).astype(np.float32)

def id_policy_random(env: gym.Env, obs: np.ndarray) -> np.ndarray:
    """
    ID Policy: Pure Random Exploration.
    Mimics 'training data' in offline RL. 
    State coverage: Low velocity, chaotic poses, falling over.
    """
    return env.action_space.sample()

def ood_policy_run(env: gym.Env, obs: np.ndarray, step_count: int) -> np.ndarray:
    """
    OOD Policy: A structured Sine-Wave Gait.
    Mimics an 'expert' or 'deployment' policy.
    State coverage: High velocity, coordinated periodic motion.
    
    This is OOD because the model has never seen 'coordinated running' 
    dynamics, only 'random flailing' dynamics.
    """
    # A simple trotting gait for HalfCheetah
    # Joints: [bthigh, bshin, bfoot, fthigh, fshin, ffoot]
    # We oscillate legs in anti-phase to create forward motion
    
    freq = 2.0  # Hz
    phase_offset = np.array([0, -1, -0.5, np.pi, np.pi-1, np.pi-0.5])
    amplitude = 0.8
    
    t = step_count * 0.05 # Assuming dt=0.05 or similar
    
    # Generate action: A * sin(wt + phi)
    action = amplitude * np.sin(freq * t + phase_offset)
    
    return action.astype(np.float32)

def collect_data(
    env_name: str, 
    steps: int, 
    policy_fn: Callable[[gym.Env, np.ndarray], np.ndarray], 
    seed: int = 0
) -> Tuple[jax.Array, jax.Array]:
    """Collects transitions from a Gym environment using a given policy."""
    print(f"Collecting {steps} steps ({str(policy_fn)})...")
    env = gym.make(env_name)
        
    inputs, targets = [], []
    obs, _ = env.reset(seed=seed)
    
    for _ in range(steps):
        action = policy_fn(env, obs)
        # Fix for Gym API quirks
        if np.isscalar(action): 
            action = np.array([action], dtype=np.float32)
        if action.ndim == 0: 
            action = action[None]
            
        next_obs, _, term, trunc, _ = env.step(action)
        
        inputs.append(np.concatenate([obs, action])) # State + Action
        targets.append(next_obs) # Predict Next State
        
        obs = next_obs
        if term or trunc:
            obs, _ = env.reset()
    
    env.close()
    return jnp.array(np.stack(inputs)), jnp.array(np.stack(targets))

class OODPolicyWrapper:
    """A stateful wrapper for ood_policy_run that keeps track of the step count."""
    def __init__(self):
        self.step_count = 0

    def __call__(self, env: gym.Env, obs: np.ndarray) -> np.ndarray:
        action = ood_policy_run(env, obs, self.step_count)
        self.step_count += 1
        return action

def _download_idx(url: str, cache_dir: str) -> np.ndarray:
    """Downloads and parses an IDX file (MNIST binary format). Caches locally."""
    import urllib.request, gzip, struct
    os.makedirs(cache_dir, exist_ok=True)
    fname = os.path.join(cache_dir, url.split('/')[-1])
    if not os.path.exists(fname):
        print(f"  Downloading {url} ...")
        urllib.request.urlretrieve(url, fname)
    with gzip.open(fname, 'rb') as f:
        magic = struct.unpack('>I', f.read(4))[0]
        ndim = magic & 0xFF
        shape = tuple(struct.unpack('>I', f.read(4))[0] for _ in range(ndim))
        data = np.frombuffer(f.read(), dtype=np.uint8).reshape(shape)
    return data

def load_mnist(cache_dir: str = '/tmp/mnist_cache') -> tuple:
    """
    Loads MNIST train and test splits.
    Downloads directly from the Google CDN — no tensorflow or keras needed.
    Returns: (x_train, y_train, x_test, y_test)
    """
    MNIST_BASE = "https://storage.googleapis.com/cvdf-datasets/mnist/"
    print("Loading MNIST dataset...")

    cache = os.path.join(cache_dir, 'mnist')
    x_train_raw = _download_idx(f"{MNIST_BASE}train-images-idx3-ubyte.gz", cache)
    y_train_raw = _download_idx(f"{MNIST_BASE}train-labels-idx1-ubyte.gz", cache)
    x_test_raw  = _download_idx(f"{MNIST_BASE}t10k-images-idx3-ubyte.gz",  cache)
    y_test_raw  = _download_idx(f"{MNIST_BASE}t10k-labels-idx1-ubyte.gz",  cache)

    preprocess_x = lambda x: jnp.array(x.reshape(-1, 784).astype(np.float32) / 255.0)
    preprocess_y = lambda y: jnp.array(y, dtype=jnp.int32)

    return preprocess_x(x_train_raw), preprocess_y(y_train_raw), preprocess_x(x_test_raw), preprocess_y(y_test_raw)
